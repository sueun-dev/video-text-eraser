"""Subtitle text-box detection across video frames using PaddleOCR."""

import sys
from functools import cached_property
from typing import Dict, List, Optional, cast

import cv2
from tqdm import tqdm

from backend.config import config, tr
from backend.scenedetect import scene_detect
from backend.scenedetect.detectors import ContentDetector

from .common_tools import get_readable_path
from .constant import Area, Box, FrameRange
from .hardware_accelerator import HardwareAccelerator
from .inpaint_tools import is_frame_number_in_ab_sections
from .model_config import ModelConfig
from .ocr import get_coordinates

FrameBoxes = Dict[int, List[Box]]                # frame number -> detected boxes


class SubtitleDetect:
    """Finds which frames contain subtitle text and where the text sits.

    Detection runs on a sample of frames (OCR is expensive) and the gaps are
    interpolated afterwards, which is safe because subtitles persist for many
    consecutive frames.
    """

    def __init__(self, video_path: str, sub_areas: Optional[List[Area]] = None):
        self.video_path = video_path
        self.sub_areas = sub_areas or []
        self.sample_step = self._choose_sample_step(video_path)

    @staticmethod
    def _choose_sample_step(video_path: str) -> int:
        """Pick an OCR sampling interval that keeps >= 8 samples per second."""
        cap = cv2.VideoCapture(get_readable_path(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps >= 60:
            return 4
        if fps >= 30:
            return 3
        return 2

    @cached_property
    def text_detector(self):
        """Lazily constructed PaddleOCR text detector (detection only, no recognition)."""
        import paddle

        paddle.disable_signal_handler()
        from paddleocr import TextDetection

        onnx_providers = HardwareAccelerator.instance().onnx_providers
        model_config = ModelConfig()
        try:
            return TextDetection(
                model_name=model_config.DET_MODEL_NAME,
                model_dir=model_config.DET_MODEL_DIR,
                device="cpu",
                enable_hpi=len(onnx_providers) > 0,
            )
        except RuntimeError:
            # High-performance inference plugins are an optional install;
            # fall back to the plain Paddle predictor when they are missing.
            return TextDetection(
                model_name=model_config.DET_MODEL_NAME,
                model_dir=model_config.DET_MODEL_DIR,
                device="cpu",
                enable_hpi=False,
            )

    def detect_subtitle(self, img) -> List[Box]:
        """Run text detection on one frame, keeping boxes inside the user areas."""
        detected: List[Box] = []
        for res in self.text_detector.predict(img):
            polys = res["dt_polys"]
            if polys is None or len(polys) == 0:
                continue
            boxes = get_coordinates(polys.tolist())
            detected.extend(self._filter_by_areas(boxes))
        return detected

    def _filter_by_areas(self, boxes: List[Box]) -> List[Box]:
        if not self.sub_areas:
            return boxes
        kept = []
        for xmin, xmax, ymin, ymax in boxes:
            for area_ymin, area_ymax, area_xmin, area_xmax in self.sub_areas:
                inside = (
                    area_xmin <= xmin and xmax <= area_xmax
                    and area_ymin <= ymin and ymax <= area_ymax
                )
                if inside:
                    kept.append((xmin, xmax, ymin, ymax))
                    break
        return kept

    def find_subtitle_frame_no(self, sub_remover=None) -> FrameBoxes:
        """Scan the whole video and map frame numbers to subtitle boxes.

        Phase 1 samples every ``sample_step``-th frame with OCR; phase 2 fills
        the frames between two positive samples with the earlier sample's
        boxes, then near-identical boxes are unified to suppress jitter.
        """
        video_cap = cv2.VideoCapture(get_readable_path(self.video_path))
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        tbar = tqdm(
            total=int(frame_count), unit="frame", position=0,
            file=sys.__stdout__, desc="Subtitle Finding",
        )
        if sub_remover:
            sub_remover.append_output(tr["Main"]["ProcessingStartFindingSubtitles"])

        ab_sections = sub_remover.ab_sections if sub_remover else None
        sampled: FrameBoxes = {}
        frame_no = 0
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break
            frame_no += 1
            if not is_frame_number_in_ab_sections(frame_no - 1, ab_sections):
                tbar.update(1)
                continue
            if (frame_no - 1) % self.sample_step == 0:
                boxes = self.detect_subtitle(frame)
                if boxes:
                    sampled[frame_no] = boxes
            tbar.update(1)
            if sub_remover:
                sub_remover.progress_total = (100 * frame_no / frame_count) // 2
        video_cap.release()

        frame_boxes = self._interpolate_samples(sampled)
        frame_boxes = self.unify_regions(frame_boxes)
        if sub_remover:
            sub_remover.append_output(tr["Main"]["FinishedFindingSubtitles"])
        return {no: boxes for no, boxes in frame_boxes.items() if boxes}

    def _interpolate_samples(self, sampled: FrameBoxes) -> FrameBoxes:
        """Mark frames between two nearby positive samples as subtitled too."""
        filled: FrameBoxes = {}
        sample_nos = sorted(sampled.keys())
        max_gap = self.sample_step * 2
        for current, following in zip(sample_nos, sample_nos[1:]):
            filled[current] = sampled[current]
            if following - current <= max_gap:
                for between in range(current + 1, following):
                    filled[between] = sampled[current]
        if sample_nos:
            filled[sample_nos[-1]] = sampled[sample_nos[-1]]
        return filled

    @staticmethod
    def are_similar(region1: Box, region2: Box) -> bool:
        """Whether two boxes are the same subtitle line allowing minor jitter."""
        xmin1, xmax1, ymin1, ymax1 = region1
        xmin2, xmax2, ymin2, ymax2 = region2
        tol_x = config.subtitleAreaPixelToleranceXPixel.value
        tol_y = config.subtitleAreaPixelToleranceYPixel.value
        return (
            abs(xmin1 - xmin2) <= tol_x and abs(xmax1 - xmax2) <= tol_x
            and abs(ymin1 - ymin2) <= tol_y and abs(ymax1 - ymax2) <= tol_y
        )

    def unify_regions(self, raw_regions: FrameBoxes) -> FrameBoxes:
        """Snap each frame's boxes to the previous frame's when nearly identical.

        Stabilised boxes produce identical masks across consecutive frames,
        which lets later stages batch those frames together.
        """
        if not raw_regions:
            return raw_regions

        keys = sorted(raw_regions.keys())
        unified: FrameBoxes = {keys[0]: raw_regions[keys[0]]}
        last_key = keys[0]
        for key in keys[1:]:
            previous_boxes = unified[last_key]
            stabilised = []
            for idx, box in enumerate(raw_regions[key]):
                reference = previous_boxes[idx] if idx < len(previous_boxes) else None
                if reference and self.are_similar(box, reference):
                    stabilised.append(reference)
                else:
                    stabilised.append(box)
            unified[key] = stabilised
            last_key = key
        return unified

    @staticmethod
    def find_continuous_ranges_with_same_mask(frame_boxes: FrameBoxes) -> List[FrameRange]:
        """Split subtitled frames into runs where the mask stays identical.

        A run breaks on a frame-number gap or whenever the box layout changes,
        so each run can be inpainted with a single mask.
        """
        numbers = sorted(frame_boxes.keys())
        ranges: List[FrameRange] = []
        start = numbers[0]
        for prev, current in zip(numbers, numbers[1:]):
            gap = current - prev != 1
            mask_changed = not gap and frame_boxes[current] != frame_boxes[prev]
            if gap or mask_changed:
                ranges.append((start, prev))
                start = current
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def get_scene_div_frame_no(video_path: str) -> List[int]:
        """Frame numbers where a scene cut occurs (used to split STTN runs)."""
        scene_starts = []
        for start, _ in scene_detect(video_path, ContentDetector()):
            # scenedetect mistypes frame_num as Optional; it is always an int.
            frame_num = cast(int, start.frame_num)
            if frame_num != 0:
                scene_starts.append(frame_num + 1)
        return scene_starts

    @staticmethod
    def split_range_by_scene(
        intervals: List[FrameRange], points: List[int]
    ) -> List[FrameRange]:
        """Split frame ranges at scene-cut points so inpainting never crosses a cut."""
        points.sort()
        result: List[FrameRange] = []
        for start, end in intervals:
            for point in (p for p in points if start <= p <= end):
                if start < point:
                    result.append((start, point - 1))
                start = point
            result.append((start, end))
        return result

    @staticmethod
    def filter_and_merge_intervals(
        intervals: List[FrameRange], target_length: int
    ) -> List[FrameRange]:
        """Guarantee every interval spans at least ``target_length`` frames.

        STTN needs a minimum temporal context; single-frame intervals are
        widened around their midpoint and short neighbours are merged.
        """
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda interval: interval[0])

        expanded: List[FrameRange] = []
        for i, (start, end) in enumerate(intervals):
            if start == end:
                half = (target_length - 1) // 2
                # Lower bound: the previous interval's end + 1, but never below
                # frame 1 (frame numbers are 1-based everywhere; cf.
                # expand_frame_ranges).
                floor = expanded[-1][1] + 1 if expanded else 1
                new_start = max(start - half, floor, 1)
                # Upper bound: stop before the next interval starts.
                ceil = intervals[i + 1][0] - 1 if i + 1 < len(intervals) else start + half
                new_end = min(start + half, ceil)
                if new_end < new_start:
                    new_start, new_end = start, start
                expanded.append((new_start, new_end))
            else:
                expanded.append((start, end))

        merged = [expanded[0]]
        for start, end in expanded[1:]:
            last_start, last_end = merged[-1]
            adjacent = start <= last_end or start == last_end + 1
            either_short = (
                end - start + 1 < target_length
                or last_end - last_start + 1 < target_length
            )
            if adjacent and either_short:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged
