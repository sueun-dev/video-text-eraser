"""Subtitle removal pipeline orchestrator.

Detects hard-coded subtitle text in a video (or picture), erases it with the
configured inpaint model, then re-encodes the result and merges the original
audio back in. See ``backend.tools.args_handler`` for the CLI entry point.
"""

import gc
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import InpaintMode, SubtitleDetectMode, config, tr
from backend.inpaint.lama_inpaint import LamaInpaint
from backend.inpaint.opencv_inpaint import OpenCVInpaint
from backend.inpaint.propainter_inpaint import PropainterInpaint
from backend.inpaint.sttn_auto_inpaint import STTNAutoInpaint
from backend.inpaint.sttn_det_inpaint import STTNDetInpaint
from backend.tools.common_tools import (
    get_readable_path,
    is_image_file,
    is_video_or_image,
    read_image,
)
from backend.tools.constant import Area
from backend.tools.ffmpeg_cli import FFmpegCLI
from backend.tools.hardware_accelerator import HardwareAccelerator
from backend.tools.inpaint_tools import batch_generator, create_mask, expand_frame_ranges
from backend.tools.model_config import ModelConfig
from backend.tools.subtitle_detect import SubtitleDetect
from backend.tools.video_io import FramePrefetcher, make_video_writer


class SubtitleRemover:
    """Runs the detect -> inpaint -> encode -> merge-audio pipeline.

    The GUI drives this class through the same public surface as the CLI:
    set ``sub_areas`` / ``ab_sections``, optionally replace ``append_output``
    and ``update_preview_with_comp``, register a progress listener, then call
    :meth:`run`.
    """

    def __init__(self, vd_path: str, gui_mode: bool = False):
        self.lock = threading.RLock()
        self.gui_mode = gui_mode
        # Subtitle areas selected by the user; empty means full frame.
        self.sub_areas: List[Area] = []
        # A/B frame sections to process; None means the whole video.
        self.ab_sections: Optional[List[range]] = None

        self.hardware_accelerator = HardwareAccelerator.instance()
        self.hardware_accelerator.set_enabled(config.hardwareAcceleration.value)
        self.model_config = ModelConfig()

        self.video_path = vd_path
        self.is_picture = is_image_file(str(vd_path))
        self.vd_name = Path(vd_path).stem
        self.ext = os.path.splitext(vd_path)[-1]

        self.video_cap = cv2.VideoCapture(get_readable_path(vd_path))
        # Guard against corrupt/missing metadata: a negative frame count would
        # drop frames, and a 0/NaN fps would make the writer emit an empty file.
        self.frame_count = max(0, int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5))
        raw_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.fps = raw_fps if raw_fps and raw_fps > 0 and raw_fps == raw_fps else 25.0
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = (self.frame_width, self.frame_height)
        self.mask_size = (self.frame_height, self.frame_width)

        # delete=False: Windows raises permission errors with delete=True.
        self.video_temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.video_writer = self._create_video_writer()
        self.video_out_path = self._default_output_path()

        self.progress_total = 0
        self.progress_remover = 0
        self.isFinished = False
        self.is_successful_merged = False
        self.progress_listeners = []
        # Active frame prefetcher, stopped in run()'s finally so its daemon
        # thread cannot leak if processing raises mid-run.
        self._active_reader: Optional[FramePrefetcher] = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _create_video_writer(self):
        """Prefer the FFmpeg/libx264 writer; fall back to OpenCV mp4v."""
        temp_path = get_readable_path(self.video_temp_file.name)
        return make_video_writer(temp_path, self.fps, self.size)

    def _default_output_path(self) -> str:
        source_dir = os.path.dirname(self.video_path)
        if self.is_picture:
            return os.path.join(source_dir, "no_sub", f"{self.vd_name}{self.ext}")
        return os.path.abspath(os.path.join(source_dir, f"{self.vd_name}_no_sub.mp4"))

    @cached_property
    def lama_inpaint(self) -> LamaInpaint:
        model_path = os.path.join(self.model_config.LAMA_MODEL_DIR, "big-lama.pt")
        accelerator = self.hardware_accelerator
        device = (
            accelerator.device
            if accelerator.has_cuda() or accelerator.has_mps()
            else torch.device("cpu")
        )
        return LamaInpaint(device, model_path)

    @cached_property
    def sttn_det_inpaint(self) -> STTNDetInpaint:
        return STTNDetInpaint(
            self.hardware_accelerator.device, self.model_config.STTN_DET_MODEL_PATH
        )

    # ------------------------------------------------------------------
    # Progress / output plumbing (replaced by the GUI at runtime)
    # ------------------------------------------------------------------

    def append_output(self, *args) -> None:
        print(*args)

    def update_preview_with_comp(self, frame_ori, frame_comp) -> None:
        """Preview hook; the GUI swaps this for a real implementation."""

    def add_progress_listener(self, listener) -> None:
        if listener not in self.progress_listeners:
            self.progress_listeners.append(listener)

    def remove_progress_listener(self, listener) -> None:
        if listener in self.progress_listeners:
            self.progress_listeners.remove(listener)

    def notify_progress_listeners(self) -> None:
        for listener in self.progress_listeners:
            try:
                listener(self.progress_total, self.isFinished)
            except Exception:
                traceback.print_exc()

    def update_progress(self, tbar, increment: int) -> None:
        tbar.update(increment)
        self.progress_remover = int(tbar.n / tbar.total * 100) if tbar.total else 0
        self.progress_total = self.progress_remover
        self.notify_progress_listeners()

    def _preview_masked(self, frame: np.ndarray, mask: np.ndarray, comp: np.ndarray) -> None:
        """Show the mask highlighted on the original next to the result."""
        highlighted = np.clip(frame + mask[:, :, np.newaxis] * 0.3, 0, 255).astype(np.uint8)
        self.update_preview_with_comp(highlighted, comp)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run(self) -> None:
        start_time = time.time()
        if not self.sub_areas:
            self.append_output(tr["Main"]["FullScreenProcessingNote"])
            self.sub_areas.append((0, self.frame_height, 0, self.frame_width))
        self.append_output(tr["Main"]["SubtitleArea"].format(self.sub_areas))
        self.append_output(tr["Main"]["ABSection"].format(self._ab_sections_text()))

        accelerator = self.hardware_accelerator
        if (
            accelerator.has_accelerator()
            and accelerator.accelerator_name == "DirectML"
            and config.inpaintMode.value not in (InpaintMode.STTN_AUTO, InpaintMode.STTN_DET)
        ):
            self.append_output(tr["Main"]["DirectMLWarning"])

        os.makedirs(os.path.dirname(self.video_out_path), exist_ok=True)
        self.progress_total = 0
        tbar = tqdm(
            total=int(self.frame_count), unit="frame", position=0,
            file=sys.__stdout__, desc="Subtitle Removing",
        )

        try:
            if self.is_picture:
                self._process_picture(tbar)
            else:
                self._process_video(tbar)
                # The writer must be flushed before the audio is muxed in.
                self.video_writer.release()
                self.merge_audio_to_video()

            self.append_output(tr["Main"]["FinishedProcessing"].format(self.video_out_path))
            self.append_output(
                tr["Main"]["ProcessingTime"].format(round(time.time() - start_time))
            )
            self.isFinished = True
            self.progress_total = 100
        finally:
            # Always release the capture, prefetcher thread, writer (ffmpeg
            # subprocess) and temp file, even if processing raised partway.
            if self._active_reader is not None:
                try:
                    self._active_reader.stop()
                except Exception:
                    pass
            self.video_cap.release()
            try:
                self.video_writer.release()  # idempotent if already released above
            except Exception:
                pass
            self._remove_temp_file()

    def _ab_sections_text(self) -> str:
        if self.ab_sections:
            return str(self.ab_sections).replace("range", "")
        return tr["Main"]["ABSectionAll"]

    def _process_picture(self, tbar) -> None:
        original = read_image(self.video_path)
        if original is None:
            self.append_output(tr["Main"]["ReadImageFailed"].format(self.video_path))
            return

        sub_detector = SubtitleDetect(self.video_path, self.sub_areas)
        boxes = sub_detector.detect_subtitle(original)
        del sub_detector
        gc.collect()

        if boxes:
            mask = create_mask(original.shape[:2], boxes)
            inpainted = self.lama_inpaint.inpaint(original, mask)
            self._preview_masked(original, mask, inpainted)
        else:
            inpainted = original
            self.update_preview_with_comp(original, inpainted)

        try:
            ok, buf = cv2.imencode(self.ext, inpainted)
        except cv2.error:
            ok = False
        if not ok:
            # Some decodable formats (gif/heic/webp) cannot be re-encoded by
            # OpenCV; fall back to PNG so a result is still produced.
            self.video_out_path = os.path.splitext(self.video_out_path)[0] + ".png"
            _, buf = cv2.imencode(".png", inpainted)
        buf.tofile(self.video_out_path)
        tbar.update(1)
        self.progress_total = 100

    def _process_video(self, tbar) -> None:
        self._log_models()
        mode = config.inpaintMode.value
        if mode == InpaintMode.PROPAINTER:
            self._run_propainter(tbar)
        elif mode == InpaintMode.STTN_AUTO:
            self._run_sttn_auto(tbar)
        elif mode == InpaintMode.STTN_DET:
            self._run_detection_inpaint(tbar, self.sttn_det_inpaint)
        elif mode == InpaintMode.LAMA:
            self._run_detection_inpaint(tbar, self.lama_inpaint)
        elif mode == InpaintMode.OPENCV:
            self._run_detection_inpaint(tbar, OpenCVInpaint())
        else:
            raise ValueError(f"inpaint mode not implemented: {mode}")

    def _log_models(self) -> None:
        mode = config.inpaintMode.value
        model_name = list(tr["InpaintMode"].values())[list(InpaintMode).index(mode)]
        accelerator = self.hardware_accelerator
        device_name = "CPU"
        if mode != InpaintMode.OPENCV and accelerator.has_accelerator():
            if (
                accelerator.accelerator_name == "DirectML"
                and mode in (InpaintMode.STTN_AUTO, InpaintMode.STTN_DET)
            ):
                device_name = "DirectML"
            if accelerator.has_cuda() or accelerator.has_mps():
                device_name = accelerator.accelerator_name
        self.append_output(
            tr["Main"]["SubtitleRemoverModel"].format(f"{model_name} ({device_name})")
        )

        providers = ", ".join(accelerator.onnx_providers)
        providers_text = f" ({providers})" if providers else ""
        detect_mode = config.subtitleDetectMode.value
        detect_name = list(tr["SubtitleDetectMode"].values())[
            list(SubtitleDetectMode).index(detect_mode)
        ]
        self.append_output(
            tr["Main"]["SubtitleDetectionModel"].format(f"{detect_name}{providers_text}")
        )

    # ------------------------------------------------------------------
    # Mode: STTN auto (no detection)
    # ------------------------------------------------------------------

    def _run_sttn_auto(self, tbar) -> None:
        """Erase the user-selected areas in every frame without OCR."""
        self.append_output(tr["Main"]["ProcessingStartRemovingSubtitles"])
        boxes = [(xmin, xmax, ymin, ymax) for ymin, ymax, xmin, xmax in self.sub_areas]
        mask = create_mask(self.mask_size, boxes)
        inpainter = STTNAutoInpaint(
            self.hardware_accelerator.device,
            self.model_config.STTN_AUTO_MODEL_PATH,
            self.video_path,
        )
        inpainter(input_mask=mask, input_sub_remover=self, tbar=tbar)

    # ------------------------------------------------------------------
    # Mode: detection-driven inpainting (STTN det / LaMa / OpenCV)
    # ------------------------------------------------------------------

    def _detect_subtitle_runs(self, for_propainter: bool = False):
        """OCR the video and return (frame->boxes, list of frame runs)."""
        sub_detector = SubtitleDetect(self.video_path, self.sub_areas)
        frame_boxes = sub_detector.find_subtitle_frame_no(sub_remover=self)
        if not frame_boxes:
            raise Exception(tr["Main"]["NoSubtitleDetected"].format(self.video_path))

        runs = sub_detector.find_continuous_ranges_with_same_mask(frame_boxes)
        if for_propainter:
            scene_cuts = sub_detector.get_scene_div_frame_no(self.video_path)
            runs = sub_detector.split_range_by_scene(runs, scene_cuts)
        del sub_detector
        gc.collect()
        return frame_boxes, runs

    def _run_detection_inpaint(self, tbar, model) -> None:
        """Detect subtitle runs, then inpaint each run with a combined mask."""
        frame_boxes, runs = self._detect_subtitle_runs()
        tbar.write(f"Subtitle detected: {runs}")

        backward = config.subtitleTimelineBackwardFrameCount.value
        forward = config.subtitleTimelineForwardFrameCount.value
        runs = expand_frame_ranges(runs, backward, forward)
        tbar.write(f"Subtitle timeline expand ({backward} <- -> {forward}): {runs}")

        runs = SubtitleDetect.filter_and_merge_intervals(
            runs, config.sttnReferenceLength.value
        )
        tbar.write(f"Subtitle filter_and_merge_intervals: {runs}")

        # Do NOT clamp run ends to self.frame_count: CAP_PROP_FRAME_COUNT is an
        # estimate that can UNDER-count (VBR/streamed mp4), and detection reads
        # the real frames, so clamping would truncate a legitimate run and leave
        # its later frames un-erased. The inner-loop EOF break below is what
        # actually guards the prefetcher-sentinel deadlock.
        run_end_by_start: Dict[int, int] = {start: end for start, end in runs}

        self.append_output(tr["Main"]["ProcessingStartRemovingSubtitles"])
        reader = FramePrefetcher(self.video_cap)
        self._active_reader = reader
        frame_index = 0
        while True:
            ret, frame = reader.read()
            if not ret:
                break
            frame_index += 1

            if frame_index not in run_end_by_start:
                self.video_writer.write(frame)
                self.update_progress(tbar, increment=1)
                self.update_preview_with_comp(frame, frame)
                continue

            run_start, run_end = frame_index, run_end_by_start[frame_index]
            tbar.write(f"processing frame {run_start} to {run_end}")
            run_frames = [frame]
            eof = False
            for _ in range(run_end - run_start):
                ret, frame = reader.read()
                if not ret:
                    eof = True
                    break
                frame_index += 1
                run_frames.append(frame)

            mask = create_mask(self.mask_size, self._collect_run_boxes(frame_boxes, run_start, run_end))
            for batch in batch_generator(run_frames, config.getSttnMaxLoadNum()):
                inpainted_frames = model(batch, mask)
                for original, inpainted in zip(batch, inpainted_frames):
                    self.video_writer.write(inpainted)
                    self._preview_masked(original, mask, inpainted)
                self.update_progress(tbar, increment=len(batch))
            # If the inner read hit EOF, the prefetcher's sole sentinel is spent;
            # reading again in the outer loop would block forever.
            if eof:
                break
        reader.stop()

    def _collect_run_boxes(self, frame_boxes, run_start: int, run_end: int):
        """Union of all text boxes seen during a run, minus implausible ones.

        A box that is much taller than it is wide is almost never a subtitle
        line, so it is treated as a false detection.
        """
        max_tall_box = config.subtitleYXAxisDifferencePixel.value
        boxes = []
        # The run is inclusive of run_end (its frame is read and inpainted), so
        # its boxes must be in the mask too — otherwise a subtitle that only
        # appears on the final frame survives.
        for frame_no in range(run_start, run_end + 1):
            for box in frame_boxes.get(frame_no, []):
                xmin, xmax, ymin, ymax = box
                if (ymax - ymin) - (xmax - xmin) > max_tall_box:
                    continue
                if box not in boxes:
                    boxes.append(box)
        return boxes

    # ------------------------------------------------------------------
    # Mode: ProPainter
    # ------------------------------------------------------------------

    def _run_propainter(self, tbar) -> None:
        """Detection-driven ProPainter; single frames fall back to LaMa."""
        frame_boxes, runs = self._detect_subtitle_runs(for_propainter=True)
        # Not clamped to self.frame_count (see _run_detection_inpaint): an
        # under-counting estimate would truncate a run and leave subtitles in.
        # The inner-loop EOF break guards the prefetcher-sentinel deadlock.
        run_end_by_start: Dict[int, int] = {start: end for start, end in runs}

        # ProPainter's ops (grid_sample, FFT, attention) all run on Apple MPS as
        # of torch 2.x — verified bit-parity with CPU — so it no longer needs the
        # CUDA-only fallback. fp16 stays CUDA-only (half grid_sample/attention can
        # NaN on MPS), so MPS runs fp32 at identical quality, ~8x faster than CPU.
        device = (
            self.hardware_accelerator.device
            if self.hardware_accelerator.has_cuda() or self.hardware_accelerator.has_mps()
            else torch.device("cpu")
        )
        propainter = PropainterInpaint(
            device, self.model_config.PROPAINTER_MODEL_DIR,
            config.propainterMaxLoadNum.value,
        )
        self.append_output(tr["Main"]["ProcessingStartRemovingSubtitles"])

        reader = FramePrefetcher(self.video_cap)
        self._active_reader = reader
        frame_index = 0
        while True:
            ret, frame = reader.read()
            if not ret:
                break
            frame_index += 1

            if frame_index not in frame_boxes:
                self.video_writer.write(frame)
                self.update_progress(tbar, increment=1)
                self.update_preview_with_comp(frame, frame)
                continue

            run_end = run_end_by_start.get(frame_index, -1)
            if run_end == -1:
                # Subtitled frame outside any run start: keep as-is.
                self.video_writer.write(frame)
                self.update_progress(tbar, increment=1)
                continue

            run_start = frame_index
            run_frames = [frame]
            eof = False
            while frame_index < run_end:
                ret, frame = reader.read()
                if not ret:
                    eof = True
                    break
                frame_index += 1
                run_frames.append(frame)

            mask = create_mask(self.mask_size, frame_boxes[run_start])
            for batch in batch_generator(run_frames, config.propainterMaxLoadNum.value):
                if len(batch) == 1:
                    # ProPainter needs temporal context; LaMa handles singles.
                    inpainted = self.lama_inpaint.inpaint(batch[0], mask)
                    self.video_writer.write(inpainted)
                else:
                    batch = list(batch)
                    for original, inpainted in zip(batch, propainter(batch, mask)):
                        self.video_writer.write(inpainted)
                        self._preview_masked(original, mask, inpainted)
                self.update_progress(tbar, increment=len(batch))
            # Inner read hit EOF -> sentinel spent -> stop before the outer read blocks.
            if eof:
                break
        reader.stop()

    # ------------------------------------------------------------------
    # Audio merge
    # ------------------------------------------------------------------

    def merge_audio_to_video(self) -> None:
        """Extract the source audio track and mux it into the output video."""
        audio_temp = tempfile.NamedTemporaryFile(suffix=".aac", delete=False)
        ffmpeg = FFmpegCLI.instance().ffmpeg_path
        use_shell = os.name == "nt"

        extract_cmd = [
            ffmpeg, "-y", "-i", self.video_path,
            "-acodec", "copy", "-vn", "-loglevel", "error", audio_temp.name,
        ]
        try:
            subprocess.check_output(
                extract_cmd, stdin=subprocess.DEVNULL, shell=use_shell, timeout=600
            )
        except Exception as e:
            traceback.print_exc()
            self.append_output(tr["Main"]["FailToExtractAudio"].format(str(e)))
            self._finalize_merge(audio_temp)
            return

        if os.path.exists(self.video_temp_file.name):
            merge_cmd = [
                ffmpeg, "-y", "-i", self.video_temp_file.name, "-i", audio_temp.name,
                "-vcodec", "copy", "-acodec", "copy",
                "-loglevel", "error", self.video_out_path,
            ]
            try:
                subprocess.check_output(
                    merge_cmd, stdin=subprocess.DEVNULL, shell=use_shell, timeout=600
                )
            except Exception as e:
                traceback.print_exc()
                self.append_output(tr["Main"]["FailToMergeAudio"].format(str(e)))
                self._finalize_merge(audio_temp)
                return

        if os.path.exists(audio_temp.name):
            try:
                os.remove(audio_temp.name)
            except Exception:
                pass
        self.is_successful_merged = True
        self._finalize_merge(audio_temp)

    def _finalize_merge(self, audio_temp) -> None:
        """If muxing failed, ship the silent video rather than nothing."""
        audio_temp.close()
        if not self.is_successful_merged:
            try:
                shutil.copy2(self.video_temp_file.name, self.video_out_path)
            except IOError as e:
                self.append_output(
                    tr["Main"]["CopyFileFailed"].format(
                        self.video_temp_file.name, self.video_out_path, str(e)
                    )
                )
        self.video_temp_file.close()

    def _remove_temp_file(self) -> None:
        if os.path.exists(self.video_temp_file.name):
            try:
                os.remove(self.video_temp_file.name)
            except Exception:
                pass


def main() -> None:
    multiprocessing.set_start_method("spawn")
    from backend.tools.args_handler import parse_args

    args = parse_args()
    config.set(config.interface, "en")
    translation_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "interface", f"{config.interface.value}.ini"
    )
    tr.read(translation_file, encoding="utf-8")

    if not is_video_or_image(args.input):
        print(f"Error: {args.input} is not a supported video/image or is corrupted.")
        sys.exit(-1)

    remover = SubtitleRemover(args.input)
    remover.sub_areas = args.subtitle_area_coords
    if args.output:
        remover.video_out_path = args.output
    config.inpaintMode.value = args.inpaint_mode
    remover.run()


if __name__ == "__main__":
    main()
