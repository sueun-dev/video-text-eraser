"""Flow-guided video inpainting with ProPainter.

Three-stage pipeline: RAFT estimates bidirectional optical flow, a recurrent
network completes the flow inside the mask, then the ProPainter transformer
propagates pixels along the completed flow and hallucinates what remains.
Highest quality on fast motion, but also the most memory-hungry mode.
"""

import gc
import os
import warnings
from typing import List, Tuple, cast

import cv2
import numpy as np
import scipy.ndimage
import torch
from PIL import Image

from backend.inpaint.video.core.utils import to_tensors
from backend.inpaint.video.model.modules.flow_comp_raft import RAFT_bi
from backend.inpaint.video.model.propainter import InpaintGenerator
from backend.inpaint.video.model.recurrent_flow_completion import RecurrentFlowCompleteNet
from backend.tools.inpaint_tools import composite_run_pde, get_inpaint_area_by_mask

warnings.filterwarnings("ignore")

# Height of the band handed to the model, as a fraction of frame width.
# Navier-Stokes blend weight (flow model + real-pixel PDE fill).
_PDE_WEIGHT = 0.7
_BAND_HEIGHT_RATIO = 3 / 16
# ProPainter's architecture requires band dimensions divisible by 8.
_SIZE_MULTIPLE = 8


def _stack_tensor(items) -> torch.Tensor:
    """Stack a list of PIL images into one tensor (typed wrapper over to_tensors())."""
    return to_tensors()(items)


def _prepare_masks(
    mask: np.ndarray, length: int, flow_mask_dilates: int = 8, mask_dilates: int = 5
) -> Tuple[List[Image.Image], List[Image.Image]]:
    """Dilate the subtitle mask into flow masks and inpaint masks.

    Flow masks are dilated further so that every pixel RAFT trusts is truly
    clean background. One mask is replicated for all frames.
    """
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask.squeeze(2)
    elif mask.ndim == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = np.array(Image.fromarray(mask).convert("L"))

    if flow_mask_dilates > 0:
        flow_mask = scipy.ndimage.binary_dilation(
            mask, iterations=flow_mask_dilates
        ).astype(np.uint8)
    else:
        flow_mask = (mask > 0).astype(np.uint8)

    if mask_dilates > 0:
        dilated = scipy.ndimage.binary_dilation(mask, iterations=mask_dilates).astype(np.uint8)
    else:
        dilated = (mask > 0).astype(np.uint8)

    flow_masks = [Image.fromarray(flow_mask * 255)] * length
    masks_dilated = [Image.fromarray(dilated * 255)] * length
    return flow_masks, masks_dilated


def _reference_frame_ids(
    mid_neighbor_id: int,
    neighbor_ids: List[int],
    length: int,
    ref_stride: int = 10,
    ref_num: int = -1,
) -> List[int]:
    """Sample global reference frames, optionally limited around the window."""
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
        return ref_index

    start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
    end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
    for i in range(start_idx, end_idx, ref_stride):
        if i not in neighbor_ids:
            if len(ref_index) > ref_num:
                break
            ref_index.append(i)
    return ref_index


class PropainterInpaint:
    """Flow-guided video inpainting (RAFT + flow completion + transformer)."""

    def __init__(
        self,
        device: torch.device,
        model_dir: str,
        sub_video_length: int = 80,
        use_fp16: bool = True,
    ) -> None:
        self.device = device
        self.model_dir = model_dir
        # fp16 only on CUDA: half-precision grid_sample/attention can NaN on MPS,
        # so MPS runs fp32 (still ~8x faster than CPU, bit-parity quality).
        self.use_half = use_fp16 and device.type == "cuda"
        # Maximum frames processed in one sub-video for long inputs.
        self.sub_video_length = sub_video_length
        # Local temporal window size for the transformer.
        self.neighbor_length = 10
        # Dilation iterations applied to the masks. Upstream ProPainter dilates
        # the flow mask more aggressively (8) than the inpaint mask (5) so RAFT
        # never samples flow across the hole boundary; matching that removes the
        # residual edge artifacts a single value of 4 left behind.
        self.flow_mask_dilation = 8
        self.mask_dilation = 5
        # Stride between sampled global reference frames.
        self.ref_stride = 10
        # RAFT refinement iterations.
        self.raft_iter = 20

        # RAFT_bi's vendored signature mistypes device as str; it accepts a torch.device.
        self.fix_raft = RAFT_bi(os.path.join(model_dir, "raft-things.pth"), device)  # type: ignore[arg-type]
        self.fix_flow_complete = self._load_flow_completion_model()
        self.model = self._load_inpaint_model()

    def _load_flow_completion_model(self) -> RecurrentFlowCompleteNet:
        model = RecurrentFlowCompleteNet(
            os.path.join(self.model_dir, "recurrent_flow_completion.pth")
        )
        for p in model.parameters():
            p.requires_grad = False
        if self.use_half:
            model = model.half()
        return model.to(self.device).eval()

    def _load_inpaint_model(self) -> InpaintGenerator:
        model = InpaintGenerator(model_path=os.path.join(self.model_dir, "ProPainter.pth"))
        if self.use_half:
            model = model.half()
        return model.to(self.device).eval()

    def inpaint(self, frames: List[np.ndarray], mask: np.ndarray) -> List[np.ndarray]:
        """Restore one band crop across a run of frames. Returns BGR arrays."""
        if isinstance(frames[0], np.ndarray):
            pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        else:
            pil_frames = cast(List[Image.Image], list(frames))
        w, h = pil_frames[0].size
        flow_mask_imgs, mask_imgs = _prepare_masks(
            mask, len(pil_frames),
            flow_mask_dilates=self.flow_mask_dilation, mask_dilates=self.mask_dilation,
        )

        frames_inp = [np.array(f).astype(np.uint8) for f in pil_frames]
        frames_t = (_stack_tensor(pil_frames).unsqueeze(0) * 2 - 1).to(self.device)
        flow_masks = _stack_tensor(flow_mask_imgs).unsqueeze(0).to(self.device)
        masks_dilated = _stack_tensor(mask_imgs).unsqueeze(0).to(self.device)
        video_length = frames_t.size(1)

        with torch.no_grad():
            gt_flows_bi = self._compute_flow(frames_t, video_length)
            if self.use_half:
                frames_t = frames_t.half()
                flow_masks = flow_masks.half()
                masks_dilated = masks_dilated.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
            pred_flows_bi = self._complete_flow(gt_flows_bi, flow_masks)
            updated_frames, updated_masks = self._propagate_images(
                frames_t, masks_dilated, pred_flows_bi, video_length, h, w
            )

        return self._run_transformer(
            frames_inp, updated_frames, updated_masks, masks_dilated,
            pred_flows_bi, video_length, h, w,
        )

    def _compute_flow(self, frames: torch.Tensor, video_length: int):
        """Estimate bidirectional optical flow with RAFT (fp32, clipped runs)."""
        width = frames.size(-1)
        if width <= 640:
            short_clip_len = 12
        elif width <= 720:
            short_clip_len = 8
        elif width <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2

        if frames.size(1) <= short_clip_len:
            gt_flows_bi = self.fix_raft(frames, iters=self.raft_iter)
            torch.cuda.empty_cache()
            return gt_flows_bi

        flows_f_list, flows_b_list = [], []
        for f in range(0, video_length, short_clip_len):
            end_f = min(video_length, f + short_clip_len)
            window = frames[:, f:end_f] if f == 0 else frames[:, f - 1:end_f]
            flows_f, flows_b = self.fix_raft(window, iters=self.raft_iter)
            flows_f_list.append(flows_f)
            flows_b_list.append(flows_b)
            torch.cuda.empty_cache()
        return torch.cat(flows_f_list, dim=1), torch.cat(flows_b_list, dim=1)

    def _complete_flow(self, gt_flows_bi, flow_masks):
        """Fill the flow field inside the mask, in sub-videos when long."""
        flow_length = gt_flows_bi[0].size(1)
        if flow_length <= self.sub_video_length:
            pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(
                gt_flows_bi, flow_masks
            )
            pred_flows_bi = self.fix_flow_complete.combine_flow(
                gt_flows_bi, pred_flows_bi, flow_masks
            )
            torch.cuda.empty_cache()
            return pred_flows_bi

        pred_flows_f, pred_flows_b = [], []
        pad_len = 5
        for f in range(0, flow_length, self.sub_video_length):
            s_f = max(0, f - pad_len)
            e_f = min(flow_length, f + self.sub_video_length + pad_len)
            pad_len_s = max(0, f) - s_f
            pad_len_e = e_f - min(flow_length, f + self.sub_video_length)
            window = (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f])
            pred_sub, _ = self.fix_flow_complete.forward_bidirect_flow(
                window, flow_masks[:, s_f:e_f + 1]
            )
            pred_sub = self.fix_flow_complete.combine_flow(
                window, pred_sub, flow_masks[:, s_f:e_f + 1]
            )
            pred_flows_f.append(pred_sub[0][:, pad_len_s:e_f - s_f - pad_len_e])
            pred_flows_b.append(pred_sub[1][:, pad_len_s:e_f - s_f - pad_len_e])
            torch.cuda.empty_cache()
        return torch.cat(pred_flows_f, dim=1), torch.cat(pred_flows_b, dim=1)

    def _propagate_images(
        self, frames, masks_dilated, pred_flows_bi, video_length, h, w
    ):
        """Warp known pixels along the completed flow into the holes."""
        masked_frames = frames * (1 - masks_dilated)
        subvideo_length = min(100, self.sub_video_length)
        if video_length <= subvideo_length:
            b, t, _, _, _ = masks_dilated.size()
            prop_imgs, updated_local_masks = self.model.img_propagation(
                masked_frames, pred_flows_bi, masks_dilated, "nearest"
            )
            updated_frames = (
                frames * (1 - masks_dilated)
                + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            )
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
            torch.cuda.empty_cache()
            return updated_frames, updated_masks

        updated_frames_list, updated_masks_list = [], []
        pad_len = 10
        for f in range(0, video_length, subvideo_length):
            s_f = max(0, f - pad_len)
            e_f = min(video_length, f + subvideo_length + pad_len)
            pad_len_s = max(0, f) - s_f
            pad_len_e = e_f - min(video_length, f + subvideo_length)
            b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
            flows_sub = (pred_flows_bi[0][:, s_f:e_f - 1], pred_flows_bi[1][:, s_f:e_f - 1])
            prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(
                masked_frames[:, s_f:e_f], flows_sub, masks_dilated[:, s_f:e_f], "nearest"
            )
            updated_frames_sub = (
                frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f])
                + prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
            )
            updated_frames_list.append(
                updated_frames_sub[:, pad_len_s:e_f - s_f - pad_len_e]
            )
            updated_masks_list.append(
                updated_local_masks_sub.view(b, t, 1, h, w)[:, pad_len_s:e_f - s_f - pad_len_e]
            )
            torch.cuda.empty_cache()
        return torch.cat(updated_frames_list, dim=1), torch.cat(updated_masks_list, dim=1)

    def _run_transformer(
        self, ori_frames, updated_frames, updated_masks, masks_dilated,
        pred_flows_bi, video_length, h, w,
    ) -> List[np.ndarray]:
        """Final transformer pass over sliding windows; blends overlaps 50/50."""
        comp_frames: List[np.ndarray] = [None] * video_length
        neighbor_stride = self.neighbor_length // 2
        ref_num = (
            self.sub_video_length // self.ref_stride
            if video_length > self.sub_video_length
            else -1
        )

        for f in range(0, video_length, neighbor_stride):
            neighbor_ids = list(
                range(max(0, f - neighbor_stride),
                      min(video_length, f + neighbor_stride + 1))
            )
            ref_ids = _reference_frame_ids(
                f, neighbor_ids, video_length, self.ref_stride, ref_num
            )
            window = neighbor_ids + ref_ids
            selected_imgs = updated_frames[:, window, :, :, :]
            selected_masks = masks_dilated[:, window, :, :, :]
            selected_update_masks = updated_masks[:, window, :, :, :]
            selected_pred_flows_bi = (
                pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
                pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :],
            )

            with torch.no_grad():
                l_t = len(neighbor_ids)
                pred_img = self.model(
                    selected_imgs, selected_pred_flows_bi, selected_masks,
                    selected_update_masks, l_t,
                )
                pred_img = pred_img.view(-1, 3, h, w)
                pred_img = ((pred_img + 1) / 2).cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = (
                    masks_dilated[0, neighbor_ids, :, :, :]
                    .cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
                )
                for i, idx in enumerate(neighbor_ids):
                    img = (
                        np.array(pred_img[i]).astype(np.uint8) * binary_masks[i]
                        + ori_frames[idx] * (1 - binary_masks[i])
                    )
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = (
                            comp_frames[idx].astype(np.float32) * 0.5
                            + img.astype(np.float32) * 0.5
                        )
                    comp_frames[idx] = comp_frames[idx].astype(np.uint8)
            torch.cuda.empty_cache()

        return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in comp_frames]

    def __call__(
        self, input_frames: List[np.ndarray], input_mask: np.ndarray
    ) -> List[np.ndarray]:
        """Restore the masked subtitle bands across a run of frames."""
        mask = input_mask[:, :, None]
        frame_height, frame_width = mask.shape[:2]
        band_height = int(frame_width * _BAND_HEIGHT_RATIO)
        bands = get_inpaint_area_by_mask(
            frame_width, frame_height, band_height, mask, multiple=_SIZE_MULTIPLE
        )

        frames = [frame.copy() for frame in input_frames]
        if not bands:
            return frames

        restored_bands = {}
        for k, (ymin, ymax, xmin, xmax) in enumerate(bands):
            cropped_frames = [frame[ymin:ymax, xmin:xmax, :] for frame in frames]
            cropped_mask = mask[ymin:ymax, xmin:xmax, :]
            restored_bands[k] = self.inpaint(cropped_frames, cropped_mask)
            del cropped_frames
            gc.collect()

        for k, (ymin, ymax, xmin, xmax) in enumerate(bands):
            box = mask[ymin:ymax, xmin:xmax, :]
            orig_bands = [frame[ymin:ymax, xmin:xmax, :].copy() for frame in frames]
            model_fills = [restored_bands[k][j] for j in range(len(frames))]
            # Blend the flow-guided fill with a temporally-fused Navier-Stokes
            # real-background interpolation of the thin strokes.
            out_bands = composite_run_pde(orig_bands, model_fills, box, _PDE_WEIGHT)
            for j, frame in enumerate(frames):
                frame[ymin:ymax, xmin:xmax, :] = out_bands[j]
        return frames
