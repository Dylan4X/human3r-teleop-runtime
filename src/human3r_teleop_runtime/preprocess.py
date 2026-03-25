from typing import Any, Optional

import cv2
import torch
import torch.nn.functional as F


class FastPreprocessor:
    def __init__(self, device: str, size: int, img_res: Optional[int], get_camera_parameters_fn):
        self.device = device
        self.size = size
        self.img_res = img_res
        self.get_camera_parameters_fn = get_camera_parameters_fn

        self.mean05 = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std05 = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.eye4 = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
        self.img_mask_true = torch.tensor([True], device=device)
        self.ray_mask_false = torch.tensor([False], device=device)
        self.true_update = torch.tensor([True], device=device)
        self.false_update = torch.tensor([False], device=device)
        self.true_reset = torch.tensor([True], device=device)
        self.false_reset = torch.tensor([False], device=device)
        self.ray_map_cache: dict[tuple[int, int], torch.Tensor] = {}
        self.k_mhmr_cache: Optional[torch.Tensor] = None

    @staticmethod
    def _resize_frame(frame_bgr, target_long_edge: int):
        h, w = frame_bgr.shape[:2]
        long_edge = max(h, w)
        scale = float(target_long_edge) / float(long_edge)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        interp = cv2.INTER_LANCZOS4 if long_edge > target_long_edge else cv2.INTER_CUBIC
        return cv2.resize(frame_bgr, (new_w, new_h), interpolation=interp)

    def _crop_like_original(self, frame_bgr):
        h0, w0 = frame_bgr.shape[:2]
        if self.size == 224:
            resized = self._resize_frame(frame_bgr, round(self.size * max(w0 / h0, h0 / w0)))
        else:
            resized = self._resize_frame(frame_bgr, self.size)

        h, w = resized.shape[:2]
        cx, cy = w // 2, h // 2
        if self.size == 224:
            half = min(cx, cy)
            x0, x1 = cx - half, cx + half
            y0, y1 = cy - half, cy + half
        else:
            halfw = ((2 * cx) // 16) * 8
            halfh = ((2 * cy) // 16) * 8
            if w == h:
                halfh = int(3 * halfw / 4)
            x0, x1 = cx - halfw, cx + halfw
            y0, y1 = cy - halfh, cy + halfh
        return resized[y0:y1, x0:x1].copy()

    def _get_ray_map(self, h: int, w: int, dtype: torch.dtype) -> torch.Tensor:
        key = (h, w)
        ray = self.ray_map_cache.get(key)
        if ray is None or ray.dtype != dtype:
            ray = torch.full((1, 6, h, w), torch.nan, dtype=dtype, device=self.device)
            self.ray_map_cache[key] = ray
        return ray

    def _pad_to_square(self, img: torch.Tensor, img_res: int) -> torch.Tensor:
        _, _, h, w = img.shape
        if h == img_res and w == img_res:
            return img
        pad_h = max(0, img_res - h)
        pad_w = max(0, img_res - w)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (left, right, top, bottom), mode="constant", value=0.0)
        if img.shape[-2] != img_res or img.shape[-1] != img_res:
            img = F.interpolate(img, size=(img_res, img_res), mode="bilinear", align_corners=False)
        return img

    @torch.inference_mode()
    def prepare_view(self, frame_bgr, frame_idx: int, reset: bool, update: bool) -> dict[str, Any]:
        cropped_bgr = self._crop_like_original(frame_bgr)
        h, w = cropped_bgr.shape[:2]

        cpu = torch.from_numpy(cropped_bgr)
        img = cpu.to(self.device, non_blocking=True)
        img = img[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).contiguous()
        img = img.to(dtype=torch.float32).div_(255.0)
        img = img.sub_(self.mean05).div_(self.std05)

        view = {
            "img": img,
            "ray_map": self._get_ray_map(h, w, img.dtype),
            "true_shape": torch.tensor([[h, w]], dtype=torch.int32, device=self.device),
            "idx": frame_idx,
            "instance": str(frame_idx),
            "camera_pose": self.eye4,
            "img_mask": self.img_mask_true,
            "ray_mask": self.ray_mask_false,
            "update": self.true_update if update else self.false_update,
            "reset": self.true_reset if reset else self.false_reset,
        }

        if self.img_res is not None:
            view["img_mhmr"] = self._pad_to_square(img, self.img_res)
            if self.k_mhmr_cache is None:
                self.k_mhmr_cache = self.get_camera_parameters_fn(self.img_res, device=self.device)
            view["K_mhmr"] = self.k_mhmr_cache

        return view
