from typing import Any, Optional

import roma
import torch

from .geometry import geotrf
from .ops import estimate_focal_knowing_depth, pose_encoding_to_camera


RICH_JOINT_INDEX = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "neck": 12,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}

RICH_JOINT_NAMES = list(RICH_JOINT_INDEX.keys())


def tensor3_to_list(x):
    if x is None:
        return None
    return x.detach().float().cpu().tolist()


class RichWorldCoordinateExporter:
    def __init__(self, device: str):
        self.device = device
        self.smpl_layer = None
        self.num_betas = None
        self.intrinsics_cache = {}

    def _ensure_smpl_layer(self, num_betas: int):
        if self.smpl_layer is not None and self.num_betas == num_betas:
            return
        from dust3r.utils.smpl_layer import SMPL_Layer

        self.smpl_layer = SMPL_Layer(
            type="smplx",
            gender="neutral",
            num_betas=num_betas,
            kid=False,
            person_center="head",
        ).to(self.device)
        self.num_betas = num_betas

    def _get_pp_and_intrinsics(self, bsz: int, h: int, w: int, dtype: torch.dtype, device: torch.device):
        key = (h, w, dtype)
        cached = self.intrinsics_cache.get(key)
        if cached is None:
            pp = torch.tensor([w // 2, h // 2], device=device, dtype=dtype).view(1, 2)
            eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
            self.intrinsics_cache[key] = (pp, eye)
            cached = (pp, eye)
        pp, eye = cached
        return pp.expand(bsz, -1), eye.expand(bsz, -1, -1).clone()

    @torch.inference_mode()
    def export(self, frame_idx: int, pred: Optional[dict[str, Any]]) -> dict[str, Any]:
        result = {
            "frame_id": int(frame_idx),
            "joint_schema": RICH_JOINT_NAMES,
            "persons": [],
        }
        if pred is None:
            return result

        camera_pose_enc = pred.get("camera_pose")
        pts3d_self = pred.get("pts3d_in_self_view")
        smpl_shape = pred.get("smpl_shape")
        smpl_rotmat = pred.get("smpl_rotmat")
        smpl_transl = pred.get("smpl_transl")
        smpl_expression = pred.get("smpl_expression")
        smpl_id = pred.get("smpl_id")

        if any(x is None for x in (camera_pose_enc, pts3d_self, smpl_shape, smpl_rotmat, smpl_transl)):
            return result
        if smpl_shape.numel() == 0 or smpl_shape.shape[1] == 0:
            return result

        c2w = pose_encoding_to_camera(camera_pose_enc)
        bsz, height, width, _ = pts3d_self.shape
        pp, intrinsics = self._get_pp_and_intrinsics(bsz, height, width, pts3d_self.dtype, pts3d_self.device)
        focal = estimate_focal_knowing_depth(pts3d_self, pp, focal_mode="weiszfeld")
        intrinsics[:, 0, 0] = focal
        intrinsics[:, 1, 1] = focal
        intrinsics[:, 0, 2] = pp[:, 0]
        intrinsics[:, 1, 2] = pp[:, 1]

        n_humans = smpl_shape.shape[1]
        self._ensure_smpl_layer(int(smpl_shape.shape[-1]))

        smpl_rotvec = roma.rotmat_to_rotvec(smpl_rotmat[0])
        betas = smpl_shape[0]
        transl = smpl_transl[0]
        expr = smpl_expression[0] if smpl_expression is not None else None
        k_mat = intrinsics[0].expand(n_humans, -1, -1)

        smpl_out = self.smpl_layer(
            smpl_rotvec,
            betas,
            transl,
            None,
            None,
            K=k_mat,
            expression=expr,
        )
        joints_world = geotrf(c2w, smpl_out["smpl_j3d"].unsqueeze(0))[0]

        for pid in range(n_humans):
            person_joints = joints_world[pid]
            pid_val = int(smpl_id[0, pid].item()) if smpl_id is not None else pid
            named = {
                name: tensor3_to_list(person_joints[idx])
                for name, idx in RICH_JOINT_INDEX.items()
                if idx < person_joints.shape[0]
            }
            result["persons"].append(
                {
                    "id": pid_val,
                    "root_world": named.get("pelvis"),
                    "head_world": named.get("head"),
                    "left_wrist_world": named.get("left_wrist"),
                    "right_wrist_world": named.get("right_wrist"),
                    "left_ankle_world": named.get("left_ankle"),
                    "right_ankle_world": named.get("right_ankle"),
                    "named_joints_world": named,
                    "joints_world": tensor3_to_list(person_joints),
                }
            )
        return result
