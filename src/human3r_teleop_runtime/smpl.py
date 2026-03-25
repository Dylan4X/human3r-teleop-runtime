from pathlib import Path

import roma
import smplx
import torch
from smplx.joint_names import JOINT_NAMES
from torch import nn

from .geometry import inverse_perspective_projection, perspective_projection
from .upstream import get_human3r_models_root


class SMPLLayer(nn.Module):
    def __init__(
        self,
        model_root: str | Path | None = None,
        model_type: str = "smplx",
        gender: str = "neutral",
        num_betas: int = 10,
        kid: bool = False,
        person_center: str | None = None,
    ):
        super().__init__()
        assert model_type == "smplx"

        self.model_type = model_type
        self.kid = kid
        self.num_betas = num_betas

        models_root = Path(model_root) if model_root is not None else get_human3r_models_root()
        self.models_root = models_root
        self.bm_x = smplx.create(
            str(models_root),
            "smplx",
            gender=gender,
            use_pca=False,
            flat_hand_mean=True,
            num_betas=num_betas,
        )

        self.joint_names = JOINT_NAMES[:127]
        self.person_center = person_center
        self.person_center_idx = None
        if self.person_center is not None:
            self.person_center_idx = self.joint_names.index(self.person_center)

    def forward(
        self,
        pose,
        shape,
        transl,
        loc,
        dist,
        K,
        expression=None,
        K_to_proj=None,
    ):
        if loc is not None and dist is not None:
            assert pose.shape[0] == shape.shape[0] == loc.shape[0] == dist.shape[0]
            assert len(loc.shape) == 2 and list(loc.shape[1:]) == [2]
            assert len(dist.shape) == 2 and list(dist.shape[1:]) == [1]

        assert len(pose.shape) == 3 and list(pose.shape[1:]) == [53, 3]
        assert len(shape.shape) == 2 and (
            list(shape.shape[1:]) == [self.num_betas] or list(shape.shape[1:]) == [self.num_betas + 1]
        )
        assert transl is not None or (loc is not None and dist is not None)

        batch_size = pose.shape[0]
        if batch_size == 0:
            return {}

        kwargs_pose = {
            "betas": shape,
            "global_orient": self.bm_x.global_orient.repeat(batch_size, 1),
            "body_pose": pose[:, 1:22].flatten(1),
            "left_hand_pose": pose[:, 22:37].flatten(1),
            "right_hand_pose": pose[:, 37:52].flatten(1),
            "jaw_pose": pose[:, 52:53].flatten(1),
            "expression": expression.flatten(1) if expression is not None else self.bm_x.expression.repeat(batch_size, 1),
            "leye_pose": self.bm_x.leye_pose.repeat(batch_size, 1),
            "reye_pose": self.bm_x.reye_pose.repeat(batch_size, 1),
        }

        output = self.bm_x(**kwargs_pose)
        verts = output.vertices
        joints = output.joints
        rotation = roma.rotvec_to_rotmat(pose[:, 0])

        pelvis = joints[:, [0]]
        joints = (rotation.unsqueeze(1) @ (joints - pelvis).unsqueeze(-1)).squeeze(-1)
        verts = (rotation.unsqueeze(1) @ (verts - pelvis).unsqueeze(-1)).squeeze(-1)

        if transl is None:
            if K.dtype == torch.float16:
                transl = inverse_perspective_projection(loc.unsqueeze(1).float(), K.float(), dist.unsqueeze(1).float())[:, 0]
                transl = transl.half()
            else:
                transl = inverse_perspective_projection(loc.unsqueeze(1), K, dist.unsqueeze(1))[:, 0]

        transl_up = transl.clone()
        if self.person_center_idx is None:
            transl_up = transl_up + pelvis[:, 0]
        else:
            person_center = joints[:, [self.person_center_idx]]
            verts = verts - person_center
            joints = joints - person_center

        joints_cam = joints + transl_up.unsqueeze(1)
        verts_cam = verts + transl_up.unsqueeze(1)

        if K_to_proj is None:
            K_to_proj = K

        joints_2d = perspective_projection(joints_cam, K_to_proj)
        verts_2d = perspective_projection(verts_cam, K_to_proj)

        return {
            "smpl_v3d": verts_cam,
            "smpl_j3d": joints_cam,
            "smpl_j2d": joints_2d,
            "smpl_v2d": verts_2d,
            "smpl_transl": transl,
            "smpl_transl_pelvis": joints_cam[:, [0]],
        }
