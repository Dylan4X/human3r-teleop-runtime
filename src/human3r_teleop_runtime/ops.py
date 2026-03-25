import numpy as np
import torch
import torch.nn.functional as F

from .geometry import xy_grid


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    out = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return out.reshape(quaternions.shape[:-1] + (3, 3))


def pose_encoding_to_camera(pose_encoding, pose_encoding_type="absT_quaR"):
    if pose_encoding_type != "absT_quaR":
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    abs_t = pose_encoding[:, :3]
    quaternion_r = pose_encoding[:, 3:7]
    rotation = quaternion_to_matrix(quaternion_r)

    c2w = torch.eye(4, 4, dtype=rotation.dtype, device=rotation.device)[None].repeat(len(rotation), 1, 1)
    c2w[:, :3, :3] = rotation
    c2w[:, :3, 3] = abs_t
    return c2w


def estimate_focal_knowing_depth(pts3d, pp, focal_mode="median", min_focal=0.0, max_focal=np.inf):
    batch, height, width, channels = pts3d.shape
    assert channels == 3

    pixels = xy_grid(width, height, device=pts3d.device).view(1, -1, 2) - pp.view(-1, 1, 2)
    pts3d = pts3d.flatten(1, 2)

    if focal_mode == "median":
        with torch.no_grad():
            u, v = pixels.unbind(dim=-1)
            x, y, z = pts3d.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y
            focal_votes = torch.cat((fx_votes.view(batch, -1), fy_votes.view(batch, -1)), dim=-1)
            focal = torch.nanmedian(focal_votes, dim=-1).values
    elif focal_mode == "weiszfeld":
        xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(posinf=0, neginf=0)
        dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
        dot_xy_xy = xy_over_z.square().sum(dim=-1)
        focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

        for _ in range(10):
            dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
            weights = dis.clip(min=1e-8).reciprocal()
            focal = (weights * dot_xy_px).mean(dim=1) / (weights * dot_xy_xy).mean(dim=1)
    else:
        raise ValueError(f"bad focal_mode={focal_mode}")

    focal_base = max(height, width) / (2 * np.tan(np.deg2rad(60) / 2))
    return focal.clip(min=min_focal * focal_base, max=max_focal * focal_base)


def unpad_uv(uv, original_size, target_height, target_width):
    max_target = max(target_height, target_width)
    scale_factor = max_target / original_size
    uv_scaled = uv * scale_factor

    pad_left = (max_target - target_width) // 2
    pad_top = (max_target - target_height) // 2
    offset = torch.tensor([pad_left, pad_top], dtype=uv.dtype, device=uv.device)

    uv_transformed = uv_scaled - offset
    uv_transformed[..., 0] = torch.clamp(uv_transformed[..., 0], 0, target_width - 1)
    uv_transformed[..., 1] = torch.clamp(uv_transformed[..., 1], 0, target_height - 1)
    return uv_transformed


def log_sinkhorn_iterations(z, log_mu, log_nu, iters):
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(z + u.unsqueeze(2), dim=1)
    return z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters):
    batch, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(batch, m, 1)
    bins1 = alpha.expand(batch, 1, n)
    alpha = alpha.expand(batch, 1, 1)

    couplings = torch.cat(
        [
            torch.cat([scores, bins0], -1),
            torch.cat([bins1, alpha], -1),
        ],
        1,
    )

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu = log_mu[None].expand(batch, -1)
    log_nu = log_nu[None].expand(batch, -1)

    z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    return z - norm


def apply_threshold(det_thresh, scores):
    if isinstance(det_thresh, list):
        det_thresh = det_thresh[0]
    return torch.where(scores >= det_thresh)


def nms(heat, kernel=3):
    if kernel not in [2, 4]:
        pad = (kernel - 1) // 2
    elif kernel == 2:
        pad = 1
    else:
        pad = 2

    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    if hmax.shape[2] > heat.shape[2]:
        hmax = hmax[:, :, : heat.shape[2], : heat.shape[3]]
    keep = (hmax == heat).float()
    return heat * keep
