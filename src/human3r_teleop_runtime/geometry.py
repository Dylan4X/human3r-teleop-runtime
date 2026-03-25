import numpy as np
import torch


def xy_grid(
    width,
    height,
    device=None,
    origin=(0, 0),
    unsqueeze=None,
    cat_dim=-1,
    homogeneous=False,
    **arange_kw,
):
    if device is None:
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    xs, ys = [arange(o, o + s, **arange_kw) for s, o in zip((width, height), origin)]
    grid = meshgrid(xs, ys, indexing="xy")
    if homogeneous:
        grid = grid + (ones((height, width)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def geotrf(transform, points, ncol=None, norm=False):
    assert transform.ndim >= 2
    if isinstance(transform, np.ndarray):
        points = np.asarray(points)
    elif isinstance(transform, torch.Tensor):
        points = torch.as_tensor(points, dtype=transform.dtype)

    output_shape = points.shape[:-1]
    ncol = ncol or points.shape[-1]

    if (
        isinstance(transform, torch.Tensor)
        and isinstance(points, torch.Tensor)
        and transform.ndim == 3
        and points.ndim == 4
    ):
        dim = points.shape[3]
        if transform.shape[-1] == dim:
            points = torch.einsum("bij,bhwj->bhwi", transform, points)
        elif transform.shape[-1] == dim + 1:
            points = (
                torch.einsum("bij,bhwj->bhwi", transform[:, :dim, :dim], points)
                + transform[:, None, None, :dim, dim]
            )
        else:
            raise ValueError(f"bad shape for {points.shape=}")
    else:
        if transform.ndim >= 3:
            batch_dims = transform.ndim - 2
            assert transform.shape[:batch_dims] == points.shape[:batch_dims], "batch size does not match"
            transform = transform.reshape(-1, transform.shape[-2], transform.shape[-1])

            if points.ndim > transform.ndim:
                points = points.reshape(transform.shape[0], -1, points.shape[-1])
            elif points.ndim == 2:
                points = points[:, None, :]

        if points.shape[-1] + 1 == transform.shape[-1]:
            transform = transform.swapaxes(-1, -2)
            points = points @ transform[..., :-1, :] + transform[..., -1:, :]
        elif points.shape[-1] == transform.shape[-1]:
            transform = transform.swapaxes(-1, -2)
            points = points @ transform
        else:
            points = transform @ points.T
            if points.ndim >= 2:
                points = points.swapaxes(-1, -2)

    if norm:
        points = points / points[..., -1:]
        if norm != 1:
            points *= norm

    return points[..., :ncol].reshape(*output_shape, ncol)


def get_focal_length_from_field_of_view(fov=60, img_size=224):
    return img_size / (2 * np.tan(np.deg2rad(fov) / 2))


def get_camera_parameters(img_size, fov=60, p_x=None, p_y=None, device=torch.device("cpu")):
    intrinsics = torch.eye(3)
    focal = get_focal_length_from_field_of_view(fov=fov, img_size=img_size)
    intrinsics[0, 0], intrinsics[1, 1] = focal, focal

    if p_x is not None and p_y is not None:
        intrinsics[0, -1], intrinsics[1, -1] = p_x * img_size, p_y * img_size
    else:
        intrinsics[0, -1], intrinsics[1, -1] = img_size // 2, img_size // 2

    return intrinsics.unsqueeze(0).to(device)


def perspective_projection(points, intrinsics):
    projected = points / points[:, :, -1].unsqueeze(-1)
    projected = torch.einsum("bij,bkj->bki", intrinsics, projected)
    return projected[:, :, :2]


def inverse_perspective_projection(points, intrinsics, distance):
    points = torch.cat([points, torch.ones_like(points[..., :1])], -1)
    points = torch.einsum("bij,bkj->bki", torch.inverse(intrinsics), points)
    if distance is None:
        return points
    return points * distance
