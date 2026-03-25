import os
import sys
from pathlib import Path

import torch


def _resolve_upstream_root(upstream_root: str | None = None) -> Path:
    root = upstream_root or os.environ.get("HUMAN3R_ROOT")
    if not root:
        raise ValueError("HUMAN3R_ROOT is not set and upstream_root was not provided")
    path = Path(root).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def ensure_human3r_imports(upstream_root: str | None = None) -> Path:
    root = _resolve_upstream_root(upstream_root)
    src_root = root / "src"

    for item in (root, src_root):
        item_str = str(item)
        if item_str not in sys.path:
            sys.path.insert(0, item_str)

    add_ckpt_path = root / "add_ckpt_path.py"
    if not add_ckpt_path.exists():
        raise FileNotFoundError(add_ckpt_path)

    return root


def add_checkpoint_paths(model_path: str, upstream_root: str | None = None) -> Path:
    root = ensure_human3r_imports(upstream_root)
    from add_ckpt_path import add_path_to_dust3r

    add_path_to_dust3r(model_path)
    return root


def load_human3r_model(
    model_path: str,
    device: str = "cuda",
    upstream_root: str | None = None,
):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    add_checkpoint_paths(model_path, upstream_root=upstream_root)
    from dust3r.model import ARCroco3DStereo

    model = ARCroco3DStereo.from_pretrained(model_path).to(device)
    model.eval()
    return model, device


def get_human3r_models_root(upstream_root: str | None = None) -> Path:
    root = _resolve_upstream_root(upstream_root)
    models_root = root / "src" / "models"
    if not models_root.exists():
        raise FileNotFoundError(models_root)
    return models_root
