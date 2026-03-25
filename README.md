# human3r-teleop-runtime

Minimal runtime wrapper around upstream Human3R for teleoperation and remote control use cases.

This repo is intentionally narrow:

- online frame-by-frame Human3R inference
- stable world-coordinate joint export
- socket server for remote clients
- a single adapter layer that depends on upstream Human3R internals

It is not a training repo, evaluation repo, or visualization playground.

## Design

The package keeps upstream Human3R as a dependency instead of trying to fork it completely.
All upstream coupling is concentrated in `human3r_teleop_runtime.upstream`.

Current exported layers:

- `Human3RStreamer`: online recurrent inference
- `RichWorldCoordinateExporter`: stable JSON-friendly world joint export
- `SocketInferenceServer`: TCP server for JPEG frame streaming

## Expected upstream layout

Set `HUMAN3R_ROOT` to a checkout of the upstream Human3R project that contains:

- `add_ckpt_path.py`
- `src/dust3r/...`
- `src/croco/...`

Example:

```bash
export HUMAN3R_ROOT=/amax/xuedingrong/projects/Human3R
python -m human3r_teleop_runtime.socket_server \
  --model-path /amax/xuedingrong/projects/Human3R/src/human3r_672S.pth \
  --port 19999
```

## Runtime dependencies

- `torch`
- `opencv-python`
- `numpy`
- `roma`
- `einops`

Everything else should come from upstream Human3R.

## Notes

- This repo is meant to be clean enough to live as a GitHub repo and be used as a submodule.
- It does not attempt to fully decouple from Human3R yet.
- The current goal is to isolate protocol and inference runtime from training / eval / demo clutter.
