# human3r-teleop-runtime

A cleaner runtime wrapper around upstream Human3R for teleoperation and remote control use cases.

This repo is intentionally narrow:

- online frame-by-frame Human3R inference
- stable world-coordinate joint export
- a socket server for remote clients
- a single adapter layer that isolates upstream Human3R coupling

It is not a training repo, evaluation repo, or demo collection.

## Why this repo exists

The original Human3R project is great for research, but for teleoperation integration it currently contains too many unrelated parts:

- training code
- evaluation code
- visualization/demo scripts
- one-off experiments
- repo-internal imports that are hard to expose cleanly to another project

This repo is meant to be a smaller integration layer that can later live as:

- a standalone GitHub repo
- a submodule inside a larger teleoperation framework

## Scope

This repo currently owns:

- runtime model loading through a thin upstream adapter
- single-frame preprocessing for online inference
- recurrent streaming inference state
- world-coordinate joint export in JSON-friendly format
- a TCP socket inference server

This repo intentionally does not own:

- Human3R model training
- Human3R evaluation scripts
- dataset preprocessing
- checkpoint hosting
- SMPL / SMPL-X asset hosting

## Current structure

```text
src/human3r_teleop_runtime/
  upstream.py       # all upstream Human3R path/import coupling
  preprocess.py     # online frame preprocessing
  runtime.py        # recurrent Human3R streamer
  export.py         # world-coordinate joint export
  server.py         # socket server implementation
  socket_server.py  # CLI entrypoint
```

## Upstream dependency model

This repo does not try to fully detach from Human3R.

Instead, it keeps upstream dependency explicit and concentrated:

- `human3r_teleop_runtime.upstream` is the main integration boundary
- the repo still depends on upstream `dust3r` / `croco` / SMPL-related internals
- model checkpoints remain external

In practice, this means the repo is "cleaner" rather than "fully standalone".

## What is still required from upstream Human3R

Set `HUMAN3R_ROOT` to an upstream Human3R checkout that contains at least:

- `add_ckpt_path.py`
- `src/dust3r/...`
- `src/croco/...`
- `src/models/...`

Typical example:

```bash
export HUMAN3R_ROOT=/amax/xuedingrong/projects/Human3R
```

## Runtime dependencies in this repo

Direct Python dependencies:

- `torch`
- `opencv-python`
- `numpy`
- `roma`
- `einops`

These are declared in `pyproject.toml`.

## Runtime dependencies that still come from upstream Human3R

This repo still relies on upstream implementations for:

- `dust3r.model.ARCroco3DStereo`
- `dust3r.utils.camera`
- `dust3r.utils.geometry`
- `dust3r.utils.image`
- `dust3r.utils.smpl_layer`
- `dust3r.post_process`
- `dust3r.smpl_model`

So if upstream Human3R changes internal APIs, this repo may also need updates.

## Checkpoints and large files

Checkpoints are intentionally not stored in this repo.

That includes:

- Human3R `.pth` weights
- large exported assets
- SMPL/SMPL-X archives

You should pass checkpoint paths from outside, for example:

```bash
python -m human3r_teleop_runtime.socket_server \
  --model-path /amax/xuedingrong/projects/Human3R/src/human3r_672S.pth \
  --port 19999 \
  --upstream-root /amax/xuedingrong/projects/Human3R
```

## Basic usage

Example server launch:

```bash
export HUMAN3R_ROOT=/amax/xuedingrong/projects/Human3R

python -m human3r_teleop_runtime.socket_server \
  --model-path /amax/xuedingrong/projects/Human3R/src/human3r_672S.pth \
  --host 127.0.0.1 \
  --port 19999 \
  --device cuda \
  --size 256
```

## Network protocol

Current socket protocol is intentionally simple:

- client sends a 4-byte big-endian payload length
- client sends one JPEG-encoded frame
- server returns one JSON line per processed frame

Returned JSON includes fields like:

- `frame_id`
- `server_latency_sec`
- `persons`
- `named_joints_world`
- `root_world`
- `head_world`

## Intended integration style

The expected integration pattern is:

1. keep upstream Human3R in a separate checkout
2. include this repo as a lighter runtime-focused submodule
3. let upper-level teleoperation code depend on this repo instead of importing raw Human3R scripts

## Current status

This is an initial extraction, not the final architecture.

The current goal is:

- reduce clutter
- isolate unstable upstream imports
- provide clearer runtime entrypoints

Future cleanup can further reduce coupling by wrapping more `dust3r` internals behind stable interfaces.
