import json
import socket
import struct
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .runtime import Human3RStreamer
from .upstream import get_human3r_models_root, load_human3r_model


def recv_exact(conn: socket.socket, n: int):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def send_jsonl(conn: socket.socket, obj):
    payload = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    conn.sendall(payload)


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 19999
    model_path: str = ""
    device: str = "cuda"
    size: int = 256
    use_ttt3r: bool = False
    tf32: bool = False
    warmup: bool = False
    reset_on_new_client: bool = False
    warmup_after_reset: bool = False
    log_every: int = 10
    upstream_root: str | None = None


class SocketInferenceServer:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.model = None
        self.device = None
        self.streamer = None

    def load(self):
        torch.set_grad_enabled(False)
        if self.config.device == "cuda" and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            if self.config.tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")

        self.model, self.device = load_human3r_model(
            model_path=self.config.model_path,
            device=self.config.device,
            upstream_root=self.config.upstream_root,
        )
        models_root = str(get_human3r_models_root(self.config.upstream_root))
        self.streamer = Human3RStreamer(
            model=self.model,
            device=self.device,
            size=self.config.size,
            use_ttt3r=self.config.use_ttt3r,
            tf32=self.config.tf32,
            models_root=models_root,
        )

    def maybe_warmup(self):
        if not self.streamer:
            raise RuntimeError("Server is not loaded")
        dummy = np.zeros((self.config.size, self.config.size, 3), dtype=np.uint8)
        self.streamer.push_frame_and_export_world(dummy, reset=True, update=True)

    def serve_forever(self):
        if self.streamer is None:
            self.load()

        if self.config.warmup:
            self.maybe_warmup()

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.config.host, self.config.port))
        srv.listen(1)

        print(f"Listening on {self.config.host}:{self.config.port}")

        while True:
            conn, addr = srv.accept()
            print(f"Client connected: {addr}")

            if self.config.reset_on_new_client:
                self.streamer.reset()
                if self.config.warmup_after_reset:
                    self.maybe_warmup()

            try:
                while True:
                    header = recv_exact(conn, 4)
                    if header is None:
                        break

                    msg_len = struct.unpack("!I", header)[0]
                    payload = recv_exact(conn, msg_len)
                    if payload is None:
                        break

                    arr = np.frombuffer(payload, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        send_jsonl(conn, {"error": "cv2.imdecode failed"})
                        continue

                    t0 = time.time()
                    try:
                        result = self.streamer.push_frame_and_export_world(
                            frame_bgr=frame,
                            reset=False,
                            update=True,
                        )
                        result["server_latency_sec"] = round(time.time() - t0, 4)
                        result["server_frame_idx"] = int(result.get("frame_id", -1))
                        send_jsonl(conn, result)

                        idx = result["server_frame_idx"]
                        if idx % self.config.log_every == 0:
                            print(
                                f"[frame {idx}] persons={len(result.get('persons', []))} "
                                f"latency={result['server_latency_sec']:.4f}s"
                            )
                    except Exception as exc:
                        err = {"error": str(exc)}
                        try:
                            send_jsonl(conn, err)
                        except Exception:
                            pass
                        print(f"Inference error: {exc}")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
