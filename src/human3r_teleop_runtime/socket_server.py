import argparse

from .server import ServerConfig, SocketInferenceServer


def build_argparser():
    parser = argparse.ArgumentParser("Human3R teleop socket server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--use-ttt3r", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--reset-on-new-client", action="store_true")
    parser.add_argument("--warmup-after-reset", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--upstream-root", type=str, default=None)
    return parser


def main():
    args = build_argparser().parse_args()
    config = ServerConfig(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        device=args.device,
        size=args.size,
        use_ttt3r=args.use_ttt3r,
        tf32=args.tf32,
        warmup=args.warmup,
        reset_on_new_client=args.reset_on_new_client,
        warmup_after_reset=args.warmup_after_reset,
        log_every=args.log_every,
        upstream_root=args.upstream_root,
    )
    server = SocketInferenceServer(config)
    server.serve_forever()


if __name__ == "__main__":
    main()
