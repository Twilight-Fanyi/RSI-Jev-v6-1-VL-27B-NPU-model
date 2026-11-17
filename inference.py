#!/usr/bin/env python3
"""Resident entry point for Cloudflare clef-omni on vLLM-Ascend."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18143)
    parser.add_argument("--device", choices=("npu", "cpu"), default="npu")
    args = parser.parse_args(argv)
    if not args.serve:
        parser.error("--serve is required")
    model_dir = args.model_dir.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    os.environ.setdefault("VLLM_RPC_BASE_PATH", "/tmp")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

    package_dir = Path(__file__).resolve().parent
    shared_dir = package_dir.parent / "laya_service"
    if shared_dir.is_dir():
        sys.path.insert(0, str(shared_dir))
    sys.path.insert(0, str(package_dir))
    from service_adapter import Adapter
    from service_runtime import make_server

    adapter = Adapter(model_dir, output_dir, args.device)
    server, info = make_server(adapter, args.host, args.port, output_dir)
    (output_dir / "service-info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"ready: http://{args.host}:{server.server_port} model={info['model']} revision={info['revision']}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
