"""Standard-library HTTP transport for a resident, model-specific Adapter.

The SystemOne schema follows the public OpenThai/TypeSafe contract. Protocol
version is local until the competition publishes a versioned specification.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

PROTOCOL = "systemone-compatible/local-1"
MAX_BODY = 16 * 1024 * 1024


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def native_request(body):
    """Convert the documented single-question Jev demo shape to SystemOne."""
    if "questions" in body:
        return body, None
    allowed = {"model", "kind", "state", "question", "options", "images", "audio", "audio_path", "head"}
    if set(body) - allowed:
        raise ValueError(f"Unsupported demo fields: {sorted(set(body) - allowed)}")
    if "state" not in body or not isinstance(body.get("question"), str):
        raise ValueError("Demo requests require state and a string question")
    kind = body.get("kind")
    kind = "noul" if kind in ("boolean", "noul") else kind
    if kind not in ("choice", "score", "noul"):
        raise ValueError("kind must be choice, score or noul")
    question = {"type": kind, "instructions": body["question"]}
    if "head" in body:
        question["head"] = body["head"]
    options = body.get("options")
    if kind == "choice":
        if options is None and "head" in body:
            pass  # A fixed-head model publishes its own labels in its native schema.
        elif isinstance(options, dict):
            question["criteria"] = options
        elif isinstance(options, list) and all(isinstance(x, str) for x in options):
            if len(set(options)) != len(options):
                raise ValueError("Duplicate choice options")
            question["criteria"] = {x: x for x in options}
        else:
            raise ValueError("choice options must be an object or string array")
    elif kind == "score":
        if options is not None and not (isinstance(options, list) and all(isinstance(x, str) for x in options)):
            raise ValueError("score options must be an ordered string array")
        question["criteria"] = options if options is not None else [str(i) for i in range(6)]
    elif options is not None:
        raise ValueError("noul does not accept options")
    request = {k: v for k, v in body.items() if k in {"state", "model", "images", "audio", "audio_path"}}
    request["questions"] = {"decision": question}
    return request, question


def validate_response(result, questions):
    # JSON serialization also rejects non-finite values elsewhere in the result.
    json.dumps(result, allow_nan=False)
    if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
        raise RuntimeError("Adapter did not return an answers object")
    if set(result["answers"]) != set(questions):
        raise RuntimeError("Adapter returned different question IDs")
    for answer in result["answers"].values():
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or not probabilities:
            raise RuntimeError("Adapter omitted full probabilities")
        values = list(probabilities.values())
        if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
            raise RuntimeError("Adapter returned invalid probabilities")
        total = sum(values)
        abstain = answer.get("abstain")
        separate_abstain = (isinstance(abstain, (int, float)) and not isinstance(abstain, bool)
                            and math.isfinite(abstain) and 0 <= abstain <= 1
                            and math.isclose(total + abstain, 1, abs_tol=1e-3))
        if not math.isclose(total, 1, abs_tol=1e-3) and not separate_abstain:
            raise RuntimeError("Adapter probabilities do not sum to one")
    return result


def make_server(adapter, host, port, output_dir):
    info = adapter.info()
    if not isinstance(info.get("model"), str) or not isinstance(info.get("revision"), str):
        raise RuntimeError("Adapter info must include model and revision")
    info = {**info, "protocol_version": PROTOCOL,
            "cache": {"enabled": False, "response_cache": False,
                      "scope": "HTTP response caching", "adapter": info.get("cache", {})},
            "endpoints": ["/health", "/v1/decide/info", "/v1/systemone", "/v1/decide"]}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        # One request per connection avoids reuse of unread invalid request bodies.
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt, *args):
            print(fmt % args, file=sys.stderr, flush=True)

        def send_json(self, status, body):
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/health":
                self.send_json(200, {"status": "ready", "model": info["model"], "revision": info["revision"]})
            elif path == "/v1/decide/info":
                self.send_json(200, info)
            else:
                self.send_json(404, {"error": {"code": "not_found", "message": "Unknown endpoint"}})

        def do_POST(self):
            request_id = uuid.uuid4().hex
            started = time.monotonic()
            path = urlsplit(self.path).path
            status = 500
            try:
                if path not in ("/v1/systemone", "/v1/decide"):
                    status = 404
                    self.send_json(404, {"error": {"code": "not_found", "message": "Unknown endpoint"}})
                    return
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Chunked requests are unsupported; send Content-Length")
                if self.headers.get_content_type() != "application/json":
                    status = 415
                    self.send_json(415, {"error": {"code": "content_type", "message": "Use application/json"}})
                    return
                length = int(self.headers.get("Content-Length", "-1"))
                if length < 1 or length > MAX_BODY:
                    raise ValueError(f"Content-Length must be between 1 and {MAX_BODY}")
                self.connection.settimeout(120)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete request body")
                body = json.loads(raw, parse_constant=reject_constant)
                if not isinstance(body, dict):
                    raise ValueError("Request must be a JSON object")
                body, native = native_request(body) if path == "/v1/decide" else (body, None)
                # State types and defaults belong to the pinned model protocol.
                # Several native SDKs accept scalar/null or omitted state.
                if not isinstance(body.get("questions"), dict) or not body["questions"]:
                    raise ValueError("questions must be a nonempty object keyed by question ID")
                if not all(isinstance(q, dict) for q in body["questions"].values()):
                    raise ValueError("Every question must be an object")
                if not all(isinstance(qid, str) and qid for qid in body["questions"]):
                    raise ValueError("Question IDs must be nonempty strings")
                # Preserve absent instructions for the native SDK's own default
                # or validation; injecting the question ID changes its prompt.
                if body.get("model", info["model"]) not in [info["model"], *info.get("model_aliases", [])]:
                    raise ValueError("Requested model does not match the loaded model")
                # Serialize all inference: no overlapping device use or mutable model caches.
                with lock:
                    result = validate_response(adapter.decide(body), body["questions"])
                if native is not None:
                    answer = result["answers"]["decision"]
                    result = {**result, **answer}
                    if native["type"] == "choice":
                        result["choice_index"] = next((i for i, key in enumerate(native.get("criteria", answer["probabilities"])) if key == answer.get("choice")), None)
                result["request_id"] = request_id
                status = 200
                self.send_json(status, result)
            except (ValueError, UnicodeError, TimeoutError) as exc:
                status = 400
                self.send_json(status, {"error": {"code": "invalid_request", "message": str(exc)}, "request_id": request_id})
            except Exception as exc:
                print(f"inference_failed[{request_id}]: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                self.send_json(500, {"error": {"code": "inference_failed", "message": "Model inference failed; see service log"}, "request_id": request_id})
            finally:
                # Keep transport evidence without recording user state or media.
                with lock:
                    with (output_dir / "requests.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"request_id": request_id, "path": path, "status": status,
                                            "seconds": time.monotonic() - started}) + "\n")

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server, info


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False, description="Resident SystemOne inference service")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18143)
    parser.add_argument("--device", choices=("npu", "cpu"), default="npu")
    args = parser.parse_args(argv)
    # Resolve paths before any chdir. start.sh preserves the evaluator's cwd.
    args.model_dir = args.model_dir.expanduser().resolve(strict=True)
    if not args.model_dir.is_dir():
        parser.error("--model-dir must be a directory")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    from service_adapter import Adapter
    # Complete load and patch initialization before exposing status=ready.
    adapter = Adapter(args.model_dir, args.output_dir, args.device)
    server, info = make_server(adapter, args.host, args.port, args.output_dir)
    (args.output_dir / "service-info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"ready: http://{args.host}:{server.server_port} model={info['model']} revision={info['revision']}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
