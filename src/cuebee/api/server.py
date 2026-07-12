"""Dependency-free JSON API for local integration and replay."""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from cuebee.branch_graph import TaskKind
from cuebee.event_schema import AudioChunk, STTEvent
from cuebee.runtime import CueBeeRuntime
from cuebee.speaker.service import SpeakerService
from cuebee.speaker.worker import DeterministicEmbeddingWorker


class Application:
    def __init__(
        self,
        runtime: CueBeeRuntime | None = None,
        speaker: SpeakerService | None = None,
    ) -> None:
        self.runtime = runtime or CueBeeRuntime()
        self.speaker = speaker or SpeakerService([DeterministicEmbeddingWorker()])
        self.lock = threading.RLock()

    def handle_stt(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = dict(payload)
        data["session_id"] = session_id
        with self.lock:
            return self.runtime.handle_event(STTEvent.from_dict(data)).to_dict()

    def submit_task(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            task = self.runtime.submit_task(
                session_id=session_id,
                kind=TaskKind(payload["kind"]),
                prompt=str(payload["prompt"]),
                deadline_ms=int(payload.get("deadline_ms", 500)),
                priority=int(payload.get("priority", 0)),
                estimated_tokens=int(payload.get("max_output_tokens", 32)),
                task_id=payload.get("task_id"),
            )
            return task.snapshot()

    def tick(self) -> dict[str, Any]:
        with self.lock:
            outcome = self.runtime.run_next()
            return outcome.to_dict() if outcome else {"status": "idle"}

    def submit_audio(self, payload: dict[str, Any]) -> dict[str, Any]:
        chunk = AudioChunk(
            session_id=str(payload["session_id"]),
            chunk_id=str(payload["chunk_id"]),
            start_ms=int(payload["start_ms"]),
            end_ms=int(payload["end_ms"]),
            samples=tuple(float(value) for value in payload["samples"]),
            sample_rate=int(payload.get("sample_rate", 16_000)),
            deadline_ms=(
                int(payload["deadline_ms"]) if payload.get("deadline_ms") is not None else None
            ),
        )
        with self.lock:
            status = self.speaker.submit(chunk, int(payload.get("now_ms", chunk.end_ms)))
            return {"status": status.value, "queued_chunks": self.speaker.batcher.queued_chunks}

    def flush_audio(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            segments = self.speaker.flush(
                int(payload.get("now_ms", 0)),
                force=bool(payload.get("force", False)),
            )
            return {
                "segments": [
                    {
                        "session_id": segment.session_id,
                        "chunk_id": segment.chunk_id,
                        "start_ms": segment.start_ms,
                        "end_ms": segment.end_ms,
                        "speaker_id": segment.speaker_id,
                        "similarity": segment.similarity,
                        "embedding_quality": segment.embedding_quality,
                    }
                    for segment in segments
                ]
            }

    def session_snapshot(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            return self.runtime.snapshot(session_id)

    def metrics(self) -> str:
        with self.lock:
            return self.runtime.metrics.prometheus() + self.speaker.metrics.prometheus()


def make_handler(application: Application) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "CueBee/0.1"

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/healthz":
                self._json(HTTPStatus.OK, {"status": "ok"})
                return
            if path == "/metrics":
                body = application.metrics().encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            parts = [part for part in path.split("/") if part]
            if len(parts) == 3 and parts[:2] == ["v1", "sessions"]:
                self._run(lambda: application.session_snapshot(parts[2]))
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "route not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            payload = self._read_json()
            if payload is None:
                return
            parts = [part for part in path.split("/") if part]
            if len(parts) == 4 and parts[:2] == ["v1", "sessions"]:
                session_id, action = parts[2], parts[3]
                if action == "stt":
                    self._run(lambda: application.handle_stt(session_id, payload))
                    return
                if action == "tasks":
                    self._run(lambda: application.submit_task(session_id, payload))
                    return
            if path == "/v1/scheduler/tick":
                self._run(application.tick)
                return
            if path == "/v1/speaker/chunks":
                self._run(lambda: application.submit_audio(payload))
                return
            if path == "/v1/speaker/flush":
                self._run(lambda: application.flush_audio(payload))
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "route not found"})

        def _run(self, operation: Any) -> None:
            try:
                result = operation()
            except (KeyError, ValueError, RuntimeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._json(HTTPStatus.OK, result)

        def _read_json(self) -> dict[str, Any] | None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object")
                return payload
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return None

        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8080, application: Application | None = None) -> None:
    app = application or Application()
    server = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"CueBee listening on http://{host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CueBee local inference gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()

