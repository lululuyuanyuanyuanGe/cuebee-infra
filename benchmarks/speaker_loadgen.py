"""Synthetic cross-session load generator for batching and queue metrics."""

from __future__ import annotations

import argparse
import json
import math

from cuebee.event_schema import AudioChunk
from cuebee.speaker.service import SpeakerService
from cuebee.speaker.worker import DeterministicEmbeddingWorker


def run(sessions: int, chunks_per_session: int, batch_size: int) -> dict[str, object]:
    worker = DeterministicEmbeddingWorker()
    service = SpeakerService([worker])
    service.batcher.max_batch_size = batch_size
    now_ms = 0
    output_segments = 0
    for chunk_index in range(chunks_per_session):
        for session_index in range(sessions):
            samples = tuple(
                0.2 * math.sin((sample + session_index) * 0.12)
                for sample in range(160)
            )
            service.submit(
                AudioChunk(
                    session_id=f"session-{session_index}",
                    chunk_id=f"{session_index}-{chunk_index}",
                    start_ms=now_ms,
                    end_ms=now_ms + 500,
                    samples=samples,
                ),
                now_ms,
            )
            output_segments += len(service.flush(now_ms))
        now_ms += 10
    output_segments += len(service.flush(now_ms + 100, force=True))
    return {
        "sessions": sessions,
        "chunks_per_session": chunks_per_session,
        "output_segments": output_segments,
        "metrics": service.metrics.snapshot(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--chunks-per-session", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(args.sessions, args.chunks_per_session, args.batch_size), indent=2))


if __name__ == "__main__":
    main()

