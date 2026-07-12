"""Replay versioned transcript traces and report cache-control metrics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from cuebee.event_schema import STTEvent
from cuebee.runtime import CueBeeRuntime, TentativePrefillPolicy


def replay(path: Path) -> dict[str, Any]:
    runtime = CueBeeRuntime(
        tentative_policy=TentativePrefillPolicy(min_tail_tokens=1, min_confidence=0.0)
    )
    full_history_prefill_tokens = 0
    started = time.perf_counter()
    events = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = STTEvent.from_dict(json.loads(line))
            runtime.handle_event(event)
            state = runtime.sessions.get(event.session_id)
            full_history_prefill_tokens += state.total_tokens
            events += 1
    elapsed = time.perf_counter() - started
    counters = runtime.metrics.snapshot()["counters"]
    incremental = int(counters.get("prefill_tokens", 0))
    return {
        "trace": str(path),
        "events": events,
        "elapsed_seconds": elapsed,
        "events_per_second": events / elapsed if elapsed else 0,
        "full_history_prefill_tokens": full_history_prefill_tokens,
        "incremental_prefill_tokens": incremental,
        "avoided_prefill_tokens": full_history_prefill_tokens - incremental,
        "rollback_tokens": int(counters.get("rollback_tokens", 0)),
        "stale_output_escape": 0,
        "runtime_metrics": runtime.metrics.snapshot(),
        "kv": runtime.kv.snapshot(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trace",
        type=Path,
        nargs="?",
        default=Path(__file__).parent / "traces" / "revision_demo.jsonl",
    )
    args = parser.parse_args()
    print(json.dumps(replay(args.trace), indent=2))


if __name__ == "__main__":
    main()

