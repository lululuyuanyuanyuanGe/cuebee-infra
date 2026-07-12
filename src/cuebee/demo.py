"""Reproducible revision, cancellation, commit, and freshness demonstration."""

from __future__ import annotations

import json

from cuebee.branch_graph import TaskKind
from cuebee.event_schema import EventType, STTEvent
from cuebee.runtime import CueBeeRuntime, TentativePrefillPolicy


def run_demo() -> list[dict[str, object]]:
    runtime = CueBeeRuntime(
        block_size=8,
        tentative_policy=TentativePrefillPolicy(min_tail_tokens=1, min_confidence=0.0),
    )
    timeline: list[dict[str, object]] = []

    partial = runtime.handle_event(
        STTEvent("demo", "budget", 41, EventType.APPEND_PARTIAL, "budget is fifty")
    )
    timeline.append({"step": "partial_v41", **partial.to_dict()})

    runtime.submit_task(
        "demo",
        TaskKind.SEARCH_DECISION,
        "should we look up the budget policy?",
        deadline_ms=300,
        task_id="search-v41",
    )
    timeline.append({"step": "branch_v41", "task_id": "search-v41"})

    revision = runtime.handle_event(
        STTEvent(
            "demo",
            "budget",
            42,
            EventType.REVISE_PARTIAL,
            "budget is one hundred fifty thousand",
        )
    )
    timeline.append({"step": "revision_v42", **revision.to_dict()})

    final = runtime.handle_event(
        STTEvent(
            "demo",
            "budget",
            43,
            EventType.COMMIT_FINAL,
            "budget is one hundred fifty thousand",
        )
    )
    timeline.append({"step": "final_v43", **final.to_dict()})

    runtime.submit_task(
        "demo",
        TaskKind.PROACTIVE_HINT,
        "give one short and actionable hint",
        deadline_ms=300,
        task_id="hint-v43",
    )
    output = runtime.run_next()
    timeline.append({"step": "fresh_output", **output.to_dict()})  # type: ignore[union-attr]
    timeline.append({"step": "metrics", "value": runtime.metrics.snapshot()})
    return timeline


def main() -> None:
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

