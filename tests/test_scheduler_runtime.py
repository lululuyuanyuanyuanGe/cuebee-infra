from __future__ import annotations

import unittest

from cuebee.branch_graph import BranchGraph, BranchStatus, TaskKind
from cuebee.event_schema import EventType, STTEvent, TokenRange
from cuebee.freshness_gate import FreshnessGate, GateDecision
from cuebee.kv_metadata import Validity, VersionedKVManager
from cuebee.runtime import CueBeeRuntime, TentativePrefillPolicy
from cuebee.scheduler import AnalysisTask, TaskStatus, VersionDeadlineScheduler
from cuebee.session_manager import ConversationSession


def make_task(
    task_id: str,
    kind: TaskKind,
    deadline: float,
    foreground: bool,
    priority: int = 0,
) -> AnalysisTask:
    return AnalysisTask(
        task_id=task_id,
        branch_id=f"branch:{task_id}",
        session_id="s",
        kind=kind,
        base_version=1,
        dependency=TokenRange(0, 10),
        deadline_at=deadline,
        priority=priority,
        estimated_tokens=10,
        shared_prefix_tokens=100,
        foreground=foreground,
        requires_current_version=kind is TaskKind.PROACTIVE_HINT,
        prompt="test",
        created_at=0.0,
    )


class SchedulerTests(unittest.TestCase):
    def test_foreground_deadline_task_runs_first(self) -> None:
        scheduler = VersionDeadlineScheduler()
        scheduler.submit(make_task("background", TaskKind.CONTEXT_SUMMARY, 10.0, False))
        scheduler.submit(make_task("hint", TaskKind.PROACTIVE_HINT, 2.0, True))
        selected = scheduler.pop_next({"s": 1}, now=1.0)
        self.assertEqual(selected.task_id, "hint")  # type: ignore[union-attr]

    def test_stale_task_is_cancelled_before_scoring(self) -> None:
        scheduler = VersionDeadlineScheduler()
        scheduler.submit(make_task("hint", TaskKind.PROACTIVE_HINT, 2.0, True))
        self.assertIsNone(scheduler.pop_next({"s": 2}, now=1.0))
        self.assertEqual(scheduler.get("hint").status, TaskStatus.CANCELLED)

    def test_overload_drops_background_first(self) -> None:
        scheduler = VersionDeadlineScheduler()
        scheduler.submit(make_task("query", TaskKind.USER_QUERY, 10.0, True))
        scheduler.submit(make_task("summary", TaskKind.CONTEXT_SUMMARY, 10.0, False))
        scheduler.submit(make_task("memory", TaskKind.MEMORY_EXTRACTION, 10.0, False))
        self.assertEqual(scheduler.shed_load(1), ("summary", "memory"))


class FreshnessTests(unittest.TestCase):
    def test_invalidated_foreground_output_restarts(self) -> None:
        kv = VersionedKVManager()
        kv.append_session("s", [1, 2], Validity.TENTATIVE, 1)
        graph = BranchGraph(kv)
        branch = graph.create("hint", "s", TaskKind.PROACTIVE_HINT, 1, 0, 2)
        graph.invalidate("s", TokenRange(1, 3), new_version=2)
        task = make_task("hint", TaskKind.PROACTIVE_HINT, 2.0, True)
        session = ConversationSession("s", version=2)
        self.assertEqual(FreshnessGate().evaluate(task, branch, session), GateDecision.RESTART)
        self.assertEqual(branch.status, BranchStatus.INVALIDATED)


class RuntimeIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = CueBeeRuntime(
            block_size=4,
            tentative_policy=TentativePrefillPolicy(min_tail_tokens=1, min_confidence=0.0),
        )

    def test_revision_rolls_back_and_cancels_tail_task(self) -> None:
        first = STTEvent("s", "seg", 1, EventType.APPEND_PARTIAL, "budget is fifty")
        self.runtime.handle_event(first)
        self.runtime.submit_task(
            "s", TaskKind.PROACTIVE_HINT, "suggest a response", 500, task_id="old-hint", now=1.0
        )
        revised = STTEvent(
            "s", "seg", 2, EventType.REVISE_PARTIAL, "budget is one hundred fifty"
        )
        outcome = self.runtime.handle_event(revised)

        self.assertIn("old-hint", outcome.invalidated_tasks)
        self.assertGreater(outcome.rollback_tokens, 0)
        self.assertEqual(self.runtime.scheduler.get("old-hint").status, TaskStatus.CANCELLED)
        self.assertIsNone(self.runtime.run_next(now=1.1))

    def test_commit_and_fresh_output(self) -> None:
        self.runtime.handle_event(
            STTEvent("s", "seg", 1, EventType.COMMIT_FINAL, "the meeting starts at nine")
        )
        self.runtime.submit_task(
            "s", TaskKind.USER_QUERY, "when is the meeting", 500, task_id="query", now=1.0
        )
        outcome = self.runtime.run_next(now=1.1)
        self.assertEqual(outcome.decision, GateDecision.ALLOW)  # type: ignore[union-attr]
        self.assertIn("cuebee:", outcome.result.text)  # type: ignore[union-attr]
        self.assertEqual(self.runtime.metrics.snapshot()["counters"]["outputs_allowed"], 1.0)

    def test_duplicate_callback_is_idempotent(self) -> None:
        event = STTEvent("s", "seg", 1, EventType.APPEND_PARTIAL, "hello world")
        self.assertTrue(self.runtime.handle_event(event).accepted)
        duplicate = self.runtime.handle_event(event)
        self.assertTrue(duplicate.duplicate_or_stale)


if __name__ == "__main__":
    unittest.main()
