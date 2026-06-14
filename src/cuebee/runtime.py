"""End-to-end orchestration of revisions, cache state, branches, and output."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from cuebee.branch_graph import BranchGraph, TaskKind
from cuebee.engine import DeterministicEngine, EngineRequest, EngineResult, InferenceEngine
from cuebee.event_schema import EventType, STTEvent, TranscriptDiff
from cuebee.freshness_gate import FreshnessGate, GateDecision
from cuebee.kv_metadata import Validity, VersionedKVManager
from cuebee.metrics import MetricRegistry
from cuebee.revision_adapter import RevisionAdapter
from cuebee.scheduler import AnalysisTask, VersionDeadlineScheduler
from cuebee.session_manager import SessionManager
from cuebee.tokenizer import Tokenizer, UTF8Tokenizer


@dataclass(frozen=True, slots=True)
class EventOutcome:
    accepted: bool
    duplicate_or_stale: bool
    diff: TranscriptDiff | None
    invalidated_tasks: tuple[str, ...]
    prefilled_tokens: int
    rollback_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "duplicate_or_stale": self.duplicate_or_stale,
            "version": self.diff.new_version if self.diff else None,
            "invalidated_tasks": list(self.invalidated_tasks),
            "prefilled_tokens": self.prefilled_tokens,
            "rollback_tokens": self.rollback_tokens,
            "changed_range": (
                [self.diff.changed_range.start, self.diff.changed_range.end]
                if self.diff
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    task_id: str
    decision: GateDecision
    result: EngineResult | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "decision": self.decision.value,
            "text": self.result.text if self.result else None,
            "input_tokens": self.result.input_tokens if self.result else None,
            "output_tokens": self.result.output_tokens if self.result else None,
        }


class TentativePrefillPolicy:
    def __init__(
        self,
        min_tail_tokens: int = 8,
        min_confidence: float = 0.50,
        max_gpu_load: float = 0.85,
    ) -> None:
        self.min_tail_tokens = min_tail_tokens
        self.min_confidence = min_confidence
        self.max_gpu_load = max_gpu_load

    def admit(self, event: STTEvent, tail_tokens: int, gpu_load: float) -> bool:
        confidence = 1.0 if event.confidence is None else event.confidence
        return (
            tail_tokens >= self.min_tail_tokens
            and confidence >= self.min_confidence
            and gpu_load <= self.max_gpu_load
        )


class CueBeeRuntime:
    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        engine: InferenceEngine | None = None,
        block_size: int = 16,
        admission_cap: int = 32,
        tentative_policy: TentativePrefillPolicy | None = None,
    ) -> None:
        self.tokenizer = tokenizer or UTF8Tokenizer()
        self.engine = engine or DeterministicEngine()
        self.revisions = RevisionAdapter()
        self.sessions = SessionManager(self.tokenizer)
        self.kv = VersionedKVManager(block_size)
        self.branches = BranchGraph(self.kv)
        self.scheduler = VersionDeadlineScheduler(admission_cap)
        self.freshness = FreshnessGate()
        self.metrics = MetricRegistry()
        self.tentative_policy = tentative_policy or TentativePrefillPolicy()
        self.gpu_load = 0.0

    def handle_event(self, event: STTEvent) -> EventOutcome:
        normalized = self.revisions.normalize(event)
        if normalized is None:
            self.metrics.inc("stt_events_dropped")
            return EventOutcome(False, True, None, (), 0, 0)

        diff = self.sessions.apply(normalized)
        self.metrics.inc("stt_events_accepted")
        invalidated = self.branches.invalidate(
            normalized.session_id,
            diff.changed_range,
            diff.new_version,
        )
        for task_id in invalidated:
            self.scheduler.cancel(task_id, "transcript_revision")
            self.engine.abort(task_id)
            self.metrics.inc("tasks_invalidated")

        prefilled = 0
        rolled_back = 0
        if normalized.type not in {EventType.SPEAKER_RELABEL, EventType.SESSION_CLOSE}:
            current_tokens = self.kv.session_tokens(normalized.session_id)
            rollback_from = diff.changed_range.start
            if len(current_tokens) > rollback_from:
                rollback = self.kv.rollback_tentative(normalized.session_id, rollback_from)
                rolled_back = rollback.removed_tokens
                self.metrics.inc("rollback_tokens", rolled_back)

            state = self.sessions.get(normalized.session_id)
            expected = state.all_tokens
            resident = self.kv.session_tokens(normalized.session_id)
            if resident != expected[: len(resident)]:
                raise AssertionError("resident cache is not a prefix of the transcript")

            should_prefill = normalized.type is EventType.COMMIT_FINAL or self.tentative_policy.admit(
                normalized,
                len(diff.new_tail_tokens),
                self.gpu_load,
            )
            if should_prefill:
                missing = expected[len(resident) :]
                if missing:
                    self.engine.prefill(
                        normalized.session_id,
                        missing,
                        diff.new_version,
                        tentative=normalized.type is not EventType.COMMIT_FINAL,
                    )
                    self.kv.append_session(
                        normalized.session_id,
                        missing,
                        Validity.TENTATIVE,
                        diff.new_version,
                    )
                    prefilled = len(missing)
                    self.metrics.inc("prefill_tokens", prefilled)
            if normalized.type is EventType.COMMIT_FINAL:
                self.kv.commit_tentative(
                    normalized.session_id,
                    state.commit_frontier,
                    diff.new_version,
                )

        self.kv.validate_ref_counts()
        return EventOutcome(True, False, diff, invalidated, prefilled, rolled_back)

    def submit_task(
        self,
        session_id: str,
        kind: TaskKind,
        prompt: str,
        deadline_ms: int,
        priority: int = 0,
        estimated_tokens: int = 32,
        task_id: str | None = None,
        now: float | None = None,
    ) -> AnalysisTask:
        state = self.sessions.get(session_id)
        timestamp = time.monotonic() if now is None else now
        resolved_task_id = task_id or uuid.uuid4().hex
        branch = self.branches.create(
            task_id=resolved_task_id,
            session_id=session_id,
            kind=kind,
            base_version=state.version,
            commit_frontier=state.commit_frontier,
            total_tokens=state.total_tokens,
        )
        private_prompt_tokens = self.tokenizer.encode(f"\n[task:{kind.value}] {prompt}")
        self.branches.append_private_prompt(branch.branch_id, private_prompt_tokens, state.version)
        task = AnalysisTask(
            task_id=resolved_task_id,
            branch_id=branch.branch_id,
            session_id=session_id,
            kind=kind,
            base_version=state.version,
            dependency=branch.dependency,
            deadline_at=timestamp + deadline_ms / 1000.0,
            priority=priority,
            estimated_tokens=estimated_tokens,
            shared_prefix_tokens=branch.dependency.length,
            foreground=kind in {TaskKind.USER_QUERY, TaskKind.PROACTIVE_HINT},
            requires_current_version=branch.requires_current_version,
            prompt=prompt,
            created_at=timestamp,
        )
        try:
            self.scheduler.submit(task)
        except Exception:
            self.branches.cancel(branch.branch_id)
            raise
        self.metrics.inc("tasks_submitted")
        return task

    def run_next(self, now: float | None = None) -> TaskOutcome | None:
        versions = {snapshot["session_id"]: snapshot["version"] for snapshot in self.sessions.snapshots()}
        task = self.scheduler.pop_next(versions, now=now)
        if task is None:
            return None
        branch = self.branches.for_task(task.task_id)
        self.branches.mark_running(branch.branch_id)
        state = self.sessions.get(task.session_id)
        transcript = state.transcript_text(include_tentative=branch.includes_tentative)
        full_prompt = f"{transcript}\n[task:{task.kind.value}] {task.prompt}"
        prompt_tokens = tuple(self.tokenizer.encode(full_prompt))
        started = time.monotonic()
        result = self.engine.generate(
            EngineRequest(
                task_id=task.task_id,
                session_id=task.session_id,
                base_version=task.base_version,
                prompt=full_prompt,
                prompt_token_ids=prompt_tokens,
                max_output_tokens=task.estimated_tokens,
            )
        )
        self.metrics.observe("task_latency_seconds", time.monotonic() - started)
        decision = self.freshness.evaluate(task, branch, self.sessions.get(task.session_id))
        if decision is GateDecision.ALLOW:
            self.branches.complete(branch.branch_id)
            self.scheduler.complete(task.task_id)
            self.metrics.inc("outputs_allowed")
            return TaskOutcome(task.task_id, decision, result)

        self.branches.cancel(branch.branch_id)
        self.scheduler.drop(task.task_id, f"freshness_{decision.value}")
        self.metrics.inc("stale_outputs_prevented")
        return TaskOutcome(task.task_id, decision, None)

    def shed_load(self, target_active: int) -> tuple[str, ...]:
        dropped = self.scheduler.shed_load(target_active)
        for task_id in dropped:
            self.branches.cancel(self.branches.for_task(task_id).branch_id)
            self.metrics.inc("tasks_overload_dropped")
        return dropped

    def snapshot(self, session_id: str) -> dict[str, Any]:
        return {
            "session": self.sessions.get(session_id).snapshot(),
            "kv": self.kv.snapshot(),
            "branches": [
                branch for branch in self.branches.snapshot() if branch["session_id"] == session_id
            ],
            "tasks": [
                task for task in self.scheduler.snapshot() if task["session_id"] == session_id
            ],
        }

