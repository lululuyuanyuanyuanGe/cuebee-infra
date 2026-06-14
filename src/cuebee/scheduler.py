"""Semantic admission layer placed ahead of the model token scheduler."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from cuebee.branch_graph import TaskKind
from cuebee.event_schema import TokenRange


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DROPPED = "dropped"


@dataclass(slots=True)
class AnalysisTask:
    task_id: str
    branch_id: str
    session_id: str
    kind: TaskKind
    base_version: int
    dependency: TokenRange
    deadline_at: float
    priority: int
    estimated_tokens: int
    shared_prefix_tokens: int
    foreground: bool
    requires_current_version: bool
    prompt: str
    created_at: float
    status: TaskStatus = TaskStatus.QUEUED
    cancel_reason: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "kind": self.kind.value,
            "base_version": self.base_version,
            "deadline_at": self.deadline_at,
            "priority": self.priority,
            "estimated_tokens": self.estimated_tokens,
            "shared_prefix_tokens": self.shared_prefix_tokens,
            "foreground": self.foreground,
            "requires_current_version": self.requires_current_version,
            "status": self.status.value,
            "cancel_reason": self.cancel_reason,
        }


@dataclass(frozen=True, slots=True)
class SchedulerWeights:
    deadline: float = 8.0
    foreground: float = 12.0
    age: float = 0.25
    reuse: float = 0.01
    cost: float = 0.02
    priority: float = 2.0


class VersionDeadlineScheduler:
    def __init__(self, admission_cap: int = 32, weights: SchedulerWeights | None = None) -> None:
        if admission_cap <= 0:
            raise ValueError("admission_cap must be positive")
        self.admission_cap = admission_cap
        self.weights = weights or SchedulerWeights()
        self._tasks: dict[str, AnalysisTask] = {}

    def submit(self, task: AnalysisTask) -> None:
        if task.task_id in self._tasks:
            raise ValueError(f"task already exists: {task.task_id}")
        if self.active_count >= self.admission_cap:
            raise RuntimeError("analysis admission cap reached")
        self._tasks[task.task_id] = task

    @property
    def active_count(self) -> int:
        return sum(
            task.status in {TaskStatus.QUEUED, TaskStatus.RUNNING}
            for task in self._tasks.values()
        )

    def cancel(self, task_id: str, reason: str) -> bool:
        task = self.get(task_id)
        if task.status not in {TaskStatus.QUEUED, TaskStatus.RUNNING}:
            return False
        task.status = TaskStatus.CANCELLED
        task.cancel_reason = reason
        return True

    def pop_next(
        self,
        current_versions: dict[str, int],
        now: float | None = None,
    ) -> AnalysisTask | None:
        timestamp = time.monotonic() if now is None else now
        candidates: list[AnalysisTask] = []
        for task in self._tasks.values():
            if task.status is not TaskStatus.QUEUED:
                continue
            if task.requires_current_version and current_versions.get(task.session_id) != task.base_version:
                task.status = TaskStatus.CANCELLED
                task.cancel_reason = "stale_base_version"
                continue
            candidates.append(task)
        if not candidates:
            return None
        selected = max(candidates, key=lambda task: self.score(task, timestamp))
        selected.status = TaskStatus.RUNNING
        return selected

    def score(self, task: AnalysisTask, now: float) -> float:
        slack = max(0.001, task.deadline_at - now)
        urgency = 1.0 / slack
        age = max(0.0, now - task.created_at)
        return (
            self.weights.deadline * urgency
            + self.weights.foreground * float(task.foreground)
            + self.weights.age * age
            + self.weights.reuse * task.shared_prefix_tokens
            - self.weights.cost * task.estimated_tokens
            + self.weights.priority * task.priority
        )

    def complete(self, task_id: str) -> None:
        task = self.get(task_id)
        if task.status is not TaskStatus.RUNNING:
            raise RuntimeError(f"task {task_id} is not running")
        task.status = TaskStatus.COMPLETED

    def drop(self, task_id: str, reason: str) -> None:
        task = self.get(task_id)
        if task.status not in {TaskStatus.RUNNING, TaskStatus.QUEUED}:
            return
        task.status = TaskStatus.DROPPED
        task.cancel_reason = reason

    def shed_load(self, target_active: int) -> tuple[str, ...]:
        if target_active < 0:
            raise ValueError("target_active must be non-negative")
        drop_order = {
            TaskKind.CONTEXT_SUMMARY: 0,
            TaskKind.MEMORY_EXTRACTION: 1,
            TaskKind.SEARCH_DECISION: 2,
            TaskKind.PROACTIVE_HINT: 3,
            TaskKind.USER_QUERY: 4,
        }
        queued = sorted(
            (task for task in self._tasks.values() if task.status is TaskStatus.QUEUED),
            key=lambda task: (drop_order[task.kind], task.priority, task.created_at),
        )
        dropped: list[str] = []
        while self.active_count > target_active and queued:
            task = queued.pop(0)
            task.status = TaskStatus.DROPPED
            task.cancel_reason = "overload_shed"
            dropped.append(task.task_id)
        return tuple(dropped)

    def get(self, task_id: str) -> AnalysisTask:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown task: {task_id}") from exc

    def snapshot(self) -> list[dict[str, Any]]:
        return [self._tasks[key].snapshot() for key in sorted(self._tasks)]

