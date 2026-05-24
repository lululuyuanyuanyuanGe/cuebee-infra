"""Explicit shared-prefix branches with version and dependency semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from cuebee.event_schema import TokenRange
from cuebee.kv_metadata import VersionedKVManager


class TaskKind(str, Enum):
    USER_QUERY = "user_query"
    PROACTIVE_HINT = "proactive_hint"
    SEARCH_DECISION = "search_decision"
    MEMORY_EXTRACTION = "memory_extraction"
    CONTEXT_SUMMARY = "context_summary"


class BranchStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    INVALIDATED = "invalidated"


_CURRENT_VERSION_KINDS = {TaskKind.PROACTIVE_HINT, TaskKind.SEARCH_DECISION}


@dataclass(slots=True)
class Branch:
    branch_id: str
    task_id: str
    session_id: str
    kind: TaskKind
    base_version: int
    dependency: TokenRange
    includes_tentative: bool
    requires_current_version: bool
    status: BranchStatus = BranchStatus.QUEUED
    invalidated_by_version: int | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "kind": self.kind.value,
            "base_version": self.base_version,
            "dependency": [self.dependency.start, self.dependency.end],
            "includes_tentative": self.includes_tentative,
            "requires_current_version": self.requires_current_version,
            "status": self.status.value,
            "invalidated_by_version": self.invalidated_by_version,
        }


class BranchGraph:
    def __init__(self, kv_manager: VersionedKVManager) -> None:
        self.kv = kv_manager
        self._branches: dict[str, Branch] = {}

    def create(
        self,
        task_id: str,
        session_id: str,
        kind: TaskKind,
        base_version: int,
        commit_frontier: int,
        total_tokens: int,
        dependency: TokenRange | None = None,
    ) -> Branch:
        branch_id = f"branch:{task_id}"
        if branch_id in self._branches:
            raise ValueError(f"branch already exists for task: {task_id}")
        includes_tentative = kind in _CURRENT_VERSION_KINDS
        dependency_end = total_tokens if includes_tentative else commit_frontier
        branch = Branch(
            branch_id=branch_id,
            task_id=task_id,
            session_id=session_id,
            kind=kind,
            base_version=base_version,
            dependency=dependency or TokenRange(0, dependency_end),
            includes_tentative=includes_tentative,
            requires_current_version=kind in _CURRENT_VERSION_KINDS,
        )
        self.kv.fork_branch(branch_id, session_id, include_tentative=includes_tentative)
        self._branches[branch_id] = branch
        return branch

    def append_private_prompt(self, branch_id: str, token_ids: list[int], version: int) -> None:
        branch = self.get(branch_id)
        if branch.status is not BranchStatus.QUEUED:
            raise RuntimeError("private prompt can only be appended before scheduling")
        self.kv.append_branch(branch_id, token_ids, version)

    def invalidate(
        self,
        session_id: str,
        changed_range: TokenRange,
        new_version: int,
    ) -> tuple[str, ...]:
        invalidated: list[str] = []
        for branch in self._branches.values():
            if branch.session_id != session_id or branch.status not in {
                BranchStatus.QUEUED,
                BranchStatus.RUNNING,
            }:
                continue
            exact_version_stale = (
                branch.requires_current_version and branch.base_version != new_version
            )
            dependency_changed = changed_range.length > 0 and branch.dependency.intersects(
                changed_range
            )
            if exact_version_stale or dependency_changed:
                branch.status = BranchStatus.INVALIDATED
                branch.invalidated_by_version = new_version
                self.kv.release_branch(branch.branch_id)
                invalidated.append(branch.task_id)
        return tuple(invalidated)

    def mark_running(self, branch_id: str) -> None:
        branch = self.get(branch_id)
        if branch.status is not BranchStatus.QUEUED:
            raise RuntimeError(f"branch {branch_id} is not queued")
        branch.status = BranchStatus.RUNNING

    def complete(self, branch_id: str) -> None:
        branch = self.get(branch_id)
        if branch.status is not BranchStatus.RUNNING:
            raise RuntimeError(f"branch {branch_id} is not running")
        branch.status = BranchStatus.COMPLETED
        self.kv.release_branch(branch_id)

    def cancel(self, branch_id: str) -> None:
        branch = self.get(branch_id)
        if branch.status in {
            BranchStatus.COMPLETED,
            BranchStatus.CANCELLED,
            BranchStatus.INVALIDATED,
        }:
            return
        branch.status = BranchStatus.CANCELLED
        self.kv.release_branch(branch_id)

    def get(self, branch_id: str) -> Branch:
        try:
            return self._branches[branch_id]
        except KeyError as exc:
            raise KeyError(f"unknown branch: {branch_id}") from exc

    def for_task(self, task_id: str) -> Branch:
        return self.get(f"branch:{task_id}")

    def snapshot(self) -> list[dict[str, Any]]:
        return [self._branches[key].snapshot() for key in sorted(self._branches)]

