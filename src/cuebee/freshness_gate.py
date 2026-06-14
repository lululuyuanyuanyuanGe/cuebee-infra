"""Final version check for user-visible output and memory writes."""

from __future__ import annotations

from enum import Enum

from cuebee.branch_graph import Branch, BranchStatus
from cuebee.scheduler import AnalysisTask
from cuebee.session_manager import ConversationSession


class GateDecision(str, Enum):
    ALLOW = "allow"
    DROP = "drop"
    RESTART = "restart"


class FreshnessGate:
    def evaluate(
        self,
        task: AnalysisTask,
        branch: Branch,
        session: ConversationSession,
    ) -> GateDecision:
        if branch.status in {BranchStatus.CANCELLED, BranchStatus.INVALIDATED}:
            return GateDecision.RESTART if task.foreground else GateDecision.DROP
        if task.requires_current_version and task.base_version != session.version:
            return GateDecision.RESTART if task.foreground else GateDecision.DROP
        if not branch.includes_tentative and branch.dependency.end > session.commit_frontier:
            return GateDecision.DROP
        return GateDecision.ALLOW

