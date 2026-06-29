"""Deadline-aware, length-bucketed micro-batching across sessions."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from cuebee.event_schema import AudioChunk


@dataclass(frozen=True, slots=True)
class QueuedChunk:
    chunk: AudioChunk
    enqueued_ms: int
    deadline_ms: int


class SpeakerMicroBatcher:
    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_ms: int = 20,
        dispatch_slack_ms: int = 5,
        per_session_pending: int = 8,
    ) -> None:
        if min(max_batch_size, max_wait_ms, per_session_pending) <= 0:
            raise ValueError("batch limits must be positive")
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.dispatch_slack_ms = dispatch_slack_ms
        self.per_session_pending = per_session_pending
        self._buckets: dict[str, deque[QueuedChunk]] = defaultdict(deque)
        self._pending_by_session: dict[str, int] = defaultdict(int)
        self.backpressure_drops = 0

    def enqueue(self, chunk: AudioChunk, now_ms: int) -> bool:
        if self._pending_by_session[chunk.session_id] >= self.per_session_pending:
            self.backpressure_drops += 1
            return False
        deadline = chunk.deadline_ms if chunk.deadline_ms is not None else now_ms + 100
        self._buckets[self._bucket(chunk)].append(QueuedChunk(chunk, now_ms, deadline))
        self._pending_by_session[chunk.session_id] += 1
        return True

    def pop_ready(self, now_ms: int, force: bool = False) -> list[list[QueuedChunk]]:
        batches: list[list[QueuedChunk]] = []
        for bucket_name in sorted(self._buckets):
            queue = self._buckets[bucket_name]
            if not queue:
                continue
            earliest_deadline = min(item.deadline_ms for item in queue)
            oldest_wait = now_ms - min(item.enqueued_ms for item in queue)
            ready = (
                force
                or len(queue) >= self.max_batch_size
                or oldest_wait >= self.max_wait_ms
                or earliest_deadline <= now_ms + self.dispatch_slack_ms
            )
            while ready and queue:
                candidates = sorted(queue, key=lambda item: (item.deadline_ms, item.enqueued_ms))
                batch = self._fair_select(candidates, self.max_batch_size)
                selected_ids = {id(item) for item in batch}
                self._buckets[bucket_name] = deque(
                    item for item in queue if id(item) not in selected_ids
                )
                queue = self._buckets[bucket_name]
                for item in batch:
                    self._pending_by_session[item.chunk.session_id] -= 1
                batches.append(batch)
                ready = force or len(queue) >= self.max_batch_size
        return batches

    @property
    def queued_chunks(self) -> int:
        return sum(len(queue) for queue in self._buckets.values())

    @property
    def backlog_seconds(self) -> float:
        return sum(
            item.chunk.duration_seconds
            for queue in self._buckets.values()
            for item in queue
        )

    @staticmethod
    def _bucket(chunk: AudioChunk) -> str:
        if chunk.duration_seconds <= 1.0:
            return "short"
        if chunk.duration_seconds <= 3.0:
            return "medium"
        return "long"

    @staticmethod
    def _fair_select(items: list[QueuedChunk], limit: int) -> list[QueuedChunk]:
        by_session: dict[str, deque[QueuedChunk]] = defaultdict(deque)
        for item in items:
            by_session[item.chunk.session_id].append(item)
        selected: list[QueuedChunk] = []
        sessions = deque(sorted(by_session))
        while sessions and len(selected) < limit:
            session_id = sessions.popleft()
            selected.append(by_session[session_id].popleft())
            if by_session[session_id]:
                sessions.append(session_id)
        return selected

