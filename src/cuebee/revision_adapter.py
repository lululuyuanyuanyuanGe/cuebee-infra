"""Normalize vendor callbacks and reject stale or duplicate transcript events."""

from __future__ import annotations

from dataclasses import dataclass

from cuebee.event_schema import EventType, STTEvent


@dataclass(slots=True)
class _SegmentCursor:
    revision: int = -1
    finalized: bool = False


class RevisionAdapter:
    def __init__(self) -> None:
        self._segments: dict[tuple[str, str], _SegmentCursor] = {}
        self.duplicates_dropped = 0
        self.stale_dropped = 0

    def normalize(self, event: STTEvent) -> STTEvent | None:
        key = (event.session_id, event.segment_id)
        cursor = self._segments.setdefault(key, _SegmentCursor())

        if cursor.finalized:
            self.stale_dropped += 1
            return None
        if event.revision == cursor.revision:
            self.duplicates_dropped += 1
            return None
        if event.revision < cursor.revision:
            self.stale_dropped += 1
            return None

        normalized_type = event.type
        if event.type in {EventType.APPEND_PARTIAL, EventType.REVISE_PARTIAL}:
            normalized_type = (
                EventType.APPEND_PARTIAL
                if cursor.revision < 0
                else EventType.REVISE_PARTIAL
            )

        cursor.revision = event.revision
        cursor.finalized = event.type is EventType.COMMIT_FINAL
        if normalized_type is event.type:
            return event
        return STTEvent(
            session_id=event.session_id,
            segment_id=event.segment_id,
            revision=event.revision,
            type=normalized_type,
            text=event.text,
            start_ms=event.start_ms,
            end_ms=event.end_ms,
            confidence=event.confidence,
            client_epoch=event.client_epoch,
            seq_no=event.seq_no,
        )

