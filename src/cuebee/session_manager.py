"""Versioned conversation state with committed spine and revisable tail."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cuebee.event_schema import EventType, STTEvent, TokenRange, TranscriptDiff
from cuebee.tokenizer import Tokenizer, token_lcp


@dataclass(slots=True)
class ConversationSession:
    session_id: str
    version: int = 0
    committed_tokens: list[int] = field(default_factory=list)
    tentative_tokens: list[int] = field(default_factory=list)
    committed_text: list[str] = field(default_factory=list)
    tentative_text: str = ""
    active_segment_id: str | None = None
    speaker_names: dict[str, str] = field(default_factory=dict)
    closed: bool = False

    @property
    def commit_frontier(self) -> int:
        return len(self.committed_tokens)

    @property
    def total_tokens(self) -> int:
        return self.commit_frontier + len(self.tentative_tokens)

    @property
    def all_tokens(self) -> tuple[int, ...]:
        return tuple(self.committed_tokens + self.tentative_tokens)

    def transcript_text(self, include_tentative: bool = True) -> str:
        parts = [part for part in self.committed_text if part]
        if include_tentative and self.tentative_text:
            parts.append(self.tentative_text)
        return " ".join(parts)

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "version": self.version,
            "commit_frontier": self.commit_frontier,
            "committed_tokens": len(self.committed_tokens),
            "tentative_tokens": len(self.tentative_tokens),
            "active_segment_id": self.active_segment_id,
            "closed": self.closed,
            "transcript": self.transcript_text(),
            "speaker_names": dict(self.speaker_names),
        }


class SessionManager:
    def __init__(self, tokenizer: Tokenizer, retokenization_guard_tokens: int = 1) -> None:
        if retokenization_guard_tokens < 0:
            raise ValueError("retokenization guard must be non-negative")
        self.tokenizer = tokenizer
        self.retokenization_guard_tokens = retokenization_guard_tokens
        self._sessions: dict[str, ConversationSession] = {}

    def get_or_create(self, session_id: str) -> ConversationSession:
        if not session_id:
            raise ValueError("session_id must not be empty")
        return self._sessions.setdefault(session_id, ConversationSession(session_id))

    def get(self, session_id: str) -> ConversationSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown session: {session_id}") from exc

    def apply(self, event: STTEvent) -> TranscriptDiff:
        session = self.get_or_create(event.session_id)
        if session.closed:
            raise RuntimeError(f"session {event.session_id} is closed")

        old_version = session.version
        session.version += 1

        if event.type is EventType.SPEAKER_RELABEL:
            speaker_id, separator, display_name = event.text.partition("=")
            if not separator or not speaker_id or not display_name:
                raise ValueError("speaker relabel text must be '<speaker_id>=<display_name>'")
            session.speaker_names[speaker_id] = display_name
            return self._metadata_diff(session, event, old_version)

        if event.type is EventType.SESSION_CLOSE:
            session.closed = True
            return self._metadata_diff(session, event, old_version)

        if session.active_segment_id not in {None, event.segment_id}:
            raise ValueError(
                f"segment {session.active_segment_id} is still tentative; "
                f"cannot start {event.segment_id}"
            )

        old_tail = tuple(session.tentative_tokens)
        segment_prefix = " " if session.committed_text else ""
        new_tail = tuple(self.tokenizer.encode(segment_prefix + event.text))
        is_append = len(new_tail) >= len(old_tail) and new_tail[: len(old_tail)] == old_tail
        guard = 0 if is_append else self.retokenization_guard_tokens
        common_prefix = token_lcp(old_tail, new_tail, guard_tokens=guard)
        changed_start = session.commit_frontier + common_prefix
        changed_end = session.commit_frontier + max(len(old_tail), len(new_tail))
        if changed_end == changed_start:
            changed_end += len(new_tail) - common_prefix

        session.active_segment_id = event.segment_id
        session.tentative_tokens = list(new_tail)
        session.tentative_text = event.text
        committed = event.type is EventType.COMMIT_FINAL

        diff = TranscriptDiff(
            session_id=session.session_id,
            segment_id=event.segment_id,
            old_version=old_version,
            new_version=session.version,
            changed_range=TokenRange(changed_start, max(changed_start, changed_end)),
            old_tail_tokens=old_tail,
            new_tail_tokens=new_tail,
            common_prefix_tokens=common_prefix,
            rollback_tokens=len(old_tail) - common_prefix,
            appended_tokens=new_tail[common_prefix:],
            committed=committed,
        )

        if committed:
            session.committed_tokens.extend(new_tail)
            session.committed_text.append(event.text)
            session.tentative_tokens.clear()
            session.tentative_text = ""
            session.active_segment_id = None
        return diff

    def _metadata_diff(
        self,
        session: ConversationSession,
        event: STTEvent,
        old_version: int,
    ) -> TranscriptDiff:
        frontier = session.total_tokens
        return TranscriptDiff(
            session_id=session.session_id,
            segment_id=event.segment_id,
            old_version=old_version,
            new_version=session.version,
            changed_range=TokenRange(frontier, frontier),
            old_tail_tokens=tuple(session.tentative_tokens),
            new_tail_tokens=tuple(session.tentative_tokens),
            common_prefix_tokens=len(session.tentative_tokens),
            rollback_tokens=0,
            appended_tokens=(),
            committed=False,
        )

    def snapshots(self) -> list[dict[str, Any]]:
        return [self._sessions[key].snapshot() for key in sorted(self._sessions)]
