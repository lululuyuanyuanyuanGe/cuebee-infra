"""Versioned event contracts shared by the gateway and inference runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EventType(str, Enum):
    APPEND_PARTIAL = "append_partial"
    REVISE_PARTIAL = "revise_partial"
    COMMIT_FINAL = "commit_final"
    SPEAKER_RELABEL = "speaker_relabel"
    SESSION_CLOSE = "session_close"


@dataclass(frozen=True, slots=True)
class TokenRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("token range must satisfy 0 <= start <= end")

    def intersects(self, other: "TokenRange") -> bool:
        return self.start < other.end and other.start < self.end

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class STTEvent:
    """A normalized Speech-to-Text event.

    Partial and final text is the complete hypothesis for ``segment_id`` rather
    than an arbitrary character delta. ``revision`` is monotonic within that
    segment; ``client_epoch`` and ``seq_no`` make replay idempotent.
    """

    session_id: str
    segment_id: str
    revision: int
    type: EventType
    text: str = ""
    start_ms: int = 0
    end_ms: int = 0
    confidence: float | None = None
    client_epoch: int = 0
    seq_no: int = 0

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if not self.segment_id.strip():
            raise ValueError("segment_id must not be empty")
        if self.revision < 0 or self.client_epoch < 0 or self.seq_no < 0:
            raise ValueError("revision, client_epoch, and seq_no must be non-negative")
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ValueError("timestamps must satisfy 0 <= start_ms <= end_ms")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.type not in {EventType.SPEAKER_RELABEL, EventType.SESSION_CLOSE}:
            if not self.text:
                raise ValueError("transcript events must contain text")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "STTEvent":
        data = dict(payload)
        data["type"] = EventType(data["type"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "segment_id": self.segment_id,
            "revision": self.revision,
            "type": self.type.value,
            "text": self.text,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "confidence": self.confidence,
            "client_epoch": self.client_epoch,
            "seq_no": self.seq_no,
        }


@dataclass(frozen=True, slots=True)
class AudioChunk:
    session_id: str
    chunk_id: str
    start_ms: int
    end_ms: int
    samples: tuple[float, ...]
    sample_rate: int = 16_000
    deadline_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.session_id or not self.chunk_id:
            raise ValueError("session_id and chunk_id must not be empty")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("audio chunk must have a positive duration")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not self.samples:
            raise ValueError("samples must not be empty")

    @property
    def duration_seconds(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0


@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    session_id: str
    chunk_id: str
    start_ms: int
    end_ms: int
    speaker_id: str
    similarity: float
    embedding_quality: float


@dataclass(frozen=True, slots=True)
class TranscriptDiff:
    session_id: str
    segment_id: str
    old_version: int
    new_version: int
    changed_range: TokenRange
    old_tail_tokens: tuple[int, ...]
    new_tail_tokens: tuple[int, ...]
    common_prefix_tokens: int
    rollback_tokens: int
    appended_tokens: tuple[int, ...]
    committed: bool

