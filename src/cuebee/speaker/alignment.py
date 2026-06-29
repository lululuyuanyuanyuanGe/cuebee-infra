"""Timestamp overlap alignment between transcripts and speaker segments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from cuebee.event_schema import SpeakerSegment


@dataclass(frozen=True, slots=True)
class AttributedTranscript:
    segment_id: str
    text: str
    start_ms: int
    end_ms: int
    speaker_id: str | None
    overlap_ms: int


def align_speaker(
    segment_id: str,
    text: str,
    start_ms: int,
    end_ms: int,
    speakers: Sequence[SpeakerSegment],
) -> AttributedTranscript:
    best: SpeakerSegment | None = None
    best_overlap = 0
    for speaker in speakers:
        overlap = max(0, min(end_ms, speaker.end_ms) - max(start_ms, speaker.start_ms))
        if overlap > best_overlap or (
            overlap == best_overlap and best is not None and speaker.similarity > best.similarity
        ):
            best, best_overlap = speaker, overlap
    return AttributedTranscript(
        segment_id,
        text,
        start_ms,
        end_ms,
        best.speaker_id if best else None,
        best_overlap,
    )

