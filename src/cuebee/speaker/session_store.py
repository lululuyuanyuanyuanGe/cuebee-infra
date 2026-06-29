"""Session-scoped stable speaker identities separated from stateless workers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from cuebee.speaker.worker import cosine_similarity, normalize


@dataclass(slots=True)
class SpeakerProfile:
    speaker_id: str
    centroid: tuple[float, ...]
    effective_samples: float
    display_name: str | None = None


@dataclass(slots=True)
class _SpeakerSession:
    profiles: dict[str, SpeakerProfile] = field(default_factory=dict)
    next_speaker_number: int = 1


class SpeakerSessionStore:
    def __init__(self, assignment_threshold: float = 0.80) -> None:
        if not -1.0 <= assignment_threshold <= 1.0:
            raise ValueError("assignment threshold must be a cosine similarity")
        self.assignment_threshold = assignment_threshold
        self._sessions: dict[str, _SpeakerSession] = {}

    def assign(
        self,
        session_id: str,
        embedding: Sequence[float],
        quality: float,
    ) -> tuple[str, float]:
        if not 0.0 <= quality <= 1.0:
            raise ValueError("quality must be between zero and one")
        vector = normalize(embedding)
        session = self._sessions.setdefault(session_id, _SpeakerSession())
        best: SpeakerProfile | None = None
        best_score = -1.0
        for profile in session.profiles.values():
            score = cosine_similarity(profile.centroid, vector)
            if score > best_score:
                best, best_score = profile, score

        if best is None or best_score < self.assignment_threshold:
            speaker_id = f"spk_{session.next_speaker_number:03d}"
            session.next_speaker_number += 1
            session.profiles[speaker_id] = SpeakerProfile(
                speaker_id,
                vector,
                max(0.1, quality),
            )
            return speaker_id, best_score

        weight = max(0.05, quality)
        total = best.effective_samples + weight
        updated = [
            (old * best.effective_samples + new * weight) / total
            for old, new in zip(best.centroid, vector, strict=True)
        ]
        best.centroid = normalize(updated)
        best.effective_samples = total
        return best.speaker_id, best_score

    def rename(self, session_id: str, speaker_id: str, display_name: str) -> None:
        if not display_name:
            raise ValueError("display name must not be empty")
        self._sessions[session_id].profiles[speaker_id].display_name = display_name

    def export_session(self, session_id: str) -> dict[str, Any]:
        session = self._sessions.get(session_id, _SpeakerSession())
        return {
            "next_speaker_number": session.next_speaker_number,
            "profiles": {
                speaker_id: {
                    "centroid": list(profile.centroid),
                    "effective_samples": profile.effective_samples,
                    "display_name": profile.display_name,
                }
                for speaker_id, profile in sorted(session.profiles.items())
            },
        }

    def restore_session(self, session_id: str, payload: dict[str, Any]) -> None:
        session = _SpeakerSession(next_speaker_number=int(payload["next_speaker_number"]))
        for speaker_id, profile in payload["profiles"].items():
            session.profiles[speaker_id] = SpeakerProfile(
                speaker_id=speaker_id,
                centroid=normalize(profile["centroid"]),
                effective_samples=float(profile["effective_samples"]),
                display_name=profile.get("display_name"),
            )
        self._sessions[session_id] = session

