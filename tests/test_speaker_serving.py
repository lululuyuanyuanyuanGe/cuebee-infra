from __future__ import annotations

import unittest

from cuebee.event_schema import AudioChunk, SpeakerSegment
from cuebee.speaker.alignment import align_speaker
from cuebee.speaker.autoscaler import AutoscalerObservation, SpeakerAutoscaler
from cuebee.speaker.batcher import SpeakerMicroBatcher
from cuebee.speaker.service import SpeakerService, SubmitStatus
from cuebee.speaker.session_store import SpeakerSessionStore
from cuebee.speaker.worker import EmbeddingResult


class SignWorker:
    def __init__(self, worker_id: str = "sign-worker") -> None:
        self.worker_id = worker_id
        self.batch_sizes: list[int] = []

    def embed_batch(self, chunks: list[AudioChunk]) -> list[EmbeddingResult]:
        self.batch_sizes.append(len(chunks))
        return [
            EmbeddingResult(
                chunk.chunk_id,
                (1.0, 0.0) if chunk.samples[0] > 0 else (0.0, 1.0),
                1.0,
                0.001,
            )
            for chunk in chunks
        ]


def chunk(session: str, chunk_id: str, sign: float, deadline_ms: int | None = None) -> AudioChunk:
    return AudioChunk(
        session,
        chunk_id,
        0,
        500,
        tuple(sign * 0.2 for _ in range(80)),
        deadline_ms=deadline_ms,
    )


class SpeakerServingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.worker = SignWorker()
        self.store = SpeakerSessionStore(assignment_threshold=0.8)
        self.service = SpeakerService(
            [self.worker],
            session_store=self.store,
            batcher=SpeakerMicroBatcher(max_batch_size=4, max_wait_ms=20),
        )

    def test_cross_session_micro_batch(self) -> None:
        self.assertEqual(self.service.submit(chunk("a", "a1", 1), 0), SubmitStatus.ACCEPTED)
        self.assertEqual(self.service.submit(chunk("b", "b1", -1), 0), SubmitStatus.ACCEPTED)
        output = self.service.flush(25)
        self.assertEqual(len(output), 2)
        self.assertEqual(self.worker.batch_sizes, [2])

    def test_stable_ids_and_worker_replacement(self) -> None:
        self.service.submit(chunk("a", "a1", 1), 0)
        first = self.service.flush(25)[0]
        self.service.replace_workers([SignWorker("replacement")])
        self.service.submit(chunk("a", "a2", 1), 30)
        second = self.service.flush(60)[0]
        self.service.submit(chunk("a", "a3", -1), 70)
        third = self.service.flush(100)[0]
        self.assertEqual(first.speaker_id, second.speaker_id)
        self.assertNotEqual(first.speaker_id, third.speaker_id)

    def test_silence_is_filtered_before_batching(self) -> None:
        silent = AudioChunk("a", "silent", 0, 500, tuple(0.0 for _ in range(80)))
        self.assertEqual(self.service.submit(silent, 0), SubmitStatus.SILENCE)
        self.assertEqual(self.service.batcher.queued_chunks, 0)

    def test_deadline_dispatches_before_max_wait(self) -> None:
        urgent = chunk("a", "urgent", 1, deadline_ms=4)
        self.service.submit(urgent, 0)
        self.assertEqual(len(self.service.flush(0)), 1)

    def test_store_can_be_restored(self) -> None:
        speaker_id, _ = self.store.assign("a", (1.0, 0.0), 1.0)
        payload = self.store.export_session("a")
        restored = SpeakerSessionStore(assignment_threshold=0.8)
        restored.restore_session("a", payload)
        restored_id, _ = restored.assign("a", (0.99, 0.01), 1.0)
        self.assertEqual(restored_id, speaker_id)


class AutoscalerTests(unittest.TestCase):
    def test_audio_capacity_and_backlog_scale_up(self) -> None:
        scaler = SpeakerAutoscaler(cooldown_seconds=0)
        replicas = scaler.observe(AutoscalerObservation(6.0, 0.5, 5.0, 100.0, 0.9, 1.0))
        self.assertGreaterEqual(replicas, 6)

    def test_scale_down_requires_consecutive_windows(self) -> None:
        scaler = SpeakerAutoscaler(cooldown_seconds=0, scale_down_windows=2)
        scaler.observe(AutoscalerObservation(6.0, 0.5, 0.0, 10.0, 0.9, 1.0))
        high = scaler.current_replicas
        first = scaler.observe(AutoscalerObservation(0.1, 0.5, 0.0, 1.0, 0.1, 2.0))
        second = scaler.observe(AutoscalerObservation(0.1, 0.5, 0.0, 1.0, 0.1, 3.0))
        self.assertEqual(first, high)
        self.assertLess(second, high)


class AlignmentTests(unittest.TestCase):
    def test_largest_overlap_wins(self) -> None:
        speakers = [
            SpeakerSegment("s", "a", 0, 300, "spk_001", 0.9, 1.0),
            SpeakerSegment("s", "b", 300, 1000, "spk_002", 0.8, 1.0),
        ]
        attributed = align_speaker("t", "hello", 200, 800, speakers)
        self.assertEqual(attributed.speaker_id, "spk_002")
        self.assertEqual(attributed.overlap_ms, 500)


if __name__ == "__main__":
    unittest.main()
