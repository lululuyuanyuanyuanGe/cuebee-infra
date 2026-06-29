"""Voice Activity Detection, micro-batching, embedding, and stable assignment."""

from __future__ import annotations

import time
from enum import Enum
from typing import Sequence

from cuebee.event_schema import AudioChunk, SpeakerSegment
from cuebee.metrics import MetricRegistry
from cuebee.speaker.batcher import SpeakerMicroBatcher
from cuebee.speaker.session_store import SpeakerSessionStore
from cuebee.speaker.vad import EnergyVAD
from cuebee.speaker.worker import EmbeddingWorker


class SubmitStatus(str, Enum):
    ACCEPTED = "accepted"
    SILENCE = "silence"
    BACKPRESSURE = "backpressure"


class SpeakerService:
    def __init__(
        self,
        workers: Sequence[EmbeddingWorker],
        session_store: SpeakerSessionStore | None = None,
        batcher: SpeakerMicroBatcher | None = None,
        vad: EnergyVAD | None = None,
        metrics: MetricRegistry | None = None,
    ) -> None:
        if not workers:
            raise ValueError("at least one speaker worker is required")
        self.workers = list(workers)
        self.store = session_store or SpeakerSessionStore()
        self.batcher = batcher or SpeakerMicroBatcher()
        self.vad = vad or EnergyVAD()
        self.metrics = metrics or MetricRegistry()
        self._next_worker = 0

    def submit(self, chunk: AudioChunk, now_ms: int) -> SubmitStatus:
        if not self.vad.is_speech(chunk.samples):
            self.metrics.inc("speaker_silence_dropped")
            return SubmitStatus.SILENCE
        if not self.batcher.enqueue(chunk, now_ms):
            self.metrics.inc("speaker_backpressure_dropped")
            return SubmitStatus.BACKPRESSURE
        self.metrics.inc("speaker_chunks_accepted")
        self.metrics.set("speaker_backlog_seconds", self.batcher.backlog_seconds)
        return SubmitStatus.ACCEPTED

    def flush(self, now_ms: int, force: bool = False) -> list[SpeakerSegment]:
        output: list[SpeakerSegment] = []
        for batch in self.batcher.pop_ready(now_ms, force=force):
            worker = self.workers[self._next_worker % len(self.workers)]
            self._next_worker += 1
            chunks = [item.chunk for item in batch]
            started = time.perf_counter()
            results = worker.embed_batch(chunks)
            elapsed = time.perf_counter() - started
            if len(results) != len(chunks):
                raise RuntimeError("worker returned a different number of embeddings")
            by_chunk = {result.chunk_id: result for result in results}
            self.metrics.observe("speaker_batch_size", float(len(chunks)))
            self.metrics.observe("speaker_batch_inference_seconds", elapsed)
            for item in batch:
                result = by_chunk[item.chunk.chunk_id]
                speaker_id, similarity = self.store.assign(
                    item.chunk.session_id,
                    result.embedding,
                    result.quality,
                )
                output.append(
                    SpeakerSegment(
                        session_id=item.chunk.session_id,
                        chunk_id=item.chunk.chunk_id,
                        start_ms=item.chunk.start_ms,
                        end_ms=item.chunk.end_ms,
                        speaker_id=speaker_id,
                        similarity=similarity,
                        embedding_quality=result.quality,
                    )
                )
                self.metrics.observe(
                    "speaker_queue_delay_ms",
                    float(max(0, now_ms - item.enqueued_ms)),
                )
        self.metrics.set("speaker_backlog_seconds", self.batcher.backlog_seconds)
        return output

    def replace_workers(self, workers: Sequence[EmbeddingWorker]) -> None:
        if not workers:
            raise ValueError("at least one speaker worker is required")
        self.workers = list(workers)
        self._next_worker = 0

