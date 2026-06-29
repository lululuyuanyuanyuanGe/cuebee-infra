"""Stateless speaker embedding worker boundary."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

from cuebee.event_schema import AudioChunk


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    chunk_id: str
    embedding: tuple[float, ...]
    quality: float
    inference_seconds: float


class EmbeddingWorker(Protocol):
    worker_id: str

    def embed_batch(self, chunks: Sequence[AudioChunk]) -> list[EmbeddingResult]: ...


class DeterministicEmbeddingWorker:
    """Dependency-free feature worker for local flow and load tests.

    Production injects an ONNX Runtime worker with the same stateless contract.
    """

    def __init__(self, worker_id: str = "worker-0", dimensions: int = 8) -> None:
        if dimensions < 4:
            raise ValueError("embedding dimensions must be at least four")
        self.worker_id = worker_id
        self.dimensions = dimensions

    def embed_batch(self, chunks: Sequence[AudioChunk]) -> list[EmbeddingResult]:
        results: list[EmbeddingResult] = []
        started = time.perf_counter()
        for chunk in chunks:
            samples = chunk.samples
            mean = sum(samples) / len(samples)
            rms = math.sqrt(sum(value * value for value in samples) / len(samples))
            zcr = sum(
                (left >= 0) != (right >= 0)
                for left, right in zip(samples, samples[1:], strict=False)
            ) / max(1, len(samples) - 1)
            peak = max(abs(value) for value in samples)
            features = [mean, rms, zcr, peak]
            windows = self.dimensions - len(features)
            for index in range(windows):
                start = index * len(samples) // windows
                end = (index + 1) * len(samples) // windows
                window = samples[start:end] or samples
                features.append(sum(abs(value) for value in window) / len(window))
            embedding = normalize(features)
            quality = max(0.0, min(1.0, rms / 0.15))
            results.append(EmbeddingResult(chunk.chunk_id, embedding, quality, 0.0))
        elapsed = time.perf_counter() - started
        per_chunk = elapsed / max(1, len(chunks))
        return [
            EmbeddingResult(result.chunk_id, result.embedding, result.quality, per_chunk)
            for result in results
        ]


def normalize(values: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return tuple(0.0 for _ in values)
    return tuple(value / norm for value in values)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    return sum(a * b for a, b in zip(left, right, strict=True))

