"""Multi-tenant speaker embedding serving components."""

from cuebee.speaker.autoscaler import AutoscalerObservation, SpeakerAutoscaler
from cuebee.speaker.service import SpeakerService, SubmitStatus
from cuebee.speaker.worker import DeterministicEmbeddingWorker

__all__ = [
    "AutoscalerObservation",
    "DeterministicEmbeddingWorker",
    "SpeakerAutoscaler",
    "SpeakerService",
    "SubmitStatus",
]

