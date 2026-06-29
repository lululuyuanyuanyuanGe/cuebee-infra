"""Audio-load autoscaler with backlog correction, hysteresis, and cooldown."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AutoscalerObservation:
    audio_arrival_seconds_per_second: float
    worker_rtf: float
    backlog_seconds: float
    p99_queue_delay_ms: float
    worker_utilization: float
    now_seconds: float


class SpeakerAutoscaler:
    def __init__(
        self,
        minimum_replicas: int = 1,
        maximum_replicas: int = 16,
        target_utilization: float = 0.70,
        backlog_target_seconds: float = 2.0,
        cooldown_seconds: float = 30.0,
        scale_down_windows: int = 3,
    ) -> None:
        if not 0 < minimum_replicas <= maximum_replicas:
            raise ValueError("invalid replica bounds")
        if not 0 < target_utilization <= 1:
            raise ValueError("target utilization must be between zero and one")
        self.minimum_replicas = minimum_replicas
        self.maximum_replicas = maximum_replicas
        self.target_utilization = target_utilization
        self.backlog_target_seconds = backlog_target_seconds
        self.cooldown_seconds = cooldown_seconds
        self.scale_down_windows = scale_down_windows
        self.current_replicas = minimum_replicas
        self._last_scale_at = float("-inf")
        self._low_load_windows = 0

    def observe(self, observation: AutoscalerObservation) -> int:
        if observation.worker_rtf <= 0:
            raise ValueError("Real-time Factor must be positive")
        base = math.ceil(
            observation.audio_arrival_seconds_per_second
            * observation.worker_rtf
            / self.target_utilization
        )
        backlog_correction = max(
            0,
            math.ceil(observation.backlog_seconds / self.backlog_target_seconds) - 1,
        )
        desired = min(
            self.maximum_replicas,
            max(self.minimum_replicas, base + backlog_correction),
        )
        in_cooldown = observation.now_seconds - self._last_scale_at < self.cooldown_seconds

        if desired > self.current_replicas:
            self._low_load_windows = 0
            if not in_cooldown:
                self.current_replicas = desired
                self._last_scale_at = observation.now_seconds
            return self.current_replicas

        low_load = (
            desired < self.current_replicas
            and observation.worker_utilization < self.target_utilization * 0.65
            and observation.backlog_seconds < self.backlog_target_seconds * 0.25
        )
        self._low_load_windows = self._low_load_windows + 1 if low_load else 0
        if (
            low_load
            and self._low_load_windows >= self.scale_down_windows
            and not in_cooldown
        ):
            self.current_replicas = desired
            self._last_scale_at = observation.now_seconds
            self._low_load_windows = 0
        return self.current_replicas

