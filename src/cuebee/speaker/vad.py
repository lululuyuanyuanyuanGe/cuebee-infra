"""Lightweight Voice Activity Detection for the serving admission path."""

from __future__ import annotations

import math
from typing import Sequence


class EnergyVAD:
    def __init__(self, rms_threshold: float = 0.01, min_active_ratio: float = 0.05) -> None:
        if rms_threshold < 0 or not 0 <= min_active_ratio <= 1:
            raise ValueError("invalid Voice Activity Detection thresholds")
        self.rms_threshold = rms_threshold
        self.min_active_ratio = min_active_ratio

    def is_speech(self, samples: Sequence[float]) -> bool:
        if not samples:
            return False
        square_mean = sum(sample * sample for sample in samples) / len(samples)
        rms = math.sqrt(square_mean)
        active = sum(abs(sample) >= self.rms_threshold for sample in samples) / len(samples)
        return rms >= self.rms_threshold and active >= self.min_active_ratio

