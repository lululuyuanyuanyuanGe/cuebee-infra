"""Small in-process metric registry with Prometheus-compatible exposition."""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from typing import Any


class MetricRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._samples: dict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, amount: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += amount

    def set(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._samples[name].append(value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(sorted(self._counters.items()))
            gauges = dict(sorted(self._gauges.items()))
            samples = {name: tuple(values) for name, values in self._samples.items()}
        histograms: dict[str, dict[str, float]] = {}
        for name, values in sorted(samples.items()):
            ordered = sorted(values)
            histograms[name] = {
                "count": float(len(ordered)),
                "sum": float(sum(ordered)),
                "p50": _percentile(ordered, 0.50),
                "p95": _percentile(ordered, 0.95),
                "p99": _percentile(ordered, 0.99),
            }
        return {"counters": counters, "gauges": gauges, "histograms": histograms}

    def prometheus(self) -> str:
        snapshot = self.snapshot()
        lines: list[str] = []
        for name, value in snapshot["counters"].items():
            lines.append(f"{_metric_name(name)}_total {value}")
        for name, value in snapshot["gauges"].items():
            lines.append(f"{_metric_name(name)} {value}")
        for name, values in snapshot["histograms"].items():
            metric = _metric_name(name)
            lines.append(f"{metric}_count {values['count']}")
            lines.append(f"{metric}_sum {values['sum']}")
            for quantile in ("p50", "p95", "p99"):
                lines.append(
                    f'{metric}{{quantile="{quantile[1:]}"}} {values[quantile]}'
                )
        return "\n".join(lines) + "\n"


def _metric_name(name: str) -> str:
    return "cuebee_" + "".join(char if char.isalnum() or char == "_" else "_" for char in name)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    index = min(len(values) - 1, math.ceil(quantile * len(values)) - 1)
    return float(values[index])

