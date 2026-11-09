"""Per-session latency telemetry, with no transcript or model reasoning.

Durations are measured using a monotonic clock (perf_counter); all public
values are milliseconds. Missing stages are omitted, not reported as zero.
"""
from dataclasses import dataclass, field
from math import ceil, isfinite
from statistics import mean
from time import perf_counter
from typing import Literal

METRICS_CONTRACT_VERSION = "turn-timing-v1"

TurnStatus = Literal["running", "completed", "interrupted", "failed", "no_speech"]


def elapsed_ms(started_at: float) -> float:
    return max(0.0, (perf_counter() - started_at) * 1000.0)


@dataclass(slots=True)
class TurnTiming:
    turn_id: int
    phase: str
    kind: Literal["candidate", "greeting"] = "candidate"
    started_at: float = field(default_factory=perf_counter, repr=False)
    stages_ms: dict[str, float] = field(default_factory=dict)
    status: TurnStatus = "running"
    vad_frames: int = 0

    def add(self, name: str, duration_ms: float) -> None:
        if isfinite(duration_ms) and duration_ms >= 0:
            self.stages_ms[name] = self.stages_ms.get(name, 0.0) + duration_ms

    def mark(self, name: str) -> None:
        self.stages_ms[name] = elapsed_ms(self.started_at)

    def finish(self, status: TurnStatus = "completed") -> None:
        if self.status != "running":
            return
        self.status = status
        self.mark("end_to_end_ms")

    def public(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "phase": self.phase,
            "kind": self.kind,
            "status": self.status,
            "vad_frames": self.vad_frames,
            "stages_ms": {key: round(value, 2) for key, value in self.stages_ms.items()},
        }


def _percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[ceil(0.95 * len(ordered)) - 1]


def summarize_timings(turns: list[TurnTiming]) -> dict:
    """Aggregate completed candidate turns only; greetings stay in history."""
    completed = [
        turn for turn in turns
        if turn.kind == "candidate" and turn.status == "completed"
    ]
    names = sorted({key for turn in completed for key in turn.stages_ms})
    average: dict[str, float] = {}
    p95: dict[str, float] = {}
    for name in names:
        values = [turn.stages_ms[name] for turn in completed if name in turn.stages_ms]
        if values:
            average[name] = round(mean(values), 2)
            p95[name] = round(_percentile_95(values), 2)
    return {
        "completed_turns": len(completed),
        "average_ms": average,
        "p95_ms": p95,
    }
