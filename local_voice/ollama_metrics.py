"""Privacy-safe Ollama profiling for non-streaming /api/chat calls.

Ollama returns durations in nanoseconds and token *counts* separately.
Wall-clock and backend durations overlap and must never be summed together.
No prompts, completions, reasoning or user content is stored here.
"""
from math import isfinite
from typing import Any, Mapping

_OPERATIONS = frozenset({
    "availability", "name", "address", "confirmation", "react", "answer", "scoring",
})
_DURATIONS = (
    ("load_duration", "load_ms"),
    ("prompt_eval_duration", "prompt_ms"),
    ("eval_duration", "decode_ms"),
    ("total_duration", "total_ms"),
)
_COUNTS = (
    ("prompt_eval_count", "input_tokens"),
    ("eval_count", "output_tokens"),
)


def _nonnegative(value: Any) -> float | None:
    if type(value) not in (int, float):
        return None
    number = float(value)
    return number if isfinite(number) and number >= 0 else None


def ollama_call_stats(
    result: Mapping[str, Any],
    *,
    operation: str,
    wall_ms: float,
    success: bool = True,
) -> dict[str, str | float | int]:
    """Normalize one Ollama call, without including model response text."""
    if operation not in _OPERATIONS:
        raise ValueError("Invalid Ollama profiling stage")
    elapsed = _nonnegative(wall_ms)
    entry: dict[str, str | float | int] = {
        "operation": operation,
        "status": "ok" if success else "error",
        "wall_ms": round(elapsed if elapsed is not None else 0, 2),
    }
    if not success:
        return entry
    for source, target in _DURATIONS:
        raw = _nonnegative(result.get(source))
        if raw is not None:
            entry[target] = round(raw / 1_000_000, 2)
    for source, target in _COUNTS:
        raw = result.get(source)
        if type(raw) is int and raw >= 0:
            entry[target] = raw
    count = entry.get("output_tokens")
    duration_ns = _nonnegative(result.get("eval_duration"))
    if type(count) is int and duration_ns and duration_ns > 0:
        entry["decode_tokens_per_s"] = round(count * 1_000_000_000 / duration_ns, 2)
    prompt_count = entry.get("input_tokens")
    prompt_ns = _nonnegative(result.get("prompt_eval_duration"))
    if type(prompt_count) is int and prompt_ns and prompt_ns > 0:
        entry["prompt_tokens_per_s"] = round(prompt_count * 1_000_000_000 / prompt_ns, 2)
    return entry


def record_ollama_call(
    result: Mapping[str, Any],
    *,
    operation: str,
    wall_ms: float,
    timings_ms: dict[str, float] | None = None,
    calls: list[dict[str, str | float | int]] | None = None,
    success: bool = True,
) -> None:
    """Store timing parts separately from token counts for the same call."""
    item = ollama_call_stats(result, operation=operation, wall_ms=wall_ms, success=success)
    if calls is not None:
        calls.append(item)
    if timings_ms is None:
        return
    for name in ("wall_ms", "load_ms", "prompt_ms", "decode_ms", "total_ms"):
        val = item.get(name)
        if type(val) in (int, float):
            # Keep separate keys for each operation, and never put token
            # counts into a field whose units are milliseconds.
            key = f"{operation}_ollama_{name}"
            timings_ms[key] = timings_ms.get(key, 0.0) + float(val)
    # Keep the original answer metrics in v1 API / dashboard.
    if operation == "answer":
        for name, key in (
            ("load_ms", "ollama_load_ms"),
            ("prompt_ms", "ollama_prompt_ms"),
            ("decode_ms", "ollama_decode_ms"),
            ("total_ms", "ollama_total_ms"),
        ):
            val = item.get(name)
            if type(val) in (int, float):
                timings_ms[key] = timings_ms.get(key, 0.0) + float(val)
