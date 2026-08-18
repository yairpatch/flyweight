"""Sampling configuration accepted by the native generation engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    temperature: float = 0.0
    top_k: int = 20
    top_p: float = 0.95
    seed: int | None = None
    # Defaults match llama.cpp's long-standing repeat_penalty/repeat_last_n,
    # deliberately on rather than off: with no penalty a low-bit checkpoint
    # locks onto a line and repeats it until the token budget runs out, which
    # is a far more visible failure than the mild cost of 1.1 -- legitimately
    # repetitive text, code especially, becomes slightly less likely.
    repetition_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    penalty_window: int = 64

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 1.0 <= self.repetition_penalty <= 2.0:
            raise ValueError("repetition_penalty must be in [1, 2]")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be in [-2, 2]")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be in [-2, 2]")
        if self.penalty_window < 0:
            raise ValueError("penalty_window must be non-negative")
