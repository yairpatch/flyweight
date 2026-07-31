"""Request/stream value objects shared by the native runtime and HTTP API."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationResult:
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    text: str
    stopped_on_eos: bool
    state_tokens: int


@dataclass(frozen=True, slots=True)
class GenerationStep:
    token_id: int | None
    text_delta: str
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    text: str
    stopped_on_eos: bool
    finished: bool
    state_tokens: int
