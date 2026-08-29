"""Request/stream value objects shared by the native runtime and HTTP API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class GenerationResult:
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    text: str
    stopped_on_eos: bool
    state_tokens: int


@dataclass(frozen=True, slots=True)
class GenerationStep:
    """One streaming event.

    Intermediate events carry lightweight sequence views; the terminal event
    owns immutable tuples and the complete text.  Consumers should use
    ``text_delta`` while streaming and ``text`` once ``finished`` is true.
    """

    token_id: int | None
    text_delta: str
    prompt_ids: Sequence[int]
    generated_ids: Sequence[int]
    text: str
    stopped_on_eos: bool
    finished: bool
    state_tokens: int
