"""Sampling configuration accepted by the native generation engine."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping


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


@dataclass(frozen=True, slots=True)
class Setting:
    """One generation setting, described once for every surface that sets it.

    The same settings are reachable from four places -- a server flag, a
    `generation_config.json` beside the checkpoint, an OpenAI-shaped request and
    an Anthropic-shaped one -- and each of those used to carry its own list.
    They drifted: `seed` was reachable only from an OpenAI request, and the
    penalties only from a flag or an OpenAI request, so a value the user set
    anywhere else was silently dropped. This table is what they now share.
    """

    name: str
    kind: type
    help: str
    # `seed` has no default: absent means "pick one", which is not the same as
    # any particular number, so it is carried as None rather than a value.
    optional: bool = False
    # Whether a server-wide default makes sense. A fixed seed for every request
    # on a shared server would make concurrent generations identical.
    per_request_only: bool = False


SETTINGS: tuple[Setting, ...] = (
    Setting("temperature", float, "sampling temperature (0 = greedy)"),
    Setting("top_k", int, "how many candidates the sampler considers"),
    Setting("top_p", float, "nucleus cut over those candidates, in (0, 1]"),
    Setting("seed", int, "fixed RNG seed for a reproducible sample",
            optional=True, per_request_only=True),
    Setting("repetition_penalty", float,
            "penalty on tokens in the recent window, in [1, 2] "
            "(1 disables it; applies to greedy decode too)"),
    Setting("presence_penalty", float,
            "flat penalty on any token already generated, in [-2, 2]"),
    Setting("frequency_penalty", float,
            "penalty per occurrence of a generated token, in [-2, 2]"),
    Setting("penalty_window", int,
            "how many recent generated tokens the penalties look at "
            "(0 disables them)"),
)

# The settings a server-wide default may set, which is every one that is not
# per-request-only.
SERVER_SETTINGS: tuple[Setting, ...] = tuple(
    setting for setting in SETTINGS if not setting.per_request_only
)


def coerce(setting: Setting, value: Any) -> int | float | None:
    """A JSON value as this setting's type, or None when it is not one.

    Booleans are rejected rather than counted as ints: JSON has no separate
    integer type for them, and `"top_k": true` is a mistake, not a 1.
    """
    if value is None or isinstance(value, bool):
        return None
    if setting.kind is int:
        return int(value) if isinstance(value, int) else None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def defaults() -> dict[str, int | float]:
    """The built-in default for every setting that has one.

    Read through `dataclasses.fields` because `SamplingConfig` uses `slots=True`,
    which makes the class attribute a slot descriptor rather than the value.
    """
    return {
        field.name: field.default
        for field in dataclasses.fields(SamplingConfig)
        if field.default is not None and not isinstance(field.default, bool)
    }


def from_values(values: Mapping[str, Any]) -> SamplingConfig:
    """Build a config from whatever subset of settings a caller supplied."""
    known = {setting.name for setting in SETTINGS}
    return SamplingConfig(**{
        name: value for name, value in values.items()
        if name in known and value is not None
    })
