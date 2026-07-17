from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .causal_lm import QwenForCausalLM


@dataclass(frozen=True, slots=True)
class LogitComparison:
    position: int
    input_token: int
    colibri_greedy_token: int
    reference_greedy_token: int
    greedy_match: bool
    max_absolute_error: float
    mean_absolute_error: float
    cosine_similarity: float
    top_k_overlap: float


@dataclass(frozen=True, slots=True)
class ValidationReport:
    input_tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    comparisons: tuple[LogitComparison, ...]
    top_k: int

    @property
    def greedy_matches(self) -> int:
        return sum(item.greedy_match for item in self.comparisons)

    @property
    def all_greedy_tokens_match(self) -> bool:
        return self.greedy_matches == len(self.comparisons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": list(self.input_tokens),
            "generated_tokens": list(self.generated_tokens),
            "top_k": self.top_k,
            "greedy_matches": self.greedy_matches,
            "comparisons_count": len(self.comparisons),
            "all_greedy_tokens_match": self.all_greedy_tokens_match,
            "comparisons": [asdict(item) for item in self.comparisons],
        }


def compare_logit_vectors(
    colibri_logits: Sequence[float],
    reference_logits: Sequence[float],
    *,
    position: int,
    input_token: int,
    top_k: int = 10,
) -> LogitComparison:
    if len(colibri_logits) != len(reference_logits):
        raise ValueError(
            "logit vocabulary sizes differ: "
            f"{len(colibri_logits)} != {len(reference_logits)}"
        )
    if not colibri_logits:
        raise ValueError("logit vectors must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    colibri = [float(value) for value in colibri_logits]
    reference = [float(value) for value in reference_logits]
    errors = [abs(left - right) for left, right in zip(colibri, reference)]
    dot = sum(left * right for left, right in zip(colibri, reference))
    left_norm = math.sqrt(sum(value * value for value in colibri))
    right_norm = math.sqrt(sum(value * value for value in reference))
    cosine = dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
    count = min(top_k, len(colibri))
    colibri_top = set(sorted(range(len(colibri)), key=colibri.__getitem__, reverse=True)[:count])
    reference_top = set(
        sorted(range(len(reference)), key=reference.__getitem__, reverse=True)[:count]
    )
    colibri_greedy = max(range(len(colibri)), key=colibri.__getitem__)
    reference_greedy = max(range(len(reference)), key=reference.__getitem__)
    return LogitComparison(
        position=position,
        input_token=input_token,
        colibri_greedy_token=colibri_greedy,
        reference_greedy_token=reference_greedy,
        greedy_match=colibri_greedy == reference_greedy,
        max_absolute_error=max(errors),
        mean_absolute_error=sum(errors) / len(errors),
        cosine_similarity=cosine,
        top_k_overlap=len(colibri_top & reference_top) / count,
    )


class TransformersReference:
    """Lazy Hugging Face model wrapper used only by the validation command."""

    def __init__(
        self,
        source: Path | str,
        *,
        device: str = "cpu",
        dtype: str = "auto",
        trust_remote_code: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as error:
            raise RuntimeError(
                "Transformers validation requires the 'validation' extra: "
                "pip install -e '.[validation]'"
            ) from error

        torch_dtype: Any = dtype
        if dtype != "auto":
            try:
                torch_dtype = getattr(torch, dtype)
            except AttributeError as error:
                raise ValueError(f"unsupported torch dtype: {dtype}") from error
        self._torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(
            str(source),
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        self.device = device

    def logits(self, token_ids: Sequence[int]) -> list[float]:
        with self._torch.inference_mode():
            inputs = self._torch.tensor([list(token_ids)], device=self.device)
            logits = self.model(input_ids=inputs).logits[0, -1]
        return logits.float().cpu().tolist()


def validate_against_reference(
    model: QwenForCausalLM,
    reference: Any,
    token_ids: Sequence[int],
    *,
    generate_tokens: int = 0,
    top_k: int = 10,
) -> ValidationReport:
    if not token_ids:
        raise ValueError("token_ids must not be empty")
    if generate_tokens < 0:
        raise ValueError("generate_tokens must not be negative")

    context = [int(token) for token in token_ids]
    state = model.new_state()
    device_decode = bool(getattr(model, "device_decode_available", False))
    result = (
        model.prefill_device(context, state)
        if device_decode
        else model.prefill(context, state)
    )
    comparisons: list[LogitComparison] = []
    generated: list[int] = []

    for step in range(generate_tokens + 1):
        reference_logits = reference.logits(context)
        comparisons.append(
            compare_logit_vectors(
                result.logits,
                reference_logits,
                position=len(context) - 1,
                input_token=context[-1],
                top_k=top_k,
            )
        )
        if step == generate_tokens:
            break
        next_token = comparisons[-1].reference_greedy_token
        generated.append(next_token)
        context.append(next_token)
        result = (
            model.forward_token_device(next_token, state)
            if device_decode
            else model.forward_token(next_token, state)
        )

    return ValidationReport(
        input_tokens=tuple(int(token) for token in token_ids),
        generated_tokens=tuple(generated),
        comparisons=tuple(comparisons),
        top_k=top_k,
    )
