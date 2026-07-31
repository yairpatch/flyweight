from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MoEModelSpec:
    id: str
    total_parameters: int
    active_parameters: int
    layers: int
    hidden_size: int
    experts_per_layer: int
    routed_experts: int
    shared_experts: int
    expert_intermediate_size: int
    state_bytes_per_token: int
    native_context_length: int

    @property
    def parameters_per_expert(self) -> int:
        return 3 * self.hidden_size * self.expert_intermediate_size

    @property
    def routed_expert_parameters(self) -> int:
        return min(
            self.total_parameters,
            self.layers * self.experts_per_layer * self.parameters_per_expert,
        )

    @property
    def static_parameters(self) -> int:
        return self.total_parameters - self.routed_expert_parameters

    def bytes_for(self, parameters: int, bits: int) -> int:
        if bits not in {4, 8, 16, 32}:
            raise ValueError("quantization bits must be one of 4, 8, 16, or 32")
        raw_bytes = parameters * bits / 8
        overhead = {4: 1.125, 8: 1.03}.get(bits, 1.0)
        return int(raw_bytes * overhead)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "parameters_per_expert": self.parameters_per_expert,
                "routed_expert_parameters": self.routed_expert_parameters,
                "static_parameters": self.static_parameters,
            }
        )
        return value


QWEN36_35B_A3B = MoEModelSpec(
    id="qwen3.6-35b-a3b",
    total_parameters=35_000_000_000,
    active_parameters=3_000_000_000,
    layers=40,
    hidden_size=2048,
    experts_per_layer=256,
    routed_experts=8,
    shared_experts=1,
    expert_intermediate_size=512,
    state_bytes_per_token=20_480,
    native_context_length=262_144,
)


_MODEL_ALIASES = {
    "qwen3.6-35b-a3b": QWEN36_35B_A3B,
    "qwen/qwen3.6-35b-a3b": QWEN36_35B_A3B,
    "qwen36-35b-a3b": QWEN36_35B_A3B,
}


def model_spec(model_id: str) -> MoEModelSpec:
    try:
        return _MODEL_ALIASES[model_id.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(set(spec.id for spec in _MODEL_ALIASES.values())))
        raise ValueError(f"unsupported model {model_id!r}; available: {supported}") from error
