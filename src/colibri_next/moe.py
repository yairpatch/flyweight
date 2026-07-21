from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .bf16 import BF16Tensor
from .kernels import Q4SwiGLUExpert
from .q4 import Q4BlockTensor
from .tensor_container import ColiTensorFile


@dataclass(frozen=True, slots=True)
class MoEResult:
    output: list[float]
    selected_experts: tuple[int, ...]
    routing_weights: tuple[float, ...]
    router_logits: tuple[float, ...]


class QwenMoELayer:
    """Executable Qwen3.5/3.6 sparse MoE feed-forward block for one token."""

    def __init__(
        self,
        layer_file: Path | str,
        expert_directory: Path | str,
    ):
        self.layer_file = Path(layer_file)
        self.expert_directory = Path(expert_directory)
        container = ColiTensorFile(self.layer_file)
        self.layer = int(container.metadata["layer"])
        self.top_k = int(container.metadata["top_k"])
        self.rms_norm_eps = float(container.metadata["rms_norm_eps"])
        self.router = BF16Tensor.from_container(container, "router.weight")
        self.shared_gate = BF16Tensor.from_container(
            container, "shared_expert_gate.weight"
        )
        self.post_attention_norm = BF16Tensor.from_container(
            container, "post_attention_layernorm.weight"
        )
        self.shared_expert = Q4SwiGLUExpert(
            gate_up=Q4BlockTensor.from_container(
                container, "shared_expert.gate_up_proj"
            ),
            down=Q4BlockTensor.from_container(container, "shared_expert.down_proj"),
        )
        self._post_attention_norm_weights = self.post_attention_norm.values()
        self._experts: dict[int, Q4SwiGLUExpert] = {}
        self.expert_device = "cuda"
        self._validate()

    @classmethod
    def from_model_directory(
        cls, root: Path | str, layer: int
    ) -> "QwenMoELayer":
        root_path = Path(root)
        return cls(
            root_path / "moe_layers" / f"layer-{layer:03d}.coli",
            root_path / "experts" / f"layer-{layer:03d}",
        )

    @property
    def hidden_size(self) -> int:
        return self.router.shape[1]

    @property
    def expert_count(self) -> int:
        return self.router.shape[0]

    @property
    def estimated_expert_storage_bytes(self) -> int:
        if self.expert_count == 0:
            return 0
        sample = self.expert_directory / "expert-0000.coli"
        return sample.stat().st_size * self.expert_count

    def preload_experts(self) -> int:
        for expert_id in range(self.expert_count):
            self._expert(expert_id)
        return self.expert_count

    def set_expert_device(self, device: str) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"unsupported expert device: {device}")
        self.expert_device = device

    def route(
        self, hidden: list[float], *, allow_cuda: bool = True
    ) -> tuple[list[float], list[int], list[float]]:
        logits = self.router.matvec(hidden, allow_cuda=allow_cuda)
        maximum = max(logits)
        probabilities = [math.exp(logit - maximum) for logit in logits]
        denominator = sum(probabilities)
        probabilities = [probability / denominator for probability in probabilities]
        selected = sorted(
            range(len(probabilities)), key=probabilities.__getitem__, reverse=True
        )[: self.top_k]
        selected_total = sum(probabilities[index] for index in selected)
        weights = [probabilities[index] / selected_total for index in selected]
        return logits, selected, weights

    def forward(
        self, hidden: list[float], *, allow_cuda: bool = True
    ) -> MoEResult:
        logits, selected, weights = self.route(hidden, allow_cuda=allow_cuda)
        shared_logit = self.shared_gate.matvec(hidden, allow_cuda=allow_cuda)[0]
        shared_weight = _sigmoid(shared_logit)

        from .cuda import active_cuda

        accelerator = active_cuda()
        experts = [self._expert(expert_id) for expert_id in selected]
        if accelerator is not None and self.expert_device == "cuda" and allow_cuda:
            output = accelerator.q4_moe(
                experts,
                weights,
                self.shared_expert,
                shared_weight,
                hidden,
            )
        else:
            from .native import active_native

            backend = active_native()
            if backend is not None:
                output = backend.q4_moe(
                    experts,
                    weights,
                    self.shared_expert,
                    shared_weight,
                    hidden,
                )
            else:
                # This branch runs only when the native CPU backend is
                # unavailable; NumPy is strictly faster than the pure-Python
                # list path (and degrades to it if NumPy is missing), so the
                # CPU-offloaded experts should always prefer it.
                prefer_numpy = True
                routed = [0.0] * self.hidden_size
                for expert, weight in zip(experts, weights):
                    expert_output = expert.forward(
                        hidden, prefer_numpy=prefer_numpy, allow_cuda=allow_cuda
                    )
                    routed = [
                        current + weight * value
                        for current, value in zip(routed, expert_output)
                    ]
                shared_output = self.shared_expert.forward(
                    hidden, prefer_numpy=prefer_numpy, allow_cuda=allow_cuda
                )
                output = [
                    routed_value + shared_weight * shared_value
                    for routed_value, shared_value in zip(routed, shared_output)
                ]
        return MoEResult(
            output=output,
            selected_experts=tuple(selected),
            routing_weights=tuple(weights),
            router_logits=tuple(logits),
        )

    def forward_residual(
        self, hidden: list[float], *, allow_cuda: bool = True
    ) -> MoEResult:
        normalized = self.normalize(hidden)
        result = self.forward(normalized, allow_cuda=allow_cuda)
        return MoEResult(
            output=[residual + value for residual, value in zip(hidden, result.output)],
            selected_experts=result.selected_experts,
            routing_weights=result.routing_weights,
            router_logits=result.router_logits,
        )

    def normalize(self, hidden: list[float]) -> list[float]:
        weights = self._post_attention_norm_weights
        variance = sum(value * value for value in hidden) / len(hidden)
        inverse_rms = 1.0 / math.sqrt(variance + self.rms_norm_eps)
        return [
            value * inverse_rms * (1.0 + weight)
            for value, weight in zip(hidden, weights)
        ]

    def _expert(self, expert_id: int) -> Q4SwiGLUExpert:
        expert = self._experts.get(expert_id)
        if expert is None:
            path = self.expert_directory / f"expert-{expert_id:04d}.coli"
            expert = Q4SwiGLUExpert.from_file(path)
            self._experts[expert_id] = expert
        return expert

    def _validate(self) -> None:
        if self.router.shape[0] < self.top_k:
            raise ValueError("router has fewer experts than requested top-k")
        if self.shared_gate.shape != (1, self.hidden_size):
            raise ValueError(f"invalid shared expert gate shape: {self.shared_gate.shape}")
        if self.post_attention_norm.shape != (self.hidden_size,):
            raise ValueError(
                f"invalid post-attention norm shape: {self.post_attention_norm.shape}"
            )
        self.shared_expert.validate()


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)
