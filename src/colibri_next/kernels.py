from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .q4 import Q4BlockTensor, np
from .tensor_container import ColiTensorFile


@dataclass(slots=True)
class Q4SwiGLUExpert:
    gate_up: Q4BlockTensor
    down: Q4BlockTensor
    # ctypes views of the packed tensors, built lazily by the native backend
    # so repeated fused-MoE calls skip per-call buffer marshalling.
    _native_pointers: object = None

    @classmethod
    def from_file(cls, path: Path | str) -> "Q4SwiGLUExpert":
        container = ColiTensorFile(path)
        return cls(
            gate_up=Q4BlockTensor.from_container(container, "gate_up_proj"),
            down=Q4BlockTensor.from_container(container, "down_proj"),
        )

    @property
    def hidden_size(self) -> int:
        return self.gate_up.shape[1]

    @property
    def intermediate_size(self) -> int:
        return self.gate_up.shape[0] // 2

    @property
    def engine(self) -> str:
        return "numpy" if np is not None else "python"

    def validate(self) -> None:
        if len(self.gate_up.shape) != 2 or self.gate_up.shape[0] % 2:
            raise ValueError(f"invalid fused gate/up shape: {self.gate_up.shape}")
        expected_down = (self.hidden_size, self.intermediate_size)
        if self.down.shape != expected_down:
            raise ValueError(f"down projection shape {self.down.shape} != {expected_down}")

    def forward(
        self,
        hidden: list[float],
        *,
        prefer_numpy: bool = True,
        allow_cuda: bool = True,
    ) -> list[float]:
        self.validate()
        if prefer_numpy:
            from .cuda import active_cuda

            accelerator = active_cuda()
            if accelerator is not None and allow_cuda:
                return accelerator.q4_swiglu(self.gate_up, self.down, hidden)
        gate_up = self.gate_up.matvec(
            hidden, prefer_numpy=prefer_numpy, allow_cuda=allow_cuda
        )
        intermediate = self.intermediate_size
        activated = [
            _silu(gate_up[index]) * gate_up[intermediate + index]
            for index in range(intermediate)
        ]
        return self.down.matvec(
            activated, prefer_numpy=prefer_numpy, allow_cuda=allow_cuda
        )


def _silu(value: float) -> float:
    if value >= 0:
        return value / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return value * exponential / (1.0 + exponential)
