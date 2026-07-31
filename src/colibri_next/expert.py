from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import tanh


@dataclass(frozen=True, slots=True)
class ExpertKey:
    layer: int
    expert: int


@dataclass(slots=True)
class Expert:
    key: ExpertKey
    width: int
    weights: array
    bias: array

    @property
    def byte_size(self) -> int:
        return (len(self.weights) + len(self.bias)) * self.weights.itemsize

    def forward(self, hidden: list[float]) -> list[float]:
        if len(hidden) != self.width:
            raise ValueError(f"expected hidden width {self.width}, got {len(hidden)}")

        output: list[float] = []
        for row in range(self.width):
            offset = row * self.width
            total = self.bias[row]
            for column, value in enumerate(hidden):
                total += self.weights[offset + column] * value
            output.append(tanh(total))
        return output
