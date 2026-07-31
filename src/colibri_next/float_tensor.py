from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .q4 import np
from .tensor_container import ColiTensorFile


@dataclass(frozen=True, slots=True)
class FloatTensor:
    """Small BF16 or FP32 tensor decoded as Python float values."""

    shape: tuple[int, ...]
    dtype: str
    data: bytes

    def __post_init__(self) -> None:
        widths = {"BF16": 2, "F32": 4}
        if self.dtype not in widths:
            raise ValueError(f"unsupported floating tensor dtype: {self.dtype}")
        if len(self.data) != math.prod(self.shape) * widths[self.dtype]:
            raise ValueError("floating payload length does not match tensor shape")

    @classmethod
    def from_container(cls, container: ColiTensorFile, name: str) -> "FloatTensor":
        info = container.tensors[name]
        return cls(shape=info.shape, dtype=info.dtype, data=container.read(name))

    def values(self, *, prefer_numpy: bool = True) -> list[float]:
        if self.dtype == "F32":
            if prefer_numpy and np is not None:
                return np.frombuffer(self.data, dtype="<f4").tolist()
            return [value for (value,) in struct.iter_unpack("<f", self.data)]
        if prefer_numpy and np is not None:
            source = np.frombuffer(self.data, dtype="<u2")
            return (source.astype(np.uint32) << 16).view(np.float32).tolist()
        return [
            struct.unpack("<f", struct.pack("<I", bits << 16))[0]
            for (bits,) in struct.iter_unpack("<H", self.data)
        ]
