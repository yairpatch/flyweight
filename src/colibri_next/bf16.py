from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .q4 import np
from .tensor_container import ColiTensorFile


@dataclass(frozen=True, slots=True)
class BF16Tensor:
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        if len(self.data) != math.prod(self.shape) * 2:
            raise ValueError("BF16 payload length does not match tensor shape")

    @classmethod
    def from_container(cls, container: ColiTensorFile, name: str) -> "BF16Tensor":
        info = container.tensors[name]
        if info.dtype != "BF16":
            raise ValueError(f"tensor {name} must be BF16, got {info.dtype}")
        return cls(shape=info.shape, data=container.read(name))

    def values(self, *, prefer_numpy: bool = True) -> list[float]:
        if prefer_numpy and np is not None:
            source = np.frombuffer(self.data, dtype="<u2")
            return (source.astype(np.uint32) << 16).view(np.float32).tolist()
        return [
            struct.unpack("<f", struct.pack("<I", bits << 16))[0]
            for (bits,) in struct.iter_unpack("<H", self.data)
        ]

    def row(self, index: int, *, prefer_numpy: bool = True) -> list[float]:
        if len(self.shape) != 2:
            raise ValueError(f"row lookup requires rank-2 BF16 tensor, got {self.shape}")
        rows, columns = self.shape
        if index < 0 or index >= rows:
            raise IndexError(f"row index {index} outside tensor with {rows} rows")
        start = index * columns * 2
        data = self.data[start : start + columns * 2]
        if prefer_numpy and np is not None:
            source = np.frombuffer(data, dtype="<u2")
            return (source.astype(np.uint32) << 16).view(np.float32).tolist()
        return [
            struct.unpack("<f", struct.pack("<I", bits << 16))[0]
            for (bits,) in struct.iter_unpack("<H", data)
        ]

    def matvec(self, vector: list[float], *, prefer_numpy: bool = True) -> list[float]:
        if len(self.shape) != 2:
            raise ValueError(f"matvec requires rank-2 BF16 tensor, got {self.shape}")
        rows, columns = self.shape
        if len(vector) != columns:
            raise ValueError(f"expected vector width {columns}, got {len(vector)}")
        if prefer_numpy:
            from .cuda import active_cuda

            accelerator = active_cuda()
            if accelerator is not None:
                return accelerator.bf16_matvec(self, vector)
        if prefer_numpy and np is not None:
            source = np.frombuffer(self.data, dtype="<u2")
            matrix = (source.astype(np.uint32) << 16).view(np.float32).reshape(
                rows, columns
            )
            return (matrix @ np.asarray(vector, dtype=np.float32)).tolist()
        values = self.values(prefer_numpy=False)
        return [
            sum(
                values[row * columns + column] * vector[column]
                for column in range(columns)
            )
            for row in range(rows)
        ]

    def matvec_chunked(
        self,
        vector: list[float],
        *,
        rows_per_chunk: int = 4096,
        prefer_numpy: bool = True,
    ) -> list[float]:
        if len(self.shape) != 2:
            raise ValueError(f"matvec requires rank-2 BF16 tensor, got {self.shape}")
        rows, columns = self.shape
        if len(vector) != columns:
            raise ValueError(f"expected vector width {columns}, got {len(vector)}")
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")
        if prefer_numpy:
            from .cuda import active_cuda

            accelerator = active_cuda()
            if accelerator is not None:
                return accelerator.bf16_matvec(self, vector)
        if not prefer_numpy or np is None:
            return self.matvec(vector, prefer_numpy=False)
        source = np.frombuffer(self.data, dtype="<u2").reshape(rows, columns)
        input_vector = np.asarray(vector, dtype=np.float32)
        output: list[float] = []
        for start in range(0, rows, rows_per_chunk):
            chunk = source[start : start + rows_per_chunk]
            matrix = (chunk.astype(np.uint32) << 16).view(np.float32)
            output.extend((matrix @ input_vector).tolist())
        return output


