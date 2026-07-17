from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any

from .tensor_container import ColiTensorFile, TensorPayload


BLOCK_SIZE = 32
PACKED_BYTES_PER_BLOCK = BLOCK_SIZE // 2

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised on minimal deployments
    np = None


@dataclass(frozen=True, slots=True)
class Q4BlockTensor:
    shape: tuple[int, ...]
    packed: bytes
    scales: bytes
    block_size: int = BLOCK_SIZE

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    @property
    def block_count(self) -> int:
        return math.ceil(self.element_count / self.block_size)

    def validate(self) -> None:
        if self.block_size != BLOCK_SIZE:
            raise ValueError(f"unsupported Q4 block size: {self.block_size}")
        if len(self.packed) != self.block_count * PACKED_BYTES_PER_BLOCK:
            raise ValueError("Q4 packed weight length does not match tensor shape")
        if len(self.scales) != self.block_count * 2:
            raise ValueError("Q4 scale length does not match tensor shape")

    @classmethod
    def from_bf16(
        cls, data: bytes, shape: tuple[int, ...], *, prefer_numpy: bool = True
    ) -> "Q4BlockTensor":
        expected_bytes = math.prod(shape) * 2
        if len(data) != expected_bytes:
            raise ValueError(
                f"BF16 payload has {len(data)} bytes; expected {expected_bytes}"
            )
        if prefer_numpy and np is not None:
            tensor = _quantize_numpy(data, shape)
        else:
            tensor = _quantize_python(data, shape)
        tensor.validate()
        return tensor

    @classmethod
    def from_container(cls, container: ColiTensorFile, name: str) -> "Q4BlockTensor":
        quantization = container.metadata["quantization"]
        if quantization["scheme"] != "q4_symmetric":
            raise ValueError(f"unsupported quantization scheme: {quantization['scheme']}")
        tensor_metadata = quantization["tensors"][name]
        tensor = cls(
            shape=tuple(tensor_metadata["shape"]),
            packed=container.read(f"{name}.qweight"),
            scales=container.read(f"{name}.scales"),
            block_size=int(quantization["block_size"]),
        )
        tensor.validate()
        return tensor

    def payloads(self, name: str) -> tuple[TensorPayload, TensorPayload]:
        self.validate()
        return (
            TensorPayload(
                f"{name}.qweight",
                "U8",
                (self.block_count, PACKED_BYTES_PER_BLOCK),
                self.packed,
            ),
            TensorPayload(
                f"{name}.scales", "F16", (self.block_count,), self.scales
            ),
        )

    def dequantize(self, *, prefer_numpy: bool = True) -> list[float]:
        self.validate()
        if prefer_numpy and np is not None:
            return _dequantize_numpy(self).tolist()
        values: list[float] = []
        for block in range(self.block_count):
            scale = struct.unpack_from("<e", self.scales, block * 2)[0]
            offset = block * PACKED_BYTES_PER_BLOCK
            for packed_value in self.packed[offset : offset + PACKED_BYTES_PER_BLOCK]:
                values.append(((packed_value & 0x0F) - 8) * scale)
                values.append(((packed_value >> 4) - 8) * scale)
        return values[: self.element_count]

    def row(self, index: int, *, prefer_numpy: bool = True) -> list[float]:
        if len(self.shape) != 2:
            raise ValueError(f"row lookup requires rank-2 tensor, got {self.shape}")
        rows, columns = self.shape
        if index < 0 or index >= rows:
            raise IndexError(f"row index {index} outside tensor with {rows} rows")
        start = index * columns
        end = start + columns
        first_block = start // self.block_size
        last_block = (end - 1) // self.block_size
        block_count = last_block - first_block + 1
        packed_start = first_block * PACKED_BYTES_PER_BLOCK
        packed_end = (last_block + 1) * PACKED_BYTES_PER_BLOCK
        scale_start = first_block * 2
        scale_end = (last_block + 1) * 2
        window = Q4BlockTensor(
            shape=(block_count * self.block_size,),
            packed=self.packed[packed_start:packed_end],
            scales=self.scales[scale_start:scale_end],
        ).dequantize(prefer_numpy=prefer_numpy)
        offset = start - first_block * self.block_size
        return window[offset : offset + columns]

    def matvec_chunked(
        self,
        vector: list[float],
        *,
        rows_per_chunk: int = 4096,
        prefer_numpy: bool = True,
    ) -> list[float]:
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")
        return self.matvec(vector, prefer_numpy=prefer_numpy)

    def matvec(self, vector: list[float], *, prefer_numpy: bool = True) -> list[float]:
        if len(self.shape) != 2:
            raise ValueError(f"matvec requires a rank-2 tensor, got {self.shape}")
        rows, columns = self.shape
        if len(vector) != columns:
            raise ValueError(f"expected vector width {columns}, got {len(vector)}")
        if prefer_numpy:
            from .cuda import active_cuda

            accelerator = active_cuda()
            if accelerator is not None:
                return accelerator.q4_matvec(self, vector)
            if np is not None:
                from .native import active_native

                backend = active_native()
                if backend is not None:
                    return backend.q4_matvec(self, vector)
        if prefer_numpy and np is not None:
            matrix = _dequantize_numpy(self).reshape(rows, columns)
            return (matrix @ np.asarray(vector, dtype=np.float32)).tolist()
        values = self.dequantize(prefer_numpy=False)
        output: list[float] = []
        for row in range(rows):
            offset = row * columns
            output.append(
                sum(values[offset + column] * value for column, value in enumerate(vector))
            )
        return output

    def metadata(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "elements": self.element_count}


def _quantize_numpy(data: bytes, shape: tuple[int, ...]) -> Q4BlockTensor:
    bf16 = np.frombuffer(data, dtype="<u2")
    bits = bf16.astype(np.uint32) << 16
    values = bits.view(np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("BF16 tensor contains non-finite values")
    block_count = math.ceil(values.size / BLOCK_SIZE)
    padded = np.zeros(block_count * BLOCK_SIZE, dtype=np.float32)
    padded[: values.size] = values
    blocks = padded.reshape(block_count, BLOCK_SIZE)
    maximum = np.max(np.abs(blocks), axis=1)
    scales = np.where(maximum == 0, 1.0, maximum / 7.0).astype(np.float32)
    if np.any(scales > np.finfo(np.float16).max):
        raise ValueError("Q4 scale exceeds the FP16 storage range")
    quantized = np.rint(blocks / scales[:, None]).clip(-8, 7).astype(np.int8)
    shifted = (quantized.astype(np.int16) + 8).astype(np.uint8)
    packed = shifted[:, 0::2] | (shifted[:, 1::2] << 4)
    return Q4BlockTensor(
        shape=shape,
        packed=packed.tobytes(order="C"),
        scales=scales.astype("<f2").tobytes(order="C"),
    )


def _quantize_python(data: bytes, shape: tuple[int, ...]) -> Q4BlockTensor:
    values = [
        struct.unpack("<f", struct.pack("<I", bits << 16))[0]
        for (bits,) in struct.iter_unpack("<H", data)
    ]
    packed = bytearray()
    scales = bytearray()
    for start in range(0, len(values), BLOCK_SIZE):
        block = values[start : start + BLOCK_SIZE]
        block.extend([0.0] * (BLOCK_SIZE - len(block)))
        maximum = max(abs(value) for value in block)
        scale = maximum / 7.0 if maximum else 1.0
        scales.extend(struct.pack("<e", scale))
        quantized = [max(-8, min(7, round(value / scale))) + 8 for value in block]
        for index in range(0, BLOCK_SIZE, 2):
            packed.append(quantized[index] | (quantized[index + 1] << 4))
    return Q4BlockTensor(shape=shape, packed=bytes(packed), scales=bytes(scales))


def _dequantize_numpy(tensor: Q4BlockTensor):
    packed = np.frombuffer(tensor.packed, dtype=np.uint8).reshape(
        tensor.block_count, PACKED_BYTES_PER_BLOCK
    )
    values = np.empty((tensor.block_count, BLOCK_SIZE), dtype=np.int8)
    values[:, 0::2] = (packed & 0x0F).astype(np.int8) - 8
    values[:, 1::2] = (packed >> 4).astype(np.int8) - 8
    scales = np.frombuffer(tensor.scales, dtype="<f2").astype(np.float32)
    return (values.astype(np.float32) * scales[:, None]).reshape(-1)[
        : tensor.element_count
    ]
