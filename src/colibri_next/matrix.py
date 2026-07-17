from __future__ import annotations

from typing import TypeAlias

from .bf16 import BF16Tensor
from .q4 import BLOCK_SIZE, Q4BlockTensor
from .tensor_container import ColiTensorFile, TensorPayload


MatrixTensor: TypeAlias = BF16Tensor | Q4BlockTensor


def load_matrix(container: ColiTensorFile, name: str) -> MatrixTensor:
    info = container.tensors.get(name)
    if info is not None:
        if info.dtype != "BF16":
            raise ValueError(f"matrix {name} must be BF16 or Q4, got {info.dtype}")
        return BF16Tensor.from_container(container, name)
    if f"{name}.qweight" in container.tensors:
        return Q4BlockTensor.from_container(container, name)
    raise KeyError(f"matrix {name} is missing from {container.path}")


def encode_matrix(
    name: str,
    tensor: tuple[str, tuple[int, ...], bytes],
    quantization: str,
) -> tuple[list[TensorPayload], dict[str, object] | None]:
    dtype, shape, data = tensor
    if quantization == "bf16":
        return [TensorPayload(name, dtype, shape, data)], None
    if quantization != "q4":
        raise ValueError(f"unsupported matrix quantization: {quantization}")
    if dtype != "BF16":
        raise ValueError(f"Q4 conversion requires BF16 matrix {name}, got {dtype}")
    encoded = Q4BlockTensor.from_bf16(data, shape)
    return list(encoded.payloads(name)), encoded.metadata()


def quantization_metadata(
    tensors: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "scheme": "q4_symmetric",
        "block_size": BLOCK_SIZE,
        "tensors": tensors,
    }
