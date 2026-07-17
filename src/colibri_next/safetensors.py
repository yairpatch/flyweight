from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_HEADER_LENGTH = struct.Struct("<Q")
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True, slots=True)
class SafeTensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int

    @property
    def byte_size(self) -> int:
        return self.data_end - self.data_start

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)


class SafeTensorFile:
    """Minimal, dependency-free safetensors metadata and byte-slice reader."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        with self.path.open("rb") as handle:
            header_length_bytes = handle.read(_HEADER_LENGTH.size)
            if len(header_length_bytes) != _HEADER_LENGTH.size:
                raise ValueError(f"truncated safetensors prefix: {self.path}")
            header_length = _HEADER_LENGTH.unpack(header_length_bytes)[0]
            if header_length == 0 or header_length > 100 * 1024**2:
                raise ValueError(f"invalid safetensors header length: {header_length}")
            header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                raise ValueError(f"truncated safetensors header: {self.path}")

        header = json.loads(header_bytes)
        self.metadata = dict(header.pop("__metadata__", {}))
        data_base = _HEADER_LENGTH.size + header_length
        file_size = self.path.stat().st_size
        tensors: dict[str, SafeTensorInfo] = {}
        for name, value in header.items():
            dtype = value["dtype"]
            shape = tuple(int(dimension) for dimension in value["shape"])
            offsets = value["data_offsets"]
            if dtype not in _DTYPE_BYTES:
                raise ValueError(f"unsupported safetensors dtype {dtype!r} for {name}")
            if len(offsets) != 2 or offsets[0] < 0 or offsets[1] < offsets[0]:
                raise ValueError(f"invalid data offsets for {name}")
            expected_bytes = math.prod(shape) * _DTYPE_BYTES[dtype]
            if offsets[1] - offsets[0] != expected_bytes:
                raise ValueError(
                    f"tensor {name} declares {expected_bytes} bytes but spans "
                    f"{offsets[1] - offsets[0]}"
                )
            info = SafeTensorInfo(
                name=name,
                dtype=dtype,
                shape=shape,
                data_start=data_base + int(offsets[0]),
                data_end=data_base + int(offsets[1]),
            )
            if info.data_end > file_size:
                raise ValueError(f"tensor {name} extends beyond {self.path}")
            tensors[name] = info
        self.tensors = tensors

    def tensor(self, name: str) -> SafeTensorInfo:
        try:
            return self.tensors[name]
        except KeyError as error:
            raise KeyError(f"tensor {name!r} is not in {self.path}") from error

    def read(self, name: str) -> bytes:
        info = self.tensor(name)
        return self._read_range(info.data_start, info.byte_size)

    def read_axis0_slice(self, name: str, index: int) -> tuple[bytes, tuple[int, ...]]:
        info = self.tensor(name)
        if not info.shape:
            raise ValueError(f"cannot slice scalar tensor {name}")
        if not 0 <= index < info.shape[0]:
            raise IndexError(f"axis-0 index {index} outside [0, {info.shape[0]})")
        slice_bytes = info.byte_size // info.shape[0]
        start = info.data_start + index * slice_bytes
        return self._read_range(start, slice_bytes), info.shape[1:]

    def _read_range(self, start: int, byte_size: int) -> bytes:
        with self.path.open("rb") as handle:
            handle.seek(start)
            data = handle.read(byte_size)
        if len(data) != byte_size:
            raise ValueError(f"truncated tensor data in {self.path}")
        return data
