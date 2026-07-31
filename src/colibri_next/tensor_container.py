from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_PREFIX = struct.Struct("<4sIQ")
_MAGIC = b"CLTX"
_VERSION = 1


@dataclass(frozen=True, slots=True)
class TensorPayload:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes


@dataclass(frozen=True, slots=True)
class ColiTensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int
    sha256: str

    @property
    def byte_size(self) -> int:
        return self.data_end - self.data_start


def write_coli_tensor_file(
    path: Path | str,
    payloads: Iterable[TensorPayload],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(payloads)
    offset = 0
    tensors: dict[str, dict[str, Any]] = {}
    for payload in materialized:
        if payload.name in tensors:
            raise ValueError(f"duplicate tensor payload: {payload.name}")
        next_offset = offset + len(payload.data)
        tensors[payload.name] = {
            "dtype": payload.dtype,
            "shape": list(payload.shape),
            "data_offsets": [offset, next_offset],
            "sha256": hashlib.sha256(payload.data).hexdigest(),
        }
        offset = next_offset

    header = json.dumps(
        {"metadata": metadata or {}, "tensors": tensors},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(_PREFIX.pack(_MAGIC, _VERSION, len(header)))
            handle.write(header)
            for payload in materialized:
                handle.write(payload.data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


class ColiTensorFile:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        with self.path.open("rb") as handle:
            prefix = handle.read(_PREFIX.size)
            if len(prefix) != _PREFIX.size:
                raise ValueError(f"truncated Colibrì tensor container: {self.path}")
            magic, version, header_length = _PREFIX.unpack(prefix)
            if magic != _MAGIC or version != _VERSION:
                raise ValueError(f"unsupported Colibrì tensor container: {self.path}")
            header = json.loads(handle.read(header_length))
        self.metadata = dict(header.get("metadata", {}))
        data_base = _PREFIX.size + header_length
        self.tensors = {
            name: ColiTensorInfo(
                name=name,
                dtype=value["dtype"],
                shape=tuple(value["shape"]),
                data_start=data_base + value["data_offsets"][0],
                data_end=data_base + value["data_offsets"][1],
                sha256=value["sha256"],
            )
            for name, value in header["tensors"].items()
        }

    def read(self, name: str, *, verify: bool = True) -> bytes:
        info = self.tensors[name]
        with self.path.open("rb") as handle:
            handle.seek(info.data_start)
            data = handle.read(info.byte_size)
        if len(data) != info.byte_size:
            raise ValueError(f"truncated tensor {name} in {self.path}")
        if verify and hashlib.sha256(data).hexdigest() != info.sha256:
            raise ValueError(f"checksum mismatch for tensor {name} in {self.path}")
        return data
