"""Re-emits a single-file GGUF as a multi-shard split GGUF.

Large checkpoints ship as `gguf-split` output: N files named
``<prefix>-00001-of-0000N.gguf``, each a complete GGUF holding part of the
tensor payload, tied together by the ``split.no`` / ``split.count`` /
``split.tensors.count`` metadata. DeepSeek-V4-Flash's IQ3_XXS build is four
shards whose first file carries every metadata key and *zero* tensors, so that
layout is reproduced here rather than assumed away.

Splitting an already-written fixture (instead of generating shards directly)
keeps the shards byte-identical to the single-file model, which is what makes
"a split model behaves exactly like the file it came from" a real assertion.
"""
from __future__ import annotations

import struct
from pathlib import Path

GGUF_UINT16 = 2
GGUF_INT32 = 5
GGUF_UINT32 = 4
GGUF_STRING = 8


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.at = 0

    def take(self, count: int) -> bytes:
        chunk = self.data[self.at : self.at + count]
        if len(chunk) != count:
            raise ValueError("truncated GGUF")
        self.at += count
        return chunk

    def scalar(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(size))[0]

    def string(self) -> str:
        return self.take(self.scalar("<Q")).decode("utf-8")

    def skip_value(self, kind: int) -> None:
        widths = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if kind == 8:
            self.string()
        elif kind == 9:
            element = self.scalar("<I")
            for _ in range(self.scalar("<Q")):
                self.skip_value(element)
        elif kind in widths:
            self.take(widths[kind])
        else:
            raise ValueError(f"unsupported GGUF metadata type {kind}")


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, kind: int, payload: bytes) -> bytes:
    return _string(key) + struct.pack("<I", kind) + payload


def split_gguf(
    source: Path | str,
    directory: Path | str,
    *,
    shards: int = 4,
    stem: str = "model",
    metadata_only_first: bool = False,
) -> Path:
    """Write `source` as `shards` files and return the path of the first.

    ``metadata_only_first`` reproduces the DeepSeek-V4-Flash layout, where the
    opened file holds all of the metadata and none of the tensors.
    """
    if shards < 1:
        raise ValueError("a split needs at least one shard")
    blob = Path(source).read_bytes()
    reader = _Reader(blob)
    if reader.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version = reader.scalar("<I")
    tensor_count = reader.scalar("<Q")
    metadata_count = reader.scalar("<Q")

    alignment = 32
    architecture = ""
    entries: list[tuple[str, bytes]] = []
    for _ in range(metadata_count):
        start = reader.at
        key = reader.string()
        kind = reader.scalar("<I")
        value_at = reader.at
        if key == "general.alignment" and kind == GGUF_UINT32:
            alignment = struct.unpack_from("<I", blob, value_at)[0]
        if key == "general.architecture" and kind == GGUF_STRING:
            architecture = struct.unpack_from(
                f"<{struct.unpack_from('<Q', blob, value_at)[0]}s", blob, value_at + 8
            )[0].decode("utf-8")
        reader.skip_value(kind)
        # Shards are re-emitted verbatim, so an unrecognised key survives.
        if not key.startswith("split."):
            entries.append((key, blob[start : reader.at]))

    tensors: list[tuple[str, tuple[int, ...], int, int]] = []
    for _ in range(tensor_count):
        name = reader.string()
        shape = tuple(reader.scalar("<Q") for _ in range(reader.scalar("<I")))
        kind = reader.scalar("<I")
        offset = reader.scalar("<Q")
        tensors.append((name, shape, kind, offset))

    data_at = _align(reader.at, alignment)
    # The runtime derives a tensor's length from where the next one starts, so
    # the payload is carved the same way here.
    bounds = sorted({offset for _, _, _, offset in tensors} | {len(blob) - data_at})
    payloads = {}
    for name, _, _, offset in tensors:
        end = next(bound for bound in bounds if bound > offset)
        payloads[name] = blob[data_at + offset : data_at + end]

    buckets: list[list[tuple[str, tuple[int, ...], int, int]]] = [[] for _ in range(shards)]
    carriers = shards - 1 if metadata_only_first and shards > 1 else shards
    first = 1 if metadata_only_first and shards > 1 else 0
    for index, tensor in enumerate(tensors):
        buckets[first + index * carriers // max(len(tensors), 1)].append(tensor)

    paths = []
    for index, bucket in enumerate(buckets):
        keys = (
            list(entries)
            if index == 0
            else [
                (key, raw)
                for key, raw in entries
                if key in ("general.architecture", "general.alignment")
            ]
        )
        if index and not any(key == "general.architecture" for key, _ in keys) and architecture:
            keys.append(("general.architecture", _kv("general.architecture", GGUF_STRING, _string(architecture))))
        keys.append(("split.no", _kv("split.no", GGUF_UINT16, struct.pack("<H", index))))
        keys.append(("split.count", _kv("split.count", GGUF_UINT16, struct.pack("<H", shards))))
        keys.append((
            "split.tensors.count",
            _kv("split.tensors.count", GGUF_INT32, struct.pack("<i", len(tensors))),
        ))

        infos = bytearray()
        body = bytearray()
        for name, shape, kind, _ in bucket:
            infos += _string(name)
            infos += struct.pack("<I", len(shape))
            infos += b"".join(struct.pack("<Q", dimension) for dimension in shape)
            infos += struct.pack("<IQ", kind, len(body))
            body += payloads[name]
            body += b"\0" * ((-len(body)) % alignment)

        header = b"GGUF" + struct.pack("<IQQ", version, len(bucket), len(keys))
        out = bytearray(header + b"".join(raw for _, raw in keys) + bytes(infos))
        out += b"\0" * ((-len(out)) % alignment)
        path = Path(directory) / f"{stem}-{index + 1:05d}-of-{shards:05d}.gguf"
        path.write_bytes(bytes(out) + bytes(body))
        paths.append(path)
    return paths[0]
