"""NumPy reference for the mmproj `qwen3vl_merger` vision tower.

The oracle the CUDA tower (native/src/v2_vision.inc) is checked against:
the same GGUF, the same patch order, f32 arithmetic throughout. It follows
llama.cpp's tools/mtmd/models/qwen3vl.cpp and transformers' Qwen3-VL vision
model, which agree on every step below:

    patchify (window-major, channel-major within a patch)
    conv over the temporal pair == (w0 + w1) @ patch + bias
    + learned position table, bilinear (corners aligned) to the patch grid
    27 x [ LN -> qkv -> 2D rope on q,k -> full attention -> out -> +res
           LN -> up -> GELU(tanh) -> down -> +res ]
    post LN -> view 4 patches as one row -> mm.0 -> GELU -> mm.2

Usage as a module: ``encode(mmproj_path, pixels_hwc_normalized)`` returns
``[tokens, projection_dim]``; deepstack layers, when the mmproj has them,
are appended as extra column blocks the way the runtime lays them out.
"""
from __future__ import annotations

import mmap
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_GGML_SIZES = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34)}


@dataclass
class Gguf:
    kv: dict[str, object]
    tensors: dict[str, tuple[list[int], int, int]]  # name -> (dims, type, offset)
    data: mmap.mmap
    data_start: int

    def array(self, name: str) -> np.ndarray:
        dims, ggml_type, offset = self.tensors[name]
        count = int(np.prod(dims))
        start = self.data_start + offset
        if ggml_type == 0:
            values = np.frombuffer(self.data, dtype="<f4", count=count, offset=start)
        elif ggml_type == 1:
            values = np.frombuffer(self.data, dtype="<f2", count=count, offset=start).astype(np.float32)
        elif ggml_type == 30:
            bits = np.frombuffer(self.data, dtype="<u2", count=count, offset=start).astype(np.uint32) << 16
            values = bits.view(np.float32)
        elif ggml_type == 8:
            blocks = count // 32
            raw = np.frombuffer(self.data, dtype=np.uint8, count=blocks * 34, offset=start).reshape(blocks, 34)
            scales = raw[:, :2].copy().view("<f2").astype(np.float32).reshape(blocks, 1)
            quants = raw[:, 2:].view(np.int8).astype(np.float32)
            values = (quants * scales).reshape(-1)
        else:
            raise ValueError(f"{name}: unsupported ggml type {ggml_type}")
        # GGUF dims are innermost-first; numpy wants row-major outermost-first.
        return np.array(values, dtype=np.float32).reshape(list(reversed(dims)))


def read_gguf(path: str | Path) -> Gguf:
    handle = open(path, "rb")
    data = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    position = 0

    def take(fmt: str):
        nonlocal position
        size = struct.calcsize(fmt)
        values = struct.unpack_from(fmt, data, position)
        position += size
        return values

    def string() -> str:
        (length,) = take("<Q")
        nonlocal position
        value = data[position:position + length].decode("utf-8", errors="replace")
        position += length
        return value

    scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}

    def value(kind: int):
        if kind == 8:
            return string()
        if kind == 9:
            (element,) = take("<I")
            (count,) = take("<Q")
            return [value(element) for _ in range(count)]
        return take(scalar[kind])[0]

    magic, version, tensor_count, kv_count = take("<IIQQ")
    if magic != 0x46554747:
        raise ValueError("not a GGUF file")
    kv: dict[str, object] = {}
    for _ in range(kv_count):
        key = string()
        (kind,) = take("<I")
        kv[key] = value(kind)
    tensors: dict[str, tuple[list[int], int, int]] = {}
    for _ in range(tensor_count):
        name = string()
        (rank,) = take("<I")
        dims = list(take(f"<{rank}Q"))
        ggml_type, offset = take("<IQ")
        tensors[name] = (dims, ggml_type, offset)
    alignment = int(kv.get("general.alignment", 32))
    data_start = (position + alignment - 1) // alignment * alignment
    return Gguf(kv, tensors, data, data_start)


@dataclass
class VisionConfig:
    patch: int
    merge: int
    width: int
    heads: int
    blocks: int
    ffn: int
    projection: int
    grid_side: int
    eps: float
    mean: np.ndarray
    std: np.ndarray
    deepstack: list[int] = field(default_factory=list)

    @staticmethod
    def from_gguf(gguf: Gguf) -> "VisionConfig":
        kv = gguf.kv
        if kv.get("clip.projector_type") != "qwen3vl_merger":
            raise ValueError("mmproj is not a qwen3vl_merger projector")
        flags = kv.get("clip.vision.is_deepstack_layers", [])
        return VisionConfig(
            patch=int(kv["clip.vision.patch_size"]),
            merge=int(kv.get("clip.vision.spatial_merge_size", 2)),
            width=int(kv["clip.vision.embedding_length"]),
            heads=int(kv["clip.vision.attention.head_count"]),
            blocks=int(kv["clip.vision.block_count"]),
            ffn=int(kv["clip.vision.feed_forward_length"]),
            projection=int(kv["clip.vision.projection_dim"]),
            grid_side=int(kv["clip.vision.image_size"]) // int(kv["clip.vision.patch_size"]),
            eps=float(kv.get("clip.vision.attention.layer_norm_epsilon", 1e-6)),
            mean=np.array(kv.get("clip.vision.image_mean", [0.5] * 3), dtype=np.float32),
            std=np.array(kv.get("clip.vision.image_std", [0.5] * 3), dtype=np.float32),
            deepstack=[index for index, flag in enumerate(flags) if flag],
        )


def smart_resize(config: VisionConfig, width: int, height: int,
                 min_tokens: int = 1, max_tokens: int = 4096) -> tuple[int, int]:
    """Pixel (width, height) the runtime resizes to; mirrors vision_smart_resize."""
    side = config.patch * config.merge
    min_pixels, max_pixels = min_tokens * side * side, max_tokens * side * side
    h = max(side, round(height / side) * side)
    w = max(side, round(width / side) * side)
    if h * w > max_pixels:
        beta = np.sqrt(height * width / max_pixels)
        h = max(side, int(np.floor(height / beta / side)) * side)
        w = max(side, int(np.floor(width / beta / side)) * side)
    elif h * w < min_pixels:
        beta = np.sqrt(min_pixels / (height * width))
        h = int(np.ceil(height * beta / side)) * side
        w = int(np.ceil(width * beta / side)) * side
    return int(w), int(h)


def window_major_order(grid_h: int, grid_w: int, merge: int) -> tuple[np.ndarray, np.ndarray]:
    """(rows, cols) of every patch in the order the tower consumes them."""
    rows, cols = [], []
    for wy in range(0, grid_h, merge):
        for wx in range(0, grid_w, merge):
            for dy in range(merge):
                for dx in range(merge):
                    rows.append(wy + dy)
                    cols.append(wx + dx)
    return np.array(rows), np.array(cols)


def patchify(config: VisionConfig, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width, _ = pixels.shape
    p = config.patch
    grid_h, grid_w = height // p, width // p
    rows, cols = window_major_order(grid_h, grid_w, config.merge)
    # [gh, gw, c, p, p]
    tiles = pixels.reshape(grid_h, p, grid_w, p, 3).transpose(0, 2, 4, 1, 3)
    patches = tiles[rows, cols].reshape(len(rows), 3 * p * p)
    return patches.astype(np.float32), rows, cols


def position_embeddings(config: VisionConfig, table: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    side = config.grid_side
    grid = table.reshape(side, side, config.width)

    def axis(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coordinates = np.linspace(0, side - 1, n) if n > 1 else np.zeros(1)
        lo = np.floor(coordinates).astype(int)
        hi = np.minimum(lo + 1, side - 1)
        return lo, hi, (coordinates - lo).astype(np.float32)

    y0, y1, wy = axis(grid_h)
    x0, x1, wx = axis(grid_w)
    top = grid[y0][:, x0] * (1 - wx)[None, :, None] + grid[y0][:, x1] * wx[None, :, None]
    bottom = grid[y1][:, x0] * (1 - wx)[None, :, None] + grid[y1][:, x1] * wx[None, :, None]
    resampled = top * (1 - wy)[:, None, None] + bottom * wy[:, None, None]  # [gh, gw, E]
    rows, cols = window_major_order(grid_h, grid_w, config.merge)
    return resampled[rows, cols].astype(np.float32)


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * weight + bias


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def rope_2d(x: np.ndarray, rows: np.ndarray, cols: np.ndarray, head_dim: int, theta: float = 10000.0) -> np.ndarray:
    """x: [n, heads, head_dim]. Pairs (j, j+half); first quarter of j by row,
    second by column, each with its own ramp over half."""
    half, quarter = head_dim // 2, head_dim // 4
    ramp = theta ** (-2.0 * np.arange(quarter) / half)           # [quarter]
    angles = np.concatenate(
        [rows[:, None] * ramp[None, :], cols[:, None] * ramp[None, :]], axis=1
    ).astype(np.float32)                                            # [n, half]
    cos, sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    first, second = x[..., :half], x[..., half:]
    return np.concatenate([first * cos - second * sin, second * cos + first * sin], axis=-1)


def attention(qkv: np.ndarray, heads: int, head_dim: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    n = qkv.shape[0]
    q, k, v = (qkv[:, i * heads * head_dim:(i + 1) * heads * head_dim].reshape(n, heads, head_dim) for i in range(3))
    q, k = rope_2d(q, rows, cols, head_dim), rope_2d(k, rows, cols, head_dim)
    scores = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(head_dim)
    scores -= scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    out = np.einsum("hqk,khd->qhd", probabilities, v)
    return out.reshape(n, heads * head_dim)


def encode(mmproj: str | Path, pixels: np.ndarray) -> np.ndarray:
    """pixels: [H, W, 3] f32, already normalized, sides multiples of patch*merge."""
    gguf = read_gguf(mmproj)
    config = VisionConfig.from_gguf(gguf)
    height, width, _ = pixels.shape
    grid_h, grid_w = height // config.patch, width // config.patch
    patches, rows, cols = patchify(config, pixels)
    head_dim = config.width // config.heads

    weight = gguf.array("v.patch_embd.weight").reshape(config.width, -1)
    if "v.patch_embd.weight.1" in gguf.tensors:
        weight = weight + gguf.array("v.patch_embd.weight.1").reshape(config.width, -1)
    hidden = patches @ weight.T + gguf.array("v.patch_embd.bias")
    hidden = hidden + position_embeddings(config, gguf.array("v.position_embd.weight"), grid_h, grid_w)

    merge = config.merge * config.merge
    tokens = len(rows) // merge
    deepstack_outputs: list[np.ndarray] = []
    for index in range(config.blocks):
        t = lambda name: gguf.array(f"v.blk.{index}.{name}")  # noqa: E731
        normed = layer_norm(hidden, t("ln1.weight"), t("ln1.bias"), config.eps)
        qkv = normed @ t("attn_qkv.weight").T + t("attn_qkv.bias")
        attended = attention(qkv, config.heads, head_dim, rows, cols)
        hidden = hidden + attended @ t("attn_out.weight").T + t("attn_out.bias")
        normed = layer_norm(hidden, t("ln2.weight"), t("ln2.bias"), config.eps)
        up = gelu(normed @ t("ffn_up.weight").T + t("ffn_up.bias"))
        hidden = hidden + up @ t("ffn_down.weight").T + t("ffn_down.bias")
        if index in config.deepstack:
            d = lambda name: gguf.array(f"v.deepstack.{index}.{name}")  # noqa: E731
            merged_rows = hidden.reshape(tokens, merge * config.width)
            feature = layer_norm(merged_rows, d("norm.weight"), d("norm.bias"), config.eps)
            feature = gelu(feature @ d("fc1.weight").T + d("fc1.bias"))
            deepstack_outputs.append(feature @ d("fc2.weight").T + d("fc2.bias"))

    hidden = layer_norm(hidden, gguf.array("v.post_ln.weight"), gguf.array("v.post_ln.bias"), config.eps)
    merged_rows = hidden.reshape(tokens, merge * config.width)
    projected = gelu(merged_rows @ gguf.array("mm.0.weight").T + gguf.array("mm.0.bias"))
    projected = projected @ gguf.array("mm.2.weight").T + gguf.array("mm.2.bias")
    return np.concatenate([projected] + deepstack_outputs, axis=1).astype(np.float32)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: qwen_vision_reference.py MMPROJ.gguf IMAGE_SIDE", file=sys.stderr)
        return 2
    side = int(argv[2])
    rng = np.random.default_rng(20260903)
    pixels = rng.uniform(-1, 1, size=(side, side, 3)).astype(np.float32)
    out = encode(argv[1], pixels)
    print(out.shape, float(out.mean()), float(out.std()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
