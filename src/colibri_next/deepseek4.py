"""DeepSeek-V4 building blocks exposed for parity checking.

These call the native CPU kernels directly, so a component can be compared
against the reference implementation before there is a whole forward pass to
run. Arrays are plain float32 numpy, laid out the way the GGUF stores them.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from colibri_next.v2 import V2Error, _library


def matvec(model, name: str, input_vector: np.ndarray, output_size: int) -> np.ndarray:
    """Multiply a checkpoint tensor by a vector, whatever type it is stored as."""
    input_vector = np.ascontiguousarray(input_vector, dtype=np.float32)
    output = np.zeros(output_size, dtype=np.float32)
    library = _library()
    status = library.colibri_v2_matvec(
        model._handle,
        name.encode(),
        _pointer(input_vector),
        input_vector.size,
        _pointer(output),
        output_size,
    )
    if status:
        raise V2Error((library.colibri_v2_last_error() or b"matvec failed").decode(errors="replace"))
    return output


def grouped_matvec(
    model, name: str, input_vector: np.ndarray, inputs: int, rank: int, groups: int
) -> np.ndarray:
    """The grouped half of the output projection.

    The input is cut into `groups` chunks of `inputs`, and chunk g goes through
    the g-th slice of the tensor's output rows rather than the whole matrix.
    """
    input_vector = np.ascontiguousarray(input_vector, dtype=np.float32)
    output = np.zeros(rank * groups, dtype=np.float32)
    library = _library()
    status = library.colibri_v2_grouped_matvec(
        model._handle, name.encode(), _pointer(input_vector), inputs,
        _pointer(output), rank, groups,
    )
    if status:
        raise V2Error((library.colibri_v2_last_error() or b"grouped matvec failed").decode(errors="replace"))
    return output


def rope(
    values: np.ndarray,
    position: int,
    rope_dim: int,
    *,
    freq_base: float,
    freq_scale: float = 1.0,
    inverse: bool = False,
) -> np.ndarray:
    """Rotate the trailing `rope_dim` of each row at `position`.

    Which frequency base and scaling apply depends on the layer kind, so both
    are the caller's to supply.
    """
    values = np.ascontiguousarray(values, dtype=np.float32).copy()
    rows = 1 if values.ndim == 1 else int(np.prod(values.shape[:-1]))
    stride = values.shape[-1]
    library = _library()
    status = library.colibri_v2_deepseek4_rope(
        _pointer(values), stride, rope_dim, rows, position,
        freq_base, freq_scale, 1 if inverse else 0,
    )
    if status:
        raise V2Error((library.colibri_v2_last_error() or b"rope failed").decode(errors="replace"))
    return values


def attention(
    queries: np.ndarray,
    latents: np.ndarray,
    sinks: np.ndarray | None = None,
    *,
    scale: float | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Attention over the shared KV latent, one sink logit per head.

    `queries` is [heads, head_dim] and `latents` is [positions, head_dim] --
    keys and values are the same tensor. `scale` defaults to 1/sqrt(head_dim).
    """
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    latents = np.ascontiguousarray(latents, dtype=np.float32)
    heads, head_dim = queries.shape
    positions = latents.shape[0]
    if latents.shape[1] != head_dim:
        raise ValueError("queries and latents must share a width")
    output = np.zeros((heads, head_dim), dtype=np.float32)
    mask_array = None if mask is None else np.ascontiguousarray(mask, dtype=np.uint8)
    library = _library()
    status = library.colibri_v2_deepseek4_attention(
        _pointer(queries), _pointer(latents),
        _pointer(None if sinks is None else np.ascontiguousarray(sinks, dtype=np.float32)),
        None if mask_array is None else mask_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        heads, head_dim, positions,
        float(head_dim) ** -0.5 if scale is None else scale,
        _pointer(output),
    )
    if status:
        raise V2Error((library.colibri_v2_last_error() or b"attention failed").decode(errors="replace"))
    return output


def rms_norm(
    values: np.ndarray, weight: np.ndarray | None = None, *, epsilon: float = 1e-6
) -> np.ndarray:
    """RMS-normalize each row, optionally applying a learned gain.

    A 2-D input is normalized row by row, which is how the per-head query norm
    works: each head's slice is normalized on its own.
    """
    values = np.ascontiguousarray(values, dtype=np.float32)
    rows = 1 if values.ndim == 1 else values.shape[0]
    size = values.shape[-1]
    output = np.zeros_like(values)
    library = _library()
    status = library.colibri_v2_deepseek4_rms_norm(
        _pointer(values),
        _pointer(None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)),
        size,
        rows,
        epsilon,
        _pointer(output),
    )
    if status:
        raise V2Error((library.colibri_v2_last_error() or b"rms norm failed").decode(errors="replace"))
    return output


@dataclass(frozen=True)
class HyperConnection:
    """One block's hyper-connection weights and the vectors derived from them."""

    mixes: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    comb: np.ndarray
    collapsed: np.ndarray
    combined: np.ndarray | None


def _pointer(array: np.ndarray | None):
    if array is None:
        return None
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def hyper_connection(
    streams: np.ndarray,
    fn: np.ndarray,
    scale: np.ndarray,
    base: np.ndarray,
    *,
    sinkhorn_iterations: int = 20,
    rms_epsilon: float = 1e-6,
    hc_epsilon: float = 1e-6,
    block: np.ndarray | None = None,
) -> HyperConnection:
    """Run one hyper-connection step.

    `streams` is [hc, n_embd] and `fn` is [(2+hc)*hc, hc*n_embd] -- the GGUF
    stores the mixer output-major, which is the layout the kernel wants.
    """
    streams = np.ascontiguousarray(streams, dtype=np.float32)
    hc, n_embd = streams.shape
    mix_dim = (2 + hc) * hc
    fn = np.ascontiguousarray(fn, dtype=np.float32)
    if fn.shape != (mix_dim, hc * n_embd):
        raise ValueError(f"mixer must be {(mix_dim, hc * n_embd)}, got {fn.shape}")

    mixes = np.zeros(mix_dim, dtype=np.float32)
    pre = np.zeros(hc, dtype=np.float32)
    post = np.zeros(hc, dtype=np.float32)
    comb = np.zeros(hc * hc, dtype=np.float32)
    collapsed = np.zeros(n_embd, dtype=np.float32)
    combined = np.zeros((hc, n_embd), dtype=np.float32) if block is not None else None
    block_array = None if block is None else np.ascontiguousarray(block, dtype=np.float32)

    library = _library()
    status = library.colibri_v2_deepseek4_hyper_connection(
        _pointer(streams),
        _pointer(fn),
        _pointer(np.ascontiguousarray(scale, dtype=np.float32)),
        _pointer(np.ascontiguousarray(base, dtype=np.float32)),
        n_embd,
        hc,
        sinkhorn_iterations,
        rms_epsilon,
        hc_epsilon,
        _pointer(block_array),
        _pointer(mixes),
        _pointer(pre),
        _pointer(post),
        _pointer(comb),
        _pointer(collapsed),
        _pointer(combined),
    )
    if status:
        message = library.colibri_v2_last_error() or b"native deepseek4 error"
        raise V2Error(message.decode(errors="replace"))
    return HyperConnection(
        mixes=mixes,
        pre=pre,
        post=post,
        comb=comb.reshape(hc, hc),
        collapsed=collapsed,
        combined=combined,
    )
