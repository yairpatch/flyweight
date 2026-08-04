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
