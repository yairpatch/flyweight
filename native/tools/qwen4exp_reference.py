"""NumPy reference for the qwen4exp (Qwen3.8-Flash-Next) novel blocks.

qwen4exp reuses the qwen35 gated DeltaNet (deltanet_reference.py), attention
gate, and MoE math wholesale; this file covers only what is NEW:

    hyper-connections   the residual stream is hc_count=4 copies of the hidden
                        vector; every block boundary mixes the streams down to
                        one block input and injects the block output back with
                        per-stream gates. Replaces the rms/add bookends.
    PLE                 hashed bigram/trigram embeddings gated against the
                        stream state, plus a dilated depthwise conv, added into
                        the stream at one layer (blk.1).

Semantics extracted from transformers `modular_qwen4_exp.py` (merged PR #48337);
see plans/qwen4exp-semantics.md for the derivation and the conversion-side
weight baking (+1 on every norm weight -- the arrays passed here are the BAKED
weights, as loaded from GGUF). Pinned against the torch modules by
qwen4exp_reference_check.py.

Layouts, all float32 unless integer:

  hyper        [rows][hc * hidden]        the residual streams, flattened
  hc weights   norm [hc*hidden], down [hc*hidden][low_rank] (x @ down),
               up [low_rank][hc*hidden], inject [hc*hidden][hc]
  ple table    [total_vocab][head_dim]    head_dim = 160 in the real model
  tokens       int64 row vector           per sequence, eos-segmented
"""
from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def silu(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


def grouped_rms(x: np.ndarray, weight: np.ndarray, group: int,
                epsilon: float = 1e-6) -> np.ndarray:
    """RMS-normalize each `group`-wide chunk of the last axis independently.

    `weight` spans the full (ungrouped) width and is the BAKED form: the HF
    module computes `normed * (1 + w)`; the GGUF stores `1 + w`, so this
    multiplies by `weight` directly. fp32 accumulation like the runtime.
    """
    shaped = x.reshape(*x.shape[:-1], -1, group).astype(np.float32)
    scale = np.sqrt(np.mean(shaped * shaped, axis=-1, keepdims=True) + epsilon)
    return (shaped / scale).reshape(x.shape) * weight


def hc_mix(hyper: np.ndarray, norm: np.ndarray, down: np.ndarray,
           up: np.ndarray, inject: np.ndarray | None,
           hc: int, hidden: int, epsilon: float = 1e-6):
    """One hyper-connection boundary. Returns (block_input, inject_weights).

    block_input   [rows][hidden]   what the attention/MoE block consumes
    inject_weights [rows][hc]     None when `inject` is None (head collapse)

    Note both /hc scalings happen BEFORE the nonlinearity, and the inject
    gates are doubled -- `2*sigmoid(...)` so an untrained (zero) inject weight
    passes the block output through at weight 1 per stream.
    """
    rows = hyper.shape[0]
    normed = grouped_rms(hyper, norm, hidden, epsilon)
    low = silu((normed @ down) / hc)
    mixw = sigmoid(low @ up)
    streams = (mixw * normed).reshape(rows, hc, hidden)
    block_input = streams.mean(axis=1)
    if inject is None:
        return block_input, None
    inject_weights = 2.0 * sigmoid((normed @ inject) / hc)
    return block_input, inject_weights


def hc_inject(hyper: np.ndarray, block_output: np.ndarray,
              inject_weights: np.ndarray, hc: int, hidden: int) -> np.ndarray:
    """streams' = streams + w[s] * block_output. Residual base is PRE-norm."""
    rows = hyper.shape[0]
    add = inject_weights[:, :, None] * block_output[:, None, :]
    return hyper + add.reshape(rows, hc * hidden)


def hc_init(embedding: np.ndarray, hc: int) -> np.ndarray:
    """Stream init: the embedding repeated hc times along the feature axis."""
    return np.tile(embedding, (1, hc))


# ---------------------------------------------------------------------------
# PLE
# ---------------------------------------------------------------------------

def ngram_rows(tokens: np.ndarray, multipliers: np.ndarray,
               vocab_sizes: np.ndarray, offsets: np.ndarray,
               heads_per_ngram: int, ngram_size: int,
               eos_token_id: int) -> np.ndarray:
    """Row ids into the ngram table: [len(tokens)][(ngram_size-1)*heads_per_ngram].

    tokens is ONE sequence (history included -- caller passes everything since
    the sequence start). Segmentation: an n-gram never crosses an eos; positions
    whose lookback would cross the previous eos (or the sequence start) read
    eos itself in that slot. All arithmetic wraps in uint64 exactly like
    torch's int64 (products stay < 2**63 because multipliers are < 2**63/vocab
    and odd; xor of sign-bit-clear values keeps the sign bit clear, so
    torch.remainder == plain unsigned mod here).
    """
    tokens = tokens.astype(np.int64)
    n = tokens.shape[0]
    positions = np.arange(n)
    eos_pos = np.where(tokens == eos_token_id, positions, -1)
    prev_eos_incl = np.maximum.accumulate(eos_pos)
    prev_eos = np.concatenate([[-1], prev_eos_incl[:-1]])
    pos_in_segment = positions - (prev_eos + 1)

    shifted = np.empty((ngram_size, n), dtype=np.uint64)
    for shift in range(ngram_size):
        src = np.clip(positions - shift, 0, None)
        vals = tokens[src]
        valid = (pos_in_segment >= shift) & (positions - shift >= 0)
        shifted[shift] = np.where(valid, vals, eos_token_id).astype(np.uint64)

    mults = multipliers.astype(np.uint64)
    heads = (ngram_size - 1) * heads_per_ngram
    rows = np.empty((n, heads), dtype=np.int64)
    with np.errstate(over="ignore"):
        for ngram in range(2, ngram_size + 1):
            mixed = shifted[0] * mults[0]
            for p in range(1, ngram):
                mixed = mixed ^ (shifted[p] * mults[p])
            base = (ngram - 2) * heads_per_ngram
            for h in range(heads_per_ngram):
                g = base + h
                rows[:, g] = (mixed % np.uint64(vocab_sizes[g])).astype(np.int64) \
                    + np.int64(offsets[g])
    return rows


def ple_forward(hyper: np.ndarray, embeddings: np.ndarray,
                key_w: np.ndarray, value_w: np.ndarray,
                norm_key: np.ndarray, norm_query: np.ndarray,
                norm_conv: np.ndarray, conv_w: np.ndarray,
                conv_state: np.ndarray | None,
                hc: int, hidden: int, ngram_size: int,
                epsilon: float = 1e-6):
    """PLE block: returns (delta_to_add_to_hyper, new_conv_state).

    embeddings   [rows][2560]  the 16 gathered table rows, concatenated
    key_w        [2560][hc*hidden]     value_w [2560][hidden]
    conv_w       [hc*hidden][kernel]   depthwise, DILATED by ngram_size
    conv_state   [(kernel-1)*ngram_size][hc*hidden] trailing columns, oldest first
    """
    rows = hyper.shape[0]
    wide = hc * hidden
    kernel = conv_w.shape[1]
    state_len = (kernel - 1) * ngram_size

    key = grouped_rms(embeddings @ key_w, norm_key, hidden, epsilon)
    value = embeddings @ value_w
    query = grouped_rms(hyper, norm_query, hidden, epsilon)
    k4 = key.reshape(rows, hc, hidden)
    q4 = query.reshape(rows, hc, hidden)
    gate = np.sum(k4 * q4, axis=-1) / np.sqrt(hidden)
    gate = np.sign(gate) * np.sqrt(np.maximum(np.abs(gate), 1e-6))
    gated = sigmoid(gate)[:, :, None] * value[:, None, :]
    gated = gated.reshape(rows, wide)
    gated_normed = grouped_rms(gated, norm_conv, hidden, epsilon)

    if conv_state is None:
        conv_state = np.zeros((state_len, wide), dtype=np.float32)
    padded = np.concatenate([conv_state, gated_normed], axis=0)
    conv = np.zeros((rows, wide), dtype=np.float32)
    for row in range(rows):
        pos = state_len + row
        for tap in range(kernel):
            conv[row] += conv_w[:, tap] * padded[pos - (kernel - 1 - tap) * ngram_size]
    conv = silu(conv)
    return gated + conv, padded[-state_len:].copy()
