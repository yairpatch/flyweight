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
    QSA indexer         a 4-head scorer on the full-attention layers that masks
                        attention down to the top `budget` tokens, selected in
                        blocks of `ratio` consecutive positions. Keys are cached
                        RAW; pooling, norm and rope happen at query time (the
                        runtime caches the pooled block keys instead -- same
                        math, a block's key never changes once it completes).

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


# ---------------------------------------------------------------------------
# MTP (nextn draft block)
# ---------------------------------------------------------------------------

def mtp_input_fusion(hyper: np.ndarray, embedding: np.ndarray,
                     enorm: np.ndarray, hnorm: np.ndarray,
                     eh_proj: np.ndarray, hc: int, hidden: int,
                     epsilon: float = 1e-6) -> np.ndarray:
    """The draft block's input fusion: streams + next-token embedding -> streams.

    `hyper` is the TARGET's hyper-connection streams [rows][hc*hidden] (the
    pre-collapse state, not the collapsed hidden the LM head sees), `embedding`
    is the raw token embedding [rows][hidden] of the token the target just
    produced. Returns [rows][hc*hidden] -- stream space in, stream space out,
    which then runs the ordinary layer bookends (hc_mix/hc_inject) around a
    DENSE attention and the MoE.

    Two things here are easy to get wrong and are silent when wrong -- a bad
    fusion only costs acceptance, never text, because verify re-scores every
    draft with the target:

    - hnorm is hc*hidden wide because the reference RMS-norms the WHOLE stream
      row as one vector and only then splits it into streams. It is NOT a
      per-stream (grouped) norm -- deepseek4's MTP is, qwen4exp's is not, and
      the two differ only in the denominator. Hence `group=hc*hidden` below.
    - the concat is EMBEDDING FIRST, hidden second, and eh_proj is applied per
      stream with the embedding term broadcast across all hc of them. The
      checkpoint's own layout is two separate projections (mtp.fc_embedding,
      mtp.fc_hidden) that conversion fuses as A*e + B*h == [A|B] @ concat(e, h);
      that identity is what pins the order.

    ref: llama.cpp qwen4exp graph_mtp (refs/qwen4_exp/llamacpp_qwen4exp.cpp),
    itself from sglang qwen4_exp_mtp.py. There is no transformers module to
    check against: upstream drops the weights (`_keys_to_ignore_on_load_
    unexpected = [r"^mtp.*"]`).
    """
    rows = hyper.shape[0]
    # Whole-row RMS, then split into streams -- not grouped_rms(..., hidden).
    h_norm = grouped_rms(hyper, hnorm, hc * hidden, epsilon).reshape(rows, hc, hidden)
    # One embedding term, shared by every stream.
    e_norm = grouped_rms(embedding, enorm, hidden, epsilon)
    e_norm = np.repeat(e_norm[:, None, :], hc, axis=1)
    fused = np.concatenate([e_norm, h_norm], axis=-1) @ eh_proj
    return fused.reshape(rows, hc * hidden)


# ---------------------------------------------------------------------------
# QSA indexer (full-attention layers)
# ---------------------------------------------------------------------------

def rope_partial_half(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Half-split rope over the first cos.shape[-1] dims of the last axis.

    cos/sin are the HF-style duplicated tables (width = rotary_dim, the two
    halves repeat the same rotary_dim/2 frequencies); the tail of x past
    rotary_dim passes through untouched. Matches `apply_rotary_pos_emb`.
    """
    rot = cos.shape[-1]
    half = rot // 2
    r, tail = x[..., :rot], x[..., rot:]
    rotated = np.concatenate([-r[..., half:], r[..., :half]], axis=-1)
    return np.concatenate([r * cos + rotated * sin, tail], axis=-1)


def qsa_query(hidden: np.ndarray, q_w: np.ndarray, q_norm: np.ndarray,
              cos: np.ndarray, sin: np.ndarray, n_heads: int, head_dim: int,
              epsilon: float = 1e-6) -> np.ndarray:
    """Indexer queries [rows][n_heads][head_dim]: project, per-head rms, rope.

    q_w [hidden][n_heads*head_dim] (x @ q_w); q_norm is the BAKED per-head
    weight [head_dim]; cos/sin [rows][rotary_dim] at the query positions.
    """
    q = (hidden @ q_w).reshape(hidden.shape[0], n_heads, head_dim)
    q = grouped_rms(q, q_norm, head_dim, epsilon)
    return rope_partial_half(q, cos[:, None, :], sin[:, None, :])


def qsa_block_keys(raw_keys: np.ndarray, k_norm: np.ndarray,
                   cos: np.ndarray, sin: np.ndarray, ratio: int,
                   epsilon: float = 1e-6) -> np.ndarray:
    """Pooled block keys [n_blocks][head_dim] from raw per-token keys.

    Only complete blocks (block b = positions [b*ratio, (b+1)*ratio)); the mean
    is taken in fp32, THEN the rms norm, THEN rope anchored at the block's
    first position (cos/sin indexed at b*ratio). Incomplete tail tokens never
    produce a block -- they are always attended instead.
    """
    n_blocks = raw_keys.shape[0] // ratio
    pooled = raw_keys[:n_blocks * ratio].reshape(n_blocks, ratio, -1)
    pooled = pooled.astype(np.float32).mean(axis=1)
    head_dim = pooled.shape[-1]
    pooled = grouped_rms(pooled, k_norm, head_dim, epsilon)
    anchors = np.arange(n_blocks) * ratio
    return rope_partial_half(pooled, cos[anchors], sin[anchors])


def qsa_scores(query: np.ndarray, block_keys: np.ndarray) -> np.ndarray:
    """Block scores [n_blocks]: sum over heads of relu(q_h . k_b) / sqrt(dim).

    query [n_heads][head_dim] (normed+roped), block_keys [n_blocks][head_dim].
    """
    head_dim = query.shape[-1]
    dots = block_keys.astype(np.float32) @ query.astype(np.float32).T
    return np.maximum(dots, 0.0).sum(axis=-1) / np.sqrt(head_dim)


def qsa_selected_tokens(scores: np.ndarray, n_visible: int, budget: int,
                        ratio: int) -> np.ndarray:
    """Token indices a query at (n_visible-1) may attend, sorted ascending.

    scores covers the query's complete blocks (n_visible // ratio of them);
    the top budget//ratio blocks win (ties broken toward the LOWER block index,
    deterministically -- torch.topk on our score dtype agrees except on exact
    float ties, which the fixtures avoid), and the incomplete tail
    [n_blocks*ratio, n_visible) is always kept.
    """
    n_blocks = n_visible // ratio
    top = min(budget // ratio, n_blocks)
    order = np.lexsort((np.arange(n_blocks), -scores[:n_blocks]))
    chosen = np.sort(order[:top])
    tokens = (chosen[:, None] * ratio + np.arange(ratio)[None, :]).reshape(-1)
    tail = np.arange(n_blocks * ratio, n_visible)
    return np.concatenate([tokens, tail])


def qsa_token_mask(hidden: np.ndarray, qk_w: np.ndarray,
                   q_norm: np.ndarray, k_norm: np.ndarray,
                   cos: np.ndarray, sin: np.ndarray,
                   n_heads: int, head_dim: int, budget: int, ratio: int,
                   epsilon: float = 1e-6) -> np.ndarray:
    """Full causal-single-sequence indexer pass: bool keep mask [rows][rows].

    qk_w [hidden][(n_heads+1)*head_dim] is the fused projection, queries first
    then the single key head, matching `index_qk_proj`. Row r's mask covers
    tokens 0..r (everything else False, as causal already closes it).
    """
    rows = hidden.shape[0]
    queries = qsa_query(hidden, qk_w[:, :n_heads * head_dim],
                        q_norm, cos, sin, n_heads, head_dim, epsilon)
    raw_keys = hidden @ qk_w[:, n_heads * head_dim:]
    block_keys = qsa_block_keys(raw_keys, k_norm, cos, sin, ratio, epsilon)
    mask = np.zeros((rows, rows), dtype=bool)
    for r in range(rows):
        n_visible = r + 1
        n_blocks = n_visible // ratio
        scores = qsa_scores(queries[r], block_keys[:n_blocks])
        mask[r, qsa_selected_tokens(scores, n_visible, budget, ratio)] = True
    return mask
