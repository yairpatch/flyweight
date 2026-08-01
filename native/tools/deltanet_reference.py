"""NumPy reference for the Qwen gated DeltaNet recurrence.

This mirrors `qwen_delta_recurrent_chunk` in colibri_v2_native_kernels.hpp
element for element, including the epsilon placements and the exact order of the
state update, so it can serve as the oracle for the chunked WY kernels.

Buffer layouts, all float32 and all matching the kernels:

  convolved  [rows][2 * key_heads * head_dim + value_heads * head_dim]
             per row: queries | keys | values
  gates      [rows][value_heads * head_dim]
  beta       [rows][value_heads]      (logits)
  decay      [rows][value_heads]      (logits)
  a_log      [value_heads]            per-head decay coefficient
  dt_bias    [value_heads]
  norm       [head_dim]               shared across heads
  state      [value_heads][head_dim(key)][head_dim(value)]
  output     [rows][value_heads * head_dim]
"""
from __future__ import annotations

import numpy as np


def softplus(x: np.ndarray | float) -> np.ndarray | float:
    # The kernel switches to the identity above 20 to avoid exp overflow.
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def reference(convolved, gates, beta_logits, decay_logits, a_log, dt_bias,
              norm, state, key_heads, value_heads, head_dim, epsilon):
    """Run the recurrence over every row; returns (output, final_state).

    `state` is not modified; the returned state is a fresh array.
    """
    rows = convolved.shape[0]
    total_key = key_heads * head_dim
    state = state.astype(np.float64).copy()
    output = np.zeros((rows, value_heads * head_dim), dtype=np.float32)

    for h in range(value_heads):
        key_head = h % key_heads
        off = key_head * head_dim
        s = state[h]  # [key][value]
        for t in range(rows):
            row = convolved[t]
            q_raw = row[off:off + head_dim].astype(np.float64)
            k_raw = row[total_key + off:total_key + off + head_dim].astype(np.float64)
            v = row[2 * total_key + h * head_dim:
                    2 * total_key + (h + 1) * head_dim].astype(np.float64)

            q = q_raw / np.sqrt(np.dot(q_raw, q_raw) + 1e-6) / np.sqrt(head_dim)
            k = k_raw / np.sqrt(np.dot(k_raw, k_raw) + 1e-6)
            beta = 1.0 / (1.0 + np.exp(-float(beta_logits[t, h])))
            decay = np.exp(float(a_log[h]) * softplus(float(decay_logits[t, h]) + float(dt_bias[h])))

            # S <- decay * (I - beta k k^T) S + beta k v^T, in the kernel's order.
            s *= decay
            memory = k @ s
            delta = (v - memory) * beta
            s += np.outer(k, delta)
            core = q @ s

            inverse_rms = 1.0 / np.sqrt(np.dot(core, core) / head_dim + epsilon)
            gate = gates[t, h * head_dim:(h + 1) * head_dim].astype(np.float64)
            output[t, h * head_dim:(h + 1) * head_dim] = (
                core * inverse_rms * norm.astype(np.float64) * silu(gate))
    return output, state.astype(np.float32)


def checkpoint_decay(path, value_heads, seed=0):
    """Real (a_log, dt_bias) for one DeltaNet layer of a GGUF checkpoint.

    Worth using rather than synthesising these two: a_log reaches -105 in
    Qwen3.6-35B-A3B, so exp(a_log * softplus(...)) underflows to zero for some
    heads.  That is the regime the chunked form has to get right, and a synthetic
    a_log drawn from -exp(randn) never reaches it.
    """
    import sys

    sys.path.insert(0, "src")
    from colibri_next.v2 import V2Model

    model = V2Model(path)
    names = {t["name"] for t in model.tensors()}
    layers = sorted({int(n.split(".")[1]) for n in names if "ssm_a" in n})
    rng = np.random.default_rng(seed)
    layer = int(rng.choice(layers))
    a_log = np.frombuffer(model.read_tensor(f"blk.{layer}.ssm_a"), dtype=np.float32)
    dt_bias = np.frombuffer(model.read_tensor(f"blk.{layer}.ssm_dt.bias"), dtype=np.float32)
    if a_log.size != value_heads:
        raise ValueError(
            f"checkpoint layer {layer} has {a_log.size} value heads, not {value_heads}")
    return layer, a_log.copy(), dt_bias.copy()


def random_inputs(rows, key_heads, value_heads, head_dim, seed=0, decay=None):
    """Inputs on the scale the real projections produce.

    `decay` optionally supplies a real (a_log, dt_bias) pair from a checkpoint.
    """
    rng = np.random.default_rng(seed)
    total_key = key_heads * head_dim
    width = 2 * total_key + value_heads * head_dim
    if decay is not None:
        a_log, dt_bias = decay
        return dict(
            convolved=rng.standard_normal((rows, width), dtype=np.float32),
            gates=rng.standard_normal((rows, value_heads * head_dim), dtype=np.float32),
            beta_logits=rng.standard_normal((rows, value_heads), dtype=np.float32),
            decay_logits=rng.standard_normal((rows, value_heads), dtype=np.float32),
            a_log=a_log.astype(np.float32),
            dt_bias=dt_bias.astype(np.float32),
            norm=(1.0 + 0.1 * rng.standard_normal(head_dim)).astype(np.float32),
            state=(0.1 * rng.standard_normal((value_heads, head_dim, head_dim))).astype(np.float32),
        )
    return dict(
        convolved=rng.standard_normal((rows, width), dtype=np.float32),
        gates=rng.standard_normal((rows, value_heads * head_dim), dtype=np.float32),
        beta_logits=rng.standard_normal((rows, value_heads), dtype=np.float32),
        decay_logits=rng.standard_normal((rows, value_heads), dtype=np.float32),
        # a_log is negative in every checkpoint, which is what keeps decay <= 1.
        a_log=(-np.exp(rng.standard_normal(value_heads))).astype(np.float32),
        dt_bias=rng.standard_normal(value_heads, dtype=np.float32),
        norm=(1.0 + 0.1 * rng.standard_normal(head_dim)).astype(np.float32),
        state=(0.1 * rng.standard_normal((value_heads, head_dim, head_dim))).astype(np.float32),
    )
