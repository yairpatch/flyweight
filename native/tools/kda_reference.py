"""NumPy reference for the Kimi Delta Attention (KDA) recurrence.

KDA is the linear-attention half of BailingMoE3 (Ling 3.0) -- 18 of its 24
layers. It is a close relative of the Qwen gated DeltaNet already implemented
here (see deltanet_reference.py), with one structural difference that changes
the kernel shape:

    DeltaNet decays the whole state by ONE scalar per head per step.
    KDA decays it by a VECTOR -- one coefficient per key channel.

So the state update is `S <- diag(exp(g)) @ S` rather than `S <- a * S`, and the
gate is a full [heads][head_dim] tensor produced by its own projection instead
of a [heads] scalar. Everything else -- the delta rule, the beta gate, the query
readout -- is the same shape as DeltaNet.

Derived from `fla.ops.kda.naive.naive_recurrent_kda` and
`fla.ops.kda.gate.naive_kda_gate`, which is what the checkpoint's own
`modeling_bailing_moe_v3.py` calls. Validated against the real Triton kernel by
`kda_reference_check.py`.

Buffer layouts, all float32, chosen to match the layouts the CPU kernels here
already use:

  q, k, v     [rows][heads * head_dim]   post-convolution, post-SiLU
  gate_raw    [rows][heads * head_dim]   f_proj output, before the gate formula
  beta_logits [rows][heads]              b_proj output, before the sigmoid
  a_log       [heads]                    per-head decay coefficient
  dt_bias     [heads * head_dim]         per-channel gate bias
  state       [heads][head_dim][head_dim]  keys x values
  output      [rows][heads * head_dim]

OPEN QUESTION, deliberately not resolved here. `modeling_bailing_moe_v3.py`
passes `safe_gate=config.kda_safe_gate` (true) and `lower_bound=-5` into
`chunk_kda`. In flash-linear-attention 0.4.1 -- the version the checkpoint runs
against -- `chunk_kda` accepts `**kwargs` and BOTH ARE SILENTLY DISCARDED;
grepping the package finds `lower_bound` implemented only for HGRN, never for
KDA. So the reference model's observable behaviour has no clamp, and that is
what this file reproduces. But llama.cpp's bailingmoe3 PR carries a
`bailingmoe3.kda.safe_gate` hyperparameter, so some implementation does honour
it. If a clamp is ever needed it belongs on `gate` immediately below, as
`np.maximum(gate, lower_bound)`, and it will change long-context decay only.
"""
from __future__ import annotations

import numpy as np


def softplus(x: np.ndarray) -> np.ndarray:
    # Same guard the DeltaNet kernel uses: above 20 the identity is exact in
    # float32 and exp() would overflow.
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def l2_normalize(x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Per-row L2 normalization, as `use_qk_l2norm_in_kernel=True` applies."""
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + epsilon)


def kda_gate(gate_raw: np.ndarray, a_log: np.ndarray, dt_bias: np.ndarray,
             heads: int, head_dim: int) -> np.ndarray:
    """g = -exp(A_log) * softplus(f_proj(x) + dt_bias), per (head, channel).

    A_log is per head and broadcasts across the head's channels; dt_bias is per
    channel. The result is a log-space decay: always negative, so exp(g) is a
    contraction, and the closer to zero the longer the state remembers.
    """
    rows = gate_raw.shape[0]
    shaped = gate_raw.reshape(rows, heads, head_dim) + dt_bias.reshape(heads, head_dim)
    decay = np.exp(a_log).reshape(1, heads, 1)
    return -decay * softplus(shaped)


def reference(q, k, v, gate_raw, beta_logits, a_log, dt_bias, state,
              heads, head_dim, scale=None, epsilon=1e-6):
    """Run the recurrence over every row; returns (output, final_state).

    `state` is not modified; the returned state is a fresh array.
    """
    rows = q.shape[0]
    if scale is None:
        scale = head_dim ** -0.5

    queries = l2_normalize(q.reshape(rows, heads, head_dim), epsilon) * scale
    keys = l2_normalize(k.reshape(rows, heads, head_dim), epsilon)
    values = v.reshape(rows, heads, head_dim)
    gate = kda_gate(gate_raw, a_log, dt_bias, heads, head_dim)
    beta = sigmoid(beta_logits.astype(np.float64)).astype(np.float32)

    running = state.astype(np.float32).copy()
    output = np.zeros((rows, heads, head_dim), dtype=np.float32)

    for row in range(rows):
        for head in range(heads):
            s = running[head]
            # 1. Per-channel decay along the key axis. This is the whole
            #    difference from DeltaNet, where this line is `s *= alpha`.
            s *= np.exp(gate[row, head])[:, None]
            # 2. Delta rule: correct the value this key currently predicts
            #    towards the value actually observed, by beta of the gap.
            predicted = keys[row, head] @ s
            correction = values[row, head] - predicted
            s += np.outer(beta[row, head] * keys[row, head], correction)
            # 3. Read out with the query.
            output[row, head] = queries[row, head] @ s
            running[head] = s

    return output.reshape(rows, heads * head_dim), running
