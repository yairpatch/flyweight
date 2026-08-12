"""Checks kda_reference.py against flash-linear-attention's own reference.

The NumPy reference is the oracle the C++ kernel will be built against, so it
has to be pinned to something authoritative first.

Primary comparison is against `naive_recurrent_kda` / `naive_chunk_kda` plus
`naive_kda_gate` -- fla's own pure-PyTorch definitions of the operator, which
run on CPU and are what its Triton kernels are themselves tested against. Both
of fla's forms are checked, because the model picks between them by sequence
length (`fused_recurrent` at <= 64 tokens, `chunk` above), so a C++ port has to
match both.

The Triton kernels are also compared when they can run. On this machine they
cannot: the installed PyTorch does not support the GPU's compute capability
(sm_120), so that comparison is skipped rather than silently passed.

    python native/tools/kda_reference_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kda_reference  # noqa: E402


def main() -> int:
    try:
        import torch
        from fla.ops.kda.gate import naive_kda_gate
        from fla.ops.kda.naive import naive_chunk_kda, naive_recurrent_kda
    except ImportError as error:
        print(f"skipped: {error}")
        return 0

    device = "cpu"
    rng = np.random.default_rng(20260810)
    # Ling-3.0-tiny's real geometry: 16 heads of 128. Rows are a multiple of the
    # 64-wide chunk so chunk_kda and the recurrence see the same sequence.
    heads, head_dim, rows = 16, 128, 128

    def sample(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    q = sample(rows, heads * head_dim)
    k = sample(rows, heads * head_dim)
    v = sample(rows, heads * head_dim)
    gate_raw = sample(rows, heads * head_dim)
    beta_logits = sample(rows, heads)
    a_log = sample(heads)
    dt_bias = sample(heads * head_dim)
    state = np.zeros((heads, head_dim, head_dim), dtype=np.float32)

    expected, expected_state = kda_reference.reference(
        q, k, v, gate_raw, beta_logits, a_log, dt_bias, state, heads, head_dim
    )

    def to_torch(array, *shape):
        return torch.tensor(array, dtype=torch.float32, device=device).reshape(*shape)

    tq = to_torch(q, 1, rows, heads, head_dim)
    tk = to_torch(k, 1, rows, heads, head_dim)
    tv = to_torch(v, 1, rows, heads, head_dim)
    tg = to_torch(gate_raw, 1, rows, heads, head_dim)
    tbeta = torch.sigmoid(to_torch(beta_logits, 1, rows, heads).float())
    ta = torch.tensor(a_log, dtype=torch.float32, device=device)
    tdt = torch.tensor(dt_bias, dtype=torch.float32, device=device)

    # fla's naive forms take the gate already applied and q/k already
    # normalized, so composing them here is itself part of what is checked:
    # the gate formula and the L2 normalization come from fla, the assembly
    # from kda_reference.
    gated = naive_kda_gate(tg, ta, tdt)
    normalized_q = torch.nn.functional.normalize(tq, p=2, dim=-1, eps=1e-6)
    normalized_k = torch.nn.functional.normalize(tk, p=2, dim=-1, eps=1e-6)

    failures = 0
    for name, kernel in (
        ("naive_recurrent_kda", naive_recurrent_kda),
        ("naive_chunk_kda", naive_chunk_kda),
    ):
        out, final = kernel(
            q=normalized_q, k=normalized_k, v=tv, g=gated, beta=tbeta,
            initial_state=None, output_final_state=True,
        )
        actual = out.reshape(rows, heads * head_dim).float().cpu().numpy()
        scale = float(np.abs(expected).mean())
        error = float(np.abs(actual - expected).max()) / scale
        state_error = float(
            np.abs(final.reshape(heads, head_dim, head_dim).float().cpu().numpy()
                   - expected_state).max()
        ) / float(np.abs(expected_state).mean())
        ok = error < 2e-3 and state_error < 2e-3
        failures += not ok
        print(f"{'OK ' if ok else 'BAD'} {name:22s} "
              f"output rel-err {error:.3e}  state rel-err {state_error:.3e}")

    print("reference matches fla's own definitions of the operator" if not failures
          else "reference DISAGREES with fla")
    print("note: the Triton kernels were not compared -- this PyTorch build "
          "does not support this GPU (sm_120)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
