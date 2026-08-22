# Prefill expert GEMM — the grouped kernels stop re-decoding weights per token

## Motivation (measured 2026-08-22)

plans/prefill-expert-stream.md closed on this wall: the grouped expert
kernels are decode-shaped — one full-width matvec block per (token, route),
524,288 blocks per `q5k_grouped_swiglu_rows` call at 1024 rows — so a
streamed expert's weights are re-decoded from VRAM once per routed token
(~70 GiB of reads per fully streamed half-layer, ~15 s per chunk). Uploads
are not the problem (26.4 GB/s measured from the registered mapping); the
execution shape is.

The fix needs no new GEMM kernels. The dense prefill MMQ kernels
(`*_matmul_q8_mmq`, the `_MIN` variants for asymmetric K-quants, the tiled
dp4a fallback) are weight-format kernels that read a contiguous matrix —
exactly what a staged expert triple is. What is missing is the grouping:

1. **Gather**: pack the routed tokens of one expert from `normalized` into
   a contiguous activation tile, quantize once with the existing
   `quantize_q8_blocks_rows`.
2. **GEMM**: gate and up through the format's MMQ (weights decoded once per
   128-token tile instead of once per token), the existing `silu_mul`, the
   down projection through MMQ again.
3. **Scatter**: accumulate the down output into `fourth` at each token's
   row, scaled by its route weight.

Per staged expert that is ~8 small launches; the weight traffic falls from
`n_tokens × triple` to `ceil(n_tokens/128) × triple` — 128× less at full
occupancy. Expected balance on the motivating config (Q5, 4 K, 5000 MiB):
CPU keeps `(1−α)` of the 5.65 s sweep, GPU adds core 3.9 s + α × (uploads
hidden on the prefetch stream + ~1–2 s of GEMM); the budget knob is α.
Projected sweet spot ~α 0.3–0.5 → **~4.5 s ≈ 900 tok/s**, from 653. That
projection died once already on execution shape — hence the same gates.

## Design

All inside the existing streaming block (arena, fences,
`prefill_streamed_bytes`, engagement conditions unchanged):

- Host builds, per half-layer, the per-expert token lists (index + route
  weight, concatenated in expert order — deterministic by construction) and
  uploads them to a small per-span scratch slice.
- Per staged expert: gather → quantize → gate MMQ → up MMQ → silu_mul →
  quantize → down MMQ → scatter-add. Dispatch mirrors `dense_rows`' MMQ
  branch (mmq / mmq_min / tiled fallback) against the staged pointers.
- Scratch (packed activations, q8 tiles, intermediates) is a separate
  device allocation, two span slices, charged into the base GPU budget.
- Eligibility: the layer's three expert formats must have batched kernels
  and the device tensor cores; otherwise the expert stays on the CPU.
  Experts below a small token floor stay on the CPU too — the upload and
  launch overhead needs tokens to amortize against.
- The per-token grouped kernels remain untouched for the resident split
  path (decode-sized route counts are exactly what they are shaped for).

### Numerics

MMQ computes on Q8-quantized activations; the resident grouped kernels
read f32 activations. Streamed output therefore differs from resident-all
in quantization, not in structure — the same trade the dense prefill path
already made (pinned by test_v2_prefill_parity across every format). The
parity class's streamed≡resident bit-identity gate is replaced by:
greedy-token equality on the fixture if it holds empirically, else
determinism + budget-0 + engagement plus the dense prefill-parity style
argument; live quality is gated by inspection of continuations plus the
existing suites.

## Gates

1. Budget 0 bit-identical to today, everywhere (default unchanged).
2. Fixed budget deterministic; engagement counter nonzero on the fixture.
3. Live sweep on the motivating config beats the 653 tok/s baseline by
   ≥15% at some budget, or the plan closes with the measurement.
4. Suites green; the resident split path's behavior untouched.

## Staging

1. Kernels (gather, scatter-add) + scratch plumbing + per-expert GEMM
   orchestration behind the same env budget. Gates 1–2.
2. Live budget sweep, interleaved A/B at the best budget, gate 3. Then
   decide default-on and the promotion of the env knob to an option.

## Status (2026-08-22) — landed, +31% measured

`6753734`. The budget sweep at 4 K (single samples, hot machine):

| budget | tok/s | cpu_moe | streamed |
|---:|---:|---:|---:|
| 32 MiB | 767 | 4.21 s | 4.8 GiB |
| **48 MiB** | **810** | 3.83 s | 6.9 GiB |
| 64 MiB | 781 | 3.47 s | 9.6 GiB |
| 128 MiB | 743 | 2.43 s | 19.9 GiB |
| 256 MiB | 661 | 1.55 s | 39.8 GiB |
| 1 GiB | 515 | 0.45 s | 77 GiB |

Interleaved A/B at the optimum: **617 → 810 tok/s (+31%)** over the best
prior config (pipeline + 2048-row chunks). The optimum is small because
per-half-layer restaging cost scales with budget while the densest-first
benefit saturates; big budgets drown in their own re-uploads. Gates: budget
0 bit-identical live, streamed-all ≡ resident-all on the Q8_0 fixture
through the GEMM path, fixed budget deterministic, suites green.

Open follow-ups, each its own measured decision:
- Default-on with a small auto-sized budget (and promotion of the env knob
  to a runtime option) — needs decode-side and multi-model coverage first.
- Persistent staging across chunks/halves for the stable dense experts
  would cut the restaging term and move the optimum right.
- The 2048-row chunk interaction re-measures under streaming (the old 653
  datum predates it).

Session arc for prompt speed on this configuration: 413 tok/s (serial, two
days ago) → 586 (pipeline) → 653 (+2048 rows) → **810 (expert GEMM)** —
**+96%** end to end, bit-identical or gated at every step.

## Decode-side datum (2026-08-22)

With the 48 MiB budget active, decode loses ~2–4% to the arena's slot cost
(hits/token 137 → 134 mutable, 43 → 40 seeded-immutable; ~40 slots) against
the +31% prefill gain — favourable for prompt-heavy serving, a per-config
call for decode-heavy. The same decomposition (COLIBRI_TIMING=1) settled a
related question: decode's expert bucket is ~96% CPU MoE compute at the
host bandwidth wall, not transfers, so the prefill fence pattern has no
decode target; decode's real lever is residency policy (immutable 43 tok/s
vs mutable+seed 62, the ornith determinism trade).
