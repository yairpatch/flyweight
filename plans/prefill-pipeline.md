# Prefill chunk pipelining — CPU experts overlapped with GPU dense work

## Motivation (measured 2026-08-21)

Qwen3.6-35B Q5_K_M, hybrid auto-fit, 4096-token prompt, RTX 5070 Ti Laptop,
fresh runtime, `COLIBRI_PREFILL_PROFILE=1`, expert history off:

| phase | time | share |
|---|---:|---:|
| prefill total | 9.72 s (421 tok/s) | 100% |
| CPU routed experts (`prefill_expert_nanoseconds`) | 5.59 s | 57.5% |
| route wait — CPU blocked on GPU (`prefill_route_wait_nanoseconds`) | 3.51 s | 36.2% |
| GPU core (events) | 3.67 s | — |

The rows forward serializes per layer: GPU pre-MoE work → route download
sync → `qwen_cpu_moe_rows` (GPU idle) → upload → combine. CPU and GPU each
sit idle while the other works; the two idle phases sum to ~94% of prefill.
Perfect overlap bounds prefill at `max(GPU, CPU) ≈ 5.6–6 s` — **~1.6–1.7×
prompt speed** on this configuration. The gain grows with the CPU-expert
share (Q6, longer prompts) and shrinks when a seeded GPU expert set absorbs
routes.

## Design: half-chunk software pipeline

Split each prefill chunk into halves A and B (B sees A's KV/DeltaNet state
through normal stream ordering). Restructure the `qwen_forward_rows` layer
loop into three phase functions over an explicit (row_begin, row_count)
range:

- `queue_core(half, L)`: norms, DeltaNet/attention, router launch, route
  downloads, event record, shared expert — all GPU, no sync.
- `cpu_phase(half, L)`: event sync, route pruning, GPU-resident table build,
  `qwen_cpu_moe_rows` for the half.
- `queue_combine(half, L)`: output upload, adds, residual swap.

Single-thread schedule (the engine thread queues GPU work between its own
CPU MoE stints; OpenMP inside the MoE uses the cores either way):

```
queue_core(A,L)
queue_core(B,L)              # B's L-input was combined last iteration
cpu_phase(A,L)               # GPU meanwhile runs core(B,L)
queue_combine(A,L); queue_core(A,L+1)
cpu_phase(B,L)               # GPU meanwhile runs core(A,L+1)
queue_combine(B,L); queue_core(B,L+1)
...
```

Steady state: the GPU always holds one half-layer of queued work while the
CPU chews the other half's experts.

### Memory

Two half-size workspace slices replace one full-size slice — net zero for
the row-indexed regions. Two route events instead of one. The per-layer
pointer-table staging must be doubled or fenced (halves alternate).

### Numerics — already derisked

- Each half flows through the code exactly as a 512-row chunk does today.
  Measured on the live Q5 model at 4096 tokens, fixed 5000 MiB cache:
  `COLIBRI_PREFILL_ROWS=512` and `256` both emit tokens **identical** to
  `1024`. Chunk-boundary effects (cuBLAS attention tiling, DeltaNet WY
  chunking at 64-row multiples) do not change greedy output.
- Standalone cost of 512-row chunks: 430 → 409 tok/s (−5%), the price the
  overlap must recover first. 256-row: −16%, so halves of the default 1024
  (=512 each) are the right grain.
- Keep the half boundary a multiple of `kDeltaChunk` (64) so the WY chunk
  sequence is unchanged.

### Interactions

- Mid-prefill checkpoint targets clamp chunk boundaries; a checkpoint may
  only be taken between full chunks (both halves combined), never between
  halves.
- MTP prompt-pair preservation copies per-row hidden; per-half copies land
  in the same preserved buffer at their row offsets.
- `hybrid_prefill_cpu` and CPU-only modes pipeline identically (they are the
  configurations with the largest CPU share).
- Cancellation checks stay at half boundaries.

## Gates (roadmap discipline)

1. Bit-identical greedy output vs the unpipelined path: full parity harness
   plus live Q5/Q6 token equality at 256/1K/4K/10K.
2. Prefill throughput on the motivating config improves ≥25% (target ~60%);
   no regression >3% on dense checkpoints (no CPU experts — the pipeline
   must no-op or stay neutral there).
3. `COLIBRI_PREFILL_PIPELINE=0` (or a runtime option per roadmap rules)
   restores the serial path for rollback and A/B.

## Status (2026-08-21) — both stages landed

- Stage 1 (`6447752`): the three phase functions over a (begin, count) span,
  single full-range span, bit-identical on live Q5 256/1K/4K/10K + Q6 4K for
  both hybrid prefill paths.
- Stage 2 (`72785f3`): the two-half schedule, second route event, engaged
  only for host-routed-expert configurations. Measured on the motivating
  config (Q5, hybrid-cpu, 4096 tokens, 5000 MiB cache, interleaved A/B):
  **413 → 586 tok/s (+41%)**, gate was +25%. Bit-identity held everywhere
  gate 1 lists, plus Qwen3.8 IQ3_XXS with mtp_drafts=2.
- Stage 3 measured (2026-08-22) and **closed: build neither option.** Host
  counters on the pipelined path, Q5 hybrid-cpu:

  | length | total | cpu_moe | route_wait | other |
  |---|---:|---:|---:|---:|
  | 4096, pipe on | 7.55–8.35 s | 6.93–7.65 s | 0.07 s | 0.6 s |
  | 4096, pipe off | 10.55 s | 6.45 s | 3.50 s | 0.6 s |
  | 10240, pipe on | 18.30 s | 16.79 s | 0.10 s | 1.4 s |
  | 10240, pipe off | 26.87 s | 18.10 s | 7.23 s | 1.5 s |

  route_wait fell from 33–27% of prefill to **under 1%**: the GPU never
  makes the engine thread wait, so the single-thread schedule already
  captures the max(GPU, CPU) bound — the plan's stage-3 condition held.
  Quarter chunks have nothing left to overlap and would only deepen the
  CPU-side halving cost (each routed expert's weights swept once per half;
  +8–19% at 4K, noise at 10K). A dedicated MoE worker thread could at most
  absorb `other` (~8%) while contending with the OpenMP MoE team for cores.
  Prefill is now purely CPU-MoE-bound; the next prompt-speed lever is a
  faster CPU expert path or shifting expert share to the GPU — different
  plans.
- Chunk-size datum for whoever tunes next: `COLIBRI_PREFILL_ROWS=2048`
  under the pipeline measures **653 tok/s** at 4K (cpu_moe 7.0 → 5.65 s —
  each half's expert sweep amortizes over 1024 rows instead of 512), tokens
  bit-identical on both hybrid paths; 4096 regresses to 599. Not made the
  default because the rows workspace doubles and costs 93 expert-cache
  slots (757 → 664) on a 5000 MiB budget — a decode-side trade that needs
  its own measurement per configuration.

## Staging

1. Mechanical: extract the three phase functions over (row_begin, row_count)
   with a single full-range half — bit-identical, parity-gated, its own
   commit.
2. Semantic: the two-half schedule behind the option, measured on the
   primary configurations.
3. Only then consider deeper grain (quarter chunks) or a dedicated MoE
   worker thread — measure first; the single-thread schedule already
   captures the bound if CPU ≥ GPU per layer.
