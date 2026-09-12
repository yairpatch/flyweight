# qwen4exp prefill headroom — four levers, measured in ranking order

## Motivation (measured 2026-09-08)

Qwen3.8-Flash-Next UD-IQ1_S, 2048-token prompt, warm page cache,
`--expert-mode auto`, RTX 5070 Ti Laptop 12 GB + 60 GB RAM:

| | |
|---|---:|
| prefill | 8.9 s, 229 tok/s |
| CPU expert sweep (`prefill_expert_nanoseconds`) | 8.5 s (96%) |
| route wait | 0.03 s |
| `prefill_streamed_bytes` | 0 |

The prefill expert-GEMM stream (plans/prefill-expert-gemm.md, 810 tok/s
on the 35B) never engaged: its gate required `dma_paging`, and auto
registration refuses when the file plus headroom exceeds host RAM, which
every large qwen4exp checkpoint does on this box. Inside the sweep
(FLYWEIGHT_MOE_PROFILE=1): f32 GEMM ~60%, dequant to f32 ~28%, activation
and store ~11%; the rows path had no int8 activation route at all.

Every expert is touched once per pipeline half, ~153 GB per 2048-token
prompt; at PCIe rate that is ~6 s, so the GPU can only ever take a share
of the sweep. The ceiling is the CPU/GPU split.

## Lever 1 — uploads off the engine thread (`ea88992`)

- **Pinned mirror.** The staged experts of a half-layer are packed into a
  pinned host mirror of the arena (two-slot ring per slice, drain events)
  with non-temporal stores and moved as one copy, so the gate no longer
  needs DMA registration. Waiting on the *same* slot's previous upload
  stalled on compute-stream queue depth (that upload is queued behind
  the previous layer's GEMMs) — hence the two-deep ring.
- **Shared staging.** One expert set per layer, in the arena slice of the
  layer's parity, chosen by whichever half arrives first and reused by
  the other with its own token lists. Both halves route to nearly the
  same experts: pack and PCIe bytes halve at the same GPU share
  (51 → 25 GB streamed per prompt at 256 MiB). Always two slices, even
  for the serial driver: a whole-arena upload fenced on slice 0's events
  alone raced the previous chunk's last odd layer.
- **Pack worker.** The leader's pack + upload runs on a worker
  (4 threads, memcpy is DRAM-bound at ~18 GB/s here regardless of thread
  count) while the engine thread sweeps; the GEMM enqueue joins it first.
- Per-expert group GEMM is launch-bound at qwen4exp's ~16 tokens per
  group (~160 µs per group); the routed block-table MMQ is what makes the
  streamed share pay. 512 MiB per-expert: 113 tok/s, worse than baseline.

Budget sweep, routed MMQ, interleaved: 48 MiB 250, 128 ~320, 256 ~385–425,
512 ~295 (GPU-bound). Knobs: FLYWEIGHT_STREAM_WORKER=0,
FLYWEIGHT_STREAM_WORKER_THREADS, FLYWEIGHT_STREAM_COPY_NT=0.

## Lever 2 — int8 rows GEMM (`4a0ea18`)

IQ1_S gate/up rows are folded once to int8 (group scale folded; the 1/8
delta rides the activation's per-group sums) and dotted against
unsigned-8 activations with dpbusd (`qwen_iq1s_fold_rows_vnni512`,
`qwen_u8_gemm_k256_vnni512`, Q8_K rounding). Two lessons: the first fold
overflowed int8 (15 × 9 = 135), and GCC kept the tile's accumulators on
the stack until the token loop was unrolled and the side accumulator
folded into the main one. Against the library's -O3 f32 path the gain
is 1.2–1.5x on the gate/up phase (the -O2 bench overstated it at 2.6x);
the gather-bound row decode is a floor both paths pay. IQ4_NL down
measured slower than f32, so it stays behind FLYWEIGHT_ROWS_Q8_DOWN=1.
Contract: `flyweight_qwen_rows_q8_vnni512_contract`.

Found on the way: the mid-prefill checkpoint spread split every chunk
(targets 256/682/1364 for a 2048 prompt → halves of 128–341 rows, every
expert decoded per half). Spread targets now snap to
`interval + k·prefill_rows` for prompts of at least two chunks:
218 → 243 tok/s at budget 0, short prompts unchanged.

## Lever 3 — defaults (`9d137b0`)

`routed_moe` is tri-state (None = auto): on wherever every expert role
of a layer has a routed kernel. The auto budget is 256 MiB when a
majority of MoE layers can take it (the UD checkpoints keep a few Q8_0
down layers on the per-expert path), 48 MiB otherwise. An explicit
`routed_moe=True` no longer forces direct paging or a cache seed.
Verified not to regress the 35B: UD-Q6_K 600–620 → 745–770 tok/s.

## Lever 4 — delta-net WY path on the halves (`9d137b0`)

`kDeltaChunkedMinimumRows` 1024 → 512 so the pipeline's 512-row halves
take the chunked path (token-identical, ~+3%; the GPU is not yet the
limiter). FLYWEIGHT_DELTA_CHUNKED_MIN_ROWS overrides.

## The race the levers uncovered

`block_reduce_sum` had a barrier after publishing warp partials and none
after reading them; back-to-back reductions in one block (the
`*_matmul_rows` four-token tile) let a fast warp overwrite a partial
warp 0 was still reading. Latent until the IQ3_S per-expert prefill path
ran on UD-IQ4_XS: greedy tokens flipped run to run with expert input,
routes, staged bytes (FLYWEIGHT_STREAM_TRACE=3 verifies the slice against
the mmap) and CPU output all bit-identical (FLYWEIGHT_STREAM_TRACE=2
per-layer checksums). Fixed with a trailing barrier.

## Status — pure defaults, 2048 tokens, warm

| checkpoint | before | after |
|---|---:|---:|
| Flash-Next UD-IQ1_S | 229 | 471–476 tok/s |
| Flash-Next UD-IQ4_XS | ~200 | 300–340 |
| Qwen3.6-35B UD-Q6_K | 600–620 | 745–770 |

Deterministic run to run on both qwen4exp files; budget 0 unchanged;
path-parity, prefill-parity, qwen4exp-parity, iq1s, imatrix, checkpoint,
CLI and GPU suites green; kernel contracts green. Decode with the 256 MiB
arena (−9% expert-cache slots) measured no slower than budget 0 in two
interleaved pairs on UD-IQ1_S.

Remaining, each its own measured decision: an int8 rows path for IQ3_S
gate/up (needs the 16-bit dpwssd shape: magnitudes × scale exceed int8);
a Q8_0 routed kernel so the UD checkpoints' Q8_0 down layers leave the
per-expert path; overlapping the fold across halves the way staging is.
