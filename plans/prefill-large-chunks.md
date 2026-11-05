# Prefill in large chunks — experts streamed once per chunk

> **STATUS 2026-11-05**: PLANNED. Milestone 1 in progress on `perf/prefill-rework`.

## Motivation (measured 2026-11-05)

Qwen3.8-Flash-Next UD-Q2_0 (35.6 GB of experts: IQ2_S / IQ2_XXS gate-up,
Q2_0 / IQ4_NL down), RTX 5070 Ti Laptop 12 GB, Ryzen 9955HX, 60 GB RAM,
PCIe 5.0 x8 (~29 GB/s pinned), warm page cache, 16K-token prompts:

| configuration | tok/s |
|---|---:|
| served settings (126K context, turbo4): 1024-row chunks | 501-518 |
| 32K context, 2048 / 4096 / 8192-row chunks (default CPU share) | 611 / 702 / 671 |
| 17K context, 4096 rows, everything streamed, routed MMQ | 612 |
| 8192+ rows | does not fit in VRAM |

Where the time goes:

- A chunk touches almost every expert, so prefill either computes them on
  the CPU (~25 core-seconds per 1000 tokens, dequant + f32 GEMM) or streams
  the whole ~35 GB expert set over PCIe **once per chunk** (twice per chunk
  with the two-half pipeline). At 4096-row chunks a 16K prompt moves ~140 GB.
- The rows workspace costs **~457 KB per row** on qwen4exp: four
  12288-float scratch rows (197 KB), three hyper-connection buffers (123 KB),
  DeltaNet chunk intermediates (~48 KB), the rest ~90 KB. A 16K-row chunk
  would need ~7.5 GB beside 4.2 GB of static weights, so the context's KV and
  the expert cache cap chunks at 1024-4096 rows.
- The routed MMQ re-decodes an expert's weights for every 32-token block: it
  plateaus at ~23 TOPS / ~144 ns per route from ~160 routes per expert up
  (fwbench routed_mmq_bench.py: 242 ns/route at 20 routes, 142 at 160).

Basalt (MIT, a Strata fork for this model) reports 2.4-2.7K tok/s at 16K on a
16 GB RTX 5060 Ti with 16K-token chunks: the prompt pass borrows the expert
cache's tail slots for its buffers and stream ring, streams every
non-resident expert once per chunk from ~6K-token chunks up, and keeps a CPU
share only for small chunks.

Model for 16K-row chunks on this box: ~35 GB uploaded once (~1.3 s at link
speed, host packing overlapped), routed experts ~1-3 s, dense/attention a few
seconds -> roughly 2000+ tok/s, ~3-4x today. None of the pieces below pays
alone except milestone 1.

## Milestones

1. **Span-local core scratch.** `queue_core` runs its mixer work in
   sub-spans of S rows with S-sized private scratch; only the hand-off
   regions (FFN input, routes, residual/hyper-connection state, expert
   output) stay chunk-wide. Target: ~85 KB/row chunk-wide + S x ~370 KB.
   Pays alone: the served 126K setup moves from 1024- to 4096-row chunks
   (measured +35-40% at equal settings).
2. **Lend the expert cache to the prompt pass.** Chunk buffers and the
   stream ring are laid over the tail slots of the expert cache for the
   prompt and refilled from RAM after it (the slots' experts become CPU
   misses meanwhile). Chunk size then follows the cache, not the context.
3. **Whole-chunk streaming.** From a chunk-size threshold up, every
   non-resident expert of a layer streams once per chunk (no two-half
   split), host packing through the pinned mirror on worker threads, routed
   MMQ on the GPU; the CPU share stays for small chunks.
4. **Decode-once routed MMQ.** A 128-token tile variant (or decode-once per
   expert) for experts with ~100+ routes, so the GPU expert time stops
   scaling with routes / 32.

Each milestone keeps the old path behind an env switch for A/B, and each
numeric change goes through `bench/bench_kernel_quality.py`.
