# Prefill expert streaming — GPU takes a measured share of the routed experts

## Motivation (measured 2026-08-22)

Qwen3.6-35B Q5_K_M, hybrid auto-fit, 4096-token prompt, 5000 MiB cache,
expert history off, RTX 5070 Ti Laptop. After the prefill pipeline
(plans/prefill-pipeline.md) the decomposition is:

| configuration | prefill | cpu_moe | route_wait |
|---|---:|---:|---:|
| serial (pre-pipeline) | 9.9 s (413 tok/s) | 6.4 s | 3.5 s |
| pipeline, 1024 rows | 7.6 s (542–586 tok/s) | 6.9 s | 0.07 s |
| pipeline, 2048 rows | 6.3 s (653 tok/s) | 5.65 s | 0.10 s |

Prefill is purely CPU-MoE-bound: route_wait under 1% means the GPU finishes
its core work and idles ~2.4 s per prompt while the engine thread sweeps
experts. The policy makes this structural, not incidental: `auto` resolves
prefill routed experts to the CPU in every current configuration (admission
is decode-only — `misses_may_be_admitted` requires phase decode), so the GPU
contributes zero expert FLOPs to prefill today. `hybrid_prefill=split`
measures the same as `=cpu` for exactly this reason, and the pipeline
correctly engages for both.

The streaming case, measured:

- Expert weights: 22.7 GiB over 41 MoE layers → 567 MiB/layer. A 1024-row
  half routes essentially every expert of every layer, so per-chunk expert
  bytes ≈ the full 22.7 GiB regardless of grain.
- H2D from the registered mmap: 18 GB/s warm, 26 GB/s truly pinned, 0.5 GB/s
  on first touch (disk). Streaming a whole chunk's experts ≈ 1.26 s.
- The CPU sweeps the same experts in ~2.8 s per 2048-row chunk (compute-bound
  at ~7 GB/s effective — dequant+GEMM, not bandwidth).

So the GPU can *upload-and-compute* an expert byte ~2.2× faster than the CPU
can compute it, and it is idle anyway. Streaming a share α of each layer's
routed experts to a staging arena and running the existing grouped kernels
projects, upload-serialized on the main stream, to α ≈ 0.22 balanced →
~4.9 s ≈ **840 tok/s**; with uploads overlapped on the prefetch stream the
balance moves toward α ≈ 0.45 and ~2 s/chunk. Host-DRAM contention (uploads
read the same 51 GB/s the CPU MoE uses) will temper both — hence measure.

## Design

In `cpu_phase` (both drivers), after the half's routes arrive:

1. **Partition** the half-layer's routed entries by expert: rank experts by
   routed-token count, assign greedily to the GPU until a byte budget is
   spent (upload cost is per expert, benefit is per routed token, so densest
   experts first). Everything else stays in `weights_host` for
   `qwen_cpu_moe_rows` exactly as today.
2. **Stage** each GPU-share expert's gate/up/down from the mmap into a
   dedicated prefill staging arena, build the existing pointer tables at the
   staged addresses, and let the existing grouped SwiGLU/accumulate kernels
   and `queue_combine`'s adds do the rest — the split path's machinery,
   pointed at staged copies instead of cache slots.
3. **Budget** via a runtime option (env override for A/B), default off in
   stage 1. The arena is carved from the same GPU budget as everything else;
   the slot cost gets documented like the 2048-row datum.

### Numerics — eyes open

GPU experts run int8-quantized activations; the CPU runs float. The expert
precision contract pins the gap at 4.5e-03 relative — real but bounded, and
it is exactly the difference decode's hybrid path already ships. A mixed
partition therefore *changes greedy tokens* relative to all-CPU prefill.
That is a semantic change gated deliberately, not a parity break discovered
late:

- Budget 0 must stay bit-identical to today's output (the default).
- On a fixture where every expert fits resident, streamed-all must be
  bit-identical to resident-all (same kernels, same tables, same order).
- A fixed budget must be deterministic run to run (partition is a pure
  function of the routes).

### Interactions

- Pipelined halves partition independently; the staging arena is per-half
  (halves alternate) or fenced on the upload event.
- `record_expert_access` runs for the GPU share too — frequency history must
  not go blind to streamed experts.
- First chunk of a cold model pays first-touch (0.5 GB/s) on streamed pages;
  the CPU sweep pays the same faults today, so this shifts cost, not adds it.
- MTP verification and decode are untouched (decode-phase policy unchanged).

## Gates (roadmap discipline)

1. Budget 0: bit-identical to the shipped pipeline on the live Q5 A/B set.
2. Fixture: streamed-all ≡ resident-all, bit-exact; parity suites green.
3. Determinism: fixed budget, two runs, identical tokens.
4. Throughput ≥25% over the 653 tok/s baseline (target ~840) on the
   motivating config; no regression on CPU-only machines (arena absent →
   path never engages).
5. Rollback: budget option 0 restores today's behavior exactly.

## Staging

1. Mechanical: arena sizing + partition + staged tables behind the option,
   default off. Gates 1–3.
2. Semantic: measured default budget on the motivating config, interleaved
   A/B, gate 4. Document the VRAM/slot trade.
3. Only if stage 2 shows upload serialization on the main stream: move
   uploads to the prefetch stream with a completion event — measure first.
