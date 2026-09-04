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

## Status (2026-08-22) — stage 1 landed, stage 2 measured and FAILED

- Stage 1 (`b6b0ed8`) passed all three gates: budget 0 bit-identical live,
  streamed-all ≡ resident-all bit-exact on a Q8_0 fixture, deterministic,
  engagement-counter-guarded. The machinery stays behind the env flag.
- Stage 2 killed the design as staged. The budget sweep on the motivating
  config measured **130–245 tok/s against the 653 baseline** — the marginal
  upload throughput through `flyweight_gpu_upload` is ~1–2 GB/s, an order of
  magnitude under the probe's sequential rate. Two compounding causes:
  per-half-layer restaging multiplies volume (39.8 GiB moved for a 4 K
  prompt at a 256 MiB budget), and the copies are `cuMemcpyHtoDAsync` from
  a **file-backed mmap the driver cannot pin** — `cuMemHostRegister`
  returns −2 on it (probe_h2d), so every upload bounce-buffers, blocks the
  engine thread (~10 s of "other"), and serializes into the busy compute
  stream (~17 s of route wait). The same mechanism taxes decode expert
  paging ~7× (probe: 15.8 ms/token observed vs 2.2 ms at pinned rates).
- The upload-free variant — seeded residents serving prefill routes through
  the split table path — landed as pipeline coverage instead (`1332c26`):
  serial 440 → 655 tok/s pipelined, bit-identical on/off, but a dead heat
  with the best CPU-only config under interleaved thermal steady state
  (median 577 vs 578; the ~18% seeded absorption is repaid by the shared
  laptop power budget).
- **Successor lever, out of scope here:** make H2D from the model real —
  registration with `CU_MEMHOSTREGISTER_READ_ONLY`, or a pinned
  double-buffered bounce arena fed off the critical path. That would speed
  decode expert paging as well as revive this plan's streaming math; it is
  a prerequisite, not a variant, and deserves its own measured plan.

## Stage 3 (2026-08-22) — fenced dual-stream uploads, and the real wall

Both stage-2 explanations above were wrong, and the corrections are the
plan's lasting output. probe_registered.py shows the loader's COW mapping
**does** register (the earlier rc=−2 was the probe's own read-only mapping)
and serves random 3 MiB expert uploads at **26.4 GB/s** — the copies were
never slow. Moving the runtime's uploads to the prefetch stream with
two-way event fencing (`af6092f`) changed the sweep not at all: 128–143
tok/s either way.

The wall is the grouped expert kernels. They are decode-shaped — one
full-width matvec block per (token, route), 524,288 blocks per
`q5k_grouped_swiglu_rows` call at 1024 rows — so every expert's weights are
re-decoded from VRAM once per routed token: ~70 GiB of reads per fully
streamed half-layer, ~0.18 s at the 391 GB/s ceiling, ~15 s per chunk.
That is also the quiet reason `hybrid_prefill=split` never won and `auto`
forces cpu prefill.

**Verdict: this plan is closed.** No placement or transfer strategy can
matter until the GPU has expert-grouped GEMM kernels — tokens sorted by
expert, an MMQ-style tile per expert over its token group, the same class
of work that took dense prefill 353 → 533 tok/s. That is a kernel project
with its own plan; the streaming arena, fences, engagement counter, and
the streamed≡resident parity class stay as its ready-made substrate and
gates.
