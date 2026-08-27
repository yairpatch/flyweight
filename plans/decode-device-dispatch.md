# Decode device dispatch: removing the per-layer host round-trip

> **Goal**: qwen4exp decode from 26 tok/s toward 60. The token is not bandwidth-bound;
> it is bound by 48 serialized host round-trips, one per MoE layer.
> **Status 2026-08-27**: Phase 0 RUN — one gate failed, and the design changed
> because of it. See "Phase 0 results" below. Phases 1-3 revised accordingly.

## Measured baseline (real UD-IQ1_S, `COLIBRI_TIMING=1`, 109-128 decode calls)

| phase | ms/token | share | what it is |
|---|---|---|---|
| `expert_page` (whole MoE phase) | 32.0 | 66% | host per-expert loop, admission memcpys, table build, queueing, CPU MoE |
| ├ `expert_compute` (CPU MoE) | 6.4-8.1 | 15% | the AVX2 expert dots for the miss set |
| └ everything else in the phase | ~25 | 51% | **host-serial, GPU idle** |
| `route_wait` | 8.4-8.7 | 18% | `cuEventSynchronize` on `route_event` |
| remainder (attn, delta, dense, LM head) | ~7.5 | 16% | |
| **total** | **42.6-48.2** | | 21-26 tok/s depending on cache warmth |

Cache: 68% hit, ~211 slot lookups/token, **40.6 admissions/token**.

**Bandwidth floor**: ~340 MB of expert weights touched per token; at 68% from VRAM
(391 GB/s) and 32% from RAM (51 GB/s) that is **~2.8 ms/token, i.e. ~350 tok/s**.
We are running 15x slower than the memory wall. Disk is not it either: a whole
64-token run took ~950 major faults / 351 MB paged in. **This is a latency and
serialization problem, not a bandwidth problem.**

## Four findings that reframe the work

1. **Routing is ALREADY on the device.** `colibri_gpu_route_topk`
   (`gpu_driver.cpp:1897`, kernel `colibri_v2_qwen_kernels.hpp:400`) does softmax +
   top-k + renormalize in one block and writes `selected_device` / `route_weights`.
   The readback is 40 bytes of expert ids (plus, in hybrid, the weights and the
   full 10 KB activation vector for CPU experts). *Moving routing on-device is not
   the work — it is already done.*

2. **The event sync exists solely so the host can resolve residency and build pointer
   tables.** `v2_runtime.cpp:16943` blocks the host; everything after it
   (`:17165-17262`) is a serial loop over the selected experts doing an
   `expert_residency.find()` per expert into a host `unordered_map`
   (`:734`), three `qwen_expert_role_scale` mmap reads per expert, then six
   `memcpy`s building `gate/up/down_pointers` and one upload
   (`:17230-17237`). The grouped kernels take **host-built arrays of device
   pointers** — that is the reason the ids must visit the host at all.

3. **DMA paging is auto-declined on this box, so every admission is a host memcpy.**
   `auto_direct` (`:13790`) needs `host_available >= model_bytes + max(4 GiB, model/4)`
   = 68 + 17 = 85 GB; the box has 52 GB available, so `dma_paging=false` and
   `direct_paging` reads 0. Result: each of the ~40 admissions/token copies a
   ~1.62 MB expert bundle from the mmap into pinned staging on the host thread
   (`:17061`) before uploading. That is **~66 MB/token of single-threaded memcpy on
   the critical path** — plausibly 5-8 ms of the unexplained 25 ms.

4. **Predictive prefetch has little headroom — measured, not assumed.**
   `COLIBRI_ROUTE_RECURRENCE=1` (the instrumentation at `:1231`, built for exactly
   this question and never consulted) over 22560 observations:
   resident 69.9%, **miss-but-in-recent-window 5.3%**, **miss-and-cold 24.7%**.
   Only ~18% of misses were predictable from the last few tokens. A whole-token or
   next-layer prefetch plan therefore cannot be the answer, which also explains why
   `--next-layer-prefetch` defaults to 0. *Do not invest here.*

## The design: device-resident slot table, pointer-free grouped kernels

Two in-tree precedents already do dispatch without a host pointer table:

- **Gemma 4** (`colibri_v2_qwen_kernels.hpp:1306-1356`): `gemma_q4_0_pinned_geglu` /
  `_accumulate` take `(layer_base, slot_bytes, const int* selected, ...)` and compute
  the weight address **in the kernel** as `layer_base + selected[rank]*slot_bytes`.
  The comment at `v2_runtime.cpp:15706` is the target state verbatim: *"the routed
  indices never need to visit the host … this layer costs no download and no event
  sync. That also makes the whole dense+router+MoE chain graph-capturable."*
  It works there because Gemma pins **whole layers**, so slot = expert.
- **Bailing prefill** (`bailing_route_rows`, `colibri_v2_native_kernels.hpp:1878`)
  goes further: it builds the inverted dispatch on device with `atomicAdd`, and the
  grouped kernels take `(base, expert_stride, counts, routes)`. Zero readback.

Qwen cannot use either as-is because its experts live at **arbitrary LRU cache-slot
addresses**. The missing piece is exactly one thing: a device-visible
`(layer, expert) -> slot` map.

**The change:**

1. **Device slot table.** `int32 slot_of[layers][experts]`, 48x512 = 98 KB on device,
   `-1` = not resident. The host already owns the only two mutation points —
   insert at `expert_residency[key]=slot` (`:1526`, `:17037`, `:17194`) and erase at
   eviction (`:1678`) — so maintaining a mirror is a 4-byte patch at each, off the
   critical path.
2. **Pointer-free grouped kernels.** New variants of the IQ grouped family
   (`iq1s`, `iq4nl`, `iq2xxs` — the three this checkpoint uses; macro at
   `colibri_v2_qwen_kernels.hpp:5795`) taking
   `(cache_base, slot_bytes, role_offsets, const int* slot_of_layer, const int* selected, const float* route_weights)`
   instead of `gate_ptrs/up_ptrs/down_ptrs`. Address is computed in-kernel, Gemma-style.
3. ~~**Misses without the host.**~~ **KILLED BY PHASE 0.** The idea was that when
   `slot_of[expert] < 0` the kernel would read the weights straight from the
   CUDA-host-registered mmap, so no routed id would ever visit the host. Measured
   20-47x slower than VRAM and ~5x slower than the CPU path it would replace — see
   below. **Misses stay on the CPU**, which means the host must still learn the miss
   set, which means the per-layer event sync survives.
4. **Admission becomes background work.** The LRU update itself (victim scan, slot
   assignment, the H2D of the bundle) does not have to happen inside the layer that
   missed — the miss is being served by the CPU regardless. Deferring it off the
   critical path is still available and is now the larger part of the win.

The end state per layer is: router matvec -> `route_topk` -> grouped swiglu for the
resident experts -> sync -> CPU experts for the miss set -> accumulate. The sync
remains; what goes away is the ~25 ms of host work wrapped around it.

## Phase 0 results (2026-08-27)

**Gate 1 — GPU reading expert weights from registered host memory: NO-GO.**
`bench_host_expert_reads.py`, real expert shape (2560->640), 10 experts per launch,
weights scattered through a 3 GiB registered host arena:

| kernel | weights | ms/launch | effective GB/s | vs VRAM |
|---|---|---|---|---|
| `iq1s_grouped_swiglu` | VRAM | 0.059 | 108.6 | - |
| `iq1s_grouped_swiglu` | host mmap | 2.745 | 2.3 | **46.6x** |
| `iq4nl_grouped_swiglu` | VRAM | 0.070 | 263.4 | - |
| `iq4nl_grouped_swiglu` | host mmap | 1.416 | 13.0 | **20.2x** |

Scaled to a real layer's miss set (~3 of 10 experts): ~0.8 ms/layer, ~39 ms/token —
against the CPU expert path's measured 6.4-8.1 ms/token for the same work. **The GPU
reading experts over PCIe is roughly 5x slower than the CPU executing them out of
RAM.** The cause is structural, not tunable: the octet decoders walk 8-element
octets per thread with no coalescing, and PCIe punishes exactly that pattern. IQ1_S
is worse than IQ4_NL because it packs more elements per byte, so the same octet
stride touches fewer bytes per transaction.

**Gate 2 — registering the expert ranges: viable, but must be range-scoped and
budgeted.** File-backed mmap registers at **~2 GiB/s** (2 GiB in 1.19 s, 8 GiB in
3.85 s), so the 40 GB expert set costs ~20 s once at load — acceptable. But
registration *locks* the pages, and 40 GB of experts alongside the 25.7 GB PLE
table exceeds this box's 60 GB. The current heuristic (`:13790`) is all-or-nothing
over the whole 68 GB mapping, which is why it declines. A range-scoped, budgeted
registration covering the expert tensors (or as much of them as a budget allows)
makes DMA paging available where it is silently off today.

**Note the distinction Phase 0 sharpened**: DMA paging is a *bulk sequential
`colibri_gpu_upload` from registered memory into a VRAM slot* — it is unaffected by
gate 1 and remains the Phase 1 win. What failed is *kernels dereferencing host
pointers*, which is a different access pattern entirely.

## Phases and gates

**Phase 0 — retire the two risky assumptions. DONE, results above.**
Gate 1 failed (host-resident expert reads are ~5x slower than the CPU path), gate 2
passed with a constraint. Cost: one afternoon and `bench_host_expert_reads.py`,
versus building a dispatch redesign around an assumption that was wrong.

**Phase 0b — localize the 25 ms before committing to Phase 2's shape.** The gap
between the MoE phase (32 ms) and its CPU compute (6.4-8.1 ms) is host-serial work
that has NOT been broken down. Candidates, all inside `:17165-17262`:
admission memcpys (~66 MB/token at `:17061`), `expert_residency.find()` per expert,
three `qwen_expert_role_scale` mmap touches per expert (`:17010` — these read 4 bytes
straight out of the mapping and can minor-fault), `select_expert_cache_slot`'s linear
victim scan (~40 admissions/token over a ~67-slot band), the six table memcpys, and
`record_expert_access`'s periodic O(layers x experts) decay sweep (`:1568`).
Add sub-timers under `COLIBRI_TIMING` and attribute the 25 ms.
*Gate: a per-cause ms/token breakdown. This decides whether Phase 1 alone gets most
of the win, or whether Phase 2 is worth its cost.*

**Phase 1 — the cheap wins, independently shippable.** Now the leading candidate,
since Phase 2 lost its best property.
- Expert-range-only, budgeted registration to enable `dma_paging`, killing the
  ~66 MB/token of admission memcpy. Registration costs ~20 s at load (measured
  ~2 GiB/s) and locks the pages, so it needs a budget and a clean fallback rather
  than the current all-or-nothing test against the whole mapping.
- Defer admission (victim scan + H2D) out of the missing layer: the miss is served
  by the CPU either way, so nothing about correctness needs the slot now.
- A/B `--expert-top-k 8` vs 10 for decode: cuts routed experts 20%, hitting fetch and
  CPU compute together. Needs a greedy-token quality check, not just tok/s.
- Move the `record_expert_access` decay sweep (`:1568`, O(layers x experts) every
  32768 ticks) off the critical path.
*Gate: interleaved A/B decode tok/s; greedy output unchanged for the paging change.*

**Phase 2 — the device slot table and pointer-free kernels.** Steps 1, 2 and 4 only
(step 3 is dead). The kernel computes resident experts' addresses in-kernel from
`slot_of`, so the host no longer builds or uploads pointer tables; the sync moves to
*after* the GPU expert work is queued and carries only the miss list, which the
kernel can write with an `atomicAdd` compaction. Keep the host path behind
`COLIBRI_DEVICE_DISPATCH=0` for A/B and rollback.
*Gate: greedy tokens bit-identical to the host path on the fixture AND on 48 real
tokens; ctest + pytest green; interleaved A/B decode tok/s recorded.*
*Only worth building if Phase 0b attributes real time to the table build and the
per-expert host loop.*

**Phase 3 — CUDA graphs.** Blocked today by the hc/PLE hooks not being capture-clean
and by QSA's host sync. Note Phase 0's finding removes one argument for this: since
the sync survives, a layer cannot be captured end to end. Lower the priority
accordingly, and recall [[decode-overhead-audit]] measured graphs-off as a wash on
dense.

## Expected arithmetic (revised after Phase 0)

The sync survives, so the target is no longer "remove the round-trip" but "make the
host work between round-trips negligible and let CPU and GPU expert work overlap".

Per layer the two expert paths are already comparable: 0.059 ms for 10 resident IQ1_S
experts on the GPU (measured above) against ~0.13-0.17 ms for the ~3-expert CPU miss
set. If the ~25 ms of host overhead goes away and those overlap, the token approaches
`48 x max(GPU, CPU) + non-MoE` ≈ 7 + 7.5 ≈ **15 ms, i.e. ~65 tok/s**. That still
clears 60 — but every bit of it now depends on Phase 0b's breakdown being addressable,
not on the dispatch redesign. **Phase 0b before Phase 2.**

Orthogonal multiplier, not in this plan: **MTP**. Decode is dominated by per-token
expert traffic, so verifying k drafted tokens in one batched pass amortizes it
directly. The released GGUF ships no draft block, so it needs a separate conversion —
that is the only reason it is out of scope, not the payoff.

## Critical files

- `native/src/v2_runtime.cpp` — decode MoE phase (`:16834-17280`), the event sync
  (`:16943`), residency map (`:734`), `select_expert_cache_slot` (`:1587`),
  paging registration (`:13780-13830`), `record_expert_access` (`:1539`)
- `native/include/colibri_v2_qwen_kernels.hpp` — IQ grouped macro (`:5795`),
  Gemma pinned kernels as the template (`:1306-1356`), `route_topk` (`:400`)
- `native/include/colibri_v2_native_kernels.hpp` — `bailing_route_rows` (`:1878`),
  the fully-on-device precedent
- `native/src/gpu_driver.cpp` — `colibri_gpu_route_topk` (`:1897`), kernel name table
- Prior art: `plans/prefill-pipeline.md` (how the rows path got route_wait <1% — by
  schedule depth, not transfer cleverness), `plans/concurrent-decode.md` (the same
  trick across sequences, 1.46x at N=2)
