# Decode device dispatch: removing the per-layer host round-trip

> **Goal**: qwen4exp decode from 26 tok/s toward 60. The token is not bandwidth-bound;
> it is bound by 48 serialized host round-trips, one per MoE layer.
> **Status 2026-08-27**: **Phase 1 SHIPPED — decode ~17 -> ~30 tok/s** by registering
> the expert tensors instead of the whole mapping. Phase 0 and Phase 0b RUN.
> **The plan's centrepiece is dead.**
> Phase 0 killed host-resident expert reads; Phase 0b then showed that the host work
> Phase 2 would remove is ~0.7 ms/token, while **95-97% of it is one thing the plan
> barely mentioned: the bundle staging memcpy**. Phase 2 is dropped. Phase 1 is the
> whole plan now. See "Phase 0b results".

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

## Phase 0b results (2026-08-27) — the 25 ms, attributed

`COLIBRI_MOE_PROFILE=1` (added in this commit), real UD-IQ1_S, two windows of the
same run:

| cause | ms/token (39 tok) | ms/token (89 tok) | share |
|---|---|---|---|
| **bundle staging memcpy** | **34.220** | **16.501** | **95.7-97.5%** |
| kernel launch + uploads | 0.451 | 0.335 | 1.3-1.9% |
| pointer table build | 0.172 | 0.138 | 0.5-0.8% |
| residency lookup | 0.143 | 0.139 | 0.4-0.8% |
| admission victim scan | 0.073 | 0.075 | 0.2-0.4% |
| access bookkeeping | 0.039 | 0.038 | 0.1-0.2% |
| role scales (mmap) | 0.009 | 0.009 | 0.0-0.1% |
| total attributed | 35.106 | 17.235 | |

480 routed experts/token (the earlier hit/miss counters undercount — they are
incremented in only some branches), **134-160 admissions/token**, and
**95-174 MiB/token memcpy'd** from the mmap into pinned staging at `:17061`.

**Consequences, in order of importance:**

1. **Phase 2 is worthless and is dropped.** The device slot table plus pointer-free
   grouped kernels would remove the pointer table build and part of the residency
   lookup: **~0.3 ms/token of a 42 ms token.** That is weeks of kernel work, a new
   device-side cache mirror, and a rewrite of the IQ grouped family, for 0.7%.
   Two rounds of measurement cost an afternoon and saved all of it.
2. **The disease is admission volume, not dispatch.** Every miss stages a ~1.24 MB
   expert bundle through a single-threaded host memcpy, ~140 times per token. The
   cache holds ~3200 slots and churns ~140 of them per token — it is thrashing, and
   [[qwen4exp-qsa]]'s recurrence measurement already said reuse is weak (only 18% of
   misses were predictable from recent tokens). We are paying 174 MiB/token of
   transfer to populate a cache whose contents largely do not survive to be reused.
3. **Two independent fixes, both cheap:**
   - *Stop copying*: DMA paging uploads straight from the registered mmap with no
     host memcpy (`:17039-17043`). Off here only because the heuristic tests the
     whole 68 GB mapping. This removes the 16-34 ms outright.
   - *Stop admitting so much*: `COLIBRI_EXPERT_CACHE_STRICT_ADMISSION=1` measures
     **27.42 / 27.36 tok/s against a 23.91 / 26.69 baseline** — about +8% and a large
     drop in run-to-run variance, from one environment variable. A miss runs on the
     CPU either way; admitting it is pure speculation about reuse.

   These compose: strict admission cuts how many bundles move, DMA paging makes the
   ones that do move cost no host time.

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

**Phase 1 — SHIPPED 2026-08-27.** Expert-range registration landed; see results
immediately below. Remaining Phase 1 items (top-k A/B, decay sweep) are unstarted
and the decay sweep is now known to be 0.04 ms/token, i.e. not worth doing.

### Phase 1 results

Registration is now scoped to the expert tensors and budgeted, instead of testing
the whole 68 GiB mapping against RAM. On the reference box it registers
**37.1 GiB covering all 48 expert layers in 22.15 s** and direct paging engages
where it never had before.

Host-serial MoE work per token, same profile as Phase 0b:

| cause | before | after |
|---|---|---|
| bundle staging memcpy | 16.5-34.2 ms | **0.000 ms** |
| kernel launch + uploads | 0.34-0.45 | 0.168-0.174 |
| pointer table build | 0.14-0.17 | 0.079-0.087 |
| residency lookup | 0.14 | 0.078-0.087 |
| admission victim scan | 0.07 | 0.060-0.062 |
| access bookkeeping | 0.04 | 0.035-0.038 |
| role scales (mmap) | 0.01 | 0.009-0.010 |
| **total** | **17.2-35.1 ms** | **0.43-0.46 ms** |

Interleaved decode A/B (48 iterations, 4 warmup), **on a warm, healthy page cache**:

| config | run 1 | run 2 |
|---|---|---|
| staged (old default) | 19.98 | 13.81 |
| **direct DMA** | **30.28** | **29.36** |
| direct DMA + strict admission | 29.77 | 30.59 |

**~17 tok/s -> ~30 tok/s**, and the run-to-run variance largely disappears (the
staged path's spread was page-cache state).

**CAVEAT, found later the same day and only partly resolved.** Those numbers were
taken with the page cache warm and the box otherwise idle. Repeating the comparison
after many back-to-back model loads, with RAM oversubscribed, told a different
story: both configurations collapsed to 7-8 tok/s, and the within-run split showed
pinning had *moved* the cost rather than removed it —

| paging | MoE phase | "other" (incl. PLE gathers) | major faults |
|---|---|---|---|
| staged | 116-119 ms | 5-9 ms | 42k-63k |
| auto (37 GiB pinned) | ~21 ms | 88-93 ms | 14k-97k |

Pinning all 37 GiB starved the 25.7 GiB n-gram table's page cache, so the memcpy
came back as major faults in the PLE gathers. The budget now subtracts the
host-resident tables it will never register (the n-gram table) on top of the OS
reserve, which on this box registers **19.1 GiB / 25 of 48 expert layers** instead
of everything. Partial coverage is exactly what the per-layer `expert_dma` flag was
built for.

**Then the corrected budget was benchmarked, and it does not win.** Interleaved,
warm cache, idle box:

| | staged | auto (19.1 GiB pinned) |
|---|---|---|
| run 1 | 25.87 | 23.10 |
| run 2 | 23.73 | 28.37 |
| run 3 | 26.51 | 21.71 |
| **mean** | **25.4** | **24.4** |

Inside the noise, with thousands of major faults per run either way. Two reasons,
and the second is the one that matters: only 25 of 48 layers are pinned so half the
admissions still stage, and **the working set does not fit RAM at all** — 39.8 GiB
of experts plus the 25.7 GiB n-gram table plus dense weights against 60 GiB. Page
pressure dominates however the pinning is arranged. The 17 -> 30 tok/s figure above
came from a page-cache state that flattered the pinned side; it is not reproducible
as a steady-state result on this box.

**So budgeted registration is now OPT-IN** (`--expert-paging direct`), not something
`auto` turns on. The all-fits path (`auto_direct`) is untouched and still automatic:
where RAM genuinely holds the checkpoint, pinning costs nothing and the staging
memcpy is pure loss. What changed is that a RAM-short box no longer locks 19 GiB to
buy nothing.

**What this says about the roof**: on *this* hardware the remaining decode cost is
not addressable by paging policy — it is that the model does not fit. The levers
that survive are the ones that shrink the working set or the traffic (a smaller
expert quant, `--expert-top-k`, MTP amortising per-token expert reads), not ones
that rearrange where the bytes are copied from.

**REVISED 2026-08-27, and the headline above is overstated.** Those arms compare
staged against direct *inside HEAD* at a moment when the page cache was in an
unusually bad state for the staged arm, so the 17 is too low. A later A/B against
the actual pre-session commit (5e8fefe), at 32k context with a warm cache, puts the
session's net decode win at **22.70 -> 27.24 tok/s, about +14-20%** -- real, but not
+77%. Quote the baseline comparison, not the staged-vs-direct one. Neither figure
has been re-measured since the registration budget narrowed to 19.1 GiB / 25 of 48
layers, so even +14-20% is provisional.

Two things worth noting:

- **Strict admission is now redundant.** It was worth +8% when every admission cost
  a host memcpy; with the memcpy gone it measures neutral (29.77/30.59 vs
  30.28/29.36). It was a workaround for the real problem, and the real problem is
  fixed. Do not make it a default.
- **The win is smaller than the 17 ms removed, because the bytes still move.**
  ~174 MiB/token of admissions now cross PCIe asynchronously instead of being
  memcpy'd first. That transfer is the next limit, so *reducing admission volume*
  (which strict admission attacks from the wrong end) is where the next lever is —
  but it must now be judged against PCIe cost, not host time.

**Trade-off RESOLVED — registration now runs in the background.** It used to cost
~22 s at prepare, breaking even only around 4000 generated tokens, which was wrong
for a one-shot `generate`. A worker now does the pinning while decode proceeds on
the staged path, and a token boundary promotes the layers once it lands. Measured
24-token `generate` wall time: **staged 23.6 / 21.3 s vs background 20.7 / 22.2 s**
— indistinguishable, where the synchronous version added ~22 s. Long runs still
reach 0.435-0.447 ms/token of host-serial MoE work and 27.7-28.9 tok/s.

**BUDGET CORRECTED, AND THE PHASE 1 NUMBERS NEED RE-VALIDATING.** The first budget
reserved a flat 8 GiB and pinned all 37.1 GiB of experts. A later run showed the
pinned config moving cost rather than removing it — MoE phase ~118 ms -> ~21 ms per
token but "other" ~6 ms -> ~90 ms, with tens of thousands of major faults — which
looks exactly like the 25.7 GiB n-gram table losing its page cache to the pin. The
budget now subtracts host-resident tables it will never register (the n-gram table)
on top of the OS reserve, so this box registers **19.1 GiB / 25 of 48 layers**
instead of all 48, and partial coverage keeps the rest staged.

*Caveat, stated plainly*: that diagnosis is **not confirmed**. The runs behind it
had a cold page cache (`buff/cache` was later 5 GiB with 49 GiB free), so both arms
were re-reading the checkpoint from NVMe and both measured 7-8 tok/s — nothing like
the 27-30 tok/s the earlier warm-cache runs showed. The corrected budget is the
conservative choice either way, but **the headline "17 -> 30 tok/s" was measured with
the old all-experts pin on a warm cache and has not been re-measured since.** Re-run
the A/B on a warmed cache before quoting it.

Three things this needed, each a small trap:

- `colibri_gpu_host_register` did not bind the CUDA primary context, unlike
  `colibri_gpu_alloc` and friends. On a worker thread that fails with
  `INVALID_CONTEXT`. Fixed in both register and unregister.
- **Promotion is all-or-nothing, not per layer, and that is forced.** The first
  attempt registered each layer separately, trimming spans inward so neighbouring
  layers could not claim a shared boundary page. It failed with
  `native hybrid MoE DMA cache upload failed`: **an upload whose source straddles
  the edge of a registration is rejected**, and per-tensor trimming puts an edge
  inside every tensor. The ranges must be merged into maximal spans (they collapse
  to roughly one per shard), so a layer cannot be promoted before the whole span
  covering it is pinned.
- `expert_dma` is written only by the decode thread, in `qwen_absorb_registration`
  at a token boundary. The worker publishes finished layers under a mutex and bumps
  an atomic counter; the decode thread drains it. That keeps the per-layer flags
  free of atomics (which `QwenLayerPlan` cannot hold — it is moved into
  `mtp_layer_plan`) and guarantees a layer never switches path mid-token.

### Remaining Phase 1 items
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

**Phase 2 — the device slot table and pointer-free kernels. DROPPED.**
Phase 0b measured the work it would remove at ~0.3 ms of a 42 ms token. Kept here
only so the next person does not re-derive it: *the reason Qwen downloads routing
is real, but what the download costs is not the download.*

**Phase 3 — CUDA graphs.** Blocked today by the hc/PLE hooks not being capture-clean
and by QSA's host sync. Note Phase 0's finding removes one argument for this: since
the sync survives, a layer cannot be captured end to end. Lower the priority
accordingly, and recall [[decode-overhead-audit]] measured graphs-off as a wash on
dense.

## Expected arithmetic (revised after Phase 0b)

Removing the staging memcpy takes the token from ~42 ms to roughly 25 ms on its own
(~40 tok/s), and strict admission is worth a further ~8% measured. Neither touches
the dispatch path. With both, plus the residual PCIe cost of whatever admissions
survive (asynchronous rather than host-blocking), **15-25 ms/token is 40-65 tok/s** —
the 60 roof is in range, from configuration and admission policy rather than from a
kernel rewrite.

The honest caveat: the fastest configuration measured so far (strict admission,
27.4 tok/s) is only +8%, because it reduces *how often* the memcpy happens rather
than removing it. The big number depends on DMA paging actually being enable-able
within this box's RAM, which is now Phase 1's real risk and the next thing to test.

## Server validation (2026-08-27) — the win does NOT show up under `serve`

Everything above was measured with `colibri-next benchmark`. The product is the
server, which goes through the cooperative engine, the prompt cache and the
multi-sequence path instead. Measured through the OpenAI streaming API against a
real `serve` process, 32k context, `--gpu-cache-mib 9000`, decode rate taken from
the server's own `colibri.decode_elapsed_seconds`:

| | baseline 5e8fefe | HEAD |
|---|---|---|
| short prompt, TTFT | 2.21 s | 2.19 s |
| short prompt, decode | 29.65 tok/s | 31.38 tok/s |
| ~3k prompt, TTFT | **33.08 s** | **32.76 s** |
| ~3k prompt, decode | 26.10 tok/s | 24.41 tok/s |
| 4 concurrent, decode | 31.17 tok/s | 32.67 tok/s |
| 4 concurrent, aggregate | 14.26 tok/s | 14.16 tok/s |

Both arms auto-fit the identical expert cache (1992 slots, 3355 MiB), so this is a
clean comparison. **It is a wash.** The +14-20% the CLI benchmark showed does not
reach a served request: the server's workload keeps the expert cache warm, so few
admissions happen, so removing the admission memcpy buys little. The benchmark
harness exercised exactly the case the fix helps and the server does not.

What the session actually delivers to the server, then: no ~22 s startup stall, the
QSA regression removed by making it opt-in, an attribution tool, and a dispatch
rewrite correctly not built. Throughput: neutral.

**And the number that matters is not decode.** TTFT for a ~3k-token prompt is
**~33 seconds**, identical on both arms and untouched by any of this. At ~30 tok/s
decode, a served request spends its first half-minute in prefill. That is the
server's real problem, and this whole plan was aimed at the wrong phase.

## What this exercise is worth remembering for

The plan was written around a dispatch redesign, and two rounds of measurement
retired it before a line of product code was written: Phase 0 killed the mechanism
(host-resident expert reads, 20-47x too slow), Phase 0b killed the motivation (the
host work it targeted is 0.7% of the token). The real cost was a `std::memcpy` on one
line that the original plan mentioned only in passing. Measure the gap before
designing for it.

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

---

# Prefill / TTFT (2026-08-27) — the phase that actually matters

The server measurement above put a ~3k-token request's TTFT at ~33 s against ~30
tok/s decode. Measured directly on a 4243-token prompt (`COLIBRI_TIMING=1`):

| | |
|---|---|
| prefill wall | **69.35 s — 61.2 tok/s** |
| expert phase | **91.7%** |
| route wait | 1.1% |
| remainder | 7.3% |
| expert cache hit rate | **21%** (decode gets 69%) |

Route wait being 1.1% is the prefill pipeline working exactly as
`plans/prefill-pipeline.md` claims. Everything else is expert execution.

**Why the hit rate is 21%.** `expert_cache_prompt_bypasses = 480`: the cache
deliberately refuses admissions during prompt processing, so ~79% of prefill's
expert work runs on the host. And `--prefill-cache-seed` — which did place 192
experts — is documented as pinning "the hottest prompt-routed experts **once the
prompt is done**, so decode does not fault them back in". The seeding is aimed at
decode and happens *after* prefill, which is precisely too late to help it.

**Configuration cannot reach it.** A sweep moves nothing:

| arm | prefill | expert phase | hit |
|---|---|---|---|
| default (split, seed auto) | 56.2 tok/s | 84.5% | 21.4% |
| `--hybrid-prefill cpu` | 55.1 tok/s | 84.3% | 21.4% |
| `--expert-mode hybrid` | 62.3 tok/s | 97.6% | 0.0% |
| `--gpu-cache-mib 10500` | 58.9 tok/s | 90.3% | 21.4% |

**The hypothesis to test next**, and it is a code change, not a flag: *admit or seed
during prefill rather than after it*. A 4243-token prompt is processed in 8 chunks;
chunk 1's routing already names most of the experts chunks 2-8 will want, and the
cache is being told to ignore exactly that signal. Unlike decode — where the
recurrence data showed only 18% of misses were predictable — a prompt re-routes the
same experts across its own chunks, so the predictor here is far better.

Note `prefill_streamed_bytes = 0`: the prefill expert-streaming arena
(`plans/prefill-expert-stream.md`, marked closed, default budget 0) never runs.
Whether the gather/GEMM/scatter path from `plans/prefill-expert-gemm.md` (810 tok/s
on Ornith) engages here at all is unverified and worth checking before anything is
designed — it is silently off whenever hybrid falls back to CPU MoE, which at a 21%
hit rate is most of the time.

## The override that made `--hybrid-prefill` a no-op (2026-08-27)

Chasing why the prefill expert-GEMM/streaming path never ran
(`prefill_streamed_bytes = 0` in every arm), a `COLIBRI_STREAM_TRACE` at its gate
showed the gate passing — and, unexpectedly, `hybrid_prefill_cpu=1` and
`routed_gpu_allowed=0` **even under `--hybrid-prefill split`**.

`src/colibri_next/v2.py`:

```python
effective_hybrid_prefill = ("cpu" if resolved_expert_mode == "auto" else hybrid_prefill)
```

Under the default `--expert-mode auto` this **replaced** the caller's choice rather
than defaulting it, so `--hybrid-prefill split` silently got `cpu` and every routed
expert ran on the host for the whole prompt. That is why the earlier sweep saw
`split` and `cpu` measure identically: they were the same configuration.

Measured cost, 4243-token prompt, two rounds each:

| | prefill |
|---|---|
| default (`auto` -> forced `cpu`) | 55.8 / 60.2 tok/s |
| `--expert-mode hybrid --hybrid-prefill split` | **63.6 / 65.0 tok/s** |

**~+11%**, and steadier. Fixed by giving the flag no parser default, so "not asked"
is distinguishable from an explicit choice; only the former is defaulted (`cpu`
under `auto`, `split` otherwise). The existing default behaviour is unchanged.

Note this does NOT explain the bulk of TTFT: the expert phase is still 97.5% of
prefill in the `split` arm. Getting routed experts onto the GPU during prompt
processing is worth ~11%, not the 2x that ~33 s TTFT needs. The phase is still the
target; this was a flag that lied about what it did.

## Why qwen3.5 prefills at 810 tok/s and qwen4exp at 61 (2026-08-27)

`plans/prefill-expert-gemm.md` is the reason for the qwen3.5-family number: the
grouped expert kernels are *decode-shaped* (one full-width matvec block per
(token, route), so a staged expert's weights are re-decoded once per routed
token), and the fix was gather -> MMQ GEMM -> scatter, decoding weights once per
128-token tile instead. That shipped and took Ornith to 810 tok/s (+31%).

qwen4exp never enters it. The engagement test is `stream_role_ok` in
`v2_mtp_verifier.inc:1640`, applied as a conjunction over gate, up AND down:

```cpp
if (format->matmul_q8_mmq && int8_tensor_cores && (in_size & 255) == 0) return true;
return format->matmul_rows != nullptr &&
       format->matmul_rows_grid != RowsMatmulGrid::per_token;
```

Against this checkpoint's formats (34 layers IQ1_S + 14 IQ2_XXS gate/up, **all 48
layers IQ4_NL down**):

| role | format | `matmul_q8_mmq` | `matmul_rows` | passes? |
|---|---|---|---|---|
| gate/up | IQ1_S (19) | `iq1s_q8_mmq` | `iq1s_matmul_rows`, quad_pack | yes |
| gate/up | IQ2_XXS (16) | `iq2xxs_q8_mmq` | `iq2xxs_matmul_rows`, quad_pack | yes |
| **down** | **IQ4_NL (20)** | **none** | **none** | **NO** |

The IQ4_NL entry carries only `iq_expert_prefix` and `cpu_expert` — the
decode-shaped grouped kernel and a CPU dot, nothing batched. One missing format
entry fails the conjunction for every layer, so prefill falls back to exactly the
wall `prefill-expert-gemm.md` was written to remove. That, not policy or budget,
is why `prefill_streamed_bytes` is 0 with the gate reporting true.

**The work**: give IQ4_NL a rows matmul. Note the MMQ branch is unavailable to it
regardless — the expert intermediate is 640 and `640 & 255 != 0` — so what is
needed is `iq4nl_matmul_rows` with a non-`per_token` grid, the shape IQ1_S/IQ2_XXS/
IQ4_XS already have. IQ4_NL is block-32 rather than 256-super-block (see the %32
admission in [[qwen4exp-support]]), so it cannot copy IQ4_XS's kernel verbatim.

Expected payoff is the *shape* change, not a constant: weight traffic per expert
falls from `n_tokens x triple` to `ceil(n_tokens/128) x triple`. Prefill is 97.5%
expert phase, so this is the only lever measured so far that could plausibly move
a 33 s TTFT by more than 11%.

### IQ4_NL rows matmul shipped — the path engages, and does not pay here

`iq4nl_matmul_rows` added (`COLIBRI_LOWBIT_MATMUL_ROWS(iq4nl_matmul_rows,
iq4nl_value)`), format entry wired, registered in the driver name table. Verified
against the independent host decoder by the IQ kernel contract at 1.3e-08.

Two traps on the way, both caught by that contract rather than by a benchmark:

- `iq4nl_value` cannot be derived from `iq4nl_octet`. IQ4_NL's nibble order is
  Q4_0-style (byte j holds element j low, j+16 high); the octet decoder walks the
  same bytes in a different traversal, and copying its indexing gave a plausible,
  wrong kernel.
- The contract harness hardcoded 256-element blocks in **two** places
  (`check_rows`'s packed sizing and `worst_error`'s row stride). IQ4_NL is a flat
  32, so the first failure was the *reference*, not the kernel. `Format` now
  carries `block_elements`, which any future non-256 format needs too.

`prefill_streamed_bytes` goes from 0 to non-zero: the path engages. Arena sweep on
a 4243-token prompt:

| arena | prefill | streamed | expert phase |
|---|---|---|---|
| 0 (off) | 53.2 tok/s | 0 | 84.0% |
| 48 (auto default) | 55.0 | 17.8 GiB | 74.4% |
| 512 | **59.5** | 133.8 GiB | 18.1% |
| 2048 | 51.6 | 184.2 GiB | 1.2% |

**The GEMM does exactly what it promises and it still does not solve this model.**
Expert compute collapses 84% -> 1.2%; throughput peaks at +12% and then regresses,
because the cost moves wholesale into staging traffic. 184 GiB moved for one 4243-
token prompt is ~2.2 GB/s effective against a 26 GB/s link — thrash, not bandwidth.

The reason is the model's routing shape. 512 experts at top-10 over 1024-row chunks
touches nearly every expert in every layer of every chunk, so a per-span arena has
almost no reuse to amortize: each chunk re-uploads what the last one used. Ornith
wins from this path because far fewer experts carry far more tokens each. Here the
per-expert token count is ~20, so a staged triple is read about 20 times before
being evicted — against ~128 needed to break even on the decode-shaped kernel it
replaces.

Default left at auto/48. 512 MiB is the measured optimum on this box but buys 12%
for 134 GiB of PCIe traffic, which is a bad trade on a machine also serving.

**What this says about prefill**: the expert phase is reachable — it went to 1.2% —
but only by paying more in weight movement than it saves in compute. Any real fix
has to cut weight *movement*, not decode cost: fewer experts touched per chunk
(smaller chunks trade the same way), or experts that stay resident across chunks
rather than per-span staging.

### AVX-512 for IQ4_NL: measured neutral, and why that matters

Added `iq4nl_dot` to `qwen_cpu_avx512.cpp` (Q4_0's structure, with
`_mm_shuffle_epi8` doing the 16-entry codebook lookup) and a type-20 admission
ahead of the AVX2 one. Contract-verified; the gate defaults on, so the branch is
genuinely taken.

| | prefill |
|---|---|
| `COLIBRI_IQ_AVX512=0` (AVX2) | 58.0 / 58.5 tok/s |
| AVX-512 | 56.9 / 58.2 tok/s |

**No gain.** Vector width is not the constraint, and the reason points at the
actual one. The down projection is `[640 -> 2560]`: 2560 output rows of only 640
elements each. At 32 elements per block that is a 20-iteration loop per dot,
after which `_mm512_reduce_add_ps` costs a log2(16) shuffle chain — so a short
dot is reduction-bound and a wider register buys nothing. AVX2's 8-wide
horizontal sum is proportionally cheaper.

What is missing is **register blocking across output rows**, so one pass over the
input feeds several rows and amortises both the loads and the reduction:

```cpp
constexpr bool qwen_simd_multi_type(std::uint32_t type) {
    return type == 8 || type == 12 || type == 13 || type == 14;   // K-quants only
}
```

The rows MoE calls `qwen_quant_dot_two_rows` / `_pair` / `_oct` in its inner loop,
but every IQ type falls off that allowlist and reaches SIMD "one row at a time
through the single-row fallback" — its own comment. The K-quants get the blocked
kernels; IQ1_S, IQ2_XXS and IQ4_NL, which is the entire expert set of this
checkpoint, do not.

**That is the next lever, and it is better targeted than vector width**: it
attacks call overhead and reduction tails on 2560 short dots per expert per token
batch, which is exactly the shape prefill spends its time in. Wide SIMD only helps
the 2560-long gate/up dots, where the loop already amortises the tail.

The AVX-512 kernel is kept: correct, contract-pinned, neutral here, and plausibly
positive wherever IQ4_NL is a wide projection rather than a 640-wide expert down.
It is NOT evidence that AVX-512 helps this model.

## Prefill, finally measured rather than inferred (2026-08-27)

Three hypotheses about the streaming path were killed by direct test, in order:

1. **Transfer shape.** An isolated upload benchmark from registered host memory
   hits **26 GB/s at every granularity** — 540 KiB, 1.6 MiB, 16 MiB, 256 MiB — and
   **26 GB/s with an event record + cross-stream wait after every copy**. Neither
   small transfers nor fencing costs anything.
2. **Source residency.** During a full-staging prefill that streamed 198.6 GB, the
   host read only **25.8 GB from disk** (13%, 0.45 GB/s). The mmap pages are
   overwhelmingly resident; the uploads are not secretly disk reads.
3. **My own "~2.9 GB/s upload rate".** That was arithmetic on cross-run deltas with
   a guessed constant baseline, and it was wrong. It is recorded here because it
   drove two proposals before anyone measured the thing directly.

Direct instrumentation (`COLIBRI_MOE_PROFILE=1` now reports the streaming split,
and prints at exit so a short probe still gets it), 4243-token prompt, 2048 MiB
arena, 55.6 s total:

| phase | time | note |
|---|---|---|
| non-expert | ~22 s | attention, DeltaNet, dense, hyper-connections, PLE |
| expert uploads | **11.75 s** | 198 GB at **16.9 GB/s** — near link speed |
| expert group compute | **21.96 s** | **122,848 groups, 16.6 tokens each** |

**What this establishes.**

- Uploads are close to the hardware and are not worth optimising.
- The expert cost is not bandwidth, not disk, not vector width, not row blocking.
  It is **122,848 per-expert-group dispatches of ~17 tokens each**, ~179 us per
  group, each doing gather -> quantize -> gate MMQ -> up MMQ -> silu_mul ->
  quantize -> down MMQ -> scatter. About a million launches for one prompt.
- **Non-expert work is ~40% of prefill and has never been looked at.** At full
  staging it is the same size as the expert phase.

**The fix this points to** is the one vLLM and llama.cpp already use, and it is
structural rather than a tuning knob: *one* kernel over a sorted, block-padded
token buffer, each CTA reading its expert id from a per-block table, instead of a
host loop issuing eight launches per expert. `moe_align_block_size` +
`fused_moe_kernel` is exactly this; llama.cpp's `mul_mat_id` fast path groups rows
by expert into one batched GEMM. With 512 fine-grained experts at top-10 the
per-group dispatch overhead is the whole cost, so collapsing 123k dispatches into
one launch per layer-role is the change worth making.

Second target, unexamined: the ~22 s of non-expert prefill.

## The non-expert half of prefill (2026-08-27)

`COLIBRI_PREFILL_TRACE=2` with `COLIBRI_PREFILL_PIPELINE=0`, 4243-token prompt.
**Read the shares, not the totals** — the tracer syncs the stream at every marker.

| phase | seconds | share | layers |
|---|---|---|---|
| pre | 39.56 | 72.1% | all |
| **proj** (DeltaNet qkv) | **8.23** | 15.0% | delta |
| moerms | 2.25 | 4.1% | all |
| outproj | 1.59 | 2.9% | delta |
| router | 1.15 | 2.1% | all |
| recurrent | 1.02 | 1.9% | delta |
| qkvproj (attention) | 0.50 | 0.9% | attn |
| ropestore | 0.37 | 0.7% | attn |
| conv | 0.19 | 0.4% | delta |

**First: `pre` is not a non-expert phase.** `phase()` reports time since the
previous marker, and the last marker of a layer is `router` — so `pre` spans the
preceding layer's entire MoE expert execution plus this layer's bookend. Its 72%
is mostly the expert phase already accounted for elsewhere. Anyone reading this
trace should not conclude the hyper-connection bookend costs 40 s.

Genuinely non-expert work is the other ~15.3 s, and `proj` dominates it.

**The number that matters is not the ranking, it is the rate.** `attn_qkv` is
Q5_K, `[2560 -> 10240]`, on 36 layers: 4 TMAC = 8 TFLOP for this prompt, done in
8.23 s = **~970 GFLOP/s**. The streamed expert GEMM independently measures
20 TFLOP in 21.96 s = **~910 GFLOP/s**. Two unrelated code paths, one dense and
one MoE, land within 7% of each other.

**Unifying hypothesis** (NOT yet confirmed): quantized matmul on this GPU runs at
~1 TFLOP/s, and that single rate sets prefill. Total prefill arithmetic is roughly
20 TFLOP of expert plus ~10 TFLOP of dense = ~30 TFLOP; at 1 TFLOP/s that is ~30 s
against the 55-70 s observed, with the remainder in CPU experts and overheads. If
true it supersedes every explanation in this document: not dispatch count, not
upload bandwidth, not vector width, not the fine-grained MoE shape — those all sit
on top of a matmul that is ~5x slower than this class of GPU should manage for
K-quant MMQ.

`int8_tensor_cores` is enabled (the probe requires compute >= 7.5; this card is
well past it) and `2560 % 256 == 0`, so Q5_K should be taking `q5k_q8_mmq`. Either
it is not, or MMQ itself is underperforming here.

**Next test, and it must come before any code**: a standalone benchmark of
`q5k_q8_mmq` against `q5k_matmul_rows` at the real shape (1024 x 2560 -> 10240),
plus a check of which kernel `dense_rows` actually selects. There is no env toggle
for MMQ today, so the dispatch needs a trace like `COLIBRI_STREAM_TRACE`. That
distinguishes "MMQ is off" (a wiring bug, cheap fix, precedent: IQ4_NL and
`--hybrid-prefill`) from "MMQ is on and slow" (a kernel problem, expensive).

## Prefill, resolved (2026-08-27)

Two hypotheses from this file are now FALSIFIED by direct per-kernel measurement,
both of them mine, and both from the same mistake.

**Dead: "quantized matmul runs at ~1 TFLOP/s and that sets prefill."** It came from
the phase tracer putting `proj` at 8.23 s. But `phase()` syncs the stream before
reading the clock, so `proj`'s window charges that phase for whatever was still
queued from the previous layer -- the same contamination already noted for `pre`.
The tracer is fine for ordering and useless for rates.

**Dead: "the kernel is 20x slower in situ than standalone."** Same cause.

`COLIBRI_PREFILL_LAUNCH_TIME=1` now brackets every rows-path launch with CUDA
events, so each kernel's own duration is measured. (Wall time under it is
meaningless -- everything is serialized -- but a kernel's duration is not affected
by serialization, which is the whole point.) On a 784-token prompt:

| | time |
|---|---|
| **every GPU kernel combined** | **6.4 s** |
| **CPU expert phase** | **17.63 s** |

Top GPU kernels: `q5k_q8_mmq` 1.57 s (24.5%, 848 calls = **1.85 ms/call**),
`q8_matmul_tiled` 0.87 s, `qwen_f32_matmul_rows` 0.79 s, `qwen4_hc_mix_rows`
0.41 s, `quantize_q8_blocks_rows` 0.38 s, `qwen4_group_rms_rows` 0.35 s,
`iq4nl_matmul_rows` 0.28 s. A standalone benchmark of `q5k_q8_mmq` at
`[1024x2560]x[2560x10240]` gives **19.06 TFLOP/s** / 2.82 ms, consistent with the
in-situ per-call figure once shapes are accounted. And `COLIBRI_DENSE_TRACE=1`
confirms every dense projection takes MMQ: `q5k_q8_mmq`, `q6k_q8_mmq`.

**The GPU is not the problem, at all.** Prefill is bound by CPU expert execution,
which is ~2.75x the entire GPU kernel budget. That is the same conclusion the very
first measurement reached (expert phase 84-92% of prefill) and every subsequent
theory -- weight re-reads, upload bandwidth, dispatch count, vector width, matmul
throughput -- was a detour off it.

**Which makes the ranking clear.** Getting experts off the CPU is the only lever
that matters, and staging already does it: at a 2048 MiB arena the expert phase
falls to 1.2% and prefill goes 69 s -> 55.6 s, with the remaining budget split
uploads 11.75 s / group compute 21.96 s / non-expert ~22 s. So:

1. **Raise the staged fraction** -- it is the measured win, and the default arena
   (48 MiB) captures almost none of it. The blocker is that a per-span arena
   re-uploads every chunk; experts persisting across chunks would amortise it ~4x
   on a 4k prompt.
2. **Then** the 122,848 tiny per-expert-group dispatches, via the sorted-buffer
   fused kernel vLLM and llama.cpp use.
3. Non-expert GPU work is ~6.4 s of kernels and is not worth touching.

## CORRECTION (2026-08-28, later): that bottleneck is the STAGING regime only

Everything in the section below was measured with a **full 2048 MiB staging arena**,
which is not the default — expert streaming defaults off. Under the shipped default
the picture is completely different, and the dispatch count is not the problem.

**Wall-clock phase split, 4158-token prefill, default config** (`COLIBRI_TIMING=1`
+ `COLIBRI_PREFILL_PROFILE=1`, 8 prefill calls):

| phase | seconds | share |
|---|---|---|
| **CPU expert phase** | **44.25** | **96.3%** |
| route wait | 0.74 | 1.6% |
| GPU core | 0.41 | 0.9% |
| unattributed | 0.54 | 1.2% |
| total | 45.94 | 90.5 tok/s |

The GPU does ~1% of prefill. Dispatch count, kernel shape and upload bandwidth are
all irrelevant here. **~65-70% of routed experts run on the CPU**, because the GPU
expert cache hits only 30-35% during prefill (against ~69% in decode), and the CPU
is ~8x slower per byte (51 vs 391 GB/s).

Why the cache cannot fix itself: a 4158-token prefill issues ~41,580 routes spread
over essentially all 24,576 expert instances, while the cache holds ~3,226 slots
(13%). Prefill touches nearly everything once; there is no reuse for an LRU to
exploit. Decode is the opposite — 480 routes per token with strong recurrence.

**Three hypotheses falsified, in order, each by measurement:**
1. *Dispatch count* — that is the staging regime, not the default (see split above).
2. *Direct-quant dots* (`COLIBRI_PREFILL_DIRECT_QUANT=1`, which lifts the Laguna-only
   gate): **2x SLOWER** — 45.8/43.8 tok/s against 89.0/84.6. Identical output. The
   existing dequant-then-GEMM decodes each weight row once and GEMMs it against all
   ~20 tokens routed to that expert; the direct path re-decodes per 8-token tile.
   The gate is correct and should stay.
3. *Bigger chunks to amortise dequant further* — flat: 89.7 / 90.5 / 89.2 tok/s at
   `COLIBRI_PREFILL_ROWS` 1024 / 2048 / 4096. Dequant is already amortised as far as
   chunking takes it.

**Lesson: phase split before mechanism.** All three cost a measurement that reading
the wall-clock split first would have avoided.

**Open correctness flag**: chunk size changed the first generated token (1024 ->
`201`, 2048/4096 -> `248046` = EOS). The prompt was synthetic near-tied garbage and
[[qwen4exp-support]] already records batch-vs-single CPU-MoE summation drift on this
quant, so it may be benign — but prefill should not be that chunk-sensitive, and
this was not run on real text. Check before treating `COLIBRI_PREFILL_ROWS` as a
tuning knob.

## RESOLVED (2026-08-28): one weight type had no vectorized row decoder

**Prefill 50.4 -> 156.5 tok/s at 2048 tokens, 36.5 -> 140.1 at 256** (100% of
experts on CPU, the same configuration llama.cpp's `-ncmoe 48` measures). The
whole gap below was `IQ2_XXS` falling through `qwen_dequant_row` to the
per-element scalar form. The section that follows it — the q8-activation
integer-dot theory — is **falsified**; keep reading only for the record.

**How it was found, in the order that worked.** The mechanism was never guessed:

1. *Isolate the kernel.* A standalone benchmark at the real shape (640x2560
   IQ1_S, 2560x640 IQ4_NL, 20 routed tokens) put dequant+f32-GEMM at **55-80
   GMAC/s per core**, with dequant only **0.28-0.69x** the GEMM it feeds — not
   the 3.75x the in-situ counters reported. A genuinely DRAM-cold working set
   (1.2 GiB, ~50x L3) changed it by 4%, so weight-read latency was not it.
2. *Account for I/O and threads.* `ru_majflt`, `ru_minflt` and
   `/proc/self/io` across the timed prefill: **zero major faults, zero block
   reads, 15.9 of 16 cores busy.** Not paging, not thread starvation, not the
   68 GB model against 60 GB of RAM.
3. *Reproduce the loop.* The same kernels in the same CSR-grouped, 4-row,
   `dynamic,4` loop at production buffer sizes hit **598 GMAC/s on 16 threads**
   with dequant/gemm at 0.76 — 5x what production delivered. Same kernel, same
   schedule, same shapes: whatever was wrong was not in any of them.
4. *Ask what production has that the reproduction does not.* Mixed weight
   types. `qwen_simd_quant_type` is `{8,10,11,12,13,14}`; types 19 and 20 got
   explicit AVX2 admissions when qwen4exp landed. **16 got none**, and the
   unsloth UD mix puts IQ2_XXS on the gate and up of **14 of 48 layers** (5.64
   GiB of 37 GiB of expert weights). Those layers re-derived a weight's block,
   group, scale, sign byte and grid entry **per element**.

**The fix**: `iq2xxs_dequant`, the store form of the `iq2xxs_dot` that was
already there, plus the admission. Bit-exact against `qwen_iq2xxs_value` —
the sign is an XOR of a magnitude and the scale multiply is in the same order,
so the contract's element-for-element check is exact, not approximate.

Effect on the in-situ counters, per gate task: **50.6 us -> 4.78 us**. The phase
split now reads gate{dequant 115.3s gemm 156.6s} against gate{dequant 893.9s
gemm 100.2s} — a dequant/gemm ratio of 0.74, which is what the isolated
benchmark said it should be all along.

**Why it hid for two sessions.** Every symptom pointed somewhere else. The
scalar path is *correct*, so nothing failed. It is spread across a third of the
layers, so no single tensor looked wrong. And the in-situ profile attributed the
cost to "dequant", which read as "decoding is inherently expensive, quantize the
activations instead" — the theory in the next section — rather than "this
format is not using the decoder that exists". The isolated kernel measurement
is what separated the two, and it should have come first.

**Two things now guard it.**
- `g_scalar_dequant_types` (v2_runtime.cpp): the scalar tail records the type,
  and the batched MoE prints one line naming the offenders. Verified by
  disabling the new admission and watching it fire, then restoring it.
  A configuration that lands there now says so instead of just being slow.
- `native/tools/bench_moe_cpu_layer.cpp`: the batched CPU MoE at production
  shape and threading, per weight type. It prints dequant/gemm per format; a
  format on the scalar path stands out immediately. Post-fix it reads IQ1_S
  0.72, IQ2_XXS 0.36.

**The same omission, two formats over — measured and fixed.** Types 17 (IQ2_XS)
and 18 (IQ3_XXS) had SIMD dots and q8-K integer dots but no store-form dequant
either. `colibri_qwen_moe_layer_bench` at the real shape, before and after:

| format | before | after | dequant/gemm |
|---|---|---|---|
| IQ2_XS (17) | 44.5 GMAC/s | **626.4** | 27.9 -> 0.58 |
| IQ3_XXS (18) | 43.6 GMAC/s | **649.7** | 28.0 -> 0.59 |

**14-15x**, worse than the IQ2_XXS case that started this. `iq2xs_dequant` and
`iq3xxs_dequant` are the store forms of the dots directly above them, pinned
element-for-element by the contract. All four IQ expert formats now sit at
0.46-0.84.

Not validated end to end: no checkpoint on this box has 17/18 *expert* stacks
(the local IQ2_XXS/IQ3_XXS models are dense). The exactness is contract-proven
and the speedup is measured at the real shape, but the first model that runs
this should be checked. That layout is Laguna's (gate/up 17, down 18), which
also makes `COLIBRI_PREFILL_DIRECT_QUANT` worth re-measuring there: the gate
exists because dequant-then-GEMM was the scalar path, and it no longer is.

### The split after the fix, and why the CPU path is now done

Re-measured immediately, because the split that justifies a piece of work is
stale the moment that work lands. 2048 tokens, all experts on CPU, 166 tok/s:

| phase | seconds | share |
|---|---|---|
| CPU expert phase | 11.89 | **96.4%** |
| route wait | 0.03 | 0.2% |
| unattributed (incl. all GPU work) | 0.41 | 3.4% |

Still 96% CPU expert — the same share as before, three times faster in absolute
terms. Inside it (thread-seconds): **GEMM 52%**, dequant 36%, down store 8.8%,
activate 2.5%. The f32 GEMM is now the top term.

**And it is close to the metal.** Production runs the expert phase at ~392
GMAC/s where the isolated harness reaches 599; that 1.5x is accounted for, not
a defect:

- **Sustained thermals, -19%.** `colibri_qwen_moe_layer_bench 1024 16 25` holds
  16 cores in AVX-512 for 25 s: 665 GMAC/s on the first pass, ~540 by pass 418.
  A 50 ms benchmark reads the clocks at boost; a prefill does not.
- **File-backed pages, -9%.** Same bytes, page-cache resident either way,
  anonymous (THP, 2 MiB) against a file mapping (4 KiB): 636.7 vs 582.3
  GMAC/s. Real, and much smaller than the TLB argument predicts.
- 665 x 0.81 x 0.91 = 490 against production's ~413 leaves ~1.2x, inside the
  error of splitting production's counters between gate+up and down.

Single-core the f32 GEMM already runs at **64% of this machine's AVX-512 FMA
peak**. There is perhaps 1.2x left in the CPU expert path, not 4x. **Further
CPU kernel work on this path is not where the remaining prefill time is.**

**Where it is instead.** With the CPU side at its limit, prefill is bounded by
how much of the expert work can leave the CPU at all, and today almost none
can: the expert cache holds **2825 slots against 24,576 expert instances**
(48 layers x 512), ~11% of a chunk's working set, and prefill touches nearly
all of it once. `prefill_cache_seed=auto` is worth **+4-5%** and no more —
measured interleaved, three paired rounds (off 184.3/169.1/164.9, auto
186.8/177.9/174.9), auto ahead every round. A single unpaired run showed +19%,
which is this box's clock drift, not a result. Default left alone.

So the bulk can only move by **streaming experts per chunk**, whose blocker is
the ~150k tiny dispatches. Rough ceiling for that route: 37 GiB of expert
weights per 1024-row chunk at ~26 GB/s H2D is ~1.4 s per chunk against the CPU
phase's ~5.4 s, so **~2-3x on prefill**, upload-bound.

The block-major fused kernels built for exactly this were then timed, and they
do not deliver it — see "STOP: step 3 must not be wired" at the end of this
document. The prize above stands; the kernel that collects it does not exist
yet.

## FALSIFIED: colibri's CPU expert path is several times slower than llama.cpp's

**This section's conclusion was wrong.** Its measurements were sound and its
inference was not: it compared configurations end to end and concluded the
*kernel* was the gap, without ever measuring the kernel. See the section above.
The q8-activation prototype was built and measured anyway, at the real shape,
against the path it would replace:

| shape | dequant+f32 GEMM | q8 activations + integer dot |
|---|---|---|
| IQ1_S 640x2560, 20 tokens | **58.4 GMAC/s** | 37.9 (0.65x) |
| IQ4_NL 2560x640, 20 tokens | **80.1 GMAC/s** | 60.5 (0.76x) |

Output agreed to 0.5-0.7% relative, so the arithmetic was right; it is just
slower. Both forms accumulate per token in vector floats with no per-block
horizontal reduction, so this is not a strawman — the first cut was 0.38x and
was fixed before drawing the comparison. The reason is that dequant-then-GEMM
already decodes a row once and amortizes it over all ~20 routed tokens through
a register-blocked f32 GEMM, while the integer form must re-apply signs and
scales per token. **The mechanism llama.cpp uses is not the mechanism that makes
llama.cpp faster here.**

The original reasoning is kept below because the measurements in it are real and
the cross-check is still the right instinct — only the conclusion drawn from it
was unfounded.

The cross-check that localises it. Same GGUF, same box, `llama-bench`:

| engine | expert placement | pp512 |
|---|---|---|
| llama.cpp `-ncmoe 48` | **100%** of experts on CPU | **196.5 tok/s** |
| llama.cpp `-ncmoe 44` | 4 layers GPU, 44 on CPU | 367.9 tok/s |
| colibri (default) | ~30-35% GPU, 65-70% CPU | ~45 tok/s at 591 tokens |

llama.cpp doing **more** CPU expert work than colibri is still ~4x faster. So the
gap is not expert placement and not the GPU path — **it is the CPU expert kernel
itself**, and that is where prefill work should go next.

The likely mechanism, and it matches what [[qwen4exp-support]] already flagged
("the remaining 98% of prefill is the f32 dot_multi GEMM over decoded rows —
quantized dot_multi templates for 19/20 are the next lever"):

- colibri decodes IQ1_S/IQ2_XXS/IQ4_NL weight rows to **f32** and runs an f32 GEMM.
  That is a 2-4x byte expansion plus float math, and the measured
  `gate{dequant=446s gemm=119s}` says the decode alone is 3.75x the GEMM it feeds.
- llama.cpp quantizes the **activations** to Q8_K once per chunk and does **integer
  dot products** straight against the packed weights (`ggml_vec_dot_iq1_s_q8_K` and
  friends, VNNI/AVX2). No weight decode at all.

So the target is a third path, distinct from both existing ones: **quantize the
chunk's activations to Q8 once, then int8-dot against the packed expert weights.**
The GPU side of the runtime already does exactly this (`stream_quantize`,
`iq1s_q8_mmq`); the CPU MoE never got it.

*Before building it*: measure a single CPU expert matmul in isolation, colibri's
dequant+GEMM against a q8-activation int-dot prototype, at the real shape
(2560x640, ~20 tokens). The end-to-end numbers above compare configurations with
different expert placement, so they bound the opportunity but do not prove the
mechanism. Do not repeat this session's mistake of designing from an inferred cause.

> That instruction was followed, and it is the reason this section is now
> marked falsified: the isolated measurement showed the prototype **losing**,
> and the same isolated number is what exposed the real cause. It was the right
> instruction. It just needed to come one session earlier.

## The prefill bottleneck, established (2026-08-28)

Measured with `COLIBRI_PREFILL_LAUNCH_TIME=1`, 784-token prompt, at the two ends
of the staging range:

| | default arena (48 MiB) | full staging (2048 MiB) |
|---|---|---|
| CPU expert phase | **17.63 s** | **0.26 s** |
| GPU kernel time | 6.4 s | 14.7 s |
| launches | 24,919 | **153,440** |

**Staging solves the CPU problem outright** — the expert phase goes to 0.26 s. It
buys that by moving the work onto the GPU as ~150k dispatches, and *that* is the
new limit: `iq1s_q8_mmq` alone is 27,942 calls at **5 blocks per call**. A 5-block
kernel cannot fill this GPU; the 66 us the events attribute to each call is
mostly launch latency, not arithmetic.

Confirmed by a chunk-size test rather than assumed. Doubling `COLIBRI_PREFILL_ROWS`
1024 -> 2048 at a 2048 MiB arena cuts uploads 184.6 -> 145.6 GiB and groups
122,333 -> 96,472, both -21%, and moves prefill 66.6 -> 68.2 tok/s: **+2.4%**.
Reducing dispatches proportionally reduces time proportionally, which is what a
launch-bound regime looks like. 21% is not enough; the gap needs ~100x.

**Why neither existing path can close it.** The two GPU expert paths trade the same
pair of costs in opposite directions:

- *Table path* (`*_grouped_swiglu_rows`): ONE launch for all routes, grid
  `(out_size, rows*top_k)` — but one block per *(token, route)*, so an expert's
  weights are re-read once per routed token. Decode-shaped.
- *Streaming path*: weights read once per expert — but a host loop issuing ~8
  launches per expert, hence the 150k.

Neither is "one launch, weights read once per expert-block". That is precisely what
vLLM's `moe_align_block_size` + `fused_moe_kernel` and llama.cpp's grouped
`mul_mat_id` are: sort tokens by expert into a block-padded buffer, then a single
kernel whoseCTA reads its expert id from a per-block table.

**Scope of the remaining work**, in order:
1. A sort/align step producing `sorted_token_ids`, `expert_ids` (one per block) and
   the padded count — self-contained and unit-testable against a host reference.
2. A fused MoE kernel over that buffer for the three IQ formats, replacing both the
   per-expert loop and the per-route grouped kernel.
3. Rewire the rows path to call it once per layer instead of per expert.

Expected: 153,440 launches -> a few hundred, with weights still read once per
expert. Everything measured this session says that is the last big factor; uploads
(near link speed) and CPU experts (already ~0 under staging) are not.

### Step 1 shipped: the block-aligned route layout

`native/include/colibri_v2_moe_align.hpp` — routes grouped by expert and padded to
a block multiple, the layout a single fused kernel walks. Host reference and
production builder in one small header, pinned by
`native/tests/qwen_moe_align_contract.cpp` (ctest is now 22).

The contract checks the properties a consumer actually depends on, over
randomised batches at the real shape rather than hand-written cases: every kept
route present exactly once, all routes in a block belong to that block's expert,
padding only ever at the tail of a run, an expert's blocks contiguous, and
determinism. A lost or duplicated route is invisible in a small example and
produces fluent-wrong output downstream, which is why it is randomised.

Zero-weight routes are skipped — the rows path uses a zero weight to mark a route
already claimed by the GPU expert cache or pruned by top-k/top-p.

**Block size, measured at the real shape** (1024 rows, top-10, 512 experts; all
512 touched, mean 20 routes each, max 36):

| block | blocks/layer | padding waste |
|---|---|---|
| 4 | 2750 | 6.9% |
| **8** | **1495** | **14.4%** |
| 16 | 904 | 29.2% |
| 32 | 513 | 37.6% |
| 64 | 512 | 68.8% |
| 128 | 512 | 84.4% |

Past 32 every expert takes exactly one block (max count is 36) and the padding
runs away. **8-16 is the usable range.**

**This has a consequence for step 2 that was not obvious before measuring it: the
existing MMQ kernels are the wrong shape for this model.** They tile 16 or 64
tokens (`kQ8MmqTokens=128`, `kQ8MmqMinTokens=64`), sized for dense projections
where every token participates. An expert here has ~20 routes, so a 64-token tile
is 69% padding. The fused kernel needs its own geometry -- closer to a warp per
8-token block -- rather than reusing `*_q8_mmq`.

The prize is unchanged and now precise: today the streaming path issues ~4096
launches per layer-chunk at **5 blocks each**; this layout is **one launch of
~1495 blocks**.

### Step 2 shipped: the fused block-major MoE kernel

`iq1s_block_swiglu`, `iq4nl_block_swiglu`, `iq2xxs_block_swiglu` — one CUDA block
per (output row, route-block), decoding an expert's weight row once and dotting it
against up to `kBlockSize`=8 tokens that share that expert. Registered in the
driver name table; CPU twins generated from the same source as always.

Verified against `*_grouped_swiglu_rows` — the route-major kernel it replaces —
on the same routing: **worst 0.000e+00**, i.e. bit-identical. That is the right
reference precisely because the arithmetic is identical and only the traversal
differs, so any disagreement is a traversal bug (a block reading the wrong
expert's weights, a padded slot overwriting a real one) rather than a numerics
question.

Weight traffic, at the real per-layer shape (640 output rows, 10240 routes over
512 experts, 1495 blocks at kBlockSize 8):

| path | launches/layer | weight-row reads |
|---|---|---|
| route-major grouped | 1 | 640 x 10240 = 6.55M |
| streaming per-expert | ~4096 | 640 x 512 = 0.33M |
| **block-major fused** | **1** | **640 x 1495 = 0.96M** |

So it keeps the grouped path's single launch while cutting its weight traffic
**6.8x**, and keeps most of the streaming path's traffic advantage without any of
its 4096 launches or its uploads. The residual gap to the streaming path's 0.33M
is the padding plus the fact that an expert spanning several blocks re-reads its
row once per block -- 14.4% and ~2.9 blocks per expert at this shape.

`kBlockSize` lives in `colibri_v2_moe_align.hpp` and `COLIBRI_MOE_BLOCK` in the
kernel corpus; each names the other, because they size the same accumulator.

**Not yet wired in.** Step 3 is the rows path calling this once per layer instead
of the per-expert loop, which also needs the down-projection twin (symmetric: one
block per (hidden row, route-block), reading `activated` by slot and scattering
into the token's output). Until then the kernel is dead code with a contract --
deliberately, because the wiring is where the sequencing risk is and it should
land on a kernel that is already proven.

### The down twin, and a constraint the corpus imposes

`iq1s_block_accumulate` / `iq4nl_` / `iq2xxs_` complete the pair. Verified against
the token-major `*_grouped_accumulate_rows` at worst 5.0e-06 and 5.9e-06 — not
0.0 like the swiglu, because block-major sums per slot and folds afterwards while
token-major reduces a token's routes in one pass, so the float reassociation
differs. Expected, and inside the 1e-5 contract.

**A design constraint worth recording, because it is not obvious and it changed
the kernel:** the block form cannot accumulate into the token's output row.
Several blocks (different experts) feed the same token, which needs an atomic —
and this corpus contains **no `atomicAdd` anywhere**, nor does the CPU-emulation
generator model one. Introducing the first would silently break every CPU twin and
the contract tests that depend on them.

So each slot writes its own weighted row and a scatter folds them into tokens,
which is what the streaming path already does with `qwen_scatter_add_rows`. Padded
slots write exactly `0.0` rather than being skipped, so a scatter can process them
unconditionally instead of needing a branch or a sentinel index — the contract
checks that zero explicitly.

Both kernels are now proven and unwired. Step 3 remains: build the aligned layout
per layer on the host, upload `sorted_routes` / `block_experts` / the per-block
pointer table, launch swiglu then accumulate then scatter, and reconcile with the
CPU-expert path for whatever the GPU does not cover. `align_blocks` already skips
zero-weight routes, which is exactly the marker the rows path uses for a route
claimed elsewhere, so that seam is already the right shape.

### STOP (2026-08-28): step 3 must not be wired -- the kernels were never timed

Everything above about these two kernels is a **weight-traffic** argument. They
were verified for correctness and their launch counts were counted, and then the
conclusion "6.8x less traffic in one launch" was carried forward as if it were a
throughput result. It is not. Timed on the real card at the real per-layer shape
(1024 rows, top-10, 512 experts, 1504 blocks, random codes so the codebook
lookups diverge -- the caution `bench_matvec_kernel.py` carries):

| kernel | ms/layer | achieved | weight read |
|---|---|---|---|
| `iq1s_block_swiglu` | 58.1 | 577 GMAC/s | 17 GB/s |
| `iq4nl_block_accumulate` | 76.5 | 219 GMAC/s | 18 GB/s |
| **fused total** | **134.6** | | **6.46 s per 48-layer chunk** |

The CPU expert phase this is meant to replace costs **~5.4 s per chunk** today.
**The fused path is slower than the CPU it would offload, before adding the
~1.4 s of uploads it needs.** And against the route-major kernel it was built to
replace it is **1.3x**, not the 6.8x the traffic ratio implied.

17-18 GB/s on a 391 GB/s card is 4-5% of bandwidth, and both kernels sit 20-36x
above their own bandwidth floor, so this is geometry, not physics. Reading them
back with that in mind, the causes are visible in the source:

- `_block_swiglu` calls `block_reduce_sum` **16 times per CUDA block** (8 slots
  x gate and up), each a 256-thread tree reduction with its own `__syncthreads`.
  The useful work between them is ~320 octets of decode.
- `_block_accumulate` is worse: `octets = 640/8 = 80` against `blockDim.x` of
  256, so **69% of its threads are idle for the whole kernel**, and it still
  pays 8 block reductions.

**What this changes.** The dispatch shape was right and the inner engine was
wrong. The streaming path's per-expert `*_q8_mmq` GEMMs are the efficient
compute engine on this card -- the plan's own standalone measurement puts
`q5k_q8_mmq` at 19.06 TFLOP/s, roughly 25x the rate these octet kernels reach --
and their only defect was that the host issues ~4096 launches per layer to use
them. So the target is not "block-major instead of MMQ"; it is **MMQ-quality
tiles driven by a block table**, which is what vLLM's `fused_moe_kernel`
actually is. These two kernels are a naive octet decoder wearing the right
dispatch shape.

Until that kernel exists, wiring step 3 would make prefill slower. The aligned
layout (`colibri_v2_moe_align.hpp`) survives unchanged and is still the right
input for it; the two `_block_*` kernels are the part to replace.

### Step 2b: the routed MMQ — the block table over the tensor-core core

`iq1s_q8_mmq_routed` and its four siblings (`iq2xxs`, `iq2xs`, `iq3xxs`,
`iq4xs`). `COLIBRI_Q8_MMQ_ROUTED` is `COLIBRI_Q8_MMQ` with the staging loop
changed and nothing else: `blockIdx.y` picks a route-block, its expert comes
from `block_experts`, and a slot's token is `sorted_routes[slot] / top_k`. The
ldmatrix pairs, the `m16n8k16` MMAs and the scale folding are the proven core,
untouched.

Measured at the real per-layer shape, against the octet kernel it replaces:

| | ms/layer (gate+up) | achieved |
|---|---|---|
| `iq1s_block_swiglu` | 58.73 | 571 GMAC/s |
| **`iq1s_q8_mmq_routed` x2** | **4.55** | **7381 GMAC/s** |

**12.9x**, and 0.22 s per 48-layer chunk against the ~3.4 s the CPU spends on
the same two projections. That is the piece the whole GPU route was missing.

**The tile is 128 rows x 32 tokens, not the plain kernel's 128x128.** An expert
holds ~20 routes here, so a 128-token tile would be 84% padding. 32 costs 37.7%
padding against kBlockSize 8's 14.4%, and buys 2.9x fewer weight decodes --
the right trade because this kernel is decode-bound, not MMA-bound (it reads
weights at 46 GB/s of a 391 GB/s bus). `kMmqBlockSize` in the align header and
`COLIBRI_MOE_MMQ_TOKENS` in the corpus name each other.

Correctness is pinned by `check_routed_mmq` against a double-precision
reference over the block table, not against the plain MMQ: a plain-kernel
reference would pass an off-by-one in the expert pointer whenever the two
experts landed adjacent in memory. Worst 2.4e-8 across all five formats, and
padded slots are checked to be exactly 0.0 — the SwiGLU and scatter behind this
run over them unconditionally, so a stale value there is a wrong answer for a
real token rather than wasted work.

**The down projection, on the same kernel.** IQ4_NL's exclusion from MMQ was
arithmetic, not layout: the macro hardcoded `input_size >> 8` and 640-wide
expert rows are not a whole number of 256-element super-blocks. Two parameters
fix that — `block_shift` / `group_shift` carry the format's blocking, (8, 3)
for a super-block and **(5, 0) for IQ4_NL's flat 32**, whose block simply *is*
a group. `iq4nl_q8_decode` reconstructs the nibbles through `kIq4nlValues` the
same way `iq4xs_q8_group` already did.

The down projection also needs **slot-major** activations — its input is the
per-slot SwiGLU output, not a token row — which `top_k == 0` selects.

| | ms/layer | achieved | against the octet kernel |
|---|---|---|---|
| `iq1s_q8_mmq_routed` x2 (gate+up) | 4.57 | 7348 GMAC/s | **12.6x** |
| `iq4nl_q8_mmq_routed` (down) | 3.05 | 5507 GMAC/s | **25.0x** |
| **whole expert phase** | **7.61** | | **0.37 s per 48-layer chunk** |

Against the ~5.4 s the CPU spends per chunk today, that is **14.6x on the
expert compute**, and it moves the bound where the earlier estimate said it
would go: **uploads**. 37 GiB of expert weights per 1024-row chunk at ~26 GB/s
is ~1.4 s, so a wired path lands near ~1.4-1.8 s per chunk against 5.4 s —
about **3x on prefill**, upload-bound, exactly the ceiling this document
predicted before the kernel existed.

### Step 3 SHIPPED: the rows path drives it, opt-in

`COLIBRI_ROUTED_MOE=1` replaces the streaming path's per-expert loop with one
aligned layout per layer and a handful of launches per tile. Measured at 2048
tokens, first token identical in every configuration:

| configuration | wall | prompt tok/s |
|---|---|---|
| streaming path (what it replaces) | 22.27 s | 92 |
| CPU expert phase (today's default) | 11.73 s | 175 |
| **routed MMQ, 512 MiB arena** | **8.33 s** | **246** |

**2.7x over the path it replaces and 1.4x over the CPU default** — the first
time the device expert route has beaten the host on this box. Greedy tokens are
*identical* to both the streaming path and the CPU path over 16 tokens, which is
the check that matters: the routed form changes only which CUDA block computes
a route, not the arithmetic.

Arena sweep, routed: 512 MiB 8.33 s, 1024 MiB 9.42 s, 2048 MiB 9.80 s. Smaller
still wins, for the reason the auto default already encodes -- restaging scales
with the budget while densest-first saturates -- but the optimum has moved up
from 48 MiB, and the knee below 512 MiB was not searched.

How it fits the existing path, which is most of why the change is small:
- the layout is built from the routes staging already claimed, keyed by
  **group** rather than expert, so the block table indexes the compact pointer
  tables directly;
- activations are quantized **once per layer** instead of once per expert,
  which also deletes the per-expert `qwen_gather_rows`;
- tiles are sized so every per-slot buffer is scratch the streaming path
  already owns (`stream_cap / kMmqBlockSize` blocks), so no new large
  allocation -- only ~200 KiB of index and pointer tables, sized in prepare and
  taken in the forward in the same order, as that walk requires;
- `qwen_slot_accumulate_rows` folds slots back onto tokens **token-major**.
  `qwen_scatter_add_rows` is only safe when a launch touches each token once,
  which holds for one-expert-per-launch and fails for a mixed-expert tile, and
  the corpus has no atomicAdd. Owning the output row is the fix.

Opt-in rather than default: this is the first path to run these kernels end to
end, and the gate combination that reaches it (`--hybrid-prefill cpu`, a cache
seed, forced DMA paging, a stream arena) is narrow enough that the default
deserves its own A/B before moving. Note that on this box the streaming path is
otherwise **unreachable**: DMA registration needs routed GPU execution allowed,
and the stream gate needs `hybrid_prefill_cpu` once it is -- so the two
conditions exclude each other unless `--hybrid-prefill cpu` is set explicitly.
