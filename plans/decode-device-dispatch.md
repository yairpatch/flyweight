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

Interleaved decode A/B (48 iterations, 4 warmup):

| config | run 1 | run 2 |
|---|---|---|
| staged (old default) | 19.98 | 13.81 |
| **direct DMA** | **30.28** | **29.36** |
| direct DMA + strict admission | 29.77 | 30.59 |

**~17 tok/s -> ~30 tok/s**, and the run-to-run variance largely disappears (the
staged path's spread was page-cache state).

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
