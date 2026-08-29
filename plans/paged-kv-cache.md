# Paged KV cache (vLLM-style)

## Where the runtime is today

Each of the `--parallel N` slots owns one **contiguous** device arena of
`runtime->state_bytes`, allocated at `prepare()` and sized for the full
`context_limit` (`v2_runtime.cpp:14018`). Within a slot, every attention layer's
K and V are head-major slabs `[kv_head][cache_capacity][head_dim]` at
`layer.state_first` / `layer.state_second`, and every kernel addresses a row the
same way:

```cuda
keys + ((long long)kv_head * capacity + token) * head_dim
```

Slot switching is a pointer swap (`runtime.state` is repointed at the chosen
slot's arena), which is exactly why phases 1-6 of [parallel-kv-slots.md] needed
no kernel changes.

Three consequences follow from "contiguous, full-context, per slot":

1. A slot costs the full context whether it holds 200 tokens or 200k.
2. Two slots sharing a system prompt hold two copies of it and each pays the
   prefill for it. There is no cross-slot reuse — the router matches a prompt to
   *one* slot, and the host prompt cache restores a *whole* conversation.
3. SWA layers ring-wrap (`(first + token) % capacity`) instead of freeing.

## What the reservation costs (measured 2026-08-28)

`Qwen3.6-35B-A3B-UD-Q5_K_M`, RTX 5070 Ti Laptop, 12227 MiB VRAM, `moe_device=hybrid`,
auto-fit expert cache. One process per row (two runtimes cannot coexist on this card):

| ctx | slots | state/slot | slots total | expert cache | expert slots |
|---|---|---|---|---|---|
| 8192 | 1 | 223.8 MiB | 223.8 MiB | 4814 MiB | 1975 |
| 8192 | 4 | 223.8 MiB | 895.0 MiB | 3661 MiB | 1502 |
| 32768 | 1 | 703.8 MiB | 703.8 MiB | 4617 MiB | 1894 |
| 32768 | 4 | 703.8 MiB | **2815.0 MiB** | 1867 MiB | **766** |

Going from one 32k slot to four costs 2.1 GiB of reservation and **−60% of the
resident expert cache** (1894 → 766 slots). Auto-fit probes free VRAM and drifts
run to run (the banner says so), but that is a ~5% effect, not a 60% one.

(Phase 1b later measured that the expert-cache column is not the payoff it looks
like: a bigger cache changes decode by less than this box's ±10% run-to-run drift,
because route wait dominates the token. The reservation is still worth reclaiming;
the reason is memory, not speed.)

This is the whole argument for reclaiming it — but **not** for what the reclaimed
bytes then buy. The original reading here was that [decode-expert-paging] makes
resident-expert coverage the decode lever, so the reservation is spending decode
speed on context a side-request slot never touches. Phase 1b measured that and it
does not hold: a doubled expert cache moves decode by less than this box's
run-to-run drift, and the device-side profile shows the GPU already ~93% busy with
the routed expert kernels as the dominant term. See phase 1b.

What survives is the memory itself. A realistic agentic occupancy — one 20k
conversation plus three ~1k side slots — genuinely needs `23k × 20 KiB +
4 × 64 MiB ≈ 716 MiB`, not the 2815 MiB reserved, and on a 12 GB card that
difference is headroom, extra slots, or a longer context. Just not throughput.

## What a session actually touches (measured 2026-08-28, phase 0)

`bench_kv_reservation.py`, same model and card, `--context 32768 --parallel 4
--gpu-cache-mib 8500`: five turns of a growing conversation with a short
side-request (title / one-line summary / one-word classification) wedged between
each pair, which is the traffic shape that motivated per-slot arenas at all.

```
  turn 1: peak live   257.5 MiB over 127 tokens (largest slot 103)
  turn 3: peak live   262.6 MiB over 390 tokens (largest slot 311)
  turn 5: peak live   266.7 MiB over 598 tokens (largest slot 519)

reserved     2815.0 MiB (703.8 MiB per slot)
peak live     266.7 MiB (9.5% of the reservation)
reclaimable  2548.3 MiB = 1045 expert slots on top of today's 715
peak tokens  598 summed, 519 in the largest slot, 131072 reserved for
```

The five points are linear, and the fit is the whole story:

```
live ≈ 255 MiB + 20.0 KiB × (tokens resident across all slots)
```

255 MiB is the four slots' DeltaNet floor, which no block pool reclaims. The
20.0 KiB/token slope independently reproduces the figure in
[parallel-kv-slots.md], and the reservation is reached only at 131072 resident
tokens — i.e. exactly when all four slots are simultaneously full.

So the honest reading, since this particular session is short (519 tokens in its
largest slot): a realistic 25k-token main conversation plus three small side
slots lands at `255 + 25k × 20 KiB ≈ 745 MiB`, still 26% of the 2815 MiB
reserved. **The case survives** — occupancy is bounded by the largest live
conversation, not by `slots × context`, and an operator raises `--parallel` for
isolation, not because the extra slots are full.

Note what is **not** pageable: the ~64 MiB/slot DeltaNet conv+recurrent state is
fixed-size, not position-indexed. On this 10-attention/30-delta geometry paging
addresses the 160 MiB (8k) / 640 MiB (32k) attention part of each slot and leaves
the 64 MiB alone. On pure-attention archs (Gemma 4) it addresses everything.

## What paging buys

1. **Reservation becomes occupancy.** Blocks are allocated when written. `--parallel`
   and `--ctx` stop multiplying into VRAM; one `--kv-pool-mib` bounds the whole thing
   and everything else goes to experts.
2. **Cross-sequence prefix sharing.** Two live conversations on a shared 20k system
   prompt reference the same blocks (copy-on-write at the divergence block) instead
   of holding and prefilling two copies.
3. **Block-granular spill.** `qwen_used_state_ranges` (`v2_runtime.cpp:18951`) already
   computes the live prefix per head to pack host spills; under paging the live set
   *is* the block list.
4. **SWA stops ring-wrapping.** A windowed layer frees its head block instead of
   modular-indexing into it, which deletes the `first`/`% capacity` variants of the
   score and value kernels.

## What it costs — the change inventory

| Site | Count | Shape of change |
|---|---|---|
| `flyweight_v2_qwen_kernels.hpp` device address sites | 28 | all `(kv_head * capacity + token) * head_dim`; one shared `__device__` helper |
| `flyweight_v2_native_kernels.hpp` CPU mirrors | 6 | same substitution (the other `capacity` uses there are score-buffer strides, not KV) |
| `flyweight_gpu_attention_f16_cublas`, `..._prefill_f16_cublas` | 2 | **the hard ones** — they pass `capacity * head_dim` as a strided-batch stride |
| dispatch sites in `v2_runtime.cpp` | ~9 | thread a block table beside `capacity` |
| `qwen_used_state_ranges` + spill/restore, KV dump, MTP rollback/fold, QSA block store | — | rework against block lists |

Two things de-risk it:

- **The corpus already has gather attention.** `kv_attention_{scores,values}[_q8]_indexed`
  (`flyweight_v2_qwen_kernels.hpp:10743`) run attention over an arbitrary per-token slot
  list for QSA. Paged addressing is the same kernel shape with the slot computed from a
  block table rather than read from a selection buffer.
- **The ring variants prove indirection is affordable.** SWA layers already compute
  `(first + token) % capacity` per token in the inner loop.

The cuBLAS tiers are the real work. They treat the K/V slab as a matrix with batch
stride `capacity * head_dim` over heads, which a block pool breaks. Options, in
preference order: (a) `cublasGemmBatchedEx` with a device pointer array built by a
tiny kernel from the block table — needs a new symbol in the dlopen'd `CublasApi`;
(b) one strided-batched call per block, which is fine at 32k/B=1024 (32 calls) and
not at 128k/B=256 (512 calls); (c) gather into a contiguous staging buffer — rejected
for f16 decode, since it doubles the KV read on the path that is already at the
bandwidth wall ([hardware-ceilings]).

## Design: coarse blocks

vLLM uses `block_size=16` because it schedules hundreds of concurrent sequences. This
runtime has `N ≤ ~8` slots, `max_workers=1`, and no concurrent decode, so fine
granularity buys nothing and costs kernel churn. **Propose B = 512 or 1024 tokens.**

- Internal fragmentation ≤ B tokens per slot per layer — at qwen4exp geometry
  (24 KiB/token over 12 attention layers) that is ≤ 24 MiB against GiB reservations.
- ≤ 256 blocks/slot at 128k: the block table fits in constant/shared memory, so the
  per-token indirection is a shared-memory read, not a global one.
- Each block holds B contiguous rows *per head*, so per-head contiguity — which the
  cuBLAS and flash tiles assume — survives at block scope.
- Prefix sharing at 1024-token granularity still captures a 10-20k system prompt.

Per-layer pool layout: `[block][kv_head][B][head_dim]`, i.e. head-major *within* a
block. One helper, used by every kernel and its CPU mirror:

```cuda
__device__ inline long long kv_row(const int* table, int token, int kv_head,
                                   int kv_heads, int block_size, int head_dim) {
    const int block = table[token / block_size];
    return ((long long)(block * kv_heads + kv_head) * block_size
            + token % block_size) * head_dim;
}
```

## Phases

0. **Occupancy instrumentation.** ✅ SHIPPED 2026-08-28. The runtime samples the
   live state across all slots at every request boundary (inside a request
   `position` only grows, so the boundary is that request's own maximum, and slot
   recycling only lowers the sum) and reports the high-water mark through
   `info` and `/health`.`prefix_cache`: `kv_reserved_bytes`, `kv_peak_live_bytes`,
   `kv_peak_tokens`, `kv_peak_tokens_max`, `kv_occupancy_samples`. Idle slots are
   counted rather than skipped — at position 0 the used ranges are exactly the
   non-pageable floor, which a pool still allocates per slot. `qwen_used_state_ranges`
   does the work, so occupancy and the host-spill packing can never disagree about
   what "live" means. `bench_kv_reservation.py` drives the agentic-shaped session;
   `tests/test_v2_kv_occupancy.py` pins the accounting on the synthetic qwen4exp
   fixture (a counter reading zero, or a peak above the reservation, would make the
   phase-2 case unmeasurable). Result above: **9.5% of the reservation touched**,
   and the linear fit says a realistic conversation reaches ~26%.

1. **Cross-slot prefix donation — no paging, no kernel changes.** ✅ SHIPPED
   2026-08-28. When a prompt shares a live slot's opening without continuing it,
   the router now gives it the LRU slot seeded by a device-to-device copy of the
   donor's position-indexed state `[0, p)` plus the donor's checkpoint at `p`,
   instead of handing over the donor's slot. Both pieces already existed:
   `qwen_used_state_ranges` (given the new `position_indexed_only` flag) computes
   the ranges, and `prefill_snapshots` holds the recurrent state at `p`. Layout is
   identical between slots, so it is a memcpy against a different base — no kernel
   touches attention indexing. New: `flyweight_gpu_copy_device` (byte-granular
   deliberately; quantized KV rows are not 4-byte multiples, so the float copy
   kernel could not stand in). `FLYWEIGHT_PREFIX_DONATE=0` is the kill switch;
   `prefix_donations` / `prefix_donated_tokens` are the telemetry.

   **Measured** (35B Q5_K_M, `--parallel 2`, 4-turn conversation, then a
   side-request sharing only the 2352-token system prompt, then the main agent's
   next turn):

   | main agent's next turn | reused | wall |
   |---|---|---|
   | donation on | 5645 / 6437 | **5.89 s** |
   | donation off | 1770 / 6437 | 28.99 s |

   **4.9× on the turn after an interleaved side-request**, and total reprefill
   across the session 11008 → 7133 tokens. The side-request itself reused 1572
   tokens in *both* arms, which is the point: donation buys the newcomer exactly
   what stealing the slot would have, and leaves the donor standing. Cost is one
   extra live conversation's KV — peak live 253.8 → 300.5 MiB.

   Two rules had to be found by measurement, not reasoning:

   * **The discriminator is the donor's residual, not the matched fraction.** The
     first version asked "did this prompt match more than half the slot?" and a
     side-request under a 2352-token system prompt covered 97% of a 2417-token
     conversation — so it declined. What is actually at stake is what taking the
     slot *overwrites*: `cached − match`. Zero means the prompt extends the
     conversation; a few hundred means a client re-rendered its last reply;
     thousands means something branched off it.
   * **No minimum on the donated length.** A 2048-token bar on `p` declined a free
     donation, because the donor's best checkpoint below the 2352-token shared
     opening sat at 1572. Taking the slot would have reused from those same
     checkpoints, so any `p > 0` buys the newcomer what stealing would have. Zero
     is the real floor: with nothing to seed, relocating trades the newcomer's
     reuse for the donor's, which is a different bet and not this one.

   **Churn check** (`bench_donation_churn.py`): the per-request argument for
   donation is that it is strictly-better-or-equal, but that says nothing about
   repeating it. Three conversations round-robin over two slots, 3 rounds, all
   sharing a 2352-token opening, host prompt cache off to isolate the mechanism:

   | | reprefilled | reused | wall |
   |---|---|---|---|
   | donation on (5 fired, 11500 tokens) | **42432** | 18552 | 253.4 s |
   | donation off | 48359 | 12625 | 290.8 s |

   **No thrash: 12% fewer tokens reprefilled and 47% more reused, on the
   deterministic counter.** Relocating conversations between slots does not
   compound into a worse eviction pattern.

   Note what the same table says about oversubscription itself, though: both arms
   degrade hard as the conversations grow, reuse stuck near the 2300-token shared
   checkpoint while prompts reach 9600. With three conversations on two slots one is
   always evicted, so it can only reuse the shared prefix, not its own history.
   Donation recovers some of that; it does not fix it. The lever for that case is
   the host prompt cache (`--prompt-cache-mib`), which this test deliberately
   disabled — measuring the two together is the obvious follow-up.

   A first attempt at this test recorded **zero donations in both arms** and looked
   like a clean null result. It was not: with ~200 tokens of divergence per
   conversation, nothing cleared the at-risk bar and the mechanism never ran. A
   churn test that never donates proves nothing about churn.

1b. **Heterogeneous slot sizing — no kernel changes either.** ✅ SHIPPED 2026-08-28,
   taken ahead of phase 2 because mapping phase 2 turned it up. `capacity` is
   *already* a per-call kernel argument and `layer.cache_capacity` is read in ~12
   places, so slots can differ in size without a single kernel or dispatch site
   learning about it: `--scratch-context K` gives slot 0 the full window and the
   rest K. The mechanism is the one the runtime already uses for arenas — the
   active slot's geometry is mirrored onto `runtime.layers` exactly as its arena
   is mirrored onto `runtime.state` (`qwen_apply_geometry`), so the ~29 dispatch
   sites are untouched.

   **Measured** (35B Q5_K_M, `--parallel 4 --context 32768`, auto-fit):

   | slots | reserved | expert cache |
   |---|---|---|
   | 4 × 32k | 2815.0 MiB | 1518 MiB (623 slots) |
   | 1 × 32k + 3 × 4k | **1135.0 MiB** | **3166 MiB (1299 slots)** |

   1680 MiB reclaimed, and at 2.44 MiB per expert slot that predicts +688 slots
   against +676 observed — the reservation converts to expert cache essentially
   one for one.

   **Scratch slots are decode-neutral**, measured at equal expert cache: 48.22 tok/s
   symmetric against a steady 48.3 for scratch with the budget cut to match (715
   slots, 49.5% hit rate both). The scratch arm needs 3-4 warmup generations to get
   there, not 1 — with a single warmup it reads ~5% slow.

   **What the reclaimed VRAM then buys is UNRESOLVED, and this box cannot answer it
   by wall clock.** Four A/B pairs were run, alternating order, `--gpu-cache-mib`
   pinned, median of 5 after warmup. The first three said a doubled expert cache
   (715 → 1405 slots, hit rate 49% → 67%) decoded 7-19% *slower*; the fourth said
   it decoded 14% *faster*. The identical symmetric configuration measured 48.22,
   47.05, 44.40 and 39.68 tok/s across those runs — **±10% run to run for a fixed
   config, which is as large as the effect being chased.** Three orderings agreeing
   was not the control it looked like; sequential-process comparisons cannot be
   interleaved here, because slot sizing and cache size are both fixed at prepare.

   The counters are the better instrument, being causally attributable rather than
   wall time. Per decoded token, `FLYWEIGHT_TIMING=1`:

   | | 715 slots | 1405 slots |
   |---|---|---|
   | decode | 20.29 ms | 17.54 ms |
   | cpu experts | 5.79 ms | 3.79 ms |
   | expert page | 6.88 ms | 4.74 ms |
   | **route wait** | **11.18 ms (55%)** | **10.81 ms (62%)** |
   | tail wait | 1.46 ms | 1.23 ms |

   (The components overlap — the CPU experts run while the GPU kernel is in flight —
   so they do not sum to the total.) A bigger cache reduces host expert work and
   staging, and this pair puts the total 14% lower; taken with the drift above, the
   fair reading is "modestly better or neutral, and smaller than this box can
   resolve by wall clock".

   **Route wait is not what it looks like, and calling it "the bottleneck" was
   wrong.** `FLYWEIGHT_CUDA_PROFILE=1` measures the device side of the same token:

   ```
   HOST  decode 21.08 ms   route_wait 13.97   cpu_experts 3.68   page 4.14
   GPU   expert 9.30 (47%)  delta 5.51 (28%)  attention 2.12 (11%)
         shared 1.51 (8%)   tail 1.31 (7%)    total 19.59 ms
   ```

   **The GPU is busy 19.59 of 21.08 ms — ~93% of the token.** There is no bubble to
   reclaim. `route_wait` is a host-side event sync placed after this layer's shared
   expert is enqueued, so it blocks until the device drains a backlog that is
   several layers deep: it measures *the host waiting on a saturated GPU*, not a
   round-trip latency. (It exceeds the 7.62 ms of pre-route work in its own layer
   for exactly that reason.) Removing or overlapping the sync would buy nothing.

   So the decode here is **GPU-bound, and the term to attack is the routed expert
   kernels at 9.30 ms/token, 47% of GPU time** — which is the "decode-shaped grouped
   kernels re-decoding weights per routed token" already named in
   [decode-expert-paging], not anything about KV or slots. Note this differs from
   [decode-overhead-audit]'s "GPU busy is 6.4 ms of a 30 ms token": that was a dense
   model serialized behind a CPU-FFN spill, a different configuration.

   **Treat reclaimed VRAM as memory, not throughput** — headroom, more slots, a
   longer context. Phase 0's framing of the reservation as "1045 expert slots the
   decode would rather have" overstated it.

   What the phasing note above got right: this is ~60% of the reservation, against
   the ~74% a paged pool would reclaim at realistic occupancy. What it cost:
   per-slot *layouts*, so anything touching an inactive slot's arena — host spill,
   host restore, phase 1's donation — now derives both range lists and refuses
   when they disagree region for region (`qwen_slot_state_ranges`,
   `qwen_ranges_are_compatible`), rather than assuming shared offsets. Routing
   gained a capacity filter that runs before every other consideration, and
   prefers the smallest slot that fits so short traffic lands on scratch.

2. **Block pool + block table, elementwise paths only.** TODO. f32/f16/bf16/q8/turbo
   score+value kernels and their CPU mirrors move to `kv_row`; the cuBLAS tiers are
   disabled while paging is on. Gate behind a flag (`--kv-pool-mib`, 0 = today's
   contiguous arenas). Gate to pass: bit-identical greedy output paged vs contiguous
   at N=1 and N=2 (`check_greedy_determinism.py`), and the path-parity harness green.

3. **Restore the cuBLAS/flash tiers under paging** via `cublasGemmBatchedEx` with a
   device pointer array. TODO. Re-run the prefill and long-context A/Bs — interleaved,
   per [prefill-gemm-status]'s 27% clock drift discipline — and keep whichever tier
   wins per shape.

4. **Copy-on-write prefix sharing.** TODO. Block-hash trie over committed prefixes;
   a slot that continues a shared prefix references the blocks read-only and copies
   the block it first writes into. Subsumes phase 1's copy for the attention half
   (the donation becomes a block-table entry rather than a memcpy, and stops
   duplicating VRAM); the DeltaNet checkpoint copy stays. Phase 1 also raised the
   ceiling on this: donation is bounded by the donor's *checkpoint spacing*, not by
   the shared prefix — the measured donation seeded 1572 of an available 2352
   tokens purely because that is where the nearest checkpoint sat. Block-granular
   sharing has no such quantization on the attention half.

5. **Retire the per-slot reservation.** TODO. `--parallel` and `--ctx` become free of
   each other, auto-fit gets the difference, and the banner reports pool occupancy
   rather than `Nx KV slot`.

## Risks

- **Decode is bandwidth-bound**, so the indirection must not become a global read per
  token. Block tables in constant/shared memory; measure, don't assume.
- **MTP rollback** rewinds `position` and the fold check depends on the batch's own
  bits ([mtp-fold-rollback]). Blocks freed by a rollback must not be recycled before
  the fold check runs, or a rejected draft's tail block reappears under another slot.
- **Two runtimes cannot coexist** on this card, so every A/B is a separate process —
  which also means the auto-fit VRAM probe differs between arms. Pin `--gpu-cache-mib`
  for any comparison, per [ornith-greedy-nondeterminism].
- **Phase 2 is the point of no return** for the 28 + 6 address sites; phase 1 is not.
  Phase 0 was the check on whether that is worth spending — it came back at 9.5%
  occupancy, so the case holds. What would still overturn it is a workload where
  several slots each carry a full-length conversation at once; the telemetry is now
  live, so that shows up as `kv_peak_live_bytes` approaching `kv_reserved_bytes` in
  `/health` rather than as a surprise after the kernels are rewritten.
