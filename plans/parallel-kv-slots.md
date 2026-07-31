# Parallel KV-cache slots (llama.cpp `--parallel` parity)

## Context

The v2 native runtime is **single-sequence**: one attention KV cache + one DeltaNet
recurrent state (`runtime->state`), one `processed_tokens`/`position`, and a small
pool of DeltaNet snapshots. Agentic clients (Claude Code, opencode) multiplex several
*logical* conversations onto it — the main agent, subagents, and small side calls
(topic/title detection, quota/summaries). Each request that isn't a clean extension
of the resident sequence overwrites the shared KV cache and evicts the snapshots, so
the **next turn of the main conversation reprefills from scratch** even when nothing
was compacted and the prompt barely changed.

### Confirmed empirically (2026-07-22, real 35B Q5)

Same append-turn, measured with vs without an interleaved side-request:

```
CONTROL (no interleave)
  turn2 (append)   prompt=616 reused=512 lcp_snap=600   # reuse works
INTERLEAVED (a 562-tok side-request between)
  turn2 (append)   prompt=616 reused=0   lcp_snap=0     # full reprefill
```

`reused=0, lcp_snap=0` after the side-request: it clobbered the KV cache *and* evicted
the main conversation's checkpoints from the limited slots. This reproduces the user's
server log, where a 362-token `/v1/messages` wedged between two big turns turned the
follow-up into a 169 s cold reprefill, while a follow-up with nothing interleaved reused
in 3 s. This is the single-slot gap vs llama.cpp, which runs N independent slots so side
traffic never evicts the main sequence.

## Goal

N independent sequence slots, each with its own KV + DeltaNet state + bookkeeping.
Route each request to the slot whose cached tokens are the longest exact prefix of the
prompt; otherwise the LRU slot (reset it). Side-requests land on a different slot and
never evict the main conversation. Runtime-configurable (`--parallel N`, per user
preference for config over env). This is the cache-preservation win only — the
generation pool stays `max_workers=1`, so slots are time-multiplexed, not decoded
concurrently (parallel throughput is a separate, later concern).

## VRAM (measured: ~20 KB/token f16 attention KV over 10 layers + ~64 MiB fixed DeltaNet)

Per-slot state = `~64 MiB + ~20 KB × context`:

| context | per-slot state |
|---|---|
| 8k  | ~224 MiB |
| 32k | ~704 MiB |
| 128k | ~2.6 GiB |

Slots compete with the auto-fit expert cache (memory: cache size is ~throughput-neutral,
so trading some for slots is favorable). **Slots need not be symmetric** (Phase 3): a
small scratch slot absorbs short side-requests at a fraction of a full-context slot's
cost. The operator sets `--parallel` (and, later, per-slot context) to trade VRAM for
conversation isolation as their hardware allows — a config choice, not a design constant.

## Design

Introduce a per-sequence `Sequence` holding what is today global on the runtime:
`state` (its own KV+DeltaNet arena), `processed_tokens`, `position`, `last_output`,
its snapshot pool, and an LRU `clock`. `runtime` holds `vector<Sequence> sequences`
and an `active` index.

Kernels already address state as `runtime.state + <fixed layer offset>` (29 sites, all
relative to one base). So switching sequences is: point `runtime.state` at the chosen
slot's arena and restore that slot's `position`/`processed_tokens`/`last_output`. **No
kernel changes.** The existing reuse + KV-safety guard logic runs unchanged against the
active slot.

Router (in `colibri_v2_qwen_runtime_generate`, before prefill): pick the slot whose
`processed_tokens` (or a snapshot) is the longest exact prefix of the incoming prompt;
tie-break by longest match. If none matches, pick the LRU slot and reset it. A prompt
that matches a slot's prefix stays on that slot (sticky conversation). Optionally bias
short/no-match prompts toward a designated small scratch slot.

## Phases

1. **Refactor to a `Sequence` struct, N=1.** ✅ SHIPPED 2026-07-22. `QwenSequence`
   holds the per-slot KV+DeltaNet arena, `position`, `last_output_token`,
   `processed_tokens`, and per-slot checkpoint pool; the active slot mirrors onto
   the runtime so kernels are untouched. `qwen_switch_sequence` saves/loads on a
   slot change (pointer + host bookkeeping swap, no device copy). Validated
   bit-identical greedy output N=1 vs N=2.
2. **Allocate N slots + router.** ✅ SHIPPED 2026-07-22. `qwen_route_sequence`:
   route to the slot the prompt genuinely *continues* (match ≥ 50% of that slot's
   committed tokens — prevents a short side-request that shares the system prefix
   from evicting the big conversation), else the LRU slot. Validated: the
   interleave experiment goes `reused 0 → 512` at N=2, and a shared-system-prefix
   side-request no longer evicts the main slot.
3. **Heterogeneous scratch slot(s).** TODO. Today all N slots are full-context
   (symmetric). Optional per-slot context sizing (a big slot + smaller scratch
   slots for side traffic) lets operators isolate conversations at lower VRAM;
   it's a config knob, not a fixed layout.
4. **Config plumbing.** ✅ SHIPPED 2026-07-22. `--parallel N` (default 1) threaded
   options → v2.py → NativeV2InferenceService → cli, mirroring `cache_type_k`.
5. **Host-backed prompt cache** (match llama.cpp's spill/restore). ✅ SHIPPED 2026-07-22.
   With N GPU slots, an LRU-recycled conversation is lost to a cold reprefill. llama.cpp
   instead spills evicted slot state to a bounded host cache (`--prompt-cache-mib`,
   analogous to its 8 GB default) and restores it into a slot on a later matching
   request, so a recycled conversation comes back from RAM instead of reprefilling.
   Design: `QwenHostPrompt { tokens, host_state=malloc(state_bytes), position,
   last_output, clock }`. On evict, DtoH-copy the victim slot's arena into a host entry
   (skip short/side-request slots via a min-token threshold; LRU-evict host entries over
   the byte limit). In the router, also match the incoming prompt against host entries;
   if a host entry beats every GPU slot, pick an LRU victim, spill it, HtoD-restore the
   host entry into it. limit=0 disables (bit-identical to Phase 2). Full-arena copy for
   v1 (correct; copying only `[0,position)` per layer is a later optimization).
   Concurrent decode is still out of scope (`max_workers=1`).
   Shipped: `QwenHostPrompt`/`QwenHostSnapshot` + `qwen_spill_slot_to_host` /
   `qwen_restore_host_to_slot` (arena **and** checkpoints spilled/restored — the
   end-of-prompt checkpoint is what lets the recalled turn reuse past the prompt
   boundary), per-entry byte accounting + LRU eviction, `--prompt-cache-mib` (needs
   `--parallel >= 2`; min 2048-token conversations spilled). Validated live 35B:
   evict-then-continue goes `reused 0 → 1024` (all checkpoints recalled) with
   bit-identical output vs cold.
6. **Packed `[0,position)` spill.** ✅ SHIPPED 2026-07-22. `qwen_used_state_ranges`
   computes the live device ranges (per attention layer, each kv-head's used
   [0,position) prefix of its head-major K/V slab — layout `head*capacity+pos`,
   verified for f32/f16/bf16 and q8_0 — plus full DeltaNet conv+recurrent);
   spill/restore copy only those, in deterministic order (layout is a pure
   function of config+position, so restore recomputes the same ranges). Bytes
   beyond `position` are never read (attention is position-bounded), so leftover
   victim data there is harmless. Measured: 121.5 MiB packed vs 223.8 full arena
   at position 3000/8192 (~46% saved, scales with position/context). Env-gated
   `COLIBRI_ROUTE_TRACE=1` prints routing decisions (debug switch, not a tunable).
   Note: spilled checkpoints (~64 MiB each, up to 4) now dominate a host entry;
   spilling only the end-of-prompt + best mid checkpoint is a future trim.
   Not done: concurrent decode.

## Risks / notes

- **VRAM**: gated by context × N; heterogeneous slots keep it small. Auto-fit expert
  cache shrinks to accommodate.
- **Correctness**: the KV-safety guard already prevents cross-sequence KV splicing;
  per-slot arenas make each slot self-consistent by construction. Phase-1 bit-identical
  check is the gate.
- **Not parallelism**: `max_workers=1` means no concurrent decode; the win is avoiding
  reprefill, not throughput. True parallel decode (batched multi-seq) is a separate,
  much larger effort.
- **Interim mitigation (no code)**: point Claude Code's small/side model at a different
  backend so only the main conversation touches colibri — recovers most of the loss today.

## Instrumentation already in place

`/health`.execution: `prefix_cache_last_{reused_tokens,prompt_tokens,lcp_snapshot}` and
`prefix_cache_reprefilled_tokens` — used to confirm the diagnosis and will measure the
before/after per slot.
