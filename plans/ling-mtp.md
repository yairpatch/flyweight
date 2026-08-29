# Ling (bailingmoe3) MTP with fold-style rollback

Decided 2026-08-24: Ling should get speculative decode. Status today: the
`nextn` draft block loads (blk.N.nextn.* since the GGUF work) and is **never
executed** — the bailing runtime has no draft loop, no verify, no rollback.
This is a build, not a port.

## Why it is worth more on Ling than the qwen fold was

Ling decode is stall-bound, not bandwidth-bound: 179 tok/s against a 542 tok/s
wall, with expert routing round-tripping to the host once per MoE layer
(23 syncs/token ≈ 3.7 ms — see ling-decode-host-routing). A batched verify of
`wanted` rows pays those syncs **once per round instead of once per token**.
At 60% acceptance and drafts=3 (~2.5 committed tokens/round) the sync tax per
token drops ~2.5x, which attacks the top-ranked decode problem directly. The
fold contribution (rejection rollback) rides on top, same as qwen's +40% on
rejection-heavy work.

## What exists to build on

- `flyweight_v2_bailing_eval_slot` already advances all rows through every layer
  (host batched path and device tiled-prefill path both); only the head is
  last-row-only. **Verify = eval + per-row argmax.**
- Host caches are plain vectors: MLA rollback is `cache.positions` truncation;
  KDA state is `state` + three conv windows per layer — snapshot is small.
- Device path mirrors: `bailing_gpu_prefill_tiled` (multi-row), per-slot
  device caches, `bailing_cache_transfer` host<->device tracked by
  `cache_on_device` (do NOT let verify split paths — see the hazard comment in
  eval_slot about separate host/device caches).
- The KDA transition inputs are the three convolved projections + decay +
  beta, all computed from hidden per row (`kda_step`): the fold retention is
  the same shape as qwen's (record inputs, not states), replay only
  conv+`kda_recurrence` over the accepted prefix.
- Draft-side bugs only cost acceptance, never text: verify's tokens are target
  outputs. The risky (correctness) surface is small: rollback semantics.

## Phases

1. **Host-path round** (correctness foundation, fixture-testable in CI):
   - nextn block forward: enorm/hnorm, eh_proj([norm(embed); norm(hidden)]),
     one decoder layer (check which flavor Ling's nextn layer is — MLA or
     KDA — from the fixture/real checkpoint), shared_head_norm + head.
     Draft KV/state cache per slot, context-sized, recycled like qwen's.
   - `bailing_verify_rows`: host batched eval + per-row argmax winners.
   - Fold rollback: retain per-KDA-layer (q/k/v projections post-conv? no —
     retain PRE-conv projections + decay + beta per row, mirror qwen), on
     reject restore snapshot + replay conv+recurrence for `valid` rows;
     truncate MLA `positions`. Bitwise check mode from day one
     (the qwen lesson: on row-count-stable host kernels the check is exact).
   - Round driver + commit, `mtp_*` counters wired into bailing info.
   - Test: bailing GGUF fixture (tests/test_v2_bailing_gguf.py has nextn) —
     token parity MTP-on vs off... NOT expected (speculative changes nothing
     only under greedy; bailing sampling lives in the server) — assert
     greedy parity + fold-vs-replay parity + bitwise state check.
2. **Device verify** (the perf payoff): route verify through the tiled
   device path + device argmax rows; device-side KDA snapshot/fold (copy
   kernels + relaunch, exactly the qwen shape); keep `cache_on_device`
   coherent across draft/verify/rollback.
3. **Server + measurement**: native round entry the server can drive (the
   Python-side sampling loop cannot stay in the loop per round), bench file
   patterned on bench_mtp_fold.py, real-Ling A/B (Ling-3.0-flash is on disk),
   acceptance-rate telemetry.

## Cautions carried over

- Same-arm nondeterminism in hybrid/expert paths is ambient — pin budget,
  history off before comparing streams.
- The GPU bitwise check oracle is only exact where kernels are row-count
  stable; on device expect ulp seams (qwen: dense_rows matvec/rows-kernel
  switch). Validate exactness on the host path.
- eh_proj input order (embedding-first vs hidden-first) differs between MTP
  families and silently only degrades acceptance — pin it against the HF
  reference with a fixture parity test, not by eye.
