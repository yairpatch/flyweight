# Prompt-lookup drafting for qwen4exp decode

> **Goal**: decode speed on Qwen3.8-Flash-Next without touching the weights or the
> sampler's distribution. The checkpoint has no draft block, so the only speculation
> available is self-speculation: draft from the sequence's own history and verify
> through the batched rows pass that MTP already uses.
> **Status 2026-09-10**: SHIPPED behind `FLYWEIGHT_LOOKUP_DRAFTS=N` on branch
> `perf/lookup-drafts`. Code rewrite +11% (33.8 -> 37.6 tok/s, 88-90% acceptance),
> prose +0.2% (drafts almost never fire). Found and fixed a pre-existing bug in the
> MTP replay rollback on the way (sampler choices were overwritten by the greedy
> replay on every checkpoint with a PLE ring).

## Where the decode time goes, and why this is the lever that was left

Flash-Next UD-IQ1_S, warm, `--expert-mode auto`, 5070 Ti 12 GB + 9955HX, 2026-09:

| | ms/token | bound by |
|---|---|---|
| routed experts (CPU misses + GPU hits) | ~13 | DDR5 at 58 GB/s, 750 MB of expert rows per token |
| route wait (dense block on GPU) | ~10 | 391 GB/s VRAM, 3.2 GB of dense weights per token |
| attention, DeltaNet, LM head, host | ~5-7 | |
| **total** | **~30** | 33-35 tok/s |

Both big terms are memory walls and they serialize per layer. Requantizing the dense
block would cut the second term but costs quality, which is off the table. Speculation
is the only lever that touches the dense term without changing a weight: a verify pass
over K rows reads the dense weights once for K tokens.

## Design

- `qwen_lookup_draft` (v2_runtime.cpp): the committed history plus the token about to
  be fed is the text. The last n tokens are searched for earlier in the text, longest
  match first (n from 32 down to `FLYWEIGHT_LOOKUP_NGRAM_MIN`, default 4; most recent
  occurrence on ties), and the tokens that followed that occurrence become the drafts.
  Linear backward scan; at 8K tokens it is well under the noise floor of a round.
- The round is `qwen_mtp_round` with a `preset_drafts` argument: same verify batch,
  same per-row sampler walk, same snapshot/restore rollback. The draft-block cache
  commit is skipped when there is no draft block. Nothing else in the round changed.
- `qwen_spec_drafts()` is the new "does this runtime verify rows" predicate. Everything
  the rows verify needs (rows workspace, DeltaNet + PLE snapshot arena, hand-over rows,
  single sequence slot, KV rewind refusal) keys off it; only the draft block's own KV
  and hand-over still key off `options.mtp_drafts`.
- Gating: a round happens only when the history offers a match. No timing calibration
  (that is the draft block's `FLYWEIGHT_MTP_ADAPTIVE`, which measured rounds against
  decodes; here the match test already decides).

## Sizing the round: what the trace said

`FLYWEIGHT_LOOKUP_TRACE=1`, 8 drafts, n-gram search to 32, code rewrite + prose:

| match length | rounds | accepted of 8 | rejected rounds |
|---|---|---|---|
| 3 (code) | 13 | 2.0 | 11 |
| 3 (prose) | 9 | 0.8 | 9 |
| 5-7 | 1 | 8.0 | 0 |
| 12-15 | 3 | 6.3 | 0 |
| 16-23 | 2 | 8.0 | 0 |
| 24+ | 15 | 7.8 | 1 |

So: matches under 4 do not draft; the budget is `match_len - 1` capped at
`FLYWEIGHT_LOOKUP_DRAFTS`. With that, prose drafts 9-12 tokens in 320 and stays at
sequential speed; code drafts ~190 and accepts 88-90%.

## The cost of a verify row, and why the ceiling is ~1.8x

`FLYWEIGHT_PREFILL_TRACE=2` (per-phase stream syncs, so absolute numbers are inflated;
ratios hold), per verify pass over all 48 layers:

| rows | experts ("pre") | dense (moerms+proj+qkv+out) | total | per row |
|---|---|---|---|---|
| 3 | 43 | 23 | ~68 | 23 |
| 5 | 57 | 25 | ~83 | 17 |
| 6 | 87 | 25 | ~113 | 19 |
| 9 | 123 | **71** | ~200 | 22 |

The expert term is linear in rows at ~13-14 ms per row: each row routes to its own
ten experts per layer, so a 5-row verify streams five tokens' worth of expert bytes
from RAM. That is the same wall decode hits, and speculation does not move it. The
dense term is flat up to 8 rows (the batched q8 matvec serves 8 rows per weight read)
and then jumps: 9 rows leave the rows kernel for the tensor-core MMQ path sized for
64-row prefill chunks. Hence the 7-draft cap (`FLYWEIGHT_LOOKUP_DRAFTS` refuses 8).

A verify row therefore costs ~17 ms against a 30 ms decode, and every row past the
first rejection is wasted. With per-draft acceptance p, a W-draft round yields
(1-p^(W+1))/(1-p) tokens:

| p | W=3 | W=4 | W=7 |
|---|---|---|---|
| 0.90 | 19.5 ms/token | 20.3 | 23.3 |
| 0.96 | 17.9 | 18.1 | 19.0 |

Four is the recommended budget. The asymptote is 30/17 = 1.8x on perfectly repetitive
text; the measured +11% on a code rewrite is that asymptote times the fraction of
tokens a match covers, minus the rejected rows.

**What would raise the ceiling**: only a cheaper verify row, which means fewer expert
bytes per row. More VRAM-resident experts help decode and verify equally (this box
runs a 1.78 GB / 1007-slot cache at ~35% hit rate with ~2.5 GB of VRAM unallocated;
`--gpu-cache-mib` is the knob). Sharing expert reads across rows needs the rows to
route to the same experts, which they do not (top-10 of 512).

## Bit-identity

Greedy output is compared exactly in `bench_lookup.py`. Two findings:

1. **The auto/hybrid baseline is not reproducible run to run** on the real model:
   a GPU-resident expert and the same expert on the CPU accumulate in different
   orders, and residency differs between runs. Baseline-vs-baseline diverged at token
   43 (prose) and 296 (code) in consecutive runs. `check_greedy_determinism.py`
   documents this; use `--moe-device cpu` for identity checks.
2. **In cpu mode the baseline is reproducible, and lookup-on matched it exactly on
   the code prompt but flipped one token (position 43) on the prose prompt.** The
   verify rows go through the batched kernels (q8 rows matvec, rows GEMM for CPU
   experts, DeltaNet rows recurrence) whose float summation order differs from the
   single-token kernels, so a near-tie can resolve differently. Same weights, same
   quantization, same sampler; it is the same class of difference as (1), which
   serving already lives with. It is not bit-exact, and no verify-based scheme on
   this codebase can be without running the verify through the single-token kernels.

## Fixed on the way: replay rollback discarded sampler choices

`qwen_mtp_round`'s full-replay branch (taken whenever the fold is off, which is every
qwen4exp checkpoint because the PLE ring has no retention arena) re-ran
`qwen_verify_target_rows` over the accepted prefix **into the `verified` array**,
overwriting the tokens the sampler had chosen with the replay's greedy argmaxes. A
penalized or temperature-sampled task drafting on qwen4exp emitted the bare greedy
token at exactly the rows where the sampler had overruled the draft. qwen35 never
showed it (fold path). Fixed by replaying into scratch; pinned by
`Qwen4ExpSampledMtpTest` (draft block) and `test_v2_lookup_drafts.py` (lookup), both
on the synthetic qwen4exp fixture.

## Files

- `native/src/v2_runtime.cpp`: `lookup_*` runtime fields, `qwen_spec_drafts`,
  `qwen_lookup_draft`, `qwen_mtp_round(..., preset_drafts)`, both call sites, the
  replay fix, env parsing at runtime create.
- `native/src/v2_mtp_verifier.inc`: `[phase]` trace now prints the row count.
- `tests/test_v2_lookup_drafts.py`, `tests/test_v2_qwen4exp_mtp_plan.py`.
- `bench_lookup.py`: interleaved A/B with exact stream comparison.

## Env

| variable | default | meaning |
|---|---|---|
| `FLYWEIGHT_LOOKUP_DRAFTS` | unset (off) | draft cap per round, 1-7; 4 recommended |
| `FLYWEIGHT_LOOKUP_NGRAM_MIN` | 4 | shortest match that drafts |
| `FLYWEIGHT_LOOKUP_NGRAM_MAX` | 32 | longest match searched for |
| `FLYWEIGHT_LOOKUP_TRACE` | unset | per-lookup and per-round stderr lines |

Ignored with a note when `--mtp-drafts` is set (the draft block drafts instead).
