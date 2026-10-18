# DFlash + LiLiCorr drafter for Nemotron 3.5 Bolt

NVIDIA ships a 1.1B draft model beside Bolt:
`dflash-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4.gguf`, GGUF arch `dflash`,
`general.name` Li5_SWA. It is a DFlash block drafter with a LiLiCorr lattice
re-ranker on top. The drafter loads through `--mtp-model` and runs on the GPU
(`native/src/v2_dflash.inc`). Its drafts feed the speculative round the
lookup drafts already use (`qwen_mtp_round` with preset drafts, the Mamba-2
fold for rejections). Semantics follow llama.cpp cd0a5b6 (`src/models/dflash.cpp`
and the draft-dflash path of `common/speculative.cpp`), which is also the
oracle for Bolt itself.

## What the drafter computes

Target side. Every forward (prompt chunks, decodes, verify rows) records the
residual stream *entering* blocks `dflash.target_layers` = [2, 19, 36, 52,
69, 86] (`capture_slot`; llama.cpp's `t_layer_inp`). Each row concatenates the
six streams in list order into 24576 floats. After the forward, the rows
are injected:

    g_p = rms(fc . feat(p) * fc.scale) * enc.output_norm
    per draft layer l: K = k_norm(W_k g_p) rotated (NEOX, base 1e4) at p, V = W_v g_p

K/V go into a per-layer ring of window + 16 positions (`position % slots`)
with a position tag per slot, so the slots a verify leaves behind past the
accepted prefix are forgotten when the next forward starts over them (tags
>= its first position are dropped). The last 16 g rows are kept as LiLiCorr
anchors. A prompt chunk longer than the window injects only its last window
of rows: no later draft can see the rest.

Draft. One block `[id_last, mask x 7]` at pos0..pos0+7 (mask = the target's
`<SPECIAL_990>` embedding; the drafter has no embeddings or head of its own).
Five layers. Each sublayer is wrapped in a dynamic two-tap depthwise conv over
the block rows:

    w(c, tap, i) = coeff_i[c/16 + 256 tap + 512 side] + base[c, tap, side]
    conv(x)[i][c] = w(c,0,i) x[i][c] + w(c,1,i) x[i-1][c]      (x[-1] = 0)

`coeff` is `conv_proj . (normed input)`. Side 0 runs on the normed input,
side 1 on the sublayer output before the residual add, both with the same
coefficients. Attention is non-causal: block row i sees all 8 block rows,
plus every context position p < pos0 with pos0 + i - p < 1024. Q/K take
per-head norms and RoPE, and each head's sink logit joins the softmax
denominator only. The FFN is SwiGLU (NVFP4). Then `output_norm`, and the
*target's* LM head on rows 1..7.

LiLiCorr. Per slot s = 0..6: the top 8 of row s+1's logits, in descending
order. The 5 features `[lp, p, lp - lp0, r/7, r == 0]` go through a
LayerNorm (eps 1e-5) and a 5 -> 1024 -> 1024 MLP. The 56 node states are

    (token_proj(E[cand]) + pass_proj(z_{s+1})) + fs + slot_emb[s] + rank_emb[r]

and run through two transformer layers with a learned [56, 56, 8] attention
bias and no mask. The anchor is `context_proj(g_{pos0-1})`. Factor heads give
l2-normed OUT and IN per node and AO for the anchor. The lattice scores are
`8 IN[0][j].AO` for slot 0 and `8 IN[s][j].OUT[s-1][p]` after that. A greedy
walk takes `argmax` (first max wins) from slot 0 onward, conditioned on the
candidate picked before. The walk's first n_max tokens are the drafts.

The converter numbers LiLiCorr's layers after the trunk's (`blk.5`, `blk.6`)
and puts the feature MLP at `blk.7`, so those blocks are not trunk layers.

## Numerics

The drafter's numerics only move the acceptance rate, since the target
re-scores every draft.

- bf16 and f32 matrices (q/k/v/o, the conv projections, the LiLiCorr
  projections and heads) go to the device as Q8_0. The LiLiCorr NVFP4
  projections do too: they run 56 node rows, where the NVFP4 kernels batch 8.
- Q8_0 matmuls whose input is a multiple of 256 take Q8 activations: the
  dp4a rows kernel for up to 8 rows, the dp4a tile above that. The float tile
  kernel this replaced ran the 570 MB LM head at ~30 GB/s (20 ms of a 22 ms
  draft).
- NVFP4 (`fc`, the trunk FFN) takes Q8 activations at every batch size, the
  way llama.cpp runs the format, in 8-row passes of the Q8 rows kernel. That
  way a prompt's injected rows match the ones decode and verify inject.

## Results (Bolt IQ2_XXS, 5070 Ti Laptop 12 GB, `--expert-mode auto`, greedy)

Interleaved A/B, 200 tokens per prompt, warm runs, quiet machine:

| prompt | plain tok/s | drafter tok/s | drafted tokens accepted |
| --- | --- | --- | --- |
| code | 45.2-45.4 | 52.6-56.3 | 86% |
| prose | 45.5-47.5 | 50.1-52.2 | 81% |
| 2117-token document | 44.2-45.1 | 50.4-50.8 | 82% |

The server runs 55.4 tok/s greedy and 52.2 at temperature 0.7 on a code
request (the engine path, every verified row through the request's sampler).
The 2117-token prompt's prefill is within noise of plain (7.2-7.7 s against
6.6-7.8). On the code prompt, the numpy port of the spec on real Bolt
features (scratch scripts, not shipped) accepts 90%. llama.cpp's own runs
accepted 65-93%.

A draft costs ~6.7 ms at short context and ~9.3 ms with a full 1024-position
window (`FLYWEIGHT_DFLASH_PROFILE=1`): the NVFP4 FFN 2.0 ms, the LM head
1.3 ms, everything else under 1 ms each. A 4-row verify costs ~55 ms.
Drafter VRAM: ~0.75 GB of weights, an 11 MiB ring, ~30 MiB of workspace and
the 96 MiB capture buffer (1024 rows x 24576 floats). With Unreal Editor
holding 2.3 GB, that ~0.9 GB was what no longer fit beside Bolt.

The verify round is the ceiling. Per 4-row pass, `FLYWEIGHT_MOE_PROFILE=1`:
the GPU expert kernels take 20 ms and read an expert once per row that routes
to it (30% of the 1434 routes are duplicates). The CPU takes 512 refused
misses (20 ms). Every verify row past the first costs ~11 ms, so 2 to 5
drafts land within 5% of each other. 3 is the default, llama.cpp's
`--spec-draft-n-max`.

## Switches

- `--mtp-model PATH` attaches the drafter (Nemotron H targets only; the
  DFlash2 selector, DSpark Markov head and drafters with their own
  embeddings or head are refused).
- `--mtp-drafts N` sets the draft count (default 3, at most 7).
  `FLYWEIGHT_DFLASH_DRAFTS` overrides it.
- `FLYWEIGHT_DFLASH_PROFILE=1` brackets every drafter launch and reports the
  per-kernel GPU time every 32 drafts.
- `FLYWEIGHT_CAPTURE_LAYERS=a,b,...` records target features without a
  drafter (`V2QwenRuntime.captured_features`).

## Tests

`tests/dflash_gguf_fixture.py` writes a toy drafter beside the Nemotron
fixture: a 16-position window and a 32-slot ring, so the window edge and the
ring wrap sit inside a 45-token prompt. Its bf16/f32 matrices are generated
Q8_0-exact. It also carries a float64 reference drafter.
`tests/test_v2_nemotron_dflash.py` checks:

- The GPU drafter against the reference (decode-fed features): logits within
  1e-5 of their scale, identical candidates and drafts.
- Prompt injection through the rows forward, and the context after rounds
  with rejections, against the reference within 3%. The rows forward and the
  verify-and-fold sum in another order than decode, and on the toy drafter
  one Q8 rounding step in `fc` moves the logits by ~1.5%.
- Rounds keeping the greedy output; reset forgetting the context; refusal for
  a non-Nemotron target; `--mtp-drafts` setting the draft count.

A stale verify row seen mid-generation cannot change the output, only the
acceptance. The end-of-generation probe cannot see one: `generate` ends on a
plain decode that rewrites the tail. The real-model acceptance is the
evidence there.

## Not done

- Verify-row cost: group a verify batch's routes by expert (see
  `bench/bench_mtp_expert_reuse.py`).
- The adaptive draft-or-decode gate (`qwen_mtp_should_draft`) always drafts
  for DFlash. A drafter that loses on some content keeps drafting.
- A conversation restored from the host or disk prompt cache, or a slot
  switch, starts with an empty drafter context (the ring is reset, since it
  held another conversation). Drafting resumes after the first injected
  rows, and the first window of the continuation drafts with less context.
