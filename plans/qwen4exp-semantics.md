# qwen4exp forward semantics — extracted from transformers (merged PR #48337) + llama.cpp PR #27739

Sources: `transformers/models/qwen4_exp/{modular,modeling,configuration}_qwen4_exp.py` on
transformers main (PR merged 2026-08-26, head b61b98be), llama.cpp conversion
`JJJYmmm/llama.cpp@add_qwen4exp:conversion/qwen4exp.py` (head dfa0c0fe), and the unsloth
UD-IQ1_S GGUF metadata/tensor inventory (parsed from file headers, see plan).
All math below quoted from HF source; verbatim class quotes captured in the conversation
that produced this file — re-fetch from transformers main if in doubt.

## Conventions that the GGUF already bakes in (conversion-side)

- **RMS norms use `(1 + weight)` with zero-init weights in HF; the conversion adds +1.0
  to every `*norm.weight` (incl. the PLE norms explicitly)** → runtime uses plain
  `rms_norm(x) * w`. EXCEPTION (found by the parity test 2026-08-26): the DeltaNet
  output norm `ssm_norm` is HF's `RMSNormGated`, which is ONES-initialized and applies
  its weight directly — no +1 anywhere for it. VERIFY on real file: range-fetch a norm
  tensor (e.g. `blk.0.hc_attn_norm.weight`, F32, offsets known from header parse) and
  check it is ≈1-centered. Fixture generator must replicate the +1 baking.
- No q/k rope permute at conversion (HF layout straight through).
- DeltaNet conversion goes through the same value-head reorder base the qwen35 GGUFs use
  (`_LinearAttentionVReorderBase`) → GGUF tensor layout matches what the existing
  `qwen_delta_recurrent*` kernels consume for qwen35.
- Vision tensors dropped; MTP fused to `NEXTN_EH_PROJ` only when converted with MTP —
  the unsloth file has neither.
- `per_layer_token_embd.weight [160, 320001536]` = ngram shards concatenated row-wise;
  row space = 16 heads at prime-sized vocab each, laid out by `ple.head_offsets`.

## RMSNorm (all sites)

`out = x * rsqrt(mean(x², group) + eps) * w_baked`, eps 1e-6, fp32 accumulation.
`group_size=hidden (2560)` for all 10240-wide norms (hc_attn/hc_ffn/output_hc, ple norm_key/
norm_query/norm_conv): normalize each 2560-chunk independently, weight is full-width 10240.
Per-head norms (attn q/k [256], ssm_norm [128]) are plain RMS over the head dim.
→ Implementation: existing `rms` kernel launched per group with weight-slice offsets (4
launches), or a grouped variant later.

## Hyper-connections (`Qwen4ExpTextGatedResidual`)

State: `hyper [4, 2560]` per token (flattened 10240). Init: `hyper = repeat(embedding, 4)`.
Per block boundary (attn and ffn each have their own weights; head has `use_combine=False`):

```
normed   = grouped_rms(hyper) * hc_norm                       # [10240]
low      = silu( (normed @ down) / hc_count )                 # [320]; down: [10240→320]
mixw     = sigmoid( low @ up )                                # [10240]; up: [320→10240]
block_in = mean_over_streams( mixw ⊙ normed )                 # [4,2560]→[2560]
injw     = 2 * sigmoid( (normed @ inject) / hc_count )        # [4];  inject: [10240→4]
# ... run block on block_in → block_out [2560] ...
hyper'   = hyper + injw[s] * block_out                        # per stream s (outer product)
```

NOTE the `/ hc_count` scaling on BOTH the down-proj output (before silu) and the inject
logits (before sigmoid), the `2 *` on inject gates, and that the residual base is the
**pre-norm** `hyper`, not `normed`. Final head collapse = same math, stops at `block_in`
(that 2560 vector goes to final norm → LM head). GGUF tensors: `hc_{attn,ffn}_{norm,down,
up,inject}` per layer, `output_hc_{norm,down,up}` global. No transposes at conversion:
`input_mix_weight_down→hc_*_down`, `_up→hc_*_up`, `block_inject_weight→hc_*_inject`.

There are NO plain attn_norm/ffn_norm tensors; hc replaces the rms→block→add bookends
entirely. PyTorch layer order: PLE add (blk.1 only, at layer top, in stream space) →
attn hyper-connection → attn block → inject → ffn hyper-connection → MoE block → inject.

## PLE (`Qwen4ExpTextPLELayer`, blk.1 only; `ple_layer_ids=[2]` is 1-based ⇒ layer_idx 1)

### N-gram ids (integer math, EXACT — pin with integer vectors)

Token history per sequence, **EOS-segmented** (eos = `ple.eos_token_id` = 248044):
`shift_right_ignore_eos(tokens, s)` = token at distance s back, but positions whose
in-segment index < s (i.e. would cross an eos or sequence start) yield eos itself.
Sequence start behaves as if preceded by eos. All in int64 (products fit: multipliers are
odd, < 2^63/vocab, so token*mult < 2^63; XOR of non-negatives stays non-negative — plain
u64 works, no 128-bit needed).

```
for ngram n in {2 (heads 0-7), 3 (heads 8-15)}:
    mixed = (t_0 * M[0]) XOR (t_-1 * M[1]) [XOR (t_-2 * M[2]) if n==3]   # int64
    for head h in the n-gram's 8 heads:
        row[h] = mixed mod prime[h] + offset[h]      # torch.remainder → non-negative
emb = concat over 16 heads of table[row[h]] (160 each) → [2560]
```

`M` = `ple.layer_multipliers` (3×u64), `prime[h]` = `ple.head_vocab_sizes`,
`offset[h]` = `ple.head_offsets` — all in GGUF metadata verbatim; do NOT re-derive
(HF derives them from seed 1234 via splitmix64 + next-prime search; conversion exports them).
Head order: bigram heads 0-7 then trigram heads 8-15, matching offset order.

### PLE forward (at layer top, before attn hyper-connection)

```
key    = grouped_rms(emb @ Wk) * ple_norm_key            # Wk: [2560→10240], → [4,2560]
value  = emb @ Wv                                        # Wv: [2560→2560]
query  = grouped_rms(hyper) * ple_norm_query             # [4,2560]
g[s]   = dot(key[s], query[s]) / sqrt(2560)              # per stream
g[s]   = sign(g) * sqrt(clamp_min(|g|, 1e-6))            # signed sqrt
gv[s]  = sigmoid(g[s]) * value                           # [4,2560] → flatten 10240
gvn    = grouped_rms(gv) * ple_norm_conv                 # [10240]
conv   = silu( causal_depthwise_conv1d(gvn; k=4, DILATION=3) )   # per-channel, [10240]
hyper += gv + conv                                       # additive into stream space
```

Conv state = `(k-1)*dilation = 9` past columns of 10240 (f32) — per-sequence, must join
the state arena + snapshot/rewind copies. The token-history "state" (last 2 token ids per
sequence) is free in colibri (tokens are tracked); decode-step hashing uses the live
history with the same EOS-segmentation rule.

## Full attention (12 layers: idx 3,7,…,47) — dense fallback semantics

```
q|gate = x @ Wq  → per head 24×[512]: chunk2 → q[256], gate[256]   # fused, interleaved per head
q      = rms(q_head) * q_norm_w ; k = rms(k_head) * k_norm_w        # per-head 256
rope   : HALF-SPLIT (rotate_half), PARTIAL: first 64 dims only, θ=1e7, pairs (i, i+32)
         M-RoPE sections [11,11,10] interleaved — for TEXT all 3 position streams are
         equal ⇒ collapses exactly to standard partial rope; ignore sections (qwen35moe precedent)
attn   = softmax(q·k / 16) @ v          # GQA 24:2, causal; scaling = 256^-0.5
out    = (attn ⊙ sigmoid(gate)) @ Wo    # sigmoid output gate then o_proj [6144→2560]
```

Matches the existing fused q|gate + `qwen_attention_gate` path (qwen35 geometry). The QSA
indexer (`Qwen4ExpTextQSAIndexer`) further masks attention to top-2048 selected tokens —
phase 3; dense fallback is exact whenever nothing would be pruned (short contexts).

## Gated DeltaNet (36 layers) — identical to qwen35

`in_proj_qkv` split order **[K(2048), K(2048)… i.e. key_dim*2 then value_dim]** →
q[16×128], k[16×128], v[48×128] after conv1d(k=4, silu, stateful) over the 10240 qkv vec;
`beta = sigmoid(x @ ssm_beta)`; `g = -exp(ssm_a) * softplus((x @ ssm_alpha) + dt_bias)`;
delta-rule recurrence; `out = gated_rms_norm(core, z=x @ attn_gate)` per head [128]
(`ssm_norm`), then `ssm_out [6144→2560]`. This is tensor-for-tensor and math-for-math the
qwen35 gated-delta block (HF class inherits `Qwen3_5GatedDeltaNet`; only the norm class
changed, semantics equal) → reuse `qwen_delta_recurrent{,_split,_rows,_chunk}` unchanged.
Confirm the gate activation inside gated norm matches qwen35's kernel (expected: silu(z)
per Qwen3-Next lineage) via the module oracle.

## MoE (48 layers)

`router_logits = x @ ffn_gate_inp` (f32) → softmax(fp32) → top-10 → renorm (norm_topk_prob
=True) → weighted sum of expert SwiGLU (silu(x@gate)*(x@up))@down, expert dim 640.
Shared expert: same SwiGLU, dim 640, output scaled by `sigmoid(x @ ffn_gate_inp_shexp)`
(scalar), added to routed sum. Identical to the existing qwen MoE + `qwen_shared_scale`.

## Top-level

- Embedding: `token_embd` lookup (no multiplier), then repeat×4 into streams.
- Final: head-collapse hyper-connection (`output_hc_*`) → final `output_norm`?? —
  **CHECK**: tensor inventory has NO `output_norm.weight`. The head-collapse `grouped_rms`
  with `output_hc_norm` IS the final norm; LM head `output [2560→248320]` applies directly
  to the collapsed 2560. (HF: `self.norm` — verify whether Qwen4ExpTextModel has a final
  `norm` after the mixer; the missing GGUF tensor says no / it's fused. Resolve in oracle.)
- No tied embeddings. eos 248046 (`<|im_end|>` presumably), bos/pad 248044, add_bos false.
- Sampling defaults from GGUF: temp 1.0, top_p 0.95, top_k 20 (thinking-mode defaults).
- Context 262144; YaRN beyond (not in scope).

## Config values (GGUF metadata, all parsed & confirmed)

48 layers, full_attention_interval 4 (pattern LLLF×12); hidden 2560; heads 24/2 kv,
head_dim 256, rotary 64; hc_count 4, low_rank 320; experts 512 top-10 + shared, dims 640;
delta: 16 QK / 48 V heads × 128, conv 4; ple: ngram 3, 8 heads/ngram, conv k4 (dilation 3
= ngram_size, NOT in GGUF metadata — hardcode from ngram_size), embed/head 160;
indexer: 4 heads × 128, 1 kv, top_k 2048, compress_ratio 4 (per-layer array, 0 on delta
layers); vocab 248320; rms_eps 1e-6; rope θ 1e7.

## Open questions — RESOLVED 2026-08-26 (oracle run + local modeling source)

1. Final norm after head collapse: **ABSENT.** `Qwen4ExpTextModel.forward` ends with
   `self.hyper_connection_mixer(hidden_states)` and returns; lm_head applies directly.
   (modeling_qwen4_exp.py:1430)
2. DeltaNet z-gate activation: `output_gate_type="sigmoid"` → RMSNormGated(sigmoid).
   CORRECTION (2026-08-26): the sequential kernels actually applied SILU (qwen3.5's
   unset output_gate_type falls back to hidden_act); the :1352 comment refers to a
   different block. `qwen_delta_recurrent{,_split}` grew a trailing `gate_sigmoid`
   arg (silu when 0, sigmoid when 1); rows/chunk variants still silu-only and get the
   flag when the rows path is hooked in phase 2.
3. Real-file norm centering (baked +1): still to eyeball on the downloaded shard
   (phase 2, cheap).
4. QSA indexer exact scoring/selection math: phase 3, extract from
   `Qwen4ExpTextQSAIndexer` + refs/qwen4_exp/llamacpp_qwen4exp.cpp.
5. `embedding_length_per_layer_input = 160` is the per-head table row width; HF
   `ple_embed_dim = 2560` is the 16-head concat. Confirmed by the oracle gather.

Oracle run (native/tools/qwen4exp_reference_check.py vs transformers 5.16.0.dev0 main):
hc_mix/hc_inject/head-collapse ≤1.3e-6, ngram gather EXACT (0.0), ple whole ≤2.2e-6,
ple split==whole ≤1e-6. All green on first run.
