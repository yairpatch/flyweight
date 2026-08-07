# DeepSeek-V4-Flash (`deepseek4`) support, targeting UD-IQ3_XXS

## Context

We want `colibri-next` to serve `unsloth/DeepSeek-V4-Flash-0731-GGUF`, primarily the
`UD-IQ3_XXS` build. Today the native runtime accepts only Qwen, Gemma 4 and Laguna
(`native/src/v2_runtime.cpp:3662`), so the model cannot even be opened.

Facts established from the checkpoint's own GGUF header (shard 1 of
`UD-IQ3_XXS`, parsed directly; the HF card's `dflash` tag is not what the file says):

- `general.architecture = deepseek4`, `general.size_label = 256x8.4B`, ~280B total params
- 43 blocks, `embedding_length` 4096, context 1 048 576, YaRN factor 16 over a
  65 536 original context
- MLA attention: `head_count` 64, `head_count_kv` 1, `key_length`/`value_length` 512,
  `rope.dimension_count` 64, `attention.q_lora_rank` 1024
- MoE: 256 experts, 6 used, 1 shared, expert FFN 2048, sigmoid gating
  (`expert_gating_func` 4), `expert_weights_scale` 1.5, `expert_weights_norm` true,
  per-layer `swiglu_clamp_exp` / `swiglu_clamp_shexp`
- New to this runtime: `attention.compress_ratios` (46 entries of 0/4/128),
  `attention.compress_rope_freq_base` 160000, `attention.indexer.{head_count 64,
  key_length 128, top_k 512}`, `attention.output_group_count` 8,
  `attention.output_lora_rank` 1024, `hyper_connection.{count 4,
  sinkhorn_iterations 20, epsilon 1e-6}`, `hash_layer_count` 3,
  `attention.sliding_window` 128
- Tokenizer: gpt2 BPE, 129 280 tokens, `tokenizer.ggml.pre = joyai-llm` (unknown to us),
  DeepSeek control tokens, DSML-flavoured tool-call chat template
- Distribution: 4-shard split GGUF, `split.tensors.count` 1328, ~104 GB total.
  Shard 1 carries all metadata and **zero** tensors.

Per the reference port (llama.cpp PR ggml-org/llama.cpp#24162 and the teamblobfish
write-up), `compress_ratios` selects the per-layer attention kind: `0` = sliding-window
(128), `4` = Compressed Sparse Attention — 4 tokens compressed to 1, lightning indexer
picks top-512 — and `128` = Heavily Compressed Attention (dense MQA over 128x-compressed
tokens plus the SWA stream). Every block carries 4 hyper-connection streams mixed by a
col-norm-first Sinkhorn router. That port is ~1500 lines of model code plus five custom
ops with hand-written kernels; there are three K caches in flight per layer.

Hardware reality on this box (60 GB RAM, 12 GB VRAM, 184 GB free disk): 104 GB cannot be
resident. It must run mmap'd off NVMe with experts paged per token. Decision taken:
correctness first, paging throughput is a separate problem later.

Throughput here is not a single number, and measuring it once gives the wrong one. On the
reference implementation:

- A cold one-shot run -- load, short prompt, six tokens -- gives 0.37 tok/s prompt eval and
  0.42 tok/s generating, after a 76-second load.
- The same machine serving successive requests reaches 1.81, then 2.75, then 3.22 tok/s
  generating, with prompt eval climbing 0.53 -> 1.20 -> 1.69 over the same three requests.

The difference is page-cache warmth and graph reuse, not variance: reused graphs went 128 ->
245 -> 425 across those requests while the expert weights settled into cache. The steady
state is roughly 3 tok/s, which is usable; the cold figure is not representative of
anything except the first request after a boot.

The lesson generalizes past this model, and this project has already written it down once
for GPU clock ramp: take throughput at steady state over sustained work, never from a cold
one-shot run. An earlier revision of this plan recorded 0.4 tok/s as the real number on the
strength of exactly such a run, which was wrong by nearly an order of magnitude.

Work lands in gated stages; each stage is independently useful and mergeable.

## What already exists and should be reused

- Memory-mapped GGUF load with `MADV_HUGEPAGE`, optional mlock, CUDA host registration:
  `colibri_v2_model_open`, `native/src/v2_runtime.cpp:3246`
- Suffix-matched GGUF metadata parsing: `parse()`, `native/src/v2_runtime.cpp:1294`.
  Most `deepseek4.*` keys (block_count, embedding_length, head_count(_kv), key/value_length,
  expert_count, expert_used_count, expert_feed_forward_length, rope.*, YaRN,
  expert_weights_scale/norm, sliding_window) already land through existing suffix rules.
- DeepSeek-V3-style sigmoid top-k routing with score-correction bias:
  `native/include/colibri_v2_native_kernels.hpp:867`
- Shared-expert and leading-dense-block config plumbing: `ModelConfig`,
  `native/include/colibri_v2_config.hpp`
- IQ3_XXS decode and dot kernels on CPU: `native/src/qwen_kquant.h:207`,
  `native/src/qwen_cpu_avx2.cpp:737`, `native/src/qwen_cpu_avx512.cpp`, block-size
  constant at `native/src/v2_runtime.cpp:620`
- Expert paging straight from the mmap and hybrid CPU/GPU expert placement
  (`native/src/v2_runtime.cpp:4778` onward)
- Hand-transcribed BPE pre-tokenizer precedent: `laguna_pretokenize`,
  `native/src/v2_runtime.cpp:3436`
- Second-GGUF sidecar precedent for the MTP module: `ColibriV2Model::mtp_sidecar`,
  `native/src/v2_runtime.cpp:88` and `colibri_v2_model_attach_mtp`, `:3323`

## Stage A — load, describe, tokenize (no execution)

Goal: `colibri-next` opens the 4-shard IQ3_XXS checkpoint, reports a correct config and
tensor plan, and tokenizes text identically to the reference.

1. **Split-GGUF mapping.** `colibri_v2_model_open` maps exactly one file today and
   `Tensor::offset` is an offset into that single mapping. Add a shard vector to
   `ColibriV2Model` (fd/handle, base pointer, size), derive sibling paths from the
   `-0000N-of-0000M.gguf` name when `split.count > 1`, map each shard, and give `Tensor` a
   shard index so `WeightProvider` resolves `shard_base + offset`. Every place that
   assumes `m->data`/`m->size` is the whole model must iterate shards: mlock, `madvise`,
   CUDA host registration, the direct expert-paging registration path
   (`native/src/v2_runtime.cpp:4778`), and the trailing-size computation in `parse()` that
   derives `Tensor::size` from the next tensor's offset (that must become per-shard).
   Validate `split.no`, `split.count` and `split.tensors.count` across shards and fail
   loudly on a missing or mismatched shard.
2. **`deepseek4` metadata.** Extend `ModelConfig` (`native/include/colibri_v2_config.hpp`)
   with: `q_lora_rank`, `kv_lora_rank` (from `key_length`), `output_lora_rank`,
   `output_group_count`, `indexer_head_count`, `indexer_key_length`, `indexer_top_k`,
   `compress_ratios` (vector), `compress_rope_freq_base`, `hyper_connection_count`,
   `sinkhorn_iterations`, `sinkhorn_epsilon`, `expert_shared_count`, `hash_layer_count`,
   and the two per-layer swiglu clamp vectors. Add matching suffix rules in `parse()`.
   Note `expert_shared_count` (1) is a *count*, not the existing
   `expert_shared_intermediate_size`; the shared FFN width is
   `expert_feed_forward_length * expert_shared_count` = 2048.
   Open item to resolve from the reference source: `compress_ratios` has 46 entries for 43
   blocks — establish what the extra 3 map to (likely the MTP/next-N and hash layers)
   before relying on the indexing.
3. **Tensor plan and quant coverage.** Enumerate the 1328 tensors across shards, group them
   into the deepseek4 roles (`attn_compressor_*`, `indexer_*`, `hc_attn_*`, `hc_ffn_*`,
   `output_hc_*`, MLA `q_a/q_b/kv_a/kv_b`, MoE, shared expert), and assert every tensor's
   GGML type is one this runtime can decode. Unsloth UD quants mix types per tensor, so
   this scan is the point where an unsupported type (e.g. an IQ variant we lack) surfaces
   as a clear error rather than garbage output.
4. **Tokenizer.** Transcribe the `joyai-llm` pre-tokenizer regex from the reference into a
   `deepseek4_pretokenize` next to `laguna_pretokenize` (`native/src/v2_runtime.cpp:3436`),
   and register the DeepSeek control tokens (`<｜begin▁of▁sentence｜>`,
   `<｜User｜>`, `<｜Assistant｜>`, `｜DSML｜`, `<think>`) so they never split.
5. **Arch gate and Python surface.** Accept `deepseek4` for inspection at
   `native/src/v2_runtime.cpp:3662`, and extend the arch branches in
   `src/colibri_next/v2.py` (`_architecture` at `:719`, the gemma4 branch at `:1071`, the
   dispatch at `:1144`) plus whatever the CLI needs to print the new config fields.

Stage A verification: a new `tests/test_v2_deepseek4.py` opens the local checkpoint (skipped
when absent) and asserts arch, 43 blocks, 4096 hidden, 256/6 experts, the compress-ratio
vector and 1328 tensors; a tokenizer round-trip test compares token ids against
`llama-tokenize` from the reference build on a fixed multilingual + control-token corpus.

### Stage A status: complete, pending real-checkpoint verification

All five items landed; 190 tests pass. Resolved along the way:

- The `compress_ratios` open item: the reference requires `size() >= n_layer` and
  reads the first `n_layer` entries, ignoring the rest. The loader now does exactly
  that, and rejects a short array.
- `Tensor::source` already existed for the MTP sidecar, so split shards reuse it
  rather than needing a new indirection.
- The `joyai-llm` patterns were transcribed from the reference and are checked
  against those same regexes run through the `regex` module, over a corpus covering
  CJK, Korean, Arabic, Cyrillic, emoji and whitespace edges. Unicode categories come
  from a table generated out of llama.cpp's `unicode-data.cpp`
  (`tools/generate_unicode_categories.py`), so category decisions match the reference
  exactly instead of using the codebase's existing range approximation.

Verified against the real UD-IQ3_XXS checkpoint (all four shards, 1328 tensors):
geometry, split mapping, compress ratios, tokenizer and the weight-type scan all
pass. Set `DEEPSEEK4_GGUF` to shard 1 to run those tests.

The type histogram over the real file is F32 662, Q8_0 321, Q6_K 170, IQ3_XXS 75,
IQ2_XS 50, BF16 43, IQ3_S 2, **I32 3**, **MXFP4 2**. Everything except the last two
already decodes, which turns two plan assumptions into facts:

- **`hash_layer_count: 3` explained.** The first three blocks carry
  `blk.{0,1,2}.ffn_gate_tid2eid.weight`, an I32 `[expert_used_count, vocabulary]`
  table — those blocks route by token id through a lookup table instead of a
  learned router. This is routing work in Stage B, not a decode kernel.
- **MXFP4 (type 39) is a new format to implement.** Two tensors,
  `blk.{26,42}.ffn_down_exps.weight`, at 17 bytes per 32 values (4.25 bits). The
  existing NVFP4 (type 40) is 4.5 bits over blocks of 16, so it cannot be reused
  as-is, though its kernel is the right starting point.

Both are pinned by `test_the_only_undecodable_types_are_the_two_known_gaps`, so a
third gap would fail the suite rather than surface mid-Stage-B.

## Stage B — CPU forward pass at logit parity

Reference-driven, correctness only. Build the llama.cpp `deepseek4` fork locally first and
dump per-layer activations on a fixed short prompt; every sub-step below is landed against
that dump rather than by inspection.

### Grounded facts (from upstream llama.cpp and the checkpoint's own shapes)

`deepseek4` is in upstream `ggml-org/llama.cpp` master; the reference is built at
`/home/yair/Desktop/llama.cpp-ref` (CPU-only) with `llama-eval-callback` for
per-tensor dumps. Tokenization is already verified byte-identical against it.

Layer layout, read off the checkpoint: ratios are `0, 0` then `4, 128` alternating
through layer 42 (so 42 is a 4). Blocks 0–2 are the hash layers. The three trailing
ratio entries are `0, 0, 0` for the draft blocks.

Per-block tensors and what their shapes imply (4096 hidden, 64 heads):

- MLA: `attn_q_a` 4096→1024, `attn_q_b` 1024→32768 (64 heads x 512), `attn_kv`
  4096→512 shared by every head, `attn_kv_a_norm` 512, `attn_sinks` 64 (one
  learned sink logit per head), output as `attn_output_a` 4096→8192 and
  `attn_output_b` 8192→4096, where 8192 = `output_group_count` 8 x
  `output_lora_rank` 1024.
- Hyper-connections: `hc_{attn,ffn}_fn` is `[hc*n_embd, (2+hc)*hc]` = 16384x24,
  `hc_*_base` 24, `hc_*_scale` 3. The head's `output_hc_fn` is 16384x4 with a
  1-wide scale, because it only needs the pre-weights.
- Compressors: width is `coff * 512` with `coff = 2` when the ratio is 4 and 1
  otherwise, and `attn_compressor_ape` is `[width, ratio]` — one position
  embedding per slot within a compressed block. Indexer compressors are the same
  shape at width `2 * 128`.
- Routing: blocks 0–2 carry an int32 `ffn_gate_tid2eid` `[6, 129280]`; blocks 3+
  carry `exp_probs_b.bias` 256 instead.

Hyper-connection maths, transcribed from `src/models/deepseek4.cpp`:

    flat      = reshape(x, [hc*n_embd, nt])          # x is [n_embd, hc, nt]
    mixes     = hc_fn @ rms_norm(flat)               # [24, nt]
    pre       = sigmoid(mixes[0:4]  * scale[0] + base[0:4])  + eps
    post      = sigmoid(mixes[4:8]  * scale[1] + base[4:8])  * 2
    comb      = sinkhorn(reshape(mixes[8:24] * scale[2] + base[8:24], [hc, hc, nt]))
    block_in  = sum_h x[:, h, :] * pre[h]
    x_out[dst]= block_out * post[dst] + sum_src x[:, src, :] * comb[dst, src]

`sinkhorn` is softmax over dst, then `+eps`, then one column normalization,
then 19 further (row, column) pairs — 20 iterations total, dividing by sums that
each have `eps` added.

### The parity loop

`llama-eval-callback` on the real checkpoint dumps every intermediate tensor with
its name, op, shapes and a `sum` checksum, and a one-token prompt completes in
about three minutes:

    cd /home/yair/Desktop/llama.cpp-ref
    ./build/bin/llama-eval-callback -m <shard 1> -p "The" -n 1 -c 256 --temp 0 > dump.txt

That checksum per tensor is what each component gets diffed against, which avoids
needing full activation dumps. The layer-0 entries already confirm the
hyper-connection derivation above: `hc_mixes-0` is `MUL_MAT(hc_attn_fn{16384,24},
{16384}) = {24}`, `hc_attn_pre-0` is `DSV4_HC_PRE(hc_init{4096,4}, {4}) = {4096}`,
and `attn_wo_a-0` is `MUL_MAT(attn_output_a{4096,1024,8}, {4096,1,8}) = {1024,1,8}`.

Running the miniature fixture through the reference instead would give a faster
loop, and the fixture is now close enough that the reference loads it (its tensor
set and shapes are accepted). It still aborts in `build_attention_impl` on a
reshape, so some geometric relationship beyond the two already preserved
(`(heads/groups)*head_dim == hidden` and `heads*head_dim == 8*hidden`) is
required. Worth finishing, but the real-checkpoint loop is unblocked and is what
the components are being diffed against for now.

### Corrections to earlier assumptions, from the reference and the dumps

Three things this plan originally got wrong, each found by checking rather than
by reading:

- **RoPE configuration is per layer, not global.** `use_compress_rope` is
  `compress_ratios[il] != 0`, so layers 0 and 1 rotate at the model's own
  `rope.freq_base` (10000) with no scaling and no YaRN at all, while the
  compressed blocks use `attention.compress_rope_freq_base` (160000) *with*
  YaRN. The plan's "YaRN scaling at factor 16" is right only for the latter.
  The attn factor the reference passes there,
  `1/(1 + 0.1*ln(1/freq_scale))`, exactly cancels the one ggml applies
  internally, so the net magnitude scaling is 1.
- **The router is `sqrt(softplus(logits))`, not sigmoid.** `expert_gating_func`
  4 and the DeepSeek-V3 precedent both suggested sigmoid; the graph shows
  `ffn_moe_probs = SQRT(SOFTPLUS(ffn_moe_logits))`. The existing
  sigmoid-with-bias router kernel is therefore not reusable as-is.
- **Hash layers do not compute a routing decision at all.** Blocks 0-2 do
  `ffn_moe_topk = GET_ROWS(ffn_gate_tid2eid, inp_tokens)`: the expert ids come
  straight out of the int32 table indexed by token id, so there is no router
  arithmetic to match, only a lookup.

Also worth noting for the attention core: keys and values are the same tensor,
so there is one K cache and no V cache, and each head's sink logit joins the
softmax without contributing a value.

### Comparing sums was measuring noise; compare values instead

Layer 2 looked badly wrong on a 376-token prompt: 8.28% on its block output,
and 154% on `attn_out` while the `derope` feeding it was 2.81%. Its experts and
weights matched the reference exactly, which for a hash layer they must, so the
arithmetic in between was the suspect.

There is no defect. The dump prints values as well as sums, and element for
element `attn_out` agrees:

    pos 0   ours  0.4221 -0.8885 -1.2575    ref  0.4269 -0.8955 -1.2629
    pos 1   ours -0.1702 -0.3229 -0.8092    ref -0.1854 -0.3372 -0.8138
    pos 2   ours -0.3866 -0.8384 -1.6104    ref -0.3943 -0.8321 -1.6048

That is half a percent to two percent per element, which is the
activation-quantization band every other component sits in. The 154% came from
summing 1.5 million values whose mean is near zero: `attn_out` totals 3663
across them, so an ordinary per-element error moves the total by multiples of
itself while changing nothing that matters.

This invalidates the method, not just this measurement. Sum comparison has been
the workhorse here because the dump makes it cheap, and on tensors with a large
coherent total -- attention output, the compressed keys -- it is informative. On
anything centred near zero it is not, and several figures recorded earlier are
suspect for that reason: the tail divergence from layer 33, the routed-expert
output, the feed-forward norm. Each was read as error growth and may be nothing
of the kind.

Anything relying on a relative error over a near-zero sum should be re-checked
against printed values before it is believed, and new checks should prefer
element comparison where the dump provides it.

Re-running the whole 43-layer sweep that way, on the 376-token prompt and
comparing the first and last three values of each block's output rather than its
sum:

- Layers 8 through 33 agree to between 0.28% and 2.0%. Twenty-six consecutive
  layers in the activation-quantization band is the strongest evidence so far
  that the implementation is right.
- Layers 0 through 7 read between 1.2% and 105%, but their values are around
  0.01 to 0.03, so this is the near-zero trap again at element scale rather than
  sum scale. Nothing can be concluded from them either way.
- Layers 36 through 42 climb from 8.7% to 47% on values of magnitude 500 to
  1300. That is large enough to be real, and it survives the change of method.

So the tail divergence is genuine and the middle of the network is not. The
onset sits around layer 36 rather than 33. Given the routed expert selections
differ from the reference and the feed-forward gain in those layers is around a
hundredfold, a single flipped expert late in the stack is the obvious candidate,
and it would be a legitimate consequence of the activation-precision difference
rather than a defect. Establishing that needs the reference's own hidden state,
which the dump does not carry.

### The stack agrees with the reference behaviourally, not token for token

On the 376-token prompt -- three identical paragraphs -- the reference's greedy
continuation is "2. The history", having noticed the repetition and started
numbering it. Our top prediction is "3", with "ĠThe" second and "2" fourth. The
reference's next token after its digit is "The", which is our second choice.

So both produce the same behaviour and differ on which digit. The reason is
visible in the logits: the top six candidates span 1.07, and five of them are
digits. The model is close to indifferent there, so the one-to-two percent
activation difference between the two implementations is more than enough to
reorder them.

This is the concrete case the acceptance criterion was rewritten for. Token-for-
token agreement fails here, and would fail on any near-tie, without indicating a
defect in either implementation. What it does establish is that a 376-token
prompt survives 43 layers -- including both compressed attention kinds and the
tail whose element-wise error reaches 47% -- and still produces a sensible,
reference-like continuation. The tail divergence does not appear to destroy the
output.

### Expert selection diverges from the reference, and probably always will

Running the full stack and comparing block outputs, layers 0-32 agree to within
7% and the tail then falls apart, reaching 50%. The mechanism is now located but
not fully settled.

At layer 34 the feed-forward takes an input summing to 151.6 and returns
-17129 -- a hundredfold gain. Late layers are dominated by a few very large
channels, so a small relative error upstream swamps the sum downstream.

More importantly, the routed expert *selections* differ from the reference at
every layer sampled (3, 20, 33, 34, 40), including layer 3 where the block
output was still within 2.9%. Selection is a top-6 of 256 on scores that are
often close together, so a one-percent perturbation is enough to swap the sixth
and seventh choice. The size of the differences -- sums off by 30 to 220, where
a single swapped id can move the sum by up to 255 -- is consistent with one or
two swaps out of sixty selections, which is what numerical noise would produce
rather than a systematic fault.

That has not been proven: distinguishing noise-driven flipping from a router
defect would need the reference's own hidden state fed through our router, and
the dump gives sums rather than activations. What is established is that
selections differ, that the difference is small in count, and that late layers
amplify whatever it causes.

The consequence for the acceptance criterion is the part worth acting on. This
runtime dots f32 against dequantized weights where ggml requantizes activations,
so the two implementations disagree slightly at every matmul by construction.
Where that disagreement crosses an expert-selection boundary the outputs diverge
sharply and legitimately. Token-for-token agreement with llama.cpp may therefore
not be achievable at all, and treating it as the pass mark risks chasing a
difference that is not a defect. A better bar is coherent output plus agreement
on most tokens, with the expert-selection rate measured rather than assumed.

### Short prompts do not exercise most of the compressed machinery

A block only compresses once `ratio` tokens have accumulated, so on a prompt
shorter than the ratio nothing is compressed and the compressed cache stays
empty. For HCA that ratio is 128, and the sliding window is also 128, so on any
prompt below 128 tokens an HCA layer attends to exactly the raw window. Confirmed
against the dump: layer 3 still projects and stores its per-token compressor
state into the 128-slot buffer, but its attention is `FLASH_ATTN_EXT` over
`cache_k` alone, with no compressed keys concatenated. Twenty-one of the 43
layers are HCA, so on a short prompt they need the state bookkeeping and no
compressed-attention path at all.

CSA is the one that bites early: at ratio 4 a ten-token prompt already completes
two blocks. Its indexer, though, selects the top 512 compressed tokens and there
are two, so selection is a no-op at that length -- the compressed latents still
have to be computed correctly, but the top-k path is not being tested by a short
prompt even when it runs.

The consequence for bring-up: a first full-model run on a short prompt needs the
CSA compressor and nothing else of the compressed machinery, and passing it says
nothing about HCA compression, the indexer's selection, or eviction. Those need
a prompt of at least a few hundred tokens, which is worth arranging deliberately
rather than assuming the short-prompt result generalizes.

### What the compressed kinds still need

The kernels are in place; what is missing is state, not arithmetic.

**HCA (ratio 128) needs no new attention kernel.** It concatenates the raw
sliding-window K cache with the compressed-block cache along the position axis,
concatenates their masks the same way, and runs one attention over the union.
That is the existing `attention()` given a longer latent array and mask, so the
work is entirely in maintaining the two caches.

**CSA (ratio 4) adds the lightning indexer** on top of the same shape: a
separate small projection (`indexer.proj`, `indexer.attn_q_b`) with its own
compressors and K cache produces a score per compressed token, top-k selects
which ones the main attention may see, and the mask is built from that
selection. `indexer_top_k` is 512, so it only bites past 512 compressed tokens.

**A Hadamard rotation wraps the compressed path.** Where `attn_inp_k_rot` is
present the reference rotates q and kv through it before attention and rotates
the output back. Being orthogonal it leaves the scores unchanged in exact
arithmetic; the point is the cache, since the stored latents are the rotated
ones and the rotation spreads quantization error evenly across channels. That
makes it a no-op to skip while the cache is f32 and a correctness requirement
once it is not -- worth knowing before choosing a cache type, and the same
motivation as the existing turboquant work.

Order of work, each diffed against the reference before moving on:

1. Hyper-connections: 4-stream expand, col-norm-first Sinkhorn (20 iterations, ε 1e-6),
   weighted sum, and the width-4 residual carried through the block.
2. MLA: q_lora 1024 down/up, kv latent 512, decoupled 64-dim RoPE, YaRN scaling at factor
   16 over the 65 536 original context.
3. Per-layer attention branch on `compress_ratios`: SWA(128) for 0; CSA for 4 — the 4:1
   token compressor, the lightning indexer with its own compressors and K cache, top-512
   selection; HCA for 128 — dense MQA over the 128x compressor plus the SWA stream.
4. The three per-layer caches (SWA K, compressed K, indexer K) and their compression plans.
5. MoE: 256 experts / 6 active with `expert_weights_scale` 1.5 and norm, the shared expert,
   and the per-layer swiglu clamps — extending the existing sigmoid-router kernel at
   `native/include/colibri_v2_native_kernels.hpp:867` rather than writing a new one.
6. Output head with `output_lora_rank` 1024 / `output_group_count` 8.

Context length must be capped well below 1M initially (32–64k) — the three-cache layout at
1M is not affordable here; make the cap explicit and error rather than silently truncate.

Stage B verification: greedy generation from a fixed prompt matches the reference token for
token for at least 64 tokens; per-layer activation max-abs diff stays within the tolerance
the quantization allows; a perplexity spot check on a short held-out text tracks the
reference.

## Stage C2 design — what the native loop must hold

The state a sequence needs is far smaller than it first appears, and working out
why changes where the effort belongs.

**The raw latent cache holds the window, not the context.** Attention over raw
tokens is bounded by the 128-token sliding window on every layer, so only 128
latents per layer need retaining -- a ring buffer, not a growing cache. At 512
wide in f16 that is 128 KiB per layer.

**The compressor state holds two blocks, not the sequence.** A 4:1 layer pools
the previous block's rows and its own, so once a block closes only its rows and
the partial block after it can still be read: 2*ratio rows, eight of them. A
128:1 layer does not overlap, so it needs only its own partial block. The
composed model keeps every position because that is simpler for an oracle, and
copying that into the runtime would waste the sequence's worth of memory for
nothing.

**The compressed caches do scale with context**, at context/4 entries for a 4:1
layer and context/128 for a 128:1 one, plus the indexer's own cache on 4:1
layers.

Totalled across 43 layers: 37.9 MiB at 4096 context, 226 MiB at 32768. Against
104 GiB of weights that is nothing, and three conclusions follow.

- Cache quantization is pointless here. The existing turbo3/turbo4 KV types earn
  their keep on models whose KV rivals their weights; on this one they would
  save single-digit megabytes and cost accuracy.
- Context length is nearly free in memory terms. What limits it is the lightning
  indexer, unimplemented and needed beyond ~2048 tokens, not the cache budget.
- Effort belongs on expert paging. The weights are the entire memory problem,
  and the existing paging machinery is what determines whether this runs at 3
  tok/s or 0.3.

The loop itself should prefill in batches rather than a token at a time. The
composed model runs one position at a time through a matvec per weight, which is
right for an oracle and wrong for a runtime: it rereads every expert row per
token instead of amortizing it across a batch, which is precisely the cost the
paging machinery exists to manage.

## Stage C — GPU / hybrid execution

Move attention, hyper-connections and the shared/dense path onto the 12 GB GPU while
experts stay CPU-side and paged from the mmap, using the existing hybrid placement policy.
Hand-written kernels are needed for the Sinkhorn mixing, the compressor, the indexer top-k
and the RoPE tail — this runtime has no graph engine to compose them, which is why this is a
separate stage from B. Success is measured against the Stage B CPU output, not re-derived.

## Stage D — MTP and serving polish

Attach the speculative-decoding module through the existing sidecar mechanism
(`colibri_v2_model_attach_mtp`, `native/src/v2_runtime.cpp:3323`), and teach the server the
DSML tool-call dialect — the chat template emits `<｜DSML｜tool_calls>` / `invoke` /
`parameter string="true|false"` blocks, which none of the existing parsers in
`src/colibri_next/server.py:83` understand. Also handle the `thinking` /
`reasoning_content` fields the template expects.

## Prerequisites

- Download `UD-IQ3_XXS` (~104 GB of 184 GB free) plus a reference llama.cpp build. Disk is
  adequate but not roomy; the reference build and the checkpoint together leave ~70 GB.
- GPU timings, when Stage C is measured, must follow the existing clock-ramp discipline for
  this laptop GPU.

## Gate

Stage A merges and is reviewed before Stage B starts; likewise B before C. If Stage A's
tensor-type scan turns up a quant this runtime cannot decode, that becomes its own piece of
work before B.
