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
resident. It must run mmap'd off NVMe with experts paged per token (~3 GB of expert reads
per decode step), so ~1–2 tok/s is the ceiling even when everything is correct. Decision
taken: correctness first, paging throughput is a separate problem later.

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
