# qwen4exp: Qwen3.8-Flash-Next support

> **STATUS 2026-08-26 end of day**: Phases 0, 1 DONE; phase 2 DONE except the
> multi-sequence hooks. The real UD-IQ1_S checkpoint SERVES: coherent text, decode
> 7.0 tok/s, prefill ~16 tok/s (expert-scalar-decode-bound — GPU IQ1_S/IQ4_NL expert
> kernels are the next perf lever). Decode AND rows paths carry the gated-residual +
> PLE hooks; every delta kernel variant (sequential/rows/chunk-128/WY) has the
> `gate_sigmoid` flag; fixture prefill==decode token-exact on CPU and CUDA backends,
> including a head_dim-128 fixture that exercises the chunk and WY paths; 64-token
> transformers parity green; ctest 21/21 + pytest 562 green. Real-model prefill vs
> decode drifts a few tokens in (batch-vs-single CPU-MoE summation order on the
> quant — fixture-exact, not a bug). PHASE 2 COMPLETE: multi-seq hooks shipped —
> interleaved==solo token-exact, --parallel guard lifted (inside qwen_decode_multi
> the engine has PARKED the active slot, so per-slot processed_tokens is
> authoritative for every slot — the usual active-mirror accessor is wrong there).
> Optional follow-ups: real-checkpoint layer-dump audit, prefill perf (GPU
> IQ1_S/IQ4_NL expert kernels), CUDA-graph re-enable for the arch.
> Phase 3 (QSA indexer) not started.
> Key findings log: no final norm (head collapse is last); z-gate sigmoid vs qwen35
> silu; ssm_norm is RMSNormGated = NO +1 baking (all other norms baked);
> qwen_dequant_row needed types 19/20; PLE conv is dilated (dilation=ngram_size).

## Context

Add support for [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) — GGUF arch `qwen4exp`, "a preview of the Qwen4 architecture". 125B total / 6B active, hidden 2560, 48 layers in a 3×(Gated DeltaNet→MoE) → 1×(sparse attention→MoE) pattern, 512 experts (10 routed + 1 sigmoid-gated shared), 4-stream hyper-connection residual, a 25.7 GB hashed n-gram embedding table (PLE) injected at blk.1, vocab 248320, ctx 262144. The unsloth UD-IQ1_S file (72.5 GB, 3 shards; shard 1 is metadata-only) has **no MTP and no vision tensors** — both stripped at conversion, so both are out of scope. No mature reference runtime exists: llama.cpp support is an unmerged PR (#27739, opened 2026-08-26); exact forward semantics live in transformers PR #48337 (`modeling_qwen4_exp.py`, transformers 5.8.0.dev).

Decisions made with the user:
- **QSA dense fallback first**: the 12 full-attention layers run as dense GQA in phase 1/2 (exact while context fits the 2048-token indexer budget, approximate beyond); the learned indexer is phase 3.
- **Full E2E on this box**: download the 72.5 GB GGUF and serve it hybrid (experts + PLE table host-side).
- GGUF loading only (no HF safetensors loader for this arch). Build on `v2-native-runtime`; the uncommitted format-dispatch (IQ4_XS) and transcript-audit changes must not be disturbed — format-dispatch edits strictly additive.

## What's already there (verified in code)

- Arch acceptance is free: `qwen*` routes to the qwen runtime (`v2_runtime.cpp:11631`); plan-builder dispatch at `:11708`.
- **DeltaNet layers are tensor-for-tensor the qwen35 set** (`attn_qkv/attn_gate/ssm_out/ssm_alpha/ssm_beta/ssm_conv1d/ssm_dt.bias/ssm_a/ssm_norm`, registered at `:2219-2223`). Geometry (48 V-heads ×128, 16 QK-heads ×128, conv 4) is derived from shapes and satisfies `use_chunked_delta`. Kernels `qwen_delta_recurrent{,_split,_rows,_chunk}` reuse as-is, pending phase-0 confirmation of alpha/beta order.
- **Fused q|gate + sigmoid output-gate attention at head_dim 256 exists** (decode `:15796-15912`, `qwen_attention_query` splits per-head `[q|gate]`, `qwen_attention_gate` sigmoid; 256-wide fused KV tiles + turbo cache OK). Only delta: RoPE is **interleaved** here (existing kernels are half-split/NeoX) over rotary_dim 64. M-RoPE sections `[11,11,10,0]` ignored for text (qwen35moe precedent).
- **Shared expert + sigmoid scalar gate (`ffn_gate_inp_shexp`) is a no-op** — plan slots `:2236-2240`, `qwen_shared_scale` at decode `:16054`, rows `:18712`, CPU `:13942`.
- Config parser already handles `hyper_connection.count`, `full_attention_interval`, indexer keys, `compress_ratios`, `embedding_length_per_layer_input`. Missing: `hyper_connection.low_rank`, all `ple.*` keys; `read_uint_array` (`:1889`) rejects u64 values → need a u64-array reader for `ple.layer_multipliers/head_offsets/head_vocab_sizes`.
- **IQ4_NL (type 20) is NOT in the format-dispatch table**, but the codebook `kIq4nlValues` exists (`qwen_iq_tables.h:781`). Needed for `ffn_down_exps` (CPU expert dot) and PLE row decode. IQ1_S has `cpu_expert=true` but no GPU expert-prefix entries → bring-up mode is hybrid with CPU experts.
- All three forward paths (decode `:15533+`, `qwen_forward_rows` in `v2_mtp_verifier.inc:95`, `qwen_decode_multi` `:18420+`) share the same per-layer bookend idiom `rms → block → add → rms → FFN → add → swap` — hyper-connections replace exactly those bookends; block bodies never touch the residual.

## The genuinely new work

1. **Hyper-connections (every layer + head)** — residual becomes 4×2560 streams. Per block boundary: RMS in 10240 stream space (`hc_*_norm`) → low-rank mix (down 10240→320, up 320→10240) with elementwise gating → 2560 block input; block output injected back via `hc_*_inject [10240,4]`; final collapse via `output_hc_norm/down/up` before LM head.
2. **PLE n-gram embeddings (blk.1)** — per token, 16 hashed lookups (8 bigram + 8 trigram heads; u64 multipliers mod ~20M primes + head offsets) into the 320M-row × 160-dim IQ4_NL table (host mmap, never uploaded) → 2560 input; depthwise causal conv (k=4, **stateful** — joins the per-sequence state arena + snapshot copies) + `ple_norm_*` + `ple_key/ple_value` projections, gated injection.
3. **Interleaved partial RoPE** — `qwen4_attention_query/key`: copies of the existing pair with `pair = index/2` layout, rotary 64 of 256.
4. **QSA indexer (phase 3)** — bf16 `indexer.q_proj/k_proj` + norms, per-layer 128-dim index-key cache, top-k 2048 block selection feeding existing score/value kernels. Deepseek4's lightning-indexer is in-repo prior art.

## Design: no forked forward paths

Arch-gated `hc_pre`/`hc_post`/`hc_head_collapse` helpers replace the `rms`/`add` bookends at the ~8 call sites per path (decode, rows, multi-seq). New `build_qwen4exp_plan()` beside `build_laguna_plan` (dispatch `:11708`), **preserving qwen positional slots** so `qwen_ffn_base()` needs no new case: slot 0 = `hc_attn_norm`, moe_base = `hc_ffn_norm`, MoE slot list unchanged; everything else (`hc_*_down/up/inject`, indexer tensors, `ple_*`, `output_hc_*`, `per_layer_token_embd`) as named `QwenLayerPlan`/runtime fields (the `router_bias` precedent).

Workspace: don't widen `hidden`; add regions (`colibri_v2_workspace.hpp`, both layouts + contract test): `streams` 4×2560, `hc_wide` 10240, `hc_low` 320, `ple_embed` 2560 (+ rows variants). ~92 KB decode — negligible.

CUDA graphs **off for qwen4exp at bring-up** (the old residual `add` is inside `enqueue_delta`'s captured region); re-enable later by moving the hooks inside capture (they're position-invariant).

PLE host stage: `qwen_stage_ple_rows` next to `qwen_stage_embedding_rows` (`:3921`); `__int128` mult-mod; sequence-start padding with `ple.eos_token_id` (confirm in phase 0). Prefix reuse/rewind need no key changes — `QwenPrefillSnapshot::tokens` already carries the n-gram history; the conv state rides in the state-arena snapshot copy (`qwen_prefill_snapshot_copy :14106` extended).

### New kernels (GPU + CPU shim twin each)

| Kernel | Shape | Role |
|---|---|---|
| `hc_init` (+`_rows`) | 2560 → 4×2560 | stream init from embedding |
| `hc_mix_collapse` (+`_rows`) | 10240 → 2560 | epilogue after rms + down/up matvecs |
| `hc_inject` (+`_rows`) | 2560 + [10240,4] → 10240 | replaces residual add |
| `hc_head_collapse` | 10240 → 2560 | before final norm |
| `qwen4_attention_query/key` | 256-dim heads, rotary 64 | interleaved-rope variants |
| `ple_gate_inject` (+`_rows`), conv reuse/clone | 2560/10240 | PLE injection at blk.1 |
| host `iq4nl_decode_row` + IQ4_NL CPU expert dot | 32-elem blocks | table gather + experts |

## Phases and gates

**Phase 0 — semantics spec + oracles (no runtime code).**
Extract from transformers PR #48337 into `plans/qwen4exp-semantics.md`: exact hc math (norm scope, gating nonlinearity, inject formula, stream init), PLE hash formula/padding/injection, DeltaNet alpha/beta order (expected == qwen35), router activation, q|gate memory order, rope layout. Cross-check llama.cpp PR #27739 conversion for tensor orientation. Write `native/tools/qwen4exp_reference.py` (pattern: `kda_reference.py`) — numpy oracles pinned ≤1e-5 vs the PR-branch torch modules.
*Gate: every oracle module passes; open-questions list empty or explicitly deferred.*

**Phase 1 — GGUF parse, plan builder, CPU-backend E2E parity.**
Config fields + u64-array reader; `build_qwen4exp_plan`; state arena + snapshot coverage for PLE conv; workspace regions (+ `qwen_workspace_contract.cpp`); all new kernels as CPU shims wired through the three paths; additive IQ4_NL dispatch entry. New `tests/qwen4exp_gguf_fixture.py` (tiny: 8 layers interval 4, hidden 64, hc 4/rank 16, 8 experts, tiny-prime PLE table) modeled on `dense_gguf_fixture.py`; `tests/test_v2_qwen4exp_parity.py` (pattern: `test_v2_qwen35_parity.py`, CPU backend, F32, per-layer activation + greedy-token agreement vs PR-branch transformers, skip-if-absent) plus transformers-free pinned-vector contract tests `native/tests/qwen4exp_hc_contract.cpp`, `qwen4exp_ple_contract.cpp` (+ CMake). Extend `test_v2_prefill_parity.py` quant loop.
*Gate: new tests green; **full `ctest` green (run it — it has been silently red before)**; existing pytest green; fixture decode == prefill == transformers, token-exact ≥64 tokens.*

**Phase 2 — GPU kernels + real-checkpoint hybrid bring-up.**
CUDA twins of the new kernels (register names in `gpu_driver.cpp` — unknown names fail silently as fallback); graphs disabled for the arch; extend `test_v2_path_parity.py` (CUDA vs CPU differential). Download UD-IQ1_S (72.5 GB); serve hybrid: dense + hc weights GPU (small), experts + PLE table host mmap, CPU-expert mode; per-layer dump vs transformers on a short prompt (pattern: `bailing_reference.py`); coherence + throughput; README families row + limitations (no MTP/vision, QSA dense fallback + context caveat).
*Gate: CUDA fixture parity token-exact vs CPU; real-checkpoint dumps within quant tolerance; coherent chat via server; ctest+pytest green; perf recorded.*

**Phase 3 — QSA indexer.**
Index-key cache (128/token × 12 layers) in the state arena; compressed-block scoring + top-k 2048 selected-slot list into existing score/value kernels. Inactive ≤2048 tokens.
*Gate: bit-exact vs phase 2 below 2048 tokens; quality spot-checks + long-context perf above.*

**Out of scope / later:** MTP (needs separate conversion; not in this GGUF), vision (deferred policy), HF safetensors loader, GPU-resident IQ1_S grouped expert kernels.

## Critical files

- `native/src/v2_runtime.cpp` — config parse (:1917-2125), plan builders (:2177+, dispatch :11708), runtime create (:11631+), state arena (:12074-12150), decode (:15308-16500), snapshot (:14106), embedding stager (:3921), multi-seq (:18420+)
- `native/src/v2_mtp_verifier.inc` — `qwen_forward_rows` (:95), same hc/ple hooks
- `native/include/colibri_v2_native_kernels.hpp` + `native/src/cpu_native_kernels.cpp` — new kernels + CPU twins; `native/src/gpu_driver.cpp` name registration
- `native/include/colibri_v2_workspace.hpp`, `colibri_v2_config.hpp`, `colibri_v2_format_dispatch.hpp` (additive only)
- `tests/qwen4exp_gguf_fixture.py`, `tests/test_v2_qwen4exp_parity.py`, `native/tests/qwen4exp_{hc,ple}_contract.cpp`, `native/CMakeLists.txt`
- Templates/prior art: `tests/dense_gguf_fixture.py`, `tests/test_v2_qwen35_parity.py`, `native/tools/kda_reference.py`, `native/include/colibri_v2_deepseek4.hpp:116-274` (hyper-connection structural precedent)

## Risks (ranked, with mitigations)

1. **HC math guessed wrong** (norm scope / gating order / [10240,4] inject interpretation) → fluent-wrong output. → phase-0 module oracles + per-layer dump parity; hc math isolated in 3 small elementwise kernels.
2. **PLE hash details** (operand order, u64 mod/overflow, metadata endianness, start padding, head ordering). → integer test vectors from HF pinned in a zero-tolerance contract test; `__int128` mult-mod.
3. **RoPE layout mismatch** (interleaved vs NeoX, and what conversion actually wrote). → keep both kernel variants; fixture parity decides.
4. **head_dim-256 assumptions in less-traveled paths** (rows attention, multi-seq, turbo cache). → fixture parity runs all three paths; qwen35 precedent covers most.
5. **Host memory pressure** (25.7 GB PLE + ~40 GB experts): PLE random-gather faults — measure, `madvise` hot heads; disable `next_layer_prefetch` (`expert_transitions` ≈ 49 MB at 512²).
6. **CUDA-graph capture** of the old residual add corrupting streams → graphs off for the arch until hooks are capture-clean.
7. **Repo hygiene** — uncommitted IQ4_XS/transcript-audit changes; ctest + pytest at every gate.
