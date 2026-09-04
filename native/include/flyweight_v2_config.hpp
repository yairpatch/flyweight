#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace flyweight::v2 {
struct ModelConfig {
    std::string architecture;
    std::uint32_t hidden_size=0, layer_count=0, attention_heads=0,
        attention_kv_heads=0, context_length=0, intermediate_size=0,
        expert_count=0, expert_used_count=0, vocabulary_size=0,
        rotary_dimension=0, full_attention_interval=0, sliding_window=0;
    // True entries identify sliding-window layers.  An empty pattern with a
    // non-zero sliding_window means every attention layer is sliding.
    std::vector<std::uint8_t> sliding_window_pattern;
    std::vector<std::uint32_t> attention_kv_heads_by_layer;
    // Laguna varies the query-head count between its full-attention and
    // sliding-window layers, so the head count arrives as an array there.
    std::vector<std::uint32_t> attention_heads_by_layer;
    std::uint32_t dense_intermediate_size=0, expert_intermediate_size=0;
    std::uint32_t per_layer_embedding_size=0, shared_kv_layers=0;
    std::uint32_t key_length=0, value_length=0;
    // The per-head width the architecture declares, which is NOT always
    // hidden_size / heads: BailingMoE3 is 1536/16 = 96 by that formula but
    // declares head_dim 128, and its KDA projections are heads*head_dim wide.
    std::uint32_t attention_head_dim=0;
    // Width of the short causal convolution in front of a linear-attention
    // layer. Zero means the architecture's own default.
    std::uint32_t conv_kernel=0;
    // Gated-delta geometry. Needed at load time on the HF path, where the
    // value-head order has to be rewritten; the GGUF path reads these off the
    // tensor shapes instead.
    std::uint32_t linear_value_heads=0, linear_key_heads=0, linear_head_dim=0;
    std::uint32_t key_length_swa=0, value_length_swa=0;
    // MLA per-head widths as a GGUF carries them. llama.cpp's bailingmoe3
    // conversion writes `attention.key_length` as the MQA cache-row width
    // (kv_lora + rope) and the true per-head q/k width under `_mla` keys;
    // the runtime geometry wants the per-head widths in key_length /
    // value_length, so these are folded over them post-parse.
    std::uint32_t key_length_mla=0, value_length_mla=0;
    std::uint32_t rotary_dimension_swa=0;
    float rope_freq_base_swa=0.0f, final_logit_softcap=0.0f;
    float rms_norm_epsilon=0.0f, rope_freq_base=0.0f;
    // Interleaved M-RoPE (`rope.dimension_sections`, Qwen3-VL / Qwen3.5):
    // rotary pairs per temporal, height, width and extra component. All zero
    // means plain RoPE, where a text token's single position rotates every
    // pair -- which is also what the sections degenerate to when the three
    // positions coincide.
    std::uint32_t rope_sections[4]={0,0,0,0};
    bool has_rope_sections() const {
        return rope_sections[0]||rope_sections[1]||rope_sections[2]||rope_sections[3];
    }
    // Muse Glimmer scales the logits by a trained constant before the softcap.
    // Zero means the checkpoint carries no scale and the head output stands.
    float logit_scale=0.0f;
    // A sliding-window pattern written as a scalar period rather than a
    // per-layer array: the layer is sliding unless it closes the cycle.
    std::uint32_t sliding_window_period=0;
    // Leading blocks that carry a dense feed-forward instead of a router.
    std::uint32_t leading_dense_block_count=0;
    std::uint32_t expert_shared_intermediate_size=0;
    // GGUF llama_expert_gating_func_type: 1 = softmax, 2 = sigmoid.
    std::uint32_t expert_gating_func=0;
    // `noaux_tc` group-limited routing: experts are split into
    // `expert_group_count` groups and only the best `expert_group_used` of them
    // supply candidates. Zero means routing is not group limited.
    std::uint32_t expert_group_count=0, expert_group_used=0;
    std::uint32_t norm_groups=0;
    std::uint32_t value_expert_count=0, value_expert_used_count=0;
    bool expert_weights_norm=false;
    float expert_weights_scale=1.0f;
    // YaRN rope extension. A zero factor means the layer runs plain RoPE.
    std::uint32_t rope_original_context_length=0;
    float rope_scaling_factor=0.0f, yarn_attn_factor=1.0f;
    float yarn_beta_fast=32.0f, yarn_beta_slow=1.0f;
    bool rope_scaling_yarn=false;
    // DeepSeek-V4 (`deepseek4`). Multi-head latent attention carries a low-rank
    // query path and a compressed KV latent instead of per-head K/V.
    std::uint32_t q_lora_rank=0, kv_lora_rank=0;
    // The output projection is low-rank and grouped rather than a single matrix.
    std::uint32_t output_lora_rank=0, output_group_count=0;
    // Lightning indexer: the top-k selector that drives compressed sparse
    // attention, with its own head count, key width and selection budget.
    std::uint32_t indexer_head_count=0, indexer_key_length=0, indexer_top_k=0;
    // Per-layer attention kind. 0 = sliding window, 4 = compressed sparse
    // attention over 4:1 compressed tokens, 128 = heavily compressed attention.
    std::vector<std::uint32_t> compress_ratios;
    float compress_rope_freq_base=0.0f;
    // Hyper-connections replace the plain residual with `hyper_connection_count`
    // parallel streams mixed by a Sinkhorn-normalized router.
    std::uint32_t hyper_connection_count=0, sinkhorn_iterations=0;
    float sinkhorn_epsilon=0.0f;
    std::uint32_t expert_shared_count=0, hash_layer_count=0;
    // qwen4exp gated residual: the same hyper_connection_count streams as
    // deepseek4 above, but mixed by a low-rank silu/sigmoid gate instead of a
    // Sinkhorn router. Zero low_rank means the architecture has no gated
    // residual.
    std::uint32_t hyper_connection_low_rank=0;
    // qwen4exp PLE: hashed n-gram embeddings added into the residual streams
    // at these layers (0-based block ids as the GGUF writes them). The u64
    // hash constants arrive verbatim from the conversion -- they are derived
    // from a training-time seed and cannot be recomputed here.
    std::vector<std::uint32_t> ple_layers;
    std::uint32_t ple_ngram_size=0, ple_heads_per_ngram=0, ple_conv_kernel=0;
    std::uint32_t ple_eos_token_id=0xffffffffu;
    std::vector<std::uint64_t> ple_multipliers, ple_head_offsets,
        ple_head_vocab_sizes;
    std::vector<std::uint32_t> target_layers;
    std::uint32_t draft_block_size=0;
    // Per-layer SwiGLU clamp bounds, routed experts and shared expert.
    std::vector<float> swiglu_clamp_exp, swiglu_clamp_shexp;
    // GGUF tokenizer terminator ids; max means the key was absent.
    std::uint32_t eos_token_id=0xffffffffu;
    std::uint32_t eot_token_id=0xffffffffu;
    std::uint32_t bos_token_id=0xffffffffu;
    std::uint32_t mask_token_id=0xffffffffu;
};
}
