#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace colibri::v2 {
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
    std::uint32_t key_length_swa=0, value_length_swa=0;
    std::uint32_t rotary_dimension_swa=0;
    float rope_freq_base_swa=0.0f, final_logit_softcap=0.0f;
    float rms_norm_epsilon=0.0f, rope_freq_base=0.0f;
    // Leading blocks that carry a dense feed-forward instead of a router.
    std::uint32_t leading_dense_block_count=0;
    std::uint32_t expert_shared_intermediate_size=0;
    // GGUF llama_expert_gating_func_type: 1 = softmax, 2 = sigmoid.
    std::uint32_t expert_gating_func=0;
    bool expert_weights_norm=false;
    float expert_weights_scale=1.0f;
    // YaRN rope extension. A zero factor means the layer runs plain RoPE.
    std::uint32_t rope_original_context_length=0;
    float rope_scaling_factor=0.0f, yarn_attn_factor=1.0f;
    float yarn_beta_fast=32.0f, yarn_beta_slow=1.0f;
    bool rope_scaling_yarn=false;
    // GGUF tokenizer terminator ids; max means the key was absent.
    std::uint32_t eos_token_id=0xffffffffu;
    std::uint32_t eot_token_id=0xffffffffu;
    std::uint32_t bos_token_id=0xffffffffu;
};
}
