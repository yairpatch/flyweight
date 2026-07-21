#pragma once
#include <cstdint>
#include <string>
namespace colibri::v2 {
struct ModelConfig {
    std::string architecture;
    std::uint32_t hidden_size=0, layer_count=0, attention_heads=0,
        attention_kv_heads=0, context_length=0, intermediate_size=0,
        expert_count=0, expert_used_count=0, vocabulary_size=0,
        rotary_dimension=0, full_attention_interval=0;
    float rms_norm_epsilon=0.0f, rope_freq_base=0.0f;
};
}
