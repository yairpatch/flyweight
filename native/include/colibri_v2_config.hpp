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
    float rms_norm_epsilon=0.0f, rope_freq_base=0.0f;
};
}
