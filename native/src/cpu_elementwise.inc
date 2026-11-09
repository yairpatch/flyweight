// Host drivers for small elementwise corpus kernels that showed up in CPU
// profiles. They are not cooperative, so they never ran on fibers, but the
// generic launcher still steps them one emulated thread at a time; these
// spread them across the pool with the corpus's expressions unchanged.

namespace elementwise {

using q8_family::argument;

// Elements one launch actually reaches: blocks x threads, one per thread.
inline long long launched(const ::flyweight::cpu::Launch& launch, int elements) {
    return std::min<long long>(
        elements, static_cast<long long>(launch.grid_x) * launch.block_x);
}

//   qwen_copy_vector(const float* input, float* output, int elements)
void copy_vector(const ::flyweight::cpu::Launch& launch, void** a) {
    const auto* input = argument<const float*>(a, 0);
    auto* output = argument<float*>(a, 1);
    const long long count = launched(launch, argument<int>(a, 2));
    if (count > 0) std::memcpy(output, input, static_cast<std::size_t>(count) * sizeof(float));
}

//   silu_mul(const float* gate_up, float* output, int elements)
// gate_up holds the gate half then the up half.
void silu_mul(const ::flyweight::cpu::Launch& launch, void** a) {
    const auto* gate_up = argument<const float*>(a, 0);
    auto* output = argument<float*>(a, 1);
    const int elements = argument<int>(a, 2);
    const long long count = launched(launch, elements);
    q8_family::parallel_ranges(static_cast<std::uint64_t>(std::max(count, 0LL)),
                               [&](std::uint64_t first, std::uint64_t last) {
        for (std::uint64_t index = first; index < last; ++index) {
            const float gate = gate_up[index];
            const float exponential = gate >= 0.0f ? expf(-gate) : expf(gate);
            const float sigmoid = gate >= 0.0f
                ? 1.0f / (1.0f + exponential)
                : exponential / (1.0f + exponential);
            output[index] = gate * sigmoid * gate_up[elements + index];
        }
    });
}

//   delta_conv_chunk(const float* mixed_qkv, const float* weights,
//                    float* state, float* output, int tokens, int channels,
//                    int kernel_size)
// A causal depthwise convolution over a prefill chunk, one channel per thread
// carrying its window across the tokens, then SiLU.
void delta_conv_chunk(const ::flyweight::cpu::Launch& launch, void** a) {
    const auto* mixed_qkv = argument<const float*>(a, 0);
    const auto* weights = argument<const float*>(a, 1);
    auto* state = argument<float*>(a, 2);
    auto* output = argument<float*>(a, 3);
    const int tokens = argument<int>(a, 4);
    const int channels = argument<int>(a, 5);
    const int kernel_size = argument<int>(a, 6);
    if (kernel_size > 8 || kernel_size <= 0) return;
    const long long count = launched(launch, channels);
    q8_family::parallel_ranges(static_cast<std::uint64_t>(std::max(count, 0LL)),
                               [&](std::uint64_t first, std::uint64_t last) {
        for (std::uint64_t c = first; c < last; ++c) {
            const long long channel = static_cast<long long>(c);
            float window[8], taps[8];
            float* channel_state = state + channel * kernel_size;
            const float* channel_weights = weights + channel * kernel_size;
            for (int i = 0; i < kernel_size; ++i) {
                window[i] = channel_state[i];
                taps[i] = channel_weights[i];
            }
            for (int token = 0; token < tokens; ++token) {
                for (int i = 0; i + 1 < kernel_size; ++i) window[i] = window[i + 1];
                window[kernel_size - 1] =
                    mixed_qkv[static_cast<long long>(token) * channels + channel];
                float value = 0.0f;
                for (int i = 0; i < kernel_size; ++i) value += window[i] * taps[i];
                output[static_cast<long long>(token) * channels + channel] =
                    value / (1.0f + expf(-value));
            }
            for (int i = 0; i < kernel_size; ++i) channel_state[i] = window[i];
        }
    });
}

}  // namespace elementwise

FLYWEIGHT_CPU_Q8_REGISTER(qwen_copy_vector, elementwise::copy_vector);
FLYWEIGHT_CPU_Q8_REGISTER(silu_mul, elementwise::silu_mul);
FLYWEIGHT_CPU_Q8_REGISTER(delta_conv_chunk, elementwise::delta_conv_chunk);
