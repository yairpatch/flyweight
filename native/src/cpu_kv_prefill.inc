// Host driver for the chunked-prefill attention kernels (KV_PREFILL and
// kv_attention_prefill_q8 in flyweight_v2_native_kernels.hpp).
//
// Included by cpu_kernels.cpp after the generated corpus, for the same reason
// as cpu_q8_family.inc: the cache element loads go through the corpus's own
// kv_ld / kv_ld_q8, so every cache precision decodes exactly as on the GPU.
//
// The corpus gives each warp four query rows of one head and streams the KV
// cache once for them with an online softmax; the score dot is a 32-lane
// shuffle reduction. Emulated, that shuffle is a fiber switch per lane per
// step per cached position: 13.6 s a launch for a ~650-token prompt on a 4B
// model, over 90% of the whole request. Here a pool task takes a head and a
// tile of query rows, decodes each cached K/V row once into floats, and runs
// the same online-softmax update per row.

namespace kv_prefill {

constexpr int kTileRows = 16;

// Element loaders: one decoded K or V row, head_dim floats.
template <class T>
struct PlainRows {
    const T* keys;
    const T* values;
    int capacity, head_dim;
    void load(int kv_head, int position, float* key, float* value) const {
        const long long base =
            (static_cast<long long>(kv_head) * capacity + position) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            key[d] = kv_ld(keys + base, d);
            value[d] = kv_ld(values + base, d);
        }
    }
};

struct Q8Rows {
    const unsigned char* keys;
    const unsigned char* values;
    int capacity, head_dim;
    void load(int kv_head, int position, float* key, float* value) const {
        const long long base = (static_cast<long long>(kv_head) * capacity + position)
            * (head_dim / 32) * 34;
        for (int d = 0; d < head_dim; ++d) {
            key[d] = kv_ld_q8(keys + base, d);
            value[d] = kv_ld_q8(values + base, d);
        }
    }
};

template <class Rows>
void run(const ::flyweight::cpu::Launch& launch, const Rows& cache,
         const float* queries, float* output, int heads, int kv_heads,
         int head_dim, int base_position, int rows, float scale,
         int attention_window) {
    if (heads <= 0 || kv_heads <= 0 || rows <= 0 || head_dim <= 0) return;
    // What the corpus grid reaches: a block per head, four rows per warp.
    const int covered_heads = std::min<long long>(heads, launch.grid_x);
    const int covered_rows = static_cast<int>(std::min<long long>(
        rows, static_cast<long long>(launch.grid_y) * (launch.block_x >> 5) * 4));
    if (covered_heads <= 0 || covered_rows <= 0) return;
    const int group = heads / kv_heads;
    const int tiles = (covered_rows + kTileRows - 1) / kTileRows;
    const std::uint64_t items = static_cast<std::uint64_t>(covered_heads) * tiles;

    q8_family::parallel_ranges(items, [&](std::uint64_t first, std::uint64_t last) {
        std::vector<float> scratch(
            static_cast<std::size_t>(head_dim) * (2 + kTileRows));
        float* key = scratch.data();
        float* value = key + head_dim;
        float* acc = value + head_dim;
        for (std::uint64_t item = first; item < last; ++item) {
            const int head = static_cast<int>(item / tiles);
            const int tile = static_cast<int>(item % tiles) * kTileRows;
            const int count = std::min(kTileRows, covered_rows - tile);
            const int kv_head = head / group;
            float m[kTileRows], l[kTileRows];
            const float* q[kTileRows];
            for (int i = 0; i < count; ++i) {
                m[i] = -3.402823466e+38F;
                l[i] = 0.0f;
                q[i] = queries + static_cast<long long>(tile + i) * heads * head_dim
                    + static_cast<long long>(head) * head_dim;
            }
            std::fill(acc, acc + static_cast<std::size_t>(count) * head_dim, 0.0f);
            const int final_position = base_position + tile + count - 1;
            for (int position = 0; position <= final_position; ++position) {
                cache.load(kv_head, position, key, value);
                for (int i = 0; i < count; ++i) {
                    const int query_position = base_position + tile + i;
                    if (position > query_position) continue;
                    if (attention_window > 0 &&
                        position < query_position - attention_window + 1)
                        continue;
                    float lanes[8] = {};
                    for (int d = 0; d < head_dim; ++d)
                        lanes[d & 7] += q[i][d] * key[d];
                    const float partial = ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6]))
                        + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
                    const float score = partial * scale;
                    const float peak = fmaxf(m[i], score);
                    const float rescale = expf(m[i] - peak);
                    const float weight = expf(score - peak);
                    l[i] = l[i] * rescale + weight;
                    float* row_acc = acc + static_cast<std::size_t>(i) * head_dim;
                    for (int d = 0; d < head_dim; ++d)
                        row_acc[d] = row_acc[d] * rescale + weight * value[d];
                    m[i] = peak;
                }
            }
            for (int i = 0; i < count; ++i) {
                const float inverse = 1.0f / l[i];
                float* out = output + static_cast<long long>(tile + i) * heads * head_dim
                    + static_cast<long long>(head) * head_dim;
                const float* row_acc = acc + static_cast<std::size_t>(i) * head_dim;
                for (int d = 0; d < head_dim; ++d) out[d] = row_acc[d] * inverse;
            }
        }
    });
}

//   name(const float* queries, const KT* keys, const VT* values, float* output,
//        int heads, int kv_heads, int head_dim, int base_position, int rows,
//        int capacity, float scale, int attention_window)
template <class T>
void plain(const ::flyweight::cpu::Launch& launch, void** arguments) {
    using q8_family::argument;
    const PlainRows<T> cache{argument<const T*>(arguments, 1),
                             argument<const T*>(arguments, 2),
                             argument<int>(arguments, 9), argument<int>(arguments, 6)};
    run(launch, cache, argument<const float*>(arguments, 0),
        argument<float*>(arguments, 3), argument<int>(arguments, 4),
        argument<int>(arguments, 5), argument<int>(arguments, 6),
        argument<int>(arguments, 7), argument<int>(arguments, 8),
        argument<float>(arguments, 10), argument<int>(arguments, 11));
}

// kv_attention_prefill_q8: the same arguments without attention_window.
void q8(const ::flyweight::cpu::Launch& launch, void** arguments) {
    using q8_family::argument;
    const Q8Rows cache{argument<const unsigned char*>(arguments, 1),
                       argument<const unsigned char*>(arguments, 2),
                       argument<int>(arguments, 9), argument<int>(arguments, 6)};
    run(launch, cache, argument<const float*>(arguments, 0),
        argument<float*>(arguments, 3), argument<int>(arguments, 4),
        argument<int>(arguments, 5), argument<int>(arguments, 6),
        argument<int>(arguments, 7), argument<int>(arguments, 8),
        argument<float>(arguments, 10), 0);
}

}  // namespace kv_prefill

FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_prefill, kv_prefill::plain<float>);
FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_prefill_f16, kv_prefill::plain<__half>);
FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_prefill_bf16, kv_prefill::plain<__nv_bfloat16>);
FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_prefill_q8, kv_prefill::q8);
