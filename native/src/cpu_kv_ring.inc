// Host drivers for the split decode attention: kv_attention_scores* (one dot
// per cached token) and kv_attention_values* (softmax, then the score-weighted
// sum of value rows), linear, ring-addressed and slot-indexed (QSA), for
// f32 / f16 / bf16 / q8_0 / turbo3 / turbo4 caches, plus the turbo cache's
// store and rotation kernels. kv_attention_values_ring and _f16_ring were
// already native in cpu_native_kernels.cpp; this covers the rest.
//
// Included by cpu_kernels.cpp after the corpus: rows decode through the
// corpus's kv_ld / kv_ld_q8. The value kernels are block-cooperative (two
// block reductions, four barriers) and cost ~6 ms a launch emulated on a q8_0
// cache; the score kernels run unemulated but one thread at a time.
//
// The per-element arithmetic follows the corpus order: score dots sum over d
// sequentially, value sums over tokens sequentially. Only the softmax
// maximum and denominator, block reductions in the corpus, are summed in a
// different order.

namespace kv_split {

// How logical token t maps to a cache slot.
enum class Slots { linear, ring, indexed };

struct SlotMap {
    Slots mode;
    int first;
    int capacity;
    const int* list;
    int operator()(int token) const {
        switch (mode) {
            case Slots::ring: return (first + token) % capacity;
            case Slots::indexed: return list[token];
            case Slots::linear: break;
        }
        return token;
    }
};

template <class T>
struct PlainCache {
    const T* base;
    int head_dim;
    float load(long long row, int d) const { return kv_ld(base + row * head_dim, d); }
};

struct Q8Cache {
    const unsigned char* base;
    int head_dim;
    float load(long long row, int d) const {
        return kv_ld_q8(base + row * (head_dim / 32) * 34, d);
    }
};

template <class Cache>
void scores(const ::flyweight::cpu::Launch& launch, const float* query,
            const Cache& keys, float* scores_out, int heads, int kv_heads,
            int head_dim, int tokens, int capacity, const SlotMap& slot_of,
            float scale) {
    if (heads <= 0 || kv_heads <= 0 || tokens <= 0 || capacity <= 0) return;
    const int covered_heads = std::min<long long>(heads, launch.grid_x);
    const int covered_tokens = static_cast<int>(std::min<long long>(
        tokens, static_cast<long long>(launch.grid_y) * launch.block_x));
    if (covered_heads <= 0 || covered_tokens <= 0) return;
    const int group = heads / kv_heads;
    q8_family::parallel_ranges(
        static_cast<std::uint64_t>(covered_heads) * covered_tokens,
        [&](std::uint64_t first_item, std::uint64_t last_item) {
            for (std::uint64_t item = first_item; item < last_item; ++item) {
                const int head = static_cast<int>(item / covered_tokens);
                const int token = static_cast<int>(item % covered_tokens);
                const long long row =
                    static_cast<long long>(head / group) * capacity + slot_of(token);
                const float* q = query + static_cast<long long>(head) * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) score += q[d] * keys.load(row, d);
                scores_out[static_cast<long long>(head) * tokens + token] = score * scale;
            }
        });
}

// `normalize_in_place`: every variant except the linear q8_0 one writes the
// normalized weights back into `scores`; that one leaves the raw scores and
// applies 1/denominator to each output instead.
template <class Cache>
void values(const ::flyweight::cpu::Launch& launch, float* scores_io,
            const Cache& cache, float* output, int heads, int kv_heads,
            int head_dim, int tokens, int capacity, const SlotMap& slot_of,
            bool normalize_in_place) {
    if (heads <= 0 || kv_heads <= 0 || capacity <= 0) return;
    const int covered_heads = std::min<long long>(heads, launch.grid_x);
    const int group = heads / kv_heads;
    q8_family::parallel_ranges(covered_heads, [&](std::uint64_t first_head,
                                                  std::uint64_t last_head) {
        std::vector<float> weights(static_cast<std::size_t>(std::max(tokens, 0)));
        for (std::uint64_t h = first_head; h < last_head; ++h) {
            const int head = static_cast<int>(h);
            const int kv_head = head / group;
            float* head_scores = scores_io + static_cast<long long>(head) * tokens;
            float maximum = -3.402823466e+38F;
            for (int t = 0; t < tokens; ++t) maximum = fmaxf(maximum, head_scores[t]);
            float denominator = 0.0f;
            for (int t = 0; t < tokens; ++t) {
                weights[t] = expf(head_scores[t] - maximum);
                denominator += weights[t];
            }
            const float inverse = 1.0f / denominator;
            if (normalize_in_place) {
                for (int t = 0; t < tokens; ++t) {
                    weights[t] *= inverse;
                    head_scores[t] = weights[t];
                }
            }
            float* out = output + static_cast<long long>(head) * head_dim;
            for (int d = 0; d < head_dim; ++d) out[d] = 0.0f;
            for (int t = 0; t < tokens; ++t) {
                const long long row =
                    static_cast<long long>(kv_head) * capacity + slot_of(t);
                const float weight = weights[t];
                for (int d = 0; d < head_dim; ++d) out[d] += weight * cache.load(row, d);
            }
            if (!normalize_in_place)
                for (int d = 0; d < head_dim; ++d) out[d] *= inverse;
        }
    });
}

using q8_family::argument;

// Argument layouts (the cache pointer is argument 1 throughout):
//   scores linear  (query, keys,        scores, heads, kv_heads, head_dim, tokens, capacity,        scale)
//   scores ring    (query, keys,        scores, heads, kv_heads, head_dim, tokens, capacity, first, scale)
//   scores indexed (query, keys, slots, scores, heads, kv_heads, head_dim, tokens, capacity,        scale)
//   values linear  (scores, values,        output, heads, kv_heads, head_dim, tokens, capacity)
//   values ring    (scores, values,        output, heads, kv_heads, head_dim, tokens, capacity, first)
//   values indexed (scores, values, slots, output, heads, kv_heads, head_dim, tokens, capacity)
// An indexed launch shifts every argument after the cache by one.
struct Layout {
    int shift;  // 1 when a slot list follows the cache pointer
    SlotMap slots;
    int heads, kv_heads, head_dim, tokens, capacity;
};

template <Slots Mode>
Layout layout(void** a) {
    const int shift = Mode == Slots::indexed ? 1 : 0;
    Layout out{};
    out.shift = shift;
    out.heads = argument<int>(a, 3 + shift);
    out.kv_heads = argument<int>(a, 4 + shift);
    out.head_dim = argument<int>(a, 5 + shift);
    out.tokens = argument<int>(a, 6 + shift);
    out.capacity = argument<int>(a, 7 + shift);
    out.slots = SlotMap{Mode, Mode == Slots::ring ? argument<int>(a, 8) : 0,
                        out.capacity,
                        Mode == Slots::indexed ? argument<const int*>(a, 2) : nullptr};
    return out;
}

template <class Cache, class Element, Slots Mode>
void scores_kernel(const ::flyweight::cpu::Launch& launch, void** a) {
    const Layout l = layout<Mode>(a);
    const Cache keys{argument<const Element*>(a, 1), l.head_dim};
    const int scale_index = Mode == Slots::ring ? 9 : 8 + l.shift;
    scores(launch, argument<const float*>(a, 0), keys, argument<float*>(a, 2 + l.shift),
           l.heads, l.kv_heads, l.head_dim, l.tokens, l.capacity, l.slots,
           argument<float>(a, scale_index));
}

template <class Cache, class Element, Slots Mode, bool InPlace = true>
void values_kernel(const ::flyweight::cpu::Launch& launch, void** a) {
    const Layout l = layout<Mode>(a);
    const Cache cache{argument<const Element*>(a, 1), l.head_dim};
    values(launch, argument<float*>(a, 0), cache, argument<float*>(a, 2 + l.shift),
           l.heads, l.kv_heads, l.head_dim, l.tokens, l.capacity, l.slots, InPlace);
}

using F32 = PlainCache<float>;
using F16 = PlainCache<__half>;
using BF16 = PlainCache<__nv_bfloat16>;

}  // namespace kv_split

#define FLYWEIGHT_CPU_KV_SPLIT(suffix, cache, element)                          \
    FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_scores##suffix,                      \
        kv_split::scores_kernel<cache, element, kv_split::Slots::linear>);      \
    FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_scores##suffix##_ring,               \
        kv_split::scores_kernel<cache, element, kv_split::Slots::ring>);        \
    FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_scores##suffix##_indexed,            \
        kv_split::scores_kernel<cache, element, kv_split::Slots::indexed>);     \
    FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_values##suffix##_indexed,            \
        kv_split::values_kernel<cache, element, kv_split::Slots::indexed>)

// The linear and ring value kernels are registered per type below: f32 and
// f16 ring already have host versions in cpu_native_kernels.cpp, and linear
// q8_0 is the one variant that does not normalize the scores in place.
FLYWEIGHT_CPU_KV_SPLIT(, kv_split::F32, float);
FLYWEIGHT_CPU_KV_SPLIT(_f16, kv_split::F16, __half);
FLYWEIGHT_CPU_KV_SPLIT(_bf16, kv_split::BF16, __nv_bfloat16);
FLYWEIGHT_CPU_KV_SPLIT(_q8, kv_split::Q8Cache, unsigned char);

FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_values,
    kv_split::values_kernel<kv_split::F32, float, kv_split::Slots::linear>);
FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_values_f16,
    kv_split::values_kernel<kv_split::F16, __half, kv_split::Slots::linear>);
FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_values_bf16,
    kv_split::values_kernel<kv_split::BF16, __nv_bfloat16, kv_split::Slots::linear>);
FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_values_q8,
    kv_split::values_kernel<kv_split::Q8Cache, unsigned char, kv_split::Slots::linear, false>);
FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_values_bf16_ring,
    kv_split::values_kernel<kv_split::BF16, __nv_bfloat16, kv_split::Slots::ring>);
FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_values_q8_ring,
    kv_split::values_kernel<kv_split::Q8Cache, unsigned char, kv_split::Slots::ring>);

// --- turbo3 / turbo4 caches ---------------------------------------------------
//
// TurboQuant rows are stored rotated (sign flip, then an orthonormal
// Walsh-Hadamard transform) and coded 3 or 4 bits per value against a fixed
// codebook with a per-32 fp16 scale. Scores rotate the query instead of the
// keys; values accumulate in the rotated domain and inverse-rotate once.
// Emulated, every Walsh-Hadamard stage was a block barrier: 208 ms for one
// turbo_rotate_rows launch on a ~650-token prefill chunk.
//
// The transform below performs the corpus's butterflies stage by stage; each
// stage's pairs are disjoint, so the result does not depend on how the corpus
// spread them over threads and is bit-identical.
namespace kv_turbo {

using q8_family::argument;

inline void fwht(float* v, int dim) {
    for (int span = 1; span < dim; span <<= 1) {
        for (int i = 0; i < (dim >> 1); ++i) {
            const int block = i / span, offset = i % span;
            const int a = block * 2 * span + offset, b = a + span;
            const float lo = v[a], hi = v[b];
            v[a] = lo + hi;
            v[b] = lo - hi;
        }
    }
    const float norm = rsqrtf((float)dim);
    for (int i = 0; i < dim; ++i) v[i] *= norm;
}

template <int Bits>
struct TurboCache {
    const unsigned char* base;
    int head_dim;
    float load(long long row, int d) const {
        return kv_ld_turbo<Bits>(
            base + row * (head_dim / 32) * turbo_block_bytes<Bits>(), d);
    }
};

// Scores: the query is rotated under stream 0, then dotted against the coded
// keys exactly as the generic kernel does.
template <int Bits, kv_split::Slots Mode>
void scores_kernel(const ::flyweight::cpu::Launch& launch, void** a) {
    const kv_split::Layout l = kv_split::layout<Mode>(a);
    if (l.heads <= 0 || l.head_dim <= 0) return;
    const auto* query = argument<const float*>(a, 0);
    const int covered_heads = std::min<long long>(l.heads, launch.grid_x);
    std::vector<float> rotated(static_cast<std::size_t>(l.heads) * l.head_dim, 0.0f);
    for (int head = 0; head < covered_heads; ++head) {
        float* rq = rotated.data() + static_cast<std::size_t>(head) * l.head_dim;
        for (int d = 0; d < l.head_dim; ++d)
            rq[d] = query[static_cast<long long>(head) * l.head_dim + d]
                * turbo_sign_d(d, 0u);
        fwht(rq, l.head_dim);
    }
    const TurboCache<Bits> keys{argument<const unsigned char*>(a, 1), l.head_dim};
    const int scale_index = Mode == kv_split::Slots::ring ? 9 : 8 + l.shift;
    kv_split::scores(launch, rotated.data(), keys, argument<float*>(a, 2 + l.shift),
                     l.heads, l.kv_heads, l.head_dim, l.tokens, l.capacity, l.slots,
                     argument<float>(a, scale_index));
}

// Values: softmax in place, the weighted sum in the rotated domain with the
// corpus's decode expression, then Walsh-Hadamard and the stream-1 sign flip.
template <int Bits, kv_split::Slots Mode>
void values_kernel(const ::flyweight::cpu::Launch& launch, void** a) {
    const kv_split::Layout l = kv_split::layout<Mode>(a);
    if (l.heads <= 0 || l.kv_heads <= 0 || l.capacity <= 0) return;
    auto* scores_io = argument<float*>(a, 0);
    const auto* values = argument<const unsigned char*>(a, 1);
    auto* output = argument<float*>(a, 2 + l.shift);
    const int covered_heads = std::min<long long>(l.heads, launch.grid_x);
    const int group = l.heads / l.kv_heads;
    const int bytes = turbo_block_bytes<Bits>();
    const int row_bytes = (l.head_dim / 32) * bytes;
    float codebook[1 << Bits];
    for (int i = 0; i < (1 << Bits); ++i) codebook[i] = turbo_cb<Bits>(i);
    constexpr unsigned mask = (1u << Bits) - 1u;

    q8_family::parallel_ranges(covered_heads, [&](std::uint64_t first_head,
                                                  std::uint64_t last_head) {
        std::vector<float> accumulated(static_cast<std::size_t>(l.head_dim));
        for (std::uint64_t h = first_head; h < last_head; ++h) {
            const int head = static_cast<int>(h);
            float* head_scores = scores_io + static_cast<long long>(head) * l.tokens;
            float maximum = -3.402823466e+38F;
            for (int t = 0; t < l.tokens; ++t) maximum = fmaxf(maximum, head_scores[t]);
            float denominator = 0.0f;
            for (int t = 0; t < l.tokens; ++t) {
                head_scores[t] = expf(head_scores[t] - maximum);
                denominator += head_scores[t];
            }
            const float inverse = 1.0f / denominator;
            for (int t = 0; t < l.tokens; ++t) head_scores[t] *= inverse;

            const unsigned char* cache_base = values
                + static_cast<long long>(head / group) * l.capacity * row_bytes;
            std::fill(accumulated.begin(), accumulated.end(), 0.0f);
            for (int t = 0; t < l.tokens; ++t) {
                const unsigned char* row =
                    cache_base + static_cast<long long>(l.slots(t)) * row_bytes;
                for (int d = 0; d < l.head_dim; ++d) {
                    const int block_offset = (d >> 5) * bytes;
                    const int bit = (d & 31) * Bits;
                    const int byte_offset = block_offset + 2 + (bit >> 3);
                    const int shift = bit & 7;
                    const int spill = (shift + Bits > 8) ? 1 : 0;
                    const float scale =
                        __half2float(*reinterpret_cast<const __half*>(row + block_offset));
                    const unsigned low = row[byte_offset], high = row[byte_offset + spill];
                    accumulated[d] += head_scores[t] * scale
                        * codebook[((low | (high << 8)) >> shift) & mask];
                }
            }
            fwht(accumulated.data(), l.head_dim);
            float* out = output + static_cast<long long>(head) * l.head_dim;
            for (int d = 0; d < l.head_dim; ++d)
                out[d] = accumulated[d] * turbo_sign_d(d, 1u);
        }
    });
}

// kv_store_turbo{3,4}_{k,v}: (current, cache, kv_heads, head_dim, position,
// capacity); the _rows twins add row_stride, take grid_y rows, and store row r
// at (position + r) % capacity. Keys rotate under stream 0, values stream 1.
template <int Bits, unsigned Stream, bool Rows>
void store_kernel(const ::flyweight::cpu::Launch& launch, void** a) {
    const auto* current = argument<const float*>(a, 0);
    auto* cache = argument<unsigned char*>(a, 1);
    const int kv_heads = argument<int>(a, 2);
    const int head_dim = argument<int>(a, 3);
    const int position = argument<int>(a, 4);
    const int capacity = argument<int>(a, 5);
    const long long row_stride = Rows ? argument<int>(a, 6) : 0;
    const int rows = Rows ? static_cast<int>(launch.grid_y) : 1;
    const int covered_heads = std::min<long long>(kv_heads, launch.grid_x);
    if (covered_heads <= 0 || head_dim <= 0 || capacity <= 0) return;
    const int blocks = head_dim / 32, bytes = turbo_block_bytes<Bits>();
    q8_family::parallel_ranges(
        static_cast<std::uint64_t>(rows) * covered_heads,
        [&](std::uint64_t first, std::uint64_t last) {
            std::vector<float> rotated(static_cast<std::size_t>(head_dim));
            for (std::uint64_t item = first; item < last; ++item) {
                const int row = static_cast<int>(item / covered_heads);
                const int head = static_cast<int>(item % covered_heads);
                const float* source = current + row * row_stride
                    + static_cast<long long>(head) * head_dim;
                for (int d = 0; d < head_dim; ++d)
                    rotated[d] = source[d] * turbo_sign_d(d, Stream);
                fwht(rotated.data(), head_dim);
                const int slot = Rows ? (position + row) % capacity : position;
                unsigned char* out = cache
                    + (static_cast<long long>(head) * capacity + slot) * blocks * bytes;
                for (int b = 0; b < blocks; ++b)
                    turbo_encode_block_d<Bits>(rotated.data() + b * 32, out + b * bytes);
            }
        });
}

// turbo_{rotate,unrotate}_rows(float* data, int rows, int dim, int stream_id)
template <bool Inverse>
void rotate_kernel(const ::flyweight::cpu::Launch& launch, void** a) {
    float* data = argument<float*>(a, 0);
    const int rows = std::min<long long>(argument<int>(a, 1), launch.grid_x);
    const int dim = argument<int>(a, 2);
    const unsigned stream = static_cast<unsigned>(argument<int>(a, 3));
    if (rows <= 0 || dim <= 0) return;
    q8_family::parallel_ranges(rows, [&](std::uint64_t first, std::uint64_t last) {
        for (std::uint64_t row = first; row < last; ++row) {
            float* base = data + static_cast<long long>(row) * dim;
            if (!Inverse)
                for (int i = 0; i < dim; ++i) base[i] *= turbo_sign_d(i, stream);
            fwht(base, dim);
            if (Inverse)
                for (int i = 0; i < dim; ++i) base[i] *= turbo_sign_d(i, stream);
        }
    });
}

}  // namespace kv_turbo

#define FLYWEIGHT_CPU_KV_TURBO(bits)                                                     \
    FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_scores_turbo##bits,                           \
        kv_turbo::scores_kernel<bits, kv_split::Slots::linear>);                         \
    FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_scores_turbo##bits##_ring,                    \
        kv_turbo::scores_kernel<bits, kv_split::Slots::ring>);                           \
    FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_scores_turbo##bits##_indexed,                 \
        kv_turbo::scores_kernel<bits, kv_split::Slots::indexed>);                        \
    FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_values_turbo##bits,                           \
        kv_turbo::values_kernel<bits, kv_split::Slots::linear>);                         \
    FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_values_turbo##bits##_ring,                    \
        kv_turbo::values_kernel<bits, kv_split::Slots::ring>);                           \
    FLYWEIGHT_CPU_Q8_REGISTER(kv_attention_values_turbo##bits##_indexed,                 \
        kv_turbo::values_kernel<bits, kv_split::Slots::indexed>);                        \
    FLYWEIGHT_CPU_Q8_REGISTER(kv_store_turbo##bits##_k,                                  \
        kv_turbo::store_kernel<bits, 0u, false>);                                        \
    FLYWEIGHT_CPU_Q8_REGISTER(kv_store_turbo##bits##_v,                                  \
        kv_turbo::store_kernel<bits, 1u, false>);                                        \
    FLYWEIGHT_CPU_Q8_REGISTER(kv_store_turbo##bits##_k_rows,                             \
        kv_turbo::store_kernel<bits, 0u, true>);                                         \
    FLYWEIGHT_CPU_Q8_REGISTER(kv_store_turbo##bits##_v_rows,                             \
        kv_turbo::store_kernel<bits, 1u, true>)

FLYWEIGHT_CPU_KV_TURBO(3);
FLYWEIGHT_CPU_KV_TURBO(4);
FLYWEIGHT_CPU_Q8_REGISTER(turbo_rotate_rows, kv_turbo::rotate_kernel<false>);
FLYWEIGHT_CPU_Q8_REGISTER(turbo_unrotate_rows, kv_turbo::rotate_kernel<true>);
