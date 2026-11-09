// Host drivers for the Q8-activation matvec families.
//
// Included by cpu_kernels.cpp after the generated corpus, inside the same
// anonymous namespace, so the corpus's own per-format `*_q8_decode` functions
// inline straight into these loops. That is the reason this lives here rather
// than in cpu_native_kernels.cpp: the weight decode of every format -- the
// bit unpacking, codebook lookups, scales and minima -- is the corpus's, and
// the per-group arithmetic around it is the corpus's FLYWEIGHT_Q8_MATVEC_ROWS
// expression. What these drivers replace is the CUDA decomposition: a block of
// fibers per output row and a warp-shuffle reduction across groups.
//
// That decomposition is what made K-quant and IQ models unusable on the CPU
// backend. Each shuffle step is a fiber barrier, so a single output row of an
// emulated FLYWEIGHT_Q8_MATVEC costs a few thousand context switches; a 4B
// Q4_K_M decode spent ~97% of its time in swapcontext and ran at minutes per
// token. Q8_0 weights had a native kernel (q8_matvec_transposed_warp) and so
// never showed it.
//
// Every family goes through the decode form, including the single-row
// FLYWEIGHT_Q8_MATVEC and FLYWEIGHT_Q8_LM_HEAD whose corpus versions call the
// fused `*_q8_group`: decode-then-dot lets the integer dot run as a plain byte
// loop the compiler vectorizes, where `*_q8_group` is a chain of emulated
// __dp4a. The two corpus forms are the same sum grouped differently, so they
// agree to fp32 rounding, which is what the parity contract checks.
//
// Across groups the drivers accumulate into kLanes interleaved partials and
// sum those pairwise. The emulated kernels sum per-thread strided partials and
// reduce through a shuffle tree; both keep the rounding error growing with
// log(groups) rather than linearly.
//
// The row loops are compiled three times -- baseline, AVX2 and AVX-512 -- and
// picked once from flyweight_cpu_features(). This translation unit is built
// for the baseline ISA, so without the wider copies the byte dots stay SSE2.

#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
#define FLYWEIGHT_Q8_FAMILY_ISA_CLONES 1
#define FLYWEIGHT_Q8_FAMILY_INLINE [[gnu::always_inline]] inline
#else
#define FLYWEIGHT_Q8_FAMILY_ISA_CLONES 0
#define FLYWEIGHT_Q8_FAMILY_INLINE inline
#endif

namespace q8_family {

constexpr int kLanes = 8;

enum class Isa { baseline, avx2, avx512 };

Isa isa() {
    static const Isa chosen = [] {
#if FLYWEIGHT_Q8_FAMILY_ISA_CLONES
        // Same bits cpu_native_kernels.cpp reads: 0 = AVX2+FMA, 1 = AVX-512
        // F/BW. FLYWEIGHT_CPU_BACKEND can force a lower path through them.
        const std::uint32_t features = flyweight_cpu_features();
        if (features & (1u << 1)) return Isa::avx512;
        if (features & (1u << 0)) return Isa::avx2;
#endif
        return Isa::baseline;
    }();
    return chosen;
}

FLYWEIGHT_Q8_FAMILY_INLINE float reduce_lanes(const float (&lanes)[kLanes]) {
    const float a = lanes[0] + lanes[4], b = lanes[1] + lanes[5];
    const float c = lanes[2] + lanes[6], d = lanes[3] + lanes[7];
    return (a + c) + (b + d);
}

// Signed int8 dot of 16 weight bytes against 16 activation bytes: the two
// __dp4a pairs one half-group of the corpus loop issues. Written over bytes so
// the compiler can vectorize it; integer arithmetic, so the order is free.
FLYWEIGHT_Q8_FAMILY_INLINE int dot16(const signed char* weights,
                                     const signed char* activations) {
    int total = 0;
    for (int index = 0; index < 16; ++index)
        total += static_cast<int>(weights[index]) * activations[index];
    return total;
}

inline int sum16(const signed char* activations) {
    int total = 0;
    for (int index = 0; index < 16; ++index) total += activations[index];
    return total;
}

template <class T>
T argument(void** arguments, int index) {
    return *reinterpret_cast<T*>(arguments[index]);
}

// Runs body(first, last) over contiguous ranges of [0, count) on the launch
// pool. Ranges rather than single items so a body can hoist per-chunk state.
template <class Body>
void parallel_ranges(std::uint64_t count, const Body& body) {
    if (count == 0) return;
    struct Job {
        const Body* body;
        std::uint64_t count;
        std::uint64_t chunk;
    };
    // Enough chunks that a straggler cannot hold up the launch, few enough that
    // the claim overhead stays negligible next to a row's worth of groups.
    const std::uint64_t target_chunks = 128;
    const std::uint64_t chunk =
        count <= target_chunks ? 1 : (count + target_chunks - 1) / target_chunks;
    Job job{&body, count, chunk};
    ::flyweight::cpu::parallel_for(
        (count + chunk - 1) / chunk,
        [](void* opaque, std::uint64_t index) {
            auto* task = static_cast<Job*>(opaque);
            const std::uint64_t first = index * task->chunk;
            const std::uint64_t last = std::min(first + task->chunk, task->count);
            (*task->body)(first, last);
        },
        &job);
}

// One launch's operands, shared by every family.
struct Operands {
    const unsigned char* packed;
    const signed char* vectors;       // rows x input_size int8
    const __half* vector_scales;      // rows x scale_stride halves
    const int* sums;                  // rows x groups x 2, Minimum formats only
    const float* activation_scales;   // rows x groups, vector_scales as floats
    int input_size;
    int groups;
    int rows;
    int scale_stride;
};

// The weight row's base. The _MIN corpus kernels address super-blocks
// directly (input_size is a multiple of 256 for every format they serve);
// the rest go through q8_weight_row, which also covers the flat formats' tail.
template <int Stride, bool Minimum>
FLYWEIGHT_Q8_FAMILY_INLINE const unsigned char* weight_row(const Operands& op,
                                                           std::uint64_t row) {
    if constexpr (Minimum) {
        return op.packed
            + static_cast<long long>(row) * (op.input_size >> 8) * Stride;
    } else {
        return q8_weight_row(op.packed, static_cast<int>(row), op.input_size,
                             Stride);
    }
}

// --- super-block decoders ---------------------------------------------------
//
// The corpus decodes one 32-value group at a time, because on a GPU each lane
// owns a group. Run from a host loop that walks a row's groups in order, the
// per-group form redoes per-super-block work eight times over -- the fp16 d
// and dmin, the packed 6-bit scales -- and assembles words through emulated
// byte intrinsics. That decode, not the dot, was what bounded the matvecs
// (~100 cycles a group on Q5_K). For the formats real models are made of, the
// decoders below unpack a whole 256-value super-block into plain bytes and
// per-group scale floats with straight-line loops the compiler vectorizes.
//
// They must produce exactly what the corpus decode would: the same bytes in
// `q`, and each scale/offset as the same single float product (d * (float)sc)
// the corpus forms. tests/cpu_parity_contract.cpp checks it against the
// emulated kernels; FLYWEIGHT_CPU_EMULATE=<name> falls back per kernel.
struct SuperBlockValues {
    alignas(64) signed char q[256];
    float scale_low[8], scale_high[8];
    float offset_low[8], offset_high[8];
};

template <auto Decode>
struct SuperBlock {
    static constexpr bool kAvailable = false;
};

FLYWEIGHT_Q8_FAMILY_INLINE void k_scale_min(const unsigned char* scales, int index,
                                            int* scale, int* minimum) {
    q5k_scale_min(scales, index, scale, minimum);
}

// Q4_K: d, dmin, 12 bytes of 6-bit scales/mins, 128 bytes of nibbles. Group j
// is the low (even j) or high (odd j) nibbles of bytes 32*(j/2)..+32.
template <>
struct SuperBlock<q4k_q8_decode> {
    static constexpr bool kAvailable = true;
    FLYWEIGHT_Q8_FAMILY_INLINE static void decode(const unsigned char* base,
                                                  SuperBlockValues& out) {
        const float d = __half2float(*reinterpret_cast<const __half*>(base));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(base + 2));
        for (int j = 0; j < 8; ++j) {
            int scale, minimum;
            k_scale_min(base + 4, j, &scale, &minimum);
            out.scale_low[j] = out.scale_high[j] = d * (float)scale;
            out.offset_low[j] = out.offset_high[j] = dmin * (float)minimum;
        }
        const unsigned char* quants = base + 16;
        for (int pair = 0; pair < 4; ++pair) {
            const unsigned char* src = quants + pair * 32;
            signed char* low = out.q + pair * 64;
            signed char* high = low + 32;
            for (int i = 0; i < 32; ++i) {
                low[i] = static_cast<signed char>(src[i] & 15);
                high[i] = static_cast<signed char>(src[i] >> 4);
            }
        }
    }
};

// Q5_K: Q4_K plus a 32-byte plane of fifth bits; group j takes bit j of each.
template <>
struct SuperBlock<q5k_q8_decode> {
    static constexpr bool kAvailable = true;
    FLYWEIGHT_Q8_FAMILY_INLINE static void decode(const unsigned char* base,
                                                  SuperBlockValues& out) {
        const float d = __half2float(*reinterpret_cast<const __half*>(base));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(base + 2));
        for (int j = 0; j < 8; ++j) {
            int scale, minimum;
            k_scale_min(base + 4, j, &scale, &minimum);
            out.scale_low[j] = out.scale_high[j] = d * (float)scale;
            out.offset_low[j] = out.offset_high[j] = dmin * (float)minimum;
        }
        const unsigned char* highs = base + 16;
        const unsigned char* quants = base + 48;
        for (int pair = 0; pair < 4; ++pair) {
            const unsigned char* src = quants + pair * 32;
            signed char* low = out.q + pair * 64;
            signed char* high = low + 32;
            const int low_bit = 2 * pair, high_bit = 2 * pair + 1;
            for (int i = 0; i < 32; ++i) {
                low[i] = static_cast<signed char>(
                    (src[i] & 15) | (((highs[i] >> low_bit) & 1) << 4));
                high[i] = static_cast<signed char>(
                    (src[i] >> 4) | (((highs[i] >> high_bit) & 1) << 4));
            }
        }
    }
};

// Q6_K: 128 bytes of low nibbles, 64 of 2-bit highs, 16 int8 scales (one per
// 16 values, so a group's two halves differ), then d. Values are q - 32.
template <>
struct SuperBlock<q6k_q8_decode> {
    static constexpr bool kAvailable = true;
    FLYWEIGHT_Q8_FAMILY_INLINE static void decode(const unsigned char* base,
                                                  SuperBlockValues& out) {
        const float d = __half2float(*reinterpret_cast<const __half*>(base + 208));
        const signed char* scales = reinterpret_cast<const signed char*>(base + 192);
        for (int j = 0; j < 8; ++j) {
            const int scale_base = (j >> 2) * 8 + (j & 3) * 2;
            out.scale_low[j] = d * (float)scales[scale_base];
            out.scale_high[j] = d * (float)scales[scale_base + 1];
        }
        for (int j = 0; j < 8; ++j) {
            const int half = j >> 2, lane_group = j & 3;
            const unsigned char* lows = base + half * 64 + ((lane_group & 1) ? 32 : 0);
            const unsigned char* highs = base + 128 + half * 32;
            const int shift = (lane_group >> 1) * 4;
            const int bit_shift = lane_group * 2;
            signed char* q = out.q + j * 32;
            for (int i = 0; i < 32; ++i)
                q[i] = static_cast<signed char>(
                    (((lows[i] >> shift) & 15) | (((highs[i] >> bit_shift) & 3) << 4))
                    - 32);
        }
    }
};

// Q8_0 as the flat 34-byte blocks the corpus addresses: fp16 d, 32 int8.
template <>
struct SuperBlock<q80_q8_decode> {
    static constexpr bool kAvailable = true;
    FLYWEIGHT_Q8_FAMILY_INLINE static void decode(const unsigned char* base,
                                                  SuperBlockValues& out) {
        for (int j = 0; j < 8; ++j) {
            const unsigned char* block = base + j * 34;
            std::uint16_t bits;
            std::memcpy(&bits, block, sizeof(bits));
            const float d = __half2float(__ushort_as_half(bits));
            out.scale_low[j] = out.scale_high[j] = d;
            std::memcpy(out.q + j * 32, block + 2, 32);
        }
    }
};

// values[r] = dot(weight row, activation row r) for r < op.rows (<= Cap).
// The per-group expression is FLYWEIGHT_Q8_MATVEC_ROWS(_MIN)'s, verbatim.
template <int Cap, bool Minimum>
FLYWEIGHT_Q8_FAMILY_INLINE void accumulate_group(
    const Operands& op, int g, const signed char* weights, float scale_low,
    float scale_high, float offset_low, float offset_high,
    float (&lanes)[Cap][kLanes]) {
    for (int r = 0; r < Cap; ++r) {
        if (r >= op.rows) break;
        const signed char* acts = op.vectors
            + static_cast<long long>(r) * op.input_size
            + static_cast<long long>(g) * 32;
        const int dot_low = dot16(weights, acts);
        const int dot_high = dot16(weights + 16, acts + 16);
        const float activation_scale =
            op.activation_scales[static_cast<std::size_t>(r) * op.groups + g];
        float value;
        if constexpr (Minimum) {
            const int* slot = op.sums
                + (static_cast<std::size_t>(r) * op.groups + g) * 2;
            value = ((float)dot_low * scale_low - (float)slot[0] * offset_low
                     + (float)dot_high * scale_high
                     - (float)slot[1] * offset_high)
                * activation_scale;
        } else {
            value = ((float)dot_low * scale_low + (float)dot_high * scale_high)
                * activation_scale;
        }
        lanes[r][g & (kLanes - 1)] += value;
    }
}

// A whole decoded super-block against every activation row. Group j of the
// block lands in lane j (the block starts on a multiple of 8, and lanes are
// g & 7), so the eight per-group values are computed and accumulated as one
// eight-wide vector. Each element is accumulate_group's expression unchanged.
template <int Cap, bool Minimum>
FLYWEIGHT_Q8_FAMILY_INLINE void accumulate_block(const Operands& op, int first_group,
                                                 const SuperBlockValues& block,
                                                 float (&lanes)[Cap][kLanes]) {
    for (int r = 0; r < Cap; ++r) {
        if (r >= op.rows) break;
        const signed char* acts = op.vectors
            + static_cast<long long>(r) * op.input_size
            + static_cast<long long>(first_group) * 32;
        int dots[16];
        for (int half = 0; half < 16; ++half)
            dots[half] = dot16(block.q + half * 16, acts + half * 16);
        const float* activation_scale = op.activation_scales
            + static_cast<std::size_t>(r) * op.groups + first_group;
        float value[8];
        if constexpr (Minimum) {
            const int* slot = op.sums
                + (static_cast<std::size_t>(r) * op.groups + first_group) * 2;
            for (int j = 0; j < 8; ++j)
                value[j] = ((float)dots[2 * j] * block.scale_low[j]
                            - (float)slot[2 * j] * block.offset_low[j]
                            + (float)dots[2 * j + 1] * block.scale_high[j]
                            - (float)slot[2 * j + 1] * block.offset_high[j])
                    * activation_scale[j];
        } else {
            for (int j = 0; j < 8; ++j)
                value[j] = ((float)dots[2 * j] * block.scale_low[j]
                            + (float)dots[2 * j + 1] * block.scale_high[j])
                    * activation_scale[j];
        }
        for (int j = 0; j < 8; ++j) lanes[r][j] += value[j];
    }
}

template <auto Decode, int Stride, int Cap, bool Minimum>
FLYWEIGHT_Q8_FAMILY_INLINE void row_values(const Operands& op,
                                           const unsigned char* row_data,
                                           float* values) {
    float lanes[Cap][kLanes] = {};
    int g = 0;
    if constexpr (SuperBlock<Decode>::kAvailable) {
        // Whole super-blocks; a flat format's partial tail (Q8_0 rows whose
        // width is not a multiple of 256) falls through to the group loop.
        SuperBlockValues block;
        for (; g + 8 <= op.groups; g += 8) {
            SuperBlock<Decode>::decode(
                row_data + static_cast<long long>(g >> 3) * Stride, block);
            accumulate_block<Cap, Minimum>(op, g, block, lanes);
        }
    }
    for (; g < op.groups; ++g) {
        int words[8];
        float scale_low = 0.0f, scale_high = 0.0f;
        float offset_low = 0.0f, offset_high = 0.0f;
        if constexpr (Minimum) {
            Decode(row_data, g, words, &scale_low, &scale_high, &offset_low,
                   &offset_high);
        } else {
            Decode(row_data, g, words, &scale_low, &scale_high);
        }
        signed char weights[32];
        std::memcpy(weights, words, sizeof(weights));
        accumulate_group<Cap, Minimum>(op, g, weights, scale_low, scale_high,
                                       offset_low, offset_high, lanes);
    }
    for (int r = 0; r < Cap; ++r) {
        if (r >= op.rows) break;
        values[r] = reduce_lanes(lanes[r]);
    }
}

// outputs[r * output_size + row] for every row in [first, last).
template <auto Decode, int Stride, int Cap, bool Minimum>
FLYWEIGHT_Q8_FAMILY_INLINE void matvec_range_body(const Operands& op,
                                                  float* outputs, int output_size,
                                                  std::uint64_t first,
                                                  std::uint64_t last) {
    for (std::uint64_t row = first; row < last; ++row) {
        float values[Cap];
        row_values<Decode, Stride, Cap, Minimum>(
            op, weight_row<Stride, Minimum>(op, row), values);
        for (int r = 0; r < Cap; ++r) {
            if (r >= op.rows) break;
            outputs[static_cast<long long>(r) * output_size + row] = values[r];
        }
    }
}

// The LM head's packed (ordered-float << 32 | ~row) maximum over [first, last).
template <auto Decode, int Stride, bool Minimum>
FLYWEIGHT_Q8_FAMILY_INLINE unsigned long long lm_head_range_body(
    const Operands& op, std::uint64_t first, std::uint64_t last) {
    unsigned long long best = 0ull;
    for (std::uint64_t row = first; row < last; ++row) {
        float value;
        row_values<Decode, Stride, 1, Minimum>(
            op, weight_row<Stride, Minimum>(op, row), &value);
        const unsigned int bits = __float_as_uint(value);
        const unsigned int ordered =
            bits ^ ((static_cast<int>(bits) < 0) ? 0xffffffffu : 0x80000000u);
        const unsigned long long key =
            (static_cast<unsigned long long>(ordered) << 32)
            | (0xffffffffu - static_cast<unsigned int>(row));
        best = std::max(best, key);
    }
    return best;
}

// ISA copies. The bodies above are always_inline and default-target, so each
// wrapper compiles them -- and the corpus decode they call -- for its own ISA.
template <auto Decode, int Stride, int Cap, bool Minimum>
void matvec_range_baseline(const Operands& op, float* outputs, int output_size,
                           std::uint64_t first, std::uint64_t last) {
    matvec_range_body<Decode, Stride, Cap, Minimum>(op, outputs, output_size,
                                                    first, last);
}
template <auto Decode, int Stride, bool Minimum>
unsigned long long lm_head_range_baseline(const Operands& op, std::uint64_t first,
                                          std::uint64_t last) {
    return lm_head_range_body<Decode, Stride, Minimum>(op, first, last);
}

#if FLYWEIGHT_Q8_FAMILY_ISA_CLONES
// Exactly the extensions flyweight_cpu_features() verifies for each bit
// (q4_dispatch.cpp): AVX2 + FMA + F16C, and AVX-512 F + BW on top. Anything
// more would let the compiler emit instructions the probe never checked.
#define FLYWEIGHT_Q8_FAMILY_AVX2 __attribute__((target("avx2,fma,f16c")))
#define FLYWEIGHT_Q8_FAMILY_AVX512 \
    __attribute__((target("avx512f,avx512bw,avx2,fma,f16c")))

template <auto Decode, int Stride, int Cap, bool Minimum>
FLYWEIGHT_Q8_FAMILY_AVX2 void matvec_range_avx2(const Operands& op, float* outputs,
                                                int output_size,
                                                std::uint64_t first,
                                                std::uint64_t last) {
    matvec_range_body<Decode, Stride, Cap, Minimum>(op, outputs, output_size,
                                                    first, last);
}
template <auto Decode, int Stride, int Cap, bool Minimum>
FLYWEIGHT_Q8_FAMILY_AVX512 void matvec_range_avx512(const Operands& op,
                                                    float* outputs,
                                                    int output_size,
                                                    std::uint64_t first,
                                                    std::uint64_t last) {
    matvec_range_body<Decode, Stride, Cap, Minimum>(op, outputs, output_size,
                                                    first, last);
}
template <auto Decode, int Stride, bool Minimum>
FLYWEIGHT_Q8_FAMILY_AVX2 unsigned long long lm_head_range_avx2(
    const Operands& op, std::uint64_t first, std::uint64_t last) {
    return lm_head_range_body<Decode, Stride, Minimum>(op, first, last);
}
template <auto Decode, int Stride, bool Minimum>
FLYWEIGHT_Q8_FAMILY_AVX512 unsigned long long lm_head_range_avx512(
    const Operands& op, std::uint64_t first, std::uint64_t last) {
    return lm_head_range_body<Decode, Stride, Minimum>(op, first, last);
}
#endif

template <auto Decode, int Stride, int Cap, bool Minimum>
auto pick_matvec_range() {
#if FLYWEIGHT_Q8_FAMILY_ISA_CLONES
    switch (isa()) {
        case Isa::avx512: return &matvec_range_avx512<Decode, Stride, Cap, Minimum>;
        case Isa::avx2: return &matvec_range_avx2<Decode, Stride, Cap, Minimum>;
        case Isa::baseline: break;
    }
#endif
    return &matvec_range_baseline<Decode, Stride, Cap, Minimum>;
}

template <auto Decode, int Stride, bool Minimum>
auto pick_lm_head_range() {
#if FLYWEIGHT_Q8_FAMILY_ISA_CLONES
    switch (isa()) {
        case Isa::avx512: return &lm_head_range_avx512<Decode, Stride, Minimum>;
        case Isa::avx2: return &lm_head_range_avx2<Decode, Stride, Minimum>;
        case Isa::baseline: break;
    }
#endif
    return &lm_head_range_baseline<Decode, Stride, Minimum>;
}

// The activation sums the _MIN formats subtract their minima against. They
// depend only on (activation row, group), so they are taken once per launch
// instead of once per weight row.
// Per-launch activation tables: the fp16 scales converted once, and for the
// _MIN formats the activation sums.
template <bool Minimum>
void fill_sums(Operands& op, std::vector<int>& storage);

template <bool Minimum>
void prepare_activations(Operands& op, std::vector<float>& scales,
                         std::vector<int>& sums) {
    scales.resize(static_cast<std::size_t>(op.rows) * op.groups);
    for (int r = 0; r < op.rows; ++r)
        for (int g = 0; g < op.groups; ++g)
            scales[static_cast<std::size_t>(r) * op.groups + g] = __half2float(
                op.vector_scales[static_cast<long long>(r) * op.scale_stride + g]);
    op.activation_scales = scales.data();
    fill_sums<Minimum>(op, sums);
}

template <bool Minimum>
void fill_sums(Operands& op, std::vector<int>& storage) {
    if constexpr (!Minimum) {
        op.sums = nullptr;
    } else {
        storage.resize(static_cast<std::size_t>(op.rows) * op.groups * 2);
        for (int r = 0; r < op.rows; ++r) {
            for (int g = 0; g < op.groups; ++g) {
                const signed char* acts = op.vectors
                    + static_cast<long long>(r) * op.input_size
                    + static_cast<long long>(g) * 32;
                int* slot = storage.data()
                    + (static_cast<std::size_t>(r) * op.groups + g) * 2;
                slot[0] = sum16(acts);
                slot[1] = sum16(acts + 16);
            }
        }
        op.sums = storage.data();
    }
}

template <bool Minimum>
int group_count(int input_size) {
    // The _MIN kernels count whole super-blocks; the rest count 32-value groups.
    return Minimum ? ((input_size >> 8) << 3) : (input_size >> 5);
}

// --- FLYWEIGHT_Q8_MATVEC ----------------------------------------------------
//
//   name(const unsigned char* packed, const signed char* vector,
//        const __half* vector_scales, float* output,
//        int input_size, int output_size)
//
// One output row per block in the corpus; the batched kernel below at a
// single activation row.
template <auto Decode, int Stride, bool Minimum>
void matvec(const ::flyweight::cpu::Launch& launch, void** arguments) {
    Operands op{};
    op.packed = argument<const unsigned char*>(arguments, 0);
    op.vectors = argument<const signed char*>(arguments, 1);
    op.vector_scales = argument<const __half*>(arguments, 2);
    float* output = argument<float*>(arguments, 3);
    op.input_size = argument<int>(arguments, 4);
    const int output_size = argument<int>(arguments, 5);
    // The single-row corpus kernel always walks input_size >> 5 groups.
    op.groups = op.input_size >> 5;
    op.rows = 1;
    op.scale_stride = 0;
    std::vector<float> scales;
    std::vector<int> sums;
    prepare_activations<Minimum>(op, scales, sums);
    // Rows past the grid are never touched by the corpus kernel either.
    const std::uint64_t rows = std::min<std::uint64_t>(
        static_cast<std::uint64_t>(std::max(output_size, 0)), launch.grid_x);
    const auto range = pick_matvec_range<Decode, Stride, 1, Minimum>();
    parallel_ranges(rows, [&](std::uint64_t first, std::uint64_t last) {
        range(op, output, output_size, first, last);
    });
}

// --- FLYWEIGHT_Q8_LM_HEAD ---------------------------------------------------
//
//   name(const unsigned char* packed, const signed char* vector,
//        const __half* vector_scales, unsigned long long* winners,
//        int input_size, int output_size)
//
// A warp per vocabulary row, eight per block, fused argmax folded into
// *winners with atomicMax. Each chunk keeps its own best and the launch folds
// them into *winners once, after the pool joins.
template <auto Decode, int Stride, bool Minimum>
void lm_head(const ::flyweight::cpu::Launch& launch, void** arguments) {
    Operands op{};
    op.packed = argument<const unsigned char*>(arguments, 0);
    op.vectors = argument<const signed char*>(arguments, 1);
    op.vector_scales = argument<const __half*>(arguments, 2);
    auto* winners = argument<unsigned long long*>(arguments, 3);
    op.input_size = argument<int>(arguments, 4);
    const int output_size = argument<int>(arguments, 5);
    op.groups = op.input_size >> 5;
    op.rows = 1;
    op.scale_stride = 0;
    std::vector<float> scales;
    std::vector<int> sums;
    prepare_activations<Minimum>(op, scales, sums);
    const std::uint64_t warps = std::min<std::uint64_t>(launch.block_x >> 5, 8);
    const std::uint64_t rows = std::min<std::uint64_t>(
        static_cast<std::uint64_t>(std::max(output_size, 0)),
        static_cast<std::uint64_t>(launch.grid_x) * warps);

    const auto range = pick_lm_head_range<Decode, Stride, Minimum>();
    std::atomic<unsigned long long> best{0ull};
    parallel_ranges(rows, [&](std::uint64_t first, std::uint64_t last) {
        const unsigned long long local = range(op, first, last);
        unsigned long long seen = best.load(std::memory_order_relaxed);
        while (local > seen &&
               !best.compare_exchange_weak(seen, local, std::memory_order_relaxed)) {
        }
    });
    if (rows > 0) *winners = std::max(*winners, best.load());
}

// --- FLYWEIGHT_Q8_MATVEC_ROWS / _ROWS_MIN / _MATMUL_TILED -------------------
//
//   name(const unsigned char* packed, const signed char* vectors,
//        const __half* vector_scales, float* outputs,
//        int input_size, int output_size, int rows, int scale_stride)
//
// Each weight group is decoded once and dotted against every activation row,
// as in the corpus. FLYWEIGHT_Q8_MATMUL_TILED is the same per-group arithmetic
// for the prefill batch (up to FLYWEIGHT_Q8_TILE_TOKENS tokens,
// FLYWEIGHT_Q8_TILE_ROWS output rows per block); its shared-memory tiling
// exists to share decoded words across warps, which a host loop gets by
// keeping them in registers.
template <auto Decode, int Stride, int Cap, bool Minimum, int RowsPerBlock = 1>
void matvec_rows(const ::flyweight::cpu::Launch& launch, void** arguments) {
    Operands op{};
    op.packed = argument<const unsigned char*>(arguments, 0);
    op.vectors = argument<const signed char*>(arguments, 1);
    op.vector_scales = argument<const __half*>(arguments, 2);
    float* outputs = argument<float*>(arguments, 3);
    op.input_size = argument<int>(arguments, 4);
    const int output_size = argument<int>(arguments, 5);
    op.rows = std::min(argument<int>(arguments, 6), Cap);
    op.scale_stride = argument<int>(arguments, 7);
    if (op.rows <= 0) return;
    op.groups = group_count<Minimum>(op.input_size);
    std::vector<float> scales;
    std::vector<int> sums;
    prepare_activations<Minimum>(op, scales, sums);
    const std::uint64_t outputs_count = std::min<std::uint64_t>(
        static_cast<std::uint64_t>(std::max(output_size, 0)),
        static_cast<std::uint64_t>(launch.grid_x) * RowsPerBlock);
    const auto range = pick_matvec_range<Decode, Stride, Cap, Minimum>();
    parallel_ranges(outputs_count, [&](std::uint64_t first, std::uint64_t last) {
        range(op, outputs, output_size, first, last);
    });
}

// --- quantize_q8_blocks / quantize_q8_blocks_rows ---------------------------
//
//   quantize_q8_blocks(const float* input, signed char* output,
//                      __half* scales, int elements)
//   quantize_q8_blocks_rows(const float* input, signed char* output,
//                           __half* scales, int elements, int scale_stride)
//
// The activation quantizer every Q8 matvec above consumes, one 32-value block
// per warp. The block maximum is order-independent and the per-value rounding
// is the corpus's own expression, so these write identical bytes. Emulated,
// the shuffle max made this ~2 ms per call at ~65 calls per decoded token.
inline void quantize_block(const float* input, signed char* output,
                           __half* scale_slot, int begin, int elements) {
    float maximum = 0.0f;
    for (int lane = 0; lane < 32; ++lane) {
        const int index = begin + lane;
        const float value = index < elements ? input[index] : 0.0f;
        maximum = fmaxf(maximum, fabsf(value));
    }
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    if (scale_slot != nullptr) *scale_slot = __float2half(scale);
    const int end = std::min(begin + 32, elements);
    for (int index = begin; index < end; ++index) {
        const int quantized =
            max(-127, min(127, __float2int_rn(input[index] / scale)));
        output[index] = static_cast<signed char>(quantized);
    }
}

void quantize_blocks(const ::flyweight::cpu::Launch& launch, void** arguments) {
    const auto* input = argument<const float*>(arguments, 0);
    auto* output = argument<signed char*>(arguments, 1);
    auto* scales = argument<__half*>(arguments, 2);
    const int elements = argument<int>(arguments, 3);
    // Lane 0 of every launched block writes its scale, in range or not.
    for (std::uint32_t block = 0; block < launch.grid_x; ++block)
        quantize_block(input, output, scales + block,
                       static_cast<int>(block) * 32, elements);
}

void quantize_blocks_rows(const ::flyweight::cpu::Launch& launch,
                          void** arguments) {
    const auto* input = argument<const float*>(arguments, 0);
    auto* output = argument<signed char*>(arguments, 1);
    auto* scales = argument<__half*>(arguments, 2);
    const int elements = argument<int>(arguments, 3);
    const int scale_stride = argument<int>(arguments, 4);
    const std::uint64_t blocks = std::min<std::uint64_t>(
        static_cast<std::uint64_t>(launch.grid_x) * (launch.block_x >> 5),
        static_cast<std::uint64_t>((std::max(elements, 0) + 31) / 32));
    parallel_ranges(launch.grid_y, [&](std::uint64_t first, std::uint64_t last) {
        for (std::uint64_t row = first; row < last; ++row) {
            const float* row_input = input + static_cast<long long>(row) * elements;
            signed char* row_output = output + static_cast<long long>(row) * elements;
            __half* row_scales = scales + static_cast<long long>(row) * scale_stride;
            for (std::uint64_t block = 0; block < blocks; ++block)
                quantize_block(row_input, row_output, row_scales + block,
                               static_cast<int>(block) * 32, elements);
        }
    });
}

}  // namespace q8_family

// Registration. Each corpus macro instantiation gets its host driver under the
// same name; register_native_kernel refuses a name the corpus lacks, so a
// format dropped from the corpus fails loudly in the parity contract rather
// than leaving a dangling entry.
#define FLYWEIGHT_CPU_Q8_REGISTER(name, ...)                                   \
    static const ::flyweight::cpu::NativeKernelRegistration                    \
        flyweight_cpu_q8_registration_##name(#name, &__VA_ARGS__)
#define FLYWEIGHT_CPU_Q8_MATVEC(name, decode_fn, stride, minimum)              \
    FLYWEIGHT_CPU_Q8_REGISTER(name, q8_family::matvec<decode_fn, stride, minimum>)
#define FLYWEIGHT_CPU_Q8_LM_HEAD(name, decode_fn, stride, minimum)             \
    FLYWEIGHT_CPU_Q8_REGISTER(name, q8_family::lm_head<decode_fn, stride, minimum>)
#define FLYWEIGHT_CPU_Q8_ROWS_CAP(name, decode_fn, stride, cap, minimum)       \
    FLYWEIGHT_CPU_Q8_REGISTER(                                                 \
        name, q8_family::matvec_rows<decode_fn, stride, cap, minimum>)
#define FLYWEIGHT_CPU_Q8_ROWS_FAMILY(name, decode_fn, stride, minimum)          \
    FLYWEIGHT_CPU_Q8_ROWS_CAP(name, decode_fn, stride, FLYWEIGHT_Q8_ROWS, minimum); \
    FLYWEIGHT_CPU_Q8_ROWS_CAP(name##_r2, decode_fn, stride, 2, minimum);       \
    FLYWEIGHT_CPU_Q8_ROWS_CAP(name##_r3, decode_fn, stride, 3, minimum);       \
    FLYWEIGHT_CPU_Q8_ROWS_CAP(name##_r4, decode_fn, stride, 4, minimum);       \
    FLYWEIGHT_CPU_Q8_ROWS_CAP(name##_r5, decode_fn, stride, 5, minimum);       \
    FLYWEIGHT_CPU_Q8_ROWS_CAP(name##_r6, decode_fn, stride, 6, minimum);       \
    FLYWEIGHT_CPU_Q8_ROWS_CAP(name##_r7, decode_fn, stride, 7, minimum)
#define FLYWEIGHT_CPU_Q8_TILED(name, decode_fn, stride)                        \
    FLYWEIGHT_CPU_Q8_REGISTER(                                                 \
        name, q8_family::matvec_rows<decode_fn, stride,                        \
            FLYWEIGHT_Q8_TILE_TOKENS, false, FLYWEIGHT_Q8_TILE_ROWS>)

// One line per format, mirroring the corpus's FLYWEIGHT_Q8_MATVEC / _LM_HEAD /
// _MATVEC_ROWS(_MIN) / _MATMUL_TILED instantiations. `true` marks the
// asymmetric K-quants the corpus serves through the _MIN macros. The parity
// contract lists the same set, so a format added to the corpus without a line
// here shows up as a gap there.
#define FLYWEIGHT_CPU_Q8_FORMAT(format, stride, minimum)                       \
    FLYWEIGHT_CPU_Q8_MATVEC(format##_q8_matvec_transposed_warp,                \
                            format##_q8_decode, stride, minimum);              \
    FLYWEIGHT_CPU_Q8_ROWS_FAMILY(format##_q8_matvec_transposed_rows,           \
                                 format##_q8_decode, stride, minimum)
#define FLYWEIGHT_CPU_Q8_FORMAT_LM_HEAD(format, stride, minimum)               \
    FLYWEIGHT_CPU_Q8_LM_HEAD(format##_q8_lm_head_argmax_warp,                  \
                             format##_q8_decode, stride, minimum)
#define FLYWEIGHT_CPU_Q8_FORMAT_TILED(format, stride)                          \
    FLYWEIGHT_CPU_Q8_TILED(format##_q8_matmul_tiled, format##_q8_decode, stride)

FLYWEIGHT_CPU_Q8_FORMAT(iq1m, 56, false);
FLYWEIGHT_CPU_Q8_FORMAT(iq1s, 50, false);
FLYWEIGHT_CPU_Q8_FORMAT(iq2s, 82, false);
FLYWEIGHT_CPU_Q8_FORMAT(iq2xs, 74, false);
FLYWEIGHT_CPU_Q8_FORMAT(iq2xxs, 66, false);
FLYWEIGHT_CPU_Q8_FORMAT(iq3s, 110, false);
FLYWEIGHT_CPU_Q8_FORMAT(iq3xxs, 98, false);
FLYWEIGHT_CPU_Q8_FORMAT(iq4nl, 144, false);
FLYWEIGHT_CPU_Q8_FORMAT(iq4xs, 136, false);
FLYWEIGHT_CPU_Q8_FORMAT(q2k, 84, true);
FLYWEIGHT_CPU_Q8_FORMAT(q3k, 110, false);
FLYWEIGHT_CPU_Q8_FORMAT(q4k, 144, true);
FLYWEIGHT_CPU_Q8_FORMAT(q5k, 176, true);
FLYWEIGHT_CPU_Q8_FORMAT(q6k, 210, false);
FLYWEIGHT_CPU_Q8_FORMAT(q80, 272, false);

FLYWEIGHT_CPU_Q8_FORMAT_LM_HEAD(iq1s, 50, false);
FLYWEIGHT_CPU_Q8_FORMAT_LM_HEAD(iq2xxs, 66, false);
FLYWEIGHT_CPU_Q8_FORMAT_LM_HEAD(iq3s, 110, false);
FLYWEIGHT_CPU_Q8_FORMAT_LM_HEAD(iq3xxs, 98, false);
FLYWEIGHT_CPU_Q8_FORMAT_LM_HEAD(iq4nl, 144, false);
FLYWEIGHT_CPU_Q8_FORMAT_LM_HEAD(q2k, 84, true);
FLYWEIGHT_CPU_Q8_FORMAT_LM_HEAD(q3k, 110, false);
FLYWEIGHT_CPU_Q8_FORMAT_LM_HEAD(q4k, 144, true);
FLYWEIGHT_CPU_Q8_FORMAT_LM_HEAD(q5k, 176, true);
FLYWEIGHT_CPU_Q8_FORMAT_LM_HEAD(q6k, 210, false);
FLYWEIGHT_CPU_Q8_FORMAT_LM_HEAD(q80, 272, false);

FLYWEIGHT_CPU_Q8_FORMAT_TILED(iq1m, 56);
FLYWEIGHT_CPU_Q8_FORMAT_TILED(iq1s, 50);
FLYWEIGHT_CPU_Q8_FORMAT_TILED(iq2s, 82);
FLYWEIGHT_CPU_Q8_FORMAT_TILED(iq2xs, 74);
FLYWEIGHT_CPU_Q8_FORMAT_TILED(iq2xxs, 66);
FLYWEIGHT_CPU_Q8_FORMAT_TILED(iq3s, 110);
FLYWEIGHT_CPU_Q8_FORMAT_TILED(iq3xxs, 98);
FLYWEIGHT_CPU_Q8_FORMAT_TILED(iq4nl, 144);
FLYWEIGHT_CPU_Q8_FORMAT_TILED(iq4xs, 136);
FLYWEIGHT_CPU_Q8_FORMAT_TILED(q3k, 110);
FLYWEIGHT_CPU_Q8_FORMAT_TILED(q6k, 210);
FLYWEIGHT_CPU_Q8_FORMAT_TILED(q80, 272);

FLYWEIGHT_CPU_Q8_REGISTER(quantize_q8_blocks, q8_family::quantize_blocks);
FLYWEIGHT_CPU_Q8_REGISTER(quantize_q8_blocks_rows, q8_family::quantize_blocks_rows);
