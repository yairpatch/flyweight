#pragma once

// K-quant *encoders*: f32 -> Q4_K / Q5_K / Q6_K super-blocks.
//
// The repo has only ever consumed these formats; the sole producer was
// qwen_pack_q8_0. Loading an HF checkpoint means quantizing at load time, which
// needs the rest.
//
// These follow llama.cpp's reference quantizers (ggml-quants.c) closely and
// deliberately. The block layouts are not the hard part -- those are pinned by
// the decoders in qwen_kquant.h, and each packer below is checked against its
// decoder. The hard part is *scale selection*: a naive absmax/min-max fit is
// materially worse than llama.cpp's search, and the difference shows up as
// perplexity, not as a crash. So the two search routines are ports of
// `make_qx_quants` and `make_qkx2_quants` rather than inventions.
//
// Not covered: the IQ types, which need an importance matrix.
//
// MUST be compiled with `-ffp-contract=off`, which is why the only supported
// entry points are the ones in qwen_kquant_pack_api.hpp: their translation unit
// carries the flag. GCC's default fuses the multiply and the add in every
// accumulation below, which changes their rounding, and that is not cosmetic --
// it flips which trial step wins on roughly one sub-block in three hundred, and
// it makes the output of a *quantizer whose result is cached and fingerprinted*
// depend on the build rather than on the input. It also decides whether the
// AVX2 and scalar paths agree, since only one of them has an FMA to fuse into.
// `hf_quantize_tiling_contract` pins the bytes so this cannot regress silently.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "qwen_kquant.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace qwen_kpack {

constexpr int kSuperBlock = 256;
constexpr std::uint32_t kQ8BlockBytes = 34;

// Pack `count` f32 values (a multiple of 32) into Q8_0 blocks: one f16 scale
// followed by 32 int8 codes. Matches qwen_q8_value and the q8 CUDA kernels.
inline void qwen_pack_q8_0(const float* values, std::uint64_t count, std::uint8_t* out) {
    for(std::uint64_t block=0;block*32<count;++block){
        const float* source=values+block*32;
        float absmax=0.0f;
        for(int i=0;i<32;++i)absmax=std::max(absmax,std::fabs(source[i]));
        const float scale=absmax/127.0f;
        // Quantize against the scale that will actually be stored, so decode
        // reproduces these codes exactly rather than the pre-rounding value.
        const std::uint16_t scale_bits=qwen_half_bits(scale);
        const float stored=qwen_half_value(scale_bits);
        const float inverse=stored>0.0f?1.0f/stored:0.0f;
        auto* destination=out+block*kQ8BlockBytes;
        std::memcpy(destination,&scale_bits,2);
        for(int i=0;i<32;++i){
            const int code=static_cast<int>(std::lround(source[i]*inverse));
            destination[2+i]=static_cast<std::uint8_t>(
                static_cast<std::int8_t>(std::min(127,std::max(-127,code))));
        }
    }
}

// llama.cpp's rounding: adding 1.5*2^23 forces the mantissa to absorb the
// fractional part, leaving the integer in the low bits.
//
// Kept in this form only because it is what the reference does. It was checked
// against `std::lrintf`, which is the obvious replacement: on real weights the
// two produce byte-identical output for all three formats, so the residual
// disagreement with llama.cpp measured below is NOT a rounding-mode artifact.
inline int nearest_int(float value) {
    float shifted = value + 12582912.0f;
    std::int32_t bits;
    std::memcpy(&bits, &shifted, sizeof(bits));
    return (bits & 0x007fffff) - 0x00400000;
}

// Symmetric signed fit for Q6_K: choose the scale minimizing weighted squared
// error for codes in [-nmax, nmax-1]. `weights` is x^2 (rmse_type 1), which is
// what the reference uses for Q6_K.
//
// Returns the scale; fills `codes` with the *biased* codes (code + nmax) so the
// caller can pack them directly.
inline float make_qx_quants(int count, int nmax, const float* x, std::int8_t* codes) {
    float max = 0.0f, absmax = 0.0f;
    for (int i = 0; i < count; ++i) {
        const float a = std::fabs(x[i]);
        if (a > absmax) { absmax = a; max = x[i]; }
    }
    if (absmax < 1e-30f) {
        for (int i = 0; i < count; ++i) codes[i] = 0;
        return 0.0f;
    }
    float iscale = -static_cast<float>(nmax) / max;
    float sumlx = 0.0f, suml2 = 0.0f;
    for (int i = 0; i < count; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = std::max(-nmax, std::min(nmax - 1, l));
        codes[i] = static_cast<std::int8_t>(l + nmax);
        const float w = x[i] * x[i];
        sumlx += w * x[i] * static_cast<float>(l);
        suml2 += w * static_cast<float>(l) * static_cast<float>(l);
    }
    float scale = suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    float best = scale * sumlx;
    // Nudge the trial scale around the absmax fit. Only a better weighted fit
    // is committed, so this can never lose to the starting point.
    for (int step = -9; step <= 9; ++step) {
        if (step == 0) continue;
        const float trial = -(static_cast<float>(nmax) + 0.1f * static_cast<float>(step)) / max;
        sumlx = suml2 = 0.0f;
        for (int i = 0; i < count; ++i) {
            int l = nearest_int(trial * x[i]);
            l = std::max(-nmax, std::min(nmax - 1, l));
            const float w = x[i] * x[i];
            sumlx += w * x[i] * static_cast<float>(l);
            suml2 += w * static_cast<float>(l) * static_cast<float>(l);
        }
        if (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < count; ++i) {
                int l = nearest_int(trial * x[i]);
                codes[i] = static_cast<std::int8_t>(
                    nmax + std::max(-nmax, std::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

// Asymmetric fit for Q4_K/Q5_K: codes in [0, nmax] with a per-sub-block
// minimum, so a block reconstructs as `scale * code + min`. Least squares over
// both parameters at each trial scale, keeping the best weighted error.
//
// `out_min` receives the *negated* minimum, matching the stored convention
// (decode subtracts dmin*m).
inline float make_qkx2_quants(int count, int nmax, const float* x, const float* weights,
                              std::uint8_t* codes, float* out_min,
                              float rmin, float rdelta, int nstep) {
    float minimum = x[0], maximum = x[0];
    float sum_w = weights[0], sum_x = sum_w * x[0];
    for (int i = 1; i < count; ++i) {
        minimum = std::min(minimum, x[i]);
        maximum = std::max(maximum, x[i]);
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    // The reconstruction offset is only ever negative; a positive minimum is
    // clamped so an all-positive block still starts from zero.
    if (minimum > 0.0f) minimum = 0.0f;
    if (maximum == minimum) {
        for (int i = 0; i < count; ++i) codes[i] = 0;
        *out_min = -minimum;
        return 0.0f;
    }

    float iscale = static_cast<float>(nmax) / (maximum - minimum);
    float scale = 1.0f / iscale;
    float best_error = 0.0f;
    for (int i = 0; i < count; ++i) {
        const int l = nearest_int(iscale * (x[i] - minimum));
        codes[i] = static_cast<std::uint8_t>(std::max(0, std::min(nmax, l)));
        const float diff = scale * static_cast<float>(codes[i]) + minimum - x[i];
        best_error += weights[i] * diff * diff;
    }

    std::uint8_t trial_codes[32];
    for (int step = 0; step <= nstep; ++step) {
        const float trial_iscale =
            (rmin + rdelta * static_cast<float>(step) + static_cast<float>(nmax)) /
            (maximum - minimum);
        float sum_l = 0.0f, sum_l2 = 0.0f, sum_xl = 0.0f;
        for (int i = 0; i < count; ++i) {
            int l = nearest_int(trial_iscale * (x[i] - minimum));
            l = std::max(0, std::min(nmax, l));
            trial_codes[i] = static_cast<std::uint8_t>(l);
            const float w = weights[i];
            sum_l += w * static_cast<float>(l);
            sum_l2 += w * static_cast<float>(l) * static_cast<float>(l);
            sum_xl += w * static_cast<float>(l) * x[i];
        }
        // Normal equations for (scale, min) against these codes.
        const float determinant = sum_w * sum_l2 - sum_l * sum_l;
        if (determinant <= 0.0f) continue;
        float this_scale = (sum_w * sum_xl - sum_x * sum_l) / determinant;
        float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / determinant;
        if (this_min > 0.0f) {
            this_min = 0.0f;
            this_scale = sum_l2 > 0.0f ? sum_xl / sum_l2 : 0.0f;
        }
        float error = 0.0f;
        for (int i = 0; i < count; ++i) {
            const float diff =
                this_scale * static_cast<float>(trial_codes[i]) + this_min - x[i];
            error += weights[i] * diff * diff;
        }
        if (error < best_error) {
            for (int i = 0; i < count; ++i) codes[i] = trial_codes[i];
            best_error = error;
            scale = this_scale;
            minimum = this_min;
        }
    }
    *out_min = -minimum;
    return scale;
}

// The 6-bit scale/min pair for sub-block `index`, as stored in the 12 shared
// bytes. Mirrors the unpacking inside qwen_q4k_value / qwen_q5_value.
inline void pack_scale_min_k4(int index, std::uint8_t* scales, std::uint8_t scale,
                              std::uint8_t minimum) {
    if (index < 4) {
        scales[index] = scale;
        scales[index + 4] = minimum;
    } else {
        scales[index + 4] = static_cast<std::uint8_t>((scale & 0xF) | ((minimum & 0xF) << 4));
        scales[index - 4] = static_cast<std::uint8_t>(scales[index - 4] | ((scale >> 4) << 6));
        scales[index] = static_cast<std::uint8_t>(scales[index] | ((minimum >> 4) << 6));
    }
}

// The scale search, eight sub-blocks at a time.
//
// This is the whole cost of Q4_K/Q5_K packing: `make_qkx2_quants` runs a
// 21-step search over every 32-element sub-block, and at ~30M elements/s/core
// that is most of an HF checkpoint's load time.
//
// The vectorization is *across* sub-blocks rather than within one. That choice
// is what makes it bit-exact rather than merely close: lane j runs sub-block j
// through the identical sequence of scalar operations in the identical order,
// so no sum is reassociated and no comparison can land differently. Vectorizing
// *within* a sub-block would reassociate the four reductions, occasionally flip
// which trial step wins, and silently change the weights a checkpoint quantizes
// to -- which is the kind of drift that shows up as perplexity, not as a test
// failure.
//
// It costs a transpose: the search wants element i of all eight sub-blocks in
// one register, and the data arrives sub-block-major. That is paid once per
// super-block and amortized over 21 steps.
//
// Only the scale and min come back. The codes `make_qkx2_quants` fills are
// scratch -- `fit_qk_super_block` re-derives them from the *stored* 6-bit
// scales afterwards -- so this does not track them at all.
#if defined(__AVX2__)

// Eight rows in, their transpose out.
inline void transpose8x8(__m256& r0, __m256& r1, __m256& r2, __m256& r3,
                         __m256& r4, __m256& r5, __m256& r6, __m256& r7) {
    const __m256 t0 = _mm256_unpacklo_ps(r0, r1);
    const __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    const __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    const __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    const __m256 t4 = _mm256_unpacklo_ps(r4, r5);
    const __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    const __m256 t6 = _mm256_unpacklo_ps(r6, r7);
    const __m256 t7 = _mm256_unpackhi_ps(r6, r7);
    const __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    r0 = _mm256_permute2f128_ps(s0, s4, 0x20);
    r1 = _mm256_permute2f128_ps(s1, s5, 0x20);
    r2 = _mm256_permute2f128_ps(s2, s6, 0x20);
    r3 = _mm256_permute2f128_ps(s3, s7, 0x20);
    r4 = _mm256_permute2f128_ps(s0, s4, 0x31);
    r5 = _mm256_permute2f128_ps(s1, s5, 0x31);
    r6 = _mm256_permute2f128_ps(s2, s6, 0x31);
    r7 = _mm256_permute2f128_ps(s3, s7, 0x31);
}

// `nearest_int`, lane-wise. The same add-and-mask trick, so it rounds
// identically -- including the ties, which is the point.
inline __m256 nearest_int_ps(__m256 value) {
    const __m256 shifted = _mm256_add_ps(value, _mm256_set1_ps(12582912.0f));
    __m256i bits = _mm256_castps_si256(shifted);
    bits = _mm256_sub_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x007fffff)),
                            _mm256_set1_epi32(0x00400000));
    return _mm256_cvtepi32_ps(bits);
}

// `a*b + c` at two roundings, which is what the scalar does. An FMA here would
// round once and produce a different number, so this must not be contracted --
// and GCC's default `-ffp-contract=fast` will happily contract a mul+add pair
// even when both are written as intrinsics. The build turns contraction off for
// this header's users; `hf_quantize_tiling_contract` is what proves it, by
// comparing these bytes against the scalar path.
inline __m256 mul_add(__m256 a, __m256 b, __m256 c) {
    const __m256 product = _mm256_mul_ps(a, b);
    return _mm256_add_ps(product, c);
}

// X and W are transposed: X[i] holds element i of all eight sub-blocks.
inline void make_qkx2_quants_x8(const __m256* X, const __m256* W, int nmax,
                                float rmin, float rdelta, int nstep,
                                __m256& out_scale, __m256& out_min) {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 nmax_ps = _mm256_set1_ps(static_cast<float>(nmax));

    __m256 minimum = X[0], maximum = X[0];
    __m256 sum_w = W[0], sum_x = _mm256_mul_ps(W[0], X[0]);
    for (int i = 1; i < 32; ++i) {
        minimum = _mm256_min_ps(minimum, X[i]);
        maximum = _mm256_max_ps(maximum, X[i]);
        sum_w = _mm256_add_ps(sum_w, W[i]);
        sum_x = mul_add(W[i], X[i], sum_x);
    }
    minimum = _mm256_min_ps(minimum, zero);

    // A sub-block whose values are all equal has no scale to find. Its lane
    // still runs the search below -- with an infinite iscale, producing
    // nonsense that never escapes -- and is overwritten at the end.
    const __m256 degenerate = _mm256_cmp_ps(maximum, minimum, _CMP_EQ_OQ);

    const __m256 iscale = _mm256_div_ps(nmax_ps, _mm256_sub_ps(maximum, minimum));
    // 1/iscale, not span/nmax. They differ in the last bit, and the last bit is
    // enough to hand a later step the win.
    __m256 scale = _mm256_div_ps(_mm256_set1_ps(1.0f), iscale);

    __m256 best_error = zero;
    for (int i = 0; i < 32; ++i) {
        const __m256 offset = _mm256_sub_ps(X[i], minimum);
        __m256 code = nearest_int_ps(_mm256_mul_ps(iscale, offset));
        code = _mm256_min_ps(_mm256_max_ps(code, zero), nmax_ps);
        const __m256 diff =
            _mm256_sub_ps(mul_add(scale, code, minimum), X[i]);
        // (w*diff)*diff, in that order: w*(diff*diff) rounds differently.
        best_error = mul_add(_mm256_mul_ps(W[i], diff), diff, best_error);
    }

    for (int step = 0; step <= nstep; ++step) {
        // Against the *current* minimum, which a winning step below moves. The
        // span is deliberately not hoisted: the search walks, and hoisting it
        // quietly turns this into a different (worse) fit.
        const __m256 trial_iscale = _mm256_div_ps(
            _mm256_set1_ps(rmin + rdelta * static_cast<float>(step) +
                           static_cast<float>(nmax)),
            _mm256_sub_ps(maximum, minimum));

        __m256 codes[32];
        __m256 sum_l = zero, sum_l2 = zero, sum_xl = zero;
        for (int i = 0; i < 32; ++i) {
            const __m256 offset = _mm256_sub_ps(X[i], minimum);
            __m256 code = nearest_int_ps(_mm256_mul_ps(trial_iscale, offset));
            code = _mm256_min_ps(_mm256_max_ps(code, zero), nmax_ps);
            codes[i] = code;
            const __m256 weighted = _mm256_mul_ps(W[i], code);
            sum_l = _mm256_add_ps(sum_l, weighted);
            sum_l2 = mul_add(weighted, code, sum_l2);
            sum_xl = mul_add(weighted, X[i], sum_xl);
        }

        const __m256 determinant =
            _mm256_sub_ps(_mm256_mul_ps(sum_w, sum_l2), _mm256_mul_ps(sum_l, sum_l));
        // Where the determinant is not positive the scalar skips the step
        // outright, so this lane must not be allowed to win below.
        const __m256 usable = _mm256_cmp_ps(determinant, zero, _CMP_GT_OQ);

        __m256 this_scale = _mm256_div_ps(
            _mm256_sub_ps(_mm256_mul_ps(sum_w, sum_xl), _mm256_mul_ps(sum_x, sum_l)),
            determinant);
        __m256 this_min = _mm256_div_ps(
            _mm256_sub_ps(_mm256_mul_ps(sum_l2, sum_x), _mm256_mul_ps(sum_l, sum_xl)),
            determinant);

        // A positive offset is not representable, so the fit is redone with the
        // offset pinned at zero.
        const __m256 positive = _mm256_cmp_ps(this_min, zero, _CMP_GT_OQ);
        const __m256 pinned = _mm256_blendv_ps(
            zero, _mm256_div_ps(sum_xl, sum_l2),
            _mm256_cmp_ps(sum_l2, zero, _CMP_GT_OQ));
        this_scale = _mm256_blendv_ps(this_scale, pinned, positive);
        this_min = _mm256_blendv_ps(this_min, zero, positive);

        __m256 error = zero;
        for (int i = 0; i < 32; ++i) {
            const __m256 diff =
                _mm256_sub_ps(mul_add(this_scale, codes[i], this_min), X[i]);
            error = mul_add(_mm256_mul_ps(W[i], diff), diff, error);
        }

        const __m256 better = _mm256_and_ps(
            usable, _mm256_cmp_ps(error, best_error, _CMP_LT_OQ));
        best_error = _mm256_blendv_ps(best_error, error, better);
        scale = _mm256_blendv_ps(scale, this_scale, better);
        minimum = _mm256_blendv_ps(minimum, this_min, better);
    }

    out_scale = _mm256_blendv_ps(scale, zero, degenerate);
    // Negate by flipping the sign bit rather than subtracting from zero, so a
    // zero minimum comes back as -0.0f exactly as `-minimum` would.
    out_min = _mm256_xor_ps(minimum, _mm256_castsi256_ps(_mm256_set1_epi32(
                                         static_cast<int>(0x80000000u))));
}

// Fills `scales` and `mins` for the eight sub-blocks of one super-block.
inline void fit_sub_block_scales_avx2(const float* x, int nmax, float rmin,
                                      int nstep, float* scales, float* mins) {
    __m256 X[32], W[32];
    for (int chunk = 0; chunk < 4; ++chunk) {
        __m256 rows[8];
        for (int sub = 0; sub < 8; ++sub)
            rows[sub] = _mm256_loadu_ps(x + 32 * sub + 8 * chunk);
        transpose8x8(rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6],
                     rows[7]);
        for (int i = 0; i < 8; ++i) X[8 * chunk + i] = rows[i];
    }

    // weights[i] = sqrt(mean of squares) + |x[i]|, per sub-block. The sum is
    // still sequential in i within each lane.
    __m256 sum_squares = _mm256_setzero_ps();
    for (int i = 0; i < 32; ++i)
        sum_squares = mul_add(X[i], X[i], sum_squares);
    const __m256 average =
        _mm256_sqrt_ps(_mm256_div_ps(sum_squares, _mm256_set1_ps(32.0f)));
    const __m256 absolute =
        _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (int i = 0; i < 32; ++i)
        W[i] = _mm256_add_ps(average, _mm256_and_ps(X[i], absolute));

    __m256 scale_v, min_v;
    make_qkx2_quants_x8(X, W, nmax, rmin, 0.1f, nstep, scale_v, min_v);
    _mm256_storeu_ps(scales, scale_v);
    _mm256_storeu_ps(mins, min_v);
}

#endif  // __AVX2__

// Shared front half of Q4_K and Q5_K: fit 8 sub-blocks of 32, quantize their
// scales and mins to 6 bits, then re-derive the codes against the scales that
// will actually be stored -- so decode reproduces exactly these codes.
//
// `rmin`/`nstep` differ between the two formats in the reference quantizer --
// Q4_K searches (-1.0, 20 steps), Q5_K searches (-0.5, 15) -- and using one
// pair for both silently costs accuracy rather than failing.
// The scalar fit, which is also the reference the AVX2 one is checked against.
inline void fit_sub_block_scales_scalar(const float* x, int nmax, float rmin,
                                        int nstep, float* scales, float* mins) {
    float weights[32];
    std::uint8_t codes[32];
    for (int sub = 0; sub < 8; ++sub) {
        const float* block = x + 32 * sub;
        float sum_squares = 0.0f;
        for (int i = 0; i < 32; ++i) sum_squares += block[i] * block[i];
        const float average = std::sqrt(sum_squares / 32.0f);
        for (int i = 0; i < 32; ++i) weights[i] = average + std::fabs(block[i]);
        mins[sub] = 0.0f;
        scales[sub] = make_qkx2_quants(32, nmax, block, weights, codes, &mins[sub],
                                       rmin, 0.1f, nstep);
    }
}

// Runtime dispatch, the same shape the rest of the CPU backend uses: the
// library is built for baseline x86-64, and each ISA lives in its own
// translation unit compiled with its own flags. Null until
// qwen_kquant_pack.cpp installs the AVX2 fit, and left null on a machine
// without AVX2 -- so this file must stay compilable at baseline.
using SubBlockScaleFit = void (*)(const float*, int, float, int, float*, float*);
inline SubBlockScaleFit fit_sub_block_scales_hook = nullptr;

inline void fit_qk_super_block(const float* x, int nmax, float rmin, int nstep,
                               std::uint8_t* scales_out,
                               std::uint16_t* d_bits, std::uint16_t* dmin_bits,
                               std::uint8_t* codes_out) {
    float scales[8], mins[8];
    float max_scale = 0.0f, max_min = 0.0f;
    if (fit_sub_block_scales_hook)
        fit_sub_block_scales_hook(x, nmax, rmin, nstep, scales, mins);
    else
        fit_sub_block_scales_scalar(x, nmax, rmin, nstep, scales, mins);
    for (int sub = 0; sub < 8; ++sub) {
        max_scale = std::max(max_scale, scales[sub]);
        max_min = std::max(max_min, mins[sub]);
    }

    const float inverse_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
    const float inverse_min = max_min > 0.0f ? 63.0f / max_min : 0.0f;
    std::memset(scales_out, 0, 12);
    for (int sub = 0; sub < 8; ++sub) {
        const auto ls = static_cast<std::uint8_t>(
            std::min(63, nearest_int(inverse_scale * scales[sub])));
        const auto lm = static_cast<std::uint8_t>(
            std::min(63, nearest_int(inverse_min * mins[sub])));
        pack_scale_min_k4(sub, scales_out, ls, lm);
    }
    *d_bits = qwen_half_bits(max_scale / 63.0f);
    *dmin_bits = qwen_half_bits(max_min / 63.0f);

    const float d = qwen_half_value(*d_bits), dmin = qwen_half_value(*dmin_bits);
    for (int sub = 0; sub < 8; ++sub) {
        // Read the scale back through the same unpacking decode uses, so a
        // packing bug shows up here rather than as silent drift.
        int scale = 0, minimum = 0;
        if (sub < 4) {
            scale = scales_out[sub] & 63;
            minimum = scales_out[sub + 4] & 63;
        } else {
            scale = (scales_out[sub + 4] & 15) | ((scales_out[sub - 4] >> 6) << 4);
            minimum = (scales_out[sub + 4] >> 4) | ((scales_out[sub] >> 6) << 4);
        }
        const float step = d * static_cast<float>(scale);
        const float offset = dmin * static_cast<float>(minimum);
        for (int i = 0; i < 32; ++i) {
            int l = step != 0.0f ? nearest_int((x[32 * sub + i] + offset) / step) : 0;
            codes_out[32 * sub + i] =
                static_cast<std::uint8_t>(std::max(0, std::min(nmax, l)));
        }
    }
}

// Q4_K: 144 bytes per 256 values. d, dmin, 12 scale bytes, 128 nibble bytes.
inline void qwen_pack_q4_k(const float* values, std::uint64_t count, std::uint8_t* out) {
    for (std::uint64_t block = 0; block * kSuperBlock < count; ++block) {
        const float* x = values + block * kSuperBlock;
        std::uint8_t* base = out + block * 144;
        std::uint16_t d_bits = 0, dmin_bits = 0;
        std::uint8_t codes[kSuperBlock];
        fit_qk_super_block(x, 15, -1.0f, 20, base + 4, &d_bits, &dmin_bits, codes);
        std::memcpy(base, &d_bits, 2);
        std::memcpy(base + 2, &dmin_bits, 2);
        // Two 32-code halves of each 64-element group share a byte: low nibble
        // is the first half, high nibble the second.
        for (int group = 0; group < 4; ++group)
            for (int i = 0; i < 32; ++i)
                base[16 + group * 32 + i] = static_cast<std::uint8_t>(
                    codes[group * 64 + i] | (codes[group * 64 + 32 + i] << 4));
    }
}

// Q5_K: 176 bytes per 256 values. Same as Q4_K plus 32 bytes holding each
// code's fifth bit.
inline void qwen_pack_q5_k(const float* values, std::uint64_t count, std::uint8_t* out) {
    for (std::uint64_t block = 0; block * kSuperBlock < count; ++block) {
        const float* x = values + block * kSuperBlock;
        std::uint8_t* base = out + block * 176;
        std::uint16_t d_bits = 0, dmin_bits = 0;
        std::uint8_t codes[kSuperBlock];
        fit_qk_super_block(x, 31, -0.5f, 15, base + 4, &d_bits, &dmin_bits, codes);
        std::memcpy(base, &d_bits, 2);
        std::memcpy(base + 2, &dmin_bits, 2);
        std::memset(base + 16, 0, 32);
        for (int group = 0; group < 4; ++group) {
            for (int i = 0; i < 32; ++i) {
                int low = codes[group * 64 + i];
                int high = codes[group * 64 + 32 + i];
                // The high bit of each 5-bit code moves to the qh plane; the
                // two halves of a group take adjacent bit positions.
                if (low > 15) { low -= 16; base[16 + i] |= static_cast<std::uint8_t>(1u << (2 * group)); }
                if (high > 15) { high -= 16; base[16 + i] |= static_cast<std::uint8_t>(1u << (2 * group + 1)); }
                base[48 + group * 32 + i] = static_cast<std::uint8_t>(low | (high << 4));
            }
        }
    }
}

// Q6_K: 210 bytes per 256 values. 128 low-nibble bytes, 64 high-bit-pair bytes,
// 16 signed sub-block scales, then d. Sub-blocks are 16 values here, not 32.
inline void qwen_pack_q6_k(const float* values, std::uint64_t count, std::uint8_t* out) {
    for (std::uint64_t block = 0; block * kSuperBlock < count; ++block) {
        const float* x = values + block * kSuperBlock;
        std::uint8_t* base = out + block * 210;
        std::int8_t codes[kSuperBlock];
        float scales[16];
        float max_scale = 0.0f, max_abs_scale = 0.0f;
        for (int sub = 0; sub < 16; ++sub) {
            scales[sub] = make_qx_quants(16, 32, x + 16 * sub, codes + 16 * sub);
            const float a = std::fabs(scales[sub]);
            if (a > max_abs_scale) { max_abs_scale = a; max_scale = scales[sub]; }
        }
        auto* scale_bytes = reinterpret_cast<std::int8_t*>(base + 192);
        if (max_abs_scale < 1e-30f) {
            std::memset(base, 0, 210);
            continue;
        }
        // Sub-block scales are themselves quantized, signed, against -128.
        const float inverse = -128.0f / max_scale;
        const std::uint16_t d_bits = qwen_half_bits(1.0f / inverse);
        std::memcpy(base + 208, &d_bits, 2);
        for (int sub = 0; sub < 16; ++sub)
            scale_bytes[sub] = static_cast<std::int8_t>(
                std::min(127, nearest_int(inverse * scales[sub])));

        const float d = qwen_half_value(d_bits);
        for (int sub = 0; sub < 16; ++sub) {
            const float step = d * static_cast<float>(scale_bytes[sub]);
            if (step == 0.0f) {
                for (int i = 0; i < 16; ++i) codes[16 * sub + i] = 32;
                continue;
            }
            for (int i = 0; i < 16; ++i) {
                const int l = std::max(-32, std::min(31, nearest_int(x[16 * sub + i] / step)));
                codes[16 * sub + i] = static_cast<std::int8_t>(l + 32);
            }
        }

        // Each 128-element half packs four 32-element lanes: lanes 0/2 share a
        // byte in the first 32, lanes 1/3 share the second 32, and all four
        // contribute a bit pair to qh.
        for (int half = 0; half < 2; ++half) {
            std::uint8_t* ql = base + half * 64;
            std::uint8_t* qh = base + 128 + half * 32;
            const std::int8_t* source = codes + half * 128;
            for (int i = 0; i < 32; ++i) {
                const int q1 = source[i] & 0xF, q2 = source[i + 32] & 0xF;
                const int q3 = source[i + 64] & 0xF, q4 = source[i + 96] & 0xF;
                ql[i] = static_cast<std::uint8_t>(q1 | (q3 << 4));
                ql[i + 32] = static_cast<std::uint8_t>(q2 | (q4 << 4));
                qh[i] = static_cast<std::uint8_t>(
                    (source[i] >> 4) | ((source[i + 32] >> 4) << 2) |
                    ((source[i + 64] >> 4) << 4) | ((source[i + 96] >> 4) << 6));
            }
        }
    }
}

}  // namespace qwen_kpack
