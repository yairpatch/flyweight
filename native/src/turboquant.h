#pragma once

// TurboQuant KV-cache codec: scalar reference implementation.
//
// TurboQuant (arXiv:2504.19874, ICLR 2026) quantizes a KV cache by first
// applying a random rotation and then a per-coordinate optimal scalar
// quantizer. The rotation is the whole trick: it spreads the energy of an
// arbitrary vector evenly across its coordinates, so the rotated coordinates
// look like independent Gaussians and a single fixed codebook is near-optimal
// for every head of every layer. Without it, the outlier channels that KV
// caches are famous for would dominate the error.
//
// This header holds the scalar definition so the runtime dispatch, any future
// SIMD or CUDA kernel and the contract test all encode from one source: the
// rotation conventions are easy to get subtly wrong (the llama.cpp ports lost
// a factor of PPL 6.2 -> 23.5 to a silently transposed rotation matrix) and
// two copies would drift.
//
// Deviations from the paper, both following the independent llama.cpp ports:
//
//   * The paper's Algorithm 2 (a 1-bit QJL transform on the residual, meant to
//     debias inner products) is not implemented. Every independent port found
//     plain MSE quantization better in practice for both K and V, because the
//     variance QJL adds gets amplified by the softmax.
//   * The rotation is a fixed per-stream sign flip followed by a Walsh-Hadamard
//     transform, not a freshly drawn random rotation per vector. A rotation
//     shared by every vector in a cache is what makes the two identities in
//     "Using this at attention time" below hold, and a Walsh-Hadamard transform
//     costs O(d log d) instead of the O(d^2) of a dense rotation.
//
// Using this at attention time
// ----------------------------
// Because the rotation R is a fixed orthogonal transform shared by every vector
// in a stream, it never has to be undone per cache entry:
//
//   scores:  <q, k> == <Rq, Rk>, so rotate the query once per step and dot it
//            straight against the stored (already rotated) keys.
//   output:  sum_i p_i v_i == R^-1 (sum_i p_i (R v_i)), so accumulate the
//            weighted sum in the rotated domain and inverse-rotate once per
//            head per token, not once per cache entry.
//
// turbo_dot_rotated and turbo_inverse_rotate are those two primitives.

#include <cmath>
#include <cstdint>
#include <cstring>

// Values per quantization block. The rotation runs over the full head
// dimension, but scales are stored per 32 values so a block matches the
// granularity flash-attention kernels like to load.
constexpr int kTurboBlock = 32;

enum class TurboType { Turbo2, Turbo3, Turbo4 };

constexpr int turbo_bits(TurboType type) {
    return type == TurboType::Turbo2 ? 2 : (type == TurboType::Turbo3 ? 3 : 4);
}

// 2 bytes of fp16 scale, then kTurboBlock packed indices: 10, 14 or 18 bytes,
// i.e. 2.5, 3.5 or 4.5 bits per value.
constexpr int turbo_block_bytes(TurboType type) {
    return 2 + turbo_bits(type) * kTurboBlock / 8;
}

// Lloyd-Max optimal scalar quantizers for the unit Gaussian, which is what the
// rotated coordinates converge to. These are fixed points of the Lloyd
// iteration (centroid = E[x | cell]); the contract test re-derives them and
// checks the resulting distortion against the paper's published figures of
// 0.034 for 3-bit and 0.009 for 4-bit rather than against these constants.
constexpr float kTurboCodebook2[4] = {
    -1.51041761f, -0.45278003f, 0.45278003f, 1.51041761f,
};
constexpr float kTurboCodebook3[8] = {
    -2.15194570f, -1.34390928f, -0.75600528f, -0.24509418f,
    0.24509418f, 0.75600528f, 1.34390928f, 2.15194570f,
};
constexpr float kTurboCodebook4[16] = {
    -2.73258956f, -2.06901721f, -1.61804637f, -1.25623118f,
    -0.94234045f, -0.65675911f, -0.38804829f, -0.12839503f,
    0.12839503f, 0.38804829f, 0.65675911f, 0.94234045f,
    1.25623118f, 1.61804637f, 2.06901721f, 2.73258956f,
};

inline const float* turbo_codebook(TurboType type) {
    switch (type) {
        case TurboType::Turbo2: return kTurboCodebook2;
        case TurboType::Turbo3: return kTurboCodebook3;
        default: return kTurboCodebook4;
    }
}

inline float turbo_half_value(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000u) << 16;
    std::uint32_t exponent = (bits >> 10) & 0x1fu, fraction = bits & 0x3ffu, result = 0;
    if (exponent == 0) {
        if (!fraction) result = sign;
        else {
            exponent = 1;
            while ((fraction & 0x400u) == 0) { fraction <<= 1; --exponent; }
            result = sign | ((exponent + 112) << 23) | ((fraction & 0x3ffu) << 13);
        }
    } else if (exponent == 31) result = sign | 0x7f800000u | (fraction << 13);
    else result = sign | ((exponent + 112) << 23) | (fraction << 13);
    float value;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

// Round-to-nearest-even float -> half. Scales are small positive numbers here,
// so the subnormal and overflow paths are cold, but they are still correct.
inline std::uint16_t turbo_half_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xffu) - 127;
    std::uint32_t fraction = bits & 0x7fffffu;
    if (exponent == 128) {  // inf / NaN
        return static_cast<std::uint16_t>(sign | 0x7c00u | (fraction ? 0x200u : 0u));
    }
    if (exponent > 15) return static_cast<std::uint16_t>(sign | 0x7bffu);
    if (exponent < -24) return static_cast<std::uint16_t>(sign);
    if (exponent < -14) {  // subnormal half
        fraction |= 0x800000u;
        const int shift = -exponent - 14;
        const std::uint32_t round = 1u << (shift + 12);
        std::uint32_t result = (fraction + round - 1 + ((fraction >> (shift + 13)) & 1)) >> (shift + 13);
        return static_cast<std::uint16_t>(sign | result);
    }
    const std::uint32_t round = 0x1000u + ((fraction >> 13) & 1u);
    fraction += round - 1;
    std::uint32_t carry = fraction >> 23;
    return static_cast<std::uint16_t>(
        sign | ((exponent + 15 + static_cast<std::int32_t>(carry)) << 10)
        | ((fraction >> 13) & 0x3ffu));
}

// Deterministic +-1 sign per coordinate, so encoder and decoder agree without
// storing the rotation. Golden-ratio multiply plus an xorshift finalizer; the
// only property that matters is that the signs look independent across both
// the coordinate and the stream.
inline float turbo_sign(int index, std::uint32_t stream) {
    std::uint32_t hash = static_cast<std::uint32_t>(index) * 0x9e3779b9u
        + stream * 0x85ebca6bu + 0x165667b1u;
    hash ^= hash >> 15;
    hash *= 0x2545f491u;
    hash ^= hash >> 13;
    return (hash & 1u) ? -1.0f : 1.0f;
}

// In-place orthonormal fast Walsh-Hadamard transform. O(d log d) butterfly;
// `dimension` must be a power of two. Orthonormal means the transform is its
// own inverse and preserves norms, which is what lets the attention-time
// identities above hold exactly.
inline void turbo_fwht(float* values, int dimension) {
    for (int span = 1; span < dimension; span <<= 1) {
        for (int start = 0; start < dimension; start += span << 1) {
            for (int offset = start; offset < start + span; ++offset) {
                const float low = values[offset], high = values[offset + span];
                values[offset] = low + high;
                values[offset + span] = low - high;
            }
        }
    }
    const float normalizer = 1.0f / std::sqrt(static_cast<float>(dimension));
    for (int i = 0; i < dimension; ++i) values[i] *= normalizer;
}

inline bool turbo_dimension_supported(int dimension) {
    return dimension >= kTurboBlock && (dimension & (dimension - 1)) == 0;
}

// Forward rotation R = H * S: flip signs, then Walsh-Hadamard.
inline void turbo_rotate(const float* input, float* output, int dimension, std::uint32_t stream) {
    for (int i = 0; i < dimension; ++i) output[i] = input[i] * turbo_sign(i, stream);
    turbo_fwht(output, dimension);
}

// Inverse rotation R^-1 = R^T = S * H: Walsh-Hadamard, then flip signs.
inline void turbo_inverse_rotate(const float* input, float* output, int dimension, std::uint32_t stream) {
    for (int i = 0; i < dimension; ++i) output[i] = input[i];
    turbo_fwht(output, dimension);
    for (int i = 0; i < dimension; ++i) output[i] *= turbo_sign(i, stream);
}

// A packed index never spans more than two bytes because bits <= 4.
inline void turbo_pack_index(std::uint8_t* packed, int slot, int bits, std::uint32_t value) {
    const int bit = slot * bits, byte = bit >> 3, shift = bit & 7;
    packed[byte] |= static_cast<std::uint8_t>(value << shift);
    if (shift + bits > 8) packed[byte + 1] |= static_cast<std::uint8_t>(value >> (8 - shift));
}

inline std::uint32_t turbo_unpack_index(const std::uint8_t* packed, int slot, int bits) {
    const int bit = slot * bits, byte = bit >> 3, shift = bit & 7;
    std::uint32_t value = static_cast<std::uint32_t>(packed[byte]) >> shift;
    if (shift + bits > 8) value |= static_cast<std::uint32_t>(packed[byte + 1]) << (8 - shift);
    return value & ((1u << bits) - 1u);
}

// Encode one already-rotated block of kTurboBlock values.
//
// The scale is chosen in two passes: the block RMS picks the codewords, then
// the scale is refit by least squares against those codewords. The refit is
// the clean form of the "store original_norm / ||reconstruction||" correction
// the llama.cpp ports found -- it costs nothing at decode time and removes the
// systematic shrinkage that quantizing to a fixed codebook otherwise induces.
inline void turbo_encode_block(const float* rotated, TurboType type, std::uint8_t* packed) {
    const int bits = turbo_bits(type), levels = 1 << bits;
    const float* codebook = turbo_codebook(type);

    float energy = 0.0f;
    for (int i = 0; i < kTurboBlock; ++i) energy += rotated[i] * rotated[i];
    const float rms = std::sqrt(energy / static_cast<float>(kTurboBlock));

    std::uint32_t indices[kTurboBlock];
    if (rms <= 0.0f) {
        // All-zero block: point every slot at the codeword nearest zero and
        // store a zero scale, so decode reproduces zeros exactly.
        for (int i = 0; i < kTurboBlock; ++i) indices[i] = static_cast<std::uint32_t>(levels / 2 - 1);
        std::memset(packed, 0, static_cast<std::size_t>(turbo_block_bytes(type)));
        for (int i = 0; i < kTurboBlock; ++i)
            turbo_pack_index(packed + 2, i, bits, indices[i]);
        return;
    }

    float numerator = 0.0f, denominator = 0.0f;
    for (int i = 0; i < kTurboBlock; ++i) {
        const float normalized = rotated[i] / rms;
        int best = 0;
        float best_distance = std::fabs(normalized - codebook[0]);
        for (int level = 1; level < levels; ++level) {
            const float distance = std::fabs(normalized - codebook[level]);
            if (distance < best_distance) { best_distance = distance; best = level; }
        }
        indices[i] = static_cast<std::uint32_t>(best);
        numerator += rotated[i] * codebook[best];
        denominator += codebook[best] * codebook[best];
    }

    const float scale = denominator > 0.0f ? numerator / denominator : rms;
    const std::uint16_t scale_bits = turbo_half_bits(scale);
    std::memset(packed, 0, static_cast<std::size_t>(turbo_block_bytes(type)));
    std::memcpy(packed, &scale_bits, sizeof(scale_bits));
    for (int i = 0; i < kTurboBlock; ++i) turbo_pack_index(packed + 2, i, bits, indices[i]);
}

inline void turbo_decode_block(const std::uint8_t* packed, TurboType type, float* rotated) {
    const int bits = turbo_bits(type);
    const float* codebook = turbo_codebook(type);
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, packed, sizeof(scale_bits));
    const float scale = turbo_half_value(scale_bits);
    for (int i = 0; i < kTurboBlock; ++i)
        rotated[i] = scale * codebook[turbo_unpack_index(packed + 2, i, bits)];
}

// Dot one stored block against the matching slice of an already-rotated query.
// This is the attention-time inner loop: no inverse rotation, and the scale
// factors out of the accumulation.
inline float turbo_dot_block(const std::uint8_t* packed, const float* rotated_query, TurboType type) {
    const int bits = turbo_bits(type);
    const float* codebook = turbo_codebook(type);
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, packed, sizeof(scale_bits));
    float sum = 0.0f;
    for (int i = 0; i < kTurboBlock; ++i)
        sum += rotated_query[i] * codebook[turbo_unpack_index(packed + 2, i, bits)];
    return sum * turbo_half_value(scale_bits);
}

// Rotate and encode a whole head-dimension vector. `packed` needs
// dimension / kTurboBlock * turbo_block_bytes(type) bytes.
inline void turbo_encode_vector(
    const float* values, int dimension, TurboType type, std::uint32_t stream, std::uint8_t* packed
) {
    float rotated[1024];
    turbo_rotate(values, rotated, dimension, stream);
    const int stride = turbo_block_bytes(type);
    for (int block = 0; block < dimension / kTurboBlock; ++block)
        turbo_encode_block(rotated + block * kTurboBlock, type, packed + block * stride);
}

// Decode back to the rotated domain -- what attention wants for both keys and
// values. Callers that need the original coordinates apply turbo_inverse_rotate.
inline void turbo_decode_vector(
    const std::uint8_t* packed, int dimension, TurboType type, float* rotated
) {
    const int stride = turbo_block_bytes(type);
    for (int block = 0; block < dimension / kTurboBlock; ++block)
        turbo_decode_block(packed + block * stride, type, rotated + block * kTurboBlock);
}

// <q, k> for a stored key and an already-rotated query, exact up to the
// quantization of k.
inline float turbo_dot_rotated(
    const std::uint8_t* packed, const float* rotated_query, int dimension, TurboType type
) {
    const int stride = turbo_block_bytes(type);
    float sum = 0.0f;
    for (int block = 0; block < dimension / kTurboBlock; ++block)
        sum += turbo_dot_block(
            packed + block * stride, rotated_query + block * kTurboBlock, type);
    return sum;
}
