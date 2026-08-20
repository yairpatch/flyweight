#pragma once

// Scalar element decoders for the low-bit K-quant super-block formats.
//
// These live in a header so the runtime dispatch, the SIMD kernels and the
// contract tests all decode from one definition: the bit layouts are fiddly
// enough that two copies would drift.

#include <cstdint>
#include <cstring>

#include "qwen_iq_tables.h"

inline float qwen_half_value(std::uint16_t bits) {
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

// Narrowing counterpart, used for every block scale this file's formats carry.
// Round-half-to-even keeps the scale error under the codes' own quantization
// error; a mantissa carry lands in the exponent field on its own because the
// fields are packed adjacently.
inline std::uint16_t qwen_half_bits(float value) {
    std::uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));
    const std::uint32_t sign=(bits>>16)&0x8000u;
    const std::int32_t exponent=
        static_cast<std::int32_t>((bits>>23)&0xffu)-127+15;
    const std::uint32_t fraction=bits&0x7fffffu;
    if(((bits>>23)&0xffu)==0xffu)
        return static_cast<std::uint16_t>(sign|0x7c00u|(fraction?0x200u:0u));
    if(exponent>=31)return static_cast<std::uint16_t>(sign|0x7c00u);
    // Subnormals are reachable and must be encoded, not flushed. A Q8_0 block
    // scale is absmax/127 of real weights and never lands here, which is why
    // this used to flush -- but a K-quant super-block scale is a scale *of
    // scales* (max_scale/63 for Q4_K/Q5_K, max_scale/-128 for Q6_K), which
    // routinely falls under the f16 normal minimum of 6.104e-5. Flushing it
    // zeroed the whole super-block.
    if(exponent<=0){
        // Below 2^-24 there is no subnormal left to round to.
        if(exponent<-10)return static_cast<std::uint16_t>(sign);
        const std::uint32_t mantissa=fraction|0x800000u;
        const int shift=14-exponent;
        std::uint32_t subnormal=mantissa>>shift;
        const std::uint32_t dropped=mantissa&((1u<<shift)-1u);
        const std::uint32_t halfway=1u<<(shift-1);
        if(dropped>halfway||(dropped==halfway&&(subnormal&1u)))++subnormal;
        // A carry out of the mantissa lands in the exponent field on its own,
        // producing the smallest normal -- which is the correct next value.
        return static_cast<std::uint16_t>(sign|subnormal);
    }
    std::uint32_t packed=(static_cast<std::uint32_t>(exponent)<<10)|(fraction>>13);
    const std::uint32_t remainder=fraction&0x1fffu;
    if(remainder>0x1000u||(remainder==0x1000u&&(packed&1u)))++packed;
    return static_cast<std::uint16_t>(sign|packed);
}


// Block geometry for the formats decoded below. These sit here rather than in
// the runtime because the decoders are here; a consumer that can see one must
// be able to see the other.
constexpr std::uint32_t kBlockElements = 256;  // super-block size for the K-quants
constexpr std::uint32_t kQ8BlockSize = 34;     // Q8_0: f16 scale + 32 codes
constexpr std::uint32_t kQ4KBlockSize = 144;
constexpr std::uint32_t kQ5KBlockSize = 176;
constexpr std::uint32_t kQ6KBlockSize = 210;

// Q8_0, Q4_K, Q5_K and Q6_K element decoders. These lived in v2_runtime.cpp
// while their Q2_K/Q3_K/IQ siblings lived here, which meant the four formats
// the HF quantizer actually emits were the only ones an out-of-runtime consumer
// could not decode. Same definitions, moved.
inline float qwen_q8_value(const std::uint8_t*packed,std::uint64_t absolute){const auto block=absolute/32,within=absolute&31;std::uint16_t scale_bits=0;std::memcpy(&scale_bits,packed+block*kQ8BlockSize,2);std::int8_t value=0;std::memcpy(&value,packed+block*kQ8BlockSize+2+within,1);return qwen_half_value(scale_bits)*value;}
inline float qwen_q4k_value(const std::uint8_t*packed,std::uint64_t absolute){const auto block=absolute/kBlockElements;const int within=static_cast<int>(absolute&(kBlockElements-1));const auto*base=packed+block*kQ4KBlockSize;std::uint16_t d_bits=0,dmin_bits=0;std::memcpy(&d_bits,base,2);std::memcpy(&dmin_bits,base+2,2);const auto*scales=base+4;const int group=within/64,offset=within&63,sub=offset/32,qindex=group*32+(offset&31);const int quant=(offset<32)?(base[16+qindex]&15):(base[16+qindex]>>4);const int index=group*2+sub;int scale=0,minimum=0;if(index<4){scale=scales[index]&63;minimum=scales[index+4]&63;}else{scale=(scales[index+4]&15)|((scales[index-4]>>6)<<4);minimum=(scales[index+4]>>4)|((scales[index]>>6)<<4);}return qwen_half_value(d_bits)*scale*quant-qwen_half_value(dmin_bits)*minimum;}
inline float qwen_q5_value(const std::uint8_t*packed,std::uint64_t absolute){const auto block=absolute/kBlockElements;const int within=static_cast<int>(absolute&(kBlockElements-1));const auto*base=packed+block*kQ5KBlockSize;std::uint16_t d_bits=0,dmin_bits=0;std::memcpy(&d_bits,base,2);std::memcpy(&dmin_bits,base+2,2);const auto*scales=base+4;const int group=within/64,offset=within&63,sub=offset/32,qindex=group*32+(offset&31);const int bit=(base[16+(offset&31)]>>(2*group+sub))&1;const int quant=((offset<32)?(base[48+qindex]&15):(base[48+qindex]>>4))+16*bit;const int index=group*2+sub;int scale=0,minimum=0;if(index<4){scale=scales[index]&63;minimum=scales[index+4]&63;}else{scale=(scales[index+4]&15)|((scales[index-4]>>6)<<4);minimum=(scales[index+4]>>4)|((scales[index]>>6)<<4);}return qwen_half_value(d_bits)*scale*quant-qwen_half_value(dmin_bits)*minimum;}
inline float qwen_q6_value(const std::uint8_t*packed,std::uint64_t absolute){const auto block=absolute/kBlockElements;const int within=static_cast<int>(absolute&(kBlockElements-1));const auto*base=packed+block*kQ6KBlockSize;const auto*ql=base;const auto*qh=base+128;const auto*scales=reinterpret_cast<const std::int8_t*>(base+192);std::uint16_t d_bits=0;std::memcpy(&d_bits,base+208,2);const int half=within/128,offset=within&127,lane=offset/32,l=offset&31,qindex=l+((lane==0||lane==2)?0:32);const auto qbyte=ql[half*64+qindex],high=qh[half*32+l];const int nibble=(lane==0||lane==1)?(qbyte&15):(qbyte>>4);const int quant=(nibble|(((high>>(lane*2))&3)<<4))-32;const int scale_index=half*8+(l/16)+lane*2;return qwen_half_value(d_bits)*scales[scale_index]*quant;}

// Row dot products for the four formats the HF quantizer emits.
//
// The `_value` decoders above are correct but decode a block's scales afresh
// for every element, which is 32-256x redundant inside a matvec. These unpack
// each block's scales once and stream the codes, which is what makes a
// quantized forward pass competitive with an f32 one rather than slower.
//
// `elements` must be a whole number of blocks; the caller's shapes are already
// constrained that way by the quantizer.
inline float qwen_q8_dot_row(const std::uint8_t* packed, const float* input,
                             std::uint64_t elements) {
    double total = 0.0;
    for (std::uint64_t block = 0; block * 32 < elements; ++block) {
        const std::uint8_t* base = packed + block * kQ8BlockSize;
        std::uint16_t scale_bits = 0;
        std::memcpy(&scale_bits, base, 2);
        const auto* codes = reinterpret_cast<const std::int8_t*>(base + 2);
        const float* x = input + block * 32;
        float partial = 0.0f;
        for (int i = 0; i < 32; ++i) partial += static_cast<float>(codes[i]) * x[i];
        total += static_cast<double>(qwen_half_value(scale_bits)) * partial;
    }
    return static_cast<float>(total);
}

// Shared 6-bit scale/min unpacking for Q4_K and Q5_K.
inline void qwen_k4_scale_min(const std::uint8_t* scales, int index,
                              int* scale, int* minimum) {
    if (index < 4) {
        *scale = scales[index] & 63;
        *minimum = scales[index + 4] & 63;
    } else {
        *scale = (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4);
        *minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
    }
}

inline float qwen_q4k_dot_row(const std::uint8_t* packed, const float* input,
                              std::uint64_t elements) {
    double total = 0.0;
    for (std::uint64_t block = 0; block * kBlockElements < elements; ++block) {
        const std::uint8_t* base = packed + block * kQ4KBlockSize;
        std::uint16_t d_bits = 0, dmin_bits = 0;
        std::memcpy(&d_bits, base, 2);
        std::memcpy(&dmin_bits, base + 2, 2);
        const float d = qwen_half_value(d_bits), dmin = qwen_half_value(dmin_bits);
        const std::uint8_t* scales = base + 4;
        const std::uint8_t* quants = base + 16;
        const float* x = input + block * kBlockElements;
        for (int group = 0; group < 4; ++group) {
            for (int half = 0; half < 2; ++half) {
                int scale = 0, minimum = 0;
                qwen_k4_scale_min(scales, group * 2 + half, &scale, &minimum);
                // value = d*scale*q - dmin*min, so the min term factors out of
                // the sum as dmin*min*sum(x).
                float dot = 0.0f, sum = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    const std::uint8_t byte = quants[group * 32 + i];
                    const int q = half == 0 ? (byte & 15) : (byte >> 4);
                    const float value = x[group * 64 + half * 32 + i];
                    dot += static_cast<float>(q) * value;
                    sum += value;
                }
                total += static_cast<double>(d) * scale * dot -
                         static_cast<double>(dmin) * minimum * sum;
            }
        }
    }
    return static_cast<float>(total);
}

inline float qwen_q5k_dot_row(const std::uint8_t* packed, const float* input,
                              std::uint64_t elements) {
    double total = 0.0;
    for (std::uint64_t block = 0; block * kBlockElements < elements; ++block) {
        const std::uint8_t* base = packed + block * kQ5KBlockSize;
        std::uint16_t d_bits = 0, dmin_bits = 0;
        std::memcpy(&d_bits, base, 2);
        std::memcpy(&dmin_bits, base + 2, 2);
        const float d = qwen_half_value(d_bits), dmin = qwen_half_value(dmin_bits);
        const std::uint8_t* scales = base + 4;
        const std::uint8_t* high = base + 16;
        const std::uint8_t* quants = base + 48;
        const float* x = input + block * kBlockElements;
        for (int group = 0; group < 4; ++group) {
            for (int half = 0; half < 2; ++half) {
                int scale = 0, minimum = 0;
                qwen_k4_scale_min(scales, group * 2 + half, &scale, &minimum);
                float dot = 0.0f, sum = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    const std::uint8_t byte = quants[group * 32 + i];
                    const int low = half == 0 ? (byte & 15) : (byte >> 4);
                    const int bit = (high[i] >> (2 * group + half)) & 1;
                    const float value = x[group * 64 + half * 32 + i];
                    dot += static_cast<float>(low + 16 * bit) * value;
                    sum += value;
                }
                total += static_cast<double>(d) * scale * dot -
                         static_cast<double>(dmin) * minimum * sum;
            }
        }
    }
    return static_cast<float>(total);
}

inline float qwen_q6k_dot_row(const std::uint8_t* packed, const float* input,
                              std::uint64_t elements) {
    double total = 0.0;
    for (std::uint64_t block = 0; block * kBlockElements < elements; ++block) {
        const std::uint8_t* base = packed + block * kQ6KBlockSize;
        const std::uint8_t* ql = base;
        const std::uint8_t* qh = base + 128;
        const auto* scales = reinterpret_cast<const std::int8_t*>(base + 192);
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, base + 208, 2);
        const float d = qwen_half_value(d_bits);
        const float* x = input + block * kBlockElements;
        for (int half = 0; half < 2; ++half) {
            for (int lane = 0; lane < 4; ++lane) {
                // Each 16-element group has its own signed scale; a lane spans
                // two of them.
                for (int part = 0; part < 2; ++part) {
                    const int scale = scales[half * 8 + part + lane * 2];
                    float dot = 0.0f;
                    for (int i = 0; i < 16; ++i) {
                        const int l = part * 16 + i;
                        const int index = l + ((lane == 0 || lane == 2) ? 0 : 32);
                        const std::uint8_t byte = ql[half * 64 + index];
                        const int nibble = (lane == 0 || lane == 1) ? (byte & 15) : (byte >> 4);
                        const int q = (nibble | (((qh[half * 32 + l] >> (lane * 2)) & 3) << 4)) - 32;
                        dot += static_cast<float>(q) * x[half * 128 + lane * 32 + l];
                    }
                    total += static_cast<double>(d) * scale * dot;
                }
            }
        }
    }
    return static_cast<float>(total);
}


// Q2_K (GGML type 10): 84 bytes per 256 values -> scales[16] qs[64] d(2) dmin(2).
// Each scales byte packs a 4-bit scale (low nibble) and a 4-bit min (high
// nibble) governing one 16-element group; value = d*scale*q - dmin*min. The
// 256 values are two 128-element halves, and within a half the 2-bit quants
// for group j sit at bit offset 2*j of that half's 32 qs bytes.
constexpr std::uint32_t kQ2KBlockBytes = 84;

// Q3_K (GGML type 11): 110 bytes per 256 values -> hmask[32] qs[64] scales[12] d(2).
// The quant is a 2-bit low part from qs plus an inverted high bit from hmask
// (a set mask bit means "do not subtract 4"), giving a signed 3-bit value;
// value = d*(scale-32)*q.
constexpr std::uint32_t kQ3KBlockBytes = 110;

// The 16 six-bit Q3_K scales are packed into 12 bytes: the low and high
// nibbles of bytes 0..7 carry the low 4 bits of each scale, and bytes 8..11
// supply the top 2 bits.
inline int qwen_q3k_scale(const std::uint8_t* scales, int index) {
    const int group = index / 4, byte = index & 3;
    const int packed_low = scales[(group & 1) ? 4 + byte : byte];
    const int nibble = (group < 2) ? (packed_low & 15) : (packed_low >> 4);
    return nibble | (((scales[8 + byte] >> (2 * group)) & 3) << 4);
}

inline float qwen_q2k_value(const std::uint8_t* packed, std::uint64_t absolute) {
    const auto block = absolute / 256;
    const int within = static_cast<int>(absolute & 255);
    const auto* base = packed + block * kQ2KBlockBytes;
    std::uint16_t d_bits = 0, dmin_bits = 0;
    std::memcpy(&d_bits, base + 80, 2);
    std::memcpy(&dmin_bits, base + 82, 2);
    const int half = within / 128, rest = within & 127;
    const int group = rest / 32, lane = rest & 31, sub = lane / 16, element = lane & 15;
    const int quant = (base[16 + half * 32 + sub * 16 + element] >> (2 * group)) & 3;
    const auto scale_byte = base[half * 8 + group * 2 + sub];
    return qwen_half_value(d_bits) * (scale_byte & 15) * quant
        - qwen_half_value(dmin_bits) * (scale_byte >> 4);
}

// Dot one Q2_K row against `input`. Hoisting the per-group scale and min out
// of the inner loop is why this walks super-blocks rather than reusing
// qwen_q2k_value per element.
inline float qwen_q2k_dot_row(
    const std::uint8_t* packed, const float* input, int elements, std::uint64_t row
) {
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(elements / 256) * kQ2KBlockBytes;
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kQ2KBlockBytes;
        std::uint16_t d_bits = 0, dmin_bits = 0;
        std::memcpy(&d_bits, base + 80, 2);
        std::memcpy(&dmin_bits, base + 82, 2);
        const float d = qwen_half_value(d_bits), dmin = qwen_half_value(dmin_bits);
        const float* vector = input + block * 256;
        for (int half = 0; half < 2; ++half)
            for (int group = 0; group < 4; ++group)
                for (int sub = 0; sub < 2; ++sub) {
                    const auto scale_byte = base[half * 8 + group * 2 + sub];
                    const float ds = d * (scale_byte & 15), dm = dmin * (scale_byte >> 4);
                    const auto* quants = base + 16 + half * 32 + sub * 16;
                    const auto* values = vector + half * 128 + group * 32 + sub * 16;
                    for (int lane = 0; lane < 16; ++lane)
                        result += (ds * ((quants[lane] >> (2 * group)) & 3) - dm) * values[lane];
                }
    }
    return result;
}

inline float qwen_q3k_value(const std::uint8_t* packed, std::uint64_t absolute) {
    const auto block = absolute / 256;
    const int within = static_cast<int>(absolute & 255);
    const auto* base = packed + block * kQ3KBlockBytes;
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, base + 108, 2);
    const int half = within / 128, rest = within & 127;
    const int group = rest / 32, lane = rest & 31, sub = lane / 16, element = lane & 15;
    const int low = (base[32 + half * 32 + sub * 16 + element] >> (2 * group)) & 3;
    const int high = (base[lane] & (1 << (half * 4 + group))) ? 0 : 4;
    const int scale = qwen_q3k_scale(base + 96, half * 8 + group * 2 + sub);
    return qwen_half_value(d_bits) * (scale - 32) * (low - high);
}

// Dot one Q3_K row against `input`, accumulating each 16-element group before
// applying its shared scale.
inline float qwen_q3k_dot_row(
    const std::uint8_t* packed, const float* input, int elements, std::uint64_t row
) {
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(elements / 256) * kQ3KBlockBytes;
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kQ3KBlockBytes;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, base + 108, 2);
        const float d = qwen_half_value(d_bits);
        const auto* hmask = base;
        const auto* scales = base + 96;
        const float* vector = input + block * 256;
        for (int half = 0; half < 2; ++half)
            for (int group = 0; group < 4; ++group)
                for (int sub = 0; sub < 2; ++sub) {
                    const float ds =
                        d * (qwen_q3k_scale(scales, half * 8 + group * 2 + sub) - 32);
                    const int mask = 1 << (half * 4 + group);
                    const auto* quants = base + 32 + half * 32 + sub * 16;
                    const auto* values = vector + half * 128 + group * 32 + sub * 16;
                    float partial = 0.0f;
                    for (int lane = 0; lane < 16; ++lane) {
                        const int low = (quants[lane] >> (2 * group)) & 3;
                        const int high = (hmask[sub * 16 + lane] & mask) ? 0 : 4;
                        partial += static_cast<float>(low - high) * values[lane];
                    }
                    result += ds * partial;
                }
    }
    return result;
}

// IQ2_XXS (GGML type 16): 66 bytes per 256 values -> d(2) qs[32] as uint16.
// Each 32-element group is two uint32: the low word holds four 8-bit indices
// into the 256-entry grid of 8-value patterns, and the high word packs four
// 7-bit sign selectors (bits 0..27) plus a 4-bit scale in its top nibble.
constexpr std::uint32_t kIq2xxsBlockBytes = 66;

inline float qwen_iq2xxs_value(const std::uint8_t* packed, std::uint64_t absolute) {
    const auto block = absolute / 256;
    const int within = static_cast<int>(absolute & 255);
    const auto* base = packed + block * kIq2xxsBlockBytes;
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, base, 2);
    const int group = within / 32, rest = within & 31;
    const int quad = rest / 8, element = rest & 7;
    std::uint32_t low = 0, high = 0;
    std::memcpy(&low, base + 2 + group * 8, 4);
    std::memcpy(&high, base + 2 + group * 8 + 4, 4);
    const float scale = qwen_half_value(d_bits) * (0.5f + (high >> 28)) * 0.25f;
    const std::uint8_t signs = kIq2xxsSigns[(high >> (7 * quad)) & 127];
    const std::uint8_t value = kIq2xxsGrid[(low >> (8 * quad)) & 255][element];
    return (signs >> element) & 1 ? -scale * value : scale * value;
}

// Dot one IQ2_XXS row. Decoding a whole 8-value grid entry at a time keeps the
// table lookup and the sign byte out of the innermost loop.
inline float qwen_iq2xxs_dot_row(
    const std::uint8_t* packed, const float* input, int elements, std::uint64_t row
) {
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(elements / 256) * kIq2xxsBlockBytes;
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq2xxsBlockBytes;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, base, 2);
        const float d = qwen_half_value(d_bits);
        const float* vector = input + block * 256;
        for (int group = 0; group < 8; ++group) {
            std::uint32_t low = 0, high = 0;
            std::memcpy(&low, base + 2 + group * 8, 4);
            std::memcpy(&high, base + 2 + group * 8 + 4, 4);
            const float scale = d * (0.5f + (high >> 28)) * 0.25f;
            float partial = 0.0f;
            for (int quad = 0; quad < 4; ++quad) {
                const std::uint8_t signs = kIq2xxsSigns[(high >> (7 * quad)) & 127];
                const std::uint8_t* pattern = kIq2xxsGrid[(low >> (8 * quad)) & 255];
                const float* values = vector + group * 32 + quad * 8;
                for (int element = 0; element < 8; ++element) {
                    const float weight = static_cast<float>(pattern[element]);
                    partial += ((signs >> element) & 1 ? -weight : weight) * values[element];
                }
            }
            result += scale * partial;
        }
    }
    return result;
}

// IQ3_XXS (GGML type 18): 98 bytes per 256 values -> d(2) qs[64] scales[32].
// Each 32-element group has one uint32 of scale-and-signs, laid out like
// IQ2_XXS, but the grid entries are only four values wide, so each 8-output
// quad consumes two consecutive qs bytes.
constexpr std::uint32_t kIq3xxsBlockBytes = 98;

inline float qwen_iq3xxs_value(const std::uint8_t* packed, std::uint64_t absolute) {
    const auto block = absolute / 256;
    const int within = static_cast<int>(absolute & 255);
    const auto* base = packed + block * kIq3xxsBlockBytes;
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, base, 2);
    const int group = within / 32, rest = within & 31;
    const int quad = rest / 8, element = rest & 7;
    std::uint32_t aux = 0;
    std::memcpy(&aux, base + 2 + 64 + group * 4, 4);
    const float scale = qwen_half_value(d_bits) * (0.5f + (aux >> 28)) * 0.5f;
    const std::uint8_t signs = kIq2xxsSigns[(aux >> (7 * quad)) & 127];
    const std::uint32_t pattern =
        kIq3xxsGrid[base[2 + group * 8 + quad * 2 + (element >> 2)]];
    const float value =
        static_cast<float>((pattern >> (8 * (element & 3))) & 0xffu);
    return (signs >> element) & 1 ? -scale * value : scale * value;
}

inline float qwen_iq3xxs_dot_row(
    const std::uint8_t* packed, const float* input, int elements, std::uint64_t row
) {
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(elements / 256) * kIq3xxsBlockBytes;
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq3xxsBlockBytes;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, base, 2);
        const float d = qwen_half_value(d_bits);
        const float* vector = input + block * 256;
        for (int group = 0; group < 8; ++group) {
            std::uint32_t aux = 0;
            std::memcpy(&aux, base + 2 + 64 + group * 4, 4);
            const float scale = d * (0.5f + (aux >> 28)) * 0.5f;
            float partial = 0.0f;
            for (int quad = 0; quad < 4; ++quad) {
                const std::uint8_t signs = kIq2xxsSigns[(aux >> (7 * quad)) & 127];
                const auto* indices = base + 2 + group * 8 + quad * 2;
                const float* values = vector + group * 32 + quad * 8;
                for (int half = 0; half < 2; ++half) {
                    const std::uint32_t pattern = kIq3xxsGrid[indices[half]];
                    for (int element = 0; element < 4; ++element) {
                        const float weight =
                            static_cast<float>((pattern >> (8 * element)) & 0xffu);
                        const int bit = half * 4 + element;
                        partial += ((signs >> bit) & 1 ? -weight : weight)
                            * values[bit];
                    }
                }
            }
            result += scale * partial;
        }
    }
    return result;
}

// IQ2_S (GGML type 22): 82 bytes per 256 values -> d(2) qs[32] signs[32]
// qh[8] scales[8]. Unlike the XXS formats the signs are stored literally, one
// byte per 8 outputs, and the grid index gains two high bits from qh.
constexpr std::uint32_t kIq2sBlockBytes = 82;

inline float qwen_iq2s_value(const std::uint8_t* packed, std::uint64_t absolute) {
    const auto block = absolute / 256;
    const int within = static_cast<int>(absolute & 255);
    const auto* base = packed + block * kIq2sBlockBytes;
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, base, 2);
    const int group = within / 16, rest = within & 15;
    const int half = rest / 8, element = rest & 7;
    const int index = group * 2 + half;
    const auto* quants = base + 2;
    const auto* signs = base + 34;
    const auto* high = base + 66;
    const auto* scales = base + 74;
    const int entry =
        quants[index] | (((high[index >> 2] >> (2 * (index & 3))) & 3) << 8);
    const int scale = (scales[group >> 1] >> (4 * (group & 1))) & 15;
    const float db = qwen_half_value(d_bits) * (0.5f + scale) * 0.25f;
    const float value =
        static_cast<float>((kIq2sGrid[entry] >> (8 * element)) & 0xffull);
    return (signs[index] >> element) & 1 ? -db * value : db * value;
}

inline float qwen_iq2s_dot_row(
    const std::uint8_t* packed, const float* input, int elements, std::uint64_t row
) {
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(elements / 256) * kIq2sBlockBytes;
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq2sBlockBytes;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, base, 2);
        const float d = qwen_half_value(d_bits);
        const auto* quants = base + 2;
        const auto* signs = base + 34;
        const auto* high = base + 66;
        const auto* scales = base + 74;
        const float* vector = input + block * 256;
        for (int group = 0; group < 16; ++group) {
            const int scale = (scales[group >> 1] >> (4 * (group & 1))) & 15;
            const float db = d * (0.5f + scale) * 0.25f;
            float partial = 0.0f;
            for (int half = 0; half < 2; ++half) {
                const int index = group * 2 + half;
                const int entry =
                    quants[index] | (((high[index >> 2] >> (2 * (index & 3))) & 3) << 8);
                const std::uint64_t pattern = kIq2sGrid[entry];
                const std::uint8_t sign = signs[index];
                const float* values = vector + group * 16 + half * 8;
                for (int element = 0; element < 8; ++element) {
                    const float weight =
                        static_cast<float>((pattern >> (8 * element)) & 0xffull);
                    partial += ((sign >> element) & 1 ? -weight : weight) * values[element];
                }
            }
            result += db * partial;
        }
    }
    return result;
}

// IQ3_S (GGML type 21): 110 bytes per 256 values -> d(2) qs[64] qh[8]
// signs[32] scales[4]. One extra index bit per grid entry comes from qh, and
// the scale is an odd multiplier rather than an offset-and-scale pair.
constexpr std::uint32_t kIq3sBlockBytes = 110;

inline float qwen_iq3s_value(const std::uint8_t* packed, std::uint64_t absolute) {
    const auto block = absolute / 256;
    const int within = static_cast<int>(absolute & 255);
    const auto* base = packed + block * kIq3sBlockBytes;
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, base, 2);
    const auto* quants = base + 2;
    const auto* high = base + 66;
    const auto* signs = base + 74;
    const auto* scales = base + 106;
    const int index = within / 4, element = within & 3;
    const int entry = quants[index] | (((high[index >> 3] >> (index & 7)) & 1) << 8);
    const int group = within / 32, quad = (within % 32) / 8, bit = within & 7;
    const int scale = (scales[group >> 1] >> (4 * (group & 1))) & 15;
    const float db = qwen_half_value(d_bits) * (1 + 2 * scale);
    const float value =
        static_cast<float>((kIq3sGrid[entry] >> (8 * element)) & 0xffu);
    return (signs[group * 4 + quad] >> bit) & 1 ? -db * value : db * value;
}

inline float qwen_iq3s_dot_row(
    const std::uint8_t* packed, const float* input, int elements, std::uint64_t row
) {
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(elements / 256) * kIq3sBlockBytes;
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq3sBlockBytes;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, base, 2);
        const float d = qwen_half_value(d_bits);
        const auto* quants = base + 2;
        const auto* high = base + 66;
        const auto* signs = base + 74;
        const auto* scales = base + 106;
        const float* vector = input + block * 256;
        for (int group = 0; group < 8; ++group) {
            const int scale = (scales[group >> 1] >> (4 * (group & 1))) & 15;
            const float db = d * (1 + 2 * scale);
            float partial = 0.0f;
            for (int quad = 0; quad < 4; ++quad) {
                const std::uint8_t sign = signs[group * 4 + quad];
                const float* values = vector + group * 32 + quad * 8;
                for (int half = 0; half < 2; ++half) {
                    const int index = group * 8 + quad * 2 + half;
                    const int entry =
                        quants[index] | (((high[index >> 3] >> (index & 7)) & 1) << 8);
                    const std::uint32_t pattern = kIq3sGrid[entry];
                    for (int element = 0; element < 4; ++element) {
                        const float weight =
                            static_cast<float>((pattern >> (8 * element)) & 0xffu);
                        const int bit = half * 4 + element;
                        partial += ((sign >> bit) & 1 ? -weight : weight) * values[bit];
                    }
                }
            }
            result += db * partial;
        }
    }
    return result;
}

// IQ2_XS (GGML type 17): 74 bytes per 256 values -> d(2) qs[32] as uint16
// scales[8]. Each uint16 carries a 9-bit grid index and a 7-bit sign selector.
constexpr std::uint32_t kIq2xsBlockBytes = 74;

inline float qwen_iq2xs_value(const std::uint8_t* packed, std::uint64_t absolute) {
    const auto block = absolute / 256;
    const int within = static_cast<int>(absolute & 255);
    const auto* base = packed + block * kIq2xsBlockBytes;
    std::uint16_t d_bits = 0, entry = 0;
    std::memcpy(&d_bits, base, 2);
    const int index = within / 8, element = within & 7;
    std::memcpy(&entry, base + 2 + index * 2, 2);
    const int group = within / 16;
    const int scale = (base[66 + (group >> 1)] >> (4 * (group & 1))) & 15;
    const float db = qwen_half_value(d_bits) * (0.5f + scale) * 0.25f;
    const std::uint8_t signs = kIq2xxsSigns[entry >> 9];
    const float value =
        static_cast<float>((kIq2xsGrid[entry & 511] >> (8 * element)) & 0xffull);
    return (signs >> element) & 1 ? -db * value : db * value;
}

inline float qwen_iq2xs_dot_row(
    const std::uint8_t* packed, const float* input, int elements, std::uint64_t row
) {
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(elements / 256) * kIq2xsBlockBytes;
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq2xsBlockBytes;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, base, 2);
        const float d = qwen_half_value(d_bits);
        const float* vector = input + block * 256;
        for (int group = 0; group < 16; ++group) {
            const int scale = (base[66 + (group >> 1)] >> (4 * (group & 1))) & 15;
            const float db = d * (0.5f + scale) * 0.25f;
            float partial = 0.0f;
            for (int half = 0; half < 2; ++half) {
                std::uint16_t entry = 0;
                std::memcpy(&entry, base + 2 + (group * 2 + half) * 2, 2);
                const std::uint64_t pattern = kIq2xsGrid[entry & 511];
                const std::uint8_t signs = kIq2xxsSigns[entry >> 9];
                const float* values = vector + group * 16 + half * 8;
                for (int element = 0; element < 8; ++element) {
                    const float weight =
                        static_cast<float>((pattern >> (8 * element)) & 0xffull);
                    partial += ((signs >> element) & 1 ? -weight : weight) * values[element];
                }
            }
            result += db * partial;
        }
    }
    return result;
}

// IQ1_S (GGML type 19): 50 bytes per 256 values -> d(2) qs[32] qh[8*2].
// One halfword of qh covers a whole 32-value group: the high three bits of each
// of its four 11-bit grid indices (the low eight come from qs), the group's own
// 3-bit scale multiplier in bits 12-14, and in bit 15 the sign of the +-0.125
// delta added to every weight in the group.
constexpr std::uint32_t kIq1sBlockBytes = 50;
constexpr float kIq1sDelta = 0.125f;

// Grid index of one group of eight weights, and the group's scale and delta.
// The element decoder and the row dot product below want exactly this much of
// the halfword each, so it is unpacked once here rather than twice there.
inline std::uint32_t qwen_iq1s_index(
    const std::uint8_t* base, int group, int part, std::uint16_t qh
) {
    return static_cast<std::uint32_t>(base[2 + group * 4 + part]) |
        ((static_cast<std::uint32_t>(qh >> (3 * part)) & 7u) << 8);
}

inline float qwen_iq1s_value(const std::uint8_t* packed, std::uint64_t absolute) {
    const auto block = absolute / 256;
    const int within = static_cast<int>(absolute & 255);
    const auto* base = packed + block * kIq1sBlockBytes;
    // Group of 32, group of 8 within it, element within that.
    const int group = within / 32, part = (within % 32) / 8, lane = within & 7;
    std::uint16_t scale_bits = 0, qh = 0;
    std::memcpy(&scale_bits, base, 2);
    std::memcpy(&qh, base + 34 + group * 2, 2);
    const float delta = (qh & 0x8000) ? -kIq1sDelta : kIq1sDelta;
    const float scale = qwen_half_value(scale_bits) *
        static_cast<float>(2 * ((qh >> 12) & 7) + 1);
    const auto weight = static_cast<std::int8_t>(
        (kIq1sGrid[qwen_iq1s_index(base, group, part, qh)] >> (8 * lane)) & 0xffull);
    return scale * (static_cast<float>(weight) + delta);
}

inline float qwen_iq1s_dot_row(
    const std::uint8_t* packed, const float* input, int elements, std::uint64_t row
) {
    const int blocks = elements / 256;
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(blocks) * kIq1sBlockBytes;
    float result = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        const auto* base = row_data + block * kIq1sBlockBytes;
        std::uint16_t scale_bits = 0;
        std::memcpy(&scale_bits, base, 2);
        const float d = qwen_half_value(scale_bits);
        const float* vector = input + block * 256;
        for (int group = 0; group < 8; ++group) {
            std::uint16_t qh = 0;
            std::memcpy(&qh, base + 34 + group * 2, 2);
            const float scale = d * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
            const float delta = (qh & 0x8000) ? -kIq1sDelta : kIq1sDelta;
            const float* values = vector + group * 32;
            float partial = 0.0f;
            for (int part = 0; part < 4; ++part) {
                const std::uint64_t entry =
                    kIq1sGrid[qwen_iq1s_index(base, group, part, qh)];
                for (int lane = 0; lane < 8; ++lane) {
                    const auto weight =
                        static_cast<std::int8_t>((entry >> (8 * lane)) & 0xffull);
                    partial += (static_cast<float>(weight) + delta) *
                        values[part * 8 + lane];
                }
            }
            result += scale * partial;
        }
    }
    return result;
}

// IQ1_M (GGML type 29): 56 bytes per 256 values -> qs[32] qh[16] scales[8].
// Same 2048-entry grid as IQ1_S, but the super-block scale is not stored as a
// field of its own: its sixteen half-precision bits are scattered four at a
// time across the top nibbles of the four scale halfwords, and each halfword
// also carries two 3-bit sub-scales. Every group of eight weights takes an
// 11-bit grid index (eight bits from qs, three from qh) and a per-group sign
// for the same +-0.125 delta IQ1_S applies.
constexpr std::uint32_t kIq1mBlockBytes = 56;
constexpr float kIq1mDelta = 0.125f;

// Super-block scale of an IQ1_M block, reassembled from the four scale
// halfwords. Shared by the element decoder and the row dot product.
inline float qwen_iq1m_scale(const std::uint8_t* base) {
    std::uint16_t sc[4];
    std::memcpy(sc, base + 48, 8);
    const std::uint16_t bits = static_cast<std::uint16_t>(
        (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) |
        (sc[3] & 0xf000));
    return qwen_half_value(bits);
}

inline float qwen_iq1m_value(const std::uint8_t* packed, std::uint64_t absolute) {
    const auto block = absolute / 256;
    const int within = static_cast<int>(absolute & 255);
    const auto* base = packed + block * kIq1mBlockBytes;
    std::uint16_t sc[4];
    std::memcpy(sc, base + 48, 8);
    // Sub-block of 32, group of 8 within it, element within the group.
    const int ib = within / 32, group = (within % 32) / 8, lane = within & 7;
    const auto qh = base[32 + ib * 2 + group / 2];
    const std::uint32_t index = static_cast<std::uint32_t>(base[ib * 4 + group]) |
        ((static_cast<std::uint32_t>(qh) << ((group & 1) ? 4 : 8)) & 0x700u);
    const float delta = (qh & ((group & 1) ? 0x80 : 0x08)) ? -kIq1mDelta : kIq1mDelta;
    // The two halves of a sub-block carry separate 3-bit scales, packed three
    // bits apart in the halfword that covers this pair of sub-blocks.
    const int shift = 6 * (ib & 1) + (group < 2 ? 0 : 3);
    const float scale = qwen_iq1m_scale(base) *
        static_cast<float>(2 * ((sc[ib / 2] >> shift) & 7) + 1);
    const auto weight =
        static_cast<std::int8_t>((kIq1sGrid[index] >> (8 * lane)) & 0xffull);
    return scale * (static_cast<float>(weight) + delta);
}

inline float qwen_iq1m_dot_row(
    const std::uint8_t* packed, const float* input, int elements, std::uint64_t row
) {
    const int blocks = elements / 256;
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(blocks) * kIq1mBlockBytes;
    float result = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        const auto* base = row_data + block * kIq1mBlockBytes;
        std::uint16_t sc[4];
        std::memcpy(sc, base + 48, 8);
        const float d = qwen_iq1m_scale(base);
        const float* vector = input + block * 256;
        for (int ib = 0; ib < 8; ++ib) {
            for (int group = 0; group < 4; ++group) {
                const auto qh = base[32 + ib * 2 + group / 2];
                const std::uint32_t index =
                    static_cast<std::uint32_t>(base[ib * 4 + group]) |
                    ((static_cast<std::uint32_t>(qh) << ((group & 1) ? 4 : 8)) & 0x700u);
                const float delta =
                    (qh & ((group & 1) ? 0x80 : 0x08)) ? -kIq1mDelta : kIq1mDelta;
                const int shift = 6 * (ib & 1) + (group < 2 ? 0 : 3);
                const float scale =
                    d * static_cast<float>(2 * ((sc[ib / 2] >> shift) & 7) + 1);
                const std::uint64_t entry = kIq1sGrid[index];
                const float* values = vector + ib * 32 + group * 8;
                float partial = 0.0f;
                for (int lane = 0; lane < 8; ++lane) {
                    const auto weight =
                        static_cast<std::int8_t>((entry >> (8 * lane)) & 0xffull);
                    partial += (static_cast<float>(weight) + delta) * values[lane];
                }
                result += scale * partial;
            }
        }
    }
    return result;
}

// IQ4_XS (GGML type 23): 136 bytes per 256 values -> d(2) scales_h(2)
// scales_l[4] qs[128]. Not a codebook of patterns like the other IQ formats:
// each 4-bit code indexes the sixteen non-uniform IQ4_NL levels, and each
// 32-value sub-block has a 6-bit signed scale split across scales_l/scales_h.
constexpr std::uint32_t kIq4xsBlockBytes = 136;

inline float qwen_iq4xs_value(const std::uint8_t* packed, std::uint64_t absolute) {
    const auto block = absolute / 256;
    const int within = static_cast<int>(absolute & 255);
    const auto* base = packed + block * kIq4xsBlockBytes;
    std::uint16_t d_bits = 0, scales_high = 0;
    std::memcpy(&d_bits, base, 2);
    std::memcpy(&scales_high, base + 2, 2);
    const int sub = within / 32, element = within & 31;
    const int low = (base[4 + (sub >> 1)] >> (4 * (sub & 1))) & 15;
    const int scale = (low | (((scales_high >> (2 * sub)) & 3) << 4)) - 32;
    const std::uint8_t byte = base[8 + sub * 16 + (element & 15)];
    const int code = element < 16 ? (byte & 15) : (byte >> 4);
    return qwen_half_value(d_bits) * scale * kIq4nlValues[code];
}

inline float qwen_iq4xs_dot_row(
    const std::uint8_t* packed, const float* input, int elements, std::uint64_t row
) {
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(elements / 256) * kIq4xsBlockBytes;
    float result = 0.0f;
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * kIq4xsBlockBytes;
        std::uint16_t d_bits = 0, scales_high = 0;
        std::memcpy(&d_bits, base, 2);
        std::memcpy(&scales_high, base + 2, 2);
        const float d = qwen_half_value(d_bits);
        const float* vector = input + block * 256;
        for (int sub = 0; sub < 8; ++sub) {
            const int low = (base[4 + (sub >> 1)] >> (4 * (sub & 1))) & 15;
            const int scale = (low | (((scales_high >> (2 * sub)) & 3) << 4)) - 32;
            const auto* quants = base + 8 + sub * 16;
            const float* values = vector + sub * 32;
            float partial = 0.0f;
            for (int element = 0; element < 16; ++element) {
                const std::uint8_t byte = quants[element];
                partial += kIq4nlValues[byte & 15] * values[element]
                    + kIq4nlValues[byte >> 4] * values[element + 16];
            }
            result += d * scale * partial;
        }
    }
    return result;
}
