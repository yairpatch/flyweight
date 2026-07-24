#include "qwen_cpu_kernel.h"

#include <cstring>
#include <immintrin.h>

namespace {

float half_value(const std::uint8_t* pointer) {
    std::uint16_t bits;
    std::memcpy(&bits, pointer, sizeof(bits));
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}

__m512 bytes_to_float(__m128i values) {
    return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(values));
}

float q5_dot(const std::uint8_t* row_data, const float* input, int elements) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    const __m128i nibble_mask = _mm_set1_epi8(15);
    const __m128i bit_mask = _mm_set1_epi8(1);
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * 176;
        const float d = half_value(base);
        const float dmin = half_value(base + 2);
        const auto* scales = base + 4;
        const auto* high = base + 16;
        const auto* low = base + 48;
        const auto* vector = input + block * 256;
        for (int group = 0; group < 4; ++group) {
            for (int sub = 0; sub < 2; ++sub) {
                const int scale_index = group * 2 + sub;
                int scale;
                int minimum;
                if (scale_index < 4) {
                    scale = scales[scale_index] & 63;
                    minimum = scales[scale_index + 4] & 63;
                } else {
                    scale = (scales[scale_index + 4] & 15)
                        | ((scales[scale_index - 4] >> 6) << 4);
                    minimum = (scales[scale_index + 4] >> 4)
                        | ((scales[scale_index] >> 6) << 4);
                }
                const __m512 ds = _mm512_set1_ps(d * scale);
                const __m512 dm = _mm512_set1_ps(dmin * minimum);
                const int shift = 2 * group + sub;
                for (int lanes = 0; lanes < 32; lanes += 16) {
                    __m128i q = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(low + group * 32 + lanes)
                    );
                    q = sub == 0 ? _mm_and_si128(q, nibble_mask)
                                 : _mm_and_si128(_mm_srli_epi16(q, 4), nibble_mask);
                    __m128i bits = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(high + lanes)
                    );
                    bits = _mm_and_si128(_mm_srli_epi16(bits, shift), bit_mask);
                    q = _mm_add_epi8(q, _mm_slli_epi16(bits, 4));
                    const __m512 weights = _mm512_sub_ps(
                        _mm512_mul_ps(bytes_to_float(q), ds), dm
                    );
                    const __m512 values = _mm512_loadu_ps(
                        vector + group * 64 + sub * 32 + lanes
                    );
                    if (lanes == 0) sum0 = _mm512_fmadd_ps(weights, values, sum0);
                    else sum1 = _mm512_fmadd_ps(weights, values, sum1);
                }
            }
        }
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(sum0, sum1));
}

float q6_dot(const std::uint8_t* row_data, const float* input, int elements) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    const __m128i nibble_mask = _mm_set1_epi8(15);
    const __m128i high_mask = _mm_set1_epi8(3);
    const __m512 offset = _mm512_set1_ps(32.0f);
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * 210;
        const auto* ql = base;
        const auto* qh = base + 128;
        const auto* scales = reinterpret_cast<const std::int8_t*>(base + 192);
        const float d = half_value(base + 208);
        const auto* vector = input + block * 256;
        for (int half = 0; half < 2; ++half) {
            for (int segment = 0; segment < 4; ++segment) {
                const int q_offset = (segment == 0 || segment == 2) ? 0 : 32;
                for (int lanes = 0; lanes < 32; lanes += 16) {
                    __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                        ql + half * 64 + q_offset + lanes
                    ));
                    q = segment < 2 ? _mm_and_si128(q, nibble_mask)
                                    : _mm_and_si128(_mm_srli_epi16(q, 4), nibble_mask);
                    __m128i high = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                        qh + half * 32 + lanes
                    ));
                    high = _mm_and_si128(
                        _mm_srli_epi16(high, segment * 2), high_mask
                    );
                    q = _mm_add_epi8(q, _mm_slli_epi16(high, 4));
                    const int scale_index = half * 8 + lanes / 16 + segment * 2;
                    const __m512 factor = _mm512_set1_ps(d * scales[scale_index]);
                    const __m512 weights = _mm512_mul_ps(
                        _mm512_sub_ps(bytes_to_float(q), offset), factor
                    );
                    const __m512 values = _mm512_loadu_ps(
                        vector + half * 128 + segment * 32 + lanes
                    );
                    if (lanes == 0) sum0 = _mm512_fmadd_ps(weights, values, sum0);
                    else sum1 = _mm512_fmadd_ps(weights, values, sum1);
                }
            }
        }
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(sum0, sum1));
}

float q8_dot(const std::uint8_t* row_data, const float* input, int elements) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    for (int block = 0; block < elements / 32; ++block) {
        const auto* base = row_data + block * 34;
        const __m512 scale = _mm512_set1_ps(half_value(base));
        const __m256i quantized = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(base + 2)
        );
        const __m512 q0 = _mm512_cvtepi32_ps(
            _mm512_cvtepi8_epi32(_mm256_castsi256_si128(quantized))
        );
        const __m512 q1 = _mm512_cvtepi32_ps(
            _mm512_cvtepi8_epi32(_mm256_extracti128_si256(quantized, 1))
        );
        sum0 = _mm512_fmadd_ps(
            _mm512_mul_ps(q0, scale), _mm512_loadu_ps(input + block * 32), sum0
        );
        sum1 = _mm512_fmadd_ps(
            _mm512_mul_ps(q1, scale), _mm512_loadu_ps(input + block * 32 + 16), sum1
        );
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(sum0, sum1));
}

void q5_dot_pair(const std::uint8_t* row_data,const float*first,const float*second,int elements,float&out_first,float&out_second){
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),b0=_mm512_setzero_ps(),b1=_mm512_setzero_ps();
    const __m128i nibble_mask=_mm_set1_epi8(15),bit_mask=_mm_set1_epi8(1);
    for(int block=0;block<elements/256;++block){const auto*base=row_data+block*176;const float d=half_value(base),dmin=half_value(base+2);const auto*scales=base+4;const auto*high=base+16;const auto*low=base+48;
        for(int group=0;group<4;++group)for(int sub=0;sub<2;++sub){const int index=group*2+sub;int scale,minimum;if(index<4){scale=scales[index]&63;minimum=scales[index+4]&63;}else{scale=(scales[index+4]&15)|((scales[index-4]>>6)<<4);minimum=(scales[index+4]>>4)|((scales[index]>>6)<<4);}const __m512 ds=_mm512_set1_ps(d*scale),dm=_mm512_set1_ps(dmin*minimum);const int shift=2*group+sub,offset=block*256+group*64+sub*32;
            for(int lanes=0;lanes<32;lanes+=16){__m128i q=_mm_loadu_si128(reinterpret_cast<const __m128i*>(low+group*32+lanes));q=sub==0?_mm_and_si128(q,nibble_mask):_mm_and_si128(_mm_srli_epi16(q,4),nibble_mask);__m128i bits=_mm_loadu_si128(reinterpret_cast<const __m128i*>(high+lanes));bits=_mm_and_si128(_mm_srli_epi16(bits,shift),bit_mask);q=_mm_add_epi8(q,_mm_slli_epi16(bits,4));const __m512 weights=_mm512_sub_ps(_mm512_mul_ps(bytes_to_float(q),ds),dm);if(lanes==0){a0=_mm512_fmadd_ps(weights,_mm512_loadu_ps(first+offset+lanes),a0);b0=_mm512_fmadd_ps(weights,_mm512_loadu_ps(second+offset+lanes),b0);}else{a1=_mm512_fmadd_ps(weights,_mm512_loadu_ps(first+offset+lanes),a1);b1=_mm512_fmadd_ps(weights,_mm512_loadu_ps(second+offset+lanes),b1);}}
        }
    }
    out_first=_mm512_reduce_add_ps(_mm512_add_ps(a0,a1));out_second=_mm512_reduce_add_ps(_mm512_add_ps(b0,b1));
}

void q6_dot_pair(const std::uint8_t*row_data,const float*first,const float*second,int elements,float&out_first,float&out_second){
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),b0=_mm512_setzero_ps(),b1=_mm512_setzero_ps();const __m128i nibble_mask=_mm_set1_epi8(15),high_mask=_mm_set1_epi8(3);const __m512 offset32=_mm512_set1_ps(32.0f);
    for(int block=0;block<elements/256;++block){const auto*base=row_data+block*210;const auto*ql=base;const auto*qh=base+128;const auto*scales=reinterpret_cast<const std::int8_t*>(base+192);const float d=half_value(base+208);
        for(int half=0;half<2;++half)for(int segment=0;segment<4;++segment){const int q_offset=(segment==0||segment==2)?0:32;
            for(int lanes=0;lanes<32;lanes+=16){__m128i q=_mm_loadu_si128(reinterpret_cast<const __m128i*>(ql+half*64+q_offset+lanes));q=segment<2?_mm_and_si128(q,nibble_mask):_mm_and_si128(_mm_srli_epi16(q,4),nibble_mask);__m128i high=_mm_loadu_si128(reinterpret_cast<const __m128i*>(qh+half*32+lanes));high=_mm_and_si128(_mm_srli_epi16(high,segment*2),high_mask);q=_mm_add_epi8(q,_mm_slli_epi16(high,4));const int scale_index=half*8+lanes/16+segment*2,index=block*256+half*128+segment*32+lanes;const __m512 weights=_mm512_mul_ps(_mm512_sub_ps(bytes_to_float(q),offset32),_mm512_set1_ps(d*scales[scale_index]));if(lanes==0){a0=_mm512_fmadd_ps(weights,_mm512_loadu_ps(first+index),a0);b0=_mm512_fmadd_ps(weights,_mm512_loadu_ps(second+index),b0);}else{a1=_mm512_fmadd_ps(weights,_mm512_loadu_ps(first+index),a1);b1=_mm512_fmadd_ps(weights,_mm512_loadu_ps(second+index),b1);}}
        }
    }
    out_first=_mm512_reduce_add_ps(_mm512_add_ps(a0,a1));out_second=_mm512_reduce_add_ps(_mm512_add_ps(b0,b1));
}

void q8_dot_pair(const std::uint8_t*row_data,const float*first,const float*second,int elements,float&out_first,float&out_second){
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),b0=_mm512_setzero_ps(),b1=_mm512_setzero_ps();
    for(int block=0;block<elements/32;++block){const auto*base=row_data+block*34;const __m512 scale=_mm512_set1_ps(half_value(base));const __m256i quantized=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(base+2));const __m512 q0=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_castsi256_si128(quantized))),scale),q1=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_extracti128_si256(quantized,1))),scale);a0=_mm512_fmadd_ps(q0,_mm512_loadu_ps(first+block*32),a0);a1=_mm512_fmadd_ps(q1,_mm512_loadu_ps(first+block*32+16),a1);b0=_mm512_fmadd_ps(q0,_mm512_loadu_ps(second+block*32),b0);b1=_mm512_fmadd_ps(q1,_mm512_loadu_ps(second+block*32+16),b1);}
    out_first=_mm512_reduce_add_ps(_mm512_add_ps(a0,a1));out_second=_mm512_reduce_add_ps(_mm512_add_ps(b0,b1));
}

void q5_dot_two_rows(
    const std::uint8_t* first_row,
    const std::uint8_t* second_row,
    const float* input,
    int elements,
    float& out_first,
    float& out_second
) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps();
    const __m128i nibble_mask = _mm_set1_epi8(15);
    const __m128i bit_mask = _mm_set1_epi8(1);
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base0 = first_row + block * 176;
        const auto* base1 = second_row + block * 176;
        const float d0 = half_value(base0);
        const float dmin0 = half_value(base0 + 2);
        const float d1 = half_value(base1);
        const float dmin1 = half_value(base1 + 2);
        const auto* scales0 = base0 + 4;
        const auto* scales1 = base1 + 4;
        const auto* high0 = base0 + 16;
        const auto* high1 = base1 + 16;
        const auto* low0 = base0 + 48;
        const auto* low1 = base1 + 48;
        const auto* vector = input + block * 256;
        for (int group = 0; group < 4; ++group) {
            for (int sub = 0; sub < 2; ++sub) {
                const int scale_index = group * 2 + sub;
                int scale0, minimum0, scale1, minimum1;
                if (scale_index < 4) {
                    scale0 = scales0[scale_index] & 63;
                    minimum0 = scales0[scale_index + 4] & 63;
                    scale1 = scales1[scale_index] & 63;
                    minimum1 = scales1[scale_index + 4] & 63;
                } else {
                    scale0 = (scales0[scale_index + 4] & 15)
                        | ((scales0[scale_index - 4] >> 6) << 4);
                    minimum0 = (scales0[scale_index + 4] >> 4)
                        | ((scales0[scale_index] >> 6) << 4);
                    scale1 = (scales1[scale_index + 4] & 15)
                        | ((scales1[scale_index - 4] >> 6) << 4);
                    minimum1 = (scales1[scale_index + 4] >> 4)
                        | ((scales1[scale_index] >> 6) << 4);
                }
                const __m512 ds0 = _mm512_set1_ps(d0 * scale0);
                const __m512 dm0 = _mm512_set1_ps(dmin0 * minimum0);
                const __m512 ds1 = _mm512_set1_ps(d1 * scale1);
                const __m512 dm1 = _mm512_set1_ps(dmin1 * minimum1);
                const int shift = 2 * group + sub;
                for (int lanes = 0; lanes < 32; lanes += 16) {
                    __m128i q0 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(low0 + group * 32 + lanes)
                    );
                    __m128i q1 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(low1 + group * 32 + lanes)
                    );
                    q0 = sub == 0 ? _mm_and_si128(q0, nibble_mask)
                                  : _mm_and_si128(_mm_srli_epi16(q0, 4), nibble_mask);
                    q1 = sub == 0 ? _mm_and_si128(q1, nibble_mask)
                                  : _mm_and_si128(_mm_srli_epi16(q1, 4), nibble_mask);
                    __m128i bits0 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(high0 + lanes)
                    );
                    __m128i bits1 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(high1 + lanes)
                    );
                    bits0 = _mm_and_si128(_mm_srli_epi16(bits0, shift), bit_mask);
                    bits1 = _mm_and_si128(_mm_srli_epi16(bits1, shift), bit_mask);
                    q0 = _mm_add_epi8(q0, _mm_slli_epi16(bits0, 4));
                    q1 = _mm_add_epi8(q1, _mm_slli_epi16(bits1, 4));
                    const __m512 values = _mm512_loadu_ps(
                        vector + group * 64 + sub * 32 + lanes
                    );
                    const __m512 weights0 = _mm512_sub_ps(
                        _mm512_mul_ps(bytes_to_float(q0), ds0), dm0
                    );
                    const __m512 weights1 = _mm512_sub_ps(
                        _mm512_mul_ps(bytes_to_float(q1), ds1), dm1
                    );
                    if (lanes == 0) {
                        a0 = _mm512_fmadd_ps(weights0, values, a0);
                        b0 = _mm512_fmadd_ps(weights1, values, b0);
                    } else {
                        a1 = _mm512_fmadd_ps(weights0, values, a1);
                        b1 = _mm512_fmadd_ps(weights1, values, b1);
                    }
                }
            }
        }
    }
    out_first = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    out_second = _mm512_reduce_add_ps(_mm512_add_ps(b0, b1));
}

void q6_dot_two_rows(
    const std::uint8_t* first_row,
    const std::uint8_t* second_row,
    const float* input,
    int elements,
    float& out_first,
    float& out_second
) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps();
    const __m128i nibble_mask = _mm_set1_epi8(15);
    const __m128i high_mask = _mm_set1_epi8(3);
    const __m512 offset = _mm512_set1_ps(32.0f);
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base0 = first_row + block * 210;
        const auto* base1 = second_row + block * 210;
        const auto* ql0 = base0;
        const auto* ql1 = base1;
        const auto* qh0 = base0 + 128;
        const auto* qh1 = base1 + 128;
        const auto* scales0 = reinterpret_cast<const std::int8_t*>(base0 + 192);
        const auto* scales1 = reinterpret_cast<const std::int8_t*>(base1 + 192);
        const float d0 = half_value(base0 + 208);
        const float d1 = half_value(base1 + 208);
        for (int half = 0; half < 2; ++half) {
            for (int segment = 0; segment < 4; ++segment) {
                const int q_offset = (segment == 0 || segment == 2) ? 0 : 32;
                for (int lanes = 0; lanes < 32; lanes += 16) {
                    __m128i q0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                        ql0 + half * 64 + q_offset + lanes
                    ));
                    __m128i q1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                        ql1 + half * 64 + q_offset + lanes
                    ));
                    q0 = segment < 2 ? _mm_and_si128(q0, nibble_mask)
                                     : _mm_and_si128(_mm_srli_epi16(q0, 4), nibble_mask);
                    q1 = segment < 2 ? _mm_and_si128(q1, nibble_mask)
                                     : _mm_and_si128(_mm_srli_epi16(q1, 4), nibble_mask);
                    __m128i high0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                        qh0 + half * 32 + lanes
                    ));
                    __m128i high1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                        qh1 + half * 32 + lanes
                    ));
                    high0 = _mm_and_si128(
                        _mm_srli_epi16(high0, segment * 2), high_mask
                    );
                    high1 = _mm_and_si128(
                        _mm_srli_epi16(high1, segment * 2), high_mask
                    );
                    q0 = _mm_add_epi8(q0, _mm_slli_epi16(high0, 4));
                    q1 = _mm_add_epi8(q1, _mm_slli_epi16(high1, 4));
                    const int scale_index = half * 8 + lanes / 16 + segment * 2;
                    const __m512 weights0 = _mm512_mul_ps(
                        _mm512_sub_ps(bytes_to_float(q0), offset),
                        _mm512_set1_ps(d0 * scales0[scale_index])
                    );
                    const __m512 weights1 = _mm512_mul_ps(
                        _mm512_sub_ps(bytes_to_float(q1), offset),
                        _mm512_set1_ps(d1 * scales1[scale_index])
                    );
                    const int index = block * 256 + half * 128 + segment * 32 + lanes;
                    const __m512 values = _mm512_loadu_ps(input + index);
                    if (lanes == 0) {
                        a0 = _mm512_fmadd_ps(weights0, values, a0);
                        b0 = _mm512_fmadd_ps(weights1, values, b0);
                    } else {
                        a1 = _mm512_fmadd_ps(weights0, values, a1);
                        b1 = _mm512_fmadd_ps(weights1, values, b1);
                    }
                }
            }
        }
    }
    out_first = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    out_second = _mm512_reduce_add_ps(_mm512_add_ps(b0, b1));
}

void q8_dot_two_rows(
    const std::uint8_t* first_row,
    const std::uint8_t* second_row,
    const float* input,
    int elements,
    float& out_first,
    float& out_second
) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps();
    for (int block = 0; block < elements / 32; ++block) {
        const auto* base0 = first_row + block * 34;
        const auto* base1 = second_row + block * 34;
        const __m512 scale0 = _mm512_set1_ps(half_value(base0));
        const __m512 scale1 = _mm512_set1_ps(half_value(base1));
        const __m256i quantized0 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(base0 + 2)
        );
        const __m256i quantized1 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(base1 + 2)
        );
        const __m512 q00 = _mm512_mul_ps(
            _mm512_cvtepi32_ps(
                _mm512_cvtepi8_epi32(_mm256_castsi256_si128(quantized0))
            ),
            scale0
        );
        const __m512 q01 = _mm512_mul_ps(
            _mm512_cvtepi32_ps(
                _mm512_cvtepi8_epi32(_mm256_extracti128_si256(quantized0, 1))
            ),
            scale0
        );
        const __m512 q10 = _mm512_mul_ps(
            _mm512_cvtepi32_ps(
                _mm512_cvtepi8_epi32(_mm256_castsi256_si128(quantized1))
            ),
            scale1
        );
        const __m512 q11 = _mm512_mul_ps(
            _mm512_cvtepi32_ps(
                _mm512_cvtepi8_epi32(_mm256_extracti128_si256(quantized1, 1))
            ),
            scale1
        );
        const __m512 values0 = _mm512_loadu_ps(input + block * 32);
        const __m512 values1 = _mm512_loadu_ps(input + block * 32 + 16);
        a0 = _mm512_fmadd_ps(q00, values0, a0);
        a1 = _mm512_fmadd_ps(q01, values1, a1);
        b0 = _mm512_fmadd_ps(q10, values0, b0);
        b1 = _mm512_fmadd_ps(q11, values1, b1);
    }
    out_first = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    out_second = _mm512_reduce_add_ps(_mm512_add_ps(b0, b1));
}

void q5_dot_quad(const std::uint8_t*row_data,const float*const inputs[4],int elements,float outputs[4]){
    __m512 sums[4][2];for(auto&pair:sums)for(auto&sum:pair)sum=_mm512_setzero_ps();
    const __m128i nibble_mask=_mm_set1_epi8(15),bit_mask=_mm_set1_epi8(1);
    for(int block=0;block<elements/256;++block){const auto*base=row_data+block*176;const float d=half_value(base),dmin=half_value(base+2);const auto*scales=base+4;const auto*high=base+16;const auto*low=base+48;
        for(int group=0;group<4;++group)for(int sub=0;sub<2;++sub){const int index=group*2+sub;int scale,minimum;if(index<4){scale=scales[index]&63;minimum=scales[index+4]&63;}else{scale=(scales[index+4]&15)|((scales[index-4]>>6)<<4);minimum=(scales[index+4]>>4)|((scales[index]>>6)<<4);}const __m512 ds=_mm512_set1_ps(d*scale),dm=_mm512_set1_ps(dmin*minimum);const int shift=2*group+sub,offset=block*256+group*64+sub*32;
            for(int lanes=0;lanes<32;lanes+=16){__m128i q=_mm_loadu_si128(reinterpret_cast<const __m128i*>(low+group*32+lanes));q=sub==0?_mm_and_si128(q,nibble_mask):_mm_and_si128(_mm_srli_epi16(q,4),nibble_mask);__m128i bits=_mm_loadu_si128(reinterpret_cast<const __m128i*>(high+lanes));bits=_mm_and_si128(_mm_srli_epi16(bits,shift),bit_mask);q=_mm_add_epi8(q,_mm_slli_epi16(bits,4));const __m512 weights=_mm512_sub_ps(_mm512_mul_ps(bytes_to_float(q),ds),dm);for(int token=0;token<4;++token)sums[token][lanes/16]=_mm512_fmadd_ps(weights,_mm512_loadu_ps(inputs[token]+offset+lanes),sums[token][lanes/16]);}
        }
    }
    for(int token=0;token<4;++token)outputs[token]=_mm512_reduce_add_ps(_mm512_add_ps(sums[token][0],sums[token][1]));
}

void q6_dot_quad(const std::uint8_t*row_data,const float*const inputs[4],int elements,float outputs[4]){
    __m512 sums[4][2];for(auto&pair:sums)for(auto&sum:pair)sum=_mm512_setzero_ps();const __m128i nibble_mask=_mm_set1_epi8(15),high_mask=_mm_set1_epi8(3);const __m512 offset32=_mm512_set1_ps(32.0f);
    for(int block=0;block<elements/256;++block){const auto*base=row_data+block*210;const auto*ql=base;const auto*qh=base+128;const auto*scales=reinterpret_cast<const std::int8_t*>(base+192);const float d=half_value(base+208);
        for(int half=0;half<2;++half)for(int segment=0;segment<4;++segment){const int q_offset=(segment==0||segment==2)?0:32;
            for(int lanes=0;lanes<32;lanes+=16){__m128i q=_mm_loadu_si128(reinterpret_cast<const __m128i*>(ql+half*64+q_offset+lanes));q=segment<2?_mm_and_si128(q,nibble_mask):_mm_and_si128(_mm_srli_epi16(q,4),nibble_mask);__m128i high=_mm_loadu_si128(reinterpret_cast<const __m128i*>(qh+half*32+lanes));high=_mm_and_si128(_mm_srli_epi16(high,segment*2),high_mask);q=_mm_add_epi8(q,_mm_slli_epi16(high,4));const int scale_index=half*8+lanes/16+segment*2,index=block*256+half*128+segment*32+lanes;const __m512 weights=_mm512_mul_ps(_mm512_sub_ps(bytes_to_float(q),offset32),_mm512_set1_ps(d*scales[scale_index]));for(int token=0;token<4;++token)sums[token][lanes/16]=_mm512_fmadd_ps(weights,_mm512_loadu_ps(inputs[token]+index),sums[token][lanes/16]);}
        }
    }
    for(int token=0;token<4;++token)outputs[token]=_mm512_reduce_add_ps(_mm512_add_ps(sums[token][0],sums[token][1]));
}

void q8_dot_quad(const std::uint8_t*row_data,const float*const inputs[4],int elements,float outputs[4]){
    __m512 sums[4][2];for(auto&pair:sums)for(auto&sum:pair)sum=_mm512_setzero_ps();
    for(int block=0;block<elements/32;++block){const auto*base=row_data+block*34;const __m512 scale=_mm512_set1_ps(half_value(base));const __m256i quantized=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(base+2));const __m512 weights[2]={_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_castsi256_si128(quantized))),scale),_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_extracti128_si256(quantized,1))),scale)};for(int token=0;token<4;++token)for(int half=0;half<2;++half)sums[token][half]=_mm512_fmadd_ps(weights[half],_mm512_loadu_ps(inputs[token]+block*32+half*16),sums[token][half]);}
    for(int token=0;token<4;++token)outputs[token]=_mm512_reduce_add_ps(_mm512_add_ps(sums[token][0],sums[token][1]));
}

void q5_dot_oct(const std::uint8_t*row_data,const float*const inputs[8],int elements,float outputs[8]){
    __m512 sums[8][2];for(auto&pair:sums)for(auto&sum:pair)sum=_mm512_setzero_ps();
    const __m128i nibble_mask=_mm_set1_epi8(15),bit_mask=_mm_set1_epi8(1);
    for(int block=0;block<elements/256;++block){const auto*base=row_data+block*176;const float d=half_value(base),dmin=half_value(base+2);const auto*scales=base+4;const auto*high=base+16;const auto*low=base+48;
        for(int group=0;group<4;++group)for(int sub=0;sub<2;++sub){const int index=group*2+sub;int scale,minimum;if(index<4){scale=scales[index]&63;minimum=scales[index+4]&63;}else{scale=(scales[index+4]&15)|((scales[index-4]>>6)<<4);minimum=(scales[index+4]>>4)|((scales[index]>>6)<<4);}const __m512 ds=_mm512_set1_ps(d*scale),dm=_mm512_set1_ps(dmin*minimum);const int shift=2*group+sub,offset=block*256+group*64+sub*32;
            for(int lanes=0;lanes<32;lanes+=16){__m128i q=_mm_loadu_si128(reinterpret_cast<const __m128i*>(low+group*32+lanes));q=sub==0?_mm_and_si128(q,nibble_mask):_mm_and_si128(_mm_srli_epi16(q,4),nibble_mask);__m128i bits=_mm_loadu_si128(reinterpret_cast<const __m128i*>(high+lanes));bits=_mm_and_si128(_mm_srli_epi16(bits,shift),bit_mask);q=_mm_add_epi8(q,_mm_slli_epi16(bits,4));const __m512 weights=_mm512_sub_ps(_mm512_mul_ps(bytes_to_float(q),ds),dm);for(int token=0;token<8;++token)sums[token][lanes/16]=_mm512_fmadd_ps(weights,_mm512_loadu_ps(inputs[token]+offset+lanes),sums[token][lanes/16]);}
        }
    }
    for(int token=0;token<8;++token)outputs[token]=_mm512_reduce_add_ps(_mm512_add_ps(sums[token][0],sums[token][1]));
}

void q6_dot_oct(const std::uint8_t*row_data,const float*const inputs[8],int elements,float outputs[8]){
    __m512 sums[8][2];for(auto&pair:sums)for(auto&sum:pair)sum=_mm512_setzero_ps();const __m128i nibble_mask=_mm_set1_epi8(15),high_mask=_mm_set1_epi8(3);const __m512 offset32=_mm512_set1_ps(32.0f);
    for(int block=0;block<elements/256;++block){const auto*base=row_data+block*210;const auto*ql=base;const auto*qh=base+128;const auto*scales=reinterpret_cast<const std::int8_t*>(base+192);const float d=half_value(base+208);
        for(int half=0;half<2;++half)for(int segment=0;segment<4;++segment){const int q_offset=(segment==0||segment==2)?0:32;
            for(int lanes=0;lanes<32;lanes+=16){__m128i q=_mm_loadu_si128(reinterpret_cast<const __m128i*>(ql+half*64+q_offset+lanes));q=segment<2?_mm_and_si128(q,nibble_mask):_mm_and_si128(_mm_srli_epi16(q,4),nibble_mask);__m128i high=_mm_loadu_si128(reinterpret_cast<const __m128i*>(qh+half*32+lanes));high=_mm_and_si128(_mm_srli_epi16(high,segment*2),high_mask);q=_mm_add_epi8(q,_mm_slli_epi16(high,4));const int scale_index=half*8+lanes/16+segment*2,index=block*256+half*128+segment*32+lanes;const __m512 weights=_mm512_mul_ps(_mm512_sub_ps(bytes_to_float(q),offset32),_mm512_set1_ps(d*scales[scale_index]));for(int token=0;token<8;++token)sums[token][lanes/16]=_mm512_fmadd_ps(weights,_mm512_loadu_ps(inputs[token]+index),sums[token][lanes/16]);}
        }
    }
    for(int token=0;token<8;++token)outputs[token]=_mm512_reduce_add_ps(_mm512_add_ps(sums[token][0],sums[token][1]));
}

void q8_dot_oct(const std::uint8_t*row_data,const float*const inputs[8],int elements,float outputs[8]){
    __m512 sums[8][2];for(auto&pair:sums)for(auto&sum:pair)sum=_mm512_setzero_ps();
    for(int block=0;block<elements/32;++block){const auto*base=row_data+block*34;const __m512 scale=_mm512_set1_ps(half_value(base));const __m256i quantized=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(base+2));const __m512 weights[2]={_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_castsi256_si128(quantized))),scale),_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_extracti128_si256(quantized,1))),scale)};for(int token=0;token<8;++token)for(int half=0;half<2;++half)sums[token][half]=_mm512_fmadd_ps(weights[half],_mm512_loadu_ps(inputs[token]+block*32+half*16),sums[token][half]);}
    for(int token=0;token<8;++token)outputs[token]=_mm512_reduce_add_ps(_mm512_add_ps(sums[token][0],sums[token][1]));
}

void q5_dequant(const std::uint8_t* row_data, float* output, int elements) {
    const __m128i nibble_mask = _mm_set1_epi8(15);
    const __m128i bit_mask = _mm_set1_epi8(1);
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * 176;
        const float d = half_value(base);
        const float dmin = half_value(base + 2);
        const auto* scales = base + 4;
        const auto* high = base + 16;
        const auto* low = base + 48;
        float* destination = output + block * 256;
        for (int group = 0; group < 4; ++group) {
            for (int sub = 0; sub < 2; ++sub) {
                const int scale_index = group * 2 + sub;
                int scale;
                int minimum;
                if (scale_index < 4) {
                    scale = scales[scale_index] & 63;
                    minimum = scales[scale_index + 4] & 63;
                } else {
                    scale = (scales[scale_index + 4] & 15)
                        | ((scales[scale_index - 4] >> 6) << 4);
                    minimum = (scales[scale_index + 4] >> 4)
                        | ((scales[scale_index] >> 6) << 4);
                }
                const __m512 ds = _mm512_set1_ps(d * scale);
                const __m512 dm = _mm512_set1_ps(dmin * minimum);
                const int shift = 2 * group + sub;
                for (int lanes = 0; lanes < 32; lanes += 16) {
                    __m128i q = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(low + group * 32 + lanes)
                    );
                    q = sub == 0 ? _mm_and_si128(q, nibble_mask)
                                 : _mm_and_si128(_mm_srli_epi16(q, 4), nibble_mask);
                    __m128i bits = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(high + lanes)
                    );
                    bits = _mm_and_si128(_mm_srli_epi16(bits, shift), bit_mask);
                    q = _mm_add_epi8(q, _mm_slli_epi16(bits, 4));
                    _mm512_storeu_ps(
                        destination + group * 64 + sub * 32 + lanes,
                        _mm512_sub_ps(_mm512_mul_ps(bytes_to_float(q), ds), dm)
                    );
                }
            }
        }
    }
}

void q6_dequant(const std::uint8_t* row_data, float* output, int elements) {
    const __m128i nibble_mask = _mm_set1_epi8(15);
    const __m128i high_mask = _mm_set1_epi8(3);
    const __m512 offset = _mm512_set1_ps(32.0f);
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * 210;
        const auto* ql = base;
        const auto* qh = base + 128;
        const auto* scales = reinterpret_cast<const std::int8_t*>(base + 192);
        const float d = half_value(base + 208);
        float* destination = output + block * 256;
        for (int half = 0; half < 2; ++half) {
            for (int segment = 0; segment < 4; ++segment) {
                const int q_offset = (segment == 0 || segment == 2) ? 0 : 32;
                for (int lanes = 0; lanes < 32; lanes += 16) {
                    __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                        ql + half * 64 + q_offset + lanes
                    ));
                    q = segment < 2 ? _mm_and_si128(q, nibble_mask)
                                    : _mm_and_si128(_mm_srli_epi16(q, 4), nibble_mask);
                    __m128i high = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                        qh + half * 32 + lanes
                    ));
                    high = _mm_and_si128(
                        _mm_srli_epi16(high, segment * 2), high_mask
                    );
                    q = _mm_add_epi8(q, _mm_slli_epi16(high, 4));
                    const int scale_index = half * 8 + lanes / 16 + segment * 2;
                    const __m512 factor = _mm512_set1_ps(d * scales[scale_index]);
                    _mm512_storeu_ps(
                        destination + half * 128 + segment * 32 + lanes,
                        _mm512_mul_ps(
                            _mm512_sub_ps(bytes_to_float(q), offset), factor
                        )
                    );
                }
            }
        }
    }
}

void q8_dequant(const std::uint8_t* row_data, float* output, int elements) {
    for (int block = 0; block < elements / 32; ++block) {
        const auto* base = row_data + block * 34;
        const __m512 scale = _mm512_set1_ps(half_value(base));
        const __m256i quantized = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(base + 2)
        );
        _mm512_storeu_ps(
            output + block * 32,
            _mm512_mul_ps(_mm512_cvtepi32_ps(
                _mm512_cvtepi8_epi32(_mm256_castsi256_si128(quantized))
            ), scale)
        );
        _mm512_storeu_ps(
            output + block * 32 + 16,
            _mm512_mul_ps(_mm512_cvtepi32_ps(
                _mm512_cvtepi8_epi32(_mm256_extracti128_si256(quantized, 1))
            ), scale)
        );
    }
}

} // namespace

void qwen_quant_dot_two_rows_avx512(
    const std::uint8_t* first_row,
    const std::uint8_t* second_row,
    std::uint32_t type,
    const float* input,
    int elements,
    float* first_output,
    float* second_output
) {
    if (type == 13) {
        q5_dot_two_rows(first_row, second_row, input, elements, *first_output, *second_output);
    } else if (type == 14) {
        q6_dot_two_rows(first_row, second_row, input, elements, *first_output, *second_output);
    } else {
        q8_dot_two_rows(first_row, second_row, input, elements, *first_output, *second_output);
    }
}

float qwen_quant_dot_avx512(
    const std::uint8_t* packed,
    std::uint32_t type,
    const float* input,
    int elements,
    std::uint64_t row
) {
    if (type == 13) {
        return q5_dot(packed + row * static_cast<std::uint64_t>(elements / 256) * 176,
                      input, elements);
    }
    if (type == 14) {
        return q6_dot(packed + row * static_cast<std::uint64_t>(elements / 256) * 210,
                      input, elements);
    }
    return q8_dot(packed + row * static_cast<std::uint64_t>(elements / 32) * 34,
                  input, elements);
}

void qwen_quant_dot_pair_avx512(
    const std::uint8_t*packed,std::uint32_t type,const float*first,
    const float*second,int elements,std::uint64_t row,
    float*first_output,float*second_output
){
    if(type==13)q5_dot_pair(packed+row*static_cast<std::uint64_t>(elements/256)*176,first,second,elements,*first_output,*second_output);
    else if(type==14)q6_dot_pair(packed+row*static_cast<std::uint64_t>(elements/256)*210,first,second,elements,*first_output,*second_output);
    else q8_dot_pair(packed+row*static_cast<std::uint64_t>(elements/32)*34,first,second,elements,*first_output,*second_output);
}

void qwen_quant_dot_quad_avx512(
    const std::uint8_t*packed,std::uint32_t type,const float*const inputs[4],
    int elements,std::uint64_t row,float outputs[4]
){
    if(type==13)q5_dot_quad(packed+row*static_cast<std::uint64_t>(elements/256)*176,inputs,elements,outputs);
    else if(type==14)q6_dot_quad(packed+row*static_cast<std::uint64_t>(elements/256)*210,inputs,elements,outputs);
    else q8_dot_quad(packed+row*static_cast<std::uint64_t>(elements/32)*34,inputs,elements,outputs);
}

void qwen_quant_dot_oct_avx512(
    const std::uint8_t*packed,std::uint32_t type,const float*const inputs[8],
    int elements,std::uint64_t row,float outputs[8]
){
    if(type==13)q5_dot_oct(packed+row*static_cast<std::uint64_t>(elements/256)*176,inputs,elements,outputs);
    else if(type==14)q6_dot_oct(packed+row*static_cast<std::uint64_t>(elements/256)*210,inputs,elements,outputs);
    else q8_dot_oct(packed+row*static_cast<std::uint64_t>(elements/32)*34,inputs,elements,outputs);
}

void qwen_dequant_row_avx512(
    const std::uint8_t* packed,
    std::uint32_t type,
    int elements,
    std::uint64_t row,
    float* output
) {
    if (type == 13) {
        q5_dequant(packed + row * static_cast<std::uint64_t>(elements / 256) * 176,
                   output, elements);
    } else if (type == 14) {
        q6_dequant(packed + row * static_cast<std::uint64_t>(elements / 256) * 210,
                   output, elements);
    } else {
        q8_dequant(packed + row * static_cast<std::uint64_t>(elements / 32) * 34,
                   output, elements);
    }
}

void qwen_f32_dot_multi_avx512(
    const float* row,
    const float* const* inputs,
    int count,
    int elements,
    float* outputs
) {
    int token = 0;
    for (; token + 4 <= count; token += 4) {
        const float* a = inputs[token];
        const float* b = inputs[token + 1];
        const float* c = inputs[token + 2];
        const float* d = inputs[token + 3];
        __m512 sum_a = _mm512_setzero_ps(), sum_b = _mm512_setzero_ps();
        __m512 sum_c = _mm512_setzero_ps(), sum_d = _mm512_setzero_ps();
        for (int index = 0; index < elements; index += 16) {
            const __m512 weights = _mm512_loadu_ps(row + index);
            sum_a = _mm512_fmadd_ps(weights, _mm512_loadu_ps(a + index), sum_a);
            sum_b = _mm512_fmadd_ps(weights, _mm512_loadu_ps(b + index), sum_b);
            sum_c = _mm512_fmadd_ps(weights, _mm512_loadu_ps(c + index), sum_c);
            sum_d = _mm512_fmadd_ps(weights, _mm512_loadu_ps(d + index), sum_d);
        }
        outputs[token] = _mm512_reduce_add_ps(sum_a);
        outputs[token + 1] = _mm512_reduce_add_ps(sum_b);
        outputs[token + 2] = _mm512_reduce_add_ps(sum_c);
        outputs[token + 3] = _mm512_reduce_add_ps(sum_d);
    }
    for (; token < count; ++token) {
        const float* vector = inputs[token];
        __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps();
        for (int index = 0; index < elements; index += 32) {
            sum0 = _mm512_fmadd_ps(
                _mm512_loadu_ps(row + index), _mm512_loadu_ps(vector + index), sum0
            );
            sum1 = _mm512_fmadd_ps(
                _mm512_loadu_ps(row + index + 16),
                _mm512_loadu_ps(vector + index + 16), sum1
            );
        }
        outputs[token] = _mm512_reduce_add_ps(_mm512_add_ps(sum0, sum1));
    }
}

// Register-blocked expert GEMM: out[i*count + j] = dot(weights[i], inputs[j]).
// A 4-weight-row x 4-token tile keeps 16 accumulators live (constant loop bounds
// so they stay in zmm registers) and reuses each activation load across 4 rows
// and each weight load across 4 tokens, so the ~256 KB activation block is read
// from L2 count/4 times instead of once per output row. ~2-2.8x over the
// GEMV-per-row path (768x2048, 32 tokens: 82 -> 175/227 GFLOP/s single-thread).
void qwen_f32_gemm_rows_avx512(
    const float* weights, int mr, const float* const* inputs,
    int count, int elements, float* out
) {
    if (mr == 4) {
        int j = 0;
        for (; j + 4 <= count; j += 4) {
            const float* x[4] = {inputs[j], inputs[j + 1], inputs[j + 2], inputs[j + 3]};
            __m512 acc[4][4];
            for (int i = 0; i < 4; ++i)
                for (int t = 0; t < 4; ++t) acc[i][t] = _mm512_setzero_ps();
            for (int k = 0; k < elements; k += 16) {
                const __m512 xv[4] = {
                    _mm512_loadu_ps(x[0] + k), _mm512_loadu_ps(x[1] + k),
                    _mm512_loadu_ps(x[2] + k), _mm512_loadu_ps(x[3] + k)
                };
                for (int i = 0; i < 4; ++i) {
                    const __m512 w = _mm512_loadu_ps(weights + i * elements + k);
                    for (int t = 0; t < 4; ++t)
                        acc[i][t] = _mm512_fmadd_ps(w, xv[t], acc[i][t]);
                }
            }
            for (int i = 0; i < 4; ++i)
                for (int t = 0; t < 4; ++t)
                    out[i * count + j + t] = _mm512_reduce_add_ps(acc[i][t]);
        }
        for (; j < count; ++j) {
            const float* v = inputs[j];
            __m512 acc[4];
            for (int i = 0; i < 4; ++i) acc[i] = _mm512_setzero_ps();
            for (int k = 0; k < elements; k += 16) {
                const __m512 xv = _mm512_loadu_ps(v + k);
                for (int i = 0; i < 4; ++i)
                    acc[i] = _mm512_fmadd_ps(_mm512_loadu_ps(weights + i * elements + k), xv, acc[i]);
            }
            for (int i = 0; i < 4; ++i) out[i * count + j] = _mm512_reduce_add_ps(acc[i]);
        }
        return;
    }
    // Row-block tail (mr < 4): fall back to the single-row multi-token dot.
    for (int i = 0; i < mr; ++i)
        qwen_f32_dot_multi_avx512(
            weights + static_cast<std::size_t>(i) * elements, inputs, count,
            elements, out + static_cast<std::size_t>(i) * count
        );
}
