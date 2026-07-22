#include "qwen_cpu_kernel.h"

#include <cstddef>
#include <cstring>
#include <immintrin.h>

namespace {

float half_value(const std::uint8_t* pointer) {
    std::uint16_t bits;
    std::memcpy(&bits, pointer, sizeof(bits));
    return _cvtsh_ss(bits);
}

float horizontal_sum(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

__m256 bytes_to_float(__m128i values) {
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(values));
}

float q5_dot(const std::uint8_t* row_data, const float* input, int elements) {
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
    const __m128i nibble_mask = _mm_set1_epi8(15);
    const __m128i bit_mask = _mm_set1_epi8(1);
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * 176;
        const float d = half_value(base), dmin = half_value(base + 2);
        const auto* scales = base + 4;
        const auto* high = base + 16;
        const auto* low = base + 48;
        const auto* vector = input + block * 256;
        for (int group = 0; group < 4; ++group) {
            for (int sub = 0; sub < 2; ++sub) {
                const int index = group * 2 + sub;
                int scale, minimum;
                if (index < 4) {
                    scale = scales[index] & 63;
                    minimum = scales[index + 4] & 63;
                } else {
                    scale = (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4);
                    minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
                }
                const __m256 ds = _mm256_set1_ps(d * scale);
                const __m256 dm = _mm256_set1_ps(dmin * minimum);
                const int shift = 2 * group + sub;
                for (int lanes = 0; lanes < 32; lanes += 8) {
                    __m128i q = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
                        low + group * 32 + lanes));
                    q = sub == 0 ? _mm_and_si128(q, nibble_mask)
                                 : _mm_and_si128(_mm_srli_epi16(q, 4), nibble_mask);
                    __m128i bits = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(high + lanes));
                    bits = _mm_and_si128(_mm_srli_epi16(bits, shift), bit_mask);
                    q = _mm_add_epi8(q, _mm_slli_epi16(bits, 4));
                    const __m256 weights = _mm256_sub_ps(
                        _mm256_mul_ps(bytes_to_float(q), ds), dm);
                    const __m256 values = _mm256_loadu_ps(
                        vector + group * 64 + sub * 32 + lanes);
                    if ((lanes & 8) == 0) sum0 = _mm256_fmadd_ps(weights, values, sum0);
                    else sum1 = _mm256_fmadd_ps(weights, values, sum1);
                }
            }
        }
    }
    return horizontal_sum(_mm256_add_ps(sum0, sum1));
}

float q6_dot(const std::uint8_t* row_data, const float* input, int elements) {
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
    const __m128i nibble_mask = _mm_set1_epi8(15);
    const __m128i high_mask = _mm_set1_epi8(3);
    const __m256 offset = _mm256_set1_ps(32.0f);
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
                for (int lanes = 0; lanes < 32; lanes += 8) {
                    __m128i q = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
                        ql + half * 64 + q_offset + lanes));
                    q = segment < 2 ? _mm_and_si128(q, nibble_mask)
                                    : _mm_and_si128(_mm_srli_epi16(q, 4), nibble_mask);
                    __m128i high = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
                        qh + half * 32 + lanes));
                    high = _mm_and_si128(_mm_srli_epi16(high, segment * 2), high_mask);
                    q = _mm_add_epi8(q, _mm_slli_epi16(high, 4));
                    const int scale_index = half * 8 + lanes / 16 + segment * 2;
                    const __m256 factor = _mm256_set1_ps(d * scales[scale_index]);
                    const __m256 weights = _mm256_mul_ps(
                        _mm256_sub_ps(bytes_to_float(q), offset), factor);
                    const __m256 values = _mm256_loadu_ps(
                        vector + half * 128 + segment * 32 + lanes);
                    if ((lanes & 8) == 0) sum0 = _mm256_fmadd_ps(weights, values, sum0);
                    else sum1 = _mm256_fmadd_ps(weights, values, sum1);
                }
            }
        }
    }
    return horizontal_sum(_mm256_add_ps(sum0, sum1));
}

float q8_dot(const std::uint8_t* row_data, const float* input, int elements) {
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
    for (int block = 0; block < elements / 32; ++block) {
        const auto* base = row_data + block * 34;
        const __m256 scale = _mm256_set1_ps(half_value(base));
        for (int lanes = 0; lanes < 32; lanes += 8) {
            const __m128i bytes = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(base + 2 + lanes));
            const __m256 q = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
            const __m256 values = _mm256_loadu_ps(input + block * 32 + lanes);
            if ((lanes & 8) == 0) sum0 = _mm256_fmadd_ps(_mm256_mul_ps(q, scale), values, sum0);
            else sum1 = _mm256_fmadd_ps(_mm256_mul_ps(q, scale), values, sum1);
        }
    }
    return horizontal_sum(_mm256_add_ps(sum0, sum1));
}

void q5_dequant(const std::uint8_t* row_data, float* output, int elements) {
    const __m128i nibble_mask = _mm_set1_epi8(15), bit_mask = _mm_set1_epi8(1);
    for (int block = 0; block < elements / 256; ++block) {
        const auto* base = row_data + block * 176;
        const float d = half_value(base), dmin = half_value(base + 2);
        const auto* scales = base + 4; const auto* high = base + 16; const auto* low = base + 48;
        for (int group = 0; group < 4; ++group) for (int sub = 0; sub < 2; ++sub) {
            const int index=group*2+sub; int scale,minimum;
            if(index<4){scale=scales[index]&63;minimum=scales[index+4]&63;}
            else{scale=(scales[index+4]&15)|((scales[index-4]>>6)<<4);minimum=(scales[index+4]>>4)|((scales[index]>>6)<<4);}
            const __m256 ds=_mm256_set1_ps(d*scale),dm=_mm256_set1_ps(dmin*minimum);
            const int shift=2*group+sub;
            for(int lanes=0;lanes<32;lanes+=8){
                __m128i q=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(low+group*32+lanes));
                q=sub==0?_mm_and_si128(q,nibble_mask):_mm_and_si128(_mm_srli_epi16(q,4),nibble_mask);
                __m128i bits=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(high+lanes));
                bits=_mm_and_si128(_mm_srli_epi16(bits,shift),bit_mask);q=_mm_add_epi8(q,_mm_slli_epi16(bits,4));
                _mm256_storeu_ps(output+block*256+group*64+sub*32+lanes,_mm256_sub_ps(_mm256_mul_ps(bytes_to_float(q),ds),dm));
            }
        }
    }
}

void q6_dequant(const std::uint8_t* row_data, float* output, int elements) {
    const __m128i nibble_mask=_mm_set1_epi8(15),high_mask=_mm_set1_epi8(3);
    const __m256 offset=_mm256_set1_ps(32.0f);
    for(int block=0;block<elements/256;++block){
        const auto*base=row_data+block*210;const auto*ql=base;const auto*qh=base+128;
        const auto*scales=reinterpret_cast<const std::int8_t*>(base+192);const float d=half_value(base+208);
        for(int half=0;half<2;++half)for(int segment=0;segment<4;++segment){
            const int q_offset=(segment==0||segment==2)?0:32;
            for(int lanes=0;lanes<32;lanes+=8){
                __m128i q=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql+half*64+q_offset+lanes));
                q=segment<2?_mm_and_si128(q,nibble_mask):_mm_and_si128(_mm_srli_epi16(q,4),nibble_mask);
                __m128i high=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(qh+half*32+lanes));
                high=_mm_and_si128(_mm_srli_epi16(high,segment*2),high_mask);q=_mm_add_epi8(q,_mm_slli_epi16(high,4));
                const int scale_index=half*8+lanes/16+segment*2;const __m256 factor=_mm256_set1_ps(d*scales[scale_index]);
                _mm256_storeu_ps(output+block*256+half*128+segment*32+lanes,_mm256_mul_ps(_mm256_sub_ps(bytes_to_float(q),offset),factor));
            }
        }
    }
}

void q8_dequant(const std::uint8_t* row_data, float* output, int elements) {
    for(int block=0;block<elements/32;++block){const auto*base=row_data+block*34;const __m256 scale=_mm256_set1_ps(half_value(base));for(int lanes=0;lanes<32;lanes+=8){const __m128i bytes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(base+2+lanes));_mm256_storeu_ps(output+block*32+lanes,_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)),scale));}}
}

} // namespace

float qwen_quant_dot_avx2(const std::uint8_t* packed,std::uint32_t type,const float* input,int elements,std::uint64_t row){
    if(type==13)return q5_dot(packed+row*static_cast<std::uint64_t>(elements/256)*176,input,elements);
    if(type==14)return q6_dot(packed+row*static_cast<std::uint64_t>(elements/256)*210,input,elements);
    return q8_dot(packed+row*static_cast<std::uint64_t>(elements/32)*34,input,elements);
}

void qwen_dequant_row_avx2(const std::uint8_t* packed,std::uint32_t type,int elements,std::uint64_t row,float* output){
    if(type==13)q5_dequant(packed+row*static_cast<std::uint64_t>(elements/256)*176,output,elements);
    else if(type==14)q6_dequant(packed+row*static_cast<std::uint64_t>(elements/256)*210,output,elements);
    else q8_dequant(packed+row*static_cast<std::uint64_t>(elements/32)*34,output,elements);
}

void qwen_f32_dot_multi_avx2(const float* row,const float*const*inputs,int count,int elements,float*outputs){
    for(int token=0;token<count;++token){__m256 sum0=_mm256_setzero_ps(),sum1=_mm256_setzero_ps();for(int index=0;index<elements;index+=16){sum0=_mm256_fmadd_ps(_mm256_loadu_ps(row+index),_mm256_loadu_ps(inputs[token]+index),sum0);sum1=_mm256_fmadd_ps(_mm256_loadu_ps(row+index+8),_mm256_loadu_ps(inputs[token]+index+8),sum1);}outputs[token]=horizontal_sum(_mm256_add_ps(sum0,sum1));}
}

void qwen_f32_gemm_rows_avx2(const float*weights,int mr,const float*const*inputs,int count,int elements,float*out){
    for(int row=0;row<mr;++row)qwen_f32_dot_multi_avx2(weights+static_cast<std::size_t>(row)*elements,inputs,count,elements,out+static_cast<std::size_t>(row)*count);
}
