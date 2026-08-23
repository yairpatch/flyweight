#include <qwen_cpu_kernel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

#include "qwen_kquant.h"

namespace {

float half_value(const std::uint8_t* pointer) {
    std::uint16_t bits;
    std::memcpy(&bits, pointer, sizeof(bits));
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}

int horizontal_sum(__m256i value) {
    __m128i sum = _mm_add_epi32(
        _mm256_castsi256_si128(value), _mm256_extracti128_si256(value, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

int dot_i8_8(__m128i left, __m128i right) {
    return horizontal_sum(_mm256_dpwssd_epi32(
        _mm256_setzero_si256(),
        _mm256_cvtepi8_epi16(left),
        _mm256_cvtepi8_epi16(right)));
}

int dot_i8_16(const std::int8_t* left, const std::int8_t* right) {
    const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(left));
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(right));
    return horizontal_sum(_mm256_dpwssd_epi32(
        _mm256_setzero_si256(),
        _mm256_cvtepi8_epi16(a),
        _mm256_cvtepi8_epi16(b)));
}

struct IqSignBytes {
    std::int8_t lanes[256][8];
};

constexpr IqSignBytes build_iq_sign_bytes() {
    IqSignBytes signs{};
    for (int pattern = 0; pattern < 256; ++pattern)
        for (int lane = 0; lane < 8; ++lane)
            signs.lanes[pattern][lane] =
                (pattern >> lane) & 1 ? -1 : 1;
    return signs;
}

constexpr IqSignBytes kIqSignBytes = build_iq_sign_bytes();

std::int64_t iq_sign_bytes(std::uint8_t pattern) {
    std::int64_t result;
    std::memcpy(&result, kIqSignBytes.lanes[pattern], sizeof(result));
    return result;
}

} // namespace

void qwen_quantize_q8_0(
    const float* input, int elements, QwenQ80Block* output
) {
    for (int block = 0; block < elements / 32; ++block) {
        const float* source = input + block * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i)
            amax = std::max(amax, std::fabs(source[i]));
        const float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
        output[block].scale = amax / 127.0f;
        for (int i = 0; i < 32; ++i)
            output[block].values[i] =
                static_cast<std::int8_t>(std::lrintf(source[i] * inverse));
    }
}

float qwen_quant_dot_q4_0_q8_0_vnni(
    const std::uint8_t* packed, const QwenQ80Block* input,
    int elements, std::uint64_t row
) {
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(elements / 32) * 18;
    const __m128i nibble_mask = _mm_set1_epi8(15);
    const __m256i zero_point = _mm256_set1_epi8(8);
    __m256 acc = _mm256_setzero_ps();
    for (int block = 0; block < elements / 32; ++block) {
        const auto* base = row_data + block * 18;
        _mm_prefetch(reinterpret_cast<const char*>(base + 1024), _MM_HINT_T0);
        const __m128i bytes = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(base + 2));
        const __m128i low = _mm_and_si128(bytes, nibble_mask);
        const __m128i high = _mm_and_si128(
            _mm_srli_epi16(bytes, 4), nibble_mask);
        // GGML Q4_0 order: values 0..15 are the low nibbles, 16..31 the high.
        const __m256i quantized = _mm256_sub_epi8(
            _mm256_set_m128i(high, low), zero_point);
        const __m256i activation = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(input[block].values));
        // dpbusd wants an unsigned left operand, so move the weight signs onto
        // the activations: |q4| * sign(q8, q4) == q4 * q8 lane for lane, and
        // 8 * 127 * 4 stays far from the i32 accumulator's edge.
        const __m256i dots = _mm256_dpbusd_epi32(
            _mm256_setzero_si256(),
            _mm256_sign_epi8(quantized, quantized),
            _mm256_sign_epi8(activation, quantized));
        acc = _mm256_fmadd_ps(
            _mm256_set1_ps(half_value(base) * input[block].scale),
            _mm256_cvtepi32_ps(dots), acc);
    }
    __m128 sum = _mm_add_ps(
        _mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

namespace {

template <int Count>
void q4_0_q8_0_multi_vnni(
    const std::uint8_t* packed, const QwenQ80Block* const* inputs,
    int elements, std::uint64_t row, float* outputs
) {
    const auto* row_data =
        packed + row * static_cast<std::uint64_t>(elements / 32) * 18;
    const __m128i nibble_mask = _mm_set1_epi8(15);
    const __m256i zero_point = _mm256_set1_epi8(8);
    __m256 acc[Count];
    for (int t = 0; t < Count; ++t) acc[t] = _mm256_setzero_ps();
    for (int block = 0; block < elements / 32; ++block) {
        const auto* base = row_data + block * 18;
        _mm_prefetch(reinterpret_cast<const char*>(base + 1024), _MM_HINT_T0);
        const __m128i bytes = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(base + 2));
        const __m128i low = _mm_and_si128(bytes, nibble_mask);
        const __m128i high = _mm_and_si128(
            _mm_srli_epi16(bytes, 4), nibble_mask);
        const __m256i quantized = _mm256_sub_epi8(
            _mm256_set_m128i(high, low), zero_point);
        const __m256i magnitudes = _mm256_sign_epi8(quantized, quantized);
        const float d4 = half_value(base);
        for (int t = 0; t < Count; ++t) {
            const auto& q8 = inputs[t][block];
            const __m256i activation = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(q8.values));
            const __m256i dots = _mm256_dpbusd_epi32(
                _mm256_setzero_si256(), magnitudes,
                _mm256_sign_epi8(activation, quantized));
            acc[t] = _mm256_fmadd_ps(
                _mm256_set1_ps(d4 * q8.scale),
                _mm256_cvtepi32_ps(dots), acc[t]);
        }
    }
    for (int t = 0; t < Count; ++t) {
        __m128 sum = _mm_add_ps(
            _mm256_castps256_ps128(acc[t]), _mm256_extractf128_ps(acc[t], 1));
        sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
        sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
        outputs[t] = _mm_cvtss_f32(sum);
    }
}

} // namespace

void qwen_quant_dot_q4_0_q8_0_multi_vnni(
    const std::uint8_t* packed, const QwenQ80Block* const* inputs,
    int count, int elements, std::uint64_t row, float* outputs
) {
    switch (count) {
    case 8: q4_0_q8_0_multi_vnni<8>(packed, inputs, elements, row, outputs); return;
    case 7: q4_0_q8_0_multi_vnni<7>(packed, inputs, elements, row, outputs); return;
    case 6: q4_0_q8_0_multi_vnni<6>(packed, inputs, elements, row, outputs); return;
    case 5: q4_0_q8_0_multi_vnni<5>(packed, inputs, elements, row, outputs); return;
    case 4: q4_0_q8_0_multi_vnni<4>(packed, inputs, elements, row, outputs); return;
    case 3: q4_0_q8_0_multi_vnni<3>(packed, inputs, elements, row, outputs); return;
    case 2: q4_0_q8_0_multi_vnni<2>(packed, inputs, elements, row, outputs); return;
    case 1: q4_0_q8_0_multi_vnni<1>(packed, inputs, elements, row, outputs); return;
    default: return;
    }
}

float qwen_quant_dot_q8_k_avx_vnni(
    const std::uint8_t* packed, std::uint32_t type,
    const QwenQ8KBlock* input, int elements, std::uint64_t row
) {
    float result=0.0f;
    if(type==17){
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/256)*kIq2xsBlockBytes;
        for(int block=0;block<elements/256;++block){
            const auto*base=row_data+block*kIq2xsBlockBytes;const auto&q8=input[block];
            __m256i sum=_mm256_setzero_si256();
            for(int group=0;group<16;group+=2){
                std::uint16_t codes[4];
                std::memcpy(codes,base+2+group*4,sizeof(codes));
                const __m256i magnitudes=_mm256_set_epi64x(
                    static_cast<long long>(kIq2xsGrid[codes[3]&511]),
                    static_cast<long long>(kIq2xsGrid[codes[2]&511]),
                    static_cast<long long>(kIq2xsGrid[codes[1]&511]),
                    static_cast<long long>(kIq2xsGrid[codes[0]&511]));
                const __m256i signs=_mm256_set_epi64x(
                    iq_sign_bytes(kIq2xxsSigns[codes[3]>>9]),
                    iq_sign_bytes(kIq2xxsSigns[codes[2]>>9]),
                    iq_sign_bytes(kIq2xxsSigns[codes[1]>>9]),
                    iq_sign_bytes(kIq2xxsSigns[codes[0]>>9]));
                const __m256i activation=_mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(q8.values+group*16));
                const __m256i dots=_mm256_dpbusd_epi32(
                    _mm256_setzero_si256(),magnitudes,
                    _mm256_sign_epi8(activation,signs));
                const int first=2*((base[66+(group>>1)]>>(4*(group&1)))&15)+1;
                const int second=2*((base[66+((group+1)>>1)]>>(4*((group+1)&1)))&15)+1;
                const __m256i scales=_mm256_set_m128i(
                    _mm_set1_epi32(second),_mm_set1_epi32(first));
                sum=_mm256_add_epi32(sum,_mm256_mullo_epi32(dots,scales));
            }
            result+=half_value(base)*q8.scale*0.125f*horizontal_sum(sum);
        }
    }else if(type==18){
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/256)*kIq3xxsBlockBytes;
        for(int block=0;block<elements/256;++block){
            const auto*base=row_data+block*kIq3xxsBlockBytes;const auto&q8=input[block];
            __m256i sum=_mm256_setzero_si256();
            for(int group=0;group<8;++group){
                std::uint32_t aux;
                std::memcpy(&aux,base+66+group*4,sizeof(aux));
                const auto*code=base+2+group*8;
                const __m256i magnitudes=_mm256_set_epi32(
                    static_cast<int>(kIq3xxsGrid[code[7]]),
                    static_cast<int>(kIq3xxsGrid[code[6]]),
                    static_cast<int>(kIq3xxsGrid[code[5]]),
                    static_cast<int>(kIq3xxsGrid[code[4]]),
                    static_cast<int>(kIq3xxsGrid[code[3]]),
                    static_cast<int>(kIq3xxsGrid[code[2]]),
                    static_cast<int>(kIq3xxsGrid[code[1]]),
                    static_cast<int>(kIq3xxsGrid[code[0]]));
                const __m256i signs=_mm256_set_epi64x(
                    iq_sign_bytes(kIq2xxsSigns[(aux>>21)&127]),
                    iq_sign_bytes(kIq2xxsSigns[(aux>>14)&127]),
                    iq_sign_bytes(kIq2xxsSigns[(aux>>7)&127]),
                    iq_sign_bytes(kIq2xxsSigns[aux&127]));
                const __m256i activation=_mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(q8.values+group*32));
                const __m256i dots=_mm256_dpbusd_epi32(
                    _mm256_setzero_si256(),magnitudes,
                    _mm256_sign_epi8(activation,signs));
                sum=_mm256_add_epi32(sum,_mm256_mullo_epi32(
                    dots,_mm256_set1_epi32(2*(aux>>28)+1)));
            }
            result+=half_value(base)*q8.scale*0.25f*horizontal_sum(sum);
        }
    }else if(type==13){
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/256)*176;
        for(int block=0;block<elements/256;++block){
            const auto*base=row_data+block*176;const auto&q8=input[block];
            const float d=half_value(base),dmin=half_value(base+2);const auto*scales=base+4;
            const auto*high=base+16;const auto*low=base+48;
            for(int group=0;group<4;++group)for(int sub=0;sub<2;++sub){
                const int index=group*2+sub;int scale,minimum;
                if(index<4){scale=scales[index]&63;minimum=scales[index+4]&63;}
                else{scale=(scales[index+4]&15)|((scales[index-4]>>6)<<4);minimum=(scales[index+4]>>4)|((scales[index]>>6)<<4);}
                int dot=0;const int offset=group*64+sub*32;
                for(int lanes=0;lanes<32;lanes+=8){
                    __m128i quant=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(low+group*32+lanes));
                    quant=sub==0?_mm_and_si128(quant,_mm_set1_epi8(15)):_mm_and_si128(_mm_srli_epi16(quant,4),_mm_set1_epi8(15));
                    __m128i bits=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(high+lanes));
                    bits=_mm_and_si128(_mm_srli_epi16(bits,2*group+sub),_mm_set1_epi8(1));
                    quant=_mm_add_epi8(quant,_mm_slli_epi16(bits,4));
                    const __m128i activation=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(q8.values+offset+lanes));
                    dot+=dot_i8_8(quant,activation);
                }
                const int activation_sum=q8.sums[offset/16]+q8.sums[offset/16+1];
                result+=q8.scale*(d*scale*dot-dmin*minimum*activation_sum);
            }
        }
    }else if(type==14){
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/256)*210;
        for(int block=0;block<elements/256;++block){
            const auto*base=row_data+block*210;const auto*low=base;const auto*high=base+128;
            const auto*scales=reinterpret_cast<const std::int8_t*>(base+192);const float factor=half_value(base+208)*input[block].scale;
            for(int half=0;half<2;++half)for(int segment=0;segment<4;++segment){
                const int q_offset=(segment==0||segment==2)?0:32;const int output_offset=half*128+segment*32;
                alignas(16) std::int8_t quant[32];
                for(int lane=0;lane<32;++lane){const auto byte=low[half*64+q_offset+lane];const int nibble=segment<2?(byte&15):(byte>>4);quant[lane]=static_cast<std::int8_t>((nibble|(((high[half*32+lane]>>(segment*2))&3)<<4))-32);}
                for(int lanes=0;lanes<32;lanes+=16){const int scale_index=half*8+lanes/16+segment*2;result+=factor*scales[scale_index]*dot_i8_16(quant+lanes,input[block].values+output_offset+lanes);}
            }
        }
    }else{
        const auto*row_data=packed+row*static_cast<std::uint64_t>(elements/32)*34;
        for(int block=0;block<elements/32;++block){const auto*base=row_data+block*34;const int super=block/8,offset=(block%8)*32;int dot=0;for(int lanes=0;lanes<32;lanes+=16)dot+=dot_i8_16(reinterpret_cast<const std::int8_t*>(base+2+lanes),input[super].values+offset+lanes);result+=half_value(base)*input[super].scale*dot;}
    }
    return result;
}
