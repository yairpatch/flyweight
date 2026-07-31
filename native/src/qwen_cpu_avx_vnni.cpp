#include <qwen_cpu_kernel.h>

#include <cstdint>
#include <cstring>
#include <immintrin.h>

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

} // namespace

float qwen_quant_dot_q8_k_avx_vnni(
    const std::uint8_t* packed, std::uint32_t type,
    const QwenQ8KBlock* input, int elements, std::uint64_t row
) {
    float result=0.0f;
    if(type==13){
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
