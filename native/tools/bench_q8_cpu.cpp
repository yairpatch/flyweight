#include <qwen_cpu_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#define NOINLINE __declspec(noinline)
#else
#include <cpuid.h>
#define NOINLINE __attribute__((noinline))
#endif

namespace {

bool has_avx_vnni(){
#if defined(_MSC_VER)
    int registers[4]{};
    __cpuid(registers,0);
    if(registers[0]<7)return false;
    __cpuidex(registers,7,0);
    if(registers[0]<1)return false;
    __cpuidex(registers,7,1);
    return (registers[0]&(1<<4))!=0;
#else
    unsigned eax=0,ebx=0,ecx=0,edx=0;
    if(__get_cpuid_max(0,nullptr)<7)return false;
    __cpuid_count(7,0,eax,ebx,ecx,edx);
    if(eax<1)return false;
    __cpuid_count(7,1,eax,ebx,ecx,edx);
    return (eax&(1u<<4))!=0;
#endif
}

bool has_avx512(){
#if defined(_MSC_VER)
    int registers[4]{};__cpuid(registers,1);if((registers[2]&(1<<27))==0)return false;
    if((_xgetbv(0)&0xe6)!=0xe6)return false;__cpuidex(registers,7,0);
    return (registers[1]&(1<<16))!=0&&(registers[1]&(1<<30))!=0;
#else
    return __builtin_cpu_supports(avx512f)&&__builtin_cpu_supports(avx512bw);
#endif
}

float half_value(const std::uint8_t*pointer){
    std::uint16_t bits=0;std::memcpy(&bits,pointer,2);
    const std::uint32_t widened=(static_cast<std::uint32_t>(bits&0x8000)<<16)|
        ((static_cast<std::uint32_t>((bits>>10)&31)+112)<<23)|
        (static_cast<std::uint32_t>(bits&1023)<<13);
    float value;std::memcpy(&value,&widened,4);return value;
}

NOINLINE float scalar_q4_dot(const std::uint8_t*packed,const float*input,int elements){
    float result=0.0f;
    for(int block=0;block<elements/256;++block){
        const auto*base=packed+block*144;const float d=half_value(base),dmin=half_value(base+2);const auto*scales=base+4;
        for(int within=0;within<256;++within){
            const int group=within/64,offset=within&63,sub=offset/32,index=group*2+sub;int scale,minimum;
            if(index<4){scale=scales[index]&63;minimum=scales[index+4]&63;}
            else{scale=(scales[index+4]&15)|((scales[index-4]>>6)<<4);minimum=(scales[index+4]>>4)|((scales[index]>>6)<<4);}
            const auto byte=base[16+group*32+(offset&31)];const int quant=sub==0?(byte&15):(byte>>4);
            result+=(d*scale*quant-dmin*minimum)*input[block*256+within];
        }
    }
    return result;
}

NOINLINE void scalar_quantize(
    const float* input,int elements,QwenQ8KBlock* output
){
    for(int block=0;block<elements/256;++block){
        const float*values=input+block*256;
        float maximum=0.0f;
        for(int index=0;index<256;++index)
            maximum=std::max(maximum,std::fabs(values[index]));
        auto&quantized=output[block];
        if(maximum==0.0f){std::memset(&quantized,0,sizeof(quantized));continue;}
        quantized.scale=maximum/127.0f;
        const float inverse=1.0f/quantized.scale;
        for(int index=0;index<256;++index){
            const int value=static_cast<int>(std::nearbyint(values[index]*inverse));
            quantized.values[index]=static_cast<std::int8_t>(
                std::max(-127,std::min(127,value)));
        }
        for(int group=0;group<16;++group){
            int sum=0;
            for(int lane=0;lane<16;++lane)sum+=quantized.values[group*16+lane];
            quantized.sums[group]=static_cast<std::int16_t>(sum);
        }
    }
}

template<class Function>
double measure(Function function,int iterations){
    const auto start=std::chrono::steady_clock::now();
    for(int index=0;index<iterations;++index)function(index);
    const auto elapsed=std::chrono::steady_clock::now()-start;
    return std::chrono::duration<double,std::nano>(elapsed).count()/iterations;
}

} // namespace

int main(){
    constexpr int elements=4096,blocks=elements/256,iterations=20000;
    std::vector<float> input(elements);
    for(int index=0;index<elements;++index)
        input[index]=std::sin(index*0.013f)*0.8f+std::cos(index*0.007f)*0.2f;
    std::vector<QwenQ8KBlock> quantized(blocks);
    volatile std::int64_t checksum=0;
    const double scalar_ns=measure([&](int index){
        scalar_quantize(input.data(),elements,quantized.data());
        checksum=checksum+quantized[index&(blocks-1)].values[index&255];
    },iterations);
    const double simd_ns=measure([&](int index){
        qwen_quantize_q8_k_avx2(input.data(),elements,quantized.data());
        checksum=checksum+quantized[index&(blocks-1)].values[index&255];
    },iterations);

    std::vector<std::uint8_t> packed(static_cast<std::size_t>(elements/32)*34);
    for(int block=0;block<elements/32;++block){
        const std::uint16_t one=0x3c00;
        std::memcpy(packed.data()+block*34,&one,sizeof(one));
        for(int lane=0;lane<32;++lane)
            packed[block*34+2+lane]=static_cast<std::uint8_t>((block*31+lane*17)&255);
    }
    qwen_quantize_q8_k_avx2(input.data(),elements,quantized.data());
    volatile float dot_checksum=0.0f;
    const double avx2_ns=measure([&](int){
        dot_checksum=dot_checksum+qwen_quant_dot_q8_k_avx2(
            packed.data(),8,quantized.data(),elements,0);
    },iterations);
    double vnni_ns=0.0;
    if(has_avx_vnni())vnni_ns=measure([&](int){
        dot_checksum=dot_checksum+qwen_quant_dot_q8_k_avx_vnni(
            packed.data(),8,quantized.data(),elements,0);
    },iterations);

    std::vector<std::uint8_t> q5(static_cast<std::size_t>(elements/256)*176);
    std::vector<std::uint8_t> q6(static_cast<std::size_t>(elements/256)*210);
    for(std::size_t index=0;index<q5.size();++index)
        q5[index]=static_cast<std::uint8_t>((index*73+19)&255);
    for(std::size_t index=0;index<q6.size();++index)
        q6[index]=static_cast<std::uint8_t>((index*61+37)&255);
    for(int block=0;block<elements/256;++block){
        const std::uint16_t one=0x3c00,half=0x3800;
        std::memcpy(q5.data()+block*176,&one,sizeof(one));
        std::memcpy(q5.data()+block*176+2,&half,sizeof(half));
        std::memcpy(q6.data()+block*210+208,&one,sizeof(one));
    }
    const int format_iterations=iterations/4;
    const double q5_avx2_ns=measure([&](int){
        dot_checksum=dot_checksum+qwen_quant_dot_q8_k_avx2(
            q5.data(),13,quantized.data(),elements,0);
    },format_iterations);
    const double q5_vnni_ns=has_avx_vnni()?measure([&](int){
        dot_checksum=dot_checksum+qwen_quant_dot_q8_k_avx_vnni(
            q5.data(),13,quantized.data(),elements,0);
    },format_iterations):0.0;
    const double q6_avx2_ns=measure([&](int){
        dot_checksum=dot_checksum+qwen_quant_dot_q8_k_avx2(
            q6.data(),14,quantized.data(),elements,0);
    },format_iterations);
    const double q6_vnni_ns=has_avx_vnni()?measure([&](int){
        dot_checksum=dot_checksum+qwen_quant_dot_q8_k_avx_vnni(
            q6.data(),14,quantized.data(),elements,0);
    },format_iterations):0.0;

    constexpr int q4_rows=4;
    const auto q4_row_bytes=static_cast<std::size_t>(elements/256)*144;
    std::vector<std::uint8_t> q4(q4_row_bytes*q4_rows);
    for(std::size_t index=0;index<q4.size();++index)
        q4[index]=static_cast<std::uint8_t>((index*47+23)&255);
    for(int row=0;row<q4_rows;++row)for(int block=0;block<elements/256;++block){
        const std::uint16_t one=0x3c00,half=0x3800;
        auto*base=q4.data()+static_cast<std::size_t>(row)*q4_row_bytes+block*144;
        std::memcpy(base,&one,2);std::memcpy(base+2,&half,2);
    }
    const double q4_scalar_ns=measure([&](int){
        dot_checksum=dot_checksum+scalar_q4_dot(q4.data(),input.data(),elements);
    },format_iterations);
    const double q4_avx2_ns=measure([&](int){
        dot_checksum=dot_checksum+qwen_quant_dot_avx2(q4.data(),12,input.data(),elements,0);
    },format_iterations);
    const double q4_avx512_ns=has_avx512()?measure([&](int){
        dot_checksum=dot_checksum+qwen_quant_dot_avx512(q4.data(),12,input.data(),elements,0);
    },format_iterations):0.0;
    float q4_outputs[q4_rows]{};
    const int row_iterations=format_iterations/2;
    const double q4_four_single_avx2_ns=measure([&](int index){
        for(int row=0;row<q4_rows;++row)
            q4_outputs[row]=qwen_quant_dot_avx2(
                q4.data(),12,input.data(),elements,row);
        dot_checksum=dot_checksum+q4_outputs[index&(q4_rows-1)];
    },row_iterations);
    const double q4_four_rows_avx2_ns=measure([&](int index){
        qwen_quant_dot_rows_avx2(
            q4.data(),12,input.data(),elements,0,q4_rows,q4_outputs);
        dot_checksum=dot_checksum+q4_outputs[index&(q4_rows-1)];
    },row_iterations);
    const double q4_four_rows_avx512_ns=has_avx512()?measure([&](int index){
        qwen_quant_dot_rows_avx512(
            q4.data(),12,input.data(),elements,0,q4_rows,q4_outputs);
        dot_checksum=dot_checksum+q4_outputs[index&(q4_rows-1)];
    },row_iterations):0.0;

    std::cout<<scalar_ns<<' '<<simd_ns<<' '<<avx2_ns<<' '<<vnni_ns<<' '
             <<q5_avx2_ns<<' '<<q5_vnni_ns<<' '
             <<q6_avx2_ns<<' '<<q6_vnni_ns<<' '
             <<q4_scalar_ns<<' '<<q4_avx2_ns<<' '<<q4_avx512_ns<<' '
             <<q4_four_single_avx2_ns<<' '<<q4_four_rows_avx2_ns<<' '
             <<q4_four_rows_avx512_ns<<' '
             <<checksum<<' '<<dot_checksum<<'\n';
    return 0;
}
