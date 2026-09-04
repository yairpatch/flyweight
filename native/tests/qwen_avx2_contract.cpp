#include "qwen_cpu_kernel.h"
#include "qwen_kquant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
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
    int registers[4]{};
    __cpuid(registers,1);
    if((registers[2]&(1<<27))==0)return false;
    const auto xcr0=_xgetbv(0);
    if((xcr0&0xe6)!=0xe6)return false;
    __cpuidex(registers,7,0);
    return (registers[1]&(1<<16))!=0&&(registers[1]&(1<<30))!=0;
#else
    return __builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512bw");
#endif
}

bool close(float left,float right){
    const float scale=std::max({1.0f,std::fabs(left),std::fabs(right)});
    return std::fabs(left-right)<=2.0e-4f*scale;
}

void set_half(std::uint8_t*pointer,std::uint16_t bits){
    std::memcpy(pointer,&bits,sizeof(bits));
}

float q4_value(const std::uint8_t*packed,int absolute){
    const int block=absolute/256,within=absolute&255;
    const auto*base=packed+block*144;const int group=within/64,offset=within&63,sub=offset/32;
    std::uint16_t d_bits=0,dmin_bits=0;std::memcpy(&d_bits,base,2);std::memcpy(&dmin_bits,base+2,2);
    const auto half_to_float=[](std::uint16_t bits){
        const std::uint32_t widened=(static_cast<std::uint32_t>(bits&0x8000)<<16)|
            ((static_cast<std::uint32_t>((bits>>10)&31)+112)<<23)|
            (static_cast<std::uint32_t>(bits&1023)<<13);
        float value;std::memcpy(&value,&widened,sizeof(value));return value;
    };
    const float d=half_to_float(d_bits),dmin=half_to_float(dmin_bits);const auto*scales=base+4;
    const int index=group*2+sub;int scale,minimum;
    if(index<4){scale=scales[index]&63;minimum=scales[index+4]&63;}
    else{scale=(scales[index+4]&15)|((scales[index-4]>>6)<<4);minimum=(scales[index+4]>>4)|((scales[index]>>6)<<4);}
    const auto byte=base[16+group*32+(offset&31)];const int quant=sub==0?(byte&15):(byte>>4);
    return d*scale*quant-dmin*minimum;
}

bool q8_quant_contract(){
    constexpr int elements=512;
    std::vector<float> input(elements);
    for(int i=0;i<elements;++i)
        input[i]=std::sin(i*0.071f)*0.75f+std::cos(i*0.019f)*0.125f;
    input[0]=0.0f; input[1]=1.0f; input[2]=-1.0f;
    input[256]=0.0f; input[257]=0.25f; input[258]=-0.25f;

    QwenQ8KBlock actual[2]{};
    qwen_quantize_q8_k_avx2(input.data(),elements,actual);
    for(int block=0;block<2;++block){
        float maximum=0.0f;
        for(int i=0;i<256;++i)
            maximum=std::max(maximum,std::fabs(input[block*256+i]));
        const float scale=maximum/127.0f;
        if(actual[block].scale!=scale)return false;
        for(int group=0;group<16;++group){
            int sum=0;
            for(int lane=0;lane<16;++lane){
                const int index=group*16+lane;
                const int expected=std::max(-127,std::min(
                    127,static_cast<int>(std::nearbyint(
                        input[block*256+index]/scale))));
                if(actual[block].values[index]!=expected)return false;
                sum+=expected;
            }
            if(actual[block].sums[group]!=sum)return false;
        }
    }

    float zeros[elements]{};
    QwenQ8KBlock zero_blocks[2]{};
    qwen_quantize_q8_k_avx2(zeros,elements,zero_blocks);
    for(const auto& block:zero_blocks){
        if(block.scale!=0.0f)return false;
        for(auto value:block.values)if(value!=0)return false;
        for(auto sum:block.sums)if(sum!=0)return false;
    }
    return true;
}

bool quant_contract(std::uint32_t type,int row_bytes,bool avx512){
    constexpr int elements=256,rows=3;
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(rows)*row_bytes);
    for(std::size_t i=0;i<packed.size();++i)
        packed[i]=static_cast<std::uint8_t>((i*73+19)&255);
    for(int row=0;row<rows;++row){
        auto*base=packed.data()+static_cast<std::size_t>(row)*row_bytes;
        if(type==12){set_half(base,0x3c00);set_half(base+2,0x3800);}
        else if(type==13){set_half(base,0x3c00);set_half(base+2,0x3800);}
        else if(type==14)set_half(base+208,0x3c00);
        else if(type==17)set_half(base,0x3c00);
        else if(type==18)set_half(base,0x3c00);
        else if(type==40)for(int block=0;block<4;++block)
            for(int scale=0;scale<4;++scale)
                base[block*36+scale]=static_cast<std::uint8_t>(0x30+scale);
        else for(int block=0;block<8;++block)set_half(base+block*34,0x3c00);
    }
    std::vector<float> input(elements),second(elements),third(elements),fourth(elements),fifth(elements),sixth(elements),seventh(elements),eighth(elements),dequant(elements),q8_input(elements);
    for(int i=0;i<elements;++i){input[i]=std::sin(i*0.071f)*0.75f;second[i]=std::cos(i*0.037f)*0.5f;third[i]=std::sin(i*0.023f)*0.3f;fourth[i]=std::cos(i*0.019f)*0.9f;fifth[i]=std::sin(i*0.017f)*0.4f;sixth[i]=std::cos(i*0.011f)*0.6f;seventh[i]=std::sin(i*0.007f)*0.8f;eighth[i]=std::cos(i*0.005f)*0.2f;}
    QwenQ8KBlock q8{};
    qwen_quantize_q8_k_avx2(input.data(),elements,&q8);
    for(int i=0;i<elements;++i)q8_input[i]=q8.scale*q8.values[i];
    if(type==12){
        float expected[rows]{},avx2_rows[rows]{},avx512_rows[rows]{};
        for(int row=0;row<rows;++row)
            for(int i=0;i<elements;++i)
                expected[row]+=q4_value(
                    packed.data()+static_cast<std::size_t>(row)*row_bytes,i)*input[i];
        qwen_quant_dot_rows_avx2(
            packed.data(),type,input.data(),elements,0,rows,avx2_rows);
        for(int row=0;row<rows;++row)
            if(!close(expected[row],avx2_rows[row]))return false;
        if(avx512){
            qwen_quant_dot_rows_avx512(
                packed.data(),type,input.data(),elements,0,rows,avx512_rows);
            for(int row=0;row<rows;++row)
                if(!close(expected[row],avx512_rows[row]))return false;
        }
    }
    for(int row=0;row<rows;++row){
        qwen_dequant_row_avx2(packed.data(),type,elements,row,dequant.data());
        if(type==12)for(int i=0;i<elements;++i)
            if(dequant[i]!=q4_value(packed.data()+static_cast<std::size_t>(row)*row_bytes,i))return false;
        float reference=0.0f;
        for(int i=0;i<elements;++i)reference+=dequant[i]*input[i];
        const float actual=qwen_quant_dot_avx2(
            packed.data(),type,input.data(),elements,row);
        if(!close(reference,actual))return false;
        const float*quad_inputs[4]={input.data(),second.data(),third.data(),fourth.data()};
        float avx2_quad[4]{};
        qwen_quant_dot_quad_avx2(packed.data(),type,quad_inputs,elements,row,avx2_quad);
        for(int token=0;token<4;++token){float expected=0.0f;for(int i=0;i<elements;++i)expected+=dequant[i]*quad_inputs[token][i];if(!close(expected,avx2_quad[token]))return false;}
        if(avx512){
            float second_reference=0.0f;
            for(int i=0;i<elements;++i)second_reference+=dequant[i]*second[i];
            float pair_first=0.0f,pair_second=0.0f;
            qwen_quant_dot_pair_avx512(packed.data(),type,input.data(),second.data(),elements,row,&pair_first,&pair_second);
            if(!close(reference,pair_first)||!close(second_reference,pair_second))return false;
            float quad_outputs[4]{};
            qwen_quant_dot_quad_avx512(packed.data(),type,quad_inputs,elements,row,quad_outputs);
            for(int token=0;token<4;++token){float expected=0.0f;for(int i=0;i<elements;++i)expected+=dequant[i]*quad_inputs[token][i];if(!close(expected,quad_outputs[token]))return false;}
            const float*oct_inputs[8]={input.data(),second.data(),third.data(),fourth.data(),fifth.data(),sixth.data(),seventh.data(),eighth.data()};
            float oct_outputs[8]{};
            qwen_quant_dot_oct_avx512(packed.data(),type,oct_inputs,elements,row,oct_outputs);
            for(int token=0;token<8;++token){float expected=0.0f;for(int i=0;i<elements;++i)expected+=dequant[i]*oct_inputs[token][i];if(!close(expected,oct_outputs[token]))return false;}
            std::vector<std::uint8_t> other=packed;
            for(std::size_t i=0;i<other.size();++i)
                other[i]^=static_cast<std::uint8_t>((i*29+7)&255);
            auto*other_base=other.data()+static_cast<std::size_t>(row)*row_bytes;
            if(type==12){set_half(other_base,0x3a00);set_half(other_base+2,0x3400);}
            else if(type==13){set_half(other_base,0x3a00);set_half(other_base+2,0x3400);}
            else if(type==14)set_half(other_base+208,0x3a00);
            else if(type==40)for(int block=0;block<4;++block)
                for(int scale=0;scale<4;++scale)
                    other_base[block*36+scale]=static_cast<std::uint8_t>(0x28+scale);
            else for(int block=0;block<8;++block)set_half(other_base+block*34,0x3a00);
            const float other_reference=qwen_quant_dot_avx512(
                other.data(),type,input.data(),elements,row);
            float two_rows_first=0.0f,two_rows_second=0.0f;
            qwen_quant_dot_two_rows_avx512(
                packed.data()+static_cast<std::size_t>(row)*row_bytes,
                other.data()+static_cast<std::size_t>(row)*row_bytes,
                type,input.data(),elements,&two_rows_first,&two_rows_second);
            if(!close(reference,two_rows_first)||
               !close(other_reference,two_rows_second))return false;
        }
        if(type==8||type==13||type==14||type==18){
            float q8_reference=0.0f;
            for(int i=0;i<elements;++i)q8_reference+=dequant[i]*q8_input[i];
            const float q8_actual=qwen_quant_dot_q8_k_avx2(
                packed.data(),type,&q8,elements,row);
            if(!close(q8_reference,q8_actual))return false;
            if(has_avx_vnni()){
                const float vnni=qwen_quant_dot_q8_k_avx_vnni(
                    packed.data(),type,&q8,elements,row);
                if(!close(q8_reference,vnni)||!close(q8_actual,vnni))return false;
            }
        }
    }
    return true;
}

bool q40_contract(bool avx512){
    // 704 elements on purpose: Gemma 4's expert intermediate, a whole number
    // of Q4_0 blocks but NOT of K-quant super-blocks -- the width the %256
    // admission gate used to exclude.
    constexpr int elements=704,rows=3,row_bytes=(elements/32)*18;
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(rows)*row_bytes);
    for(std::size_t i=0;i<packed.size();++i)
        packed[i]=static_cast<std::uint8_t>((i*73+19)&255);
    for(int row=0;row<rows;++row)
        for(int block=0;block<elements/32;++block)
            set_half(packed.data()+static_cast<std::size_t>(row)*row_bytes+block*18,
                     0x3c00-static_cast<std::uint16_t>(block%3)*0x400);
    std::vector<float> input(elements);
    for(int i=0;i<elements;++i)input[i]=std::sin(i*0.071f)*0.75f;
    const auto half_to_float=[](std::uint16_t bits){
        const std::uint32_t widened=(static_cast<std::uint32_t>(bits&0x8000)<<16)|
            ((static_cast<std::uint32_t>((bits>>10)&31)+112)<<23)|
            (static_cast<std::uint32_t>(bits&1023)<<13);
        float value;std::memcpy(&value,&widened,sizeof(value));return value;
    };
    const auto q40_value=[&](const std::uint8_t*row_data,int index){
        const auto*base=row_data+(index/32)*18;
        std::uint16_t bits;std::memcpy(&bits,base,2);
        const int within=index%32;
        const auto byte=base[2+within%16];
        const int quant=(within<16?(byte&15):(byte>>4))-8;
        return half_to_float(bits)*quant;
    };
    for(int row=0;row<rows;++row){
        const auto*row_data=packed.data()+static_cast<std::size_t>(row)*row_bytes;
        float reference=0.0f;
        for(int i=0;i<elements;++i)reference+=q40_value(row_data,i)*input[i];
        const float avx2=qwen_quant_dot_avx2(
            packed.data(),2,input.data(),elements,row);
        if(!close(reference,avx2))return false;
        if(avx512){
            const float wide=qwen_quant_dot_avx512(
                packed.data(),2,input.data(),elements,row);
            if(!close(reference,wide))return false;
        }
    }
    return true;
}

bool iq_q8_contract(std::uint32_t type,int row_bytes){
    constexpr int elements=256;
    std::vector<std::uint8_t> packed(row_bytes);
    for(int i=0;i<row_bytes;++i)
        packed[i]=static_cast<std::uint8_t>((i*73+19)&255);
    set_half(packed.data(),0x3c00);
    std::vector<float> input(elements),dequant(elements);
    for(int i=0;i<elements;++i)
        input[i]=std::sin(i*0.071f)*0.75f;
    QwenQ8KBlock q8{};
    qwen_quantize_q8_k_avx2(input.data(),elements,&q8);
    qwen_dequant_row_avx2(
        packed.data(),type,elements,0,dequant.data());
    float expected=0.0f;
    for(int i=0;i<elements;++i)
        expected+=dequant[i]*(q8.scale*q8.values[i]);
    const float actual=qwen_quant_dot_q8_k_avx2(
        packed.data(),type,&q8,elements,0);
    if(!close(expected,actual))return false;
    if(has_avx_vnni()){
        const float vnni=qwen_quant_dot_q8_k_avx_vnni(
            packed.data(),type,&q8,elements,0);
        if(!close(expected,vnni)||!close(actual,vnni))return false;
    }
    return true;
}

bool f32_contract(){
    constexpr int elements=256,count=3,rows=4;
    std::vector<float> weights(rows*elements),vectors(count*elements),output(rows*count);
    const float*inputs[count];
    for(int i=0;i<rows*elements;++i)weights[i]=std::cos(i*0.013f);
    for(int token=0;token<count;++token){
        for(int i=0;i<elements;++i)vectors[token*elements+i]=std::sin((token+1)*(i+1)*0.009f);
        inputs[token]=vectors.data()+token*elements;
    }
    qwen_f32_gemm_rows_avx2(weights.data(),rows,inputs,count,elements,output.data());
    for(int row=0;row<rows;++row)for(int token=0;token<count;++token){
        float reference=0.0f;
        for(int i=0;i<elements;++i)
            reference+=weights[row*elements+i]*inputs[token][i];
        if(!close(reference,output[row*count+token]))return false;
    }
    return true;
}

// f16 (type 1) and bf16 (type 30) rows: the entry points decode these with
// dedicated branches added alongside the bailing all-types work. The odd
// element count is deliberate -- it exercises the scalar tails the kernels
// carry, since halfword rows have no block-size granule.
bool half_contract(bool bf16,bool avx512){
    constexpr int elements=300,rows=3;
    const std::uint32_t type=bf16?30u:1u;
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(rows)*elements*2);
    std::vector<float> decoded(rows*elements);
    for(int row=0;row<rows;++row)for(int i=0;i<elements;++i){
        const float value=std::sin((row+1)*(i+1)*0.003f)*2.5f;
        std::uint16_t bits;
        if(bf16){
            std::uint32_t wide;std::memcpy(&wide,&value,sizeof(wide));
            bits=static_cast<std::uint16_t>(wide>>16);
            const std::uint32_t back=static_cast<std::uint32_t>(bits)<<16;
            std::memcpy(&decoded[row*elements+i],&back,sizeof(float));
        }else{
            bits=qwen_half_bits(value);
            decoded[row*elements+i]=qwen_half_value(bits);
        }
        set_half(packed.data()+(static_cast<std::size_t>(row)*elements+i)*2,bits);
    }
    std::vector<float> vectors(4*elements);
    const float*inputs[4];
    for(int token=0;token<4;++token){
        for(int i=0;i<elements;++i)
            vectors[token*elements+i]=std::cos((token+2)*(i+1)*0.005f);
        inputs[token]=vectors.data()+token*elements;
    }
    for(int row=0;row<rows;++row){
        float expected[4]{};
        for(int token=0;token<4;++token)
            for(int i=0;i<elements;++i)
                expected[token]+=decoded[row*elements+i]*inputs[token][i];
        if(!close(expected[0],
                  qwen_quant_dot_avx2(packed.data(),type,inputs[0],elements,row)))
            return false;
        float quad[4]{};
        qwen_quant_dot_quad_avx2(packed.data(),type,inputs,elements,row,quad);
        for(int token=0;token<4;++token)
            if(!close(expected[token],quad[token]))return false;
        if(avx512){
            if(!close(expected[0],
                      qwen_quant_dot_avx512(packed.data(),type,inputs[0],elements,row)))
                return false;
            float quad512[4]{};
            qwen_quant_dot_quad_avx512(packed.data(),type,inputs,elements,row,quad512);
            for(int token=0;token<4;++token)
                if(!close(expected[token],quad512[token]))return false;
            float first=0.0f,second=0.0f;
            qwen_quant_dot_pair_avx512(
                packed.data(),type,inputs[0],inputs[1],elements,row,&first,&second);
            if(!close(expected[0],first)||!close(expected[1],second))return false;
        }
    }
    return true;
}

bool half_oct_contract(bool bf16){
    constexpr int elements=144;
    const std::uint32_t type=bf16?30u:1u;
    std::vector<std::uint8_t> packed(elements*2);
    std::vector<float> decoded(elements);
    for(int i=0;i<elements;++i){
        const float value=std::sin((i+1)*0.011f);
        std::uint16_t bits;
        if(bf16){
            std::uint32_t wide;std::memcpy(&wide,&value,sizeof(wide));
            bits=static_cast<std::uint16_t>(wide>>16);
            const std::uint32_t back=static_cast<std::uint32_t>(bits)<<16;
            std::memcpy(&decoded[i],&back,sizeof(float));
        }else{
            bits=qwen_half_bits(value);
            decoded[i]=qwen_half_value(bits);
        }
        set_half(packed.data()+static_cast<std::size_t>(i)*2,bits);
    }
    std::vector<float> vectors(8*elements);
    const float*inputs[8];
    for(int token=0;token<8;++token){
        for(int i=0;i<elements;++i)
            vectors[token*elements+i]=std::cos((token+1)*(i+3)*0.007f);
        inputs[token]=vectors.data()+token*elements;
    }
    float expected[8]{},outputs[8]{};
    for(int token=0;token<8;++token)
        for(int i=0;i<elements;++i)
            expected[token]+=decoded[i]*inputs[token][i];
    qwen_quant_dot_oct_avx512(packed.data(),type,inputs,elements,0,outputs);
    for(int token=0;token<8;++token)
        if(!close(expected[token],outputs[token]))return false;
    return true;
}

} // namespace

int main(){
#if defined(__x86_64__) || defined(_M_X64)
#if !defined(_MSC_VER)
    if(!__builtin_cpu_supports("avx2")||!__builtin_cpu_supports("fma"))return 0;
    const bool avx512=__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512bw");
#else
    const bool avx512=has_avx512();
#endif
    const auto require=[](bool passed,const char*name){
        if(!passed)std::fprintf(stderr,"failed: %s\n",name);
        return passed;
    };
    if(!require(q8_quant_contract(),"q8 quant")||
       !require(quant_contract(12,144,avx512),"Q4_K")||
       !require(quant_contract(13,176,avx512),"Q5_K")||
       !require(quant_contract(14,210,avx512),"Q6_K")||
       !require(iq_q8_contract(17,74),"IQ2_XS x Q8_K")||
       !require(iq_q8_contract(18,98),"IQ3_XXS x Q8_K")||
       !require(quant_contract(40,4*36,avx512),"NVFP4")||
       !require(quant_contract(8,8*34,avx512),"Q8_0")||
       !require(q40_contract(avx512),"Q4_0")||
       !require(f32_contract(),"F32")||
       !require(half_contract(false,avx512),"F16")||
       !require(half_contract(true,avx512),"BF16")||
       (avx512&&(!require(half_oct_contract(false),"F16 oct")||
                 !require(half_oct_contract(true),"BF16 oct"))))return 1;
#endif
    return 0;
}
