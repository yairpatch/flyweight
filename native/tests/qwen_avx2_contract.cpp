#include "qwen_cpu_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

bool close(float left,float right){
    const float scale=std::max({1.0f,std::fabs(left),std::fabs(right)});
    return std::fabs(left-right)<=2.0e-4f*scale;
}

void set_half(std::uint8_t*pointer,std::uint16_t bits){
    std::memcpy(pointer,&bits,sizeof(bits));
}

bool quant_contract(std::uint32_t type,int row_bytes){
    constexpr int elements=256,rows=3;
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(rows)*row_bytes);
    for(std::size_t i=0;i<packed.size();++i)
        packed[i]=static_cast<std::uint8_t>((i*73+19)&255);
    for(int row=0;row<rows;++row){
        auto*base=packed.data()+static_cast<std::size_t>(row)*row_bytes;
        if(type==13){set_half(base,0x3c00);set_half(base+2,0x3800);}
        else if(type==14)set_half(base+208,0x3c00);
        else for(int block=0;block<8;++block)set_half(base+block*34,0x3c00);
    }
    std::vector<float> input(elements),dequant(elements),q8_input(elements);
    for(int i=0;i<elements;++i)input[i]=std::sin(i*0.071f)*0.75f;
    QwenQ8KBlock q8{};
    qwen_quantize_q8_k_avx2(input.data(),elements,&q8);
    for(int i=0;i<elements;++i)q8_input[i]=q8.scale*q8.values[i];
    for(int row=0;row<rows;++row){
        qwen_dequant_row_avx2(packed.data(),type,elements,row,dequant.data());
        float reference=0.0f;
        for(int i=0;i<elements;++i)reference+=dequant[i]*input[i];
        const float actual=qwen_quant_dot_avx2(
            packed.data(),type,input.data(),elements,row);
        if(!close(reference,actual))return false;
        float q8_reference=0.0f;
        for(int i=0;i<elements;++i)q8_reference+=dequant[i]*q8_input[i];
        const float q8_actual=qwen_quant_dot_q8_k_avx2(
            packed.data(),type,&q8,elements,row);
        if(!close(q8_reference,q8_actual))return false;
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

} // namespace

int main(){
#if defined(__x86_64__) || defined(_M_X64)
#if !defined(_MSC_VER)
    if(!__builtin_cpu_supports("avx2")||!__builtin_cpu_supports("fma"))return 0;
#endif
    if(!quant_contract(13,176)||!quant_contract(14,210)||
       !quant_contract(8,8*34)||!f32_contract())return 1;
#endif
    return 0;
}
