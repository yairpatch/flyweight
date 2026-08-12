// The AVX2 half of the K-quant scale search.
//
// Its own translation unit because the library is built for baseline x86-64 and
// selects SIMD at runtime -- the same arrangement as q4_avx2.cpp and
// qwen_cpu_avx2.cpp. Built with `-mavx2` and, critically, `-ffp-contract=off`:
// see the note at the top of qwen_kquant_pack.h. `-mfma` is deliberately NOT
// requested, so the compiler has no FMA instruction available to fuse into even
// if the contraction flag were ever lost.

#include "qwen_kquant_pack.h"

namespace qwen_kpack {

#if defined(__AVX2__)
void fit_sub_block_scales_avx2_entry(const float* x, int nmax, float rmin,
                                     int nstep, float* scales, float* mins) {
    fit_sub_block_scales_avx2(x, nmax, rmin, nstep, scales, mins);
}
#endif

}  // namespace qwen_kpack
