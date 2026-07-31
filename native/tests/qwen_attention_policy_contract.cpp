#include "colibri_v2_attention_policy.hpp"

namespace attention = colibri::v2::attention;

int main() {
    constexpr auto threshold = attention::kDefaultCublasMinTokens;
    if (attention::cublas_eligible(1, 1, threshold - 1, 0, 8192, true, threshold))
        return 1;
    if (!attention::cublas_eligible(1, 1, threshold, 0, 8192, true, threshold))
        return 2;
    if (attention::cublas_eligible(0, 1, threshold, 0, 8192, true, threshold))
        return 3;
    if (attention::cublas_eligible(1, 1, threshold, 0, 8192, false, threshold))
        return 4;
    if (attention::cublas_eligible(
            1, 1, threshold, 8192 - threshold + 1, 8192, true, threshold))
        return 5;
    if (!attention::cublas_eligible(1, 1, 1024, 0, 8192, true, 1024))
        return 6;
    return 0;
}
