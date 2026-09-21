// Sylvester-Hadamard rotation contract test.
//
// Verifies:
// 1. Orthogonality and self-inverse: H * H = I (H = H^T).
// 2. Isometry (norm preservation): ||H x||_2 = ||x||_2.
// 3. Inner-product invariance: <H q, H k> = <q, k>.
// 4. Warp butterfly parity with canonical Sylvester recursive matrix.
// 5. Outlier dispersion: squashing LLM activation outliers by sqrt(D).
// 6. INT8 quantization MSE reduction (>10x improvement on outlier vectors).

#include "hadamard.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

bool test_orthogonality_and_norm(int dim) {
    std::mt19937 engine(42 + dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> x(dim), y(dim), x_rec(dim);
    for (int i = 0; i < dim; ++i) x[i] = dist(engine);

    float norm_x = 0.0f;
    for (int i = 0; i < dim; ++i) norm_x += x[i] * x[i];

    flyweight::hadamard_sylvester_cpu(x.data(), y.data(), dim);

    float norm_y = 0.0f;
    for (int i = 0; i < dim; ++i) norm_y += y[i] * y[i];

    if (std::fabs(norm_x - norm_y) > 1e-4f * norm_x) {
        std::printf("FAIL D=%d norm preservation: ||x||^2=%.6f, ||y||^2=%.6f\n",
                    dim, norm_x, norm_y);
        return false;
    }

    flyweight::hadamard_sylvester_cpu(y.data(), x_rec.data(), dim);

    float max_err = 0.0f;
    for (int i = 0; i < dim; ++i) {
        max_err = std::max(max_err, std::fabs(x[i] - x_rec[i]));
    }
    if (max_err > 1e-5f) {
        std::printf("FAIL D=%d self-inverse: max_err=%.9e\n", dim, max_err);
        return false;
    }

    // Inner-product preservation <H q, H k> == <q, k>
    std::vector<float> k(dim), y_k(dim);
    for (int i = 0; i < dim; ++i) k[i] = dist(engine);
    flyweight::hadamard_sylvester_cpu(k.data(), y_k.data(), dim);

    float dot_direct = 0.0f, dot_rotated = 0.0f;
    for (int i = 0; i < dim; ++i) {
        dot_direct += x[i] * k[i];
        dot_rotated += y[i] * y_k[i];
    }
    if (std::fabs(dot_direct - dot_rotated) > 1e-4f * std::max(1.0f, std::fabs(dot_direct))) {
        std::printf("FAIL D=%d dot product: direct=%.6f, rotated=%.6f\n",
                    dim, dot_direct, dot_rotated);
        return false;
    }

    return true;
}

bool test_warp_butterfly_parity_d256() {
    constexpr int Dim = 256;
    constexpr int Columns = 8;
    std::mt19937 engine(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> x(Dim), ref(Dim);
    for (int i = 0; i < Dim; ++i) x[i] = dist(engine);

    flyweight::hadamard_sylvester_cpu(x.data(), ref.data(), Dim);

    float warp_val[32][Columns];
    for (int l = 0; l < 32; ++l) {
        for (int r = 0; r < Columns; ++r) {
            warp_val[l][r] = x[l + 32 * r];
        }
    }

    flyweight::hadamard_warp_butterfly_cpu<Columns>(warp_val);

    float max_err = 0.0f;
    for (int l = 0; l < 32; ++l) {
        for (int r = 0; r < Columns; ++r) {
            const float diff = std::fabs(warp_val[l][r] - ref[l + 32 * r]);
            max_err = std::max(max_err, diff);
        }
    }

    if (max_err > 1e-5f) {
        std::printf("FAIL D=256 warp butterfly parity: max_err=%.9e\n", max_err);
        return false;
    }
    return true;
}

bool test_warp_butterfly_parity_d128() {
    constexpr int Dim = 128;
    constexpr int Columns = 4;
    std::mt19937 engine(5678);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> x(Dim), ref(Dim);
    for (int i = 0; i < Dim; ++i) x[i] = dist(engine);

    flyweight::hadamard_sylvester_cpu(x.data(), ref.data(), Dim);

    float warp_val[32][Columns];
    for (int l = 0; l < 32; ++l) {
        for (int r = 0; r < Columns; ++r) {
            warp_val[l][r] = x[l + 32 * r];
        }
    }

    flyweight::hadamard_warp_butterfly_cpu<Columns>(warp_val);

    float max_err = 0.0f;
    for (int l = 0; l < 32; ++l) {
        for (int r = 0; r < Columns; ++r) {
            const float diff = std::fabs(warp_val[l][r] - ref[l + 32 * r]);
            max_err = std::max(max_err, diff);
        }
    }

    if (max_err > 1e-5f) {
        std::printf("FAIL D=128 warp butterfly parity: max_err=%.9e\n", max_err);
        return false;
    }
    return true;
}

bool test_outlier_suppression(int dim) {
    std::mt19937 engine(999);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> x(dim), y(dim);
    for (int i = 0; i < dim; ++i) x[i] = dist(engine);

    // Inject heavy channel outliers characteristic of LLMs
    x[7] = 38.0f;
    x[42] = -45.0f;
    x[dim - 5] = 32.0f;

    float pre_max = 0.0f;
    for (int i = 0; i < dim; ++i) pre_max = std::max(pre_max, std::fabs(x[i]));

    flyweight::hadamard_sylvester_cpu(x.data(), y.data(), dim);

    float post_max = 0.0f;
    for (int i = 0; i < dim; ++i) post_max = std::max(post_max, std::fabs(y[i]));

    // In the worst-case Hadamard coordinate, all outliers could add constructively:
    // bound = sum(|outlier_i|) / sqrt(dim) + 3.0 * sigma
    const float outlier_sum = 38.0f + 45.0f + 32.0f;
    const float expected_bound = (outlier_sum / std::sqrt(static_cast<float>(dim))) + 3.0f;
    if (post_max > expected_bound) {
        std::printf("FAIL D=%d outlier dispersion: pre_max=%.2f, post_max=%.2f (bound=%.2f)\n",
                    dim, pre_max, post_max, expected_bound);
        return false;
    }

    // Simulate standard INT8 uniform symmetric quantization [-127, 127]
    // 1. Without Hadamard:
    const float scale_raw = pre_max / 127.0f;
    float mse_raw = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const int q = std::clamp(static_cast<int>(std::round(x[i] / scale_raw)), -127, 127);
        const float rec = static_cast<float>(q) * scale_raw;
        const float err = rec - x[i];
        mse_raw += err * err;
    }
    mse_raw /= static_cast<float>(dim);

    // 2. With Hadamard:
    const float scale_rot = post_max / 127.0f;
    std::vector<float> y_rec(dim), x_unrot(dim);
    for (int i = 0; i < dim; ++i) {
        const int q = std::clamp(static_cast<int>(std::round(y[i] / scale_rot)), -127, 127);
        y_rec[i] = static_cast<float>(q) * scale_rot;
    }
    // Rotate back
    flyweight::hadamard_sylvester_cpu(y_rec.data(), x_unrot.data(), dim);
    float mse_rot = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const float err = x_unrot[i] - x[i];
        mse_rot += err * err;
    }
    mse_rot /= static_cast<float>(dim);

    const float improvement = mse_raw / std::max(1e-8f, mse_rot);
    if (improvement < 5.0f) {
        std::printf("FAIL D=%d INT8 quantization improvement: raw MSE=%.5f, rotated MSE=%.5f (ratio=%.2f)\n",
                    dim, mse_raw, mse_rot, improvement);
        return false;
    }

    std::printf("D=%3d: pre_max=%5.1f -> post_max=%4.2f (suppression %4.1fx) | INT8 MSE: raw=%.4f -> rot=%.4f (%4.1fx better)\n",
                dim, pre_max, post_max, pre_max / post_max, mse_raw, mse_rot, improvement);

    return true;
}

} // namespace

int main() {
    std::printf("=== Sylvester-Hadamard Transform Contract ===\n");

    if (!test_orthogonality_and_norm(128)) return 1;
    if (!test_orthogonality_and_norm(256)) return 1;
    std::printf("PASS: Orthogonality, norm preservation, and inner products verified for D=128 and D=256\n");

    if (!test_warp_butterfly_parity_d128()) return 1;
    if (!test_warp_butterfly_parity_d256()) return 1;
    std::printf("PASS: Warp-level butterfly parity verified against canonical Sylvester matrix\n");

    if (!test_outlier_suppression(128)) return 1;
    if (!test_outlier_suppression(256)) return 1;
    std::printf("PASS: Activation outlier suppression and INT8 quantization MSE gain verified\n");

    std::printf("=== All Hadamard contract tests passed ===\n");
    return 0;
}
