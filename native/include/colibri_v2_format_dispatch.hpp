#pragma once

#include <cstdint>

// One table answering "which kernel serves this GGUF tensor type in this
// role", replacing the per-site switch statements that used to answer it
// independently in single-token decode, multi-sequence decode, the rows
// forward, MTP verification, and the LM-head/sampler/embedding helpers.
//
// The point is not the lookup -- it is that a new format becomes one row here
// plus its kernels, and that a *missing* kernel is a visible null instead of a
// silent fall-through to a slower or wrong path. Two such drifts were found
// while building this table and are preserved as-is (they are behavior, and
// changing behavior is a separate, measured commit):
//
//   - IQ4_XS (23) has a Q8-activation warp matvec, a tiled kernel and an MMQ
//     kernel, but the rows forward's admission gate never listed it, so its
//     chunked prefill ran the reconstruct-in-float rows matmul at 4 rows per
//     block -- a full weight read per 4-row group where MMQ covers 128 tokens.
//     Admitted 2026-08-26 after measuring on the 27B dense checkpoint, whose
//     16 attn_v projections (and the whole MTP draft layer) carry this type.
//   - IQ1_M (29) is absent from the CPU expert set even though the CPU dot
//     decodes IQ1_S; `cpu_expert` records it.
//
// Names must match the kernels registered with colibri_gpu_launch_named; a
// Python source-scan test cross-checks every literal below against the kernel
// header and the driver.

namespace colibri::v2 {

// Grid layout of the reconstruct-in-float batch matmul (`matmul_rows`):
//   per_token: one block column per row       (output_size, rows)
//   tiled32:   32x32 tiles                    ((out+31)/32, (rows+31)/32)
//   quad_pack: four rows per block            (output_size, (rows+3)/4)
enum class RowsMatmulGrid : std::uint8_t { none, per_token, tiled32, quad_pack };

struct QwenFormatKernels {
    std::uint32_t type = 0;
    // Kernel-name stem; every non-null name below must contain it, which is
    // what the contract test uses to catch cross-wired copy-paste.
    const char* family = nullptr;

    // Q8-quantized-activation group-decode matvec (decode fast path and the
    // sampler's full-logits projection).
    const char* matvec_q8_warp = nullptr;
    // Whether the rows forward may take its Q8-activation block for this type.
    bool rows_q8_gate = false;
    // Rows-forward batch kernels over Q8 activations: 8-row batch, 32-token
    // tile, tensor-core MMQ (mmq_min: the asymmetric K-quant tile geometry).
    const char* matvec_q8_rows = nullptr;
    const char* matmul_q8_tiled = nullptr;
    const char* matmul_q8_mmq = nullptr;
    bool mmq_min = false;

    // Reconstruct-in-float batch fallback and its grid shape.
    const char* matmul_rows = nullptr;
    RowsMatmulGrid matmul_rows_grid = RowsMatmulGrid::none;

    // LM head: fused f32-activation argmax, and the Q8-activation variant.
    const char* lm_head_argmax = nullptr;
    const char* lm_head_argmax_q8 = nullptr;

    // Embedding gather (single token / row batch).
    const char* embedding = nullptr;
    const char* embedding_rows = nullptr;

    // Grouped IQ expert kernel family stem (qwen_iq_grouped_kernel), null
    // where the format has no grouped expert kernels.
    const char* iq_expert_prefix = nullptr;

    // Whether qwen_quant_dot can execute this type on the CPU expert path.
    bool cpu_expert = false;
};

inline constexpr QwenFormatKernels kQwenFormats[] = {
    {.type = 0, .family = "f32",
     .matmul_rows = "qwen_f32_matmul_rows",
     .matmul_rows_grid = RowsMatmulGrid::per_token,
     .lm_head_argmax = "f32_lm_head_argmax_warp",
     .embedding = "qwen_f32_embedding", .embedding_rows = "qwen_f32_embedding_rows",
     .cpu_expert = true},
    // f16: warp matvec plus embedding/rows; the LM head projects through the
    // matvec dispatch, so no fused argmax kernel exists yet.
    {.type = 1, .family = "f16",
     .matmul_rows = "qwen_f16_matmul_rows",
     .matmul_rows_grid = RowsMatmulGrid::per_token,
     .embedding = "qwen_f16_embedding", .embedding_rows = "qwen_f16_embedding_rows",
     .cpu_expert = true},
    // GGML type 2 is Q4_0 (this row was historically mislabeled "f16"; the
    // family string is only a name stem for the contract test, and every
    // kernel field here is null, so the label was inert either way). Written
    // "q40" because the table-scan test reads underscored strings as kernel
    // names.
    {.type = 2, .family = "q40", .cpu_expert = true},
    {.type = 8, .family = "q8",
     .matmul_rows = "q8_matmul_tiled",
     .matmul_rows_grid = RowsMatmulGrid::tiled32,
     .lm_head_argmax = "q8_lm_head_argmax_warp",
     .embedding = "qwen_q8_embedding", .embedding_rows = "qwen_q8_embedding_rows",
     .cpu_expert = true},
    {.type = 10, .family = "q2k",
     .matvec_q8_warp = "q2k_q8_matvec_transposed_warp", .rows_q8_gate = true,
     .matvec_q8_rows = "q2k_q8_matvec_transposed_rows",
     .matmul_q8_mmq = "q2k_q8_mmq", .mmq_min = true,
     .matmul_rows = "q2k_matmul_rows",
     .matmul_rows_grid = RowsMatmulGrid::quad_pack,
     .lm_head_argmax = "q2k_lm_head_argmax_warp",
     .lm_head_argmax_q8 = "q2k_q8_lm_head_argmax_warp",
     .embedding = "qwen_q2k_embedding", .embedding_rows = "qwen_q2k_embedding_rows",
     .cpu_expert = true},
    {.type = 11, .family = "q3k",
     .matvec_q8_warp = "q3k_q8_matvec_transposed_warp", .rows_q8_gate = true,
     .matvec_q8_rows = "q3k_q8_matvec_transposed_rows",
     .matmul_q8_tiled = "q3k_q8_matmul_tiled", .matmul_q8_mmq = "q3k_q8_mmq",
     .matmul_rows = "q3k_matmul_rows",
     .matmul_rows_grid = RowsMatmulGrid::quad_pack,
     .lm_head_argmax = "q3k_lm_head_argmax_warp",
     .lm_head_argmax_q8 = "q3k_q8_lm_head_argmax_warp",
     .embedding = "qwen_q3k_embedding", .embedding_rows = "qwen_q3k_embedding_rows",
     .cpu_expert = true},
    {.type = 12, .family = "q4k",
     .matvec_q8_warp = "q4k_q8_matvec_transposed_warp", .rows_q8_gate = true,
     .matvec_q8_rows = "q4k_q8_matvec_transposed_rows",
     .matmul_q8_mmq = "q4k_q8_mmq", .mmq_min = true,
     .matmul_rows = "q4k_matmul_rows",
     .matmul_rows_grid = RowsMatmulGrid::quad_pack,
     .lm_head_argmax = "q4k_lm_head_argmax_warp",
     .lm_head_argmax_q8 = "q4k_q8_lm_head_argmax_warp",
     .embedding = "qwen_q4k_embedding", .embedding_rows = "qwen_q4k_embedding_rows",
     .cpu_expert = true},
    {.type = 13, .family = "q5k",
     .matvec_q8_warp = "q5k_q8_matvec_transposed_warp", .rows_q8_gate = true,
     .matvec_q8_rows = "q5k_q8_matvec_transposed_rows",
     .matmul_q8_mmq = "q5k_q8_mmq", .mmq_min = true,
     .matmul_rows = "q5k_matmul_rows",
     .matmul_rows_grid = RowsMatmulGrid::quad_pack,
     .lm_head_argmax = "q5k_lm_head_argmax_warp",
     .lm_head_argmax_q8 = "q5k_q8_lm_head_argmax_warp",
     .embedding = "qwen_q5k_embedding", .embedding_rows = "qwen_q5k_embedding_rows",
     .cpu_expert = true},
    {.type = 14, .family = "q6k",
     .matvec_q8_warp = "q6k_q8_matvec_transposed_warp", .rows_q8_gate = true,
     .matvec_q8_rows = "q6k_q8_matvec_transposed_rows",
     .matmul_q8_tiled = "q6k_q8_matmul_tiled", .matmul_q8_mmq = "q6k_q8_mmq",
     .matmul_rows = "q6k_matmul_rows",
     .matmul_rows_grid = RowsMatmulGrid::quad_pack,
     .lm_head_argmax = "q6k_lm_head_argmax_warp",
     .lm_head_argmax_q8 = "q6k_q8_lm_head_argmax_warp",
     .embedding = "qwen_q6k_embedding", .embedding_rows = "qwen_q6k_embedding_rows",
     .cpu_expert = true},
    {.type = 16, .family = "iq2xxs",
     .matvec_q8_warp = "iq2xxs_q8_matvec_transposed_warp", .rows_q8_gate = true,
     .matvec_q8_rows = "iq2xxs_q8_matvec_transposed_rows",
     .matmul_q8_tiled = "iq2xxs_q8_matmul_tiled", .matmul_q8_mmq = "iq2xxs_q8_mmq",
     .matmul_rows = "iq2xxs_matmul_rows",
     .matmul_rows_grid = RowsMatmulGrid::quad_pack,
     .lm_head_argmax = "iq2xxs_lm_head_argmax_warp",
     .lm_head_argmax_q8 = "iq2xxs_q8_lm_head_argmax_warp",
     .embedding = "qwen_iq2xxs_embedding",
     .embedding_rows = "qwen_iq2xxs_embedding_rows",
     .iq_expert_prefix = "iq2xxs", .cpu_expert = true},
    {.type = 17, .family = "iq2xs",
     .matvec_q8_warp = "iq2xs_q8_matvec_transposed_warp", .rows_q8_gate = true,
     .matvec_q8_rows = "iq2xs_q8_matvec_transposed_rows",
     .matmul_q8_tiled = "iq2xs_q8_matmul_tiled", .matmul_q8_mmq = "iq2xs_q8_mmq",
     .matmul_rows = "iq2xs_matmul_rows",
     .matmul_rows_grid = RowsMatmulGrid::quad_pack,
     .lm_head_argmax = "iq2xs_lm_head_argmax_warp",
     .embedding = "qwen_iq2xs_embedding",
     .embedding_rows = "qwen_iq2xs_embedding_rows",
     .iq_expert_prefix = "iq2xs", .cpu_expert = true},
    {.type = 18, .family = "iq3xxs",
     .matvec_q8_warp = "iq3xxs_q8_matvec_transposed_warp", .rows_q8_gate = true,
     .matvec_q8_rows = "iq3xxs_q8_matvec_transposed_rows",
     .matmul_q8_tiled = "iq3xxs_q8_matmul_tiled", .matmul_q8_mmq = "iq3xxs_q8_mmq",
     .matmul_rows = "iq3xxs_matmul_rows",
     .matmul_rows_grid = RowsMatmulGrid::quad_pack,
     .lm_head_argmax = "iq3xxs_lm_head_argmax_warp",
     .lm_head_argmax_q8 = "iq3xxs_q8_lm_head_argmax_warp",
     .embedding = "qwen_iq3xxs_embedding",
     .embedding_rows = "qwen_iq3xxs_embedding_rows",
     .iq_expert_prefix = "iq3xxs", .cpu_expert = true},
    {.type = 19, .family = "iq1s",
     .matvec_q8_warp = "iq1s_q8_matvec_transposed_warp", .rows_q8_gate = true,
     .matvec_q8_rows = "iq1s_q8_matvec_transposed_rows",
     .matmul_q8_tiled = "iq1s_q8_matmul_tiled", .matmul_q8_mmq = "iq1s_q8_mmq",
     .matmul_rows = "iq1s_matmul_rows",
     .matmul_rows_grid = RowsMatmulGrid::quad_pack,
     .iq_expert_prefix = "iq1s", .cpu_expert = true},
    {.type = 21, .family = "iq3s",
     .matmul_rows = "iq3s_matmul_rows",
     .matmul_rows_grid = RowsMatmulGrid::quad_pack,
     .lm_head_argmax = "iq3s_lm_head_argmax_warp",
     .embedding = "qwen_iq3s_embedding",
     .embedding_rows = "qwen_iq3s_embedding_rows",
     .cpu_expert = true},
    {.type = 22, .family = "iq2s",
     .matvec_q8_warp = "iq2s_q8_matvec_transposed_warp", .rows_q8_gate = true,
     .matvec_q8_rows = "iq2s_q8_matvec_transposed_rows",
     .matmul_q8_tiled = "iq2s_q8_matmul_tiled", .matmul_q8_mmq = "iq2s_q8_mmq",
     .matmul_rows = "iq2s_matmul_rows",
     .matmul_rows_grid = RowsMatmulGrid::quad_pack,
     .lm_head_argmax = "iq2s_lm_head_argmax_warp",
     .embedding = "qwen_iq2s_embedding",
     .embedding_rows = "qwen_iq2s_embedding_rows",
     .cpu_expert = true},
    {.type = 23, .family = "iq4xs",
     .matvec_q8_warp = "iq4xs_q8_matvec_transposed_warp", .rows_q8_gate = true,
     .matvec_q8_rows = "iq4xs_q8_matvec_transposed_rows",
     .matmul_q8_tiled = "iq4xs_q8_matmul_tiled", .matmul_q8_mmq = "iq4xs_q8_mmq",
     .matmul_rows = "iq4xs_matmul_rows",
     .matmul_rows_grid = RowsMatmulGrid::quad_pack,
     .lm_head_argmax = "iq4xs_lm_head_argmax_warp",
     .embedding = "qwen_iq4xs_embedding",
     .embedding_rows = "qwen_iq4xs_embedding_rows",
     .iq_expert_prefix = "iq4xs", .cpu_expert = true},
    // IQ4_NL: IQ4_XS's non-superblock sibling (18B per 32 values, same
    // codebook, no sub-block scales). qwen4exp carries its ffn_down_exps and
    // the PLE n-gram table in it; the table is host row-gathered, the experts
    // run on the CPU dots or the grouped device kernels below.
    {.type = 20, .family = "iq4nl",
     // No MMQ entry on purpose: the MMQ branch needs (in_size % 256) == 0 and
     // this format's consumer here is the 640-wide expert down projection.
     // The rows matmul is what puts IQ4_NL experts on the prefill expert-GEMM
     // path -- stream_role_ok is a conjunction over gate/up/down, so without it
     // a whole checkpoint's experts fall back to the decode-shaped grouped
     // kernels that re-decode weights once per routed token.
     .matmul_rows = "iq4nl_matmul_rows",
     .matmul_rows_grid = RowsMatmulGrid::quad_pack,
     .iq_expert_prefix = "iq4nl", .cpu_expert = true},
    {.type = 29, .family = "iq1m",
     .matvec_q8_warp = "iq1m_q8_matvec_transposed_warp", .rows_q8_gate = true,
     .matvec_q8_rows = "iq1m_q8_matvec_transposed_rows",
     .matmul_q8_tiled = "iq1m_q8_matmul_tiled", .matmul_q8_mmq = "iq1m_q8_mmq",
     .matmul_rows = "iq1m_matmul_rows",
     .matmul_rows_grid = RowsMatmulGrid::quad_pack,
     // No CPU expert dot today; see the file comment.
     .cpu_expert = false},
    {.type = 30, .family = "bf16",
     .matmul_rows = "bf16_matmul_rows",
     .matmul_rows_grid = RowsMatmulGrid::per_token,
     .lm_head_argmax = "bf16_lm_head_argmax_warp",
     .embedding = "qwen_bf16_embedding",
     .embedding_rows = "qwen_bf16_embedding_rows",
     .cpu_expert = true},
    // MXFP4 and NVFP4 run through dedicated paths (grouped/cuBLASLt); only
    // their CPU expert support is recorded here.
    {.type = 39, .family = "mxfp4", .cpu_expert = true},
    {.type = 40, .family = "nvfp4", .cpu_expert = true},
};

inline constexpr const QwenFormatKernels* qwen_format(std::uint32_t type) {
    for (const auto& format : kQwenFormats)
        if (format.type == type) return &format;
    return nullptr;
}

}  // namespace colibri::v2
