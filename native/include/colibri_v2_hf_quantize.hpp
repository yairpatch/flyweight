#pragma once

// Load-time quantization for the HF path.
//
// A GGUF arrives pre-quantized; an HF checkpoint arrives as bf16 and has to be
// converted before it will fit anywhere useful. Ling-3.0-tiny is 15.79 GB of
// bf16 -- more than the GPUs this targets have -- so this is not an
// optimization, it is what makes the HF path runnable at all.
//
// This also settles the multi-part descriptor question from the loader. A
// stacked expert tensor spans several shard mappings and cannot be handed to a
// consumer that expects contiguity. Quantization has to copy those bytes
// anyway, so the output arena is where multi-part tensors become contiguous
// again -- and every descriptor this produces is single-part.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "colibri_v2_hf.hpp"
// The decode side is header-only; the packers are not -- they need
// `-ffp-contract=off` and so live in their own translation unit.
#include "qwen_kquant.h"
#include "qwen_kquant_pack_api.hpp"

namespace colibri::v2::hf {

// GGML type codes this can emit.
enum class Target : std::uint32_t {
    F32 = 0,
    Q8_0 = 8,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
};

inline std::uint64_t target_block_bytes(Target target) {
    switch (target) {
        case Target::F32: return 4;
        case Target::Q8_0: return 34;
        case Target::Q4_K: return 144;
        case Target::Q5_K: return 176;
        case Target::Q6_K: return 210;
    }
    throw std::runtime_error("unknown quantization target");
}

inline std::uint64_t target_block_elements(Target target) {
    switch (target) {
        case Target::F32: return 1;
        case Target::Q8_0: return 32;
        case Target::Q4_K:
        case Target::Q5_K:
        case Target::Q6_K: return 256;
    }
    throw std::runtime_error("unknown quantization target");
}

struct Policy {
    // Applied to the bulk 2-D weights, including the stacked experts.
    //
    // Q6_K rather than the smaller Q4_K because the device path is not
    // format-agnostic: the tiled prompt kernels and the grouped expert decode
    // both decode Q6_K only, and a checkpoint packed as Q4_K silently loses
    // BOTH -- prefill collapses to stepping the prompt one token at a time.
    // Measured on Ling-3.0-tiny (RTX 5070 Ti laptop), Q4_K -> Q6_K:
    //   prefill  8192 tokens   63 -> 486 tok/s
    //   prefill  4096 tokens   73 -> 739 tok/s
    //   decode  @8192          54 -> 113 tok/s
    // for 4354 -> 6176 MiB of weights. Paying 1.8 GB of VRAM for 8x prefill is
    // not a close call on any checkpoint that still fits; one that does not fit
    // wants COLIBRI_HF_QUANT=Q4_K, which is why the override stays.
    Target weights = Target::Q6_K;
    // The embedding table and the output head. These two are ~1 GB of a 15.8 GB
    // checkpoint and are read every token, and low-bit embeddings cost more
    // quality per byte saved than the FFN does -- which is why the mixed GGUF
    // recipes keep them higher than the rest.
    Target embedding = Target::Q6_K;
    // Norms, biases and the router. 1-D tensors are tiny, are read at full
    // width every token, and are exactly where quantization error shows up
    // worst, so the GGUF convention keeps them f32. Follow it.
    Target small = Target::F32;
};

// Which target a given tensor gets. Kept as one function so the policy is
// auditable in one place rather than spread through the loop.
inline Target target_for(const HfTensor& tensor, const Policy& policy) {
    if (tensor.name == "token_embd.weight" || tensor.name == "output.weight")
        return policy.embedding;
    // 1-D: norms, dt_bias, A_log, the router bias.
    if (tensor.shape.size() < 2) return policy.small;

    // A row shorter than 256 cannot hold a K-quant super-block -- the block
    // would span two rows, and every matvec kernel here indexes by row. That
    // rules out the K-quants; it does not rule out quantization. Q8_0's block
    // is 32, so any row that is a multiple of 32 fits it exactly.
    //
    // This is not a size optimization. The MLA up-projections (attn_q_b,
    // attn_kv_b) have q_lora_rank-wide rows -- 128 on this family -- and they
    // are matvecs on every full-attention layer. The GPU decode path has no f32
    // matvec kernel at all, so leaving them f32 made `COLIBRI_BAILING_GPU=1`
    // fail outright with "bailing GPU matvec failed for type 0": an HF
    // checkpoint could not use the GPU, while the same model from a GGUF could.
    //
    // ssm_*_conv1d is the case that still lands on f32: shaped
    // [kernel, 1, channels], its row is the 4-tap kernel. It is not a matvec.
    if (tensor.shape[0] < 256) {
        if (policy.weights == Target::F32) return policy.small;
        return tensor.shape[0] >= 32 && tensor.shape[0] % 32 == 0 ? Target::Q8_0
                                                                  : policy.small;
    }

    std::uint64_t elements = 1;
    for (const auto extent : tensor.shape) elements *= extent;
    const auto block = target_block_elements(policy.weights);
    // A partial trailing block would need a padding path that no decoder here
    // implements; fall back rather than emit something unreadable.
    if (block && elements % block) return policy.small;
    return policy.weights;
}

struct QuantizedTensor {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::uint32_t type = 0;
    std::uint64_t offset = 0;  // into the arena
    std::uint64_t size = 0;
};

struct QuantizedModel {
    std::vector<std::uint8_t> arena;
    std::vector<QuantizedTensor> tensors;
    std::uint64_t source_bytes = 0;
};

// Widen whatever the checkpoint stores into f32, which is what every packer
// takes. `type` is the safetensors-derived code: 0=f32, 1=f16, 30=bf16.
inline void widen_to_f32(const std::uint8_t* source, std::uint32_t type,
                         std::uint64_t elements, float* out) {
    if (type == 0) {
        std::memcpy(out, source, elements * 4);
        return;
    }
    const auto* bits = reinterpret_cast<const std::uint16_t*>(source);
    if (type == 30) {
        // bf16 is the top half of an f32.
        for (std::uint64_t i = 0; i < elements; ++i) {
            const std::uint32_t widened = static_cast<std::uint32_t>(bits[i]) << 16;
            std::memcpy(out + i, &widened, 4);
        }
        return;
    }
    if (type == 1) {
        for (std::uint64_t i = 0; i < elements; ++i) out[i] = qwen_half_value(bits[i]);
        return;
    }
    throw std::runtime_error("cannot widen tensor type " + std::to_string(type));
}

// Bytes per element in the checkpoint, which is what turns an element range
// into a byte range in the source mapping.
inline std::uint64_t source_element_bytes(std::uint32_t type) {
    switch (type) {
        case 0: return 4;   // f32
        case 1: return 2;   // f16
        case 30: return 2;  // bf16
        default: break;
    }
    throw std::runtime_error("cannot widen tensor type " + std::to_string(type));
}

inline void pack_to(Target target, const float* values, std::uint64_t elements,
                    std::uint8_t* out) {
    switch (target) {
        case Target::F32: std::memcpy(out, values, elements * 4); return;
        case Target::Q8_0: qwen_kpack::pack_q8_0(values, elements, out); return;
        case Target::Q4_K: qwen_kpack::pack_q4_k(values, elements, out); return;
        case Target::Q5_K: qwen_kpack::pack_q5_k(values, elements, out); return;
        case Target::Q6_K: qwen_kpack::pack_q6_k(values, elements, out); return;
    }
    throw std::runtime_error("unknown quantization target");
}

// Two passes: size the arena from the descriptors (deterministic, no data
// touched), then fill it. The split is what lets the fill run in parallel --
// every tensor's destination is known before any work starts, so the threads
// never coordinate.
inline QuantizedModel quantize(const std::vector<HfTensor>& tensors,
                               const Policy& policy) {
    QuantizedModel model;
    model.tensors.resize(tensors.size());
    std::vector<Target> targets(tensors.size());

    std::uint64_t cursor = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& source = tensors[i];
        std::uint64_t elements = 1;
        for (const auto extent : source.shape) elements *= extent;
        const Target target = target_for(source, policy);
        targets[i] = target;

        const auto block = target_block_elements(target);
        if (block > 1 && elements % block)
            throw std::runtime_error("tensor is not a whole number of blocks: " +
                                     source.name);
        const std::uint64_t bytes =
            block > 1 ? elements / block * target_block_bytes(target)
                      : elements * target_block_bytes(target);

        auto& out = model.tensors[i];
        out.name = source.name;
        out.shape = source.shape;
        out.type = static_cast<std::uint32_t>(target);
        out.offset = cursor;
        out.size = bytes;
        cursor += bytes;
        model.source_bytes += source.size;
    }
    model.arena.resize(cursor);

    // Work items are tiles, not tensors. Two things this buys:
    //
    // * The widen buffer stops being tensor-sized. It used to be a whole-tensor
    //   f32 allocation per iteration -- 400 MB per thread on the 201 MB expert
    //   stacks -- written once and read back once, so every weight made two
    //   round trips through main memory. A tile is 256 KB, stays in L2 between
    //   the widen and the pack, and is allocated once per thread rather than
    //   once per tensor.
    // * The tail goes away. 69 of these tensors are ~100M elements and the
    //   rest are small, so per-tensor items left most threads idle at the end
    //   waiting on the last few expert stacks.
    //
    // K-quant blocks are independent and a tile is a whole number of them, so
    // this is bit-exact against packing each tensor in one call.
    constexpr std::uint64_t kTileElements = 64 * 1024;
    struct Tile {
        std::size_t tensor;
        std::uint64_t element_begin;
        std::uint64_t elements;
    };
    std::vector<Tile> tiles;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        std::uint64_t elements = 1;
        for (const auto extent : tensors[i].shape) elements *= extent;
        // Validated here rather than inside the parallel region, where throwing
        // would cross an OpenMP boundary.
        (void)source_element_bytes(tensors[i].type);
        const auto block = target_block_elements(targets[i]);
        const std::uint64_t stride =
            block > 1 ? kTileElements / block * block : kTileElements;
        for (std::uint64_t begin = 0; begin < elements; begin += stride)
            tiles.push_back({i, begin, std::min(stride, elements - begin)});
    }

    // A part boundary inside a tile is the only thing that can fail here, and
    // build_tensors has already rejected the tensors where that could happen
    // for a missing reason -- so this stays false in practice and exists so a
    // failure is a diagnosable error rather than a silently wrong arena.
    bool gather_failed = false;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<float> tile(kTileElements);
        // Only allocated if some tile straddles two shard mappings.
        std::vector<std::uint8_t> gathered;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (std::int64_t t = 0; t < static_cast<std::int64_t>(tiles.size()); ++t) {
            const auto& item = tiles[static_cast<std::size_t>(t)];
            const auto& source = tensors[item.tensor];
            const auto target = targets[item.tensor];
            const auto width = source_element_bytes(source.type);
            const std::uint64_t byte_begin = item.element_begin * width;
            const std::uint64_t byte_size = item.elements * width;

            // Straight out of the mapping wherever the tile is interior to one
            // part, which is every tile but the few that cross an expert
            // boundary inside a stacked block.
            const std::uint8_t* bytes = source.window(byte_begin, byte_size);
            if (!bytes) {
                gathered.resize(static_cast<std::size_t>(byte_size));
                if (!source.read_range(byte_begin, byte_size, gathered.data())) {
                    gather_failed = true;
                    continue;
                }
                bytes = gathered.data();
            }

            widen_to_f32(bytes, source.type, item.elements, tile.data());
            const auto block = target_block_elements(target);
            const std::uint64_t offset =
                block > 1 ? item.element_begin / block * target_block_bytes(target)
                          : item.element_begin * target_block_bytes(target);
            pack_to(target, tile.data(), item.elements,
                    model.arena.data() + model.tensors[item.tensor].offset + offset);
        }
    }
    if (gather_failed)
        throw std::runtime_error("failed to gather a tensor from its shards");
    return model;
}

}  // namespace colibri::v2::hf
