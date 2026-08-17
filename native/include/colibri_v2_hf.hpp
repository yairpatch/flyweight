#pragma once

// HF checkpoint loading: `config.json` + sharded safetensors -> the same
// descriptors the GGUF path produces.
//
// Two things here are not mechanical.
//
// 1. Expert stacking. GGUF packs a layer's routed experts into one 3-D tensor
//    (`ffn_gate_exps.weight`) and the expert cache indexes into that block by
//    offset. HF stores one tensor per expert per projection -- and on this
//    checkpoint a single layer's 128 experts are spread across *8 different
//    shard mappings*. So a stacked descriptor cannot be one base pointer plus
//    an offset, which is all `Tensor::source` can express. Hence `Part`: a
//    descriptor may name an ordered list of disjoint source ranges whose
//    concatenation is the tensor. Single-part descriptors are the common case
//    and stay exactly as cheap as before.
//
// 2. Name translation. Everything downstream looks tensors up by the GGUF
//    `blk.N.*` convention, so the HF names are rewritten at load rather than
//    teaching every consumer a second vocabulary.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "colibri_v2_config.hpp"
#include "colibri_v2_json.hpp"
#include "colibri_v2_provider.hpp"
#include "colibri_v2_safetensors.hpp"

namespace colibri::v2::hf {

// One contiguous run of bytes inside one mapping.
struct Part {
    const std::uint8_t* source = nullptr;
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
};

// A descriptor plus where its bytes live. `parts` is authoritative; a
// single-element list is an ordinary tensor.
struct HfTensor : TensorDescriptor {
    std::vector<Part> parts;

    bool contiguous() const { return parts.size() == 1; }
    // Copies the tensor out regardless of how many mappings it spans. This is
    // the only correct way to read a stacked expert block; consumers that
    // alias straight into a mapping must check `contiguous()` first.
    int read(void* destination, std::uint64_t bytes) const {
        if (!destination || bytes < size) return -1;
        auto* out = static_cast<std::uint8_t*>(destination);
        for (const auto& part : parts) {
            if (!part.source) return -1;
            std::memcpy(out, part.source + part.offset, part.size);
            out += part.size;
        }
        return 0;
    }

    // A pointer to [begin, begin+bytes) when that range lies inside a single
    // part, and null when it straddles two. Quantization walks a tensor in
    // tiles, and the overwhelming majority of tiles are interior to one part
    // even on a stacked expert block -- so this is what lets the common tile
    // read the mapping directly instead of copying first.
    const std::uint8_t* window(std::uint64_t begin, std::uint64_t bytes) const {
        if (bytes > size || begin > size - bytes) return nullptr;
        std::uint64_t at = 0;
        for (const auto& part : parts) {
            if (begin < at + part.size) {
                if (begin + bytes > at + part.size) return nullptr;
                return part.source ? part.source + part.offset + (begin - at) : nullptr;
            }
            at += part.size;
        }
        return nullptr;
    }

    // Copies an arbitrary byte range, crossing parts as needed.
    bool read_range(std::uint64_t begin, std::uint64_t bytes, void* destination) const {
        if (!destination || bytes > size || begin > size - bytes) return false;
        auto* out = static_cast<std::uint8_t*>(destination);
        std::uint64_t at = 0;
        for (const auto& part : parts) {
            if (!bytes) break;
            const std::uint64_t end = at + part.size;
            if (begin < end) {
                if (!part.source) return false;
                const std::uint64_t from = begin - at;
                const std::uint64_t take = std::min(bytes, part.size - from);
                std::memcpy(out, part.source + part.offset + from, take);
                out += take;
                begin += take;
                bytes -= take;
            }
            at = end;
        }
        return bytes == 0;
    }
};

// ---------------------------------------------------------------------------
// config.json -> ModelConfig
// ---------------------------------------------------------------------------

// `bailing_hybrid` / BailingMoeV3 (Ling 3.0). The upstream llama.cpp PR calls
// this `bailingmoe3`; prometheusAIR's GGUFs call it `bailing-hybrid`. Accept
// either spelling on input, emit the upstream one.
inline bool is_bailing_hybrid(const std::string& model_type,
                              const std::string& architecture) {
    return model_type == "bailing_hybrid" || model_type == "bailingmoe3" ||
           model_type == "bailing-hybrid" || architecture == "BailingMoeV3ForCausalLM";
}

// Qwen3.5 (`Qwen3_5ForConditionalGeneration`). The checkpoint is the
// vision-language one: `text_config` holds the decoder this runtime implements
// and `vision_config` a tower it does not, so the visual tensors are dropped
// rather than mapped. The GGUFs of this family call the text model `qwen35`.
inline bool is_qwen3_5(const std::string& model_type,
                       const std::string& architecture) {
    return model_type == "qwen3_5" || model_type == "qwen3_5_text" ||
           model_type == "qwen35" ||
           architecture == "Qwen3_5ForConditionalGeneration" ||
           architecture == "Qwen3_5ForCausalLM";
}

// Everything the decoder needs comes out of `text_config`; the SSM widths do
// not, because the layer plan reads those off the tensor shapes instead (see
// qwen_forward_rows, which derives channels, value_heads and the conv kernel
// from attn_qkv, ssm_a and ssm_conv1d).
// INCOMPLETE. The names and the config below are right -- verified against the
// 27B release, where all 1199 tensors resolve and the 866 kept are set-equal to
// that model's GGUF -- but three numeric transformations the GGUF conversion
// applies are not implemented here, so the weights land in the wrong form and
// the model generates noise. Measured against the GGUF, on several layers:
//
//   perm[i] = (i % 16) * 3 + i / 16        (48 value heads over 16 key heads)
//   ssm_dt.bias = dt_bias[perm]
//   ssm_a       = -exp(A_log[perm])
//   attn_norm, post_attention_norm, output_norm = hf + 1   (ssm_norm does not)
//
// The permutation additionally has to reach the rows of in_proj_a and in_proj_b
// and the per-head blocks inside in_proj_qkv and in_proj_z, which is the part
// that needs care rather than transcription.
//
// Until that lands the path refuses, because a checkpoint that loads and then
// produces plausible-looking rubbish is worse than one that will not open.
inline void require_qwen3_5_opt_in() {
    const char* allow = std::getenv("COLIBRI_HF_QWEN35_INCOMPLETE");
    if (allow && allow[0] == '1') return;
    throw std::runtime_error(
        "Qwen3.5 safetensors loading is incomplete: tensor names and config "
        "translate correctly, but the gated-delta head permutation, the "
        "-exp(A_log) transform and the +1 on the RMS norms are not applied, so "
        "the model would generate noise. Use a GGUF of this checkpoint. Set "
        "COLIBRI_HF_QWEN35_INCOMPLETE=1 to load it anyway (for development).");
}

inline ModelConfig config_from_qwen3_5(const json::Value& config) {
    require_qwen3_5_opt_in();
    const auto& text =
        config.contains("text_config") ? config["text_config"] : config;
    ModelConfig out;
    out.architecture = "qwen35";
    out.hidden_size = static_cast<std::uint32_t>(text["hidden_size"].as_uint());
    out.layer_count =
        static_cast<std::uint32_t>(text["num_hidden_layers"].as_uint());
    out.attention_heads =
        static_cast<std::uint32_t>(text["num_attention_heads"].as_uint());
    out.attention_kv_heads =
        static_cast<std::uint32_t>(text["num_key_value_heads"].as_uint());
    out.context_length =
        static_cast<std::uint32_t>(text["max_position_embeddings"].as_uint());
    out.vocabulary_size = static_cast<std::uint32_t>(text["vocab_size"].as_uint());
    out.intermediate_size =
        static_cast<std::uint32_t>(text["intermediate_size"].as_uint());
    out.dense_intermediate_size = out.intermediate_size;
    out.rms_norm_epsilon =
        static_cast<float>(text["rms_norm_eps"].as_double(1e-6));
    out.attention_head_dim =
        static_cast<std::uint32_t>(text["head_dim"].as_uint());
    out.key_length = out.attention_head_dim;
    out.value_length = out.attention_head_dim;
    out.conv_kernel =
        static_cast<std::uint32_t>(text["linear_conv_kernel_dim"].as_uint(4));

    // rope_theta and the partial factor live under `rope_parameters` on this
    // family rather than at the top level.
    const auto& rope = text.contains("rope_parameters")
        ? text["rope_parameters"] : text;
    out.rope_freq_base = static_cast<float>(rope["rope_theta"].as_double(1.0e7));
    const auto partial = rope["partial_rotary_factor"].as_double(1.0);
    out.rotary_dimension = static_cast<std::uint32_t>(
        static_cast<double>(out.attention_head_dim) * partial);

    // One full-attention layer closes each group of `full_attention_interval`;
    // the rest are gated-delta. True marks the linear layers, matching the
    // convention the bailing path above uses.
    out.full_attention_interval =
        static_cast<std::uint32_t>(text["full_attention_interval"].as_uint(4));
    if (out.full_attention_interval) {
        out.sliding_window_pattern.assign(out.layer_count, 0);
        for (std::uint32_t layer = 0; layer < out.layer_count; ++layer)
            out.sliding_window_pattern[layer] =
                ((layer + 1) % out.full_attention_interval == 0) ? 0 : 1;
    }

    // config.json's eos is <|endoftext|>; the turn actually ends on <|im_end|>,
    // which only generation_config.json names (and lists first). The caller
    // overrides this when that file is present -- see hf_generation_eos.
    if (text.contains("eos_token_id"))
        out.eos_token_id =
            static_cast<std::uint32_t>(text["eos_token_id"].as_uint());
    if (text.contains("bos_token_id"))
        out.bos_token_id =
            static_cast<std::uint32_t>(text["bos_token_id"].as_uint());
    return out;
}

// First entry of generation_config.json's eos_token_id, which is the real
// end-of-turn token where config.json carries end-of-document. Returns false
// when the file says nothing useful, leaving the config value alone.
inline bool hf_generation_eos(const json::Value& generation, std::uint32_t& out) {
    if (!generation.contains("eos_token_id")) return false;
    const auto& eos = generation["eos_token_id"];
    if (eos.kind == json::Kind::Array) {
        if (eos.size() == 0) return false;
        out = static_cast<std::uint32_t>(eos[0].as_uint());
        return true;
    }
    out = static_cast<std::uint32_t>(eos.as_uint());
    return true;
}

inline ModelConfig config_from_json(const json::Value& config) {
    const auto model_type = config["model_type"].as_string();
    const auto architecture = config["architectures"][0].as_string();

    if (is_qwen3_5(model_type, architecture))
        return config_from_qwen3_5(config);

    ModelConfig out;
    if (!is_bailing_hybrid(model_type, architecture))
        throw std::runtime_error(
            "unsupported HF architecture \"" +
            (model_type.empty() ? architecture : model_type) + "\"");
    out.architecture = "bailingmoe3";

    out.hidden_size = static_cast<std::uint32_t>(config["hidden_size"].as_uint());
    out.layer_count = static_cast<std::uint32_t>(config["num_hidden_layers"].as_uint());
    out.attention_heads = static_cast<std::uint32_t>(config["num_attention_heads"].as_uint());
    out.attention_kv_heads = static_cast<std::uint32_t>(config["num_key_value_heads"].as_uint());
    out.context_length = static_cast<std::uint32_t>(config["max_position_embeddings"].as_uint());
    out.vocabulary_size = static_cast<std::uint32_t>(config["vocab_size"].as_uint());
    out.rms_norm_epsilon = static_cast<float>(config["rms_norm_eps"].as_double(1e-6));
    out.rope_freq_base = static_cast<float>(config["rope_theta"].as_double());

    // Dense FFN width (layer 0) vs routed expert width. `intermediate_size` is
    // the dense one; the MoE layers use `moe_intermediate_size`.
    out.intermediate_size = static_cast<std::uint32_t>(config["intermediate_size"].as_uint());
    out.dense_intermediate_size = out.intermediate_size;
    out.expert_intermediate_size =
        static_cast<std::uint32_t>(config["moe_intermediate_size"].as_uint());
    out.expert_count = static_cast<std::uint32_t>(config["num_experts"].as_uint());
    out.expert_used_count =
        static_cast<std::uint32_t>(config["num_experts_per_tok"].as_uint());
    out.expert_shared_count =
        static_cast<std::uint32_t>(config["num_shared_experts"].as_uint());
    // The shared expert's width is per-expert; the module is built at
    // shared_intermediate * num_shared (modeling_bailing_moe_v3.py:420-423).
    out.expert_shared_intermediate_size = static_cast<std::uint32_t>(
        config["moe_shared_expert_intermediate_size"].as_uint() *
        (out.expert_shared_count ? out.expert_shared_count : 1));
    out.leading_dense_block_count =
        static_cast<std::uint32_t>(config["first_k_dense_replace"].as_uint());

    // Router: sigmoid scoring (`score_function`/`scoring_func`), normalized
    // over the selected experts, then scaled. Gating func 2 = sigmoid, matching
    // the GGUF llama_expert_gating_func_type encoding.
    const auto score = config["score_function"].as_string(
        config["scoring_func"].as_string("sigmoid"));
    out.expert_gating_func = score == "softmax" ? 1u : 2u;
    out.expert_weights_norm = config["norm_topk_prob"].as_bool(true);
    // Group-limited top-k. `topk_method` is "noaux_tc" on this family; the two
    // counts are what make the router select over groups rather than a flat
    // expert list, so they are not optional decoration.
    out.expert_group_count = static_cast<std::uint32_t>(config["n_group"].as_uint());
    out.expert_group_used = static_cast<std::uint32_t>(config["topk_group"].as_uint());
    out.expert_weights_scale =
        static_cast<float>(config["routed_scaling_factor"].as_double(1.0));

    // MLA. qk_head_dim = qk_nope + qk_rope, and the attention scale is over
    // that total, not over head_dim (modeling:636).
    out.q_lora_rank = static_cast<std::uint32_t>(config["q_lora_rank"].as_uint());
    out.kv_lora_rank = static_cast<std::uint32_t>(config["kv_lora_rank"].as_uint());
    out.key_length = static_cast<std::uint32_t>(config["qk_head_dim"].as_uint());
    out.value_length = static_cast<std::uint32_t>(config["v_head_dim"].as_uint());
    out.rotary_dimension =
        static_cast<std::uint32_t>(config["qk_rope_head_dim"].as_uint());
    out.attention_head_dim =
        static_cast<std::uint32_t>(config["head_dim"].as_uint());

    // The 3:1 cadence. A layer is full attention when it closes a group, or
    // when it sits past the last whole group -- the second clause is dead at
    // 24 layers / group 4 but fires on layer counts that are not a multiple
    // (modeling:1004-1009), so it is carried rather than assumed away.
    const auto group = static_cast<std::uint32_t>(config["layer_group_size"].as_uint(1));
    out.full_attention_interval = group;
    if (group) {
        const std::uint32_t whole = out.layer_count / group * group;
        out.sliding_window_pattern.assign(out.layer_count, 0);
        for (std::uint32_t layer = 0; layer < out.layer_count; ++layer) {
            const bool full = ((layer + 1) % group == 0) || layer >= whole;
            // True marks the *linear* (KDA) layers here: the pattern names the
            // non-full layers, as it does for the sliding-window architectures.
            out.sliding_window_pattern[layer] = full ? 0 : 1;
        }
    }

    // Per-layer SwiGLU clamp. Null on Ling-3.0-tiny but present on the flash
    // checkpoints -- llama.cpp shipped a metadata repair script for GGUFs that
    // omitted it, so the absence here is not evidence it can be dropped.
    const auto& clamp = config["expert_swiglu_limit_list"];
    if (clamp.kind == json::Kind::Array)
        for (std::size_t i = 0; i < clamp.size(); ++i)
            out.swiglu_clamp_exp.push_back(static_cast<float>(clamp[i].as_double()));
    const auto& shared_clamp = config["share_expert_swiglu_limit_list"];
    if (shared_clamp.kind == json::Kind::Array)
        for (std::size_t i = 0; i < shared_clamp.size(); ++i)
            out.swiglu_clamp_shexp.push_back(
                static_cast<float>(shared_clamp[i].as_double()));

    if (config.contains("eos_token_id"))
        out.eos_token_id = static_cast<std::uint32_t>(config["eos_token_id"].as_uint());
    if (config.contains("bos_token_id"))
        out.bos_token_id = static_cast<std::uint32_t>(config["bos_token_id"].as_uint());

    return out;
}

// ---------------------------------------------------------------------------
// name translation
// ---------------------------------------------------------------------------

// Non-layer tensors.
inline const std::map<std::string, std::string>& global_names() {
    static const std::map<std::string, std::string> table = {
        {"model.word_embeddings.weight", "token_embd.weight"},
        {"model.norm.weight", "output_norm.weight"},
        {"lm_head.weight", "output.weight"},
    };
    return table;
}

// Per-layer suffix translation, shared across both layer kinds. KDA tensors
// take the `ssm_*` names the Qwen3-Next gated-delta path already uses here, so
// the two linear-attention implementations share vocabulary; the MLA names
// follow llama.cpp's DeepSeek-2 convention.
//
// NOTE: if these GGUFs are ever to interoperate with llama.cpp's `bailingmoe3`,
// the KDA suffixes must be reconciled with whatever PR #26608 lands on. They
// are an internal convention until then.
inline const std::map<std::string, std::string>& layer_names() {
    static const std::map<std::string, std::string> table = {
        {"input_layernorm.weight", "attn_norm.weight"},
        {"post_attention_layernorm.weight", "ffn_norm.weight"},

        // MLA (full-attention layers)
        {"attention.q_a_proj.weight", "attn_q_a.weight"},
        {"attention.q_a_layernorm.weight", "attn_q_a_norm.weight"},
        {"attention.q_b_proj.weight", "attn_q_b.weight"},
        {"attention.kv_a_proj_with_mqa.weight", "attn_kv_a_mqa.weight"},
        {"attention.kv_a_layernorm.weight", "attn_kv_a_norm.weight"},
        {"attention.kv_b_proj.weight", "attn_kv_b.weight"},
        {"attention.dense.weight", "attn_output.weight"},

        // KDA (linear-attention layers)
        {"attention.q_proj.weight", "ssm_q.weight"},
        {"attention.k_proj.weight", "ssm_k.weight"},
        {"attention.v_proj.weight", "ssm_v.weight"},
        {"attention.q_conv1d.weight", "ssm_q_conv1d.weight"},
        {"attention.k_conv1d.weight", "ssm_k_conv1d.weight"},
        {"attention.v_conv1d.weight", "ssm_v_conv1d.weight"},
        {"attention.f_proj.weight", "ssm_f.weight"},
        {"attention.b_proj.weight", "ssm_b.weight"},
        {"attention.A_log", "ssm_a"},
        {"attention.dt_bias", "ssm_dt.bias"},
        {"attention.o_norm.weight", "ssm_norm.weight"},
        {"attention.o_proj.weight", "ssm_out.weight"},

        // Feed-forward: dense (layer 0), router, shared expert.
        {"mlp.gate_proj.weight", "ffn_gate.weight"},
        {"mlp.up_proj.weight", "ffn_up.weight"},
        {"mlp.down_proj.weight", "ffn_down.weight"},
        {"mlp.gate.weight", "ffn_gate_inp.weight"},
        {"mlp.gate.expert_bias", "exp_probs_b.bias"},
        {"mlp.shared_experts.gate_proj.weight", "ffn_gate_shexp.weight"},
        {"mlp.shared_experts.up_proj.weight", "ffn_up_shexp.weight"},
        {"mlp.shared_experts.down_proj.weight", "ffn_down_shexp.weight"},
    };
    return table;
}

// The three routed-expert projections, in HF spelling -> GGUF stacked name.
inline const std::map<std::string, std::string>& expert_names() {
    static const std::map<std::string, std::string> table = {
        {"gate_proj.weight", "ffn_gate_exps.weight"},
        {"up_proj.weight", "ffn_up_exps.weight"},
        {"down_proj.weight", "ffn_down_exps.weight"},
    };
    return table;
}

// `attention.g_proj.weight` exists on both layer kinds and means different
// things: the head-wise attention output gate on MLA layers (hidden -> heads),
// the KDA output gate on linear layers (hidden -> heads*head_dim). They must
// not collide, so the caller disambiguates by layer kind.
inline std::string gate_name(bool full_attention) {
    return full_attention ? "attn_gate.weight" : "ssm_g.weight";
}

// Qwen3.5 name tables. Layer tensors sit under
// `model.language_model.layers.N.`, the MTP block under `mtp.`, and the vision
// tower under `model.visual.`.
//
// torch stores a Linear as [out, in] row-major, which is byte-identical to
// GGUF's ne = [in, out]; nothing here needs transposing. The one exception is
// linear_attn.conv1d.weight, [channels, 1, kernel] against the GGUF's
// [kernel, channels] -- same bytes, one singleton dimension to drop.
inline const std::map<std::string, std::string>& qwen3_5_global_names() {
    static const std::map<std::string, std::string> table = {
        {"model.language_model.embed_tokens.weight", "token_embd.weight"},
        {"model.language_model.norm.weight", "output_norm.weight"},
        {"lm_head.weight", "output.weight"},
    };
    return table;
}

inline const std::map<std::string, std::string>& qwen3_5_layer_names() {
    static const std::map<std::string, std::string> table = {
        {"input_layernorm.weight", "attn_norm.weight"},
        {"post_attention_layernorm.weight", "post_attention_norm.weight"},

        // Full-attention layers.
        {"self_attn.q_proj.weight", "attn_q.weight"},
        {"self_attn.k_proj.weight", "attn_k.weight"},
        {"self_attn.v_proj.weight", "attn_v.weight"},
        {"self_attn.o_proj.weight", "attn_output.weight"},
        {"self_attn.q_norm.weight", "attn_q_norm.weight"},
        {"self_attn.k_norm.weight", "attn_k_norm.weight"},

        // Gated-delta layers. in_proj_a/in_proj_b are the per-head alpha and
        // beta projections (hidden -> value_heads); in_proj_z is the output
        // gate, which the GGUF calls attn_gate on these layers.
        {"linear_attn.in_proj_qkv.weight", "attn_qkv.weight"},
        {"linear_attn.in_proj_z.weight", "attn_gate.weight"},
        {"linear_attn.in_proj_a.weight", "ssm_alpha.weight"},
        {"linear_attn.in_proj_b.weight", "ssm_beta.weight"},
        {"linear_attn.out_proj.weight", "ssm_out.weight"},
        {"linear_attn.conv1d.weight", "ssm_conv1d.weight"},
        {"linear_attn.norm.weight", "ssm_norm.weight"},
        {"linear_attn.dt_bias", "ssm_dt.bias"},
        {"linear_attn.A_log", "ssm_a"},

        {"mlp.gate_proj.weight", "ffn_gate.weight"},
        {"mlp.up_proj.weight", "ffn_up.weight"},
        {"mlp.down_proj.weight", "ffn_down.weight"},
    };
    return table;
}

// The MTP block's own tensors, which land on the block past the decoder stack.
inline const std::map<std::string, std::string>& qwen3_5_mtp_names() {
    static const std::map<std::string, std::string> table = {
        {"fc.weight", "nextn.eh_proj.weight"},
        {"pre_fc_norm_embedding.weight", "nextn.enorm.weight"},
        {"pre_fc_norm_hidden.weight", "nextn.hnorm.weight"},
        {"norm.weight", "nextn.shared_head_norm.weight"},
    };
    return table;
}

struct ParsedName {
    bool matched = false;
    // Recognised and deliberately dropped, as against unrecognised: the
    // vision tower is 333 tensors this runtime has no kernels for, and
    // silently ignoring an unmapped name would hide a real gap.
    bool skip = false;
    std::string gguf;       // translated name
    std::uint32_t layer = 0;
    bool is_expert = false;
    std::uint32_t expert = 0;
};

// Peels `<prefix><digits>.` off `name`, yielding the index and the remainder.
inline bool split_indexed(const std::string& name, const std::string& prefix,
                          std::uint32_t& index, std::string& rest) {
    if (name.rfind(prefix, 0) != 0) return false;
    std::size_t cursor = prefix.size();
    if (cursor >= name.size() ||
        !std::isdigit(static_cast<unsigned char>(name[cursor])))
        return false;
    index = 0;
    while (cursor < name.size() &&
           std::isdigit(static_cast<unsigned char>(name[cursor])))
        index = index * 10 + static_cast<std::uint32_t>(name[cursor++] - '0');
    if (cursor >= name.size() || name[cursor] != '.') return false;
    rest = name.substr(cursor + 1);
    return true;
}

// Qwen3.5. The MTP block is emitted as the block just past the decoder stack,
// which is where the runtime looks for it -- a block is recognised as MTP by
// carrying `.nextn.` tensors, so its transformer half takes ordinary attention
// and feed-forward names on that same index.
inline ParsedName translate_qwen3_5(const std::string& name,
                                    const ModelConfig& config) {
    ParsedName parsed;

    // The vision tower and its merger: recognised, not mapped.
    if (name.rfind("model.visual.", 0) == 0) {
        parsed.matched = true;
        parsed.skip = true;
        return parsed;
    }

    const auto& globals = qwen3_5_global_names();
    if (const auto found = globals.find(name); found != globals.end()) {
        parsed.matched = true;
        parsed.gguf = found->second;
        return parsed;
    }

    const std::uint32_t mtp_block = config.layer_count;
    std::uint32_t index = 0;
    std::string rest;

    // mtp.<name> and mtp.layers.<N>.<rest> both land on mtp_block.
    static const std::string kMtp = "mtp.";
    if (name.rfind(kMtp, 0) == 0) {
        const auto tail = name.substr(kMtp.size());
        const auto& mtp = qwen3_5_mtp_names();
        if (const auto found = mtp.find(tail); found != mtp.end()) {
            parsed.matched = true;
            parsed.layer = mtp_block;
            parsed.gguf = "blk." + std::to_string(mtp_block) + "." + found->second;
            return parsed;
        }
        if (split_indexed(tail, "layers.", index, rest)) {
            const auto& layers = qwen3_5_layer_names();
            if (const auto found = layers.find(rest); found != layers.end()) {
                parsed.matched = true;
                parsed.layer = mtp_block;
                parsed.gguf =
                    "blk." + std::to_string(mtp_block) + "." + found->second;
                return parsed;
            }
        }
        return parsed;
    }

    if (!split_indexed(name, "model.language_model.layers.", index, rest))
        return parsed;
    const auto& layers = qwen3_5_layer_names();
    const auto found = layers.find(rest);
    if (found == layers.end()) return parsed;
    parsed.matched = true;
    parsed.layer = index;
    parsed.gguf = "blk." + std::to_string(index) + "." + found->second;
    return parsed;
}

// Splits `model.layers.<N>.<rest>` and translates. `full_attention` decides the
// two ambiguous g_proj cases.
inline ParsedName translate(const std::string& name,
                            const ModelConfig& config) {
    if (config.architecture == "qwen35") return translate_qwen3_5(name, config);
    const auto& linear_layer = config.sliding_window_pattern;
    ParsedName parsed;
    const auto& globals = global_names();
    if (const auto found = globals.find(name); found != globals.end()) {
        parsed.matched = true;
        parsed.gguf = found->second;
        return parsed;
    }

    static const std::string kPrefix = "model.layers.";
    if (name.rfind(kPrefix, 0) != 0) return parsed;
    std::size_t cursor = kPrefix.size();
    std::uint32_t layer = 0;
    if (cursor >= name.size() || !std::isdigit(static_cast<unsigned char>(name[cursor])))
        return parsed;
    while (cursor < name.size() && std::isdigit(static_cast<unsigned char>(name[cursor])))
        layer = layer * 10 + static_cast<std::uint32_t>(name[cursor++] - '0');
    if (cursor >= name.size() || name[cursor] != '.') return parsed;
    const std::string rest = name.substr(cursor + 1);
    parsed.layer = layer;

    const bool full = layer >= linear_layer.size() || !linear_layer[layer];

    static const std::string kExpert = "mlp.experts.";
    if (rest.rfind(kExpert, 0) == 0) {
        std::size_t at = kExpert.size();
        std::uint32_t expert = 0;
        if (at >= rest.size() || !std::isdigit(static_cast<unsigned char>(rest[at])))
            return parsed;
        while (at < rest.size() && std::isdigit(static_cast<unsigned char>(rest[at])))
            expert = expert * 10 + static_cast<std::uint32_t>(rest[at++] - '0');
        if (at >= rest.size() || rest[at] != '.') return parsed;
        const auto projection = rest.substr(at + 1);
        const auto& experts = expert_names();
        const auto found = experts.find(projection);
        if (found == experts.end()) return parsed;
        parsed.matched = true;
        parsed.is_expert = true;
        parsed.expert = expert;
        parsed.gguf = "blk." + std::to_string(layer) + "." + found->second;
        return parsed;
    }

    if (rest == "attention.g_proj.weight") {
        parsed.matched = true;
        parsed.gguf = "blk." + std::to_string(layer) + "." + gate_name(full);
        return parsed;
    }

    const auto& layers = layer_names();
    const auto found = layers.find(rest);
    if (found == layers.end()) return parsed;
    parsed.matched = true;
    parsed.gguf = "blk." + std::to_string(layer) + "." + found->second;
    return parsed;
}

// ---------------------------------------------------------------------------
// tokenizer.json
// ---------------------------------------------------------------------------

// What the runtime's tokenizer needs, in the same shape the GGUF metadata
// path produces it. GGUF stores vocabulary entries already byte-level encoded
// (the "Ġ" spelling), which is exactly how tokenizer.json spells them too, so
// the strings pass through untouched.
struct Tokenizer {
    std::vector<std::string> vocabulary;   // indexed by token id
    std::vector<std::string> merges;       // "left right", ordered by rank
    std::vector<std::uint32_t> token_types;  // 1 = normal, 3 = control
    std::string pre;                       // pre-tokenizer selector
};

// Maps the HF pre-tokenizer regex onto one of the transcribed pre-tokenizers.
//
// Matching on the pattern text rather than on a model name is deliberate: the
// name is a label a publisher chooses, the regex is what the tokenizer actually
// does. GPT-4o and llama3 differ by exactly one quantifier, and getting it
// wrong mis-splits every multi-digit number without ever failing.
inline std::string pretokenizer_from_regex(const std::string& pattern) {
    if (pattern.empty()) return {};
    // Qwen2 family (Qwen2 through Qwen3.5): the contraction alternation is
    // spelled with apostrophes where the GPT one uses a bracketed [sdmt]. The
    // GGUFs of this family carry "qwen35", and returning the same name is what
    // keeps the two loaders on one splitter rather than two that agree by
    // accident.
    if (pattern.find("(?i:'s|'t|'re|'ve|'m|'ll|'d)") != std::string::npos)
        return "qwen35";
    const bool gpt_family =
        pattern.find("(?i:[sdmt]|ll|ve|re)") != std::string::npos;
    if (!gpt_family) return {};
    return pattern.find("\\p{N}{1,3}") != std::string::npos ? "llama4"
                                                           : "llama-bpe";
}

// Digs the split pattern out of whatever pre_tokenizer shape the file uses:
// either a bare Split or a Sequence whose first entry is one.
inline std::string split_pattern(const json::Value& pre_tokenizer) {
    const auto direct = pre_tokenizer["pattern"]["Regex"].as_string();
    if (!direct.empty()) return direct;
    const auto& sequence = pre_tokenizer["pretokenizers"];
    for (std::size_t i = 0; i < sequence.size(); ++i) {
        const auto pattern = sequence[i]["pattern"]["Regex"].as_string();
        if (!pattern.empty()) return pattern;
    }
    return {};
}

// Pulls the chat template out of a parsed tokenizer_config.json.
//
// Only checkpoints published after the split carry chat_template.jinja; older
// ones keep the template in tokenizer_config.json, under either a bare string
// or the multi-template array shape ([{name, template}, ...]) that predates it.
// A checkpoint with neither renders through the runtime's generic fallback
// markup, which is markup no Bailing turn was trained on, so it is worth
// reading both.
inline std::string chat_template_from_tokenizer_config(
        const json::Value& document) {
    const auto& value = document["chat_template"];
    if (value.kind == json::Kind::String) return value.string;
    if (value.kind != json::Kind::Array) return {};
    std::string first;
    for (std::size_t i = 0; i < value.size(); ++i) {
        const auto text = value[i]["template"].as_string();
        if (text.empty()) continue;
        if (value[i]["name"].as_string() == "default") return text;
        if (first.empty()) first = text;
    }
    return first;
}

inline Tokenizer tokenizer_from_json(const json::Value& document,
                                     std::uint32_t vocabulary_size) {
    Tokenizer out;
    const auto& model = document["model"];
    const auto type = model["type"].as_string();
    if (type != "BPE")
        throw std::runtime_error("unsupported HF tokenizer model \"" + type +
                                 "\" (only byte-level BPE is supported)");

    // The vocabulary is a token -> id map, and the ids are not dense: added
    // tokens sit above the base vocabulary and the embedding table is usually
    // padded past both. Size to the model's own vocabulary_size so the holes
    // are explicit rather than shifting every id after them.
    const auto& vocabulary = model["vocab"];
    std::uint64_t highest = 0;
    for (const auto& [token, id] : vocabulary.object)
        highest = std::max(highest, id.as_uint());
    const auto& added = document["added_tokens"];
    for (std::size_t i = 0; i < added.size(); ++i)
        highest = std::max(highest, added[i]["id"].as_uint());

    const auto count = std::max<std::uint64_t>(vocabulary_size, highest + 1);
    out.vocabulary.assign(static_cast<std::size_t>(count), std::string());
    out.token_types.assign(static_cast<std::size_t>(count), 1u);

    for (const auto& [token, id] : vocabulary.object)
        out.vocabulary[static_cast<std::size_t>(id.as_uint())] = token;

    for (std::size_t i = 0; i < added.size(); ++i) {
        const auto& entry = added[i];
        const auto id = static_cast<std::size_t>(entry["id"].as_uint());
        out.vocabulary[id] = entry["content"].as_string();
        // Control tokens are split off ahead of BPE; ordinary added tokens are
        // not. `special` is what draws that line.
        if (entry["special"].as_bool()) out.token_types[id] = 3u;
    }

    // Merges are ordered by rank. Older files spell each as one string, newer
    // ones as a two-element array; both mean the same pair.
    const auto& merges = model["merges"];
    out.merges.reserve(merges.size());
    for (std::size_t i = 0; i < merges.size(); ++i) {
        const auto& merge = merges[i];
        if (merge.kind == json::Kind::String) {
            out.merges.push_back(merge.string);
        } else if (merge.kind == json::Kind::Array && merge.size() == 2) {
            out.merges.push_back(merge[0].as_string() + " " + merge[1].as_string());
        } else {
            throw std::runtime_error("unrecognised merge entry in tokenizer.json");
        }
    }

    out.pre = pretokenizer_from_regex(split_pattern(document["pre_tokenizer"]));
    return out;
}

// ---------------------------------------------------------------------------
// assembly
// ---------------------------------------------------------------------------

// One mapped shard: its parsed header plus the base pointer its offsets are
// relative to.
struct Shard {
    safetensors::Header header;
    const std::uint8_t* base = nullptr;
};

// Merges every shard into one descriptor list under GGUF names, stacking the
// routed experts. Shard order does not matter: experts are placed by their
// parsed index, not by the order they are encountered.
inline std::vector<HfTensor> build_tensors(const std::vector<Shard>& shards,
                                           const ModelConfig& config) {
    std::vector<HfTensor> out;
    // Stacked tensors are assembled out of order, so they are gathered by name
    // first and flushed once every shard has been walked.
    struct Stack {
        std::vector<Part> parts;                 // indexed by expert
        std::vector<std::uint64_t> element_shape;  // per-expert, GGUF order
        std::uint32_t type = 0;
        bool seen = false;
    };
    std::map<std::string, Stack> stacks;

    for (const auto& shard : shards) {
        for (const auto& entry : shard.header.entries) {
            const auto parsed = translate(entry.name, config);
            if (parsed.skip) continue;
            if (!parsed.matched)
                throw std::runtime_error("unmapped HF tensor: " + entry.name);
            // Only now that the tensor is known to be wanted is an
            // undescribable rank an error.
            if (entry.rank_exceeded)
                throw std::runtime_error(
                    "safetensors rank exceeds the v2 ABI: " + entry.name);

            if (!parsed.is_expert) {
                HfTensor tensor;
                tensor.name = parsed.gguf;
                tensor.shape = entry.shape;
                tensor.type = entry.type;
                tensor.size = entry.size;
                // Kept for consumers that still read `offset` directly; only
                // meaningful because this descriptor is single-part.
                tensor.offset = entry.offset;
                tensor.parts.push_back({shard.base, entry.offset, entry.size});
                out.push_back(std::move(tensor));
                continue;
            }

            if (parsed.expert >= config.expert_count)
                throw std::runtime_error("HF expert index is outside num_experts: " +
                                         entry.name);
            auto& stack = stacks[parsed.gguf];
            if (!stack.seen) {
                stack.parts.assign(config.expert_count, Part{});
                stack.element_shape = entry.shape;
                stack.type = entry.type;
                stack.seen = true;
            } else {
                // Every expert in a stack must agree, or the concatenation is
                // not a tensor.
                if (stack.type != entry.type || stack.element_shape != entry.shape)
                    throw std::runtime_error(
                        "HF experts disagree on shape or dtype: " + entry.name);
            }
            if (stack.parts[parsed.expert].source)
                throw std::runtime_error("duplicate HF expert tensor: " + entry.name);
            stack.parts[parsed.expert] = {shard.base, entry.offset, entry.size};
        }
    }

    for (auto& [name, stack] : stacks) {
        HfTensor tensor;
        tensor.name = name;
        tensor.type = stack.type;
        // GGUF stacks experts as a trailing dimension: [inputs, outputs, experts].
        tensor.shape = stack.element_shape;
        tensor.shape.push_back(config.expert_count);
        tensor.size = 0;
        for (std::uint32_t expert = 0; expert < config.expert_count; ++expert) {
            const auto& part = stack.parts[expert];
            if (!part.source)
                throw std::runtime_error("HF checkpoint is missing expert " +
                                         std::to_string(expert) + " of " + name);
            tensor.size += part.size;
        }
        tensor.parts = std::move(stack.parts);
        // A stacked tensor spans several mappings, so there is no single base
        // its offset could be relative to. Anything reading `offset` without
        // checking `contiguous()` is a bug; leave it zero so that bug is loud.
        tensor.offset = 0;
        out.push_back(std::move(tensor));
    }
    return out;
}

}  // namespace colibri::v2::hf
