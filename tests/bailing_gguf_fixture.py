"""Converts the BailingMoE3 safetensors fixture into a GGUF, llama.cpp style.

A GGUF of this architecture is not the same file with a different container:
llama.cpp's converter renames most of the linear-attention tensors, splits the
MLA kv_b projection in two and stores one half transposed, and writes A rather
than A_log. None of that is visible from a tensor name, and getting any of it
wrong produces fluent, wrong text -- so the test built on this fixture converts
the *same weights* and requires both files to give the same logits.

Everything is written f32, so any difference between the two loads is the
mapping and not quantization.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from tests import hf_safetensors_fixture as hf

GGUF_UINT32 = 4
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
GGUF_STRING = 8
GGUF_ARRAY = 9
GGML_F32 = 0
ALIGNMENT = 32

# BailingMoE3 closes every turn with this single control token.
ROLE_END = "<|role_end|>"
ROLE_END_ID = 300


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, kind: int, payload: bytes) -> bytes:
    return _string(key) + struct.pack("<I", kind) + payload


def _uint(value: int) -> bytes:
    return struct.pack("<I", value)


def _uint_array(values) -> bytes:
    return struct.pack("<IQ", GGUF_UINT32, len(values)) + b"".join(
        struct.pack("<I", value) for value in values
    )


def _float_array(values) -> bytes:
    return struct.pack("<IQ", GGUF_FLOAT32, len(values)) + b"".join(
        struct.pack("<f", value) for value in values
    )


def _string_array(values) -> bytes:
    return struct.pack("<IQ", GGUF_STRING, len(values)) + b"".join(
        _string(value) for value in values
    )


def _read_safetensors(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
        body = handle.read()
    out: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        raw = np.frombuffer(body[start:end], dtype=np.uint16)
        values = (raw.astype(np.uint32) << 16).view(np.float32)
        out[name] = values.reshape(info["shape"])
    return out


def _tokenizer_tokens() -> tuple[list[str], list[int], list[str]]:
    """The fixture tokenizer, in the flat form GGUF stores."""
    document = hf._tokenizer()
    vocabulary = document["model"]["vocab"]
    tokens = [""] * hf.VOCAB
    types = [1] * hf.VOCAB
    for token, index in vocabulary.items():
        if index < hf.VOCAB:
            tokens[index] = token
    for added in document["added_tokens"]:
        if added["id"] < hf.VOCAB:
            tokens[added["id"]] = added["content"]
            types[added["id"]] = 3 if added["special"] else 1
    # Ids the fixture vocabulary leaves unused still need a distinct spelling.
    for index, token in enumerate(tokens):
        if not token:
            tokens[index] = f"<|unused{index}|>"
    # The turn terminator is a CONTROL token in the real checkpoint, and that
    # is a property the conversation markup depends on: the server splices it
    # between a reused turn and the next one, so a spelling that tokenizes as
    # ordinary text puts a literal string of junk mid-conversation.
    tokens[ROLE_END_ID] = ROLE_END
    types[ROLE_END_ID] = 3
    merges = [" ".join(pair) for pair in document["model"]["merges"]]
    return tokens, types, merges


def build(directory: Path, source: Path, flash: bool = False,
          clamps: bool = True) -> Path:
    """Writes `<directory>/bailing.gguf` from the safetensors fixture at `source`.

    `flash` mirrors the Ling 3.0 Flash shape: an un-factored MLA query, the
    per-layer SwiGLU clamps, and a trailing MTP draft block that the decoder
    stack must ignore. `clamps=False` writes the same file without the clamp
    metadata, which is what makes "the clamp is read" testable at all.
    """
    directory.mkdir(parents=True, exist_ok=True)
    weights: dict[str, np.ndarray] = {}
    for shard in sorted(source.glob("*.safetensors")):
        weights.update(_read_safetensors(shard))

    heads = hf.HEADS
    qk_nope = hf.CONFIG["qk_nope_head_dim"]
    qk_rope = hf.CONFIG["qk_rope_head_dim"]
    v_head = hf.CONFIG["v_head_dim"]
    kv_lora = hf.CONFIG["kv_lora_rank"]

    tensors: dict[str, np.ndarray] = {
        "token_embd.weight": weights["model.word_embeddings.weight"],
        "output_norm.weight": weights["model.norm.weight"],
        "output.weight": weights["lm_head.weight"],
    }
    for layer in range(hf.LAYERS):
        source_prefix = f"model.layers.{layer}."
        prefix = f"blk.{layer}."
        tensors[prefix + "attn_norm.weight"] = weights[
            source_prefix + "input_layernorm.weight"]
        tensors[prefix + "ffn_norm.weight"] = weights[
            source_prefix + "post_attention_layernorm.weight"]
        full_attention = (layer + 1) % hf.CONFIG["layer_group_size"] == 0
        attention = source_prefix + "attention."
        if full_attention:
            if flash:
                tensors[prefix + "attn_q.weight"] = weights[attention + "q_proj.weight"]
            else:
                tensors[prefix + "attn_q_a.weight"] = weights[
                    attention + "q_a_proj.weight"]
                tensors[prefix + "attn_q_a_norm.weight"] = weights[
                    attention + "q_a_layernorm.weight"]
                tensors[prefix + "attn_q_b.weight"] = weights[
                    attention + "q_b_proj.weight"]
            tensors[prefix + "attn_kv_a_mqa.weight"] = weights[
                attention + "kv_a_proj_with_mqa.weight"]
            tensors[prefix + "attn_kv_a_norm.weight"] = weights[
                attention + "kv_a_layernorm.weight"]
            # kv_b splits per head into a key half -- stored transposed, which
            # is what llama.cpp's absorbed matmul reads -- and a value half.
            kv_b = weights[attention + "kv_b_proj.weight"].reshape(
                heads, qk_nope + v_head, kv_lora)
            tensors[prefix + "attn_k_b.weight"] = np.ascontiguousarray(
                kv_b[:, :qk_nope, :].transpose(0, 2, 1))
            tensors[prefix + "attn_v_b.weight"] = np.ascontiguousarray(
                kv_b[:, qk_nope:, :])
            tensors[prefix + "attn_output.weight"] = weights[attention + "dense.weight"]
            tensors[prefix + "attn_gate.weight"] = weights[attention + "g_proj.weight"]
        else:
            # A linear-attention layer's projections take the ordinary
            # attention names, and its output projection is attn_output -- both
            # of which mean different weights on a full-attention layer.
            tensors[prefix + "attn_q.weight"] = weights[attention + "q_proj.weight"]
            tensors[prefix + "attn_k.weight"] = weights[attention + "k_proj.weight"]
            tensors[prefix + "attn_v.weight"] = weights[attention + "v_proj.weight"]
            tensors[prefix + "attn_output.weight"] = weights[attention + "o_proj.weight"]
            tensors[prefix + "ssm_conv1d_q.weight"] = weights[
                attention + "q_conv1d.weight"]
            tensors[prefix + "ssm_conv1d_k.weight"] = weights[
                attention + "k_conv1d.weight"]
            tensors[prefix + "ssm_conv1d_v.weight"] = weights[
                attention + "v_conv1d.weight"]
            tensors[prefix + "ssm_f_a.weight"] = weights[attention + "f_proj.weight"]
            tensors[prefix + "ssm_beta.weight"] = weights[attention + "b_proj.weight"]
            tensors[prefix + "ssm_g_a.weight"] = weights[attention + "g_proj.weight"]
            # A, not A_log: the converter exponentiates on the way out.
            tensors[prefix + "ssm_a"] = np.exp(
                weights[attention + "A_log"].astype(np.float32))
            tensors[prefix + "ssm_dt.bias"] = weights[attention + "dt_bias"]
            tensors[prefix + "ssm_norm.weight"] = weights[attention + "o_norm.weight"]

        mlp = source_prefix + "mlp."
        if layer < hf.CONFIG["first_k_dense_replace"]:
            tensors[prefix + "ffn_gate.weight"] = weights[mlp + "gate_proj.weight"]
            tensors[prefix + "ffn_up.weight"] = weights[mlp + "up_proj.weight"]
            tensors[prefix + "ffn_down.weight"] = weights[mlp + "down_proj.weight"]
            continue
        tensors[prefix + "ffn_gate_inp.weight"] = weights[mlp + "gate.weight"]
        tensors[prefix + "exp_probs_b.bias"] = weights[mlp + "gate.expert_bias"]
        tensors[prefix + "ffn_gate_shexp.weight"] = weights[
            mlp + "shared_experts.gate_proj.weight"]
        tensors[prefix + "ffn_up_shexp.weight"] = weights[
            mlp + "shared_experts.up_proj.weight"]
        tensors[prefix + "ffn_down_shexp.weight"] = weights[
            mlp + "shared_experts.down_proj.weight"]
        for role in ("gate", "up", "down"):
            tensors[prefix + f"ffn_{role}_exps.weight"] = np.stack([
                weights[mlp + f"experts.{expert}.{role}_proj.weight"]
                for expert in range(hf.EXPERTS)
            ])

    # A trailing MTP draft block: llama.cpp counts it in `block_count`, it sits
    # past the last whole attention group (so its kv-head entry is nonzero),
    # and nothing executes it. It is written at a storage the device path has
    # no kernel for, because the bug it guards against is exactly that -- one
    # unexecuted tensor vetoing the GPU for the whole model.
    blocks = hf.LAYERS + 1 if flash else hf.LAYERS
    if flash:
        draft = f"blk.{hf.LAYERS}."
        tensors[draft + "nextn.enorm.weight"] = weights["model.norm.weight"]
        tensors[draft + "nextn.hnorm.weight"] = weights["model.norm.weight"]
        tensors[draft + "layer_output_norm.weight"] = weights["model.norm.weight"]
        # A complete, runnable draft block, the shape the real Ling 3.0 Flash
        # ships: eh_proj plus a full MLA+MoE layer of its own. The last decoder
        # layer's tensors stand in for the draft layer's -- the merge weights
        # are what the speculative path actually exercises, and sharing the
        # rest keeps the fixture small. The three norm-only entries above stay
        # exactly as they were: they are the stub-tolerance guard.
        last = f"blk.{hf.LAYERS - 1}."
        for name in [n for n in tensors if n.startswith(last)]:
            tensors.setdefault(draft + name[len(last):], tensors[name])
        rng = np.random.default_rng(424242)
        tensors[draft + "nextn.eh_proj.weight"] = (
            rng.standard_normal((hf.HIDDEN, 2 * hf.HIDDEN)).astype(np.float32)
            * 0.05)

    tokens, types, merges = _tokenizer_tokens()
    interval = hf.CONFIG["layer_group_size"]
    metadata = [
        _kv("general.architecture", GGUF_STRING, _string("bailingmoe3")),
        _kv("general.name", GGUF_STRING, _string("bailing-gguf-fixture")),
        _kv("bailingmoe3.block_count", GGUF_UINT32, _uint(blocks)),
        _kv("bailingmoe3.context_length", GGUF_UINT32, _uint(4096)),
        _kv("bailingmoe3.embedding_length", GGUF_UINT32, _uint(hf.HIDDEN)),
        _kv("bailingmoe3.feed_forward_length", GGUF_UINT32, _uint(hf.DENSE_FFN)),
        _kv("bailingmoe3.attention.head_count", GGUF_UINT32, _uint(hf.HEADS)),
        # The full-attention layers are marked by a per-layer array rather than
        # by an interval; the loader reads the period off it.
        _kv("bailingmoe3.attention.head_count_kv", GGUF_ARRAY, _uint_array([
            hf.HEADS
            if (layer + 1) % interval == 0 or layer >= blocks // interval * interval
            else 0
            for layer in range(blocks)
        ])),
        _kv("bailingmoe3.rope.freq_base", GGUF_FLOAT32,
            struct.pack("<f", hf.CONFIG["rope_theta"])),
        _kv("bailingmoe3.attention.layer_norm_rms_epsilon", GGUF_FLOAT32,
            struct.pack("<f", hf.CONFIG["rms_norm_eps"])),
        _kv("bailingmoe3.expert_count", GGUF_UINT32, _uint(hf.EXPERTS)),
        _kv("bailingmoe3.expert_used_count", GGUF_UINT32,
            _uint(hf.CONFIG["num_experts_per_tok"])),
        _kv("bailingmoe3.expert_group_count", GGUF_UINT32, _uint(hf.CONFIG["n_group"])),
        _kv("bailingmoe3.expert_group_used_count", GGUF_UINT32,
            _uint(hf.CONFIG["topk_group"])),
        _kv("bailingmoe3.expert_gating_func", GGUF_UINT32, _uint(2)),
        # The MQA row width, which is NOT the per-head key width below it.
        _kv("bailingmoe3.attention.key_length", GGUF_UINT32, _uint(kv_lora + qk_rope)),
        _kv("bailingmoe3.attention.value_length", GGUF_UINT32, _uint(v_head)),
        _kv("bailingmoe3.attention.key_length_mla", GGUF_UINT32,
            _uint(qk_nope + qk_rope)),
        _kv("bailingmoe3.attention.value_length_mla", GGUF_UINT32, _uint(v_head)),
        _kv("bailingmoe3.vocab_size", GGUF_UINT32, _uint(hf.VOCAB)),
        _kv("bailingmoe3.ssm.conv_kernel", GGUF_UINT32,
            _uint(hf.CONFIG["short_conv_kernel_size"])),
        _kv("bailingmoe3.kda.head_dim", GGUF_UINT32, _uint(hf.HEAD_DIM)),
        _kv("bailingmoe3.attention.kv_lora_rank", GGUF_UINT32, _uint(kv_lora)),
        _kv("bailingmoe3.rope.dimension_count", GGUF_UINT32, _uint(qk_rope)),
        _kv("bailingmoe3.expert_feed_forward_length", GGUF_UINT32, _uint(hf.MOE_FFN)),
        _kv("bailingmoe3.expert_shared_feed_forward_length", GGUF_UINT32,
            _uint(hf.MOE_FFN)),
        _kv("bailingmoe3.expert_shared_count", GGUF_UINT32,
            _uint(hf.CONFIG["num_shared_experts"])),
        _kv("bailingmoe3.leading_dense_block_count", GGUF_UINT32,
            _uint(hf.CONFIG["first_k_dense_replace"])),
        _kv("bailingmoe3.expert_weights_scale", GGUF_FLOAT32,
            struct.pack("<f", hf.CONFIG["routed_scaling_factor"])),
        _kv("bailingmoe3.expert_weights_norm", GGUF_BOOL, struct.pack("<B", 1)),
    ]
    if hf.CONFIG["q_lora_rank"] and not flash:
        metadata.append(_kv("bailingmoe3.attention.q_lora_rank", GGUF_UINT32,
                            _uint(hf.CONFIG["q_lora_rank"])))
    if flash and clamps:
        # One entry per BLOCK, draft block included -- the arrays are as long
        # as block_count, not as the executed stack.
        for key, values in (
            ("swiglu_clamp_exp", hf.FLASH_OVERRIDES["expert_swiglu_limit_list"]),
            ("swiglu_clamp_shexp",
             hf.FLASH_OVERRIDES["share_expert_swiglu_limit_list"]),
        ):
            metadata.append(_kv("bailingmoe3." + key, GGUF_ARRAY,
                                _float_array(list(values) + [0.0])))
    metadata += [
        _kv("tokenizer.ggml.model", GGUF_STRING, _string("gpt2")),
        _kv("tokenizer.ggml.pre", GGUF_STRING, _string("bailingmoe2")),
        _kv("tokenizer.ggml.tokens", GGUF_ARRAY, _string_array(tokens)),
        _kv("tokenizer.ggml.token_type", GGUF_ARRAY, _uint_array(types)),
        _kv("tokenizer.ggml.merges", GGUF_ARRAY, _string_array(merges)),
        _kv("tokenizer.ggml.eos_token_id", GGUF_UINT32,
            _uint(hf.CONFIG["eos_token_id"])),
        # Shaped like the real template where it matters: a role tag per turn,
        # every turn closed by the control token, and nothing between turns.
        _kv("tokenizer.chat_template", GGUF_STRING, _string(
            "{%- for message in messages %}"
            "<role>{{ message.role | upper }}</role>{{ message.content }}"
            + ROLE_END +
            "{%- endfor %}"
            "{%- if add_generation_prompt %}<role>ASSISTANT</role>{%- endif %}"
        )),
    ]

    body = b"".join(metadata)
    descriptors = bytearray()
    payload = bytearray()
    for name, array in tensors.items():
        values = np.ascontiguousarray(array, dtype=np.float32)
        # GGUF dimensions run fastest-first, the reverse of the numpy shape.
        dimensions = list(values.shape)[::-1]
        descriptors += _string(name)
        descriptors += struct.pack("<I", len(dimensions))
        descriptors += b"".join(struct.pack("<Q", extent) for extent in dimensions)
        descriptors += struct.pack("<IQ", GGML_F32, len(payload))
        payload += values.tobytes()
        pad = (ALIGNMENT - len(payload) % ALIGNMENT) % ALIGNMENT
        payload += b"\0" * pad

    header = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(metadata))
    prefix_length = len(header) + len(body) + len(descriptors)
    pad = (ALIGNMENT - prefix_length % ALIGNMENT) % ALIGNMENT
    path = directory / "bailing.gguf"
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(body)
        handle.write(descriptors)
        handle.write(b"\0" * pad)
        handle.write(payload)
    return path
