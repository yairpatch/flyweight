"""Builds a tiny BailingMoE3 checkpoint in HF layout.

The real reference checkpoint is 15.8 GB across 32 shards, which is no use in a
test. This synthesizes the same *structure* at a size that runs in under a
second: the 3:1 KDA/MLA layer cadence, a dense leading block, stacked routed
experts, and -- importantly -- experts deliberately split across two shards, so
the multi-part gather is exercised rather than assumed.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

# Small but structurally faithful. hidden_size must stay >= 256 so the weight
# tensors are actually quantized rather than falling back to f32.
HIDDEN = 256
LAYERS = 4  # layer_group_size 4 -> layers 0,1,2 are KDA, layer 3 is MLA
HEADS = 2
HEAD_DIM = 128
EXPERTS = 4
VOCAB = 512
MOE_FFN = 256
DENSE_FFN = 512

CONFIG = {
    "architectures": ["BailingMoeV3ForCausalLM"],
    "model_type": "bailing_hybrid",
    "hidden_size": HIDDEN,
    "num_hidden_layers": LAYERS,
    "num_attention_heads": HEADS,
    "num_key_value_heads": HEADS,
    "head_dim": HEAD_DIM,
    "max_position_embeddings": 4096,
    "vocab_size": VOCAB,
    "rms_norm_eps": 1e-6,
    "rope_theta": 600000.0,
    "intermediate_size": DENSE_FFN,
    "moe_intermediate_size": MOE_FFN,
    "moe_shared_expert_intermediate_size": MOE_FFN,
    "num_experts": EXPERTS,
    "num_experts_per_tok": 2,
    "num_shared_experts": 1,
    "first_k_dense_replace": 1,
    "layer_group_size": 4,
    "n_group": 2,
    "topk_group": 1,
    "score_function": "sigmoid",
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "q_lora_rank": 128,
    "kv_lora_rank": 128,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "qk_head_dim": 192,
    "v_head_dim": 128,
    "rope_interleave": True,
    "short_conv_kernel_size": 4,
    "expert_swiglu_limit_list": None,
    "eos_token_id": 7,
    "pad_token_id": 6,
}


def _bf16_bytes(count: int, seed: int) -> bytes:
    """Deterministic pseudo-random bf16 payload.

    Values are kept in a modest range so that block scales land in the f16
    normal range for some tensors and the subnormal range for others -- the
    latter is what caught the qwen_half_bits flush-to-zero bug.
    """
    out = bytearray()
    state = seed | 1
    for _ in range(count):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        # bf16 with a small exponent: sign + exponent around 2^-8..2^-1.
        sign = (state >> 31) & 1
        exponent = 118 + ((state >> 8) % 8)
        mantissa = (state >> 3) & 0x7F
        bits = (sign << 15) | (exponent << 7) | mantissa
        out += struct.pack("<H", bits)
    return bytes(out)


def _write_safetensors(path: Path, tensors: dict[str, tuple[list[int], bytes]]) -> None:
    header: dict[str, object] = {}
    offset = 0
    for name, (shape, payload) in tensors.items():
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    blob = json.dumps(header).encode()
    # The format requires the header be padded to 8-byte alignment.
    pad = (8 - len(blob) % 8) % 8
    blob += b" " * pad
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for _, payload in tensors.values():
            handle.write(payload)


def _tensor(shape: list[int], seed: int) -> tuple[list[int], bytes]:
    count = 1
    for extent in shape:
        count *= extent
    return shape, _bf16_bytes(count, seed)


def _tokenizer() -> dict:
    """A minimal byte-level BPE tokenizer.

    The split pattern is the llama3/BailingMoE3 one verbatim, because the
    loader selects a pre-tokenizer by matching on this regex -- a label would
    not exercise that.
    """
    # Byte-level alphabet, the same 256 single-character tokens a real
    # byte-level BPE starts from.
    direct = (
        list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    )
    mapping = {}
    extra = 0
    for byte in range(256):
        if byte in direct:
            mapping[byte] = byte
        else:
            mapping[byte] = 256 + extra
            extra += 1
    vocab = {chr(mapping[byte]): byte for byte in range(256)}
    # A couple of merges so the merge table is not trivially empty.
    merges = [["a", "b"], ["ab", "c"]]
    vocab["ab"] = 256
    vocab["abc"] = 257
    added = [
        {"id": 258, "content": "<|startoftext|>", "special": True},
        {"id": 259, "content": "<|endoftext|>", "special": True},
        {"id": 260, "content": "<|not_special|>", "special": False},
    ]
    return {
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {
                        "Regex": "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+"
                        "\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|"
                        "\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"
                    },
                    "behavior": "Isolated",
                    "invert": False,
                },
                {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False},
            ],
        },
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
    }


# Ling 3.0 Flash differs from the larger checkpoints in two ways that reach the
# kernels: it sets `q_lora_rank: null`, so the MLA query is one un-factored
# projection instead of q_a -> RMS norm -> q_b, and it turns the SwiGLU clamps
# on for its last few blocks -- separately for the routed and the shared half.
# Both are silent when dropped: the first fails to find a tensor, the second
# only changes the tails of the last layers' activations.
# The limits are far below the real checkpoint's 4.0/5.0/7.0 on purpose: this
# fixture's weights are deliberately small, so a realistic limit would never
# bind and "the clamp is applied" would be untestable.
FLASH_OVERRIDES = {
    "q_lora_rank": None,
    "expert_swiglu_limit_list": [0.05, 0.0, 0.0, 0.05],
    "share_expert_swiglu_limit_list": [0.0, 0.0, 0.07, 0.09],
}


def config(flash: bool = False) -> dict:
    merged = dict(CONFIG)
    if flash:
        merged.update(FLASH_OVERRIDES)
    return merged


def build(directory: Path, flash: bool = False) -> Path:
    """Writes the checkpoint and returns its directory."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config(flash)))
    (directory / "chat_template.jinja").write_text("{{ messages }}")
    (directory / "tokenizer.json").write_text(json.dumps(_tokenizer()))

    seed = 1
    def nxt() -> int:
        nonlocal seed
        seed += 7919
        return seed

    # Shard 0 holds everything except half the experts; shard 1 holds the rest.
    # HF stores nn.Linear as [out, in], the reverse of the GGUF convention.
    first: dict[str, tuple[list[int], bytes]] = {
        "model.word_embeddings.weight": _tensor([VOCAB, HIDDEN], nxt()),
        "model.norm.weight": _tensor([HIDDEN], nxt()),
        "lm_head.weight": _tensor([VOCAB, HIDDEN], nxt()),
    }
    second: dict[str, tuple[list[int], bytes]] = {}

    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}."
        first[prefix + "input_layernorm.weight"] = _tensor([HIDDEN], nxt())
        first[prefix + "post_attention_layernorm.weight"] = _tensor([HIDDEN], nxt())

        full_attention = (layer + 1) % CONFIG["layer_group_size"] == 0
        if full_attention:
            proj = HEADS * CONFIG["qk_head_dim"]
            if flash:
                # `q_proj` on a full-attention layer is the whole query; on a
                # linear layer below it is the KDA query. Same name, different
                # weight -- the loader has to disambiguate by layer kind.
                first[prefix + "attention.q_proj.weight"] = _tensor(
                    [proj, HIDDEN], nxt()
                )
            else:
                first[prefix + "attention.q_a_proj.weight"] = _tensor(
                    [128, HIDDEN], nxt()
                )
                first[prefix + "attention.q_a_layernorm.weight"] = _tensor([128], nxt())
                first[prefix + "attention.q_b_proj.weight"] = _tensor([proj, 128], nxt())
            first[prefix + "attention.kv_a_proj_with_mqa.weight"] = _tensor(
                [128 + 64, HIDDEN], nxt()
            )
            first[prefix + "attention.kv_a_layernorm.weight"] = _tensor([128], nxt())
            first[prefix + "attention.kv_b_proj.weight"] = _tensor(
                [HEADS * (128 + 128), 128], nxt()
            )
            first[prefix + "attention.dense.weight"] = _tensor(
                [HIDDEN, HEADS * 128], nxt()
            )
            # head_wise gate: one channel per head.
            first[prefix + "attention.g_proj.weight"] = _tensor([HEADS, HIDDEN], nxt())
        else:
            inner = HEADS * HEAD_DIM
            for name in ("q_proj", "k_proj", "v_proj", "f_proj", "o_proj"):
                shape = [HIDDEN, inner] if name == "o_proj" else [inner, HIDDEN]
                first[prefix + f"attention.{name}.weight"] = _tensor(shape, nxt())
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                first[prefix + f"attention.{name}.weight"] = _tensor([inner, 1, 4], nxt())
            first[prefix + "attention.b_proj.weight"] = _tensor([HEADS, HIDDEN], nxt())
            # KDA gate is per channel, not per head -- the two g_proj meanings
            # this fixture exists partly to keep apart.
            first[prefix + "attention.g_proj.weight"] = _tensor([inner, HIDDEN], nxt())
            first[prefix + "attention.A_log"] = _tensor([HEADS], nxt())
            first[prefix + "attention.dt_bias"] = _tensor([inner], nxt())
            first[prefix + "attention.o_norm.weight"] = _tensor([HEAD_DIM], nxt())

        if layer < CONFIG["first_k_dense_replace"]:
            first[prefix + "mlp.gate_proj.weight"] = _tensor([DENSE_FFN, HIDDEN], nxt())
            first[prefix + "mlp.up_proj.weight"] = _tensor([DENSE_FFN, HIDDEN], nxt())
            first[prefix + "mlp.down_proj.weight"] = _tensor([HIDDEN, DENSE_FFN], nxt())
            continue

        first[prefix + "mlp.gate.weight"] = _tensor([EXPERTS, HIDDEN], nxt())
        first[prefix + "mlp.gate.expert_bias"] = _tensor([EXPERTS], nxt())
        first[prefix + "mlp.shared_experts.gate_proj.weight"] = _tensor(
            [MOE_FFN, HIDDEN], nxt()
        )
        first[prefix + "mlp.shared_experts.up_proj.weight"] = _tensor(
            [MOE_FFN, HIDDEN], nxt()
        )
        first[prefix + "mlp.shared_experts.down_proj.weight"] = _tensor(
            [HIDDEN, MOE_FFN], nxt()
        )
        for expert in range(EXPERTS):
            # Split experts across shards so reassembly is genuinely tested.
            target = first if expert < EXPERTS // 2 else second
            base = prefix + f"mlp.experts.{expert}."
            target[base + "gate_proj.weight"] = _tensor([MOE_FFN, HIDDEN], nxt())
            target[base + "up_proj.weight"] = _tensor([MOE_FFN, HIDDEN], nxt())
            target[base + "down_proj.weight"] = _tensor([HIDDEN, MOE_FFN], nxt())

    _write_safetensors(directory / "model-00000-of-00002.safetensors", first)
    _write_safetensors(directory / "model-00001-of-00002.safetensors", second)

    weight_map = {name: "model-00000-of-00002.safetensors" for name in first}
    weight_map.update(
        {name: "model-00001-of-00002.safetensors" for name in second}
    )
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    return directory
