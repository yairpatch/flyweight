"""A tiny Qwen3.5 HuggingFace checkpoint, for the safetensors loader.

Shaped like the real Qwen3.8-27B release rather than like a convenient test:
the config nests the decoder under ``text_config`` beside a ``vision_config``,
the tensors sit under ``model.language_model.``, the MTP block is its own
``mtp.`` prefix, and a handful of ``model.visual.`` tensors are present so the
loader is actually made to drop them. Those four properties are the whole point
-- a flat single-architecture fixture would pass without exercising any of them.

Layer cadence follows ``full_attention_interval``: every fourth layer is full
attention, the rest are gated-delta, which is what decides whether a layer
carries ``self_attn.*`` or ``linear_attn.*``.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.hf_safetensors_fixture import (
    _tensor,
    _tokenizer,
    _write_safetensors,
)

HIDDEN = 256  # a whole number of K-quant blocks per row, as the release is
LAYERS = 8
FULL_INTERVAL = 4
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 64
VOCAB = 512
FFN = 256

# Gated-delta widths. value_heads is what ssm_a/ssm_dt.bias are sized by, and
# the qkv projection is value_dim + 2 * key_heads * key_head_dim.
VALUE_HEADS = 4
KEY_HEADS = 2
HEAD_STATE = 16
VALUE_DIM = VALUE_HEADS * HEAD_STATE
QKV_DIM = VALUE_DIM + 2 * KEY_HEADS * HEAD_STATE
CONV_KERNEL = 4

VISION_HIDDEN = 32

CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "attn_output_gate": True,
        "max_position_embeddings": 4096,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6,
        "intermediate_size": FFN,
        "full_attention_interval": FULL_INTERVAL,
        "linear_conv_kernel_dim": CONV_KERNEL,
        "linear_num_value_heads": VALUE_HEADS,
        "linear_num_key_heads": KEY_HEADS,
        "linear_key_head_dim": HEAD_STATE,
        "linear_value_head_dim": HEAD_STATE,
        "mtp_num_hidden_layers": 1,
        # Nested exactly as the real checkpoint does; a flat rope_theta would
        # silently fall back to the default and hide a parsing bug.
        "rope_parameters": {
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        },
        # End-of-document. The turn actually ends on the token
        # generation_config.json names, which must win.
        "eos_token_id": 5,
        "bos_token_id": 5,
    },
    "vision_config": {
        "model_type": "qwen3_5_vision",
        "depth": 2,
        "hidden_size": VISION_HIDDEN,
        "patch_size": 16,
    },
}

GENERATION_CONFIG = {
    "bos_token_id": 5,
    # First entry wins, and it differs from text_config's eos on purpose.
    "eos_token_id": [7, 5],
    "pad_token_id": 5,
}


def is_full_attention(layer: int) -> bool:
    return (layer + 1) % FULL_INTERVAL == 0


def _language_tensors(seed) -> dict[str, tuple[list[int], bytes]]:
    tensors: dict[str, tuple[list[int], bytes]] = {}
    tensors["model.language_model.embed_tokens.weight"] = _tensor([VOCAB, HIDDEN], seed())
    tensors["model.language_model.norm.weight"] = _tensor([HIDDEN], seed())
    tensors["lm_head.weight"] = _tensor([VOCAB, HIDDEN], seed())

    for layer in range(LAYERS):
        prefix = f"model.language_model.layers.{layer}."
        tensors[prefix + "input_layernorm.weight"] = _tensor([HIDDEN], seed())
        tensors[prefix + "post_attention_layernorm.weight"] = _tensor([HIDDEN], seed())
        _block(tensors, prefix, seed, full=is_full_attention(layer))

    # The MTP block: its own projections plus a complete transformer layer,
    # which is full attention on this family.
    tensors["mtp.fc.weight"] = _tensor([HIDDEN, 2 * HIDDEN], seed())
    tensors["mtp.pre_fc_norm_embedding.weight"] = _tensor([HIDDEN], seed())
    tensors["mtp.pre_fc_norm_hidden.weight"] = _tensor([HIDDEN], seed())
    tensors["mtp.norm.weight"] = _tensor([HIDDEN], seed())
    mtp = "mtp.layers.0."
    tensors[mtp + "input_layernorm.weight"] = _tensor([HIDDEN], seed())
    tensors[mtp + "post_attention_layernorm.weight"] = _tensor([HIDDEN], seed())
    _block(tensors, mtp, seed, full=True)
    return tensors


def _block(tensors, prefix: str, seed, *, full: bool) -> None:
    if full:
        # attn_output_gate packs the per-head output gate into q_proj, so it is
        # twice as wide as the head count would suggest: [q | gate] per head.
        tensors[prefix + "self_attn.q_proj.weight"] = _tensor(
            [2 * HEADS * HEAD_DIM, HIDDEN], seed())
        tensors[prefix + "self_attn.k_proj.weight"] = _tensor(
            [KV_HEADS * HEAD_DIM, HIDDEN], seed())
        tensors[prefix + "self_attn.v_proj.weight"] = _tensor(
            [KV_HEADS * HEAD_DIM, HIDDEN], seed())
        tensors[prefix + "self_attn.o_proj.weight"] = _tensor(
            [HIDDEN, HEADS * HEAD_DIM], seed())
        tensors[prefix + "self_attn.q_norm.weight"] = _tensor([HEAD_DIM], seed())
        tensors[prefix + "self_attn.k_norm.weight"] = _tensor([HEAD_DIM], seed())
    else:
        tensors[prefix + "linear_attn.in_proj_qkv.weight"] = _tensor(
            [QKV_DIM, HIDDEN], seed())
        tensors[prefix + "linear_attn.in_proj_z.weight"] = _tensor(
            [VALUE_DIM, HIDDEN], seed())
        tensors[prefix + "linear_attn.in_proj_a.weight"] = _tensor(
            [VALUE_HEADS, HIDDEN], seed())
        tensors[prefix + "linear_attn.in_proj_b.weight"] = _tensor(
            [VALUE_HEADS, HIDDEN], seed())
        tensors[prefix + "linear_attn.out_proj.weight"] = _tensor(
            [HIDDEN, VALUE_DIM], seed())
        # [channels, 1, kernel] on disk, which reverses to the GGUF's
        # [kernel, 1, channels].
        tensors[prefix + "linear_attn.conv1d.weight"] = _tensor(
            [QKV_DIM, 1, CONV_KERNEL], seed())
        tensors[prefix + "linear_attn.norm.weight"] = _tensor([HEAD_STATE], seed())
        tensors[prefix + "linear_attn.dt_bias"] = _tensor([VALUE_HEADS], seed())
        tensors[prefix + "linear_attn.A_log"] = _tensor([VALUE_HEADS], seed())

    tensors[prefix + "mlp.gate_proj.weight"] = _tensor([FFN, HIDDEN], seed())
    tensors[prefix + "mlp.up_proj.weight"] = _tensor([FFN, HIDDEN], seed())
    tensors[prefix + "mlp.down_proj.weight"] = _tensor([HIDDEN, FFN], seed())


def _vision_tensors(seed) -> dict[str, tuple[list[int], bytes]]:
    """A stub tower. Present only so the loader has something to drop."""
    tensors: dict[str, tuple[list[int], bytes]] = {}
    tensors["model.visual.patch_embed.proj.weight"] = _tensor(
        [VISION_HIDDEN, VISION_HIDDEN], seed())
    for block in range(2):
        prefix = f"model.visual.blocks.{block}."
        tensors[prefix + "attn.qkv.weight"] = _tensor(
            [3 * VISION_HIDDEN, VISION_HIDDEN], seed())
        tensors[prefix + "attn.proj.weight"] = _tensor(
            [VISION_HIDDEN, VISION_HIDDEN], seed())
    tensors["model.visual.merger.linear_fc1.weight"] = _tensor(
        [VISION_HIDDEN, VISION_HIDDEN], seed())
    return tensors


def language_tensor_names() -> set[str]:
    """The GGUF-side names the loader should produce, for the test to assert on."""
    names = {"token_embd.weight", "output_norm.weight", "output.weight"}
    for layer in list(range(LAYERS)) + [LAYERS]:
        block = f"blk.{layer}."
        names |= {block + "attn_norm.weight", block + "post_attention_norm.weight",
                  block + "ffn_gate.weight", block + "ffn_up.weight",
                  block + "ffn_down.weight"}
        full = layer == LAYERS or is_full_attention(layer)
        if full:
            names |= {block + "attn_q.weight", block + "attn_k.weight",
                      block + "attn_v.weight", block + "attn_output.weight",
                      block + "attn_q_norm.weight", block + "attn_k_norm.weight"}
        else:
            names |= {block + "attn_qkv.weight", block + "attn_gate.weight",
                      block + "ssm_alpha.weight", block + "ssm_beta.weight",
                      block + "ssm_out.weight", block + "ssm_conv1d.weight",
                      block + "ssm_norm.weight", block + "ssm_dt.bias",
                      block + "ssm_a"}
    names |= {f"blk.{LAYERS}.nextn.eh_proj.weight",
              f"blk.{LAYERS}.nextn.enorm.weight",
              f"blk.{LAYERS}.nextn.hnorm.weight",
              f"blk.{LAYERS}.nextn.shared_head_norm.weight"}
    return names


def build(directory: Path) -> Path:
    """Writes the checkpoint and returns its directory."""
    directory.mkdir(parents=True, exist_ok=True)
    counter = 0

    def seed() -> int:
        nonlocal counter
        counter += 1
        return counter

    tensors = _language_tensors(seed)
    tensors.update(_vision_tensors(seed))
    # Two shards, so the loader's multi-shard path is what gets exercised.
    names = list(tensors)
    half = len(names) // 2
    _write_safetensors(
        directory / "model-00001-of-00002.safetensors",
        {name: tensors[name] for name in names[:half]})
    _write_safetensors(
        directory / "model-00002-of-00002.safetensors",
        {name: tensors[name] for name in names[half:]})

    (directory / "config.json").write_text(json.dumps(CONFIG))
    (directory / "generation_config.json").write_text(json.dumps(GENERATION_CONFIG))
    (directory / "tokenizer.json").write_text(json.dumps(_tokenizer()))
    return directory
