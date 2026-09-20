"""A tiny Qwen-Image-2.1 diffusers snapshot, for the diffusion loader and tower.

Laid out like the real release: ``text_encoder/`` (a Qwen3-VL checkpoint whose
decoder lives under ``text_config`` and whose vision tower the loader drops,
with the tokenizer in the sibling ``processor/`` directory),
``transformer/`` (``QwenImage21Transformer2DModel``) and ``vae/``
(``AutoencoderKLQwenImage21``, carrying encoder and ``time_conv`` tensors the
decoder never reads so the loader is made to drop them).

The shapes are small but keep every structural property the runtime relies
on: the DiT's head width is the sum of its rope axes, its feed-forward is
``mlp_ratio`` times the width, the autoencoder has five levels off one
``decoder_base_dim``, its duplicating shortcuts divide evenly, and the
channel changes inside an up block carry a ``conv_shortcut``.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.hf_safetensors_fixture import _tensor, _tokenizer, _write_safetensors

# Text encoder (Qwen3-VL's decoder half).
ENC_HIDDEN = 64
ENC_LAYERS = 3
ENC_HEADS = 2
ENC_KV_HEADS = 1
ENC_HEAD_DIM = 32
ENC_FFN = 96
VOCAB = 512

# DiT.
DIT_HEADS = 2
DIT_HEAD_DIM = 48  # 16 + 16 + 16
DIT_DIM = DIT_HEADS * DIT_HEAD_DIM  # 96
AXES_DIMS = [16, 16, 16]
DIT_LAYERS = 2
MLP_RATIO = 3
DIT_FFN = DIT_DIM * MLP_RATIO
TIME_EMBED = 256
CHANNELS = 16  # z_dim

# Autoencoder decoder: dim_mult [1, 2, 4, 4, 4] off base 4 gives level channels
# [4, 8, 16, 16, 16], so the decoder walks 16 -> 16 -> 16 -> 8 -> 4.
VAE_BASE = 4
VAE_DIM_MULT = [1, 2, 4, 4, 4]
VAE_LEVELS = [VAE_BASE * m for m in VAE_DIM_MULT]
VAE_TEMPORAL_DOWNSAMPLE = [False, True, True, True]
VAE_RES_BLOCKS = 1
VAE_OUT_CHANNELS = 4


def encoder_config() -> dict:
    return {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "image_token_id": 9,
        "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": ENC_HIDDEN,
            "num_hidden_layers": ENC_LAYERS,
            "num_attention_heads": ENC_HEADS,
            "num_key_value_heads": ENC_KV_HEADS,
            "head_dim": ENC_HEAD_DIM,
            "intermediate_size": ENC_FFN,
            "max_position_embeddings": 4096,
            "vocab_size": VOCAB,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5000000,
            "rope_scaling": {"mrope_interleaved": True, "mrope_section": [8, 4, 4],
                             "rope_type": "default"},
            "eos_token_id": 7,
            "bos_token_id": 5,
        },
        "vision_config": {"depth": 2, "hidden_size": 8, "out_hidden_size": ENC_HIDDEN,
                          "patch_size": 16, "num_heads": 1, "intermediate_size": 16,
                          "deepstack_visual_indexes": [0]},
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
    }


def transformer_config() -> dict:
    return {
        "_class_name": "QwenImage21Transformer2DModel",
        "_diffusers_version": "0.37.0.dev0",
        "attention_head_dim": DIT_HEAD_DIM,
        "axes_dims_rope": AXES_DIMS,
        "causal_condition": True,
        "context_in_dim": ENC_HIDDEN,
        "eps": 1e-06,
        "in_channels": CHANNELS,
        "mlp_ratio": MLP_RATIO,
        "num_attention_heads": DIT_HEADS,
        "num_layers": DIT_LAYERS,
        "out_channels": CHANNELS,
        "patch_size": 1,
    }


def vae_config() -> dict:
    return {
        "_class_name": "AutoencoderKLQwenImage21",
        "_diffusers_version": "0.37.0.dev0",
        "attn_scales": [],
        "base_dim": VAE_BASE,
        "decoder_base_dim": VAE_BASE,
        "dim_mult": VAE_DIM_MULT,
        "dropout": 0.0,
        "in_channels": VAE_OUT_CHANNELS,
        "is_residual": True,
        "latents_mean": [0.25 * (i % 7) - 0.5 for i in range(CHANNELS)],
        "latents_std": [1.0 + 0.1 * (i % 5) for i in range(CHANNELS)],
        "num_res_blocks": VAE_RES_BLOCKS,
        "out_channels": VAE_OUT_CHANNELS,
        "patch_size": None,
        "scale_factor_spatial": 16,
        "scale_factor_temporal": 8,
        "temperal_downsample": VAE_TEMPORAL_DOWNSAMPLE,
        "z_dim": CHANNELS,
    }


def _encoder_tensors() -> dict[str, tuple[list[int], bytes]]:
    seed = iter(range(1000, 100000))
    t = lambda shape: _tensor(shape, next(seed))  # noqa: E731
    tensors = {
        "model.language_model.embed_tokens.weight": t([VOCAB, ENC_HIDDEN]),
        "model.language_model.norm.weight": t([ENC_HIDDEN]),
        "lm_head.weight": t([VOCAB, ENC_HIDDEN]),
        # The vision tower: recognised by the loader and dropped.
        "model.visual.patch_embed.proj.weight": t([8, 3, 2, 16, 16]),
        "model.visual.merger.linear_fc1.weight": t([ENC_HIDDEN, 32]),
    }
    q_width = ENC_HEADS * ENC_HEAD_DIM
    kv_width = ENC_KV_HEADS * ENC_HEAD_DIM
    for layer in range(ENC_LAYERS):
        prefix = f"model.language_model.layers.{layer}."
        tensors[prefix + "input_layernorm.weight"] = t([ENC_HIDDEN])
        tensors[prefix + "post_attention_layernorm.weight"] = t([ENC_HIDDEN])
        tensors[prefix + "self_attn.q_proj.weight"] = t([q_width, ENC_HIDDEN])
        tensors[prefix + "self_attn.k_proj.weight"] = t([kv_width, ENC_HIDDEN])
        tensors[prefix + "self_attn.v_proj.weight"] = t([kv_width, ENC_HIDDEN])
        tensors[prefix + "self_attn.o_proj.weight"] = t([ENC_HIDDEN, q_width])
        tensors[prefix + "self_attn.q_norm.weight"] = t([ENC_HEAD_DIM])
        tensors[prefix + "self_attn.k_norm.weight"] = t([ENC_HEAD_DIM])
        tensors[prefix + "mlp.gate_proj.weight"] = t([ENC_FFN, ENC_HIDDEN])
        tensors[prefix + "mlp.up_proj.weight"] = t([ENC_FFN, ENC_HIDDEN])
        tensors[prefix + "mlp.down_proj.weight"] = t([ENC_HIDDEN, ENC_FFN])
    return tensors


def _transformer_tensors() -> dict[str, tuple[list[int], bytes]]:
    seed = iter(range(200000, 400000))
    t = lambda shape: _tensor(shape, next(seed))  # noqa: E731
    tensors = {
        "img_in.weight": t([DIT_DIM, CHANNELS]),
        "txt_in.text_norm.weight": t([ENC_HIDDEN]),
        "txt_in.in_layer.weight": t([DIT_DIM, ENC_HIDDEN]),
        "txt_in.out_layer.weight": t([DIT_DIM, DIT_DIM]),
        "time_text_embed.timestep_embedder.linear_1.weight": t([DIT_DIM, TIME_EMBED]),
        "time_text_embed.timestep_embedder.linear_2.weight": t([DIT_DIM, DIT_DIM]),
        "modulation.1.weight": t([4 * DIT_DIM, DIT_DIM]),
        "norm_out.linear.weight": t([DIT_DIM, DIT_DIM]),
        "proj_out.weight": t([CHANNELS, DIT_DIM]),
    }
    for index in range(DIT_LAYERS):
        prefix = f"transformer_blocks.{index}."
        for name in ("to_q", "to_k", "to_v", "to_out.0"):
            tensors[prefix + f"attn.{name}.weight"] = t([DIT_DIM, DIT_DIM])
        tensors[prefix + "attn.norm_q.weight"] = t([DIT_HEAD_DIM])
        tensors[prefix + "attn.norm_k.weight"] = t([DIT_HEAD_DIM])
        tensors[prefix + "img_mlp.gate_layer.weight"] = t([DIT_FFN, DIT_DIM])
        tensors[prefix + "img_mlp.proj.weight"] = t([DIT_FFN, DIT_DIM])
        tensors[prefix + "img_mlp.out.weight"] = t([DIT_DIM, DIT_FFN])
    return tensors


def _resnet(tensors, prefix: str, t, in_channels: int, out_channels: int) -> None:
    tensors[prefix + "norm1.gamma"] = t([in_channels, 1, 1, 1])
    tensors[prefix + "conv1.weight"] = t([out_channels, in_channels, 3, 3])
    tensors[prefix + "conv1.bias"] = t([out_channels])
    tensors[prefix + "norm2.gamma"] = t([out_channels, 1, 1, 1])
    tensors[prefix + "conv2.weight"] = t([out_channels, out_channels, 3, 3])
    tensors[prefix + "conv2.bias"] = t([out_channels])
    if in_channels != out_channels:
        tensors[prefix + "conv_shortcut.weight"] = t([out_channels, in_channels, 1, 1])
        tensors[prefix + "conv_shortcut.bias"] = t([out_channels])


def _vae_tensors() -> dict[str, tuple[list[int], bytes]]:
    seed = iter(range(500000, 600000))
    t = lambda shape: _tensor(shape, next(seed))  # noqa: E731
    top = VAE_LEVELS[-1]
    levels = len(VAE_LEVELS)
    tensors = {
        "post_quant_conv.weight": t([CHANNELS, CHANNELS, 1, 1]),
        "post_quant_conv.bias": t([CHANNELS]),
        "decoder.conv_in.weight": t([top, CHANNELS, 3, 3]),
        "decoder.conv_in.bias": t([top]),
        "decoder.mid_block.attentions.0.norm.gamma": t([top, 1, 1]),
        "decoder.mid_block.attentions.0.to_qkv.weight": t([3 * top, top, 1, 1]),
        "decoder.mid_block.attentions.0.to_qkv.bias": t([3 * top]),
        "decoder.mid_block.attentions.0.proj.weight": t([top, top, 1, 1]),
        "decoder.mid_block.attentions.0.proj.bias": t([top]),
    }
    _resnet(tensors, "decoder.mid_block.resnets.0.", t, top, top)
    _resnet(tensors, "decoder.mid_block.resnets.1.", t, top, top)
    for level in range(levels):
        prefix = f"decoder.up_blocks.{level}."
        in_channels = top if level == 0 else VAE_LEVELS[levels - level]
        out = VAE_LEVELS[levels - 1 - level]
        for index in range(VAE_RES_BLOCKS + 1):
            _resnet(tensors, prefix + f"resnets.{index}.", t,
                    in_channels if index == 0 else out, out)
        if level < levels - 1:
            tensors[prefix + "upsampler.resample.1.weight"] = t([out, out, 3, 3])
            tensors[prefix + "upsampler.resample.1.bias"] = t([out])
            # The temporal half of the upsample, which one frame never reaches.
            tensors[prefix + "upsampler.time_conv.weight"] = t([2 * out, out, 1, 1])
            tensors[prefix + "upsampler.time_conv.bias"] = t([2 * out])
    tensors["decoder.norm_out.gamma"] = t([VAE_LEVELS[0], 1, 1, 1])
    tensors["decoder.conv_out.weight"] = t([VAE_OUT_CHANNELS, VAE_LEVELS[0], 3, 3])
    tensors["decoder.conv_out.bias"] = t([VAE_OUT_CHANNELS])
    # Encoder-side tensors the decoder never reads; the loader must drop them.
    tensors["encoder.conv_in.weight"] = t([VAE_LEVELS[0], VAE_OUT_CHANNELS, 3, 3])
    tensors["encoder.conv_in.bias"] = t([VAE_LEVELS[0]])
    tensors["quant_conv.weight"] = t([2 * CHANNELS, 2 * CHANNELS, 1, 1])
    tensors["quant_conv.bias"] = t([2 * CHANNELS])
    return tensors


def build(directory: Path) -> Path:
    """Write the snapshot under `directory` and return it."""
    directory = Path(directory)
    for part, config, tensors in (
        ("text_encoder", encoder_config(), _encoder_tensors()),
        ("transformer", transformer_config(), _transformer_tensors()),
        ("vae", vae_config(), _vae_tensors()),
    ):
        folder = directory / part
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text(json.dumps(config, indent=1))
        name = "model.safetensors" if part == "text_encoder" else "diffusion_pytorch_model.safetensors"
        _write_safetensors(folder / name, tensors)
    # The vision-language pipeline keeps the tokenizer in `processor/`, not the
    # `tokenizer/` a text-only one uses.
    processor = directory / "processor"
    processor.mkdir(parents=True, exist_ok=True)
    (processor / "tokenizer.json").write_text(json.dumps(_tokenizer()))
    (processor / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "{% for message in messages %}<|im_start|>{{ message.role }}\n"
                         "{{ message.content }}<|im_end|>\n{% endfor %}"
                         "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
    }))
    (directory / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline"}))
    return directory
