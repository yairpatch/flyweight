"""A tiny Z-Image-Turbo diffusers snapshot, for the diffusion loader and tower.

Laid out like the real release: ``text_encoder/`` (a plain Qwen3 decoder
whose tokenizer lives in the sibling ``tokenizer/`` directory),
``transformer/`` (``ZImageTransformer2DModel``, identified by ``_class_name``
rather than ``model_type``) and ``vae/`` (``AutoencoderKL`` with a few
encoder tensors present so the loader is made to drop them). The shapes are
small but keep every structural property the runtime relies on: the DiT's
head width is the sum of its rope axes, its feed-forward is dim/3*8 wide,
the VAE's channel counts are whole numbers of norm groups, and a channel
change inside an up block carries a ``conv_shortcut``.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.hf_safetensors_fixture import _tensor, _tokenizer, _write_safetensors

# Text encoder (Qwen3).
ENC_HIDDEN = 64
ENC_LAYERS = 3
ENC_HEADS = 2
ENC_KV_HEADS = 1
ENC_HEAD_DIM = 32
ENC_FFN = 96
VOCAB = 512

# DiT.
DIT_DIM = 96
DIT_HEADS = 2
DIT_HEAD_DIM = DIT_DIM // DIT_HEADS  # 48 = 16 + 16 + 16
AXES_DIMS = [16, 16, 16]
AXES_LENS = [64, 32, 32]
DIT_LAYERS = 2
DIT_REFINERS = 1
DIT_FFN = DIT_DIM // 3 * 8  # 256
ADALN = 256
T_MID = 1024
CHANNELS = 16
PATCH = 2
PATCH_INPUTS = PATCH * PATCH * CHANNELS  # 64

# VAE decoder.
VAE_CHANNELS = [8, 8, 16, 16]  # block_out_channels, walked in reverse by the decoder
VAE_GROUPS = 4
VAE_LAYERS_PER_BLOCK = 1


def encoder_config() -> dict:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": ENC_HIDDEN,
        "num_hidden_layers": ENC_LAYERS,
        "num_attention_heads": ENC_HEADS,
        "num_key_value_heads": ENC_KV_HEADS,
        "head_dim": ENC_HEAD_DIM,
        "intermediate_size": ENC_FFN,
        "max_position_embeddings": 4096,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
        "tie_word_embeddings": True,
        "eos_token_id": 7,
        "bos_token_id": 5,
        "torch_dtype": "bfloat16",
    }


def transformer_config() -> dict:
    return {
        "_class_name": "ZImageTransformer2DModel",
        "_diffusers_version": "0.36.0",
        "all_patch_size": [PATCH],
        "all_f_patch_size": [1],
        "axes_dims": AXES_DIMS,
        "axes_lens": AXES_LENS,
        "cap_feat_dim": ENC_HIDDEN,
        "dim": DIT_DIM,
        "in_channels": CHANNELS,
        "n_heads": DIT_HEADS,
        "n_kv_heads": DIT_HEADS,
        "n_layers": DIT_LAYERS,
        "n_refiner_layers": DIT_REFINERS,
        "norm_eps": 1e-5,
        "qk_norm": True,
        "rope_theta": 256.0,
        "t_scale": 1000.0,
    }


def vae_config() -> dict:
    return {
        "_class_name": "AutoencoderKL",
        "_diffusers_version": "0.36.0",
        "act_fn": "silu",
        "block_out_channels": VAE_CHANNELS,
        "down_block_types": ["DownEncoderBlock2D"] * 4,
        "up_block_types": ["UpDecoderBlock2D"] * 4,
        "force_upcast": True,
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": CHANNELS,
        "layers_per_block": VAE_LAYERS_PER_BLOCK,
        "mid_block_add_attention": True,
        "norm_num_groups": VAE_GROUPS,
        "sample_size": 64,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
        "use_post_quant_conv": False,
        "use_quant_conv": False,
    }


def _encoder_tensors() -> dict[str, tuple[list[int], bytes]]:
    seed = iter(range(1000, 100000))
    t = lambda shape: _tensor(shape, next(seed))  # noqa: E731
    tensors = {
        "model.embed_tokens.weight": t([VOCAB, ENC_HIDDEN]),
        "model.norm.weight": t([ENC_HIDDEN]),
    }
    q_width = ENC_HEADS * ENC_HEAD_DIM
    kv_width = ENC_KV_HEADS * ENC_HEAD_DIM
    for layer in range(ENC_LAYERS):
        prefix = f"model.layers.{layer}."
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


def _dit_block(tensors, prefix: str, t, *, modulated: bool) -> None:
    tensors[prefix + "attention.to_q.weight"] = t([DIT_DIM, DIT_DIM])
    tensors[prefix + "attention.to_k.weight"] = t([DIT_DIM, DIT_DIM])
    tensors[prefix + "attention.to_v.weight"] = t([DIT_DIM, DIT_DIM])
    tensors[prefix + "attention.to_out.0.weight"] = t([DIT_DIM, DIT_DIM])
    tensors[prefix + "attention.norm_q.weight"] = t([DIT_HEAD_DIM])
    tensors[prefix + "attention.norm_k.weight"] = t([DIT_HEAD_DIM])
    for name in ("attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2"):
        tensors[prefix + name + ".weight"] = t([DIT_DIM])
    tensors[prefix + "feed_forward.w1.weight"] = t([DIT_FFN, DIT_DIM])
    tensors[prefix + "feed_forward.w3.weight"] = t([DIT_FFN, DIT_DIM])
    tensors[prefix + "feed_forward.w2.weight"] = t([DIT_DIM, DIT_FFN])
    if modulated:
        tensors[prefix + "adaLN_modulation.0.weight"] = t([4 * DIT_DIM, ADALN])
        tensors[prefix + "adaLN_modulation.0.bias"] = t([4 * DIT_DIM])


def _transformer_tensors() -> dict[str, tuple[list[int], bytes]]:
    seed = iter(range(200000, 400000))
    t = lambda shape: _tensor(shape, next(seed))  # noqa: E731
    key = f"{PATCH}-1"
    tensors = {
        f"all_x_embedder.{key}.weight": t([DIT_DIM, PATCH_INPUTS]),
        f"all_x_embedder.{key}.bias": t([DIT_DIM]),
        f"all_final_layer.{key}.adaLN_modulation.1.weight": t([DIT_DIM, ADALN]),
        f"all_final_layer.{key}.adaLN_modulation.1.bias": t([DIT_DIM]),
        f"all_final_layer.{key}.linear.weight": t([PATCH_INPUTS, DIT_DIM]),
        f"all_final_layer.{key}.linear.bias": t([PATCH_INPUTS]),
        "cap_embedder.0.weight": t([ENC_HIDDEN]),
        "cap_embedder.1.weight": t([DIT_DIM, ENC_HIDDEN]),
        "cap_embedder.1.bias": t([DIT_DIM]),
        "t_embedder.mlp.0.weight": t([T_MID, ADALN]),
        "t_embedder.mlp.0.bias": t([T_MID]),
        "t_embedder.mlp.2.weight": t([ADALN, T_MID]),
        "t_embedder.mlp.2.bias": t([ADALN]),
        "x_pad_token": t([1, DIT_DIM]),
        "cap_pad_token": t([1, DIT_DIM]),
    }
    for index in range(DIT_REFINERS):
        _dit_block(tensors, f"noise_refiner.{index}.", t, modulated=True)
        _dit_block(tensors, f"context_refiner.{index}.", t, modulated=False)
    for index in range(DIT_LAYERS):
        _dit_block(tensors, f"layers.{index}.", t, modulated=True)
    return tensors


def _resnet(tensors, prefix: str, t, in_channels: int, out_channels: int) -> None:
    tensors[prefix + "norm1.weight"] = t([in_channels])
    tensors[prefix + "norm1.bias"] = t([in_channels])
    tensors[prefix + "conv1.weight"] = t([out_channels, in_channels, 3, 3])
    tensors[prefix + "conv1.bias"] = t([out_channels])
    tensors[prefix + "norm2.weight"] = t([out_channels])
    tensors[prefix + "norm2.bias"] = t([out_channels])
    tensors[prefix + "conv2.weight"] = t([out_channels, out_channels, 3, 3])
    tensors[prefix + "conv2.bias"] = t([out_channels])
    if in_channels != out_channels:
        tensors[prefix + "conv_shortcut.weight"] = t([out_channels, in_channels, 1, 1])
        tensors[prefix + "conv_shortcut.bias"] = t([out_channels])


def _vae_tensors() -> dict[str, tuple[list[int], bytes]]:
    seed = iter(range(500000, 600000))
    t = lambda shape: _tensor(shape, next(seed))  # noqa: E731
    top = VAE_CHANNELS[-1]
    tensors = {
        "decoder.conv_in.weight": t([top, CHANNELS, 3, 3]),
        "decoder.conv_in.bias": t([top]),
        "decoder.mid_block.attentions.0.group_norm.weight": t([top]),
        "decoder.mid_block.attentions.0.group_norm.bias": t([top]),
    }
    for name in ("to_q", "to_k", "to_v", "to_out.0"):
        tensors[f"decoder.mid_block.attentions.0.{name}.weight"] = t([top, top])
        tensors[f"decoder.mid_block.attentions.0.{name}.bias"] = t([top])
    _resnet(tensors, "decoder.mid_block.resnets.0.", t, top, top)
    _resnet(tensors, "decoder.mid_block.resnets.1.", t, top, top)
    previous = top
    for level in range(4):
        out = VAE_CHANNELS[3 - level]
        for index in range(VAE_LAYERS_PER_BLOCK + 1):
            _resnet(tensors, f"decoder.up_blocks.{level}.resnets.{index}.", t,
                    previous if index == 0 else out, out)
        if level < 3:
            tensors[f"decoder.up_blocks.{level}.upsamplers.0.conv.weight"] = t([out, out, 3, 3])
            tensors[f"decoder.up_blocks.{level}.upsamplers.0.conv.bias"] = t([out])
        previous = out
    tensors["decoder.conv_norm_out.weight"] = t([previous])
    tensors["decoder.conv_norm_out.bias"] = t([previous])
    tensors["decoder.conv_out.weight"] = t([3, previous, 3, 3])
    tensors["decoder.conv_out.bias"] = t([3])
    # Encoder-side tensors the decoder never reads; the loader must drop them.
    tensors["encoder.conv_in.weight"] = t([VAE_CHANNELS[0], 3, 3, 3])
    tensors["encoder.conv_in.bias"] = t([VAE_CHANNELS[0]])
    tensors["quant_conv.weight"] = t([2 * CHANNELS, 2 * CHANNELS, 1, 1])
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
    tokenizer = directory / "tokenizer"
    tokenizer.mkdir(parents=True, exist_ok=True)
    (tokenizer / "tokenizer.json").write_text(json.dumps(_tokenizer()))
    (tokenizer / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "{% for message in messages %}<|im_start|>{{ message.role }}\n"
                         "{{ message.content }}<|im_end|>\n{% endfor %}"
                         "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
    }))
    (directory / "model_index.json").write_text(json.dumps({"_class_name": "ZImagePipeline"}))
    return directory
