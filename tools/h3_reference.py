"""Reference computations for the MiniMax-H3 port, straight from the GGUF weights.

diffusers implements the original time-embedder DiT, not the pruned adaLN
table the released GGUFs carry, so the reference for those files is this:
a torch (CPU, f32) transcription of the MiniMax-H3 blocks fed with the same
Q4_K / Q6_K / Q8_0 weights, dequantized one layer at a time so a 32B encoder
and a 20B DiT stream through 60 GB of RAM. Same weights as the native tower,
so agreement is tight, not "within quantization".

Subcommands:
    encode   hidden state of the Qwen3-VL text encoder for a prompt
             -> npz with `input_ids`, `hidden`
"""
from __future__ import annotations

import argparse
import time

import math

import numpy as np
import torch


def read_gguf(path: str):
    from gguf import GGUFReader

    reader = GGUFReader(path)
    return {tensor.name: tensor for tensor in reader.tensors}


def dequantize(tensor) -> np.ndarray:
    """A GGUF tensor as f32 in its logical [rows][columns] shape."""
    from gguf.quants import dequantize as gguf_dequantize

    data = np.asarray(tensor.data)
    array = gguf_dequantize(data, tensor.tensor_type)
    # GGUF stores shapes innermost-first; numpy wants the reverse.
    shape = tuple(int(dim) for dim in reversed(tensor.shape))
    return np.ascontiguousarray(array.reshape(shape).astype(np.float32))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def encode(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ids = tokenizer(args.prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
    print("tokens", ids.tolist(), flush=True)
    tensors = read_gguf(args.gguf)
    layers = 1 + max(int(name.split(".")[2]) for name in tensors if name.startswith("model.layers."))
    hidden_size, heads, kv_heads, head_dim, eps, theta = 5120, 64, 8, 128, 1e-6, 5_000_000.0
    embed = dequantize(tensors["model.embed_tokens.weight"])  # [vocab][hidden]
    x = torch.from_numpy(embed[ids.numpy()])
    positions = torch.arange(len(ids), dtype=torch.float64)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    angles = positions[:, None] * inv_freq[None, :]
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).float()
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).float()
    started = time.time()
    for layer in range(layers if args.layers is None else args.layers):
        prefix = f"model.layers.{layer}."
        w = {name: torch.from_numpy(dequantize(tensors[prefix + name])) for name in (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")}
        h = rms_norm(x, w["input_layernorm.weight"], eps)
        q = (h @ w["self_attn.q_proj.weight"].T).view(-1, heads, head_dim)
        k = (h @ w["self_attn.k_proj.weight"].T).view(-1, kv_heads, head_dim)
        v = (h @ w["self_attn.v_proj.weight"].T).view(-1, kv_heads, head_dim)
        q = rms_norm(q, w["self_attn.q_norm.weight"], eps)
        k = rms_norm(k, w["self_attn.k_norm.weight"], eps)
        q = q * cos[:, None, :] + rotate_half(q) * sin[:, None, :]
        k = k * cos[:, None, :] + rotate_half(k) * sin[:, None, :]
        k = k.repeat_interleave(heads // kv_heads, dim=1)
        v = v.repeat_interleave(heads // kv_heads, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) / head_dim ** 0.5
        mask = torch.triu(torch.ones(len(ids), len(ids), dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask[None], float("-inf")).softmax(-1)
        attn = torch.einsum("hqk,khd->qhd", scores, v).reshape(-1, heads * head_dim)
        x = x + attn @ w["self_attn.o_proj.weight"].T
        h = rms_norm(x, w["post_attention_layernorm.weight"], eps)
        gate = h @ w["mlp.gate_proj.weight"].T
        up = h @ w["mlp.up_proj.weight"].T
        x = x + (torch.nn.functional.silu(gate) * up) @ w["mlp.down_proj.weight"].T
        print(f"layer {layer} done, {time.time() - started:.0f}s, |x| {x.norm():.2f}", flush=True)
    np.savez(args.out, input_ids=ids.numpy().astype(np.int64), hidden=x.numpy(), prompt=np.array(args.prompt))
    print("wrote", args.out, x.shape)


def dit(args: argparse.Namespace) -> None:
    """One forward of the pruned DiT from the GGUF, mirroring diffusers' MiniMaxH3Transformer3DModel with
    stable-diffusion.cpp's pruned adaLN table and fused fc1 = [gate; up] convention, on the inputs the native
    step used (tools/h3_reference.py is the reference, so the layout is rebuilt here from the pipeline's rules)."""
    inputs = np.load(args.inputs)
    video = torch.from_numpy(inputs["video"])      # [C][T][H][W]
    audio = torch.from_numpy(inputs["audio"])      # [2][A][32]
    caption = torch.from_numpy(inputs["caption"])  # [text][5120]
    t_video, t_audio = float(inputs["t_video"]), float(inputs["t_audio"])
    tensors = read_gguf(args.gguf)
    hidden, heads, head_dim, ffn, patch, eps = 5376, 56, 128, 14336, 2, 1e-5
    inner = heads * head_dim
    channels, T, H, W = video.shape
    A = audio.shape[1]
    text = caption.shape[0]
    w = lambda name: torch.from_numpy(dequantize(tensors[name]))  # noqa: E731

    # ---- packed layout (before_denoise.build_packed_sequence, t2va) ----
    th, tw = H // patch, W // patch
    sqrt_area = float(np.sqrt(H * W))
    def axis(dim):
        ratio = dim / sqrt_area; left = (1.0 - ratio) / 2.0
        return np.linspace(left, left + ratio, dim // patch, endpoint=False) * 32.0
    h_axis, w_axis = axis(H), axis(W)
    rows_per_frame = th * tw
    audio_rows, video_rows = A * 2, T * rows_per_frame
    rows = text + audio_rows + video_rows
    positions = np.zeros((rows, 3), dtype=np.float64)
    positions[:text, 0] = np.arange(text)
    audio_start, video_start = text, text + audio_rows
    positions[audio_start:video_start, 0] = np.tile(text + np.arange(A), 2)
    positions[audio_start:video_start, 2] = np.concatenate([np.full(A, w_axis[0]), np.full(A, w_axis[-1])])
    spans = np.array([5.0 / 3.0 * (1, 4, 4, 4, 4)[i % 5] for i in range(T)])
    times = text + np.concatenate([[0.0], np.cumsum(spans[:-1])])
    grid = np.stack(np.meshgrid(h_axis, w_axis, indexing="ij"), -1).reshape(-1, 2)
    for t in range(T):
        block = positions[video_start + t * rows_per_frame: video_start + (t + 1) * rows_per_frame]
        block[:, 0] = times[t]; block[:, 1:] = grid
    tags = np.concatenate([np.ones(text), np.full(audio_rows, 2), np.zeros(video_rows)]).astype(np.int64)
    timesteps = [t_video] if t_video == t_audio else [t_video, t_audio]
    t_index = np.concatenate([np.zeros(text), np.full(audio_rows, timesteps.index(t_audio)), np.zeros(video_rows)]).astype(np.int64)
    adaln_indices = torch.from_numpy(t_index * 3 + tags)

    # ---- rope (MiniMaxH3RotaryPosEmbed) ----
    inv_freq = w("rope.inv_freq")  # [16]
    freqs = torch.from_numpy(positions).float()[:, :, None] * inv_freq[None, None, :]  # [rows][3][16]
    freqs = freqs.flatten(1)
    freqs = torch.cat([freqs, freqs], -1)
    cos, sin = freqs.cos(), freqs.sin()
    rotary = cos.shape[-1]
    def rope(x):  # x [rows][heads][head_dim]
        xr, xp = x[..., :rotary], x[..., rotary:]
        x1, x2 = xr.chunk(2, -1)
        rot = torch.cat([-x2, x1], -1)
        return torch.cat([xr * cos[:, None, :] + rot * sin[:, None, :], xp], -1)

    # ---- pruned adaLN table: linear interpolation in t ----
    table = w("adaln_t_table")  # [1025][8]
    def curve(t):
        pos = min(max(t, 0.0), 1.0) * (table.shape[0] - 1)
        lo = min(int(np.floor(pos)), table.shape[0] - 2); fr = pos - lo
        return table[lo] + (table[lo + 1] - table[lo]) * fr
    temb = torch.stack([curve(t) for t in timesteps])  # [n_t][8]

    def attention(x, q_norm, k_norm, qkv_w, out_w, use_rope):
        qkv = x @ qkv_w.T
        q, k, v = qkv.split(inner, -1)
        q = rms_norm(q.view(-1, heads, head_dim), q_norm, eps); k = rms_norm(k.view(-1, heads, head_dim), k_norm, eps)
        v = v.view(-1, heads, head_dim)
        if use_rope: q, k = rope(q), rope(k)
        scores = torch.einsum("qhd,khd->hqk", q, k) / head_dim ** 0.5
        o = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).reshape(-1, inner)
        return o @ out_w.T

    def swiglu(x, fc1, fc2):
        gate, up = (x @ fc1.T).chunk(2, -1)   # the GGUF's [gate; up] order
        return (torch.nn.functional.silu(gate) * up) @ fc2.T

    # ---- embeddings ----
    started = time.time()
    x = torch.zeros(rows, hidden)
    ctx = caption @ w("condition_proj.weight").T + w("condition_proj.bias")
    for i in range(2):
        p = f"token_refiner.blocks.{i}."
        ctx = ctx + attention(rms_norm(ctx, w(p + "norm1.weight"), eps), w(p + "attn.q_norm.weight"), w(p + "attn.k_norm.weight"),
                              w(p + "attn.qkv_proj.weight"), w(p + "attn.out_proj.weight"), False)
        ctx = ctx + swiglu(rms_norm(ctx, w(p + "norm2.weight"), eps), w(p + "mlp.fc1.weight"), w(p + "mlp.fc2.weight"))
    x[:text] = rms_norm(ctx, w("token_refiner.final_norm.weight"), eps)
    audio_rows_t = audio.reshape(-1, 32)   # channel-major rows
    x[audio_start:video_start] = audio_rows_t @ w("audio_patch_proj.weight").T + w("audio_patch_proj.bias")
    # patchify (c, ph, pw) features, rows (t, h, w)
    patches = video.view(channels, T, th, patch, tw, patch).permute(1, 2, 4, 0, 3, 5).reshape(video_rows, -1)
    x[video_start:] = patches @ w("video_patch_proj.weight").T + w("video_patch_proj.bias")
    print(f"embedded {rows} rows in {time.time() - started:.0f}s", flush=True)

    for i in range(50):
        p = f"blocks.{i}."
        mods = temb @ w(p + "adaln_proj.linear.weight").T + w(p + "adaln_proj.linear.bias")  # [n_t][3*6*hidden]
        mods = mods.view(-1, 6 * hidden)  # [n_t*3][6*hidden], row = t*3 + modality
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, -1)
        h = rms_norm(x, w(p + "norm1.weight"), eps) * (1 + scale_msa[adaln_indices]) + shift_msa[adaln_indices]
        x = x + gate_msa[adaln_indices] * attention(h, w(p + "attn.q_norm.weight"), w(p + "attn.k_norm.weight"),
                                                    w(p + "attn.qkv_proj.weight"), w(p + "attn.out_proj.weight"), True)
        h = rms_norm(x, w(p + "norm2.weight"), eps) * (1 + scale_mlp[adaln_indices]) + shift_mlp[adaln_indices]
        x = x + gate_mlp[adaln_indices] * swiglu(h, w(p + "mlp.fc1.weight"), w(p + "mlp.fc2.weight"))
        if i % 10 == 9: print(f"block {i} done, {time.time() - started:.0f}s, |x| {x.norm():.2f}", flush=True)
    fmods = temb @ w("final_layer.adaln_proj.linear.weight").T + w("final_layer.adaln_proj.linear.bias")
    shift, scale = fmods.chunk(2, -1)
    t_rows = torch.from_numpy(t_index)
    h = rms_norm(x, w("final_layer.norm.weight"), eps) * (1 + scale[t_rows]) + shift[t_rows]
    video_out = h[video_start:] @ w("final_layer.video_out.weight").T + w("final_layer.video_out.bias")
    audio_out = h[audio_start:video_start] @ w("final_layer.audio_out.weight").T + w("final_layer.audio_out.bias")
    video_out = video_out.view(T, th, tw, channels, patch, patch).permute(3, 0, 1, 4, 2, 5).reshape(channels, T, H, W)
    np.savez(args.out, video=video_out.numpy(), audio=audio_out.view(2, A, 32).numpy())
    print("wrote", args.out)


def vae(args: argparse.Namespace) -> None:
    """Decode normalized latents with the ViT decoder from the fp16 safetensors, mirroring diffusers'
    AutoencoderKLMiniMaxH3._decode (chunking, blending) with the file's own conventions: per-head-interleaved
    to_qkv and fc1 = [gate; up]. Writes the frames as uint8 [F][H][W][3]."""
    import json as json_module
    from safetensors.numpy import load_file
    from safetensors import safe_open

    config = json_module.load(open(args.config))
    weights = load_file(args.weights)
    w = lambda name: torch.from_numpy(weights[name].astype(np.float32))  # noqa: E731
    latents = torch.from_numpy(np.load(args.inputs)["latents"])  # [C][T][H][W] normalized
    mean = torch.tensor(config["latents_mean"]).view(-1, 1, 1, 1); std = torch.tensor(config["latents_std"]).view(-1, 1, 1, 1)
    z = latents * std + mean
    dim, heads, head_dim, layers, registers = 2048, 32, 64, 36, 4
    ffn = dim * 4; eps = 1e-5; ps, pt = 16, 4
    inv_freq = 1.0 / (100.0 ** torch.arange(0, 1, 6 / 48))   # 8 per axis
    post_w, post_b = w("post_quant_conv.weight").reshape(24, 24), w("post_quant_conv.bias")

    def decode_clip(clip):  # [C][F][H][W] -> [3][F*4][H*16][W*16]
        C, F, Hh, Ww = clip.shape
        x = clip.permute(1, 2, 3, 0).reshape(-1, C) @ post_w.T + post_b
        h = x @ w("decoder.x_embedder.weight").T + w("decoder.x_embedder.bias")
        n = h.shape[0]
        h = torch.cat([h, w("decoder.register_tokens").reshape(registers, dim), torch.zeros(1, dim)], 0)
        grids = [2.0 * (torch.arange(0.5, size) / size) - 1.0 for size in (F, Hh, Ww)]
        pos = torch.stack(torch.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 3)
        pos = torch.cat([pos, torch.zeros(registers + 1, 3)], 0)
        angles = 2 * math.pi * pos[:, :, None] * inv_freq[None, None, :]
        angles = angles.flatten(1); angles = torch.cat([angles, angles], -1)
        cos, sin = angles.cos(), angles.sin(); rot = cos.shape[-1]
        def rope(t):
            tr, tp = t[..., :rot], t[..., rot:]; a, b = tr.chunk(2, -1)
            return torch.cat([tr * cos[:, None] + torch.cat([-b, a], -1) * sin[:, None], tp], -1)
        for i in range(layers):
            p = f"decoder.transformer_blocks.{i}."
            hn = rms_norm(h, w(p + "norm1.weight"), eps)
            qkv = hn @ w(p + "attn.to_qkv.weight").T + w(p + "attn.to_qkv.bias")
            qkv = qkv.view(-1, heads, 3, head_dim)   # per-head interleaved [head][q k v][d]
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps); k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + eps)
            q, k = rope(q), rope(k)
            scores = torch.einsum("qhd,khd->hqk", q, k) / head_dim ** 0.5
            o = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).reshape(-1, dim)
            h = h + (o @ w(p + "attn.to_out.weight").T + w(p + "attn.to_out.bias")) * w(p + "scale1")
            hn = rms_norm(h, w(p + "norm2.weight"), eps)
            gate, up = (hn @ w(p + "ff.w1.weight").T + w(p + "ff.w1.bias")).chunk(2, -1)
            h = h + ((torch.nn.functional.silu(gate) * up) @ w(p + "ff.w2.weight").T + w(p + "ff.w2.bias")) * w(p + "scale2")
        h = torch.nn.functional.layer_norm(h, (dim,), w("decoder.norm_out.weight"), w("decoder.norm_out.bias"), eps)
        out = (h @ w("decoder.proj_out.weight").T + w("decoder.proj_out.bias"))[:n]
        out = out.view(F, Hh, Ww, 3, pt, ps, ps).permute(3, 0, 4, 1, 5, 2, 6).reshape(3, F * pt, Hh * ps, Ww * ps)
        return out

    # diffusers' spatial tiling (`use_tiling` defaults on): 256-pixel tiles, >= 64 overlap, latent-aligned.
    def split_tiles(length, tile=256, min_overlap=64, ratio=16):
        if tile >= length:
            return [0], [length], []
        num = math.ceil(length / tile)
        while tile * num - min_overlap * (num - 1) - length < 0:
            num += 1
        overlaps = [min_overlap] * (num - 1)
        for i in range((tile * num - sum(overlaps) - length) // ratio):
            overlaps[i % (num - 1)] += ratio
        starts = [0]
        for i in range(num - 1):
            starts.append(starts[-1] + tile - overlaps[i])
        return starts, [tile] * num, overlaps

    def blend(a, b, extent, dim):
        extent = min(a.shape[dim], b.shape[dim], extent)
        weights_ = torch.arange(extent).float() / extent
        shape = [1] * a.ndim; shape[dim] = extent
        head = a.narrow(dim, a.shape[dim] - extent, extent) * (1 - weights_.view(shape)) + b.narrow(dim, 0, extent) * weights_.view(shape)
        return head if extent == b.shape[dim] else torch.cat([head, b.narrow(dim, extent, b.shape[dim] - extent)], dim)

    def decode_clip_tiled(clip):
        C, F, Hh, Ww = clip.shape
        ys, ylens, yov = split_tiles(Hh * 16); xs, xlens, xov = split_tiles(Ww * 16)
        tiles = [[decode_clip(clip[:, :, y // 16: (y + yl) // 16, x // 16: (x + xl) // 16]) for x, xl in zip(xs, xlens)] for y, yl in zip(ys, ylens)]
        rows = []
        for i, row in enumerate(tiles):
            out_row = []
            for j, tile in enumerate(row):
                if i > 0: tile = blend(tiles[i - 1][j], tile, yov[i - 1], -2)
                if j > 0: tile = blend(row[j - 1], tile, xov[j - 1], -1)
                if i < len(tiles) - 1: tile = tile[..., : tile.shape[-2] - yov[i], :]
                if j < len(row) - 1: tile = tile[..., : tile.shape[-1] - xov[j]]
                out_row.append(tile)
            rows.append(torch.cat(out_row, -1))
        return torch.cat(rows, -2)

    # _decode chunking
    clip_length, token_drop, temporal = 17, 3, 4
    tokens_chunk = math.ceil(clip_length / temporal); token_overlap = (-token_drop) % tokens_chunk
    pre_padding = (-clip_length) % temporal; frame_overlap = max(token_overlap * temporal - pre_padding, 0)
    chunk_frames = tokens_chunk * temporal
    num_tokens = z.shape[1] + token_drop; pad_tokens = (-num_tokens) % tokens_chunk
    num_chunks = (num_tokens + pad_tokens) // tokens_chunk - int(token_drop > 0)
    if num_chunks < 1:
        pad_tokens += tokens_chunk; num_chunks += 1
    if pad_tokens > 0: z = torch.cat([z, z[:, -1:].repeat(1, pad_tokens, 1, 1)], 1)
    decoded, overlap = [], None
    started = time.time()
    for i in range(num_chunks):
        start = i * tokens_chunk
        clip = decode_clip_tiled(z[:, start: start + tokens_chunk + token_overlap])
        print(f"clip {i} decoded, {time.time() - started:.0f}s", flush=True)
        for j in range(int(token_drop > 0) + 1):
            fs = j * chunk_frames
            chunk = clip[:, fs: fs + chunk_frames][:, pre_padding:]
            if chunk.shape[1] == 0: continue
            if j == 0:
                if overlap is not None:
                    blend = min(frame_overlap, overlap.shape[1], chunk.shape[1])
                    weights_ = torch.arange(blend).float() / blend
                    chunk = chunk.clone()
                    chunk[:, :blend] = overlap[:, -blend:] * (1 - weights_.view(1, -1, 1, 1)) + chunk[:, :blend] * weights_.view(1, -1, 1, 1)
                decoded.append(chunk)
            else:
                overlap = chunk
    if overlap is not None: decoded.append(overlap)
    dec = torch.cat(decoded, 1)
    if pad_tokens > 0:
        intra_tail = clip_length % temporal
        before = z.shape[1] - pad_tokens
        pad_frames = sum(intra_tail if intra_tail and (before + k) % tokens_chunk == 0 else temporal for k in range(pad_tokens))
        dec = dec[:, :-pad_frames]
    frames = ((dec.permute(1, 2, 3, 0) * 0.5 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    np.savez(args.out, frames=frames)
    print("wrote", args.out, frames.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    enc = sub.add_parser("encode")
    enc.add_argument("--gguf", required=True, help="qwen3vl_32b_minimax_h3-*.gguf")
    enc.add_argument("--tokenizer", required=True, help="the official repo's tokenizer/ directory")
    enc.add_argument("--prompt", default="a red panda stepping along a mossy log in a misty forest, cinematic")
    enc.add_argument("--layers", type=int, default=None, help="stop early (debugging)")
    enc.add_argument("--out", required=True)
    enc.set_defaults(func=encode)
    step = sub.add_parser("dit")
    step.add_argument("--gguf", required=True, help="minimax_h3_*_pruned-*.gguf")
    step.add_argument("--inputs", required=True, help="npz with video, audio, caption, t_video, t_audio")
    step.add_argument("--out", required=True)
    step.set_defaults(func=dit)
    dec = sub.add_parser("vae")
    dec.add_argument("--weights", required=True, help="minimax_h3_video_vae_fp16.safetensors")
    dec.add_argument("--config", required=True, help="the official vae/config.json")
    dec.add_argument("--inputs", required=True, help="npz with `latents` [C][T][H][W], normalized")
    dec.add_argument("--out", required=True)
    dec.set_defaults(func=vae)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
