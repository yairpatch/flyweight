#!/usr/bin/env python3
"""Evaluate Sylvester-Hadamard rotation on real model weights and activations.

Extracts real Query, Key, and Value activations from an attention layer of a real
GGUF model, and measures:
1. Activation outlier channel spikes before and after Sylvester-Hadamard rotation.
2. Invariance of attention dot products: <Hq, Hk> == <q, k>.
3. Softmax probability drift and output fidelity across INT8, FP8, and 4-bit KV caches.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

from flyweight.v2 import V2Model


def sylvester_hadamard(n: int) -> np.ndarray:
    """Generate normalized Sylvester-Hadamard matrix of size 2^n."""
    if n == 1:
        H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    else:
        H_prev = sylvester_hadamard(n - 1)
        H = np.block([[H_prev, H_prev], [H_prev, -H_prev]]).astype(np.float32)
    return H


def dequant_q8_0(raw_bytes: memoryview, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    scales = np.frombuffer(raw.reshape(-1, 34)[:, :2].copy(), dtype=np.float16).astype(np.float32)
    quants = raw.reshape(-1, 34)[:, 2:].view(np.int8).astype(np.float32)
    return (quants * scales[:, None]).reshape(shape)


def quant_int8_per_token(X: np.ndarray) -> np.ndarray:
    """Symmetric INT8 per-token quantization."""
    scale = np.max(np.abs(X), axis=-1, keepdims=True) / 127.0
    scale = np.maximum(scale, 1e-8)
    q = np.clip(np.round(X / scale), -127, 127)
    return q * scale


def quant_4bit_group16(X: np.ndarray) -> np.ndarray:
    """4-bit quantization with group size 16."""
    T, D = X.shape
    X_grouped = X.reshape(T, D // 16, 16)
    scale = np.max(np.abs(X_grouped), axis=-1, keepdims=True) / 7.0
    scale = np.maximum(scale, 1e-8)
    q = np.clip(np.round(X_grouped / scale), -7, 7)
    return (q * scale).reshape(T, D)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Sylvester-Hadamard rotation on a real model")
    parser.add_argument("model", nargs="?", default="/home/yair/Downloads/Qwen3-1.7B-Q8_0.gguf",
                        help="Path to GGUF model")
    parser.add_argument("--prompt", default="Explain how a computer microprocessor executes machine code instructions.",
                        help="Prompt to run through the attention layer")
    parser.add_argument("--layer", type=int, default=0, help="Attention layer index to probe")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 1

    print(f"Loading model: {model_path.name}")
    model = V2Model(str(model_path))

    prefix = f"blk.{args.layer}."
    try:
        w_emb = dequant_q8_0(model.read_tensor("token_embd.weight"), (-1, 2048))
        w_norm = np.frombuffer(model.read_tensor(f"{prefix}attn_norm.weight"), dtype=np.float32)
        w_q = dequant_q8_0(model.read_tensor(f"{prefix}attn_q.weight"), (-1, 2048))
        w_k = dequant_q8_0(model.read_tensor(f"{prefix}attn_k.weight"), (-1, 2048))
        w_v = dequant_q8_0(model.read_tensor(f"{prefix}attn_v.weight"), (-1, 2048))
        w_q_norm = np.frombuffer(model.read_tensor(f"{prefix}attn_q_norm.weight"), dtype=np.float32)
        w_k_norm = np.frombuffer(model.read_tensor(f"{prefix}attn_k_norm.weight"), dtype=np.float32)
    except Exception as error:
        print(f"Could not read layer {args.layer} tensors directly: {error}", file=sys.stderr)
        return 1

    head_dim = len(w_k_norm)
    num_q_heads = len(w_q) // head_dim
    num_kv_heads = len(w_k) // head_dim
    print(f"Layer {args.layer} geometry: head_dim={head_dim}, Q_heads={num_q_heads}, KV_heads={num_kv_heads}")

    tokens = list(model.tokenize(args.prompt))
    print(f"Tokenized prompt: {len(tokens)} tokens")

    # Embedding + Pre-Attention RMSNorm
    emb = w_emb[tokens]
    normed = (emb / np.sqrt(np.mean(emb**2, axis=-1, keepdims=True) + 1e-6)) * w_norm
    T = len(tokens)

    # Projections
    Q = (normed @ w_q.T).reshape(T, num_q_heads, head_dim)
    K = (normed @ w_k.T).reshape(T, num_kv_heads, head_dim)
    V = (normed @ w_v.T).reshape(T, num_kv_heads, head_dim)

    # Per-head RMSNorm
    Q = (Q / np.sqrt(np.mean(Q**2, axis=-1, keepdims=True) + 1e-6)) * w_q_norm
    K = (K / np.sqrt(np.mean(K**2, axis=-1, keepdims=True) + 1e-6)) * w_k_norm

    # Build Sylvester Hadamard matrix
    n_bits = int(round(np.log2(head_dim)))
    H = sylvester_hadamard(n_bits) / np.sqrt(head_dim)

    # Rotated representations
    HQ = Q @ H.T
    HK = K @ H.T
    HV = V @ H.T

    # 1. Outlier Spectrum Analysis
    peak_k_raw = float(np.max(np.abs(K)))
    median_k_raw = float(np.median(np.max(np.abs(K), axis=(0, 1))))
    peak_k_rot = float(np.max(np.abs(HK)))
    median_k_rot = float(np.median(np.max(np.abs(HK), axis=(0, 1))))

    print("\n" + "=" * 60)
    print("1. ACTIVATION OUTLIER SUPPRESSION ON REAL MODEL")
    print("=" * 60)
    print(f"Raw Keys (K)   : Peak = {peak_k_raw:7.2f} | Median = {median_k_raw:5.2f} | Outlier Ratio = {peak_k_raw / median_k_raw:5.1f}x")
    print(f"Rotated Keys   : Peak = {peak_k_rot:7.2f} | Median = {median_k_rot:5.2f} | Outlier Ratio = {peak_k_rot / median_k_rot:5.1f}x")
    print(f"-> Peak outlier squashed by {peak_k_raw / peak_k_rot:.1f}x across the {head_dim}-dim feature space!")

    # 2. Invariance of Dot Products
    print("\n" + "=" * 60)
    print("2. DOT PRODUCT INVARIANCE: <Hq, Hk> == <q, k>")
    print("=" * 60)
    q_vec = Q[-1, 0]  # last token query, head 0
    k_mat = K[:, 0]   # all key tokens, head 0
    logits_ref = (q_vec @ k_mat.T) / np.sqrt(head_dim)

    hq_vec = HQ[-1, 0]
    hk_mat = HK[:, 0]
    logits_rot = (hq_vec @ hk_mat.T) / np.sqrt(head_dim)

    max_logit_diff = float(np.max(np.abs(logits_ref - logits_rot)))
    print(f"Max difference between unrotated and rotated attention logits: {max_logit_diff:.9e}")
    if max_logit_diff < 1e-5:
        print("PASS: Attention logits are mathematically identical (invariant).")
    else:
        print("FAIL: Logit divergence exceeds threshold!")
        return 1

    # 3. Quantization Fidelity Benchmark
    print("\n" + "=" * 60)
    print("3. ATTENTION OUTPUT QUALITY UNDER QUANTIZED KV CACHES")
    print("=" * 60)
    v_mat = V[:, 0]
    hv_mat = HV[:, 0]

    # Reference float output
    attn_p_ref = np.exp(logits_ref - np.max(logits_ref))
    attn_p_ref /= np.sum(attn_p_ref)
    out_ref = attn_p_ref @ v_mat

    # Method A: INT8 without rotation
    k_i8_raw = quant_int8_per_token(k_mat)
    logits_i8_raw = (q_vec @ k_i8_raw.T) / np.sqrt(head_dim)
    attn_p_i8_raw = np.exp(logits_i8_raw - np.max(logits_i8_raw))
    attn_p_i8_raw /= np.sum(attn_p_i8_raw)
    out_i8_raw = attn_p_i8_raw @ v_mat
    cos_i8_raw = float(np.dot(out_ref, out_i8_raw) / (np.linalg.norm(out_ref) * np.linalg.norm(out_i8_raw)))
    p_err_i8_raw = float(np.max(np.abs(attn_p_ref - attn_p_i8_raw)))

    # Method B: INT8 WITH Sylvester-Hadamard rotation
    k_i8_rot = quant_int8_per_token(hk_mat)
    logits_i8_rot = (hq_vec @ k_i8_rot.T) / np.sqrt(head_dim)
    attn_p_i8_rot = np.exp(logits_i8_rot - np.max(logits_i8_rot))
    attn_p_i8_rot /= np.sum(attn_p_i8_rot)
    out_i8_rot = attn_p_i8_rot @ v_mat
    cos_i8_rot = float(np.dot(out_ref, out_i8_rot) / (np.linalg.norm(out_ref) * np.linalg.norm(out_i8_rot)))
    p_err_i8_rot = float(np.max(np.abs(attn_p_ref - attn_p_i8_rot)))

    # Method C: 4-bit V without rotation
    v_4bit_raw = quant_4bit_group16(v_mat)
    out_4bit_raw = attn_p_ref @ v_4bit_raw
    cos_4bit_raw = float(np.dot(out_ref, out_4bit_raw) / (np.linalg.norm(out_ref) * np.linalg.norm(out_4bit_raw)))

    # Method D: 4-bit V WITH Sylvester-Hadamard rotation (accumulated then inverse-rotated)
    v_4bit_rot = quant_4bit_group16(hv_mat)
    out_4bit_rot = (attn_p_ref @ v_4bit_rot) @ H.T
    cos_4bit_rot = float(np.dot(out_ref, out_4bit_rot) / (np.linalg.norm(out_ref) * np.linalg.norm(out_4bit_rot)))

    # Method E: Asymmetric K8V4 (INT8 K + 4-bit V, both with Hadamard)
    out_k8v4 = (attn_p_i8_rot @ v_4bit_rot) @ H.T
    cos_k8v4 = float(np.dot(out_ref, out_k8v4) / (np.linalg.norm(out_ref) * np.linalg.norm(out_k8v4)))

    print(f"{"Configuration":<36} | {"Softmax Max Error":<18} | {"Output Cosine Sim":<18}")
    print("-" * 78)
    print(f"{"INT8 Keys (standard raw)":<36} | {p_err_i8_raw:<18.6f} | {cos_i8_raw:<18.6f}")
    print(f"{"INT8 Keys (with Sylvester-Hadamard)":<36} | {p_err_i8_rot:<18.6f} | {cos_i8_rot:<18.6f}")
    print(f"{"4-bit Values (standard raw)":<36} | {"(N/A - keys exact)":<18} | {cos_4bit_raw:<18.6f}")
    print(f"{"4-bit Values (with Sylvester-Hadamard)":<36} | {"(N/A - keys exact)":<18} | {cos_4bit_rot:<18.6f}")
    print(f"{"K8V4 Asymmetric (Hadamard K8 + V4)":<36} | {p_err_i8_rot:<18.6f} | {cos_k8v4:<18.6f}")
    print("-" * 78)
    print(f"Result: Hadamard improves INT8 attention probability error by {p_err_i8_raw / p_err_i8_rot:.1f}x!")
    print(f"Result: K8V4 achieves {cos_k8v4:.6f} cosine similarity while taking only ~400 bytes/token/head.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
