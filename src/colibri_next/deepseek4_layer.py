"""A whole DeepSeek-V4 block, composed from the individual components.

This runs a sliding-window block -- compress ratio zero -- end to end over a
batch of positions, which is what turns a pile of separately-checked kernels
into something that can be compared against the reference as a unit. The
compressed-attention kinds are not here yet.

It is deliberately written for clarity rather than speed: one token at a time,
no batching, no caching between calls. Its job is to establish that the
components compose correctly, so the runtime path can be built against a known
good answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colibri_next.deepseek4 import (
    attention,
    expert_matvec,
    grouped_matvec,
    hyper_connection,
    matvec,
    rms_norm,
    rope,
    route,
    swiglu,
)


def _f32(model, name: str) -> np.ndarray:
    info = model.tensor(name)
    count = 1
    for dimension in info["shape"]:
        count *= int(dimension)
    return np.frombuffer(
        bytes(model.read_tensor_slice(name, 0, count * 4)), dtype=np.float32, count=count
    )


def _i32(model, name: str) -> np.ndarray:
    info = model.tensor(name)
    shape = tuple(int(dimension) for dimension in info["shape"])
    count = shape[0] * shape[1]
    raw = model.read_tensor_slice(name, 0, count * 4)
    # GGUF reports [used, vocabulary]; the payload is vocabulary-major.
    return np.frombuffer(bytes(raw), dtype=np.int32, count=count).reshape(
        shape[1], shape[0]
    )


@dataclass
class BlockTrace:
    """Intermediates worth comparing against the reference, per step."""

    hc_attn_pre: np.ndarray
    attention: np.ndarray
    derope: np.ndarray
    attn_out: np.ndarray
    hc_attn_post: np.ndarray
    hc_ffn_pre: np.ndarray
    ffn_norm: np.ndarray
    experts: np.ndarray
    expert_weights: np.ndarray
    moe_out: np.ndarray
    ffn_out: np.ndarray
    output: np.ndarray


class SlidingWindowBlock:
    """One ratio-0 block, holding the weights it needs."""

    def __init__(self, model, layer: int):
        self.model = model
        self.layer = layer
        config = model.config
        self.n_embd = int(config["hidden_size"])
        self.heads = int(config["attention_heads"])
        self.head_dim = int(config["kv_lora_rank"])
        self.rope_dim = int(config["rotary_dimension"])
        self.q_lora = int(config["q_lora_rank"])
        self.hc = int(config["hyper_connection_count"])
        self.sinkhorn_iterations = int(config["sinkhorn_iterations"])
        self.experts_used = int(config["expert_used_count"])
        self.experts = int(config["expert_count"])
        self.expert_ffn = int(config["intermediate_size"])
        self.groups = int(config["output_group_count"])
        self.lora_rank = int(config["output_lora_rank"])
        self.weight_scale = 1.5
        self.epsilon = float(config["rms_norm_epsilon"])
        self.freq_base = float(config["rope_freq_base"])
        self.window = int(config["sliding_window"]) or None
        self.hashed = layer < int(config["hash_layer_count"])
        self.clamp = 10.0

        prefix = f"blk.{layer}."
        mix = (2 + self.hc) * self.hc
        self.hc_attn = (
            _f32(model, prefix + "hc_attn_fn.weight").reshape(mix, self.hc * self.n_embd),
            _f32(model, prefix + "hc_attn_scale.weight"),
            _f32(model, prefix + "hc_attn_base.weight"),
        )
        self.hc_ffn = (
            _f32(model, prefix + "hc_ffn_fn.weight").reshape(mix, self.hc * self.n_embd),
            _f32(model, prefix + "hc_ffn_scale.weight"),
            _f32(model, prefix + "hc_ffn_base.weight"),
        )
        self.attn_norm = _f32(model, prefix + "attn_norm.weight")
        self.q_a_norm = _f32(model, prefix + "attn_q_a_norm.weight")
        self.kv_a_norm = _f32(model, prefix + "attn_kv_a_norm.weight")
        self.sinks = _f32(model, prefix + "attn_sinks.weight")
        self.ffn_norm_weight = _f32(model, prefix + "ffn_norm.weight")
        self.bias = None if self.hashed else _f32(model, prefix + "exp_probs_b.bias")
        self.table = _i32(model, prefix + "ffn_gate_tid2eid.weight") if self.hashed else None

    def _mix(self, streams, weights):
        mixer, scale, base = weights
        return hyper_connection(
            streams, mixer, scale, base,
            sinkhorn_iterations=self.sinkhorn_iterations,
            rms_epsilon=self.epsilon, hc_epsilon=self.epsilon,
        )

    def _feed_forward(self, hidden, token):
        prefix = f"blk.{self.layer}."
        logits = matvec(self.model, prefix + "ffn_gate_inp.weight", hidden, self.experts)
        chosen, weights = route(
            logits, self.bias, used=self.experts_used,
            weight_scale=self.weight_scale,
            experts=self.table[token] if self.hashed else None,
        )
        total = np.zeros(self.n_embd, dtype=np.float32)
        for slot, expert in enumerate(chosen):
            gate = expert_matvec(
                self.model, prefix + "ffn_gate_exps.weight", int(expert), hidden, self.expert_ffn
            )
            up = expert_matvec(
                self.model, prefix + "ffn_up_exps.weight", int(expert), hidden, self.expert_ffn
            )
            activated = swiglu(gate, up, self.clamp)
            down = expert_matvec(
                self.model, prefix + "ffn_down_exps.weight", int(expert), activated, self.n_embd
            )
            total += down * weights[slot]
        shared_gate = matvec(self.model, prefix + "ffn_gate_shexp.weight", hidden, self.expert_ffn)
        shared_up = matvec(self.model, prefix + "ffn_up_shexp.weight", hidden, self.expert_ffn)
        shared = matvec(
            self.model, prefix + "ffn_down_shexp.weight",
            swiglu(shared_gate, shared_up, self.clamp), self.n_embd,
        )
        return chosen, weights, total, total + shared

    def forward(self, streams: np.ndarray, tokens: list[int]) -> tuple[np.ndarray, BlockTrace]:
        """Run the block over every position. `streams` is [positions, hc, n_embd]."""
        positions = len(tokens)
        prefix = f"blk.{self.layer}."

        attn_mix = [self._mix(streams[p], self.hc_attn) for p in range(positions)]
        collapsed = np.stack([m.collapsed for m in attn_mix])

        queries, latents = [], []
        for position in range(positions):
            normed = rms_norm(collapsed[position], self.attn_norm, epsilon=self.epsilon)
            low_rank = matvec(self.model, prefix + "attn_q_a.weight", normed, self.q_lora)
            query = matvec(
                self.model, prefix + "attn_q_b.weight",
                rms_norm(low_rank, self.q_a_norm, epsilon=self.epsilon),
                self.heads * self.head_dim,
            ).reshape(self.heads, self.head_dim)
            query = rms_norm(query, None, epsilon=self.epsilon)
            queries.append(rope(query, position, self.rope_dim, freq_base=self.freq_base))

            latent = rms_norm(
                matvec(self.model, prefix + "attn_kv.weight", normed, self.head_dim),
                self.kv_a_norm, epsilon=self.epsilon,
            )
            # The cache holds f16, and the reference's own numbers reflect that.
            latents.append(
                rope(latent, position, self.rope_dim, freq_base=self.freq_base)
                .astype(np.float16).astype(np.float32)
            )
        latents = np.stack(latents)

        raw, deroped, outputs = [], [], []
        for position in range(positions):
            visible = np.zeros(positions, dtype=np.uint8)
            first = 0 if self.window is None else max(0, position - self.window + 1)
            visible[first : position + 1] = 1
            out = attention(
                queries[position], latents, self.sinks,
                scale=float(self.head_dim) ** -0.5, mask=visible,
            )
            raw.append(out)
            # Undo the rotation before projecting.
            back = rope(
                out, position, self.rope_dim, freq_base=self.freq_base, inverse=True
            )
            deroped.append(back)
            grouped = grouped_matvec(
                self.model, prefix + "attn_output_a.weight", back.reshape(-1),
                back.size // self.groups, self.lora_rank, self.groups,
            )
            outputs.append(
                matvec(self.model, prefix + "attn_output_b.weight", grouped, self.n_embd)
            )

        after_attention = np.stack([
            hyper_connection_combine_wrapper(attn_mix[p], outputs[p], streams[p])
            for p in range(positions)
        ])

        ffn_mix = [self._mix(after_attention[p], self.hc_ffn) for p in range(positions)]
        chosen_all, weights_all, moe_all, ffn_all, final = [], [], [], [], []
        for position in range(positions):
            hidden = rms_norm(
                ffn_mix[position].collapsed, self.ffn_norm_weight, epsilon=self.epsilon
            )
            chosen, weights, moe, ffn = self._feed_forward(hidden, tokens[position])
            chosen_all.append(chosen)
            weights_all.append(weights)
            moe_all.append(moe)
            ffn_all.append(ffn)
            final.append(
                hyper_connection_combine_wrapper(
                    ffn_mix[position], ffn, after_attention[position]
                )
            )

        trace = BlockTrace(
            hc_attn_pre=collapsed,
            attention=np.stack(raw),
            derope=np.stack(deroped),
            attn_out=np.stack(outputs),
            hc_attn_post=after_attention,
            hc_ffn_pre=np.stack([m.collapsed for m in ffn_mix]),
            ffn_norm=np.stack([
                rms_norm(m.collapsed, self.ffn_norm_weight, epsilon=self.epsilon)
                for m in ffn_mix
            ]),
            experts=np.stack(chosen_all),
            expert_weights=np.stack(weights_all),
            moe_out=np.stack(moe_all),
            ffn_out=np.stack(ffn_all),
            output=np.stack(final),
        )
        return trace.output, trace


def hyper_connection_combine_wrapper(mix, block_output, streams):
    """Write a block's output back into the streams using an existing mix."""
    result = np.empty_like(streams)
    for dst in range(streams.shape[0]):
        value = block_output * mix.post[dst]
        for src in range(streams.shape[0]):
            value = value + streams[src] * mix.comb[src, dst]
        result[dst] = value
    return result
