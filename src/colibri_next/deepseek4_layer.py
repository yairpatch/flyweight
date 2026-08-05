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
    compress,
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


class CompressedState:
    """The per-layer compressor for a CSA (ratio 4) block.

    Each token projects to a state row twice the latent width: a "prev" half and
    a "cur" half. A block pools eight entries -- the previous `ratio` tokens'
    prev halves followed by its own `ratio` tokens' cur halves -- which is the
    overlap the architecture describes as keeping a window of the last eight
    tokens at each four-token boundary. Before the sequence starts there is no
    previous block, so those rows read a padding entry whose value is zero and
    whose score is -inf, contributing nothing to the softmax.

    The compressed latent is then normalized, and its rope half rotated at the
    *block* index using the compressed frequency base with YaRN -- a different
    rotation from the one the raw tokens get.
    """

    def __init__(self, model, layer: int, ratio: int):
        config = model.config
        self.model = model
        self.layer = layer
        self.ratio = ratio
        self.head_dim = int(config["kv_lora_rank"])
        self.rope_dim = int(config["rotary_dimension"])
        self.width = 2 * self.head_dim
        self.epsilon = float(config["rms_norm_epsilon"])
        self.freq_base = float(config["compress_rope_freq_base"])
        factor = float(config.get("rope_scaling_factor", 0.0)) or 1.0
        self.freq_scale = 1.0 / factor
        # The reference passes an attenuation that exactly cancels the one ggml
        # applies for YaRN, leaving the magnitude unscaled.
        self.attn_factor = 1.0 / (1.0 + 0.1 * float(np.log(1.0 / self.freq_scale)))
        self.original_context = int(config["rope_original_context_length"]) \
            if "rope_original_context_length" in config else 65536
        prefix = f"blk.{layer}."
        self.norm = _f32(model, prefix + "attn_compressor_norm.weight")
        self.ape = _f32(model, prefix + "attn_compressor_ape.weight").reshape(
            ratio, self.width
        )

    def states(self, hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project every token to its compressor state, with the slot embedding."""
        prefix = f"blk.{self.layer}."
        values, scores = [], []
        for position, vector in enumerate(hidden):
            values.append(matvec(self.model, prefix + "attn_compressor_kv.weight", vector, self.width))
            score = matvec(self.model, prefix + "attn_compressor_gate.weight", vector, self.width)
            scores.append(score + self.ape[position % self.ratio])
        return np.stack(values), np.stack(scores)

    def compress_blocks(self, hidden: np.ndarray) -> np.ndarray:
        """Compress every complete block; returns [blocks, head_dim]."""
        blocks = len(hidden) // self.ratio
        if blocks == 0:
            # Nothing to pool, and projecting the partial block would be wasted
            # work: it only matters once its remaining tokens arrive.
            return np.zeros((0, self.head_dim), dtype=np.float32)
        values, scores = self.states(hidden)
        out = []
        for block in range(blocks):
            pooled_values = np.zeros((2 * self.ratio, self.head_dim), dtype=np.float32)
            pooled_scores = np.full((2 * self.ratio, self.head_dim), -np.inf, dtype=np.float32)
            for slot in range(self.ratio):
                previous = (block - 1) * self.ratio + slot
                if previous >= 0:
                    # The prev half is the first `head_dim` of the state row.
                    pooled_values[slot] = values[previous][: self.head_dim]
                    pooled_scores[slot] = scores[previous][: self.head_dim]
                current = block * self.ratio + slot
                # The cur half is the second.
                pooled_values[self.ratio + slot] = values[current][self.head_dim :]
                pooled_scores[self.ratio + slot] = scores[current][self.head_dim :]
            pooled = compress(pooled_values, pooled_scores)
            pooled = rms_norm(pooled, self.norm, epsilon=self.epsilon)
            out.append(rope(
                pooled, block, self.rope_dim,
                freq_base=self.freq_base, freq_scale=self.freq_scale,
                ext_factor=1.0, attn_factor=self.attn_factor,
                original_context=self.original_context,
            ))
        return np.stack(out)


def csa_attention_latents(
    raw: np.ndarray, compressed: np.ndarray, position: int, ratio: int, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keys a CSA query at `position` may attend to, and their mask.

    The raw sliding window and the compressed blocks are both visible, and
    deliberately overlap: a token can be attended directly and again through its
    block's summary. The reference confirms this -- DSV4 raw attention uses the
    sliding-window half of its cache, and the compressed keys are concatenated
    after it rather than replacing anything.

    A block becomes visible once every token it covers is at or before the
    query, so block b is available from position b*ratio + ratio - 1 onward --
    the query that completes a block may already attend to it.

    That boundary was settled by comparing candidates against the reference
    rather than reasoned from the source, which does it through mask tensors
    built elsewhere. The winner is unambiguous: at layer 2 it lands within 0.85%
    while requiring the block to strictly precede the query gives 4.10%, keying
    on the block's first token 11.65%, and ignoring causality 40.25%.
    """
    first = max(0, position - window + 1)
    visible_raw = np.zeros(len(raw), dtype=np.uint8)
    visible_raw[first : position + 1] = 1

    blocks = len(compressed)
    visible_blocks = np.zeros(blocks, dtype=np.uint8)
    for block in range(blocks):
        if block * ratio + ratio - 1 <= position:
            visible_blocks[block] = 1

    latents = np.concatenate([raw, compressed]) if blocks else raw
    mask = np.concatenate([visible_raw, visible_blocks]) if blocks else visible_raw
    return latents, mask
