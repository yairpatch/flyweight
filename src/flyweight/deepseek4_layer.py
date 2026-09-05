"""DeepSeek-V4 blocks and the whole stack, composed from the native kernels.

All three attention kinds are here: the plain sliding window, and both
compressed kinds with their block compressors. What is not here is the
lightning indexer's top-k selection, which only bites once there are more
compressed tokens than `indexer_top_k` -- 512 of them, so beyond roughly 2048
raw tokens on a 4:1 layer.

It is deliberately written for clarity rather than speed: one token at a time,
no batching, no state carried between calls. Its job is to be the known-good
answer the runtime path gets built against, not to be that path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flyweight.deepseek4 import (
    attention,
    head_collapse,
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


class DeepSeek4Block:
    """One block of any kind, holding the weights it needs.

    The kind is set by the layer's compress ratio: 0 is a plain sliding window,
    4 is compressed sparse attention, 128 is heavily compressed. All three share
    the same latent attention and feed-forward; they differ in which keys the
    query may see and, for a non-zero ratio, in which rotation the query and key
    take -- the compressed frequency base with YaRN rather than the model's own.
    """

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
        ratios = model.compress_ratios
        self.ratio = int(ratios[layer]) if layer < len(ratios) else 0
        # A non-zero ratio moves the query and key rotation onto the compressed
        # frequency base with YaRN. This follows the reference rather than
        # measurement: at short prompt lengths the two are hard to tell apart.
        self.rope_kwargs: dict[str, Any]
        if self.ratio:
            scale = 1.0 / float(config["rope_scaling_factor"])
            self.rope_kwargs = dict(
                freq_base=float(config["compress_rope_freq_base"]),
                freq_scale=scale,
                ext_factor=1.0,
                attn_factor=1.0 / (1.0 + 0.1 * float(np.log(1.0 / scale))),
                beta_fast=float(config["yarn_beta_fast"]),
                beta_slow=float(config["yarn_beta_slow"]),
                original_context=int(config["rope_original_context_length"]),
            )
        else:
            self.rope_kwargs = dict(freq_base=self.freq_base)
        self.compressor = CompressedState(model, layer, self.ratio) if self.ratio else None

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
            queries.append(rope(query, position, self.rope_dim, **self.rope_kwargs))

            latent = rms_norm(
                matvec(self.model, prefix + "attn_kv.weight", normed, self.head_dim),
                self.kv_a_norm, epsilon=self.epsilon,
            )
            # The cache holds f16, and the reference's own numbers reflect that.
            latents.append(
                rope(latent, position, self.rope_dim, **self.rope_kwargs)
                .astype(np.float16).astype(np.float32)
            )
        latents = np.stack(latents)

        # A compressed layer summarizes completed blocks; below its ratio there
        # are none, which is why a short prompt runs a 128:1 layer as a plain
        # sliding window.
        compressed = (
            self.compressor.compress_blocks(
                np.stack([
                    rms_norm(m.collapsed, self.attn_norm, epsilon=self.epsilon)
                    for m in attn_mix
                ])
            ).astype(np.float16).astype(np.float32)
            if self.compressor is not None
            else np.zeros((0, self.head_dim), dtype=np.float32)
        )

        raw, deroped, outputs = [], [], []
        for position in range(positions):
            keys, visible = csa_attention_latents(
                latents, compressed, position, max(self.ratio, 1),
                self.window or positions,
            )
            out = attention(
                queries[position], keys, self.sinks,
                scale=float(self.head_dim) ** -0.5, mask=visible,
            )
            raw.append(out)
            # Undo the rotation before projecting.
            back = rope(
                out, position, self.rope_dim, inverse=True, **self.rope_kwargs
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
    """The per-layer compressor, for either compressed attention kind.

    The two kinds pool differently, and the difference is not just the ratio.

    A 4:1 layer overlaps its blocks. Each token projects to a state row twice
    the latent width -- a "prev" half and a "cur" half -- and a block pools
    eight entries: the previous four tokens' prev halves followed by its own
    four tokens' cur halves. That is the window of the last eight tokens the
    architecture keeps at each four-token boundary. The first block has no
    predecessor, so those rows read a padding entry scored -inf, which the
    softmax drops.

    A 128:1 layer does not overlap. Its state row is one latent wide and a block
    pools its own 128 tokens and nothing else.

    Either way the pooled latent is normalized, and its rope half rotated at the
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
        # A 4:1 layer keeps two half-rows per token to overlap its blocks; a
        # 128:1 layer keeps one. The checkpoint's compressor widths say so
        # directly: 1024 at layer 2 against 512 at layer 3.
        self.overlapped = ratio == 4
        self.width = (2 if self.overlapped else 1) * self.head_dim
        self.epsilon = float(config["rms_norm_epsilon"])
        self.freq_base = float(config["compress_rope_freq_base"])
        # No fallback here on purpose. These came from a default until the ABI
        # exposed them, and a wrong-but-plausible 1.0 silently disabled YaRN --
        # invisible on short prompts, because the block index barely moves.
        self.freq_scale = 1.0 / float(config["rope_scaling_factor"])
        # The reference passes an attenuation that exactly cancels the one ggml
        # applies for YaRN, leaving the magnitude unscaled.
        self.attn_factor = 1.0 / (1.0 + 0.1 * float(np.log(1.0 / self.freq_scale)))
        self.original_context = int(config["rope_original_context_length"])
        self.beta_fast = float(config["yarn_beta_fast"])
        self.beta_slow = float(config["yarn_beta_slow"])
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
            if self.overlapped:
                rows = 2 * self.ratio
                pooled_values = np.zeros((rows, self.head_dim), dtype=np.float32)
                pooled_scores = np.full((rows, self.head_dim), -np.inf, dtype=np.float32)
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
            else:
                span = slice(block * self.ratio, (block + 1) * self.ratio)
                pooled_values = values[span]
                pooled_scores = scores[span]
            pooled = compress(pooled_values, pooled_scores)
            pooled = rms_norm(pooled, self.norm, epsilon=self.epsilon)
            out.append(self._rotate(pooled, block))
        return np.stack(out)

    def _rotate(self, pooled: np.ndarray, block: int) -> np.ndarray:
        """Rotate a pooled latent at its block index."""
        return rope(
            pooled, block, self.rope_dim,
            freq_base=self.freq_base, freq_scale=self.freq_scale,
            ext_factor=1.0, attn_factor=self.attn_factor,
            beta_fast=self.beta_fast, beta_slow=self.beta_slow,
            original_context=self.original_context,
        )

    def pool_block(self, values, scores, block: int) -> np.ndarray:
        """Pool the newest complete block from accumulated state rows."""
        if self.overlapped:
            rows = 2 * self.ratio
            pv = np.zeros((rows, self.head_dim), dtype=np.float32)
            ps = np.full((rows, self.head_dim), -np.inf, dtype=np.float32)
            for slot in range(self.ratio):
                previous = (block - 1) * self.ratio + slot
                if previous >= 0:
                    pv[slot] = values[previous][: self.head_dim]
                    ps[slot] = scores[previous][: self.head_dim]
                current = block * self.ratio + slot
                pv[self.ratio + slot] = values[current][self.head_dim :]
                ps[self.ratio + slot] = scores[current][self.head_dim :]
        else:
            span = slice(block * self.ratio, (block + 1) * self.ratio)
            pv, ps = values[span], scores[span]
        return self._rotate(rms_norm(compress(pv, ps), self.norm, epsilon=self.epsilon), block)


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


class DeepSeek4Model:
    """Every block, then the output head. Clarity over speed.

    One token at a time, nothing cached between calls, no batching -- this is
    the known-good answer the runtime path gets built against, not the runtime
    path itself.
    """

    def __init__(self, model):
        self.model = model
        config = model.config
        self.n_embd = int(config["hidden_size"])
        self.hc = int(config["hyper_connection_count"])
        self.layers = int(config["layer_count"])
        self.vocabulary = int(config["vocabulary_size"])
        self.epsilon = float(config["rms_norm_epsilon"])
        self.blocks = [DeepSeek4Block(model, layer) for layer in range(self.layers)]
        self.head_fn = _f32(model, "output_hc_fn.weight").reshape(
            self.hc, self.hc * self.n_embd
        )
        self.head_scale = _f32(model, "output_hc_scale.weight")
        self.head_base = _f32(model, "output_hc_base.weight")
        self.output_norm = _f32(model, "output_norm.weight")

    def forward(self, tokens: list[int], *, last_only: bool = True) -> np.ndarray:
        """Return logits, for the final position by default."""
        streams = np.stack([
            np.repeat(
                np.asarray(self.model.qwen_embedding(token, self.n_embd), dtype=np.float32)[None, :],
                self.hc, axis=0,
            )
            for token in tokens
        ])
        for block in self.blocks:
            streams, _ = block.forward(streams, tokens)
        positions = [len(tokens) - 1] if last_only else range(len(tokens))
        out = []
        for position in positions:
            _, collapsed = head_collapse(
                streams[position], self.head_fn, self.head_scale, self.head_base,
                rms_epsilon=self.epsilon, hc_epsilon=self.epsilon,
            )
            normed = rms_norm(collapsed, self.output_norm, epsilon=self.epsilon)
            out.append(matvec(self.model, "output.weight", normed, self.vocabulary))
        return out[0] if last_only else np.stack(out)


class LayerCache:
    """Everything a block must remember between decode steps.

    A runtime cannot reprocess the prompt on every token, so each block carries
    its own state forward. Three things persist, and the third is the one that
    makes this more than an append-only cache:

    - the raw latents the sliding window attends to,
    - the compressed blocks completed so far,
    - the compressor's *partial* block: rows already projected for tokens whose
      block has not filled yet. They cannot be pooled until the block completes,
      and they cannot be recomputed later because the hidden state that produced
      them is gone.
    """

    def __init__(self, block: "DeepSeek4Block"):
        self.block = block
        self.latents: list[np.ndarray] = []
        self.compressed: list[np.ndarray] = []
        self.state_values: list[np.ndarray] = []
        self.state_scores: list[np.ndarray] = []

    @property
    def positions(self) -> int:
        return len(self.latents)

    def append_latent(self, latent: np.ndarray) -> None:
        self.latents.append(latent)

    def append_compressor_state(self, hidden: np.ndarray) -> None:
        """Project one token's compressor state and close the block if it fills."""
        state = self.block.compressor
        if state is None:
            return
        position = len(self.state_values)
        prefix = f"blk.{self.block.layer}."
        self.state_values.append(
            matvec(self.block.model, prefix + "attn_compressor_kv.weight", hidden, state.width)
        )
        score = matvec(
            self.block.model, prefix + "attn_compressor_gate.weight", hidden, state.width
        )
        self.state_scores.append(score + state.ape[position % state.ratio])
        if (position + 1) % state.ratio == 0:
            self.compressed.append(state.pool_block(
                np.stack(self.state_values), np.stack(self.state_scores),
                len(self.compressed),
            ))

    def keys(self, position: int) -> tuple[np.ndarray, np.ndarray]:
        raw = np.stack(self.latents)
        compressed = (
            np.stack(self.compressed) if self.compressed
            else np.zeros((0, self.block.head_dim), dtype=np.float32)
        )
        return csa_attention_latents(
            raw, compressed, position, max(self.block.ratio, 1),
            self.block.window or (position + 1),
        )
