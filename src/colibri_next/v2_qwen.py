"""Format-neutral CPU reference execution for Qwen3.6 DeltaNet blocks."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .q4 import np
from .v2 import V2Model


@dataclass
class QwenDeltaState:
    conv: object
    recurrent: object
    tokens: int = 0
    cuda_conv: object = None
    cuda_recurrent: object = None

    def reset(self) -> None:
        self.conv.fill(0.0)
        self.recurrent.fill(0.0)
        if self.cuda_conv is not None:
            self.cuda_conv.fill(0.0)
            self.cuda_recurrent.fill(0.0)
        self.tokens = 0


class QwenDeltaLayer:
    def __init__(self, model: V2Model, layer: int, *, cuda_only: bool = False):
        if np is None:
            raise RuntimeError("Qwen v2 CPU reference requires numpy")
        self.model, self.layer = model, layer
        self.hidden = int(model.config["hidden_size"])
        self._raw_by_role = {}
        self.norm = self._vector("input_norm")
        if cuda_only:
            qkv_info = self.model.qwen_layer_tensor(self.layer, "qkv")
            gate_info = self.model.qwen_layer_tensor(self.layer, "attention_gate")
            output_info = self.model.qwen_layer_tensor(self.layer, "ssm_output")
            for role, info in (("qkv", qkv_info), ("attention_gate", gate_info), ("ssm_output", output_info)):
                self._raw_by_role[role] = self.model.view_tensor(info["name"])
            self.qkv = self.gate = self.output = None
        else:
            self.qkv = self._matrix("qkv")
            self.gate = self._matrix("attention_gate")
            output_info = None
        self.alpha = self._matrix("ssm_alpha")
        self.beta = self._matrix("ssm_beta")
        conv_decoded = self._tensor("ssm_conv")
        # GGML stores the 4-tap dimension contiguously.  The logical
        # convolution table is [channels, kernel].
        self.conv = conv_decoded.reshape(tuple(reversed(conv_decoded.shape)))
        self.dt_bias = self._vector("ssm_dt_bias")
        self.a_log = self._vector("ssm_a")
        self.ssm_norm = self._vector("ssm_norm")
        if not cuda_only:
            self.output = self._matrix("ssm_output")
        self.conv_dim = int(qkv_info["shape"][1]) if cuda_only else self.qkv.shape[1]
        self.value_heads = len(self.a_log)
        self.value_dim = int(output_info["shape"][0]) if cuda_only else self.output.shape[0]
        self.head_dim = self.value_dim // self.value_heads
        self.key_dim = self.conv_dim - self.value_dim
        if self.key_dim <= 0 or self.key_dim % 2:
            raise ValueError("invalid Qwen DeltaNet Q/K/V projection geometry")
        self.key_dim //= 2
        self.key_heads = self.key_dim // self.head_dim
        self.head_repeats = self.value_heads // self.key_heads
        self.kernel = self.conv.shape[1]
        self.epsilon = float(model.config.get("rms_norm_epsilon") or 1e-6)

    def new_state(self) -> QwenDeltaState:
        return QwenDeltaState(
            np.zeros((self.conv_dim, self.kernel), dtype=np.float32),
            np.zeros((self.value_heads, self.head_dim, self.head_dim), dtype=np.float32),
        )

    def forward(self, hidden: list[float], state: QwenDeltaState) -> list[float]:
        if self.qkv is None or self.gate is None or self.output is None:
            raise RuntimeError("CUDA-only DeltaNet layer cannot execute on CPU")
        if len(hidden) != self.hidden:
            raise ValueError(f"expected hidden width {self.hidden}, got {len(hidden)}")
        vector = np.asarray(hidden, dtype=np.float32)
        # GGUF stores Qwen RMSNorm weights in their one-centered form
        # (the original weight is already 1 + delta).
        normalized = vector / np.sqrt(np.mean(vector * vector) + self.epsilon) * self.norm
        mixed = normalized @ self.qkv
        gates = normalized @ self.gate
        beta_logits = normalized @ self.beta
        decay_logits = normalized @ self.alpha
        state.conv[:, :-1] = state.conv[:, 1:]
        state.conv[:, -1] = mixed
        convolved = (state.conv * self.conv).sum(axis=1)
        convolved = convolved / (1.0 + np.exp(-convolved))
        queries = convolved[: self.key_dim].reshape(self.key_heads, self.head_dim)
        keys = convolved[self.key_dim : self.key_dim * 2].reshape(self.key_heads, self.head_dim)
        values = convolved[self.key_dim * 2 :].reshape(self.value_heads, self.head_dim)
        # The GGUF Qwen3-Next conversion stores value heads as
        # [0, 2, ..., 30, 1, 3, ..., 31].  Q/K heads therefore expand as
        # [0..15, 0..15], rather than adjacent repeated pairs.
        queries = np.tile(queries, (self.head_repeats, 1))
        keys = np.tile(keys, (self.head_repeats, 1))
        # Queries and keys are independently normalized.  Do not alias these
        # arrays: mutating queries must not renormalize keys a second time.
        # Queries and keys are independently normalized after GQA expansion.
        queries = queries.copy()
        queries /= np.sqrt(np.sum(queries * queries, axis=1, keepdims=True) + 1e-6)
        keys /= np.sqrt(np.sum(keys * keys, axis=1, keepdims=True) + 1e-6)
        queries *= self.head_dim ** -0.5
        beta = 1.0 / (1.0 + np.exp(-np.clip(beta_logits, -80.0, 80.0)))
        # GGUF's ``ssm_a`` is already the transformed negative decay
        # coefficient (-exp(A_log)), unlike the source checkpoint's A_log.
        decay = self.a_log * np.logaddexp(0.0, decay_logits + self.dt_bias)
        state.recurrent *= np.exp(decay)[:, None, None]
        memory = np.einsum("hkv,hk->hv", state.recurrent, keys)
        delta = (values - memory) * beta[:, None]
        state.recurrent += keys[:, :, None] * delta[:, None, :]
        core = np.einsum("hkv,hk->hv", state.recurrent, queries)
        core /= np.sqrt(np.mean(core * core, axis=1, keepdims=True) + self.epsilon)
        core *= self.ssm_norm.reshape(1, -1)
        gate_values = gates.reshape(self.value_heads, self.head_dim)
        clipped_gates = np.clip(gate_values, -80.0, 80.0)
        core *= gate_values / (1.0 + np.exp(-clipped_gates))
        state.tokens += 1
        return (core.reshape(-1) @ self.output).tolist()

    def forward_residual(self, hidden: list[float], state: QwenDeltaState) -> list[float]:
        mixed = self.forward(hidden, state)
        return [left + right for left, right in zip(hidden, mixed)]

    def forward_cuda(self, hidden: list[float], state: QwenDeltaState, accelerator=None):
        """Execute one real GGUF Q8 DeltaNet block on CUDA.

        The recurrent state is allocated once and remains device-resident
        across calls. Qwen's matrices are [input, output], so the accelerator
        uses its GGML-layout-aware transposed Q8 matvec.
        """
        if accelerator is None:
            from .cuda import active_cuda
            accelerator = active_cuda()
        if accelerator is None:
            raise RuntimeError("CUDA is not configured")
        cp = accelerator.cp
        vector = cp.asarray(hidden, dtype=cp.float32)
        norm = accelerator._float32_array(self.norm)
        normalized = vector / cp.sqrt(cp.mean(vector * vector) + self.epsilon)
        normalized *= norm

        qkv_info = self.model.qwen_layer_tensor(self.layer, "qkv")
        gate_info = self.model.qwen_layer_tensor(self.layer, "attention_gate")
        output_info = self.model.qwen_layer_tensor(self.layer, "ssm_output")
        qkv_raw = self._raw_by_role["qkv"]
        gate_raw = self._raw_by_role["attention_gate"]
        output_raw = self._raw_by_role["ssm_output"]
        mixed = accelerator.q8_matvec_transposed(
            qkv_raw, self.hidden, int(qkv_info["shape"][1]), normalized,
            return_device=True, cache_weight=True, protect_weight=True,
        )
        gates = accelerator.q8_matvec_transposed(
            gate_raw, self.hidden, int(gate_info["shape"][1]), normalized,
            return_device=True, cache_weight=True, protect_weight=True,
        )
        alpha = normalized @ accelerator._float32_array(self.alpha)
        beta_logits = normalized @ accelerator._float32_array(self.beta)
        if state.cuda_conv is None:
            state.cuda_conv = cp.asarray(state.conv, dtype=cp.float32)
            state.cuda_recurrent = cp.asarray(state.recurrent, dtype=cp.float32)
        state.cuda_conv[:, :-1] = state.cuda_conv[:, 1:]
        state.cuda_conv[:, -1] = mixed
        conv_weights = accelerator._float32_array(self.conv)
        convolved = cp.sum(state.cuda_conv * conv_weights, axis=1)
        convolved = convolved / (cp.float32(1.0) + cp.exp(-convolved))
        queries = convolved[:self.key_dim].reshape(self.key_heads, self.head_dim)
        keys = convolved[self.key_dim:self.key_dim * 2].reshape(self.key_heads, self.head_dim)
        values = convolved[self.key_dim * 2:].reshape(self.value_heads, self.head_dim)
        queries = cp.tile(queries, (self.head_repeats, 1))
        keys = cp.tile(keys, (self.head_repeats, 1))
        queries = queries / cp.sqrt(cp.sum(queries * queries, axis=1, keepdims=True) + cp.float32(1e-6))
        keys = keys / cp.sqrt(cp.sum(keys * keys, axis=1, keepdims=True) + cp.float32(1e-6))
        queries *= cp.float32(self.head_dim ** -0.5)
        beta = cp.float32(1.0) / (cp.float32(1.0) + cp.exp(-beta_logits))
        decay = accelerator._float32_array(self.a_log) * cp.logaddexp(
            cp.float32(0.0), alpha + accelerator._float32_array(self.dt_bias)
        )
        state.cuda_recurrent *= cp.exp(decay)[:, None, None]
        memory = cp.einsum("hkv,hk->hv", state.cuda_recurrent, keys)
        delta = (values - memory) * beta[:, None]
        state.cuda_recurrent += keys[:, :, None] * delta[:, None, :]
        core = cp.einsum("hkv,hk->hv", state.cuda_recurrent, queries)
        core /= cp.sqrt(cp.mean(core * core, axis=1, keepdims=True) + self.epsilon)
        core *= accelerator._float32_array(self.ssm_norm).reshape(1, -1)
        gate_values = gates.reshape(self.value_heads, self.head_dim)
        core *= gate_values / (cp.float32(1.0) + cp.exp(-cp.clip(gate_values, -80.0, 80.0)))
        output = accelerator.q8_matvec_transposed(
            output_raw, self.value_dim, self.hidden, core.reshape(-1),
            return_device=True, cache_weight=True, protect_weight=True,
        )
        state.tokens += 1
        return output + vector

    def _tensor(self, role: str):
        info = self.model.qwen_layer_tensor(self.layer, role)
        raw = self.model.view_tensor(info["name"])
        self._raw_by_role[role] = raw
        count = math.prod(info["shape"])
        kind = info["ggml_type"]
        if kind == 0:
            return np.frombuffer(raw, dtype="<f4", count=count).reshape(info["shape"])
        if kind == 30:
            bits = np.frombuffer(raw, dtype="<u2", count=count).astype(np.uint32) << 16
            return bits.view(np.float32).reshape(info["shape"])
        if kind == 8:
            blocks = (count + 31) // 32
            scales = np.frombuffer(raw, dtype="<f2", count=blocks, offset=0)
            values = np.frombuffer(raw, dtype="i1", count=blocks * 32, offset=2 * blocks)
            # GGML Q8_0 stores each scale immediately before its 32 int8 values.
            output = np.empty(count, dtype=np.float32)
            for block in range(blocks):
                start = block * 34
                scale = np.frombuffer(raw, dtype="<f2", count=1, offset=start)[0]
                quant = np.frombuffer(raw, dtype="i1", count=min(32, count - block * 32), offset=start + 2)
                output[block * 32 : block * 32 + len(quant)] = quant * scale
            return output.reshape(info["shape"])
        raise ValueError(f"unsupported Qwen reference tensor type {kind} for {info['name']}")

    def _vector(self, role: str):
        return self._tensor(role).reshape(-1).astype(np.float32)

    def _matrix(self, role: str):
        # GGML keeps dimension 0 contiguous.  GGUF reports Qwen projection
        # shapes as [input, output], so the byte stream is physically
        # [output, input] and must be transposed after decoding.
        decoded = self._tensor(role)
        shape = tuple(decoded.shape)
        return decoded.reshape(tuple(reversed(shape))).T.astype(np.float32)


@dataclass
class QwenAttentionState:
    keys: list
    values: list
    tokens: int = 0
    cuda_keys: object = None
    cuda_values: object = None
    cuda_capacity: int = 0

    def reset(self) -> None:
        self.keys.clear()
        self.values.clear()
        self.tokens = 0
        self.cuda_keys = None
        self.cuda_values = None
        self.cuda_capacity = 0


class QwenFullAttentionLayer:
    def __init__(self, model: V2Model, layer: int, *, cuda_only: bool = False):
        if np is None:
            raise RuntimeError("Qwen v2 CPU reference requires numpy")
        self.model, self.layer = model, layer
        self.hidden = int(model.config["hidden_size"])
        self._raw_by_role = {}
        self.heads = int(model.config["attention_heads"])
        self.kv_heads = int(model.config["attention_kv_heads"])
        if cuda_only:
            infos = {
                role: self.model.qwen_layer_tensor(self.layer, role)
                for role in ("attention_q", "attention_k", "attention_v", "attention_output")
            }
            for role, info in infos.items():
                self._raw_by_role[role] = self.model.view_tensor(info["name"])
            self.q = self.k = self.v = self.output = None
        else:
            self.q = self._matrix("attention_q")
            self.k = self._matrix("attention_k")
            self.v = self._matrix("attention_v")
            self.output = self._matrix("attention_output")
            infos = None
        self.input_norm = self._vector("input_norm")
        self.q_norm = self._tensor("attention_q_norm").reshape(-1).astype(np.float32)
        self.k_norm = self._tensor("attention_k_norm").reshape(-1).astype(np.float32)
        self.head_dim = (
            int(infos["attention_k"]["shape"][1]) // self.kv_heads
            if cuda_only else self.k.shape[1] // self.kv_heads
        )
        self.rotary_dim = int(model.config.get("rotary_dimension") or self.head_dim)
        self.rope_theta = float(model.config.get("rope_freq_base") or 1_000_000.0)
        self.epsilon = float(model.config.get("rms_norm_epsilon") or 1e-6)

    def new_state(self) -> QwenAttentionState:
        return QwenAttentionState([[] for _ in range(self.kv_heads)], [[] for _ in range(self.kv_heads)])

    def forward_residual(self, hidden: list[float], state: QwenAttentionState) -> list[float]:
        if self.q is None or self.k is None or self.v is None or self.output is None:
            raise RuntimeError("CUDA-only attention layer cannot execute on CPU")
        vector = np.asarray(hidden, dtype=np.float32)
        normalized = vector / np.sqrt(np.mean(vector * vector) + self.epsilon) * self.input_norm
        projected_q = normalized @ self.q
        projected_k = normalized @ self.k
        projected_v = normalized @ self.v
        queries, gates = [], []
        for head in range(self.heads):
            start = head * self.head_dim * 2
            queries.append(self._rope(self._norm(projected_q[start:start + self.head_dim], self.q_norm), state.tokens))
            gates.extend(projected_q[start + self.head_dim:start + self.head_dim * 2])
        keys, values = [], []
        for head in range(self.kv_heads):
            start = head * self.head_dim
            keys.append(self._rope(self._norm(projected_k[start:start + self.head_dim], self.k_norm), state.tokens))
            values.append(projected_v[start:start + self.head_dim].tolist())
            state.keys[head].append(keys[-1])
            state.values[head].append(values[-1])
        groups = self.heads // self.kv_heads
        attended = []
        for head, query in enumerate(queries):
            kv = head // groups
            scores = np.asarray([np.dot(query, key) / math.sqrt(self.head_dim) for key in state.keys[kv]], dtype=np.float32)
            scores = np.exp(scores - np.max(scores)); scores /= np.sum(scores)
            attended.extend(np.sum(np.asarray(state.values[kv]) * scores[:, None], axis=0))
        gate_values = np.asarray(gates, dtype=np.float32)
        gated = np.asarray(attended, dtype=np.float32) / (
            1.0 + np.exp(-np.clip(gate_values, -80.0, 80.0))
        )
        state.tokens += 1
        return (vector + gated @ self.output).tolist()

    def forward_cuda(self, hidden: list[float], state: QwenAttentionState, accelerator=None):
        if accelerator is None:
            from .cuda import active_cuda
            accelerator = active_cuda()
        if accelerator is None:
            raise RuntimeError("CUDA is not configured")
        cp = accelerator.cp
        vector = cp.asarray(hidden, dtype=cp.float32)
        norm = accelerator._float32_array(self.input_norm)
        normalized = vector / cp.sqrt(cp.mean(vector * vector) + self.epsilon)
        normalized *= norm

        def q8(role, vector_value):
            info = self.model.qwen_layer_tensor(self.layer, role)
            raw = self._raw_by_role[role]
            return accelerator.q8_matvec_transposed(
                raw, int(info["shape"][0]), int(info["shape"][1]),
                vector_value, return_device=True, cache_weight=True,
                protect_weight=True,
            )

        projected_q = q8("attention_q", normalized)
        projected_k = q8("attention_k", normalized)
        projected_v = q8("attention_v", normalized)
        query_gate = projected_q.reshape(self.heads, 2, self.head_dim)
        queries = query_gate[:, 0, :]
        gates = query_gate[:, 1, :]
        q_norm = accelerator._float32_array(self.q_norm)
        k_norm = accelerator._float32_array(self.k_norm)
        queries = queries / cp.sqrt(cp.mean(queries * queries, axis=1, keepdims=True) + self.epsilon)
        queries *= q_norm
        keys = projected_k.reshape(self.kv_heads, self.head_dim)
        keys = keys / cp.sqrt(cp.mean(keys * keys, axis=1, keepdims=True) + self.epsilon)
        keys *= k_norm
        position = state.tokens
        half = self.head_dim // 2
        indices = cp.arange(half, dtype=cp.float32)
        angles = cp.float32(position) / cp.power(cp.float32(self.rope_theta), 2.0 * indices / self.head_dim)
        cosine, sine = cp.cos(angles), cp.sin(angles)
        queries = cp.concatenate((queries[:, :half] * cosine - queries[:, half:] * sine,
                                  queries[:, half:] * cosine + queries[:, :half] * sine), axis=1)
        keys = cp.concatenate((keys[:, :half] * cosine - keys[:, half:] * sine,
                               keys[:, half:] * cosine + keys[:, :half] * sine), axis=1)
        if state.cuda_keys is None or state.cuda_capacity <= position:
            capacity = max(16, state.cuda_capacity * 2, position + 1)
            new_keys = cp.zeros((self.kv_heads, capacity, self.head_dim), dtype=cp.float32)
            new_values = cp.zeros_like(new_keys)
            if state.cuda_keys is not None and state.tokens:
                new_keys[:, :state.tokens] = state.cuda_keys[:, :state.tokens]
                new_values[:, :state.tokens] = state.cuda_values[:, :state.tokens]
            state.cuda_keys, state.cuda_values, state.cuda_capacity = new_keys, new_values, capacity
        state.cuda_keys[:, position] = keys
        state.cuda_values[:, position] = projected_v.reshape(self.kv_heads, self.head_dim)
        scores = cp.einsum("ghd,gtd->ght", queries.reshape(self.kv_heads, self.heads // self.kv_heads, self.head_dim),
                           state.cuda_keys[:, :position + 1]) * cp.float32(self.head_dim ** -0.5)
        scores -= cp.max(scores, axis=2, keepdims=True)
        probabilities = cp.exp(scores)
        probabilities /= cp.sum(probabilities, axis=2, keepdims=True)
        attended = cp.einsum("ght,gtd->ghd", probabilities, state.cuda_values[:, :position + 1])
        attended = attended.reshape(self.heads, self.head_dim)
        gated = attended / (
            cp.float32(1.0) + cp.exp(-cp.clip(gates, -80.0, 80.0))
        )
        output = q8("attention_output", gated.reshape(-1))
        state.tokens += 1
        return output + vector

    def _norm(self, vector, weights):
        vector = np.asarray(vector, dtype=np.float32)
        return vector / np.sqrt(np.mean(vector * vector) + self.epsilon) * weights

    def _rope(self, vector, position):
        half = self.rotary_dim // 2
        first, second = vector[:half], vector[half:self.rotary_dim]
        indices = np.arange(half, dtype=np.float32)
        angles = position / (self.rope_theta ** (2.0 * indices / self.rotary_dim))
        return np.concatenate((first * np.cos(angles) - second * np.sin(angles),
                               second * np.cos(angles) + first * np.sin(angles), vector[self.rotary_dim:]))

    def _tensor(self, role):
        try:
            info = self.model.qwen_layer_tensor(self.layer, role)
        except Exception:
            aliases = {"attention_q_norm": "attn_q_norm.weight", "attention_k_norm": "attn_k_norm.weight"}
            if role not in aliases:
                raise
            info = self.model.tensor(f"blk.{self.layer}.{aliases[role]}")
        raw = self.model.view_tensor(info["name"])
        self._raw_by_role[role] = raw
        count = math.prod(info["shape"])
        if info["ggml_type"] == 0:
            return np.frombuffer(raw, dtype="<f4", count=count).reshape(info["shape"])
        if info["ggml_type"] == 30:
            bits = np.frombuffer(raw, dtype="<u2", count=count).astype(np.uint32) << 16
            return bits.view(np.float32).reshape(info["shape"])
        if info["ggml_type"] == 8:
            result = np.empty(count, dtype=np.float32)
            for block in range((count + 31) // 32):
                scale = np.frombuffer(raw, dtype="<f2", count=1, offset=block * 34)[0]
                size = min(32, count - block * 32)
                quant = np.frombuffer(raw, dtype="i1", count=size, offset=block * 34 + 2)
                result[block * 32:block * 32 + size] = quant * scale
            return result.reshape(info["shape"])
        raise ValueError(f"unsupported attention tensor type {info['ggml_type']}")

    def _vector(self, role): return self._tensor(role).reshape(-1).astype(np.float32)
    def _matrix(self, role):
        decoded = self._tensor(role)
        return decoded.reshape(tuple(reversed(decoded.shape))).T.astype(np.float32)


class QwenMoELayer:
    """CPU reference for Qwen router and shared-expert execution."""
    def __init__(
        self, model: V2Model, layer: int, *, cuda_only: bool = False,
        capture_debug: bool = True,
    ):
        if np is None:
            raise RuntimeError("Qwen v2 CPU reference requires numpy")
        self.model, self.layer = model, layer
        self.capture_debug = capture_debug
        self.hidden = int(model.config["hidden_size"])
        self.experts = int(model.config["expert_count"])
        self.top_k = int(model.config["expert_used_count"])
        self.router = self._matrix("router")
        if cuda_only:
            self.shared_gate = self.shared_up = self.shared_down = None
        else:
            self.shared_gate = self._matrix("shared_gate")
            self.shared_up = self._matrix("shared_up")
            self.shared_down = self._matrix("shared_down")
        self.shared_input_gate = self._tensor("shared_input_gate").reshape(-1)
        self.post_attention_norm = self._tensor("post_attention_norm").reshape(-1)
        self.epsilon = float(model.config.get("rms_norm_epsilon") or 1e-6)
        self.expert_gate_info = self._info("blk.%d.ffn_gate_exps.weight" % layer)
        self.expert_up_info = self._info("blk.%d.ffn_up_exps.weight" % layer)
        self.expert_down_info = self._info("blk.%d.ffn_down_exps.weight" % layer)
        self.shared_gate_info = self._info("blk.%d.ffn_gate_shexp.weight" % layer)
        self.shared_up_info = self._info("blk.%d.ffn_up_shexp.weight" % layer)
        self.shared_down_info = self._info("blk.%d.ffn_down_shexp.weight" % layer)
        # Expert tensors are large but shared by every routed token.  Keep one
        # mapped/read copy per tensor instead of re-reading all three tensors
        # once for every selected expert.
        self._expert_raw = {}
        self._expert_slices = {}
        self._cuda_raw = {}
        self._last_selected_ids: list[int] = []
        self._hinted_ids: set[int] = set()
        self._prefetch_event = None

    def prefetch_cuda(self, accelerator) -> None:
        if not accelerator.expert_prefetch_enabled or not self._last_selected_ids:
            return
        groups = [
            (
                self._expert_bytes(self.expert_gate_info, expert),
                self._expert_bytes(self.expert_up_info, expert),
                self._expert_bytes(self.expert_down_info, expert),
            )
            for expert in self._last_selected_ids
        ]
        self._hinted_ids = set(self._last_selected_ids)
        self._prefetch_event = accelerator.prefetch_v2_expert_groups(groups)

    def route(self, hidden: list[float]) -> tuple[list[int], list[float]]:
        vector = np.asarray(hidden, dtype=np.float32)
        logits = vector @ self.router
        selected = np.argpartition(logits, -self.top_k)[-self.top_k:]
        selected = selected[np.argsort(logits[selected])[::-1]]
        scores = np.exp(logits[selected] - np.max(logits[selected]))
        scores /= np.sum(scores)
        return selected.astype(int).tolist(), scores.astype(float).tolist()

    def shared_forward(self, hidden: list[float]) -> list[float]:
        if self.shared_gate is None or self.shared_up is None or self.shared_down is None:
            raise RuntimeError("CUDA-only MoE layer cannot execute on CPU")
        vector = np.asarray(hidden, dtype=np.float32)
        gate = vector @ self.shared_gate
        up = vector @ self.shared_up
        activated = gate / (1.0 + np.exp(-np.clip(gate, -80.0, 80.0))) * up
        output = activated @ self.shared_down
        scale_input = float(vector @ self.shared_input_gate)
        scale = 1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, scale_input))))
        return (output * scale).tolist()

    def forward(self, hidden: list[float]) -> tuple[list[float], list[int], list[float]]:
        selected, weights = self.route(hidden)
        output = np.asarray(self.shared_forward(hidden), dtype=np.float32)
        shared_output = output.copy()
        vector = np.asarray(hidden, dtype=np.float32)
        for expert, weight in zip(selected, weights):
            # GGUF stores these matrices as [input, output].
            gate = vector @ self._expert_matrix(self.expert_gate_info, expert)
            up = vector @ self._expert_matrix(self.expert_up_info, expert)
            activated = gate / (1.0 + np.exp(-np.clip(gate, -80.0, 80.0))) * up
            down = activated @ self._expert_matrix(self.expert_down_info, expert)
            if expert == selected[0]:
                self.last_cpu_expert = (gate.copy(), activated.copy(), down.copy())
            output += float(weight) * down
        self.last_cpu_shared = shared_output
        self.last_cpu_routed = output - shared_output
        return output.tolist(), selected, weights

    def forward_residual(self, hidden: list[float]) -> tuple[list[float], list[int], list[float]]:
        vector = np.asarray(hidden, dtype=np.float32)
        normalized = vector / np.sqrt(np.mean(vector * vector) + self.epsilon)
        normalized *= self.post_attention_norm
        output, selected, weights = self.forward(normalized.tolist())
        return (vector + np.asarray(output, dtype=np.float32)).tolist(), selected, weights

    def forward_cuda(
        self, hidden: list[float], accelerator=None, *,
        copy_route_weights: bool = True,
    ):
        """Route and execute this GGUF MoE layer on CUDA.

        Dense shared weights use Q8_0; routed Qwen experts use Q5_K/Q6_K.
        Expert bytes are sliced directly from the model mapping and are not
        materialized as Python/Numpy matrices.
        """
        if accelerator is None:
            from .cuda import active_cuda
            accelerator = active_cuda()
        if accelerator is None:
            raise RuntimeError("CUDA is not configured")
        cp = accelerator.cp
        route_profile = accelerator.profile_start()
        vector = cp.asarray(hidden, dtype=cp.float32)
        post_norm = accelerator._float32_array(self.post_attention_norm)
        normalized = vector / cp.sqrt(cp.mean(vector * vector) + self.epsilon)
        normalized *= post_norm
        router = accelerator._float32_array(self.router)
        logits = normalized @ router
        selected, weights = accelerator.route_topk_device(logits, self.top_k)
        accelerator.profile_end("router", route_profile)

        def q8(info, vector_value):
            raw = self._cuda_raw.get(info["name"])
            if raw is None:
                raw = self.model.view_tensor(info["name"])
                self._cuda_raw[info["name"]] = raw
            return accelerator.q8_matvec_transposed(
                raw, int(info["shape"][0]), int(info["shape"][1]),
                vector_value, return_device=True, cache_weight=True,
                protect_weight=True,
            )

        shared_profile = accelerator.profile_start()
        shared_gate = q8(self.shared_gate_info, normalized)
        shared_up = q8(self.shared_up_info, normalized)
        shared_activated = shared_gate / (cp.float32(1.0) + cp.exp(-cp.clip(shared_gate, -80.0, 80.0))) * shared_up
        shared = q8(self.shared_down_info, shared_activated)
        shared_input = accelerator._float32_array(self.shared_input_gate)
        shared_scale = cp.float32(1.0) / (cp.float32(1.0) + cp.exp(-cp.dot(normalized, shared_input)))
        output = shared * shared_scale
        accelerator.profile_end("shared_expert", shared_profile)
        if self.capture_debug:
            self.last_cuda_shared = output.copy()
            self.last_cuda_expert = None
        selected_ids = selected.get().tolist()
        if self._prefetch_event is not None:
            if not self._prefetch_event.done:
                accelerator.expert_prefetch_waits += 1
            cp.cuda.get_current_stream().wait_event(self._prefetch_event)
            accelerator.expert_prefetch_uses += sum(
                expert in self._hinted_ids for expert in selected_ids
            )
            self._prefetch_event = None
            self._hinted_ids.clear()
        self._last_selected_ids = [int(expert) for expert in selected_ids]

        fused_experts = (
            not self.capture_debug
            and self.expert_gate_info["ggml_type"] == 13
            and self.expert_up_info["ggml_type"] == 13
            and self.expert_down_info["ggml_type"] in {8, 14}
        ) or (
            not self.capture_debug
            and self.expert_gate_info["ggml_type"] == 40
            and self.expert_up_info["ggml_type"] == 40
            and self.expert_down_info["ggml_type"] == 40
        )
        expert_groups = []
        routed_profile = accelerator.profile_start()
        if fused_experts:
            slices_started = time.perf_counter()
            expert_groups = [
                (
                    self._expert_bytes(self.expert_gate_info, int(expert)),
                    self._expert_bytes(self.expert_up_info, int(expert)),
                    self._expert_bytes(self.expert_down_info, int(expert)),
                )
                for expert in selected_ids
            ]
            accelerator.profile_host(
                "routed_expert_slices", time.perf_counter() - slices_started
            )
            accelerator.q5k_q6k_grouped_swiglu_accumulate(
                expert_groups,
                int(self.expert_gate_info["shape"][0]),
                int(self.expert_gate_info["shape"][1]),
                int(self.expert_down_info["shape"][1]),
                normalized,
                output,
                weights,
                down_ggml_type=int(self.expert_down_info["ggml_type"]),
                gate_ggml_type=int(self.expert_gate_info["ggml_type"]),
            )

        expert_matvec_methods = {
            8: accelerator.q8_matvec_transposed,
            12: accelerator.q4k_matvec_transposed,
            13: accelerator.q5k_matvec_transposed,
            14: accelerator.q6k_matvec_transposed,
            40: accelerator.nvfp4_matvec_transposed,
        }

        def expert_matvec(info, raw, vec):
            method = expert_matvec_methods.get(info["ggml_type"])
            if method is None:
                raise ValueError(f"unsupported CUDA expert type {info['ggml_type']}")
            return method(
                raw, int(info["shape"][0]), int(info["shape"][1]), vec,
                return_device=True, cache_weight=True,
            )

        for index in range(self.top_k):
            expert = int(selected_ids[index])
            if fused_experts:
                continue
            gate_raw = self._expert_bytes(self.expert_gate_info, expert)
            up_raw = self._expert_bytes(self.expert_up_info, expert)
            down_raw = self._expert_bytes(self.expert_down_info, expert)
            gate = expert_matvec(self.expert_gate_info, gate_raw, normalized)
            up = expert_matvec(self.expert_up_info, up_raw, normalized)
            activated = gate / (cp.float32(1.0) + cp.exp(-cp.clip(gate, -80.0, 80.0))) * up
            down = expert_matvec(self.expert_down_info, down_raw, activated)
            if index == 0:
                self.last_cuda_expert = (gate.copy(), activated.copy(), down.copy())
            output += weights[index] * down
        accelerator.profile_end("routed_experts", routed_profile)
        if self.capture_debug:
            self.last_cuda_routed = output - self.last_cuda_shared
        route_weights = (
            weights.get().tolist() if copy_route_weights else weights
        )
        return vector + output, selected_ids, route_weights

    def _info(self, name):
        return self.model.tensor(name)

    def _expert_matrix(self, info, expert):
        shape = tuple(info["shape"])
        per_expert = math.prod(shape[:-1])
        raw = self._expert_raw.get(info["name"])
        if raw is None:
            raw = self.model.view_tensor(info["name"])
            self._expert_raw[info["name"]] = raw
        decoded = _decode_ggml(raw, info["ggml_type"], expert * per_expert, per_expert)
        # Expert is the slowest dimension in GGML.  Within one expert the
        # contiguous stream is [output, input], while the logical matmul is
        # [input, output].
        matrix_shape = tuple(shape[:-1])
        return decoded.reshape((matrix_shape[1], matrix_shape[0])).T

    def _expert_bytes(self, info, expert):
        key = (info["name"], int(expert))
        cached = self._expert_slices.get(key)
        if cached is not None:
            return cached
        raw = self._expert_raw.get(info["name"])
        per_expert = int(info["size"]) // self.experts
        start = expert * per_expert
        if raw is None:
            expert_bytes = self.model.view_tensor_slice(
                info["name"], start, per_expert
            )
        else:
            expert_bytes = memoryview(raw)[start:start + per_expert]
        self._expert_slices[key] = expert_bytes
        return expert_bytes

    def _tensor(self, role):
        aliases = {
            "router": "ffn_gate_inp.weight",
            "shared_gate": "ffn_gate_shexp.weight",
            "shared_up": "ffn_up_shexp.weight",
            "shared_down": "ffn_down_shexp.weight",
            "shared_input_gate": "ffn_gate_inp_shexp.weight",
        }
        try:
            info = self.model.qwen_layer_tensor(self.layer, role)
        except Exception:
            info = self.model.tensor(f"blk.{self.layer}.{aliases[role]}")
        raw = self.model.view_tensor(info["name"])
        count = math.prod(info["shape"])
        if info["ggml_type"] == 0:
            return np.frombuffer(raw, dtype="<f4", count=count).reshape(info["shape"])
        if info["ggml_type"] == 30:
            bits = np.frombuffer(raw, dtype="<u2", count=count).astype(np.uint32) << 16
            return bits.view(np.float32).reshape(info["shape"])
        if info["ggml_type"] == 8:
            result = np.empty(count, dtype=np.float32)
            for block in range((count + 31) // 32):
                scale = np.frombuffer(raw, dtype="<f2", count=1, offset=block * 34)[0]
                size = min(32, count - block * 32)
                quant = np.frombuffer(raw, dtype="i1", count=size, offset=block * 34 + 2)
                result[block * 32:block * 32 + size] = quant * scale
            return result.reshape(info["shape"])
        raise ValueError(f"unsupported shared MoE tensor type {info['ggml_type']}")

    def _matrix(self, role):
        decoded = self._tensor(role)
        return decoded.reshape(tuple(reversed(decoded.shape))).T.astype(np.float32)


class QwenDenseMLPLayer:
    """CPU reference for a dense Qwen3.5/3.6 SwiGLU block.

    The dense checkpoints replace the router, shared expert and stacked routed
    experts of the MoE variants with a single ffn_gate/ffn_up/ffn_down triple,
    so there is nothing to route and the whole block is one SwiGLU.
    """

    def __init__(self, model: V2Model, layer: int):
        if np is None:
            raise RuntimeError("Qwen v2 CPU reference requires numpy")
        self.model, self.layer = model, layer
        self.hidden = int(model.config["hidden_size"])
        self.gate = self._matrix("ffn_gate")
        self.up = self._matrix("ffn_up")
        self.down = self._matrix("ffn_down")
        self.post_attention_norm = self._tensor("post_attention_norm").reshape(-1)
        self.epsilon = float(model.config.get("rms_norm_epsilon") or 1e-6)

    def forward(self, hidden: list[float]) -> list[float]:
        vector = np.asarray(hidden, dtype=np.float32)
        gate = vector @ self.gate
        up = vector @ self.up
        activated = gate / (1.0 + np.exp(-np.clip(gate, -80.0, 80.0))) * up
        return (activated @ self.down).tolist()

    def forward_residual(self, hidden: list[float]) -> list[float]:
        vector = np.asarray(hidden, dtype=np.float32)
        normalized = vector / np.sqrt(np.mean(vector * vector) + self.epsilon)
        normalized *= self.post_attention_norm
        output = np.asarray(self.forward(normalized.tolist()), dtype=np.float32)
        return (vector + output).tolist()

    def _info(self, role: str):
        suffix = {
            "ffn_gate": "ffn_gate.weight",
            "ffn_up": "ffn_up.weight",
            "ffn_down": "ffn_down.weight",
            "post_attention_norm": "post_attention_norm.weight",
        }[role]
        return self.model.tensor(f"blk.{self.layer}.{suffix}")

    def _tensor(self, role: str):
        info = self._info(role)
        raw = self.model.view_tensor(info["name"])
        return _decode_tensor(raw, info["ggml_type"], math.prod(info["shape"])).reshape(info["shape"])

    def _matrix(self, role: str):
        # GGML keeps dimension 0 contiguous, so a GGUF shape of [input, output]
        # is physically [output, input] and has to be transposed after decoding.
        decoded = self._tensor(role)
        return decoded.reshape(tuple(reversed(decoded.shape))).T.astype(np.float32)


@dataclass
class QwenV2DecoderState:
    mixer_states: list[object]
    tokens: int = 0

    def reset(self) -> None:
        for state in self.mixer_states:
            if state is not None:
                state.reset()
        self.tokens = 0


class QwenV2Decoder:
    """Lazy, format-neutral Qwen block stack used as the v2 CPU contract.

    The stack deliberately owns only model-independent execution state. GGUF
    is consulted through V2Model's provider interface, so another provider can
    be used without changing this forward loop.
    """
    def __init__(self, model: V2Model):
        if np is None:
            raise RuntimeError("Qwen v2 CPU reference requires numpy")
        model.validate_qwen()
        self.model = model
        self.layers = int(model.config["layer_count"])
        self.hidden = int(model.config["hidden_size"])
        self._blocks: dict[int, object] = {}
        self._moe: dict[int, object] = {}
        # Dense checkpoints (for example Qwen3.6-27B) carry an ffn_gate triple
        # per block instead of a router and stacked experts.
        self.dense_ffn = self._has_tensor("blk.0.ffn_gate.weight")
        self._final_norm = None
        self._lm_head_info = None
        self._lm_head_raw = None

    def _has_tensor(self, name: str) -> bool:
        try:
            self.model.tensor(name)
        except Exception:
            return False
        return True

    def _feed_forward(self, layer: int, *, cuda_only: bool):
        block = self._moe.get(layer)
        if block is None:
            block = (
                QwenDenseMLPLayer(self.model, layer)
                if self.dense_ffn
                else QwenMoELayer(
                    self.model, layer, cuda_only=cuda_only, capture_debug=not cuda_only
                )
            )
            self._moe[layer] = block
        return block

    def new_state(self) -> QwenV2DecoderState:
        return QwenV2DecoderState([None] * self.layers)

    def forward_token(self, token_id: int, state: QwenV2DecoderState):
        if len(state.mixer_states) != self.layers:
            raise ValueError("decoder state does not match model layer count")
        hidden = self.model.qwen_embedding(token_id, self.hidden)
        routes = []
        for layer in range(self.layers):
            block = self._block(layer, cuda_only=False)
            if state.mixer_states[layer] is None:
                state.mixer_states[layer] = block.new_state()
            mixer_state = state.mixer_states[layer]
            if isinstance(block, QwenDeltaLayer):
                hidden = block.forward_residual(hidden, mixer_state)
            else:
                hidden = block.forward_residual(hidden, mixer_state)
            feed_forward = self._feed_forward(layer, cuda_only=False)
            if isinstance(feed_forward, QwenDenseMLPLayer):
                hidden = feed_forward.forward_residual(hidden)
            else:
                hidden, selected, weights = feed_forward.forward_residual(hidden)
                routes.append((selected, weights))
        state.tokens += 1
        return hidden, routes

    def forward_token_cuda(self, token_id: int, state: QwenV2DecoderState, accelerator=None):
        if accelerator is None:
            from .cuda import active_cuda
            accelerator = active_cuda()
        if accelerator is None:
            raise RuntimeError("CUDA is not configured")
        cp = accelerator.cp
        hidden = cp.asarray(self.model.qwen_embedding(token_id, self.hidden), dtype=cp.float32)
        routes = []
        for layer in range(self.layers):
            block = self._block(layer, cuda_only=True)
            moe = self._moe.get(layer)
            if moe is not None:
                moe.prefetch_cuda(accelerator)
            if state.mixer_states[layer] is None:
                state.mixer_states[layer] = block.new_state()
            mixer_state = state.mixer_states[layer]
            mixer_profile = accelerator.profile_start()
            hidden = block.forward_cuda(hidden, mixer_state, accelerator)
            accelerator.profile_end("mixer", mixer_profile)
            if moe is None:
                moe = QwenMoELayer(
                    self.model, layer, cuda_only=True, capture_debug=False
                )
                self._moe[layer] = moe
            hidden, selected, weights = moe.forward_cuda(
                hidden, accelerator, copy_route_weights=False
            )
            routes.append((selected, weights))
        state.tokens += 1
        return hidden, routes

    def logits_cuda(self, hidden, accelerator=None):
        if accelerator is None:
            from .cuda import active_cuda
            accelerator = active_cuda()
        if accelerator is None:
            raise RuntimeError("CUDA is not configured")
        cp = accelerator.cp
        lm_head_profile = accelerator.profile_start()
        if self._final_norm is None:
            norm_info = self.model.qwen_tensor("final_norm")
            norm_raw = self.model.view_tensor(norm_info["name"])
            self._final_norm = np.frombuffer(
                norm_raw, dtype="<f4", count=self.hidden
            ).copy()
        norm = accelerator._float32_array(self._final_norm)
        vector = cp.asarray(hidden, dtype=cp.float32)
        epsilon = cp.float32(float(self.model.config.get("rms_norm_epsilon") or 1e-6))
        normalized = vector / cp.sqrt(cp.mean(vector * vector) + epsilon)
        normalized *= norm
        if self._lm_head_info is None:
            self._lm_head_info = self.model.qwen_tensor("lm_head")
            self._lm_head_raw = self.model.view_tensor(self._lm_head_info["name"])
        head = self._lm_head_info
        raw = self._lm_head_raw
        lm_type = int(head["ggml_type"])
        if lm_type == 8:
            logits = accelerator.q8_matvec_transposed(
                raw, int(head["shape"][0]), int(head["shape"][1]), normalized,
                return_device=True, cache_weight=True, protect_weight=True,
            )
        else:
            lm_matvec = {
                12: accelerator.q4k_matvec_transposed,
                13: accelerator.q5k_matvec_transposed,
                14: accelerator.q6k_matvec_transposed,
            }.get(lm_type)
            if lm_matvec is None:
                raise ValueError(f"unsupported lm_head type {lm_type}")
            logits = lm_matvec(
                raw, int(head["shape"][0]), int(head["shape"][1]), normalized,
                return_device=True, cache_weight=True,
            )
        accelerator.profile_end("lm_head", lm_head_profile)
        return logits

    def _block(self, layer: int, *, cuda_only: bool):
        block = self._blocks.get(layer)
        if block is not None:
            return block
        try:
            self.model.qwen_layer_tensor(layer, "attention_q")
        except Exception:
            block = QwenDeltaLayer(self.model, layer, cuda_only=cuda_only)
        else:
            block = QwenFullAttentionLayer(self.model, layer, cuda_only=cuda_only)
        self._blocks[layer] = block
        return block


def _decode_q2k_blocks(raw, blocks: int):
    """Dequantize whole Q2_K super-blocks.

    Layout per 84-byte block: scales[16] qs[64] d(2) dmin(2). Each scales byte
    packs a 4-bit scale (low nibble) and 4-bit min (high nibble) for one
    16-element group. The 256 values are two 128-element halves; within a half
    the 2-bit quants for group j sit at bit offset 2*j of that half's 32 qs
    bytes, so the natural axis order is [half, group, sub, element].
    """
    data = np.frombuffer(raw, dtype=np.uint8, count=blocks * 84).reshape(blocks, 84)
    d = data[:, 80:82].copy().view("<f2").astype(np.float32).reshape(blocks, 1, 1, 1, 1)
    dmin = data[:, 82:84].copy().view("<f2").astype(np.float32).reshape(blocks, 1, 1, 1, 1)
    scales = data[:, 0:16].reshape(blocks, 2, 4, 2, 1)
    quants = data[:, 16:80].reshape(blocks, 2, 1, 2, 16)
    shifts = (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 4, 1, 1)
    values = (quants >> shifts) & 3
    decoded = d * (scales & 15).astype(np.float32) * values - dmin * (scales >> 4).astype(np.float32)
    return decoded.reshape(blocks * 256)


def _q3k_scales(data, blocks: int):
    """Unpack the sixteen 6-bit Q3_K scales from their 12-byte encoding.

    The low and high nibbles of bytes 0..7 carry each scale's low 4 bits, and
    bytes 8..11 supply the top 2 bits.
    """
    packed = data[:, 96:108].astype(np.uint16)
    index = np.arange(16)
    group, byte = index // 4, index % 4
    low = packed[:, np.where(group & 1, 4 + byte, byte)]
    nibble = np.where(group < 2, low & 15, low >> 4)
    high = (packed[:, 8 + byte] >> (2 * group)) & 3
    return (nibble | (high << 4)).astype(np.float32).reshape(blocks, 2, 4, 2, 1)


def _decode_q3k_blocks(raw, blocks: int):
    """Dequantize whole Q3_K super-blocks.

    Layout per 110-byte block: hmask[32] qs[64] scales[12] d(2). The quant is a
    2-bit low part from qs plus an inverted high bit from hmask -- a set mask
    bit means "do not subtract 4" -- giving a signed 3-bit value.
    """
    data = np.frombuffer(raw, dtype=np.uint8, count=blocks * 110).reshape(blocks, 110)
    d = data[:, 108:110].copy().view("<f2").astype(np.float32).reshape(blocks, 1, 1, 1, 1)
    quants = data[:, 32:96].reshape(blocks, 2, 1, 2, 16)
    shifts = (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 4, 1, 1)
    low = (quants >> shifts) & 3
    masks = (1 << (np.arange(2).reshape(2, 1) * 4 + np.arange(4).reshape(1, 4))).astype(np.uint8)
    hmask = data[:, 0:32].reshape(blocks, 1, 1, 2, 16)
    high = np.where(hmask & masks.reshape(1, 2, 4, 1, 1), 0, 4)
    return (d * (_q3k_scales(data, blocks) - 32) * (low.astype(np.float32) - high)).reshape(blocks * 256)


def _decode_tensor(raw, kind: int, count: int):
    """Decode a whole GGUF tensor to float32, vectorized over super-blocks."""
    if kind == 0:
        return np.frombuffer(raw, dtype="<f4", count=count).astype(np.float32)
    if kind == 30:
        bits = np.frombuffer(raw, dtype="<u2", count=count).astype(np.uint32) << 16
        return bits.view(np.float32)
    if kind == 8:
        blocks = count // 32
        data = np.frombuffer(raw, dtype=np.uint8, count=blocks * 34).reshape(blocks, 34)
        scale = data[:, 0:2].copy().view("<f2").astype(np.float32)
        return (data[:, 2:34].view(np.int8).astype(np.float32) * scale).reshape(count)
    if count % 256:
        raise ValueError(f"K-quant tensor length {count} is not a multiple of 256")
    blocks = count // 256
    if kind == 10:
        return _decode_q2k_blocks(raw, blocks)
    if kind == 11:
        return _decode_q3k_blocks(raw, blocks)
    if kind in (12, 13, 14):
        decoder = {12: _decode_q4k, 13: _decode_q5k, 14: _decode_q6k}[kind]
        return decoder(raw, 0, count)
    raise ValueError(f"unsupported GGUF tensor type {kind}")


def _decode_ggml(raw: bytes, kind: int, start: int, count: int):
    if kind == 0:
        return np.frombuffer(raw, dtype="<f4", count=count, offset=start * 4).astype(np.float32)
    if kind == 8:
        return _decode_q8(raw, start, count)
    if kind == 30:
        bits = np.frombuffer(raw, dtype="<u2", count=count, offset=start * 2).astype(np.uint32) << 16
        return bits.view(np.float32)
    if kind == 12:
        return _decode_q4k(raw, start, count)
    if kind == 13:
        return _decode_q5k(raw, start, count)
    if kind == 14:
        return _decode_q6k(raw, start, count)
    if kind == 40:
        return _decode_nvfp4(raw, start, count)
    raise ValueError(f"unsupported GGML expert type {kind}")


def _decode_q8(raw, start, count):
    output = np.empty(count, dtype=np.float32)
    for index in range(count):
        absolute = start + index
        block = absolute // 32
        scale = np.frombuffer(raw, dtype="<f2", count=1, offset=block * 34)[0]
        output[index] = np.frombuffer(raw, dtype="i1", count=1, offset=block * 34 + 2 + absolute % 32)[0] * scale
    return output


def _scale_min(scales, index):
    if index < 4:
        return scales[index] & 63, scales[index + 4] & 63
    return (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4), (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4)


def _decode_q4k(raw, start, count):
    # Q4_K is Q5_K without the 5th (high) bit: same super-block scales/mins,
    # but the qh block is absent so ql starts at +16 and quants are 4-bit.
    block_size, bytes_per_block = 256, 144  # d(2)+dmin(2)+scales(12)+qs(128)
    output = np.empty(count, dtype=np.float32)
    for local in range(count):
        absolute = start + local
        block, within = divmod(absolute, block_size)
        base = block * bytes_per_block
        d = float(np.frombuffer(raw, dtype="<f2", count=1, offset=base)[0])
        dmin = float(np.frombuffer(raw, dtype="<f2", count=1, offset=base + 2)[0])
        scales = raw[base + 4:base + 16]
        group, offset = divmod(within, 64)
        sub = offset // 32
        qindex = group * 32 + offset % 32
        low = raw[base + 16 + qindex]
        scale, minimum = _scale_min(scales, group * 2 + sub)
        quant = (low & 15) if offset < 32 else (low >> 4)
        output[local] = d * scale * quant - dmin * minimum
    return output


def _decode_q5k(raw, start, count):
    block_size, bytes_per_block = 256, 176
    output = np.empty(count, dtype=np.float32)
    for local in range(count):
        absolute = start + local
        block, within = divmod(absolute, block_size)
        base = block * bytes_per_block
        d = float(np.frombuffer(raw, dtype="<f2", count=1, offset=base)[0])
        dmin = float(np.frombuffer(raw, dtype="<f2", count=1, offset=base + 2)[0])
        scales = raw[base + 4:base + 16]
        group, offset = divmod(within, 64)
        sub = offset // 32
        qindex = group * 32 + offset % 32
        low = raw[base + 48 + qindex]
        high = (raw[base + 16 + (offset % 32)] >> (2 * group + sub)) & 1
        scale, minimum = _scale_min(scales, group * 2 + sub)
        quant = ((low & 15) if offset < 32 else (low >> 4)) + 16 * high
        output[local] = d * scale * quant - dmin * minimum
    return output


def _decode_q6k(raw, start, count):
    block_size, bytes_per_block = 256, 210
    output = np.empty(count, dtype=np.float32)
    for local in range(count):
        absolute = start + local
        block, within = divmod(absolute, block_size)
        base = block * bytes_per_block
        ql, qh = raw[base:base + 128], raw[base + 128:base + 192]
        scales = np.frombuffer(raw, dtype="i1", count=16, offset=base + 192)
        d = float(np.frombuffer(raw, dtype="<f2", count=1, offset=base + 208)[0])
        half, offset = divmod(within, 128)
        lane, l = divmod(offset, 32)
        # GGML's Q6_K layout interleaves four 32-value lanes.  The low
        # quantiles use ql[l], the high quantiles use ql[l + 32], while all
        # four 2-bit groups for a lane come from qh[l].
        qbyte = ql[half * 64 + (l if lane in (0, 2) else l + 32)]
        high = qh[half * 32 + l]
        nibble = (qbyte & 15) if lane in (0, 1) else (qbyte >> 4)
        high_shift = lane * 2
        quant = (nibble | (((high >> high_shift) & 3) << 4)) - 32
        scale_index = half * 8 + (l // 16) + lane * 2
        output[local] = d * scales[scale_index] * quant
    return output


_NVFP4_LUT = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)

def _ue4m3_to_float(bits):
    s = (bits >> 7) & 1
    e = (bits >> 3) & 0xF
    m = bits & 7
    if e == 0:
        val = (m / 8.0) * (2.0 ** -6)
    elif e == 0xF:
        val = float('inf') if m == 0 else float('nan')
    else:
        val = (2.0 ** (e - 7)) * (1.0 + m / 8.0)
    return -val if s else val


def _decode_nvfp4(raw, start, count):
    output = np.empty(count, dtype=np.float32)
    for local in range(count):
        absolute = start + local
        sub, within = divmod(absolute, 16)
        base = sub * 9
        scale = _ue4m3_to_float(raw[base])
        nibble_pair = raw[base + 1 + (within >> 1)]
        val = (nibble_pair >> 4) if (within & 1) else (nibble_pair & 0x0F)
        output[local] = scale * _NVFP4_LUT[val]
    return output
