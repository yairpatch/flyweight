from __future__ import annotations

from pathlib import Path
from typing import Any

from .attention import AttentionKVCache
from .bf16 import BF16Tensor
from .tensor_container import ColiTensorFile


class _MtpAttention:
    """Attribute bag matching what CudaAccelerator.full_attention expects."""


class QwenMtpHead:
    """GPU runtime for the Qwen multi-token-prediction draft head.

    The head is one standard full-attention + MoE decoder layer fed by
    ``fc(concat(norm(embed(token)), norm(decoder_hidden)))``, sharing the main
    model's embedding table and LM head. Everything runs at BF16: quantizing
    the draft path is known to collapse draft acceptance.
    """

    def __init__(self, container: ColiTensorFile):
        metadata = container.metadata
        self.hidden_size = int(metadata["hidden_size"])
        self.top_k = int(metadata["top_k"])
        self.rms_norm_eps = float(metadata["rms_norm_eps"])
        self.num_experts = int(metadata["num_experts"])
        self.intermediate_size = int(metadata["moe_intermediate_size"])
        self.num_key_value_heads = int(metadata["num_key_value_heads"])
        self.head_dim = int(metadata["head_dim"])

        def tensor(name: str) -> BF16Tensor:
            info = container.tensors[name]
            return BF16Tensor(shape=info.shape, data=container.read(name))

        def weights(name: str) -> list[float]:
            return tensor(name).values()

        self.fc = tensor("fc.weight")
        self._pre_fc_norm_embedding = weights("pre_fc_norm_embedding.weight")
        self._pre_fc_norm_hidden = weights("pre_fc_norm_hidden.weight")
        self._final_norm = weights("norm.weight")

        attention = _MtpAttention()
        attention.q_projection = tensor("layers.0.self_attn.q_proj.weight")
        attention.k_projection = tensor("layers.0.self_attn.k_proj.weight")
        attention.v_projection = tensor("layers.0.self_attn.v_proj.weight")
        attention.o_projection = tensor("layers.0.self_attn.o_proj.weight")
        attention._input_norm_weights = weights(
            "layers.0.input_layernorm.weight"
        )
        attention._q_norm_weights = weights("layers.0.self_attn.q_norm.weight")
        attention._k_norm_weights = weights("layers.0.self_attn.k_norm.weight")
        attention.rms_norm_eps = self.rms_norm_eps
        attention.num_attention_heads = int(metadata["num_attention_heads"])
        attention.num_key_value_heads = self.num_key_value_heads
        attention.head_dim = self.head_dim
        attention.rotary_dim = int(metadata["rotary_dim"])
        attention.rope_theta = float(metadata["rope_theta"])
        self.attention = attention

        self.router = tensor("layers.0.mlp.gate.weight")
        self.shared_gate = tensor("layers.0.mlp.shared_expert_gate.weight")
        self._post_attention_norm = weights(
            "layers.0.post_attention_layernorm.weight"
        )
        gate_info = container.tensors["layers.0.mlp.experts.gate_up_proj"]
        down_info = container.tensors["layers.0.mlp.experts.down_proj"]
        self._experts_gate_up = container.read(
            "layers.0.mlp.experts.gate_up_proj"
        )
        self._experts_down = container.read("layers.0.mlp.experts.down_proj")
        self._gate_up_shape = tuple(gate_info.shape[1:])
        self._down_shape = tuple(down_info.shape[1:])
        shared_gate_proj = tensor("layers.0.mlp.shared_expert.gate_proj.weight")
        shared_up_proj = tensor("layers.0.mlp.shared_expert.up_proj.weight")
        self.shared_gate_up = BF16Tensor(
            shape=(
                shared_gate_proj.shape[0] + shared_up_proj.shape[0],
                shared_gate_proj.shape[1],
            ),
            data=shared_gate_proj.data + shared_up_proj.data,
        )
        self.shared_down = tensor("layers.0.mlp.shared_expert.down_proj.weight")
        self._expert_cache: dict[int, tuple[BF16Tensor, BF16Tensor]] = {}

    @classmethod
    def from_model_directory(cls, root: Path | str) -> "QwenMtpHead":
        path = Path(root) / "mtp.coli"
        if not path.is_file():
            raise FileNotFoundError(f"missing MTP head container: {path}")
        return cls(ColiTensorFile(path))

    def new_cache(self) -> AttentionKVCache:
        return AttentionKVCache(self.num_key_value_heads, self.head_dim)

    def _expert(self, expert_id: int) -> tuple[BF16Tensor, BF16Tensor]:
        cached = self._expert_cache.get(expert_id)
        if cached is not None:
            return cached
        gate_rows, gate_columns = self._gate_up_shape
        down_rows, down_columns = self._down_shape
        gate_bytes = gate_rows * gate_columns * 2
        down_bytes = down_rows * down_columns * 2
        cached = (
            BF16Tensor(
                shape=(gate_rows, gate_columns),
                data=self._experts_gate_up[
                    expert_id * gate_bytes : (expert_id + 1) * gate_bytes
                ],
            ),
            BF16Tensor(
                shape=(down_rows, down_columns),
                data=self._experts_down[
                    expert_id * down_bytes : (expert_id + 1) * down_bytes
                ],
            ),
        )
        self._expert_cache[expert_id] = cached
        return cached

    def fuse_inputs(
        self, accelerator: Any, token_id: int, hidden: Any, model_io: Any
    ) -> Any:
        """fc(concat(norm(embed(token)), norm(decoder_hidden))) on device."""
        cp = accelerator.cp
        embedding = accelerator.device_vector(model_io.embed(token_id))
        embedding = accelerator.rms_norm_device(
            embedding, self._pre_fc_norm_embedding, self.rms_norm_eps
        )
        hidden = accelerator.rms_norm_device(
            hidden, self._pre_fc_norm_hidden, self.rms_norm_eps
        )
        fused = cp.concatenate((embedding, hidden))
        return accelerator._matrix_matvec_device(self.fc, fused)

    def forward(
        self,
        accelerator: Any,
        token_id: int,
        hidden: Any,
        position: int,
        cache: AttentionKVCache,
        model_io: Any,
        *,
        return_logits: bool = True,
    ) -> tuple[Any, Any]:
        """One MTP step; returns (logits, layer_hidden) as device arrays.

        ``hidden`` is the pre-final-norm decoder output for the position the
        embedded token follows. Appends this position to ``cache``; rolling
        back speculative steps is a cache-length truncation.
        """
        fused = self.fuse_inputs(accelerator, token_id, hidden, model_io)
        mixed, _ = accelerator.full_attention(
            self.attention,
            fused,
            position,
            cache,
            residual=True,
            return_attention_weights=False,
            return_device=True,
        )
        output = self._moe_residual(accelerator, mixed)
        if not return_logits:
            return None, output
        normalized = accelerator.rms_norm_device(
            output, self._final_norm, self.rms_norm_eps
        )
        logits = accelerator.matrix_matvec_device(model_io.lm_head, normalized)
        return logits, output

    def advance(
        self,
        accelerator: Any,
        token_id: int,
        hidden: Any,
        position: int,
        cache: AttentionKVCache,
        model_io: Any,
    ) -> None:
        """Append one true (token, hidden) pair to the draft KV cache."""
        fused = self.fuse_inputs(accelerator, token_id, hidden, model_io)
        accelerator.full_attention(
            self.attention,
            fused,
            position,
            cache,
            residual=True,
            return_attention_weights=False,
            return_device=True,
        )

    def prefill_cache(
        self,
        accelerator: Any,
        token_ids: list[int],
        hiddens: Any,
        cache: AttentionKVCache,
        model_io: Any,
        *,
        start_position: int = 0,
    ) -> None:
        """Batch-populate the draft KV cache for a prompt.

        ``token_ids[j]`` must be the token following the position whose
        pre-final-norm decoder hidden is ``hiddens[j]``.
        """
        if not token_ids:
            return
        cp = accelerator.cp
        if len(token_ids) != int(hiddens.shape[0]):
            raise ValueError("token and hidden counts must match")
        embeddings = accelerator.device_vector(
            [model_io.embed(token_id) for token_id in token_ids]
        )
        embeddings = accelerator.rms_norm_rows_device(
            embeddings, self._pre_fc_norm_embedding, self.rms_norm_eps
        )
        hiddens = accelerator.rms_norm_rows_device(
            hiddens, self._pre_fc_norm_hidden, self.rms_norm_eps
        )
        fused = accelerator.matrix_matmul_device(
            self.fc, cp.concatenate((embeddings, hiddens), axis=1)
        )
        accelerator._full_attention_sequence_batched(
            self.attention, fused, start_position, cache
        )

    def _moe_residual(self, accelerator: Any, hidden: Any) -> Any:
        import numpy as np

        cp = accelerator.cp
        normalized = accelerator.rms_norm_device(
            hidden, self._post_attention_norm, self.rms_norm_eps
        )
        router_logits = accelerator._matrix_matvec_device(
            self.router, normalized
        ).get()
        probabilities = np.exp(router_logits - router_logits.max())
        probabilities /= probabilities.sum()
        selected = np.argpartition(probabilities, -self.top_k)[-self.top_k :]
        weights = probabilities[selected]
        weights /= weights.sum()
        shared_logit = accelerator._matrix_matvec_device(
            self.shared_gate, normalized
        )
        shared_weight = cp.float32(1.0) / (
            cp.float32(1.0) + cp.exp(-shared_logit[0])
        )
        output = cp.zeros(self.hidden_size, dtype=cp.float32)
        for expert_id, weight in zip(selected, weights):
            gate_up, down = self._expert(int(expert_id))
            output += cp.float32(float(weight)) * self._swiglu(
                accelerator, gate_up, down, normalized
            )
        output += shared_weight * self._swiglu(
            accelerator, self.shared_gate_up, self.shared_down, normalized
        )
        return hidden + output

    def _swiglu(
        self, accelerator: Any, gate_up: BF16Tensor, down: BF16Tensor, vector: Any
    ) -> Any:
        cp = accelerator.cp
        projected = accelerator._matrix_matvec_device(
            gate_up, vector, protected=False
        )
        intermediate = gate_up.shape[0] // 2
        gates = projected[:intermediate]
        activated = (
            gates
            / (cp.float32(1.0) + cp.exp(-gates))
            * projected[intermediate:]
        )
        return accelerator._matrix_matvec_device(
            down, activated, protected=False
        )
