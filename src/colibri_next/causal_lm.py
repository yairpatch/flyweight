from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .decoder import DecoderResult, DecoderState, QwenDecoderStack
from .model_io import QwenModelIO


@dataclass(frozen=True, slots=True)
class CausalLMResult:
    token_id: int
    hidden: list[float]
    logits: list[float]
    decoder_result: DecoderResult

    @property
    def greedy_token(self) -> int:
        return max(range(len(self.logits)), key=self.logits.__getitem__)


@dataclass(frozen=True, slots=True)
class DeviceCausalLMResult:
    token_id: int
    hidden: Any
    logits: Any


class CausalLMState:
    def __init__(self, decoder_state: DecoderState):
        self.decoder_state = decoder_state
        # Speculative-decoding companions: the MTP draft head's KV cache and
        # the pre-final-norm decoder hidden of the last forwarded position.
        # Living on the state, they travel through the prefix cache for free.
        self.mtp_cache: Any = None
        self.mtp_hidden: Any = None

    @property
    def tokens(self) -> int:
        return self.decoder_state.tokens

    def clear(self) -> None:
        self.decoder_state.clear()
        self.mtp_cache = None
        self.mtp_hidden = None


class QwenForCausalLM:
    """Token-ID to logits runtime over the converted Qwen decoder stack."""

    def __init__(self, decoder: QwenDecoderStack, model_io: QwenModelIO):
        self.decoder = decoder
        self.model_io = model_io
        self.hidden_size = decoder.hidden_size
        self.vocab_size = model_io.vocab_size
        self.root: Path | None = None
        self.mtp: Any = None
        self._mtp_attempted = False
        if model_io.hidden_size != self.hidden_size:
            raise ValueError("model I/O and decoder hidden sizes do not match")

    @classmethod
    def from_model_directory(
        cls, root: Path | str, *, rows_per_chunk: int = 4096
    ) -> "QwenForCausalLM":
        model = cls(
            QwenDecoderStack.from_model_directory(root),
            QwenModelIO.from_model_directory(root, rows_per_chunk=rows_per_chunk),
        )
        model.root = Path(root)
        return model

    def load_mtp(self) -> Any:
        """Load the MTP draft head if the model directory ships one."""
        if not self._mtp_attempted:
            self._mtp_attempted = True
            if self.root is not None and (self.root / "mtp.coli").is_file():
                from .mtp import QwenMtpHead

                self.mtp = QwenMtpHead.from_model_directory(self.root)
        return self.mtp

    @property
    def cpu_moe_layers(self) -> int:
        return self.decoder.cpu_moe_layers

    def configure_moe_placement(self, cpu_layers: int) -> None:
        self.decoder.configure_moe_placement(cpu_layers)

    @property
    def device_decode_available(self) -> bool:
        from .cuda import active_cuda

        return active_cuda() is not None

    @property
    def estimated_expert_storage_bytes(self) -> int:
        return self.decoder.estimated_expert_storage_bytes

    def preload_experts(
        self,
        *,
        mode: str = "all",
        available_ram_bytes: int | None = None,
        reserve_bytes: int = 8 * 1024**3,
        progress: Callable[[int, int], None] | None = None,
    ) -> int:
        if mode not in {"none", "auto", "all"}:
            raise ValueError(f"unsupported expert preload mode: {mode}")
        if mode == "none":
            return 0
        required = self.estimated_expert_storage_bytes
        if mode == "auto" and (
            available_ram_bytes is None
            or available_ram_bytes < required + reserve_bytes
        ):
            return 0
        return self.decoder.preload_experts(progress)

    def new_state(self) -> CausalLMState:
        return CausalLMState(self.decoder.new_state())

    def forward_token(
        self, token_id: int, state: CausalLMState
    ) -> CausalLMResult:
        embedding = self.model_io.embed(token_id)
        decoder_result = self.decoder.forward_token(
            embedding, state.decoder_state
        )
        hidden = self.model_io.normalize(decoder_result.output)
        logits = self.model_io.lm_head.matvec_chunked(
            hidden, rows_per_chunk=self.model_io.rows_per_chunk
        )
        return CausalLMResult(
            token_id=token_id,
            hidden=hidden,
            logits=logits,
            decoder_result=decoder_result,
        )

    def forward_token_device(
        self, token_id: int, state: CausalLMState
    ) -> DeviceCausalLMResult:
        accelerator = self._device_accelerator()
        accelerator.device_resident_decode_tokens += 1
        embedding = accelerator.device_vector(self.model_io.embed(token_id))
        hidden = self.decoder.forward_token_device(
            embedding, state.decoder_state, accelerator
        )
        hidden = accelerator.rms_norm_device(
            hidden,
            self.model_io._norm_weights,
            self.model_io.rms_norm_eps,
        )
        logits = accelerator.matrix_matvec_device(
            self.model_io.lm_head, hidden
        )
        return DeviceCausalLMResult(token_id, hidden, logits)

    def prefill_device(
        self, token_ids: list[int], state: CausalLMState
    ) -> DeviceCausalLMResult:
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        accelerator = self._device_accelerator()
        embeddings = accelerator.device_vector(
            [self.model_io.embed(token_id) for token_id in token_ids]
        )
        hidden = self.decoder.prefill_device(
            embeddings, state.decoder_state, accelerator
        )[-1]
        hidden = accelerator.rms_norm_device(
            hidden,
            self.model_io._norm_weights,
            self.model_io.rms_norm_eps,
        )
        logits = accelerator.matrix_matvec_device(
            self.model_io.lm_head, hidden
        )
        accelerator.device_resident_decode_tokens += len(token_ids)
        return DeviceCausalLMResult(token_ids[-1], hidden, logits)

    def prefill_device_with_hidden(
        self, token_ids: list[int], state: CausalLMState
    ) -> tuple[DeviceCausalLMResult, Any]:
        """Prefill returning last-position logits plus all decoder hiddens.

        Unlike :meth:`verify_device` the LM head runs only on the final
        position, so long prompts do not materialize a (tokens, vocab)
        logits matrix.
        """
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        accelerator = self._device_accelerator()
        embeddings = accelerator.device_vector(
            [self.model_io.embed(token_id) for token_id in token_ids]
        )
        decoder_hidden = self.decoder.prefill_device(
            embeddings, state.decoder_state, accelerator
        )
        hidden = accelerator.rms_norm_device(
            decoder_hidden[-1],
            self.model_io._norm_weights,
            self.model_io.rms_norm_eps,
        )
        logits = accelerator.matrix_matvec_device(
            self.model_io.lm_head, hidden
        )
        accelerator.device_resident_decode_tokens += len(token_ids)
        return (
            DeviceCausalLMResult(token_ids[-1], hidden, logits),
            decoder_hidden,
        )

    def verify_device(
        self, token_ids: list[int], state: CausalLMState
    ) -> tuple[Any, Any]:
        """Forward a token batch returning logits at every position.

        Returns ``(logits, decoder_hidden)`` as device arrays of shape
        ``(tokens, vocab)`` and ``(tokens, hidden)``; ``decoder_hidden`` is
        the pre-final-norm decoder output that the MTP draft head consumes.
        """
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        accelerator = self._device_accelerator()
        embeddings = accelerator.device_vector(
            [self.model_io.embed(token_id) for token_id in token_ids]
        )
        decoder_hidden = self.decoder.prefill_device(
            embeddings, state.decoder_state, accelerator
        )
        normalized = accelerator.rms_norm_rows_device(
            decoder_hidden,
            self.model_io._norm_weights,
            self.model_io.rms_norm_eps,
        )
        logits = accelerator.matrix_matmul_device(
            self.model_io.lm_head, normalized
        )
        accelerator.device_resident_decode_tokens += len(token_ids)
        return logits, decoder_hidden

    def _device_accelerator(self) -> Any:
        from .cuda import active_cuda

        accelerator = active_cuda()
        if accelerator is None:
            raise RuntimeError("device decode requires CUDA")
        return accelerator

    def prefill(
        self, token_ids: list[int], state: CausalLMState
    ) -> CausalLMResult:
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        for token_id in token_ids[:-1]:
            embedding = self.model_io.embed(token_id)
            self.decoder.forward_token(embedding, state.decoder_state)
        return self.forward_token(token_ids[-1], state)

    def forward_ids(
        self, token_ids: list[int], state: CausalLMState
    ) -> list[CausalLMResult]:
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        return [self.forward_token(token_id, state) for token_id in token_ids]
