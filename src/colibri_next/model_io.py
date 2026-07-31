from __future__ import annotations

import math
from pathlib import Path

from .bf16 import BF16Tensor
from .matrix import load_matrix
from .tensor_container import ColiTensorFile


class QwenModelIO:
    """Token embedding, final one-centered RMSNorm, and BF16 vocabulary head."""

    def __init__(self, path: Path | str, *, rows_per_chunk: int = 4096):
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")
        self.path = Path(path)
        self.rows_per_chunk = rows_per_chunk
        container = ColiTensorFile(self.path)
        metadata = container.metadata
        self.hidden_size = int(metadata["hidden_size"])
        self.vocab_size = int(metadata["vocab_size"])
        self.rms_norm_eps = float(metadata["rms_norm_eps"])
        self.tie_word_embeddings = bool(metadata["tie_word_embeddings"])
        self.embedding = load_matrix(container, "embed_tokens.weight")
        self.final_norm = BF16Tensor.from_container(container, "norm.weight")
        self.lm_head = (
            self.embedding
            if self.tie_word_embeddings
            else load_matrix(container, "lm_head.weight")
        )
        self._norm_weights = self.final_norm.values()
        self._validate()

    @classmethod
    def from_model_directory(
        cls, root: Path | str, *, rows_per_chunk: int = 4096
    ) -> "QwenModelIO":
        return cls(Path(root) / "model_io.coli", rows_per_chunk=rows_per_chunk)

    def embed(self, token_id: int) -> list[float]:
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(
                f"token id {token_id} outside vocabulary of {self.vocab_size}"
            )
        return self.embedding.row(token_id)

    def normalize(self, hidden: list[float]) -> list[float]:
        if len(hidden) != self.hidden_size:
            raise ValueError(
                f"expected hidden width {self.hidden_size}, got {len(hidden)}"
            )
        inverse_rms = 1.0 / math.sqrt(
            sum(value * value for value in hidden) / len(hidden)
            + self.rms_norm_eps
        )
        return [
            value * inverse_rms * (1.0 + weight)
            for value, weight in zip(hidden, self._norm_weights)
        ]

    def logits(self, hidden: list[float]) -> list[float]:
        return self.lm_head.matvec_chunked(
            self.normalize(hidden), rows_per_chunk=self.rows_per_chunk
        )

    def _validate(self) -> None:
        expected_matrix = (self.vocab_size, self.hidden_size)
        if self.embedding.shape != expected_matrix:
            raise ValueError(f"invalid embedding shape: {self.embedding.shape}")
        if self.final_norm.shape != (self.hidden_size,):
            raise ValueError(f"invalid final norm shape: {self.final_norm.shape}")
        if self.lm_head.shape != expected_matrix:
            raise ValueError(f"invalid LM head shape: {self.lm_head.shape}")
