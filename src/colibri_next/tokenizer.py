from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


try:
    from tokenizers import Tokenizer as BackendTokenizer
except ImportError:  # pragma: no cover - exercised on minimal deployments
    BackendTokenizer = None


class HuggingFaceTokenizer:
    """Tokenizer JSON runtime with a text-only ChatML prompt formatter."""

    def __init__(self, root: Path | str, *, backend: Any | None = None):
        self.root = Path(root)
        config_path = self.root / "tokenizer_config.json"
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        generation_path = self.root / "generation_config.json"
        self.generation_config = (
            json.loads(generation_path.read_text(encoding="utf-8"))
            if generation_path.exists()
            else {}
        )
        if backend is not None:
            self.backend = backend
        else:
            if BackendTokenizer is None:
                raise RuntimeError(
                    "text generation requires the 'tokenizers' package; "
                    "install with: python -m pip install -e '.[generation]'"
                )
            self.backend = BackendTokenizer.from_file(
                str(self.root / "tokenizer.json")
            )
        self.eos_token_ids = self._eos_token_ids()

    @classmethod
    def from_model_directory(
        cls, root: Path | str, *, backend: Any | None = None
    ) -> "HuggingFaceTokenizer":
        return cls(Path(root) / "tokenizer", backend=backend)

    def encode(self, text: str) -> list[int]:
        return list(self.backend.encode(text, add_special_tokens=False).ids)

    def decode(
        self, token_ids: list[int], *, skip_special_tokens: bool = True
    ) -> str:
        return str(
            self.backend.decode(
                token_ids, skip_special_tokens=skip_special_tokens
            )
        )

    def format_chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        enable_thinking: bool = False,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.format_messages(messages, enable_thinking=enable_thinking)

    def format_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        enable_thinking: bool = False,
    ) -> str:
        if not messages:
            raise ValueError("messages must not be empty")
        sections: list[str] = []
        for message in messages:
            role = message["role"]
            content = message["content"].strip()
            if role not in ("system", "user", "assistant"):
                raise ValueError(f"unsupported chat role: {role}")
            if not content:
                raise ValueError("chat message content must not be empty")
            if role == "assistant" and not content.startswith("<think>"):
                thinking_prefix = (
                    "<think>\n"
                    if enable_thinking
                    else "<think>\n\n</think>\n\n"
                )
                content = thinking_prefix + content
            sections.append(
                f"<|im_start|>{role}\n{content}<|im_end|>\n"
            )
        sections.append("<|im_start|>assistant\n")
        sections.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
        return "".join(sections)

    def encode_chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        enable_thinking: bool = False,
    ) -> list[int]:
        return self.encode(
            self.format_chat(
                prompt, system=system, enable_thinking=enable_thinking
            )
        )

    def encode_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        enable_thinking: bool = False,
    ) -> list[int]:
        return self.encode(
            self.format_messages(messages, enable_thinking=enable_thinking)
        )
    def _eos_token_ids(self) -> tuple[int, ...]:
        configured = self.generation_config.get("eos_token_id")
        if configured is None:
            configured = self.config.get("eos_token_id")
        if configured is None:
            token = self.config.get("eos_token")
            if token is None:
                return ()
            token_id = self.backend.token_to_id(token)
            return () if token_id is None else (int(token_id),)
        if isinstance(configured, list):
            return tuple(int(value) for value in configured)
        return (int(configured),)
