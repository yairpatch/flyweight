from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from .causal_lm import QwenForCausalLM
from .cuda import active_cuda
from .sampling import LogitsSampler, SamplingConfig
from .tokenizer import HuggingFaceTokenizer


@dataclass(frozen=True, slots=True)
class GenerationResult:
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    text: str
    stopped_on_eos: bool
    state_tokens: int


@dataclass(frozen=True, slots=True)
class GenerationStep:
    token_id: int | None
    text_delta: str
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    text: str
    stopped_on_eos: bool
    finished: bool
    state_tokens: int


@dataclass(slots=True)
class _PrefixCacheEntry:
    state: Any
    logits: Any


class TextGenerator:
    def __init__(
        self,
        model: QwenForCausalLM,
        tokenizer: HuggingFaceTokenizer,
        *,
        prefix_cache_entries: int | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        capacity = (
            int(os.environ.get("COLIBRI_PREFIX_CACHE_ENTRIES", "4"))
            if prefix_cache_entries is None
            else prefix_cache_entries
        )
        if capacity < 0:
            raise ValueError("prefix_cache_entries must be non-negative")
        self.prefix_cache_entries = capacity
        self._prefix_cache: OrderedDict[
            tuple[int, ...], _PrefixCacheEntry
        ] = OrderedDict()
        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0
        self.prefix_cache_evictions = 0
        self.prefix_cache_reused_tokens = 0

    def prefix_cache_stats(self) -> dict[str, int]:
        return {
            "entries": len(self._prefix_cache),
            "capacity": self.prefix_cache_entries,
            "hits": self.prefix_cache_hits,
            "misses": self.prefix_cache_misses,
            "evictions": self.prefix_cache_evictions,
            "reused_tokens": self.prefix_cache_reused_tokens,
        }

    def clear_prefix_cache(self) -> None:
        self._prefix_cache.clear()

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 8,
        sampling: SamplingConfig | None = None,
        system: str | None = None,
        enable_thinking: bool = False,
    ) -> GenerationResult:
        prompt_ids = self.tokenizer.encode_chat(
            prompt,
            system=system,
            enable_thinking=enable_thinking,
        )
        return self._generate_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
        )

    def generate_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int = 8,
        sampling: SamplingConfig | None = None,
        enable_thinking: bool = False,
    ) -> GenerationResult:
        prompt_ids = self.tokenizer.encode_messages(
            messages, enable_thinking=enable_thinking
        )
        return self._generate_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
        )

    def stream_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int = 8,
        sampling: SamplingConfig | None = None,
        enable_thinking: bool = False,
    ) -> Iterator[GenerationStep]:
        prompt_ids = self.tokenizer.encode_messages(
            messages, enable_thinking=enable_thinking
        )
        return self._stream_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
        )

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 8,
        sampling: SamplingConfig | None = None,
    ) -> GenerationResult:
        return self._generate_ids(
            self.tokenizer.encode(prompt),
            max_new_tokens=max_new_tokens,
            sampling=sampling,
        )

    def stream_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 8,
        sampling: SamplingConfig | None = None,
    ) -> Iterator[GenerationStep]:
        return self._stream_ids(
            self.tokenizer.encode(prompt),
            max_new_tokens=max_new_tokens,
            sampling=sampling,
        )
    def _generate_ids(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        sampling: SamplingConfig | None,
    ) -> GenerationResult:
        final_step = None
        for step in self._stream_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
        ):
            if step.finished:
                final_step = step
        if final_step is None:
            raise RuntimeError("generation ended without a final result")
        return GenerationResult(
            prompt_ids=final_step.prompt_ids,
            generated_ids=final_step.generated_ids,
            text=final_step.text,
            stopped_on_eos=final_step.stopped_on_eos,
            state_tokens=final_step.state_tokens,
        )

    def _stream_ids(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        sampling: SamplingConfig | None,
    ) -> Iterator[GenerationStep]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not prompt_ids:
            raise ValueError("formatted prompt produced no token IDs")
        prompt_tuple = tuple(prompt_ids)
        cached = self._take_prefix_state(prompt_tuple)
        if cached is None:
            state = self.model.new_state()
            cached_tokens = 0
        else:
            cached_tokens, entry = cached
            state = entry.state
        sampler = LogitsSampler(sampling or SamplingConfig())
        accelerator = (
            active_cuda() if self.model.device_decode_available else None
        )
        remaining_ids = prompt_ids[cached_tokens:]
        if remaining_ids:
            if accelerator is not None:
                prompt_result = self.model.prefill_device(remaining_ids, state)
            else:
                prompt_result = self.model.prefill(remaining_ids, state)
            logits = prompt_result.logits
        else:
            assert cached is not None
            logits = entry.logits
        processed_ids = list(prompt_ids)
        generated: list[int] = []
        stopped = False
        previous_text = ""
        for index in range(max_new_tokens):
            token_id = (
                sampler.sample_device(logits, accelerator)
                if accelerator is not None
                else sampler.sample(logits)
            )
            generated.append(token_id)
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            text_delta = (
                text[len(previous_text) :]
                if text.startswith(previous_text)
                else text
            )
            previous_text = text
            stopped = token_id in self.tokenizer.eos_token_ids
            yield GenerationStep(
                token_id=token_id,
                text_delta=text_delta,
                prompt_ids=prompt_tuple,
                generated_ids=tuple(generated),
                text=text,
                stopped_on_eos=stopped,
                finished=False,
                state_tokens=state.tokens,
            )
            if stopped:
                break
            if index + 1 < max_new_tokens:
                if accelerator is not None:
                    logits = self.model.forward_token_device(
                        token_id, state
                    ).logits
                else:
                    logits = self.model.forward_token(token_id, state).logits
                processed_ids.append(token_id)
        self._store_prefix_state(tuple(processed_ids), state, logits)
        yield GenerationStep(
            token_id=None,
            text_delta="",
            prompt_ids=prompt_tuple,
            generated_ids=tuple(generated),
            text=previous_text,
            stopped_on_eos=stopped,
            finished=True,
            state_tokens=state.tokens,
        )

    def _take_prefix_state(
        self, prompt_ids: tuple[int, ...]
    ) -> tuple[int, _PrefixCacheEntry] | None:
        best_key = max(
            (
                key
                for key in self._prefix_cache
                if len(key) <= len(prompt_ids)
                and prompt_ids[: len(key)] == key
            ),
            key=len,
            default=None,
        )
        if best_key is None:
            self.prefix_cache_misses += 1
            return None
        entry = self._prefix_cache.pop(best_key)
        if entry.state.tokens != len(best_key):
            raise RuntimeError("cached decoder state token count is inconsistent")
        self.prefix_cache_hits += 1
        self.prefix_cache_reused_tokens += len(best_key)
        return len(best_key), entry

    def _store_prefix_state(
        self, token_ids: tuple[int, ...], state: Any, logits: Any
    ) -> None:
        if self.prefix_cache_entries == 0 or not token_ids:
            return
        self._prefix_cache[token_ids] = _PrefixCacheEntry(state, logits)
        self._prefix_cache.move_to_end(token_ids)
        while len(self._prefix_cache) > self.prefix_cache_entries:
            self._prefix_cache.popitem(last=False)
            self.prefix_cache_evictions += 1
