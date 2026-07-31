from __future__ import annotations

import inspect
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

from .causal_lm import QwenForCausalLM
from .cuda import active_cuda
from .sampling import LogitsSampler, SamplingConfig
from .tokenizer import HuggingFaceTokenizer


def _call_prefill(
    method: Callable[..., Any],
    token_ids: list[int],
    state: Any,
    progress: Callable[[int, int], None] | None,
) -> Any:
    """Call model prefill while preserving compatibility with older adapters."""
    if progress is None:
        return method(token_ids, state)
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        # C-extension or proxy methods may not expose a signature. The current
        # model protocol supports progress, so prefer the new call shape.
        return method(token_ids, state, progress=progress)
    if any(
        parameter.name == "progress"
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    ):
        return method(token_ids, state, progress=progress)
    return method(token_ids, state)


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
        progress: Callable[[int, int], None] | None = None,
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
            progress=progress,
        )

    def generate_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int = 8,
        sampling: SamplingConfig | None = None,
        enable_thinking: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> GenerationResult:
        prompt_ids = self.tokenizer.encode_messages(
            messages, enable_thinking=enable_thinking
        )
        return self._generate_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            progress=progress,
        )

    def stream_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int = 8,
        sampling: SamplingConfig | None = None,
        enable_thinking: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[GenerationStep]:
        prompt_ids = self.tokenizer.encode_messages(
            messages, enable_thinking=enable_thinking
        )
        return self._stream_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            progress=progress,
        )

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 8,
        sampling: SamplingConfig | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> GenerationResult:
        return self._generate_ids(
            self.tokenizer.encode(prompt),
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            progress=progress,
        )

    def stream_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 8,
        sampling: SamplingConfig | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[GenerationStep]:
        return self._stream_ids(
            self.tokenizer.encode(prompt),
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            progress=progress,
        )
    def _generate_ids(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        sampling: SamplingConfig | None,
        progress: Callable[[int, int], None] | None = None,
    ) -> GenerationResult:
        final_step = None
        for step in self._stream_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            progress=progress,
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
        progress: Callable[[int, int], None] | None = None,
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
        # Speculative decoding is opt-in for now: rounds are verified
        # bit-identical to sequential decoding, but on this codebase the
        # per-forward GPU dispatch cost still eats the batching win, so it
        # roughly breaks even until that comes down.
        use_mtp = (
            accelerator is not None
            and max_new_tokens > 1
            and os.environ.get("COLIBRI_MTP", "0") == "1"
            and self.model.load_mtp() is not None
        )
        remaining_ids = prompt_ids[cached_tokens:]
        if progress is not None:
            progress(cached_tokens, len(prompt_ids))

        def report_prefill(done: int, _total: int) -> None:
            if progress is not None:
                progress(cached_tokens + done, len(prompt_ids))

        if remaining_ids:
            if use_mtp:
                prompt_result, decoder_hidden = (
                    _call_prefill(
                        self.model.prefill_device_with_hidden,
                        remaining_ids,
                        state,
                        report_prefill,
                    )
                )
                mtp = self.model.mtp
                if cached_tokens == 0 or state.mtp_cache is None:
                    state.mtp_cache = mtp.new_cache()
                    state.mtp_cache.tokens = 0
                    pair_tokens = list(remaining_ids[1:])
                    pair_hiddens = decoder_hidden[:-1]
                    start_position = 0
                elif state.mtp_hidden is not None:
                    cp = accelerator.cp
                    pair_tokens = list(remaining_ids)
                    pair_hiddens = cp.concatenate(
                        (
                            state.mtp_hidden.reshape(1, -1),
                            decoder_hidden[:-1],
                        )
                    )
                    start_position = cached_tokens - 1
                else:
                    use_mtp = False
                if use_mtp and pair_tokens:
                    mtp.prefill_cache(
                        accelerator,
                        pair_tokens,
                        pair_hiddens,
                        state.mtp_cache,
                        self.model.model_io,
                        start_position=start_position,
                    )
                if use_mtp:
                    state.mtp_hidden = decoder_hidden[-1]
            elif accelerator is not None:
                prompt_result = _call_prefill(
                    self.model.prefill_device,
                    remaining_ids,
                    state,
                    report_prefill,
                )
            else:
                prompt_result = _call_prefill(
                    self.model.prefill,
                    remaining_ids,
                    state,
                    report_prefill,
                )
            logits = prompt_result.logits
        else:
            assert cached is not None
            logits = entry.logits
        if use_mtp and (state.mtp_cache is None or state.mtp_hidden is None):
            use_mtp = False
        processed_ids = list(prompt_ids)
        generated: list[int] = []
        stopped = False
        previous_text = ""
        if use_mtp:
            yield from self._speculative_steps(
                state=state,
                sampler=sampler,
                accelerator=accelerator,
                logits=logits,
                prompt_tuple=prompt_tuple,
                processed_ids=processed_ids,
                max_new_tokens=max_new_tokens,
            )
            return
        if accelerator is not None:
            # A non-speculative pass will not maintain the draft cache; drop
            # it so a later speculative resume cannot use stale entries.
            state.mtp_cache = None
            state.mtp_hidden = None
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

    def _speculative_steps(
        self,
        *,
        state: Any,
        sampler: LogitsSampler,
        accelerator: Any,
        logits: Any,
        prompt_tuple: tuple[int, ...],
        processed_ids: list[int],
        max_new_tokens: int,
    ) -> Iterator[GenerationStep]:
        """MTP speculative decode: draft k tokens, verify in one forward.

        Every committed token is sampled from the main model's logits, so the
        output stream is identical to sequential decoding for the same seed;
        drafts only decide how many positions each forward can batch.
        """
        mtp = self.model.mtp
        model_io = self.model.model_io
        cache = state.mtp_cache
        hidden = state.mtp_hidden
        draft_budget = max(1, int(os.environ.get("COLIBRI_MTP_DRAFTS", "2")))
        generated: list[int] = []
        stopped = False
        previous_text = ""

        def emit(token_id: int) -> GenerationStep:
            nonlocal previous_text, stopped
            generated.append(token_id)
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            text_delta = (
                text[len(previous_text) :]
                if text.startswith(previous_text)
                else text
            )
            previous_text = text
            stopped = token_id in self.tokenizer.eos_token_ids
            return GenerationStep(
                token_id=token_id,
                text_delta=text_delta,
                prompt_ids=prompt_tuple,
                generated_ids=tuple(generated),
                text=text,
                stopped_on_eos=stopped,
                finished=False,
                state_tokens=state.tokens,
            )

        pending = sampler.sample_device(logits, accelerator)
        yield emit(pending)
        while len(generated) < max_new_tokens and not stopped:
            drafts_wanted = min(draft_budget, max_new_tokens - len(generated))
            base_position = state.tokens - 1
            snapshot = state.decoder_state.snapshot()
            true_cache_length = cache.tokens
            drafts: list[int] = []
            draft_token, draft_hidden = pending, hidden
            draft_position = base_position
            for _ in range(drafts_wanted):
                draft_logits, draft_hidden = mtp.forward(
                    accelerator,
                    draft_token,
                    draft_hidden,
                    draft_position,
                    cache,
                    model_io,
                )
                draft_token = int(draft_logits.argmax())
                drafts.append(draft_token)
                draft_position += 1
            batch = [pending, *drafts]
            batch_logits, batch_hidden = self.model.verify_device(batch, state)
            commit: list[int] = []
            for index in range(len(batch)):
                target = sampler.sample_device(batch_logits[index], accelerator)
                commit.append(target)
                if not (index < len(drafts) and target == drafts[index]):
                    break
            valid = len(commit)
            if valid < len(batch):
                state.decoder_state.restore(snapshot)
                self.model.verify_device(batch[:valid], state)
            cache.tokens = true_cache_length
            previous_hidden, position = hidden, base_position
            for index in range(valid):
                mtp.advance(
                    accelerator,
                    batch[index],
                    previous_hidden,
                    position,
                    cache,
                    model_io,
                )
                previous_hidden = batch_hidden[index]
                position += 1
            hidden = batch_hidden[valid - 1]
            logits = batch_logits[valid - 1]
            processed_ids.extend(batch[:valid])
            pending = commit[-1]
            for token_id in commit:
                if len(generated) >= max_new_tokens:
                    break
                yield emit(token_id)
                if stopped:
                    break
        state.mtp_cache = cache
        state.mtp_hidden = hidden
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
