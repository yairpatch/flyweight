from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

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


class TextGenerator:
    def __init__(
        self, model: QwenForCausalLM, tokenizer: HuggingFaceTokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

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
        state = self.model.new_state()
        sampler = LogitsSampler(sampling or SamplingConfig())
        accelerator = (
            active_cuda() if self.model.device_decode_available else None
        )
        if accelerator is not None:
            prompt_result = self.model.prefill_device(prompt_ids, state)
        else:
            prompt_result = self.model.prefill(prompt_ids, state)
        logits = prompt_result.logits
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