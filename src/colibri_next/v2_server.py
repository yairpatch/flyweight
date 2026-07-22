from __future__ import annotations

import codecs
import threading
from pathlib import Path
from queue import SimpleQueue
from typing import Callable, Iterator, Mapping, Sequence

from .generation import GenerationResult, GenerationStep
from .sampling import SamplingConfig
from .server import APIError, InferenceService
from .v2 import (
    TASK_EVENT_DONE,
    TASK_EVENT_ERROR,
    TASK_EVENT_TOKEN,
    V2Error,
    V2Model,
    V2QwenRuntime,
)


class _NativeEngine:
    """Drives the native cooperative engine from one thread and fans per-task
    events out to per-request queues, so concurrent HTTP requests interleave
    (a short request no longer waits behind a long prefill) while all CUDA work
    stays on a single thread."""

    def __init__(self, runtime: V2QwenRuntime):
        self.runtime = runtime
        self._lock = threading.Lock()
        self._queues: dict[int, SimpleQueue] = {}
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def submit(
        self, prompt_ids: list[int], max_new_tokens: int, stop_tokens: tuple[int, ...]
    ) -> tuple[int, SimpleQueue]:
        queue: SimpleQueue = SimpleQueue()
        with self._lock:
            task_id = self.runtime.task_submit(prompt_ids, max_new_tokens, stop_tokens)
            self._queues[task_id] = queue
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="colibri-v2-engine", daemon=True
                )
                self._thread.start()
        self._wake.set()
        return task_id, queue

    def cancel(self, task_id: int) -> None:
        try:
            self.runtime.task_cancel(task_id)
        except V2Error:
            pass  # runtime may already be closed; the task queue is dropped below

    def _run(self) -> None:
        while True:
            with self._lock:
                idle = not self._queues
            if idle:
                self._wake.clear()
                self._wake.wait()
                continue
            try:
                events = self.runtime.engine_step()
            except Exception as error:
                # Engine-level failure: fail every waiting request, not just one.
                with self._lock:
                    queues, self._queues = self._queues, {}
                for queue in queues.values():
                    queue.put(("error", str(error)))
                continue
            if not events:
                # Tasks exist but none progressed (e.g. waiting for a busy
                # slot); avoid a hot spin.
                threading.Event().wait(0.002)
                continue
            for task_id, token, kind in events:
                with self._lock:
                    queue = self._queues.get(task_id)
                if queue is None:
                    continue
                if kind == TASK_EVENT_TOKEN:
                    queue.put(("token", token))
                elif kind == TASK_EVENT_DONE:
                    queue.put(("done", None))
                    with self._lock:
                        self._queues.pop(task_id, None)
                elif kind == TASK_EVENT_ERROR:
                    queue.put(("error", "native v2 engine task failed"))
                    with self._lock:
                        self._queues.pop(task_id, None)

    def forget(self, task_id: int) -> None:
        with self._lock:
            self._queues.pop(task_id, None)


class NativeV2Tokenizer:
    """Chat formatting and tokenizer facade backed directly by GGUF metadata."""

    def __init__(self, model: V2Model):
        self.model = model
        eos: list[int] = []
        for text in ("<|im_end|>", "<|endoftext|>"):
            try:
                eos.append(model.token_id(text))
            except V2Error:
                pass
        self.eos_token_ids = tuple(dict.fromkeys(eos))

    def encode(self, text: str) -> list[int]:
        return self.model.tokenize(text)

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        values = (
            [token for token in token_ids if token not in self.eos_token_ids]
            if skip_special_tokens
            else token_ids
        )
        return self.model.decode_tokens(values)

    def token_bytes(self, token_id: int) -> bytes:
        """Raw bytes of one token, for incremental (streaming) UTF-8 decoding."""
        return self.model.decode_token_bytes([token_id])

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
            if role not in ("system", "developer", "user", "assistant", "tool"):
                raise ValueError(f"unsupported chat role: {role}")
            if not content:
                raise ValueError("chat message content must not be empty")
            if role == "assistant" and not content.startswith("<think>"):
                thinking_prefix = (
                    "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
                )
                content = thinking_prefix + content
            sections.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        sections.append("<|im_start|>assistant\n")
        sections.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
        return "".join(sections)

    def encode_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        enable_thinking: bool = False,
    ) -> list[int]:
        return self.encode(
            self.format_messages(messages, enable_thinking=enable_thinking)
        )


class NativeV2Generator:
    """Greedy single-stream generator over the one-token native C ABI."""

    def __init__(
        self, model: V2Model, runtime: V2QwenRuntime, tokenizer: NativeV2Tokenizer
    ):
        self.model = model
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.engine = _NativeEngine(runtime)
        self._chat_lock = threading.Lock()
        self._chat_messages: tuple[tuple[str, str], ...] | None = None
        self._chat_prompt_ids: tuple[int, ...] = ()
        self._chat_generated_ids: tuple[int, ...] = ()
        self._chat_text = ""
        self._chat_thinking = False

    def prefix_cache_stats(self) -> dict[str, int]:
        info = self.runtime.info
        return {
            "entries": 1 if info["position"] else 0,
            "capacity": 1,
            "hits": int(info["prefix_cache_hits"]),
            "misses": int(info["prefix_cache_misses"]),
            "evictions": 0,
            "reused_tokens": int(info["prefix_cache_reused_tokens"]),
        }

    @staticmethod
    def _require_greedy(sampling: SamplingConfig | None) -> None:
        # The native v2 decode path is greedy (argmax). Client sampling params
        # (temperature/top_p/top_k) are accepted and ignored rather than
        # rejected, so agentic clients that always send temperature (Claude
        # Code defaults to 1.0) still work instead of getting a 400.
        return

    def generate_messages(
        self, messages: Sequence[Mapping[str, str]], **options: object
    ) -> GenerationResult:
        final: GenerationStep | None = None
        for step in self.stream_messages(messages, **options):
            if step.finished:
                final = step
        if final is None:
            raise RuntimeError("native v2 generation ended without a final result")
        return GenerationResult(
            final.prompt_ids,
            final.generated_ids,
            final.text,
            final.stopped_on_eos,
            final.state_tokens,
        )

    def stream_messages(
        self, messages: Sequence[Mapping[str, str]], **options: object
    ) -> Iterator[GenerationStep]:
        thinking = bool(options.get("enable_thinking", False))
        normalized = tuple(
            (message["role"], message["content"].strip()) for message in messages
        )
        with self._chat_lock:
            prompt_ids = self._continued_chat_prompt(normalized, thinking)
        final: GenerationStep | None = None
        for step in self._stream(prompt_ids, **options):
            if step.finished:
                final = step
            yield step
        if final is not None:
            # Concurrent conversations race for this continuation state; last
            # writer wins and the losers simply fall back to snapshot reuse.
            with self._chat_lock:
                self._chat_messages = normalized
                self._chat_prompt_ids = tuple(prompt_ids)
                self._chat_generated_ids = final.generated_ids
                self._chat_text = final.text
                self._chat_thinking = thinking

    def _continued_chat_prompt(
        self, messages: tuple[tuple[str, str], ...], thinking: bool
    ) -> list[int]:
        previous = self._chat_messages
        if (
            previous is not None
            and thinking == self._chat_thinking
            and len(messages) > len(previous) + 1
            and messages[: len(previous)] == previous
            and messages[len(previous)][0] == "assistant"
            and messages[len(previous)][1] == self._chat_text.strip()
        ):
            remaining = messages[len(previous) + 1 :]
            if remaining:
                generated = list(self._chat_generated_ids)
                ended = bool(
                    generated and generated[-1] in self.tokenizer.eos_token_ids
                )
                separator = "\n" if ended else "<|im_end|>\n"
                suffix_messages = [
                    {"role": role, "content": content} for role, content in remaining
                ]
                return (
                    list(self._chat_prompt_ids)
                    + generated
                    + self.tokenizer.encode(separator)
                    + self.tokenizer.encode_messages(
                        suffix_messages, enable_thinking=thinking
                    )
                )
        return self.tokenizer.encode_messages(
            [{"role": role, "content": content} for role, content in messages],
            enable_thinking=thinking,
        )

    def generate_text(self, prompt: str, **options: object) -> GenerationResult:
        self._chat_messages = None
        return self._generate(self.tokenizer.encode(prompt), **options)

    def stream_text(self, prompt: str, **options: object) -> Iterator[GenerationStep]:
        self._chat_messages = None
        return self._stream(self.tokenizer.encode(prompt), **options)

    def _generate(self, prompt_ids: list[int], **options: object) -> GenerationResult:
        final: GenerationStep | None = None
        for step in self._stream(prompt_ids, **options):
            if step.finished:
                final = step
        if final is None:
            raise RuntimeError("native v2 generation ended without a final result")
        return GenerationResult(
            final.prompt_ids,
            final.generated_ids,
            final.text,
            final.stopped_on_eos,
            final.state_tokens,
        )

    def _stream(
        self, prompt_ids: list[int], **options: object
    ) -> Iterator[GenerationStep]:
        max_new_tokens = int(options.get("max_new_tokens", 8))
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        sampling = options.get("sampling")
        self._require_greedy(sampling if isinstance(sampling, SamplingConfig) else None)
        progress = options.get("progress")
        progress_callback = progress if callable(progress) else None
        if not prompt_ids:
            raise ValueError("formatted prompt produced no token IDs")
        generated: list[int] = []
        previous_text = ""
        stopped = False
        # Streamed tokens are decoded through an INCREMENTAL UTF-8 decoder: a
        # BPE token often carries only part of a multi-byte character, and
        # decoding token-by-token turned those halves into U+FFFD - corrupted
        # text that then flowed into tool calls and files written on disk.
        # The incremental decoder buffers partial sequences across tokens and
        # only emits complete characters.
        utf8 = codecs.getincrementaldecoder("utf-8")("replace")
        # The cooperative engine interleaves this task with any other in-flight
        # requests (each on its own KV slot); EOS is detected natively via the
        # stop-token list so no token is decoded past it.
        task_id, queue = self.engine.submit(
            prompt_ids, max_new_tokens, self.tokenizer.eos_token_ids
        )
        try:
            progress_reported = False
            while True:
                kind, value = queue.get()
                if kind == "done":
                    break
                if kind == "error":
                    raise RuntimeError(str(value))
                token = int(value)
                if progress_callback is not None and not progress_reported:
                    progress_callback(len(prompt_ids), len(prompt_ids))
                    progress_reported = True
                generated.append(token)
                stopped = token in self.tokenizer.eos_token_ids
                try:
                    delta = (
                        "" if stopped
                        else utf8.decode(self.tokenizer.token_bytes(token))
                    )
                except Exception:
                    # A single undecodable token must never abort the whole
                    # stream; keep the previously decoded text and continue.
                    delta = ""
                previous_text += delta
                yield GenerationStep(
                    token,
                    delta,
                    tuple(prompt_ids),
                    tuple(generated),
                    previous_text,
                    stopped,
                    False,
                    len(prompt_ids) + len(generated),
                )
                if stopped:
                    continue
            # Flush any dangling partial UTF-8 sequence (a truncated character
            # at the very end of generation becomes a single visible U+FFFD).
            tail = utf8.decode(b"", True)
            previous_text += tail
            yield GenerationStep(
                None,
                tail,
                tuple(prompt_ids),
                tuple(generated),
                previous_text,
                stopped,
                True,
                len(prompt_ids) + len(generated),
            )
        except GeneratorExit:
            self.engine.cancel(task_id)
            raise
        finally:
            self.engine.forget(task_id)


class NativeV2InferenceService(InferenceService):
    def __init__(
        self,
        model_path: Path | str,
        *,
        model_name: str | None = None,
        device: int = 0,
        context_window: int = 32768,
        max_new_tokens: int = 4096,
        gpu_cache_mib: int = 0,  # 0 = auto-fit the GPU expert cache to free VRAM
        moe_device: str = "hybrid",
        mtp_drafts: int = 0,
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
        prefill_checkpoint_interval: int = 256,
        prefill_checkpoint_slots: int = 4,
        parallel_sequences: int = 1,
        prompt_cache_mib: int = 0,
        api_key: str | None = None,
        cors_origin: str = "*",
        strict_model: bool = False,
    ):
        self.v2_model = V2Model(model_path)
        self.v2_runtime: V2QwenRuntime | None = None
        try:
            self.v2_runtime = self.v2_model.native_qwen_runtime(
                device=device,
                context_limit=context_window,
                gpu_cache_bytes=gpu_cache_mib * 1024**2,
                moe_device=moe_device,
                mtp_drafts=mtp_drafts,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                prefill_checkpoint_interval=prefill_checkpoint_interval,
                prefill_checkpoint_slots=prefill_checkpoint_slots,
                parallel_sequences=parallel_sequences,
                prompt_cache_mib=prompt_cache_mib,
            )
            self.v2_runtime.prepare()
        except Exception:
            if self.v2_runtime is not None:
                self.v2_runtime.close()
            self.v2_model.close()
            raise
        assert self.v2_runtime is not None
        tokenizer = NativeV2Tokenizer(self.v2_model)
        super().__init__(
            model_name or Path(model_path).stem,
            NativeV2Generator(self.v2_model, self.v2_runtime, tokenizer),
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            api_key=api_key,
            cors_origin=cors_origin,
            strict_model=strict_model,
        )
        self.moe_device = moe_device
        self.mtp_drafts = mtp_drafts
        self.gpu_cache_mib = gpu_cache_mib
        # The native cooperative engine interleaves concurrent requests itself
        # (per-slot KV, single CUDA thread), so the HTTP layer must not
        # serialize them.
        self._serialize_generation = False

    def close(self) -> None:
        if self.v2_runtime is not None:
            self.v2_runtime.close()
            self.v2_runtime = None
        self.v2_model.close()

    def health(self) -> dict[str, object]:
        if self.v2_runtime is None:
            raise RuntimeError("native v2 service is closed")
        value = super().health()
        value["execution"] = {
            "backend": "native-v2-cpp-cuda",
            **self.v2_runtime.info,
            "moe_device": self.moe_device,
            "mtp_drafts": self.mtp_drafts,
            "gpu_cache_mib": self.gpu_cache_mib,
        }
        return value
