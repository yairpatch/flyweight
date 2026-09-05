"""Serving DeepSeek-V4 through the engine the rest of the server expects.

The cooperative engine the other architectures share schedules slots of one KV
pair per layer, which this architecture does not have: every block carries a
sliding-window latent ring, a compressed-block cache and the compressor's own
partial-block state, and a 4:1 block carries the indexer's cache besides. None
of that is expressible as a KV pair, so the scheduler is written here against
``Deepseek4Runtime`` instead of bent around the Qwen one.

What is *not* rewritten is everything above the scheduler. Chat templates, tool
calls, continuation reuse, incremental UTF-8 decoding and the SSE transport all
live in :class:`ChatGenerator`, which talks to an engine only through
``submit``/``cancel``/``forget`` and the four event kinds below. This module
supplies an engine of that shape and nothing more.

Two properties of this model shaped the scheduler:

- A sequence's state is tens of megabytes against 104 GiB of weights, so a slot
  is cheap and several can be held at once. What is expensive is the weights,
  which every slot reads the same way, so slots share the model and interleave
  rather than run in parallel.
- The runtime advances a sequence and cannot rewind it, so a slot can only be
  reused by a prompt that *extends* what it already holds. That is exactly the
  shape a chat turn has, which is where the reuse pays.
"""

from __future__ import annotations

import inspect
import threading
from collections import OrderedDict
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Mapping

import numpy as np

from .deepseek4 import Deepseek4Runtime
from .dspark import DsparkSession
from .sampling import SamplingConfig
from .server import InferenceService
from .v2 import V2Error, V2Model
from .v2_server import (
    ChatGenerator,
    NativeV2Tokenizer,
    _generation_config_for_model,
)


# What a direct caller gets when it passes no sampling config: greedy decode,
# so a continuation served from a warm slot can be compared token-for-token
# with a cold run. The HTTP layer always passes an explicit config built from
# the request and the server defaults, so this never shapes a served answer.
_GREEDY = SamplingConfig(temperature=0.0, top_k=1, top_p=1.0, min_p=0.0)


def sample_token(
    logits: np.ndarray, sampling: SamplingConfig, generator: np.random.Generator
) -> int:
    """Pick one token from a logit vector.

    Greedy at temperature zero -- which is both the default and what every
    parity check against the reference uses -- and otherwise temperature, then
    top-k, nucleus, then min-p, in llama.cpp's order. Sampling happens here
    rather than natively because the native sampler is wired into the Qwen
    engine's task state; moving it is worth doing only once this path needs
    the speed.
    """
    if sampling.temperature <= 0:
        return int(np.argmax(logits))
    scaled = logits.astype(np.float32) / sampling.temperature
    order = np.argsort(scaled)[::-1]
    if sampling.top_k > 0:
        order = order[: sampling.top_k]
    ranked = scaled[order]
    probabilities = np.exp(ranked - ranked.max())
    probabilities /= probabilities.sum()
    if sampling.top_p < 1.0:
        cumulative = np.cumsum(probabilities)
        # Keep the first entry whose cumulative mass reaches top_p, so the
        # nucleus is never empty even when one token holds more than top_p.
        keep = int(np.searchsorted(cumulative, sampling.top_p) + 1)
        order, probabilities = order[:keep], probabilities[:keep]
        probabilities = probabilities / probabilities.sum()
    min_p = float(getattr(sampling, "min_p", 0.0) or 0.0)
    if min_p > 0.0 and probabilities.size > 1:
        # llama.cpp's min-p after top_p, relative to the best candidate.
        keep = max(1, int(np.count_nonzero(probabilities >= min_p * probabilities[0])))
        order, probabilities = order[:keep], probabilities[:keep]
        probabilities = probabilities / probabilities.sum()
    return int(generator.choice(order, p=probabilities))


def _target_runtime(slot: "Deepseek4Runtime | DsparkSession") -> Deepseek4Runtime:
    """The target-model runtime behind a slot, drafted or not."""
    return slot.target if isinstance(slot, DsparkSession) else slot


class _Slot:
    """One sequence's native state, plus the tokens it has consumed."""

    __slots__ = ("runtime", "tokens", "serial", "dirty")

    def __init__(self, runtime: "Deepseek4Runtime | DsparkSession"):
        self.runtime = runtime
        self.tokens: list[int] = []
        self.serial = 0
        self.dirty = False

    def extends(self, prompt: list[int]) -> bool:
        """Whether `prompt` continues what this slot already holds.

        Strictly: an exact match cannot be reused, because the last token would
        have to be forwarded again to produce logits and the runtime has no way
        to take a position back.
        """
        return (
            not self.dirty
            and 0 < len(self.tokens) < len(prompt)
            and self.tokens == prompt[: len(self.tokens)]
        )

    def reset(self) -> None:
        self.runtime.reset()
        self.tokens = []
        self.dirty = False


class _Task:
    __slots__ = (
        "task_id", "prompt", "max_new_tokens", "stop", "sampling", "queue",
        "slot", "fed", "generated", "pending", "rng", "cancelled", "logits",
    )

    def __init__(
        self,
        task_id: int,
        prompt: list[int],
        max_new_tokens: int,
        stop: tuple[int, ...],
        sampling: SamplingConfig,
        queue: Queue,
    ):
        self.task_id = task_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.stop = set(stop)
        self.sampling = sampling
        self.queue = queue
        self.slot: _Slot | None = None
        self.fed = 0            # prompt tokens whose forward has completed
        self.generated = 0
        self.pending = 0        # the token whose forward produces the next logits
        self.rng = np.random.default_rng(sampling.seed)
        self.cancelled = False
        self.logits = None


class Deepseek4Engine:
    """Round-robin scheduler over a pool of DeepSeek-V4 sequence slots.

    One thread drives every task, because the slots share the weights and the
    expert reads are the cost; running two forwards at once would double the
    page pressure to no benefit. Interleaving is at chunk granularity instead,
    so a short request behind a long prefill waits for a chunk rather than for
    the whole prompt.
    """

    _MAX_ACTIVE_TASKS = 64
    _MAX_BUFFERED_EVENTS = 256
    _PREFILL_CHUNK = 32
    _SHUTDOWN_TIMEOUT_SECONDS = 60.0

    def __init__(
        self, model: V2Model, context_limit: int, slots: int = 1,
        device: int | None = None, dspark_model: V2Model | None = None,
        dspark_drafts: int = 0,
    ):
        if slots <= 0:
            raise ValueError("slots must be positive")
        self.context_limit = int(context_limit)
        self.device = device
        if dspark_drafts < 0:
            raise ValueError("DSpark draft count must be non-negative")
        self.dspark_drafts = int(dspark_drafts)
        self._slots: list[Deepseek4Runtime | DsparkSession] = [
            DsparkSession(model, dspark_model, context_limit) if dspark_model is not None
            else Deepseek4Runtime(model, context_limit)
            for _ in range(slots)
        ]
        if device is not None:
            owner = _target_runtime(self._slots[0])
            owner.use_gpu(device)
            for runtime in self._slots[1:]:
                _target_runtime(runtime).share_gpu(owner)
        self._pool = [_Slot(runtime) for runtime in self._slots]
        self._lock = threading.Lock()
        self._tasks: OrderedDict[int, _Task] = OrderedDict()
        self._next_task_id = 1
        self._serial = 0
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._closing = False
        self.hits = 0
        self.misses = 0
        self.reused_tokens = 0
        self.last_prompt_tokens = 0
        self.last_reused_tokens = 0

    # -- submission -------------------------------------------------------

    def submit(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        stop_tokens: tuple[int, ...],
        sampling: SamplingConfig | None = None,
        tools: list[dict[str, object]] | None = None,
        response_format: dict[str, object] | None = None,
        forbid_tool_calls: bool = False,
    ) -> tuple[int, Queue[tuple[str, object]]]:
        # As `tools` below: this runtime's sampler has none of the constraint
        # machinery, the markup ban included.
        del response_format, forbid_tool_calls
        # Accepted and unused: constrained tool-call decoding lives in the Qwen
        # sampler, and this runtime has its own. A DeepSeek-V4 tool call is
        # still recovered by the tolerant parser in the server, which is where
        # every call was handled before the constraint existed. Taking the
        # argument keeps one calling convention across the engines rather than
        # making the shared streaming path ask which one it holds.
        del tools
        prompt = [int(token) for token in prompt_ids]
        if not prompt:
            raise ValueError("prompt must not be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        needed = len(prompt) + max_new_tokens
        if needed > self.context_limit:
            raise ValueError(
                f"{len(prompt)} prompt tokens plus {max_new_tokens} new ones "
                f"exceed the {self.context_limit}-token context limit"
            )
        queue: Queue[tuple[str, object]] = Queue(maxsize=self._MAX_BUFFERED_EVENTS)
        with self._lock:
            if self._closing:
                raise RuntimeError("deepseek4 engine is shutting down")
            if len(self._tasks) >= self._MAX_ACTIVE_TASKS:
                raise RuntimeError("deepseek4 engine queue is full; retry later")
            task_id = self._next_task_id
            self._next_task_id += 1
            self._tasks[task_id] = _Task(
                task_id, prompt, max_new_tokens, stop_tokens,
                sampling or _GREEDY, queue,
            )
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="flyweight-deepseek4-engine", daemon=True
                )
                self._thread.start()
        self._wake.set()
        return task_id, queue

    def cancel(self, task_id: int) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task.cancelled = True
        self._wake.set()

    def forget(self, task_id: int) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task.cancelled = True

    def close(self) -> None:
        with self._lock:
            already = self._closing
            self._closing = True
            tasks = list(self._tasks.values())
            thread = self._thread
        for task in tasks:
            task.cancelled = True
            self._replace_with_error(task.queue, "deepseek4 engine is shutting down")
        self._wake.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._SHUTDOWN_TIMEOUT_SECONDS)
            if thread.is_alive():
                raise RuntimeError(
                    "deepseek4 engine did not stop in time; runtime teardown "
                    "was abandoned rather than free state a thread still reads"
                )
        if not already:
            # Borrowers hold the owner's device pointers; destroy them first so
            # that ownership order remains explicit even though teardown no
            # longer forwards work.
            for runtime in reversed(self._slots):
                runtime.close()

    # -- scheduling -------------------------------------------------------

    @staticmethod
    def _replace_with_error(queue: Queue, message: str) -> None:
        """Guarantee a terminal event stays readable without unbounded memory."""
        while True:
            try:
                queue.get_nowait()
            except Empty:
                break
        try:
            queue.put_nowait(("error", message))
        except Full:  # pragma: no cover - the queue was just drained
            pass

    def _emit(self, task: _Task, event: tuple[str, object]) -> bool:
        try:
            task.queue.put_nowait(event)
        except Full:
            task.cancelled = True
            self._replace_with_error(
                task.queue, "deepseek4 client output queue overflow; request cancelled"
            )
            return False
        return True

    def _acquire(self, task: _Task) -> bool:
        """Give the task a slot, reusing one whose sequence its prompt extends."""
        with self._lock:
            free = [slot for slot in self._pool if slot not in self._held()]
        if not free:
            return False
        reusable = [slot for slot in free if slot.extends(task.prompt)]
        if reusable:
            slot = max(reusable, key=lambda candidate: len(candidate.tokens))
            task.fed = len(slot.tokens)
            self.hits += 1
            self.reused_tokens += task.fed
            self.last_reused_tokens = task.fed
        else:
            slot = min(free, key=lambda candidate: candidate.serial)
            slot.reset()
            task.fed = 0
            self.misses += 1
            self.last_reused_tokens = 0
        self.last_prompt_tokens = len(task.prompt)
        self._serial += 1
        slot.serial = self._serial
        task.slot = slot
        return True

    def _held(self) -> set[int]:
        return {id(task.slot) for task in self._tasks.values() if task.slot is not None}

    @property
    def slot_count(self) -> int:
        return len(self._pool)

    @property
    def warm_slots(self) -> int:
        """Slots holding a sequence a later prompt could still extend."""
        return sum(1 for slot in self._pool if slot.tokens and not slot.dirty)

    def _release(self, task: _Task) -> None:
        if task.slot is not None and task.slot.dirty:
            try:
                task.slot.reset()
            except V2Error:
                pass
        task.slot = None

    def _finish(self, task: _Task, event: tuple[str, object]) -> None:
        self._release(task)
        with self._lock:
            self._tasks.pop(task.task_id, None)
        if not task.cancelled:
            self._emit(task, event)

    def _step(self, task: _Task) -> bool:
        """Advance one task by a prefill chunk or a single token.

        Returns whether anything happened, so an idle pass can back off instead
        of spinning on tasks that are waiting for a slot.
        """
        if task.slot is None:
            if not self._acquire(task):
                return False
            # The first progress report is read as the count served from cache,
            # so it has to be the reused prefix rather than the first chunk.
            self._emit(task, ("prefill", task.fed))
        slot = task.slot
        assert slot is not None
        dspark = slot.runtime if isinstance(slot.runtime, DsparkSession) else None
        # Prefill: everything but the last prompt token, whose logits are the
        # first thing generation needs.
        if task.fed < len(task.prompt) - 1:
            end = min(task.fed + self._PREFILL_CHUNK, len(task.prompt) - 1)
            chunk = task.prompt[task.fed:end]
            if isinstance(slot.runtime, DsparkSession):
                for token in chunk:
                    slot.runtime.forward_target(token, logits=False)
            else:
                slot.runtime.prefill(chunk)
            slot.tokens.extend(chunk)
            task.fed = end
            self._emit(task, ("prefill", task.fed))
            return True
        current = (
            task.prompt[-1] if task.fed == len(task.prompt) - 1 else task.pending
        )
        if isinstance(slot.runtime, DsparkSession):
            if task.logits is None:
                task.logits = slot.runtime.forward_target(current)
                slot.tokens.append(current)
            logits = task.logits
        else:
            logits = slot.runtime.forward(current)
            slot.tokens.append(current)
        if task.fed == len(task.prompt) - 1:
            task.fed = len(task.prompt)
            self._emit(task, ("prefill", task.fed))
        if dspark is not None and task.sampling.temperature <= 0:
            remaining = task.max_new_tokens - task.generated
            candidates = min(remaining, self.dspark_drafts) if self.dspark_drafts else remaining
            accepted, correction, next_logits = dspark.speculative_round(
                slot.tokens[-1], logits, p_min=0.1, max_candidates=candidates
            )
            emitted = list(accepted)
            slot.tokens.extend(emitted)
            task.logits = next_logits
            if correction is not None and len(emitted) < remaining:
                emitted.append(correction)
                task.pending = correction
                task.logits = None
            elif not emitted:
                correction = int(np.argmax(logits))
                emitted.append(correction)
                task.pending = correction
                task.logits = None
            for index, token in enumerate(emitted):
                task.generated += 1
                if not self._emit(task, ("token", token)):
                    return True
                if token in task.stop or task.generated >= task.max_new_tokens:
                    if index + 1 < len(emitted):
                        slot.dirty = True
                    self._finish(task, ("done", None))
                    return True
            return True
        token = sample_token(logits, task.sampling, task.rng)
        task.logits = None
        task.pending = token
        task.generated += 1
        if not self._emit(task, ("token", token)):
            return True
        if token in task.stop or task.generated >= task.max_new_tokens:
            self._finish(task, ("done", None))
        return True

    def _run(self) -> None:
        # The weights were uploaded from whichever thread built the engine, and
        # a CUDA context is current per thread.
        try:
            if self.device is not None:
                for slot in self._pool:
                    runtime = slot.runtime.target if isinstance(slot.runtime, DsparkSession) else slot.runtime
                    runtime.attach_gpu()
        except Exception as error:
            # Initialization happens on the worker because CUDA contexts are
            # current per thread.  It must still follow the same terminal-event
            # contract as a failed forward; otherwise every caller blocks on an
            # event queue whose only producer has exited.
            with self._lock:
                tasks = list(self._tasks.values())
                self._tasks.clear()
            for task in tasks:
                task.cancelled = True
                self._replace_with_error(
                    task.queue, f"deepseek4 engine initialization failed: {error}"
                )
            return
        while True:
            with self._lock:
                if self._closing:
                    return
                tasks = list(self._tasks.values())
            if not tasks:
                self._wake.clear()
                with self._lock:
                    idle = not self._tasks and not self._closing
                if idle:
                    self._wake.wait()
                continue
            progressed = False
            for task in tasks:
                if task.cancelled:
                    self._release(task)
                    with self._lock:
                        self._tasks.pop(task.task_id, None)
                    progressed = True
                    continue
                try:
                    progressed |= self._step(task)
                except Exception as error:  # a failed forward leaves state unusable
                    if task.slot is not None:
                        task.slot.dirty = True
                    self._release(task)
                    with self._lock:
                        self._tasks.pop(task.task_id, None)
                    self._replace_with_error(
                        task.queue, f"deepseek4 engine failure: {error}"
                    )
                    progressed = True
            if not progressed:
                # Every task is waiting for a slot held by a task that is
                # itself finished but not yet reaped; yield rather than spin.
                threading.Event().wait(0.002)


class Deepseek4Generator(ChatGenerator):
    """The server's generator surface over the DeepSeek-V4 scheduler."""

    def __init__(
        self,
        model: V2Model,
        tokenizer: NativeV2Tokenizer,
        *,
        context_limit: int,
        slots: int = 1,
        device: int | None = None,
        dspark_model: V2Model | None = None,
        dspark_drafts: int = 0,
    ):
        super().__init__(
            model, Deepseek4Engine(
                model, context_limit, slots, device, dspark_model, dspark_drafts
            ), tokenizer
        )

    def prefix_cache_stats(self) -> dict[str, int]:
        engine = self.engine
        return {
            "entries": engine.warm_slots,
            "capacity": engine.slot_count,
            "ram_entries": 0,
            "ram_bytes": 0,
            "hits": engine.hits,
            "misses": engine.misses,
            "evictions": 0,
            "reused_tokens": engine.reused_tokens,
            "last_prompt_tokens": engine.last_prompt_tokens,
            "last_reused_tokens": engine.last_reused_tokens,
            "last_lcp_live": engine.last_reused_tokens,
            "last_lcp_snapshot": 0,
        }


def _reject_unsupported(options: Mapping[str, object]) -> None:
    """Fail on a runtime knob this path cannot honour, unless it is untouched.

    The defaults come from the Qwen service's own signature rather than a copy
    of them, so a knob added there cannot silently start being accepted and
    ignored here.
    """
    from .v2_server import NativeV2InferenceService

    known = inspect.signature(NativeV2InferenceService.__init__).parameters
    requested = []
    for key, value in options.items():
        parameter = known.get(key)
        if parameter is None:
            raise TypeError(f"unexpected keyword argument {key!r}")
        if value != parameter.default:
            requested.append(key)
    if requested:
        raise ValueError(
            "the DeepSeek-V4 runtime does not support "
            + ", ".join(sorted(requested))
            + " yet; it uses its dedicated CPU/hybrid runtime with half-precision caches"
        )


class NativeDeepseek4InferenceService(InferenceService):
    """The OpenAI-compatible service, backed by the DeepSeek-V4 scheduler.

    With `device` set it runs hybrid: the dense half of the model -- 6.3 GiB,
    read in full on every token -- is resident on the GPU, and the routed
    experts stay on the CPU because they are 90 GiB. Immutable device weights
    and the serialized activation workspace are shared by all sequence slots.
    The other runtime knobs the
    Qwen service accepts (GPU cache, expert placement, KV quantization, MTP
    drafts) still have no counterpart here, and are rejected rather than
    accepted and ignored.
    """

    def __init__(
        self,
        model_path: Path | str,
        *,
        model_name: str | None = None,
        context_window: int = 8192,
        max_new_tokens: int = 1024,
        parallel_sequences: int = 1,
        device: int | None = None,
        dspark_model_path: Path | str | None = None,
        dspark_drafts: int = 0,
        api_key: str | None = None,
        cors_origin: str = "*",
        strict_model: bool = False,
        max_concurrent_requests: int = 64,
        request_timeout_seconds: float = 30.0,
        sse_keepalive_seconds: float = 10.0,
        default_thinking_budget: int | None = None,
        **unsupported: object,
    ):
        _reject_unsupported(unsupported)
        generation_defaults, generation_defaults_source = _generation_config_for_model(
            model_path
        )
        self.v2_model = V2Model(model_path)
        self.dspark_model = V2Model(dspark_model_path) if dspark_model_path else None
        try:
            architecture = str(self.v2_model.config["architecture"])
            if architecture != "deepseek4":
                raise ValueError(
                    f"{architecture} is not a DeepSeek-V4 checkpoint"
                )
            tokenizer = NativeV2Tokenizer(self.v2_model)
            generator = Deepseek4Generator(
                self.v2_model,
                tokenizer,
                context_limit=context_window,
                slots=parallel_sequences,
                device=device,
                dspark_model=self.dspark_model,
                dspark_drafts=dspark_drafts,
            )
        except Exception:
            if self.dspark_model is not None:
                self.dspark_model.close()
            self.v2_model.close()
            raise
        super().__init__(
            model_name or Path(model_path).stem,
            generator,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            api_key=api_key,
            cors_origin=cors_origin,
            strict_model=strict_model,
            max_concurrent_requests=max_concurrent_requests,
            request_timeout_seconds=request_timeout_seconds,
            sse_keepalive_seconds=sse_keepalive_seconds,
            default_thinking_budget=default_thinking_budget,
            generation_defaults=generation_defaults,
        )
        self.generation_defaults_source = generation_defaults_source
        # The base class holds it as the Generator protocol; this service also
        # needs its engine and close(), which the protocol does not carry.
        self.deepseek_generator = generator
        self.parallel_sequences = parallel_sequences
        self.device = device
        self.dspark_drafts = dspark_drafts
        # The scheduler interleaves requests itself, so the HTTP layer must not
        # serialize them on top of it.
        self._serialize_generation = False

    def close(self) -> None:
        self.deepseek_generator.close()
        if self.dspark_model is not None:
            self.dspark_model.close()
        self.v2_model.close()

    def health(self) -> dict[str, object]:
        value = super().health()
        runtime = self.deepseek_generator.engine._pool[0].runtime
        target = runtime.target if isinstance(runtime, DsparkSession) else runtime
        value["execution"] = {
            "backend": (
                "native-v2-deepseek4-cpu" if self.device is None
                else "native-v2-deepseek4-hybrid"
            ),
            "slots": self.parallel_sequences,
            "device": self.device,
            **target.info,
        }
        if isinstance(runtime, DsparkSession):
            value["execution"]["dspark"] = dict(runtime.stats)
        return value

    def properties(self) -> dict[str, object]:
        value = super().properties()
        tokenizer = self.generator.tokenizer
        source = getattr(tokenizer, "chat_template_source", "fallback")
        value["chat_template"] = (
            "tokenizer.chat_template" if source == "gguf" else "deepseek4-fallback"
        )
        value["chat_template_source"] = source
        value["generation_defaults_source"] = self.generation_defaults_source
        return value
