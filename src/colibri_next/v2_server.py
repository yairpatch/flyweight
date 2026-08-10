from __future__ import annotations

import codecs
import datetime
import json
import threading
from collections import OrderedDict
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Iterator, Mapping, Sequence, overload

from jinja2 import StrictUndefined, Template, nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment

from .generation import GenerationResult, GenerationStep
from .sampling import SamplingConfig
from .server import InferenceService, _parse_tool_calls
from .v2 import (
    TASK_EVENT_DONE,
    TASK_EVENT_ERROR,
    TASK_EVENT_PREFILL,
    TASK_EVENT_TOKEN,
    V2Error,
    V2Model,
    V2QwenRuntime,
)


class _GenerationExtension(Extension):
    """Render Hugging Face's generation block without token-span tracking."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.CallBlock(self.call_method("_render"), [], [], body).set_lineno(
            lineno
        )

    def _render(self, caller):
        return caller()


def _generation_config_for_model(
    model_path: Path | str,
) -> tuple[dict[str, int | float], str]:
    """Load Hugging Face generation defaults adjacent to a GGUF, if present.

    The two stem-specific names avoid ambiguity in directories containing more
    than one model. The conventional directory-level name supports GGUF files
    kept inside an exported Hugging Face model directory.
    """
    path = Path(model_path)
    candidates = (
        path.with_name(path.name + ".generation_config.json"),
        path.with_suffix(".generation_config.json"),
        path.parent / "generation_config.json",
    )
    config_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if config_path is None:
        return {}, "engine"
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read generation config {config_path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"generation config {config_path} must contain a JSON object")

    defaults: dict[str, int | float] = {}
    for key in ("temperature", "top_p"):
        value = raw.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            defaults[key] = float(value)
    top_k = raw.get("top_k")
    if isinstance(top_k, int) and not isinstance(top_k, bool):
        defaults["top_k"] = top_k
    max_new_tokens = raw.get("max_new_tokens")
    if isinstance(max_new_tokens, int) and not isinstance(max_new_tokens, bool):
        defaults["max_new_tokens"] = max_new_tokens
    if raw.get("do_sample") is False:
        defaults["temperature"] = 0.0
    return defaults, str(config_path)


class _TokenSnapshot(Sequence[int]):
    """Constant-time, immutable-length view of an append-only token buffer."""

    __slots__ = ("_tokens", "_length")

    def __init__(self, tokens: list[int], length: int):
        self._tokens = tokens
        self._length = length

    def __len__(self) -> int:
        return self._length

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[int]: ...

    def __getitem__(self, index: int | slice) -> int | Sequence[int]:
        if isinstance(index, slice):
            return self._tokens[: self._length][index]
        normalized = index + self._length if index < 0 else index
        if normalized < 0 or normalized >= self._length:
            raise IndexError(index)
        return self._tokens[normalized]


class _NativeEngine:
    """Drives the native cooperative engine from one thread and fans per-task
    events out to per-request queues, so concurrent HTTP requests interleave
    (a short request no longer waits behind a long prefill) while all CUDA work
    stays on a single thread."""

    _MAX_ACTIVE_TASKS = 64
    _MAX_BUFFERED_EVENTS = 256
    _SHUTDOWN_TIMEOUT_SECONDS = 30.0

    def __init__(self, runtime: V2QwenRuntime):
        self.runtime = runtime
        self._lock = threading.Lock()
        self._queues: dict[int, Queue[tuple[str, object]]] = {}
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._closing = False

    def submit(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        stop_tokens: tuple[int, ...],
        sampling: SamplingConfig | None = None,
    ) -> tuple[int, Queue[tuple[str, object]]]:
        task_queue: Queue[tuple[str, object]] = Queue(
            maxsize=self._MAX_BUFFERED_EVENTS
        )
        sampling_config = sampling or SamplingConfig()
        with self._lock:
            if self._closing:
                raise RuntimeError("native v2 engine is shutting down")
            if len(self._queues) >= self._MAX_ACTIVE_TASKS:
                raise RuntimeError("native v2 engine queue is full; retry later")
            task_id = self.runtime.task_submit(
                prompt_ids,
                max_new_tokens,
                stop_tokens,
                temperature=sampling_config.temperature,
                top_k=sampling_config.top_k,
                top_p=sampling_config.top_p,
                seed=sampling_config.seed,
            )
            self._queues[task_id] = task_queue
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="colibri-v2-engine", daemon=True
                )
                self._thread.start()
        self._wake.set()
        return task_id, task_queue

    @staticmethod
    def _replace_with_error(
        task_queue: Queue[tuple[str, object]], message: str
    ) -> None:
        """Bound memory while guaranteeing a terminal event remains readable."""
        while True:
            try:
                task_queue.get_nowait()
            except Empty:
                break
        task_queue.put_nowait(("error", message))

    def cancel(self, task_id: int) -> None:
        try:
            self.runtime.task_cancel(task_id)
        except V2Error:
            pass  # runtime may already be closed; the task queue is dropped below

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._closing:
                    return
                idle = not self._queues
                if idle:
                    # Clear while holding the same lock used by submit/close:
                    # otherwise a wake arriving between the idle check and
                    # clear can be lost, leaving shutdown stuck in wait().
                    self._wake.clear()
            if idle:
                self._wake.wait()
                continue
            try:
                events = self.runtime.engine_step()
            except OSError as error:
                # Windows SEH exceptions (e.g. C++ exceptions escaping
                # ctypes) surface as OSError with a Windows error code.
                # These indicate a bug in the native code (an unhandled
                # C++ exception or memory corruption).  Surface the
                # Windows error text so it is diagnosable.
                with self._lock:
                    queues, self._queues = self._queues, {}
                for task_queue in queues.values():
                    self._replace_with_error(
                        task_queue, f"native engine failure: {error}"
                    )
                continue
            except Exception as error:
                # Engine-level failure: fail every waiting request, not just one.
                with self._lock:
                    queues, self._queues = self._queues, {}
                for task_queue in queues.values():
                    self._replace_with_error(
                        task_queue, f"native engine failure: {error}"
                    )
                continue
            if not events:
                # Tasks exist but none progressed (e.g. waiting for a busy
                # slot); avoid a hot spin.
                threading.Event().wait(0.002)
                continue
            for task_id, token, kind in events:
                with self._lock:
                    task_queue = self._queues.get(task_id)
                if task_queue is None:
                    continue
                try:
                    if kind == TASK_EVENT_TOKEN:
                        task_queue.put_nowait(("token", token))
                    elif kind == TASK_EVENT_PREFILL:
                        task_queue.put_nowait(("prefill", token))
                    elif kind == TASK_EVENT_DONE:
                        task_queue.put_nowait(("done", None))
                        with self._lock:
                            self._queues.pop(task_id, None)
                    elif kind == TASK_EVENT_ERROR:
                        task_queue.put_nowait(
                            ("error", "native v2 engine task failed")
                        )
                        with self._lock:
                            self._queues.pop(task_id, None)
                except Full:
                    self.cancel(task_id)
                    with self._lock:
                        self._queues.pop(task_id, None)
                    self._replace_with_error(
                        task_queue,
                        "native v2 client output queue overflow; request cancelled",
                    )

    def forget(self, task_id: int) -> None:
        with self._lock:
            self._queues.pop(task_id, None)

    def close(self) -> None:
        """Cancel active requests and stop touching the runtime before teardown."""
        with self._lock:
            if self._closing:
                queues = {}
            else:
                self._closing = True
                queues, self._queues = self._queues, {}
            thread = self._thread
        for task_id, task_queue in queues.items():
            try:
                self.runtime.task_cancel(task_id)
            except V2Error:
                pass
            self._replace_with_error(
                task_queue, "native v2 engine is shutting down"
            )
        self._wake.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._SHUTDOWN_TIMEOUT_SECONDS)
            if thread.is_alive():
                raise RuntimeError(
                    "native v2 engine did not stop within 30 seconds; "
                    "runtime teardown was aborted to avoid a use-after-free"
                )


class NativeV2Tokenizer:
    """Chat formatting and tokenizer facade backed directly by GGUF metadata."""

    def __init__(self, model: V2Model):
        self.model = model
        self.architecture = str(model.info["architecture"])
        self.chat_template = getattr(model, "chat_template", None)
        self.chat_template_source = "gguf" if self.chat_template else "fallback"
        self._compiled_chat_template: Template | None = None
        eos: list[int] = []
        # The GGUF's own terminator ids first. eot ends a chat turn where eos
        # ends generation, and a model that closes its turn with a dedicated
        # end-of-turn token (Laguna's </assistant>) never emits eos mid-chat,
        # so leaving eot out lets generation run straight into the next turn.
        try:
            config = dict(getattr(model, "config", None) or {})
        except (V2Error, KeyError, TypeError):
            config = {}
        for field in ("eos_token_id", "eot_token_id"):
            value = config.get(field)
            if value is not None and value != 0xFFFFFFFF:
                eos.append(int(value))
        # Names still cover checkpoints whose metadata omits the ids.
        for text in ("<|im_end|>", "<|endoftext|>", "<turn|>", "<eos>"):
            try:
                eos.append(model.token_id(text))
            except (V2Error, KeyError):
                pass
        self.eos_token_ids = tuple(dict.fromkeys(eos))
        self._token_byte_cache: dict[int, bytes] = {}
        bos = config.get("bos_token_id")
        self._bos_token_id = (
            int(bos) if bos is not None and bos != 0xFFFFFFFF else None
        )
        self._template_tokens = {
            "bos_token": self._configured_token(config, "bos_token_id"),
            "eos_token": self._configured_token(config, "eos_token_id"),
        }
        if self.chat_template:
            environment = ImmutableSandboxedEnvironment(
                trim_blocks=True,
                lstrip_blocks=True,
                undefined=StrictUndefined,
                extensions=[_GenerationExtension],
            )
            environment.globals["raise_exception"] = self._raise_template_exception
            environment.globals["strftime_now"] = (
                lambda pattern: datetime.datetime.now().strftime(pattern)
            )
            environment.filters["tojson"] = self._to_json
            # DeepSeek-V4's template re-renders a tool call the API handed back
            # as a JSON string, so it needs the inverse of ``tojson``. Without
            # it the filter is undefined and a second tool-using turn fails to
            # render at all.
            environment.filters["from_json"] = json.loads
            self._compiled_chat_template = environment.from_string(self.chat_template)

    def _configured_token(self, config: Mapping[str, object], field: str) -> str:
        token_id = config.get(field)
        if token_id is None or token_id == 0xFFFFFFFF:
            return ""
        try:
            return self.model.token_text(int(token_id))
        except (V2Error, KeyError, ValueError):
            return ""

    @staticmethod
    def _raise_template_exception(message: object) -> None:
        raise ValueError(str(message))

    @staticmethod
    def _to_json(value: object, indent: int | None = None) -> str:
        if indent is None:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return json.dumps(value, ensure_ascii=False, indent=indent)

    @property
    def turn_separator(self) -> str:
        """Markup that closes an unterminated assistant turn before the next one.

        Used when resuming a cached conversation whose last generation stopped
        short of an EOS token, so the reused prefix stays well formed.
        """
        if self.architecture == "laguna":
            return "</assistant>\n"
        if self.architecture == "deepseek4":
            # The template closes every turn -- assistant included -- with the
            # end-of-sentence token; there is no separate end-of-turn markup.
            return "<｜end▁of▁sentence｜>"
        if self.architecture == "muse-glimmer":
            # An assistant turn is one or more messages; <|eom|> only ends a
            # message and generation runs on past it, so the token that closes
            # the turn is <|eot|>.
            return "<|eot|>"
        return "<|im_end|>\n"

    @property
    def finished_turn_separator(self) -> str:
        """What follows a turn the model closed with a terminator of its own.

        Qwen-style templates put the next turn on its own line; DeepSeek-V4
        runs the next role token straight on from end-of-sentence, and a
        newline there is markup no training example contains.
        """
        # Muse Glimmer runs <|start|> straight on from <|eot|> the same way
        # DeepSeek-V4 does; a newline there is markup the template never emits.
        return "" if self.architecture in ("deepseek4", "muse-glimmer") else "\n"

    @property
    def bos_token_id(self) -> int | None:
        """The id the template emits at the very start of a conversation.

        Needed when a cached prefix is extended: the suffix is rendered as a
        fresh conversation, so its leading BOS has to be dropped rather than
        planted in the middle of the sequence.
        """
        return self._bos_token_id

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
        cached = self._token_byte_cache.get(token_id)
        if cached is None:
            cached = self.model.decode_token_bytes([token_id])
            self._token_byte_cache[token_id] = cached
        return cached

    def format_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        enable_thinking: bool = False,
    ) -> str:
        if not messages:
            raise ValueError("messages must not be empty")
        compiled_template = getattr(self, "_compiled_chat_template", None)
        if compiled_template is not None:
            normalized: list[dict[str, object]] = []
            for message in messages:
                role = message["role"]
                content = message["content"]
                if role not in ("system", "developer", "user", "assistant", "tool"):
                    raise ValueError(f"unsupported chat role: {role}")
                if not content.strip() and not (
                    role == "assistant" and message.get("tool_calls")
                ):
                    raise ValueError("chat message content must not be empty")
                # Some tokenizer.chat_template variants access tool_calls
                # unconditionally instead of guarding it with ``is defined``.
                # Tool-call history has already been rendered into content by
                # the OpenAI compatibility layer, so expose an empty collection
                # here to preserve that rendered representation while matching
                # the message shape expected by those templates.
                normalized.append(
                    {
                        "role": role,
                        "content": content,
                        "tool_calls": message.get("tool_calls", []),
                        "reasoning_content": message.get("reasoning_content", ""),
                        "tools": message.get("tools", []),
                    }
                )
            return compiled_template.render(
                messages=normalized,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                # Muse Glimmer's template ignores enable_thinking and reads a
                # `reasoning_strength` of its own (low / medium / high / xhigh),
                # defaulting to 'high'. The model has no setting that stops it
                # reasoning, so the flag cannot mean off; it means "think
                # harder". Leaving it unset therefore has to keep the model's
                # own default rather than quietly asking for less, because
                # enable_thinking is false on every request that never mentions
                # it. Templates that do not use the variable ignore it.
                reasoning_strength="xhigh" if enable_thinking else "high",
                tools=None,
                documents=None,
                **self._template_tokens,
            )
        if self.architecture == "gemma4":
            return self._format_gemma4(messages, enable_thinking=enable_thinking)
        if self.architecture == "laguna":
            return self._format_laguna(messages, enable_thinking=enable_thinking)
        if self.architecture == "deepseek4":
            return self._format_deepseek4(messages, enable_thinking=enable_thinking)
        sections: list[str] = []
        for message in messages:
            role = message["role"]
            content = message["content"].strip()
            if role not in ("system", "developer", "user", "assistant", "tool"):
                raise ValueError(f"unsupported chat role: {role}")
            tool_calls = message.get("tool_calls", [])
            if not content and not (role == "assistant" and tool_calls):
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

    @staticmethod
    def _format_gemma4(
        messages: Sequence[Mapping[str, str]], *, enable_thinking: bool
    ) -> str:
        """Render the text-only subset of Gemma 4's GGUF chat template."""
        sections = ["<bos>"]
        start = 0
        if enable_thinking or messages[0]["role"] in ("system", "developer"):
            sections.append("<|turn>system\n")
            if enable_thinking:
                sections.append("<|think|>\n")
            if messages[0]["role"] in ("system", "developer"):
                content = messages[0]["content"].strip()
                if not content:
                    raise ValueError("chat message content must not be empty")
                sections.append(content)
                start = 1
            sections.append("<turn|>\n")
        for message in messages[start:]:
            role = message["role"]
            if role not in ("user", "assistant"):
                raise ValueError(
                    "Gemma 4 text chat currently supports system, developer, user, and assistant messages"
                )
            content = message["content"].strip()
            if not content:
                raise ValueError("chat message content must not be empty")
            rendered_role = "model" if role == "assistant" else role
            sections.append(f"<|turn>{rendered_role}\n{content}<turn|>\n")
        sections.append("<|turn>model\n")
        if not enable_thinking:
            sections.append("<|channel>thought\n<channel|>")
        return "".join(sections)

    @staticmethod
    def _format_deepseek4(
        messages: Sequence[Mapping[str, str]], *, enable_thinking: bool
    ) -> str:
        """Render the text-only core of DeepSeek-V4's GGUF chat template.

        System messages are concatenated at the head with no markup of their
        own; user, developer and tool turns all render into a user block; and
        the assistant role token is emitted as a *transition* on the preceding
        user-side turn rather than by the assistant turn itself, so an assistant
        message that follows anything else carries no role token at all.

        Checked against the checkpoint's own template over every role sequence
        up to length four, with and without thinking.
        """
        sections = ["<｜begin▁of▁sentence｜>"]
        sections.append("\n\n".join(
            message["content"].strip() for message in messages
            if message["role"] == "system"
        ))
        opening = "<think>" if enable_thinking else "</think>"
        user_side = ("user", "developer", "tool")
        last_user = max(
            (
                index for index, message in enumerate(messages)
                if message["role"] in user_side
            ),
            default=-1,
        )
        # Thinking mode drops the developer turns the conversation has moved
        # past -- there is no tool schema in this fallback, which is the other
        # thing that would keep them.
        dropped = {
            index for index, message in enumerate(messages)
            if message["role"] == "developer" and enable_thinking and index < last_user
        }
        # A tool result belongs to the user turn it answers, and consecutive
        # user turns join rather than repeating the role token.
        in_user = False
        for index, message in enumerate(messages):
            role = message["role"]
            if role == "system" or index in dropped:
                continue
            content = message["content"].strip()
            tool_calls = message.get("tool_calls", [])
            if not content and not (role == "assistant" and tool_calls):
                raise ValueError("chat message content must not be empty")
            if role in ("user", "tool"):
                sections.append("\n\n" if in_user else "<｜User｜>")
                in_user = True
                sections.append(
                    f"<tool_result>{content}</tool_result>" if role == "tool"
                    else content
                )
            elif role == "developer":
                in_user = False
                sections.append(f"<｜User｜>{content}")
            elif role == "assistant":
                in_user = False
                predecessor = index - 1
                while predecessor in dropped:
                    predecessor -= 1
                follows_user = (
                    predecessor >= 0 and messages[predecessor]["role"] in user_side
                )
                # Reasoning survives only on a turn the conversation has not
                # moved past; ours never carries any, so a kept one is an empty
                # block and a dropped one leaves just the closing tag.
                keeps_reasoning = enable_thinking and index > last_user
                reasoning = str(message.get("reasoning_content", "") or "")
                if follows_user:
                    sections.append("<｜Assistant｜>")
                    sections.append(
                        f"<think>{reasoning}</think>" if keeps_reasoning else "</think>"
                    )
                elif keeps_reasoning:
                    sections.append(f"{reasoning}</think>")
                sections.append(content)
                if tool_calls:
                    sections.append("\n\n<｜DSML｜tool_calls>\n")
                    for call in tool_calls:
                        function = call["function"]
                        arguments = function.get("arguments", {})
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        sections.append(f'<｜DSML｜invoke name="{function["name"]}">\n')
                        for key, value in arguments.items():
                            string = isinstance(value, str)
                            rendered = value if string else json.dumps(
                                value, ensure_ascii=False, separators=(",", ":")
                            )
                            sections.append(
                                f'<｜DSML｜parameter name="{key}" '
                                f'string="{str(string).lower()}">{rendered}'
                                f'</｜DSML｜parameter>\n'
                            )
                        sections.append("</｜DSML｜invoke>\n")
                    sections.append("</｜DSML｜tool_calls>")
                sections.append("<｜end▁of▁sentence｜>")
            else:
                raise ValueError(f"unsupported chat role: {role}")
        sections.append(f"<｜Assistant｜>{opening}")
        return "".join(sections)

    _LAGUNA_SYSTEM = (
        "You are a helpful, conversationally-fluent assistant made by Poolside. "
        "You are here to be helpful to users through natural language conversations."
    )

    @classmethod
    def _format_laguna(
        cls, messages: Sequence[Mapping[str, str]], *, enable_thinking: bool
    ) -> str:
        """Render the text-only subset of Laguna's GGUF chat template.

        The template opens with the EOS token, emits a ``<system>`` block when
        there is a system message or thinking is on, and wraps each turn in role
        tags. An assistant turn always carries a ``<think>`` block, empty when
        thinking is off, because the model is trained to expect it.
        """
        sections = ["〈|EOS|〉"]
        system = cls._LAGUNA_SYSTEM
        start = 0
        if messages[0]["role"] in ("system", "developer"):
            system = messages[0]["content"].strip()
            start = 1
        if system or enable_thinking:
            sections.append(f"<system>{system}</system>\n")
        for message in messages[start:]:
            role = message["role"]
            if role not in ("user", "assistant"):
                raise ValueError(
                    "Laguna text chat currently supports system, developer, "
                    "user, and assistant messages"
                )
            content = message["content"].strip()
            if not content:
                raise ValueError("chat message content must not be empty")
            if role == "user":
                sections.append(f"<user>{content}</user>\n")
            else:
                sections.append(f"<assistant><think></think>{content}</assistant>\n")
        sections.append("<assistant>")
        sections.append("<think>" if enable_thinking else "<think></think>")
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


class ChatGenerator:
    """Chat formatting, continuation reuse and streaming over a task engine.

    Everything here depends on the engine only through ``submit``/``cancel``/
    ``forget`` and the events they produce, so an architecture whose state does
    not fit the Qwen engine's one-KV-pair-per-layer slots can bring its own
    scheduler and keep the whole server surface -- templates, tool calls,
    continuation reuse, incremental UTF-8 decoding -- unchanged.
    """

    def __init__(self, model: V2Model, engine, tokenizer: NativeV2Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.engine = engine
        self._chat_lock = threading.Lock()
        self._chat_messages: tuple[tuple[str, str], ...] | None = None
        self._chat_prompt_ids: tuple[int, ...] = ()
        self._chat_generated_ids: tuple[int, ...] = ()
        self._chat_text = ""
        self._chat_thinking = False
        # Exact generated token IDs must survive unrelated concurrent requests.
        # A single "last chat" record let short agent/title/tool side-calls
        # overwrite the main conversation and forced its next turn to re-tokenize
        # (usually diverging at structured tool markup). Keep a small LRU keyed
        # by the request history instead.
        self._chat_continuations: OrderedDict[
            tuple[tuple[str, str], ...],
            tuple[tuple[int, ...], tuple[int, ...], str, bool],
        ] = OrderedDict()
        self._chat_continuation_capacity = 32

    def prefix_cache_stats(self) -> dict[str, int]:
        """Reuse counters for ``/health``; shape is the engine's business."""
        raise NotImplementedError

    def close(self) -> None:
        self.engine.close()

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
            tuple(final.prompt_ids),
            tuple(final.generated_ids),
            final.text,
            final.stopped_on_eos,
            final.state_tokens,
        )

    def prepare_messages(
        self, messages: Sequence[Mapping[str, str]], **options: object
    ) -> list[int]:
        thinking = bool(options.get("enable_thinking", False))
        normalized = tuple(
            (message["role"], message["content"].strip()) for message in messages
        )
        with self._chat_lock:
            return self._continued_chat_prompt(normalized, thinking)

    def stream_messages(
        self, messages: Sequence[Mapping[str, str]], **options: object
    ) -> Iterator[GenerationStep]:
        thinking = bool(options.get("enable_thinking", False))
        normalized = tuple(
            (message["role"], message["content"].strip()) for message in messages
        )
        prepared = options.get("prepared_prompt_ids")
        prompt_ids = (
            [int(token) for token in prepared]
            if isinstance(prepared, Sequence)
            else self.prepare_messages(messages, enable_thinking=thinking)
        )
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
                self._chat_generated_ids = tuple(final.generated_ids)
                self._chat_text = final.text
                self._chat_thinking = thinking
                self._chat_continuations[normalized] = (
                    tuple(prompt_ids),
                    tuple(final.generated_ids),
                    final.text,
                    thinking,
                )
                self._chat_continuations.move_to_end(normalized)
                while (
                    len(self._chat_continuations)
                    > self._chat_continuation_capacity
                ):
                    self._chat_continuations.popitem(last=False)

    def _continued_chat_prompt(
        self, messages: tuple[tuple[str, str], ...], thinking: bool
    ) -> list[int]:
        candidates = list(self._chat_continuations.items())
        if self._chat_messages is not None:
            candidates.append(
                (
                    self._chat_messages,
                    (
                        self._chat_prompt_ids,
                        self._chat_generated_ids,
                        self._chat_text,
                        self._chat_thinking,
                    ),
                )
            )
        # Prefer the longest matching history. A shorter conversation can be a
        # literal prefix of a later turn but cannot reuse as much live state.
        candidates.sort(key=lambda item: len(item[0]), reverse=True)
        for previous, record in candidates:
            prompt_ids, generated_ids, raw_text, record_thinking = record
            if not (
                thinking == record_thinking
                and len(messages) > len(previous) + 1
                and messages[: len(previous)] == previous
                and messages[len(previous)][0] == "assistant"
                and self._assistant_continues_previous(
                    messages[len(previous)][1], raw_text
                )
            ):
                continue
            remaining = messages[len(previous) + 1 :]
            if remaining:
                generated = list(generated_ids)
                ended = bool(
                    generated and generated[-1] in self.tokenizer.eos_token_ids
                )
                separator = (
                    getattr(self.tokenizer, "finished_turn_separator", "\n")
                    if ended
                    else self.tokenizer.turn_separator
                )
                suffix_messages = [
                    {"role": role, "content": content} for role, content in remaining
                ]
                suffix = self.tokenizer.encode_messages(
                    suffix_messages, enable_thinking=thinking
                )
                # The suffix is rendered as though it were a whole
                # conversation, so a template that opens with BOS emits one
                # here too. Kept, it would sit in the middle of the sequence,
                # which no training example contains.
                bos = getattr(self.tokenizer, "bos_token_id", None)
                if (
                    bos is not None
                    and suffix[:1] == [bos]
                    and list(prompt_ids[:1]) == [bos]
                ):
                    suffix = suffix[1:]
                return (
                    list(prompt_ids)
                    + generated
                    + self.tokenizer.encode(separator)
                    + suffix
                )
        return self.tokenizer.encode_messages(
            [{"role": role, "content": content} for role, content in messages],
            enable_thinking=thinking,
        )

    def _assistant_continues_previous(
        self, candidate: str, raw_text: str | None = None
    ) -> bool:
        """Recognize the API's structured round-trip of our last tool call.

        OpenAI/Anthropic clients receive a native ``<tool_call>`` block as
        structured JSON, then send it back on the next turn.  Rendering that
        JSON reconstructs equivalent markup, but UUIDs, JSON whitespace and
        hidden reasoning text need not be byte-identical.  Requiring exact text
        discarded the generated token IDs and forced a near-full conversation
        prefill.  Compare parsed calls instead; plain assistant text remains an
        exact-match check so edited/regenerated replies never reuse stale state.
        """
        raw = (self._chat_text if raw_text is None else raw_text).strip()
        if candidate == raw:
            return True
        raw_content, raw_calls = _parse_tool_calls(raw)
        candidate_content, candidate_calls = _parse_tool_calls(candidate)
        if not raw_calls or not candidate_calls or len(raw_calls) != len(candidate_calls):
            return False
        # A client may omit reasoning that preceded a structured tool call.
        # If it keeps visible content, require that content to remain exact.
        if candidate_content and candidate_content != raw_content:
            return False

        def signature(call: Mapping[str, object]) -> tuple[str, object]:
            function = call["function"]
            assert isinstance(function, Mapping)
            name = function["name"]
            arguments = function["arguments"]
            assert isinstance(name, str) and isinstance(arguments, str)
            return name, json.loads(arguments)

        return [signature(call) for call in candidate_calls] == [
            signature(call) for call in raw_calls
        ]

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
            tuple(final.prompt_ids),
            tuple(final.generated_ids),
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
        sampling_config = (
            sampling if isinstance(sampling, SamplingConfig) else SamplingConfig()
        )
        progress = options.get("progress")
        progress_callback = progress if callable(progress) else None
        if not prompt_ids:
            raise ValueError("formatted prompt produced no token IDs")
        generated: list[int] = []
        text_parts: list[str] = []
        prompt_snapshot = tuple(prompt_ids)
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
            prompt_ids,
            max_new_tokens,
            self.tokenizer.eos_token_ids,
            sampling_config,
        )
        try:
            prefill_complete = False
            while True:
                kind, value = queue.get()
                if kind == "done":
                    break
                if kind == "error":
                    raise RuntimeError(str(value))
                if kind == "prefill":
                    processed = min(int(value), len(prompt_ids))
                    if progress_callback is not None:
                        progress_callback(processed, len(prompt_ids))
                    prefill_complete = processed >= len(prompt_ids)
                    continue
                token = int(value)
                # Compatibility fallback for a runtime which predates native
                # prefill events: complete the old one-shot callback here.
                if progress_callback is not None and not prefill_complete:
                    progress_callback(0, len(prompt_ids))
                    progress_callback(len(prompt_ids), len(prompt_ids))
                    prefill_complete = True
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
                text_parts.append(delta)
                yield GenerationStep(
                    token,
                    delta,
                    prompt_snapshot,
                    _TokenSnapshot(generated, len(generated)),
                    "",
                    stopped,
                    False,
                    len(prompt_ids) + len(generated),
                )
                if stopped:
                    continue
            # Flush any dangling partial UTF-8 sequence (a truncated character
            # at the very end of generation becomes a single visible U+FFFD).
            tail = utf8.decode(b"", True)
            text_parts.append(tail)
            yield GenerationStep(
                None,
                tail,
                prompt_snapshot,
                tuple(generated),
                "".join(text_parts),
                stopped,
                True,
                len(prompt_ids) + len(generated),
            )
        except GeneratorExit:
            self.engine.cancel(task_id)
            raise
        finally:
            self.engine.forget(task_id)


class NativeV2Generator(ChatGenerator):
    """Streaming generator backed by the cooperative native Qwen engine."""

    def __init__(
        self, model: V2Model, runtime: V2QwenRuntime, tokenizer: NativeV2Tokenizer
    ):
        super().__init__(model, _NativeEngine(runtime), tokenizer)
        self.runtime = runtime

    def prefix_cache_stats(self) -> dict[str, int]:
        info = self.runtime.info
        capacity = int(getattr(self.runtime, "parallel_sequences", 1))
        live_entries = 1 if info["position"] else 0
        ram_entries = int(info.get("prompt_cache_entries", 0))
        return {
            "entries": live_entries + ram_entries,
            "capacity": capacity,
            "ram_entries": ram_entries,
            "ram_bytes": int(info.get("prompt_cache_used_bytes", 0)),
            "hits": int(info["prefix_cache_hits"]),
            "misses": int(info["prefix_cache_misses"]),
            "evictions": 0,
            "reused_tokens": int(info["prefix_cache_reused_tokens"]),
            "last_prompt_tokens": int(
                info.get("prefix_cache_last_prompt_tokens", 0)
            ),
            "last_reused_tokens": int(
                info.get("prefix_cache_last_reused_tokens", 0)
            ),
            "last_lcp_live": int(info.get("prefix_cache_last_lcp_live", 0)),
            "last_lcp_snapshot": int(
                info.get("prefix_cache_last_lcp_snapshot", 0)
            ),
        }


class NativeV2InferenceService(InferenceService):
    def __init__(
        self,
        model_path: Path | str,
        *,
        mtp_model_path: Path | str | None = None,
        model_name: str | None = None,
        device: int = 0,
        context_window: int = 32768,
        max_new_tokens: int = 4096,
        gpu_cache_mib: int = 0,  # 0 = auto-fit the GPU expert cache to free VRAM
        expert_mode: str | None = None,
        moe_device: str | None = None,
        mtp_drafts: int = 0,
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
        prefill_checkpoint_interval: int = 256,
        prefill_checkpoint_slots: int = 4,
        parallel_sequences: int = 1,
        prompt_cache_mib: int = 0,
        swa_full: bool = False,
        prefill_cache_seed: int | str | None = None,
        expert_paging: str = "auto",
        cpu_prefetch_mib: int = 0,
        cpu_prefetch_auto: bool = False,
        next_layer_prefetch: int = 0,
        cpu_threads: int = 0,
        hybrid_prefill: str = "split",
        expert_residency: str | None = None,
        dense_requant: str = "auto",
        api_key: str | None = None,
        cors_origin: str = "*",
        strict_model: bool = False,
        max_concurrent_requests: int = 64,
        request_timeout_seconds: float = 30.0,
        sse_keepalive_seconds: float = 10.0,
    ):
        generation_defaults, generation_defaults_source = _generation_config_for_model(
            model_path
        )
        self.v2_model = V2Model(model_path, mtp_model=mtp_model_path)
        self.v2_runtime: V2QwenRuntime | None = None
        try:
            self.v2_runtime = self.v2_model.native_runtime(
                device=device,
                context_limit=context_window,
                gpu_cache_bytes=gpu_cache_mib * 1024**2,
                expert_mode=expert_mode,
                moe_device=moe_device,
                mtp_drafts=mtp_drafts,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                prefill_checkpoint_interval=prefill_checkpoint_interval,
                prefill_checkpoint_slots=prefill_checkpoint_slots,
                parallel_sequences=parallel_sequences,
                prompt_cache_mib=prompt_cache_mib,
                swa_full=swa_full,
                prefill_cache_seed=prefill_cache_seed,
                expert_paging=expert_paging,
                cpu_prefetch_mib=cpu_prefetch_mib,
                cpu_prefetch_auto=cpu_prefetch_auto,
                next_layer_prefetch=next_layer_prefetch,
                cpu_threads=cpu_threads,
                hybrid_prefill=hybrid_prefill,
                expert_residency=expert_residency,
                dense_requant=dense_requant,
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
            max_concurrent_requests=max_concurrent_requests,
            request_timeout_seconds=request_timeout_seconds,
            sse_keepalive_seconds=sse_keepalive_seconds,
            generation_defaults=generation_defaults,
        )
        self.generation_defaults_source = generation_defaults_source
        runtime_info = self.v2_runtime.info
        self.requested_expert_mode = str(runtime_info["requested_expert_mode"])
        self.expert_mode = str(runtime_info["expert_mode"])
        self.expert_fallback_reason = str(
            runtime_info["expert_fallback_reason"]
        )
        # Compatibility attribute retained for callers that displayed the old
        # low-level device name.
        self.moe_device = self.expert_mode
        self.mtp_drafts = mtp_drafts
        self.gpu_cache_mib = gpu_cache_mib
        # The native cooperative engine interleaves concurrent requests itself
        # (per-slot KV, single CUDA thread), so the HTTP layer must not
        # serialize them.
        self._serialize_generation = False

    def close(self) -> None:
        generator_close = getattr(self.generator, "close", None)
        if callable(generator_close):
            generator_close()
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
            "expert_mode": self.expert_mode,
            "requested_expert_mode": self.requested_expert_mode,
            "expert_fallback_reason": self.expert_fallback_reason,
            "moe_device": self.moe_device,
            "mtp_drafts": self.mtp_drafts,
            "gpu_cache_mib": self.gpu_cache_mib,
        }
        return value

    def properties(self) -> dict[str, object]:
        value = super().properties()
        tokenizer = self.generator.tokenizer
        source = getattr(tokenizer, "chat_template_source", "fallback")
        architecture = getattr(tokenizer, "architecture", "unknown")
        value["chat_template"] = (
            "tokenizer.chat_template" if source == "gguf" else f"{architecture}-fallback"
        )
        value["chat_template_source"] = source
        value["generation_defaults_source"] = self.generation_defaults_source
        return value
