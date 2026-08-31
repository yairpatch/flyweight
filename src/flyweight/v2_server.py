from __future__ import annotations

import codecs
import dataclasses
import datetime
import json
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Iterator, Mapping, Sequence, overload

from jinja2 import StrictUndefined, Template, nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment

from .generation import GenerationResult, GenerationStep
from .sampling import (
    SERVER_SETTINGS,
    SamplingConfig,
    coerce as sampling_coerce,
)
from .server import InferenceService, _parse_tool_calls, _split_reasoning_content
from .v2 import (
    AUTO_PROMPT_CACHE_MIB,
    TASK_EVENT_DONE,
    TASK_EVENT_ERROR,
    TASK_EVENT_PREFILL,
    TASK_EVENT_TOKEN,
    V2Error,
    BailingRuntime,
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
    candidates = (() if not path.is_dir() else (path / "generation_config.json",)) + (
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

    # Every server-settable sampling setting, not the three this used to read:
    # a checkpoint that ships a `repetition_penalty` in its generation config --
    # which is a standard Hugging Face field -- was having it silently ignored.
    defaults: dict[str, int | float] = {}
    for setting in SERVER_SETTINGS:
        value = sampling_coerce(setting, raw.get(setting.name))
        if value is not None:
            defaults[setting.name] = value
    max_new_tokens = raw.get("max_new_tokens")
    if isinstance(max_new_tokens, int) and not isinstance(max_new_tokens, bool):
        defaults["max_new_tokens"] = max_new_tokens
    # Hugging Face spells "greedy" as do_sample=false rather than temperature 0,
    # and it wins over any temperature in the same file.
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
        tools: list[dict[str, object]] | None = None,
        response_format: dict[str, object] | None = None,
        forbid_tool_calls: bool = False,
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
                repetition_penalty=sampling_config.repetition_penalty,
                presence_penalty=sampling_config.presence_penalty,
                frequency_penalty=sampling_config.frequency_penalty,
                penalty_window=sampling_config.penalty_window,
                seed=sampling_config.seed,
                tools=tools,
                response_format=response_format,
                forbid_tool_calls=forbid_tool_calls,
            )
            self._queues[task_id] = task_queue
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="flyweight-engine", daemon=True
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

    def task_is_live(self, task_id: int) -> bool:
        """Whether `task_id` can still produce events.

        A request waiting for a free KV slot is live and merely silent, which
        looking at its queue cannot distinguish from a terminal event that was
        never delivered -- both are "nothing arrived". The difference is here: a
        task the engine still holds, on a thread still running, gets scheduled
        eventually. Anything else never will, and a consumer blocked on it would
        wait forever.
        """
        with self._lock:
            if task_id not in self._queues:
                return False
            thread = self._thread
        return thread is not None and thread.is_alive()

    def active_task_count(self) -> int:
        with self._lock:
            return len(self._queues)

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


def _check_content(role: str, content: str, index: int) -> None:
    """The empty-turn rule every renderer below shares.

    An assistant turn may carry no text, and so may a tool result: a
    generation the user cancelled, one the token ceiling cut mid-tool-call,
    a turn of pure reasoning the client stripped before replaying it, a tool
    that printed nothing. The protocol layer preserves all of those on
    purpose (`server._chat_messages`) precisely so a conversation can be
    replayed -- and then this renderer refused them, so the 400 landed one
    layer further down as "unable to tokenize the formatted prompt". Every
    later request in that conversation failed the same way, because the
    offending turn is history: the client re-sends it forever and cannot
    edit it. Rendering an empty turn is harmless -- a role header with no
    body, which is what the turn was.

    An empty user, system or developer turn stays refused: nothing
    legitimately produces one, so it is a client bug worth reporting rather
    than absorbing. The index and role travel with the message because the
    caller sees this through a wrapper that says only that tokenization
    failed, and "some message was empty" is not a diagnosis.
    """
    if content.strip() or role in ("assistant", "tool"):
        return
    raise ValueError(
        f"chat message content must not be empty: messages[{index}] "
        f"has role {role!r}"
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
    def _to_json(
        value: object, indent: int | None = None, ensure_ascii: bool = False,
        separators: tuple[str, str] | None = None, sort_keys: bool = False,
        **_ignored: object,
    ) -> str:
        """``tojson`` with the signature and defaults template authors target.

        This mirrors the filter transformers installs for
        `apply_chat_template`, which is what a checkpoint's template is written
        and tested against -- keywords included, since BailingMoE3 renders a
        non-string tool argument with ``tojson(ensure_ascii=False)`` and a
        filter that refused the keyword failed the whole render.

        The separators matter as much as the keywords: this used to force the
        compact form, so a tool schema reached the model as
        `{"name":"write_file"}` where every other runtime -- and the training
        data -- has `{"name": "write_file"}`. Unknown keywords are swallowed;
        a rendered prompt is worth more than a formatting argument.
        """
        return json.dumps(
            value, ensure_ascii=ensure_ascii, indent=indent,
            separators=separators, sort_keys=sort_keys,
        )

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
        if self.architecture == "bailingmoe3":
            # Every turn ends with <|role_end|>, which is one special token and
            # also this checkpoint's EOS. The ChatML default below is not merely
            # the wrong markup here -- it is not markup at all: <|im_end|>
            # tokenizes as five ORDINARY text tokens, so a resumed conversation
            # carried that literal string in its middle. Which is every turn of
            # an agentic loop, because a turn that ends in a tool call is
            # cancelled rather than finished on EOS, and the model answered the
            # nonsense by repeating itself.
            return "<|role_end|>"
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
        # BailingMoE3 is the same: its template emits <role>HUMAN</role>
        # immediately after <|role_end|>, with nothing between them.
        return "" if self.architecture in (
            "deepseek4", "muse-glimmer", "bailingmoe3"
        ) else "\n"

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
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        preserve_thinking: bool | None = None,
    ) -> str:
        """Render the prompt. None leaves the checkpoint's own thinking default.

        `preserve_thinking` decides whether an assistant turn's replayed
        chain-of-thought reaches the prompt; None leaves that to the
        architecture's own convention (see keeps_replayed_reasoning). It is
        only a full override for the formatters written here -- on the jinja
        path a false can be enforced by withholding the text, but a true cannot
        make a template that strips history reasoning keep it.

        `reasoning_effort` is the same idea one level down, for a checkpoint
        that grades its reasoning rather than switching it. Qwen3.5 reads
        low / medium / xhigh and *defaults to xhigh*, which is its maximum:
        leaving the variable unset asks a 27B to "think carefully through the
        task, validate key assumptions, consider plausible alternatives" before
        answering anything at all. Measured on Qwen3.8-27B at Q2_K, that is 500
        tokens and a hit output cap for "explain in a short paragraph", against
        291 at medium. None still means the checkpoint's own default, because
        quietly asking a model for less reasoning than it was tuned for is the
        mirror of the enable_thinking mistake described above.

        A template that reasons by default -- BailingMoE3 is one -- turns it off
        only when `enable_thinking` is defined and false. Passing false on every
        request that simply never mentioned thinking therefore overrode the
        checkpoint's choice: it rendered `<think></think>` where llama.cpp
        renders `<think>`, telling a model trained to reason first not to.
        """
        if not messages:
            raise ValueError("messages must not be empty")
        compiled_template = getattr(self, "_compiled_chat_template", None)
        if compiled_template is not None:
            normalized: list[dict[str, object]] = []
            for index, message in enumerate(messages):
                role = message["role"]
                content = message["content"] or ""
                if role not in ("system", "developer", "user", "assistant", "tool"):
                    raise ValueError(f"unsupported chat role: {role}")
                _check_content(role, content, index)
                # Some tokenizer.chat_template variants access tool_calls
                # unconditionally instead of guarding it with ``is defined``,
                # so the key is always present. It carries real calls only for
                # the architectures whose template renders them itself; for the
                # rest the compatibility layer has already put that history in
                # content, and an empty collection preserves it.
                # An explicit false is enforced here rather than left to the
                # template: withholding the text is the only way to make a
                # template that always replays reasoning stop. The reverse does
                # not work -- passing it cannot make a template that strips it
                # keep it -- so `preserve_thinking` is advisory in that
                # direction, and is also exposed as a template variable for a
                # template that reads one.
                reasoning = message.get("reasoning_content", "")
                normalized.append(
                    {
                        "role": role,
                        "content": content,
                        "tool_calls": message.get("tool_calls", []),
                        "reasoning_content": (
                            "" if preserve_thinking is False else reasoning
                        ),
                        "tools": message.get("tools", []),
                    }
                )
            thinking_variables: dict[str, object] = (
                {} if enable_thinking is None
                else {"enable_thinking": enable_thinking}
            )
            if reasoning_effort:
                thinking_variables["reasoning_effort"] = reasoning_effort
            if preserve_thinking is not None:
                thinking_variables["preserve_thinking"] = preserve_thinking
            return compiled_template.render(
                messages=normalized,
                add_generation_prompt=True,
                **thinking_variables,
                # Muse Glimmer's template ignores enable_thinking and reads a
                # `reasoning_strength` of its own (low / medium / high / xhigh),
                # defaulting to 'high'. The model has no setting that stops it
                # reasoning, so the flag cannot mean off; it means "think
                # harder". Leaving it unset therefore has to keep the model's
                # own default rather than quietly asking for less, because
                # enable_thinking is false on every request that never mentions
                # it. Templates that do not use the variable ignore it.
                reasoning_strength="xhigh" if enable_thinking else "high",
                # The schemas the caller declared, for a template that renders
                # its own tool section. The compatibility layer attaches them to
                # the first message for exactly this hand-off; a template that
                # does not use the variable ignores it, and one that does now
                # instructs the model in the format it was trained on rather
                # than the generic prompt this server would otherwise inject.
                tools=next(
                    (message["tools"] for message in normalized
                     if message.get("tools")),
                    None,
                ),
                documents=None,
                **self._template_tokens,
            )
        if self.architecture == "gemma4":
            return self._format_gemma4(messages, enable_thinking=bool(enable_thinking))
        if self.architecture == "laguna":
            return self._format_laguna(messages, enable_thinking=bool(enable_thinking))
        if self.architecture == "deepseek4":
            return self._format_deepseek4(
                messages, enable_thinking=bool(enable_thinking),
                preserve_thinking=preserve_thinking)
        sections: list[str] = []
        last_user = max(
            (index for index, message in enumerate(messages)
             if message["role"] == "user"),
            default=-1,
        )
        for index, message in enumerate(messages):
            role = message["role"]
            content = (message["content"] or "").strip()
            if role not in ("system", "developer", "user", "assistant", "tool"):
                raise ValueError(f"unsupported chat role: {role}")
            _check_content(role, content, index)
            if role == "assistant" and not content.startswith("<think>"):
                reasoning = str(message.get("reasoning_content", "") or "").strip()
                if reasoning and keeps_replayed_reasoning(
                    preserve_thinking, enable_thinking, index, last_user
                ):
                    content = f"<think>\n{reasoning}\n</think>\n\n{content}"
                else:
                    # A history turn's block is CLOSED whatever this request
                    # asked for: enable_thinking governs the turn about to be
                    # generated, and opening one here left an unterminated
                    # <think> in the middle of the conversation, which no
                    # training example contains.
                    content = f"<think>\n\n</think>\n\n{content}"
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
                content = (messages[0]["content"] or "").strip()
                _check_content(messages[0]["role"], content, 0)
                sections.append(content)
                start = 1
            sections.append("<turn|>\n")
        for index, message in enumerate(messages[start:], start):
            role = message["role"]
            if role not in ("user", "assistant"):
                raise ValueError(
                    "Gemma 4 text chat currently supports system, developer, user, and assistant messages"
                )
            content = (message["content"] or "").strip()
            _check_content(role, content, index)
            rendered_role = "model" if role == "assistant" else role
            sections.append(f"<|turn>{rendered_role}\n{content}<turn|>\n")
        sections.append("<|turn>model\n")
        if not enable_thinking:
            sections.append("<|channel>thought\n<channel|>")
        return "".join(sections)

    @staticmethod
    def _format_deepseek4(
        messages: Sequence[Mapping[str, str]], *, enable_thinking: bool,
        preserve_thinking: bool | None = None,
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
            content = (message["content"] or "").strip()
            tool_calls = message.get("tool_calls", [])
            _check_content(role, content, index)
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
                # moved past, unless the caller overrode it. A turn of ours
                # carries reasoning only when a client replayed it, so a kept
                # empty one is an empty block and a dropped one leaves just the
                # closing tag.
                keeps_reasoning = keeps_replayed_reasoning(
                    preserve_thinking, enable_thinking, index, last_user)
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
            system = (messages[0]["content"] or "").strip()
            start = 1
        if system or enable_thinking:
            sections.append(f"<system>{system}</system>\n")
        for index, message in enumerate(messages[start:], start):
            role = message["role"]
            if role not in ("user", "assistant"):
                raise ValueError(
                    "Laguna text chat currently supports system, developer, "
                    "user, and assistant messages"
                )
            content = (message["content"] or "").strip()
            _check_content(role, content, index)
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
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        preserve_thinking: bool | None = None,
    ) -> list[int]:
        return self.encode(
            self.format_messages(
                messages, enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
                preserve_thinking=preserve_thinking)
        )


def _merge_generation_defaults(
    file_defaults: Mapping[str, float | int],
    source: str,
    overrides: Mapping[str, float | int] | None,
) -> tuple[dict[str, float | int], str]:
    """Server flags over a generation_config.json next to the checkpoint.

    A flag the operator passed wins: the file describes what the model shipped
    with, the flag is what this server was told to do. An absent flag must not
    overwrite the file with an argparse default nobody chose, which is why the
    caller passes only the values it was actually given.

    The source string records both, so /health cannot report a value as coming
    from a file that never set it.
    """
    merged = dict(file_defaults)
    given = {
        key: value
        for key, value in (overrides or {}).items()
        if value is not None
    }
    if not given:
        return merged, source
    merged.update(given)
    return merged, f"{source}+flags({','.join(sorted(given))})"


def _declared_value_type(properties: object, key: str) -> str:
    """`array` or `object` when the schema says so, else `string`.

    A union (`anyOf`, or a list of types) only counts when every branch agrees:
    a parameter that may be either an array or a string is legitimately either,
    and constraining it to JSON would make the string form unsamplable.
    """
    if not isinstance(properties, Mapping):
        return "string"
    schema = properties.get(key)
    if not isinstance(schema, Mapping):
        return "string"
    declared = schema.get("type")
    if isinstance(declared, str):
        return declared if declared in ("array", "object") else "string"
    if isinstance(declared, list):
        named = {item for item in declared if isinstance(item, str)}
        if named in ({"array"}, {"object"}):
            return next(iter(named))
    return "string"


def _tool_grammar_specification(
    declarations: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    """What the sampler needs out of the caller's tool schemas: each tool's
    name, and which of its parameters are required.

    Only names and requiredness, because that is the whole of what the
    constraint enforces -- a call may not close while a required parameter is
    unwritten. Types and descriptions belong to the prompt, not the sampler.

    A schema without a `required` list constrains nothing, which is correct:
    JSON Schema says every property is optional unless listed, so a tool that
    declares none has no call this could reject.
    """
    if not isinstance(declarations, Sequence) or isinstance(declarations, str):
        return []
    specification: list[dict[str, object]] = []
    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            continue
        function = declaration.get("function")
        source = function if isinstance(function, Mapping) else declaration
        name = source.get("name")
        if not isinstance(name, str) or not name:
            continue
        schema = source.get("parameters")
        if not isinstance(schema, Mapping):
            schema = source.get("input_schema")
        properties = schema.get("properties") if isinstance(schema, Mapping) else None
        required = schema.get("required") if isinstance(schema, Mapping) else None
        required_names = {
            value for value in (required or []) if isinstance(value, str)
        }
        parameters = [
            {
                "name": key,
                "required": key in required_names,
                # Only array and object matter to the sampler. Those are the
                # values the server reconstructs by parsing the parameter text
                # as JSON, so a value that is not JSON reaches the client as a
                # string and fails its schema. Scalars are already text.
                "type": _declared_value_type(properties, key),
            }
            for key in (properties or {})
            if isinstance(key, str)
        ]
        specification.append({"name": name, "parameters": parameters})
    return specification


# (role, visible content, replayed reasoning) per turn.
ChatKey = tuple[tuple[str, str, str], ...]
# What a matched prefix carries: its prompt and generated ids, the raw text, and
# the three render settings it was produced under, all of which have to agree
# before it is a prefix of THIS conversation rather than a similar one.
ChatRecord = tuple[
    tuple[int, ...], tuple[int, ...], str, bool | None, str | None, bool | None
]


def _chat_key(messages: Sequence[Mapping[str, object]]) -> ChatKey:
    """The continuation cache's key: role, visible content, replayed reasoning.

    Index 0 and 1 stay role and content, which the prefix matcher indexes
    positionally.
    """
    return tuple(
        (
            str(message["role"]),
            str(message["content"]).strip(),
            str(message.get("reasoning_content", "") or "").strip(),
        )
        for message in messages
    )


def _optional_thinking(options: Mapping[str, object]) -> bool | None:
    """`enable_thinking` as the caller left it, preserving "unstated"."""
    value = options.get("enable_thinking")
    return None if value is None else bool(value)


def _optional_preserve(options: Mapping[str, object]) -> bool | None:
    """`preserve_thinking` as the caller left it, preserving "unstated"."""
    value = options.get("preserve_thinking")
    return None if value is None else bool(value)


def keeps_replayed_reasoning(
    preserve_thinking: bool | None,
    enable_thinking: bool | None,
    index: int,
    last_user: int,
) -> bool:
    """Whether an assistant turn's replayed chain-of-thought reaches the prompt.

    The default is the convention every reasoning checkpoint is trained on:
    keep the reasoning of turns AFTER the last user message -- the tool-call
    loop the model is still inside, where its own thought is what justifies the
    call it is about to see the result of -- and drop everything older, which
    is what stops a long agentic session re-reading every thought it ever had.

    `preserve_thinking` overrides that in either direction and is only ever set
    when a client asked for it explicitly.
    """
    if preserve_thinking is not None:
        return preserve_thinking
    return bool(enable_thinking) and index > last_user


# Qwen's own budget-exhaustion wrap-up, from their thinking-budget reference
# implementation. A model conditions better on a sentence that says time is
# up than on a bare closing tag dropped mid-thought, and the phrase is short
# enough that even a 512-token budget barely notices it.
THINKING_BUDGET_CLOSE = (
    "\n\nConsidering the limited time by the user, I have to give the "
    "solution based on the thinking directly now.\n</think>\n\n"
)


class _ThinkingBudget:
    """Token meter over a stream's thinking block, for a hard budget.

    `reasoning_effort` asks a checkpoint to think less and the checkpoint is
    free to overrun; this is the enforcement the request names in tokens.
    The block boundaries are tracked textually because the markers are the
    one signal every <think>-family architecture shares, with a rolling
    window so a marker split across tokens is still seen. spend() is called
    once per sampled token and answers whether the budget just ran out, at
    which point the caller forces the block closed. The meter stays armed
    after a forced close: a checkpoint that immediately reopens a block is
    closed again on its first counted token rather than granted a second
    budget.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self, budget: int, thinking_open: bool):
        self.budget = budget
        self.inside = thinking_open
        self.spent = 0
        self._window = ""

    def close(self) -> None:
        """Record that the caller forced the block shut."""
        self.inside = False
        self._window = ""

    def spend(self, delta: str) -> bool:
        self._window += delta
        if not self.inside:
            if self._OPEN not in self._window:
                self._window = self._window[-(len(self._OPEN) - 1):]
                return False
            self.inside = True
            self._window = self._window.split(self._OPEN, 1)[1]
        if self._CLOSE in self._window:
            self.inside = False
            self._window = self._window.split(self._CLOSE, 1)[1]
            self._window = self._window[-(len(self._OPEN) - 1):]
            return False
        self._window = self._window[-(len(self._CLOSE) - 1):]
        # Undecodable and partial-UTF-8 tokens carry an empty delta but were
        # sampled all the same; the budget counts tokens, not characters.
        self.spent += 1
        return self.spent >= self.budget


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
        self._chat_messages: ChatKey | None = None
        self._chat_prompt_ids: tuple[int, ...] = ()
        self._chat_generated_ids: tuple[int, ...] = ()
        self._chat_text = ""
        self._chat_thinking: bool | None = False
        self._chat_effort: str | None = None
        self._chat_preserve: bool | None = None
        # Exact generated token IDs must survive unrelated concurrent requests.
        # A single "last chat" record let short agent/title/tool side-calls
        # overwrite the main conversation and forced its next turn to re-tokenize
        # (usually diverging at structured tool markup). Keep a small LRU keyed
        # by the request history instead.
        self._chat_continuations: OrderedDict[ChatKey, ChatRecord] = OrderedDict()
        self._chat_continuation_capacity = 32
        self._forced_close_ids: list[int] | None = None

    # How long a request may hear nothing before its scheduling is checked on.
    # Generous: it costs one wakeup per request per interval and only ever
    # decides whether to keep waiting, so the only thing a shorter value buys
    # is noticing a dead engine sooner.
    _STALL_POLL_SECONDS = 5.0

    def prefix_cache_stats(self) -> dict[str, int]:
        """Reuse counters for ``/health``; shape is the engine's business."""
        raise NotImplementedError

    def _task_is_live(self, task_id: int) -> bool:
        """Defer to the engine, and assume live for one that cannot say.

        An engine is free to bring its own scheduler; one that does not report
        liveness gets the old behaviour of waiting indefinitely rather than a
        spurious failure.
        """
        reporter = getattr(self.engine, "task_is_live", None)
        return True if not callable(reporter) else bool(reporter(task_id))

    def _report_waiting(self, prompt_tokens: int) -> None:
        """Say why a request that was accepted is producing nothing.

        Queueing for a slot is invisible from the client: it holds a 200 and an
        SSE stream carrying keepalives, so a request stuck behind a long
        generation looks identical to one that is thinking. Naming it in the log
        is what turns "the runtime hung" into "raise --parallel".
        """
        counter = getattr(self.engine, "active_task_count", None)
        active = counter() if callable(counter) else 0
        if active <= 1:
            return
        sys.stderr.write(
            f"[queue] request of {prompt_tokens} prompt tokens is waiting for a "
            f"KV slot ({active} requests in flight); raise --parallel to overlap "
            f"them\n"
        )
        sys.stderr.flush()

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
        # Tri-state, not a bool. `None` means the request never mentioned
        # thinking, and only the checkpoint's template can answer that: Qwen3.5
        # reasons by default and renders `<think>` to start the turn inside a
        # reasoning block, while an explicit false renders `<think></think>`
        # and tells a model trained to reason first not to. Coercing None to
        # False here silently did the latter on every request -- 20 prompt
        # tokens where the checkpoint's own default is 60 -- and took the
        # reasoning-effort instruction with it, since the template only grades
        # reasoning it is doing.
        thinking = _optional_thinking(options)
        preserve = _optional_preserve(options)
        effort = options.get("reasoning_effort")
        # Replayed reasoning is part of the KEY, not just of the render. It
        # changes the prompt whenever the architecture keeps it, so a reduction
        # to (role, content) would match a turn rendered from a different
        # chain-of-thought and reuse its tokens -- the same trap the docstring
        # of _continued_chat_prompt records for tool schemas.
        normalized = _chat_key(messages)
        with self._chat_lock:
            return self._continued_chat_prompt(
                normalized, thinking, messages,
                reasoning_effort=effort if isinstance(effort, str) else None,
                preserve_thinking=preserve)

    def stream_messages(
        self, messages: Sequence[Mapping[str, str]], **options: object
    ) -> Iterator[GenerationStep]:
        thinking = _optional_thinking(options)
        effort = options.get("reasoning_effort")
        # Where the schemas are depends on the architecture: the templates that
        # render their own tool section get them attached to the first message,
        # and everything else has them rendered into the tool prompt, with the
        # declarations passed alongside. The sampler wants them either way.
        if "tool_grammar" not in options:
            declared = options.get("tools")
            if not isinstance(declared, Sequence) or isinstance(declared, str):
                declared = next(
                    (message["tools"] for message in messages
                     if isinstance(message, Mapping) and message.get("tools")),
                    None,
                )
            options["tool_grammar"] = _tool_grammar_specification(declared)
        normalized = _chat_key(messages)
        prepared = options.get("prepared_prompt_ids")
        prompt_ids = (
            [int(token) for token in prepared]
            if isinstance(prepared, Sequence)
            else self.prepare_messages(
                messages, enable_thinking=thinking, reasoning_effort=effort,
                preserve_thinking=_optional_preserve(options))
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
                self._chat_effort = effort if isinstance(effort, str) else None
                self._chat_preserve = _optional_preserve(options)
                self._chat_continuations[normalized] = (
                    tuple(prompt_ids),
                    tuple(final.generated_ids),
                    final.text,
                    thinking,
                    self._chat_effort,
                    self._chat_preserve,
                )
                self._chat_continuations.move_to_end(normalized)
                while (
                    len(self._chat_continuations)
                    > self._chat_continuation_capacity
                ):
                    self._chat_continuations.popitem(last=False)

    def _continued_chat_prompt(
        self, messages: ChatKey, thinking: bool | None,
        full: Sequence[Mapping[str, object]] | None = None,
        *, reasoning_effort: str | None = None,
        preserve_thinking: bool | None = None,
    ) -> list[int]:
        """Prompt ids for `messages`, reusing a cached prefix where one fits.

        `messages` is the (role, content, reasoning) reduction the continuation
        cache is keyed and matched on; `full` is what actually gets rendered. The two are
        separate because a message carries more than role and content -- the
        tool schemas and the structured tool calls a native template renders
        itself -- and reducing before rendering silently dropped them, so a
        tool-capable model was prompted as though no tools existed.
        """
        rendered = list(full) if full is not None else [
            {"role": role, "content": content, "reasoning_content": reasoning}
            for role, content, reasoning in messages
        ]
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
                        self._chat_effort,
                        self._chat_preserve,
                    ),
                )
            )
        # Prefer the longest matching history. A shorter conversation can be a
        # literal prefix of a later turn but cannot reuse as much live state.
        candidates.sort(key=lambda item: len(item[0]), reverse=True)
        for previous, record in candidates:
            (prompt_ids, generated_ids, raw_text, record_thinking, record_effort,
             record_preserve) = record
            if not (
                thinking == record_thinking
                # A prefix rendered at another effort opens with a different
                # system prompt, so it is not a prefix of this conversation.
                and reasoning_effort == record_effort
                # ...and one rendered while replaying a different amount of the
                # conversation's own reasoning is a different prefix too.
                and preserve_thinking == record_preserve
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
                suffix_messages = rendered[len(previous) + 1:]
                suffix = self.tokenizer.encode_messages(
                    suffix_messages, enable_thinking=thinking,
                    reasoning_effort=reasoning_effort,
                    preserve_thinking=preserve_thinking,
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
            rendered, enable_thinking=thinking, reasoning_effort=reasoning_effort,
            preserve_thinking=preserve_thinking)

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
        # The client replays what it was served, and the response path splits
        # the leading reasoning out of `content` (_split_reasoning_content)
        # while `raw` still carries it -- so a thinking model's every plain
        # answer failed this check and the turn re-rendered cold. Compare
        # against the same visible reading the client received.
        visible = _split_reasoning_content(raw)[0].strip()
        if candidate == visible:
            return True
        raw_content, raw_calls = _parse_tool_calls(visible)
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

    def _thinking_close_ids(self) -> list[int]:
        if self._forced_close_ids is None:
            self._forced_close_ids = [
                int(token) for token in self.tokenizer.encode(THINKING_BUDGET_CLOSE)
            ]
        return self._forced_close_ids

    def _prompt_opens_thinking(self, prompt_ids: Sequence[int]) -> bool:
        """Whether the rendered prompt leaves the turn inside a think block."""
        try:
            tail = self.tokenizer.decode(
                list(prompt_ids[-8:]), skip_special_tokens=False
            )
        except Exception:
            return False
        return tail.rstrip().endswith("<think>")

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
        tool_grammar = options.get("tool_grammar")
        response_format = options.get("response_format")
        # The response constraint reads Qwen-family thinking markup. Muse
        # Glimmer frames its reasoning with channel tags instead, which the
        # grammar would reject on the first token -- disarming itself and
        # logging failure counters for a request that was never constrainable.
        if getattr(self.tokenizer, "architecture", None) == "muse-glimmer":
            response_format = None
        # A hard thinking budget, enforced here because only this loop can
        # cancel the task and force the block closed. Muse Glimmer is out for
        # the same reason as above: no <think> markers to meter.
        budget = options.get("reasoning_budget_tokens")
        meter = None
        if (
            isinstance(budget, int)
            and not isinstance(budget, bool)
            and budget > 0
            and getattr(self.tokenizer, "architecture", None) != "muse-glimmer"
        ):
            opens = options.get("thinking_open")
            meter = _ThinkingBudget(
                budget,
                self._prompt_opens_thinking(prompt_ids)
                if opens is None
                else bool(opens),
            )
        task_id, queue = self.engine.submit(
            prompt_ids,
            max_new_tokens,
            self.tokenizer.eos_token_ids,
            sampling_config,
            tools=tool_grammar if isinstance(tool_grammar, list) else None,
            response_format=(
                dict(response_format)
                if isinstance(response_format, Mapping)
                else None
            ),
            # No tools on this request means no parser for tool markup either:
            # the sampler refuses to open a <tool_call>, so a model steeped in
            # a tool-heavy transcript cannot leak one into plain text (an
            # opencode compaction stored exactly that as its summary).
            forbid_tool_calls=not tool_grammar,
        )
        try:
            prefill_complete = False
            waiting_reported = False
            while True:
                try:
                    kind, value = queue.get(timeout=self._STALL_POLL_SECONDS)
                except Empty:
                    # Silence is normal: with fewer KV slots than concurrent
                    # requests, a task sits in the engine's pending phase
                    # producing nothing until a slot frees. Silence from a task
                    # the engine no longer holds is not -- that is a terminal
                    # event that never arrived, and an untimed get() would wait
                    # on it forever while the SSE layer kept the client in a
                    # "working" state with no output and no error.
                    if not self._task_is_live(task_id):
                        raise RuntimeError(
                            "the native engine stopped scheduling this request"
                        )
                    if not waiting_reported and not generated:
                        waiting_reported = True
                        self._report_waiting(len(prompt_ids))
                    continue
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
                if meter is not None and meter.spend(delta):
                    # The budget is spent with the block still open: force it
                    # closed and resume the answer on a fresh task whose
                    # prompt is everything decoded so far plus the close. The
                    # engine treats that prompt like any other, so a prefix
                    # cache absorbs the restage where one exists. Named in
                    # the log because from outside, a capped think and a hung
                    # one look identical until the answer arrives.
                    sys.stderr.write(
                        f"[gen ] thinking budget of {meter.budget} tokens "
                        f"spent; closing the block and resuming the answer\n"
                    )
                    self.engine.cancel(task_id)
                    self.engine.forget(task_id)
                    meter.close()
                    for forced in self._thinking_close_ids():
                        generated.append(forced)
                        try:
                            forced_delta = utf8.decode(
                                self.tokenizer.token_bytes(forced)
                            )
                        except Exception:
                            forced_delta = ""
                        text_parts.append(forced_delta)
                        yield GenerationStep(
                            forced,
                            forced_delta,
                            prompt_snapshot,
                            _TokenSnapshot(generated, len(generated)),
                            "",
                            False,
                            False,
                            len(prompt_ids) + len(generated),
                        )
                    remaining = max_new_tokens - len(generated)
                    if remaining <= 0:
                        break
                    if isinstance(response_format, Mapping):
                        # The old task's constraint tracked its own text; the
                        # new one starts after a close the prompt now carries.
                        response_format = {
                            **response_format, "thinking_open": False
                        }
                    task_id, queue = self.engine.submit(
                        list(prompt_snapshot) + generated,
                        remaining,
                        self.tokenizer.eos_token_ids,
                        sampling_config,
                        tools=(
                            tool_grammar
                            if isinstance(tool_grammar, list)
                            else None
                        ),
                        response_format=(
                            dict(response_format)
                            if isinstance(response_format, Mapping)
                            else None
                        ),
                        forbid_tool_calls=not tool_grammar,
                    )
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
            "last_slot": int(info.get("prefix_cache_last_slot", 0)),
            # What the slots reserve against what they were ever asked to hold.
            # Slots are sized for the full context at load, so on a small card
            # the difference is expert cache that decode would rather have --
            # see plans/paged-kv-cache.md. Samples of zero means no request has
            # finished yet, not that occupancy was zero.
            "kv_reserved_bytes": int(info.get("kv_reserved_bytes", 0)),
            "kv_peak_live_bytes": int(info.get("kv_peak_live_bytes", 0)),
            "kv_peak_tokens": int(info.get("kv_peak_tokens", 0)),
            "kv_peak_tokens_max": int(info.get("kv_peak_tokens_max", 0)),
            "kv_occupancy_samples": int(info.get("kv_occupancy_samples", 0)),
            # Prompts that shared a live conversation's opening without
            # continuing it, and were given their own slot seeded from it.
            "donations": int(info.get("prefix_donations", 0)),
            "donated_tokens": int(info.get("prefix_donated_tokens", 0)),
        }

    def prefix_diagnostics(self) -> dict[str, object] | None:
        """Where the last admitted prompt split from its slot's history.

        Decoded from the token snippets the runtime captures at admission,
        so the prefill log can show WHAT the client rewrote rather than only
        how much the rewrite cost. None until a prompt has been admitted.
        Special tokens are kept: a divergence at a turn boundary IS the
        interesting case, and hiding <|im_end|> would hide it.
        """
        info = self.runtime.info
        prompt_tokens = int(info.get("prefix_cache_last_prompt_tokens", 0))
        if not prompt_tokens:
            return None
        old_count = int(info.get("prefix_cache_last_old_count", 0))
        new_count = int(info.get("prefix_cache_last_new_count", 0))
        old_ids = [int(t) for t in info.get("prefix_cache_last_old_tokens", [])]
        new_ids = [int(t) for t in info.get("prefix_cache_last_new_tokens", [])]
        return {
            "slot": int(info.get("prefix_cache_last_slot", 0)),
            "cached_tokens": int(info.get("prefix_cache_last_cached_tokens", 0)),
            "prompt_tokens": prompt_tokens,
            "reused_tokens": int(info.get("prefix_cache_last_reused_tokens", 0)),
            "divergence": int(info.get("prefix_cache_last_lcp_live", 0)),
            "old_text": self.tokenizer.decode(
                old_ids[:old_count], skip_special_tokens=False
            ),
            "new_text": self.tokenizer.decode(
                new_ids[:new_count], skip_special_tokens=False
            ),
        }


@dataclasses.dataclass
class _BailingActive:
    """One generation in flight, and everything a step of it needs.

    Mutable and held by the engine thread alone: `pending` is the token the
    next step feeds its slot, which is the prompt's last token to begin with
    and the previous sample after that.
    """

    task_id: int
    slot: int
    pending: int
    stops: set
    sampling: object
    remaining: int


class BailingEngine:
    """Engine adapter for the BailingMoE3 host/GPU runtime.

    Presents the same submit/cancel/forget/close surface as `_NativeEngine` so
    `ChatGenerator` needs no changes, but it is deliberately much simpler: the
    bailing runtime owns ONE sequence of caches, so tasks run strictly one at a
    time rather than being interleaved across slots. Concurrent requests queue.
    """

    _MAX_ACTIVE_TASKS = 32
    _MAX_BUFFERED_EVENTS = 256
    # Host budget for put-aside sequences when the caller does not say. A
    # snapshot costs the tokens it holds (~12 KB/token on Ling-3.0-tiny), so
    # this is a few long conversations or many short ones, and re-prefilling
    # one of them costs seconds.
    _DEFAULT_SNAPSHOT_BUDGET_BYTES = 512 * 1024**2
    # A prefill this long is worth a snapshot of its own. Saving costs one copy
    # of the live cache; re-running the prompt costs seconds, so the ratio only
    # goes wrong for prompts that were cheap to begin with.
    _SNAPSHOT_PREFILL_THRESHOLD = 256

    def __init__(self, runtime: "BailingRuntime", snapshot_budget_bytes: int | None = None):
        self.runtime = runtime
        self._snapshot_budget = (
            self._DEFAULT_SNAPSHOT_BUDGET_BYTES
            if snapshot_budget_bytes is None
            else max(0, snapshot_budget_bytes)
        )
        self._lock = threading.Lock()
        self._queues: dict[int, Queue[tuple[str, object]]] = {}
        self._pending: list[tuple[int, list[int], int, tuple[int, ...], object]] = []
        self._cancelled: set[int] = set()
        self._next_id = 1
        self._closing = False
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        # One live sequence per slot, and the exact tokens each holds.
        # Ordinary conversation continuation extends one of them exactly.
        self._slot_count = max(1, int(getattr(runtime, "slot_count", 1) or 1))
        self._slot_tokens: list[list[int]] = [[] for _ in range(self._slot_count)]
        self._slot_initialized = [False] * self._slot_count
        self._free_slots = list(range(self._slot_count))
        # task_id -> the in-flight generation state a step advances.
        self._active: dict[int, _BailingActive] = {}
        # Sequences that are not live, kept so they need not be re-prefilled.
        #
        # A coding harness does not send one conversation: a short side-call
        # (title, summary, a second agent) lands between two turns of the main
        # one, and with a single set of caches that side-call left nothing of
        # the main conversation behind -- every turn paid a full prefill of a
        # prompt the runtime had already seen. Snapshots are also what makes a
        # prompt that is a strict PREFIX of the live sequence reusable, which is
        # every regenerate and every client that does not send our reply back:
        # the recurrent KDA state cannot be rewound, only restored.
        self._snapshots: OrderedDict[tuple[int, ...], bytes] = OrderedDict()
        self._snapshot_bytes = 0
        self.reused_tokens = 0
        self.prefilled_tokens = 0

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
        # As in the DeepSeek-V4 engine: BailingMoE3 runs its own runtime, whose
        # sampler has no constrained decoding -- the markup ban included -- so
        # its tool calls and a JSON response format stay with the tolerant
        # parser and the prompt. The arguments are accepted so the shared
        # streaming path has one calling convention.
        del tools, response_format, forbid_tool_calls
        task_queue: Queue[tuple[str, object]] = Queue(maxsize=self._MAX_BUFFERED_EVENTS)
        with self._lock:
            if self._closing:
                raise RuntimeError("bailing engine is shutting down")
            if len(self._queues) >= self._MAX_ACTIVE_TASKS:
                raise RuntimeError("bailing engine queue is full; retry later")
            task_id = self._next_id
            self._next_id += 1
            self._queues[task_id] = task_queue
            self._pending.append(
                (task_id, list(prompt_ids), max_new_tokens, stop_tokens,
                 sampling or SamplingConfig())
            )
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="flyweight-bailing-engine", daemon=True
                )
                self._thread.start()
        self._wake.set()
        return task_id, task_queue

    def cancel(self, task_id: int) -> None:
        with self._lock:
            self._cancelled.add(task_id)

    def forget(self, task_id: int) -> None:
        with self._lock:
            self._queues.pop(task_id, None)
            self._cancelled.discard(task_id)

    def task_is_live(self, task_id: int) -> bool:
        with self._lock:
            if task_id not in self._queues:
                return False
            thread = self._thread
        return thread is not None and thread.is_alive()

    def active_task_count(self) -> int:
        with self._lock:
            return len(self._queues)

    def close(self) -> None:
        with self._lock:
            self._closing = True
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=30.0)

    def _emit(self, task_id: int, event: tuple[str, object]) -> bool:
        """False when the consumer has gone away or the task was cancelled."""
        with self._lock:
            task_queue = self._queues.get(task_id)
            cancelled = task_id in self._cancelled or self._closing
        if task_queue is None or cancelled:
            return False
        try:
            task_queue.put_nowait(event)
        except Exception:
            return False
        return True

    def _run(self) -> None:
        """Admit what fits, then advance every live generation one token.

        One task used to run start to finish before the next was looked at, so
        a request arriving behind a long generation waited out all of it. With
        a slot each they interleave: a 4000-token answer no longer holds up
        the 50-token one behind it.

        Prefill is still done at admission, in one call, so a very long prompt
        does hold the others for its duration -- a far smaller window than a
        whole generation, and chunking it is the next step if it bites.
        """
        while True:
            with self._lock:
                if self._closing and not self._pending and not self._active:
                    return
                idle = not self._pending and not self._active
            if idle:
                self._wake.wait(timeout=0.05)
                self._wake.clear()
                continue
            self._admit_pending()
            if not self._step_active():
                # Nothing advanced: everything is either finished or waiting
                # for a slot that has not freed yet.
                self._wake.wait(timeout=0.002)
                self._wake.clear()

    def _admit_pending(self) -> None:
        while True:
            with self._lock:
                if not self._pending or not self._free_slots or self._closing:
                    return
                task = self._pending.pop(0)
                slot = self._take_slot(task[1])
            task_id, prompt_ids, max_new_tokens, stop_tokens, sampling = task
            try:
                active = self._prefill(task_id, slot, prompt_ids,
                                       max_new_tokens, stop_tokens, sampling)
            except Exception as error:  # keep the worker alive across failures
                self._forget_slot(slot)
                self._emit(task_id, ("error", str(error)))
                continue
            if active is None:
                self._release_slot(slot)
                continue
            with self._lock:
                self._active[task_id] = active

    def _take_slot(self, prompt_ids: list[int]) -> int:
        """The free slot that already holds the most of this prompt.

        Affinity, not round-robin: a conversation's next turn extends the
        sequence its own slot is still holding, and handing it a different
        slot would throw that away and re-prefill from a snapshot at best.
        Called with the lock held.
        """
        best, best_live = self._free_slots[0], -1
        for slot in self._free_slots:
            live = len(self._slot_tokens[slot])
            if not self._slot_initialized[slot]:
                live = 0
            elif live >= len(prompt_ids) or prompt_ids[:live] != self._slot_tokens[slot]:
                live = 0  # not a prefix of this prompt: no affinity at all
            if live > best_live:
                best, best_live = slot, live
        self._free_slots.remove(best)
        return best

    def _release_slot(self, slot: int) -> None:
        with self._lock:
            if slot not in self._free_slots:
                self._free_slots.append(slot)

    def _forget_slot(self, slot: int) -> None:
        """Give up on what a slot holds, and hand it back.

        Evaluation advances the runtime's position token by token while the
        token list is only extended once a call returns, so a throw partway
        through a prompt leaves the two disagreeing -- and a snapshot taken
        from that state would be handed back later as a valid prefix it is
        not. Existing snapshots are unaffected: restoring one sets the
        position itself.
        """
        with self._lock:
            self._slot_initialized[slot] = False
            self._slot_tokens[slot].clear()
            if slot not in self._free_slots:
                self._free_slots.append(slot)

    def _step_active(self) -> bool:
        """Advance every live generation by one token. False if none moved."""
        with self._lock:
            active = list(self._active.values())
        moved = False
        for task in active:
            try:
                finished = self._step(task)
            except Exception as error:
                self._forget_slot(task.slot)
                with self._lock:
                    self._active.pop(task.task_id, None)
                self._emit(task.task_id, ("error", str(error)))
                continue
            moved = True
            if finished:
                self._release_slot(task.slot)
                with self._lock:
                    self._active.pop(task.task_id, None)
        return moved

    def _step(self, task: "_BailingActive") -> bool:
        """One token for one slot. True when the generation is over.

        The eval and the sample are one step on purpose. The runtime keeps a
        single logits buffer that the last evaluation filled, whichever slot
        it belonged to, so sampling a slot that another slot has evaluated
        since would read the wrong sequence's logits -- and produce fluent,
        confidently wrong text rather than an error.
        """
        self.runtime.eval_into([task.pending], slot=task.slot)
        self._slot_tokens[task.slot].append(task.pending)
        token = self.runtime.sample(task.sampling)
        if not self._emit(task.task_id, ("token", token)):
            return True
        task.remaining -= 1
        if token in task.stops:
            # A stop token ends the turn and is not part of what comes back,
            # so it is never evaluated -- as before slots existed.
            self._emit(task.task_id, ("done", None))
            return True
        if task.remaining <= 0:
            # Out of budget, but the reply is what the next turn will send
            # back, so leave the slot holding all of it rather than one token
            # short. Without this every continuation re-evaluates the last
            # token it already had.
            self.runtime.eval_into([token], slot=task.slot)
            self._slot_tokens[task.slot].append(token)
            self._emit(task.task_id, ("done", None))
            return True
        task.pending = token
        return False

    def cache_stats(self) -> dict[str, int]:
        """Sequences put aside, and how much prefill they have saved.

        Counted in tokens rather than the Qwen runtime's block hits/misses:
        this cache holds whole sequences, so "how much prefill did it save" is
        the only number here that means anything.
        """
        with self._lock:
            return {
                "capacity": len(self._snapshots) + 1,
                "used": self._snapshot_bytes,
                "hits": self.reused_tokens,
                "misses": self.prefilled_tokens,
            }

    def _drop_snapshot(self, key: tuple[int, ...]) -> None:
        self._snapshot_bytes -= len(self._snapshots.pop(key))

    def _remember_live_sequence(self, slot: int,
                                protect: tuple[int, ...] | None = None) -> None:
        """Put a slot's live sequence aside so it need not be prefilled again.

        `protect` is a snapshot the caller is about to restore from. Two turns
        of one conversation are nearly the same size, so storing the second can
        push the store over budget -- and the LRU end is the older snapshot,
        which is exactly the one being restored. Evicting it turned a hit into a
        full prefill of a prompt the cache was holding.
        """
        if not self._slot_initialized[slot] or not self._slot_tokens[slot]:
            return
        if self._snapshot_budget <= 0:  # --cache off
            return
        key = tuple(self._slot_tokens[slot])
        if key in self._snapshots:
            self._snapshots.move_to_end(key)
            return
        snapshot = self.runtime.save_state(slot=slot)
        self._snapshots[key] = snapshot
        self._snapshot_bytes += len(snapshot)
        for candidate in list(self._snapshots):
            if self._snapshot_bytes <= self._snapshot_budget:
                break
            if candidate == protect or candidate == key:
                continue
            self._drop_snapshot(candidate)
        # Nothing else was expendable: the sequence just stored is the one to
        # give up, not the one about to be used.
        if self._snapshot_bytes > self._snapshot_budget:
            self._drop_snapshot(key)

    def _restore_longest_prefix(self, slot: int, prompt_ids: list[int]) -> int:
        """Put one slot on the longest known prefix of this prompt.

        Returns how many of the prompt's tokens the caches already hold. At
        least one token is always left to evaluate: the sampler reads the logits
        of the last token evaluated, and a fully restored prompt would sample
        from whatever the previous sequence left in the buffer.
        """
        limit = len(prompt_ids) - 1
        live = len(self._slot_tokens[slot])
        if (
            self._slot_initialized[slot]
            and live <= limit
            and prompt_ids[:live] == self._slot_tokens[slot]
        ):
            self.reused_tokens += live
            return live
        # Choose the restore target BEFORE putting the live sequence aside, so
        # that storing the live one cannot evict the target.
        best: tuple[int, ...] | None = None
        for candidate in self._snapshots:
            length = len(candidate)
            if length > limit or (best is not None and length <= len(best)):
                continue
            if prompt_ids[:length] == list(candidate):
                best = candidate
        # Switching away from this slot's live sequence: keep it before it is
        # lost. Only this slot's is at risk; the others are untouched.
        self._remember_live_sequence(slot, protect=best)
        if best is not None:
            self.runtime.load_state(self._snapshots[best], slot=slot)
            self._snapshots.move_to_end(best)
            self._slot_tokens[slot] = list(best)
            self._slot_initialized[slot] = True
            self.reused_tokens += len(best)
            return len(best)
        self.runtime.reset(slot=slot)
        self._slot_tokens[slot].clear()
        self._slot_initialized[slot] = True
        return 0

    def _prefill_progress(self, task_id: int, processed: int, total: int) -> bool:
        """Report a prompt's progress; False abandons it.

        A full event queue is not a cancellation -- progress is the one event
        stream that may be dropped, since the next one supersedes it -- so only
        a cancelled or closing task stops the prompt.
        """
        with self._lock:
            if task_id in self._cancelled or self._closing:
                return False
            task_queue = self._queues.get(task_id)
            if task_queue is None:
                return False
        try:
            task_queue.put_nowait(("prefill", processed))
        except Full:
            pass
        return True

    def _prefill(self, task_id, slot, prompt_ids, max_new_tokens, stop_tokens,
                 sampling) -> "_BailingActive | None":
        """Put a slot on this prompt. None when there is nothing to generate.

        Everything but the prompt's final token is evaluated here; that last
        token is left for the first step, which is what keeps every sample
        immediately behind its own slot's evaluation.
        """
        if not prompt_ids:
            self._emit(task_id, ("done", None))
            return None
        cached = self._restore_longest_prefix(slot, prompt_ids)
        # Make reuse and long prefills visible to the HTTP progress logger. The
        # runtime reports per tile while the prompt runs, so a long one shows
        # movement instead of one line minutes later, and a request whose client
        # has gone is dropped mid-prompt rather than run to completion.
        self._emit(task_id, ("prefill", cached))
        suffix = prompt_ids[cached:]
        if suffix:
            self.prefilled_tokens += len(suffix)
            self.runtime.set_progress(
                lambda processed, total: self._prefill_progress(
                    task_id, cached + processed, len(prompt_ids)
                )
            )
            # The prompt is evaluated in two steps so its snapshot can be taken
            # one token short of the end. A snapshot AT the end is unusable --
            # restoring it would leave nothing to evaluate and the sampler would
            # read the logits of whatever ran before -- so the same prompt
            # arriving again would have to be re-run in full. One token short
            # serves the repeat and every continuation equally.
            #
            # Worth a copy only when the prefill was long enough to dwarf it;
            # this is what a re-send, a retry after a cancel and a regenerate
            # land on, none of which carry our reply.
            try:
                head = suffix[:-1]
                if head:
                    self.runtime.eval_into(head, slot=slot)
                    self._slot_tokens[slot].extend(head)
                    if len(head) >= self._SNAPSHOT_PREFILL_THRESHOLD:
                        self._remember_live_sequence(slot)
            finally:
                self.runtime.set_progress(None)
        self._emit(task_id, ("prefill", len(prompt_ids)))
        if max_new_tokens <= 0:
            self._emit(task_id, ("done", None))
            return None
        # The prompt's last token is the first step's input, so the sample that
        # follows it reads this slot's own logits.
        return _BailingActive(
            task_id=task_id, slot=slot, pending=prompt_ids[-1],
            stops=set(stop_tokens), sampling=sampling,
            remaining=max_new_tokens,
        )


class BailingGenerator(ChatGenerator):
    """Streaming generator backed by the BailingMoE3 runtime."""

    def __init__(self, model: V2Model, runtime: "BailingRuntime",
                 tokenizer: NativeV2Tokenizer,
                 snapshot_budget_bytes: int | None = None):
        super().__init__(model, BailingEngine(runtime, snapshot_budget_bytes),
                         tokenizer)
        self.runtime = runtime

    def prefix_cache_stats(self) -> dict[str, int]:
        return self.engine.cache_stats()


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
        scratch_context: int = 0,
        prompt_cache_mib: int = 0,
        swa_full: bool = False,
        prefill_cache_seed: int | str | None = None,
        routed_moe: bool = False,
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
        max_tool_call_tokens: int = 0,
        default_thinking_budget: int = 2048,
        reasoning_effort: str | None = None,
        generation_defaults: Mapping[str, float | int] | None = None,
    ):
        generation_defaults, generation_defaults_source = _merge_generation_defaults(
            *_generation_config_for_model(model_path), generation_defaults
        )
        self.v2_model = V2Model(model_path, mtp_model=mtp_model_path)
        # BailingMoE3 runs on its own runtime rather than the Qwen one: 24 of
        # its 24 layers use attention the Qwen path does not implement. It is a
        # narrower runtime -- one sequence, no prefix cache, no expert paging --
        # so the options below that describe those features do not apply to it.
        self.architecture = str(self.v2_model.info["architecture"])
        self.bailing_runtime: BailingRuntime | None = None
        if self.architecture == "bailingmoe3":
            self._init_bailing(
                model_name=model_name or Path(model_path).stem,
                max_new_tokens=max_new_tokens,
                context_window=context_window,
                api_key=api_key,
                cors_origin=cors_origin,
                strict_model=strict_model,
                max_concurrent_requests=max_concurrent_requests,
                request_timeout_seconds=request_timeout_seconds,
                sse_keepalive_seconds=sse_keepalive_seconds,
                max_tool_call_tokens=max_tool_call_tokens,
                default_thinking_budget=default_thinking_budget,
                reasoning_effort=reasoning_effort,
                generation_defaults=generation_defaults,
                prompt_cache_mib=prompt_cache_mib,
                parallel_sequences=parallel_sequences,
            )
            self.generation_defaults_source = generation_defaults_source
            self.gpu_cache_mib = gpu_cache_mib
            self.mtp_drafts = 0
            return
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
                scratch_context=scratch_context,
                prompt_cache_mib=prompt_cache_mib,
                swa_full=swa_full,
                prefill_cache_seed=prefill_cache_seed,
                routed_moe=routed_moe,
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
            max_tool_call_tokens=max_tool_call_tokens,
            default_thinking_budget=default_thinking_budget,
            reasoning_effort=reasoning_effort,
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

    def _init_bailing(self, *, model_name, max_new_tokens, context_window, api_key,
                      cors_origin, strict_model, max_concurrent_requests,
                      request_timeout_seconds, sse_keepalive_seconds,
                      max_tool_call_tokens, generation_defaults,
                      default_thinking_budget=2048,
                      reasoning_effort=None, prompt_cache_mib=0,
                      parallel_sequences=1) -> None:
        self.v2_runtime = None
        tokenizer = NativeV2Tokenizer(self.v2_model)
        # --parallel means the same thing here as on the Qwen runtime:
        # independent sequences that decode without waiting for each other.
        # A slot costs its caches alone, and the runtime refuses a count that
        # will not fit rather than allocating until it cannot.
        self.bailing_runtime = BailingRuntime(
            self.v2_model, capacity=context_window,
            slots=max(1, int(parallel_sequences or 1)),
        )
        # This runtime's prompt cache is the snapshot store: the same budget,
        # spent on put-aside sequences rather than the Qwen runtime's blocks.
        # `auto` is a sentinel the caller passes through, not a size.
        snapshot_budget = (
            None if prompt_cache_mib >= AUTO_PROMPT_CACHE_MIB
            else prompt_cache_mib * 1024**2
        )
        super().__init__(
            model_name,
            BailingGenerator(self.v2_model, self.bailing_runtime, tokenizer,
                             snapshot_budget),
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            api_key=api_key,
            cors_origin=cors_origin,
            strict_model=strict_model,
            max_concurrent_requests=max_concurrent_requests,
            request_timeout_seconds=request_timeout_seconds,
            sse_keepalive_seconds=sse_keepalive_seconds,
            max_tool_call_tokens=max_tool_call_tokens,
            default_thinking_budget=default_thinking_budget,
            reasoning_effort=reasoning_effort,
            generation_defaults=generation_defaults,
        )
        self.requested_expert_mode = "n/a"
        self.expert_mode = "n/a"
        self.expert_fallback_reason = ""
        self.moe_device = "n/a"
        # One sequence of caches, so requests must not overlap.
        self._serialize_generation = True

    def close(self) -> None:
        generator_close = getattr(self.generator, "close", None)
        if callable(generator_close):
            generator_close()
        if getattr(self, "bailing_runtime", None) is not None:
            self.bailing_runtime.close()
            self.bailing_runtime = None
        if self.v2_runtime is not None:
            self.v2_runtime.close()
            self.v2_runtime = None
        self.v2_model.close()

    def health(self) -> dict[str, object]:
        # getattr rather than attribute access: tests construct this service
        # without running __init__, so the field may not exist.
        if getattr(self, "bailing_runtime", None) is not None:
            value = super().health()
            value["execution"] = {
                "backend": "native-v2-bailingmoe3",
                "architecture": self.architecture,
                "device": "gpu" if self.bailing_runtime.uses_gpu else "cpu",
                "parallel_sequences": int(
                    getattr(self.bailing_runtime, "slot_count", 1)
                ),
            }
            return value
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
