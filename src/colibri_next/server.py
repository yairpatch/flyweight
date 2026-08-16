from __future__ import annotations

import contextlib
import hmac
import json
import queue
import re
import select
import socket
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence, overload
from urllib.parse import unquote, urlsplit

from .generation import GenerationResult, GenerationStep
from .sampling import SamplingConfig


class Generator(Protocol):
    @property
    def tokenizer(self) -> Tokenizer: ...

    def prepare_messages(
        self, messages: Sequence[Mapping[str, str]], **options: object
    ) -> list[int]: ...
    def generate_messages(
        self, messages: Sequence[Mapping[str, str]], **options: object
    ) -> GenerationResult: ...
    def stream_messages(
        self, messages: Sequence[Mapping[str, str]], **options: object
    ) -> Iterator[GenerationStep]: ...
    def generate_text(self, prompt: str, **options: object) -> GenerationResult: ...
    def stream_text(self, prompt: str, **options: object) -> Iterator[GenerationStep]: ...


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def encode_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        enable_thinking: bool | None = None,
    ) -> list[int]: ...
    def decode(
        self, tokens: list[int], *, skip_special_tokens: bool = True
    ) -> str: ...


MAX_REQUEST_BYTES = 1024 * 1024
UI_DIRECTORY = Path(__file__).with_name("ui")
UI_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
    "/preview.html": ("preview.html", "text/html; charset=utf-8"),
}
UI_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
    "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
)
# The preview shell runs model-generated code. The sandbox directive gives it
# an opaque origin with no access to the app origin, its storage, or the API,
# while inline scripts and styles stay allowed inside the sandbox.
PREVIEW_CSP = (
    "sandbox allow-scripts allow-modals; "
    "default-src 'unsafe-inline' 'unsafe-eval' data: blob: https: http:; "
    "frame-ancestors 'self'"
)
VALID_ROLES = frozenset(("system", "developer", "user", "assistant", "tool"))
TEXT_PART_TYPES = frozenset(("text", "input_text", "output_text"))
# Different model families emit tool calls in different shapes. Accept any
# <tool_call>...</tool_call> block and decode the body as either the Hermes
# style (<function=name><parameter=k>v</parameter></function>) or the JSON
# style ({"name": ..., "arguments": {...}}) used by Qwen3, DeepSeek, GLM, etc.
TOOL_CALL_MARKER = "<tool_call>"
TOOL_CALL_END_MARKER = "</tool_call>"
TOOL_CALL_BLOCK_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_FUNCTION_PATTERN = re.compile(r"<function=([^>\n]+)>", re.DOTALL)
# The value is captured verbatim: \s* here would eat the first line's
# indentation, which is content for tools that match on exact text (Edit).
TOOL_PARAMETER_PATTERN = re.compile(
    r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL
)
# Every opening tag, so a body can be checked for parameters the strict pattern
# above could not close.
TOOL_PARAMETER_OPEN_TAG_PATTERN = re.compile(r"<parameter=[^>\n]+>")
# The same parameter with its closing tag optional, ending instead at whatever
# tag comes next. Only ever used to recover keys the strict pattern missed, so a
# well-formed value that happens to contain </function> as text still decodes
# through the strict pattern and keeps its literal content.
TOOL_PARAMETER_LOOSE_PATTERN = re.compile(
    r"<parameter=([^>\n]+)>"
    r"(.*?)"
    r"(?:</parameter>|(?=<parameter=)|(?=</function>)|(?=</tool_call>)|\Z)",
    re.DOTALL,
)
# One newline on each side of a Hermes value is framing, per the layout
# _tool_prompt() shows the model. Trailing horizontal space is only consumed
# after that newline, where it is the closing tag's indentation.
# BailingMoE3's tagged argument pairs. The value is captured verbatim for the
# same reason a Hermes parameter is: it may be file content whose leading
# whitespace is meaningful.
TOOL_ARG_KEY_PATTERN = re.compile(r"<arg_key>")
TOOL_ARG_PAIR_PATTERN = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
)
# The same pairs with the value still open, so a parameter the model is part way
# through can be streamed rather than held until its closing tag arrives.
TOOL_ARG_OPEN_PATTERN = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>((?:(?!</arg_value>).)*)\Z", re.DOTALL
)
TOOL_PARAMETER_OPEN_PATTERN = re.compile(
    r"<parameter=([^>\n]+)>((?:(?!</parameter>).)*)\Z", re.DOTALL
)
TOOL_PARAMETER_LEAD = re.compile(r"\A\r?\n")
TOOL_PARAMETER_TAIL = re.compile(r"\r?\n[ \t]*\Z")
DSML_TOOL_CALL_MARKER = "<｜DSML｜tool_calls>"
DSML_TOOL_CALL_END_MARKER = "</｜DSML｜tool_calls>"
DSML_TOOL_CALL_BLOCK_PATTERN = re.compile(
    r"<｜DSML｜tool_calls>\s*(.*?)\s*</｜DSML｜tool_calls>", re.DOTALL
)
DSML_INVOKE_PATTERN = re.compile(
    r'<｜DSML｜invoke\s+name="([^"]+)">\s*(.*?)\s*</｜DSML｜invoke>', re.DOTALL
)
DSML_PARAMETER_PATTERN = re.compile(
    r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)">'
    r'(.*?)</｜DSML｜parameter>', re.DOTALL
)
THINKING_BLOCK_PATTERN = re.compile(r"\A\s*<think>(.*?)</think>\s*", re.DOTALL)
# Muse Glimmer renders an assistant turn as a run of recipient-tagged messages
# rather than one block with a thinking prefix. The header is optionally
# preceded by <|start|>assistant (the generation prompt already supplied the
# first one) and the recipient itself is optional on the final message.
MUSE_MESSAGE_HEADER = re.compile(
    r"(?:<\|start\|>assistant)?[ \t]*(?:to=(?P<recipient>[^\s<|]+))?[ \t]*<\|message\|>"
)
MUSE_MESSAGE_END = re.compile(r"<\|eom\|>|<\|eot\|>")
# Anything that can begin the markup following a message body. A body is only
# streamed once its tail cannot still be growing into one of these.
_MUSE_MARKERS = ("<|eom|>", "<|eot|>", "<|start|>")
_MUSE_LONGEST_MARKER = max(len(marker) for marker in _MUSE_MARKERS)


@dataclass(frozen=True, slots=True)
class APIError(Exception):
    status: int
    message: str
    error_type: str = "invalid_request_error"
    parameter: str | None = None
    # OpenAI's machine-readable error code. Agentic clients branch on
    # "context_length_exceeded" to compact a conversation and retry, and a null
    # code leaves them nothing to match on but the prose.
    code: str | None = None


@dataclass(frozen=True, slots=True)
class _GenerationRequest:
    messages: list[dict[str, str]]
    prompt_ids: tuple[int, ...]
    max_new_tokens: int
    sampling: SamplingConfig
    enable_thinking: bool | None
    tools_enabled: bool = False
    tools: tuple[dict[str, Any], ...] = ()
    # Route a reasoning model's chain-of-thought to `reasoning_content` rather
    # than streaming it as ordinary content. Off by default so output stays
    # live for clients that render only `content`.
    separate_reasoning: bool = False
    # True when the rendered prompt ends with an open <think>, so the turn
    # begins inside a reasoning block that only the closing tag will end.
    thinking_open: bool = False


@dataclass(frozen=True, slots=True)
class _TextRequest:
    prompt: str
    max_new_tokens: int
    sampling: SamplingConfig


class InferenceService:
    """Protocol adapter for a persistent native model generator."""

    def __init__(
        self,
        model_name: str,
        generator: Generator,
        *,
        max_new_tokens: int = 64,
        context_window: int = 4096,
        api_key: str | None = None,
        cors_origin: str = "*",
        strict_model: bool = False,
        max_concurrent_requests: int = 64,
        request_timeout_seconds: float = 30.0,
        # How long a pooled connection may sit idle between requests. Separate
        # from the request timeout, and much longer: a harness holds
        # connections open across a generation it is waiting on elsewhere, and
        # closing one at the request timeout cost a reconnect and logged an
        # error for a connection that was behaving perfectly.
        keepalive_timeout_seconds: float = 900.0,
        sse_keepalive_seconds: float = 10.0,
        # Tokens a single tool call may run to before it is abandoned. 0 leaves
        # it bounded only by max_new_tokens, which is the default because a
        # legitimate call can be large -- writing a file puts the whole file in
        # one parameter, and a cap that fits a "normal" call would truncate it.
        # Set it where a model is known to loop inside <tool_call>.
        max_tool_call_tokens: int = 0,
        generation_defaults: Mapping[str, int | float] | None = None,
    ):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if context_window <= 0:
            raise ValueError("context_window must be positive")
        if max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if keepalive_timeout_seconds <= 0:
            raise ValueError("keepalive_timeout_seconds must be positive")
        if sse_keepalive_seconds <= 0:
            raise ValueError("sse_keepalive_seconds must be positive")
        if max_tool_call_tokens < 0:
            raise ValueError("max_tool_call_tokens must be non-negative")
        self.model_name = model_name
        self.generator = generator
        self.max_new_tokens = max_new_tokens
        self.context_window = context_window
        self.api_key = api_key
        self.cors_origin = cors_origin
        self.strict_model = strict_model
        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout_seconds = request_timeout_seconds
        self.keepalive_timeout_seconds = keepalive_timeout_seconds
        self.sse_keepalive_seconds = sse_keepalive_seconds
        self.max_tool_call_tokens = max_tool_call_tokens
        defaults: dict[str, int | float] = {
            "temperature": 0.0,
            "top_k": 20,
            "top_p": 0.95,
            "max_new_tokens": max_new_tokens,
        }
        defaults.update(generation_defaults or {})
        try:
            SamplingConfig(
                temperature=float(defaults["temperature"]),
                top_k=int(defaults["top_k"]),
                top_p=float(defaults["top_p"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid generation defaults: {error}") from error
        configured_max = int(defaults["max_new_tokens"])
        if configured_max <= 0:
            raise ValueError("generation default max_new_tokens must be positive")
        self.generation_defaults = {
            "temperature": float(defaults["temperature"]),
            "top_k": int(defaults["top_k"]),
            "top_p": float(defaults["top_p"]),
            "max_new_tokens": min(configured_max, max_new_tokens),
        }
        self.loaded_at = int(time.time())
        self._generation_lock = threading.Lock()
        # Generators that multiplex concurrent requests natively (the v2
        # cooperative engine) set this False so requests interleave instead of
        # queueing behind one long generation.
        self._serialize_generation = True
        self._request_slots = threading.BoundedSemaphore(max_concurrent_requests)
        self._request_count_lock = threading.Lock()
        self._active_requests = 0
        self._response_lock = threading.Lock()
        self._response_records: OrderedDict[
            str, tuple[dict[str, Any], list[dict[str, str]]]
        ] = OrderedDict()

    @contextlib.contextmanager
    def _generation_guard(self):
        if not self._request_slots.acquire(blocking=False):
            raise APIError(
                429,
                "the inference queue is full; retry later",
                "rate_limit_error",
            )
        with self._request_count_lock:
            self._active_requests += 1
        try:
            if self._serialize_generation:
                with self._generation_lock:
                    yield
            else:
                yield
        finally:
            with self._request_count_lock:
                self._active_requests -= 1
            self._request_slots.release()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.model_name,
            "loaded_at": self.loaded_at,
            "busy": self._active_requests > 0,
            "active_requests": self._active_requests,
            "request_capacity": self.max_concurrent_requests,
            "context_window": self.context_window,
            "prefix_cache": (
                self.generator.prefix_cache_stats()
                if hasattr(self.generator, "prefix_cache_stats")
                else {"entries": 0, "capacity": 0}
            ),
            "execution": {"backend": "native-v2"},
        }

    def model(self) -> dict[str, Any]:
        return {
            "id": self.model_name,
            "object": "model",
            "created": self.loaded_at,
            "owned_by": "colibri-next",
        }

    def models(self) -> dict[str, Any]:
        return {"object": "list", "data": [self.model()]}

    def me(self) -> dict[str, Any]:
        return {
            "id": "local",
            "object": "user",
            "owned_by": "colibri-next",
        }

    def chat_completion(
        self,
        payload: Mapping[str, Any],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        if _boolean_option(payload, "stream", False):
            raise APIError(400, "use the streaming response path", parameter="stream")
        request = self._prepare_chat(payload)
        with self._generation_guard():
            result = self._generate_request(
                request, tools=request.tools, progress=progress
            )
        return self._chat_response(result, tools=request.tools)

    def _generate_request(
        self,
        request: _GenerationRequest,
        *,
        tools: Sequence[dict[str, Any]],
        progress: Callable[[int, int], None] | None,
    ) -> GenerationResult:
        """Generate one request, stopping as soon as a valid tool call closes.

        Native tool syntax is converted to structured API output only after it
        parses successfully. Once a complete call has parsed, any further model
        output is both invalid (the tool prompt requires no suffix) and harmful:
        clients otherwise wait silently until EOS or the output-token ceiling.
        Closing the iterator cancels the native task immediately.
        """
        if not request.tools_enabled:
            return self.generator.generate_messages(
                request.messages,
                prepared_prompt_ids=request.prompt_ids,
                max_new_tokens=request.max_new_tokens,
                sampling=request.sampling,
                enable_thinking=request.enable_thinking,
                progress=progress,
            )

        final_step: GenerationStep | None = None
        text_parts: list[str] = []
        end_marker_tail = ""
        steps = self.generator.stream_messages(
            request.messages,
            prepared_prompt_ids=request.prompt_ids,
            max_new_tokens=request.max_new_tokens,
            sampling=request.sampling,
            enable_thinking=request.enable_thinking,
            progress=progress,
        )
        try:
            for step in steps:
                final_step = step
                if step.text_delta:
                    text_parts.append(step.text_delta)
                    marker_window = end_marker_tail + step.text_delta
                    end_marker_tail = marker_window[-len(TOOL_CALL_END_MARKER) :]
                complete_call = (
                    TOOL_CALL_END_MARKER in marker_window
                    if step.text_delta
                    else False
                )
                if complete_call:
                    complete_call = _has_complete_tool_call(
                        "".join(text_parts), tools=tools
                    )
                if step.finished or complete_call:
                    break
        finally:
            close = getattr(steps, "close", None)
            if close is not None:
                close()
        if final_step is None:
            raise RuntimeError("generation stream ended without a final result")
        if final_step.finished:
            return _generation_result(final_step)
        return GenerationResult(
            prompt_ids=tuple(final_step.prompt_ids),
            generated_ids=tuple(final_step.generated_ids),
            text="".join(text_parts),
            stopped_on_eos=final_step.stopped_on_eos,
            state_tokens=final_step.state_tokens,
        )

    def stream_chat_completion(
        self,
        payload: Mapping[str, Any],
        *,
        progress: Callable[[int, int], None] | None = None,
        _prepared_request: _GenerationRequest | None = None,
    ) -> Iterator[dict[str, Any] | str]:
        request = _prepared_request or self._prepare_chat(payload)
        stream_options = payload.get("stream_options") or {}
        if not isinstance(stream_options, dict):
            raise APIError(
                400, "stream_options must be an object", parameter="stream_options"
            )
        include_usage = stream_options.get("include_usage", False)
        if not isinstance(include_usage, bool):
            raise APIError(
                400,
                "stream_options.include_usage must be a boolean",
                parameter="stream_options",
            )
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        def events() -> Iterator[dict[str, Any] | str]:
            yield self._chat_chunk(
                completion_id,
                created,
                {"role": "assistant", "content": ""},
            )
            if request.tools_enabled:
                # Stream text deltas token-by-token so clients (Claude Code,
                # Codex, ...) see live output and never hang waiting for the
                # first token. Raw tool syntax is held back until one complete,
                # valid call parses; at that point generation is cancelled and
                # the structured call is emitted immediately.
                final_step: GenerationStep | None = None
                text_parts: list[str] = []
                pending = ""  # bounded tail that may be a partial tool marker
                tool_start: int | None = None  # index of a <tool_call> marker once seen
                tool_end_tail = ""
                tool_calls: list[dict[str, Any]] = []
                decode_started: float | None = None
                last_tool_progress_at = 0.0
                last_tool_progress_tokens = 0
                tool_body = ""  # text after the marker, fed to the streamer
                tool_start_tokens = 0  # generated-token count when the call opened
                tool_streamer: _ToolCallStreamer | None = None
                tool_stream_id: str | None = None
                marker = TOOL_CALL_MARKER
                holdback = len(marker) - 1
                tool_channels = (
                    MuseChannelStream()
                    if getattr(self.generator.tokenizer, "architecture", None)
                    == "muse-glimmer"
                    else ThinkingPrefixStream() if request.thinking_open
                    else None
                )

                def _content_chunk(text: str, tokens: int, elapsed: float):
                    chunk = self._chat_chunk(completion_id, created, {"content": text})
                    chunk["colibri"] = {
                        "generated_tokens": tokens,
                        "decode_elapsed_seconds": elapsed,
                    }
                    return chunk

                def _tool_progress_chunk(tokens: int, elapsed: float):
                    chunk = self._chat_chunk(completion_id, created, {})
                    chunk["colibri"] = {
                        "generated_tokens": tokens,
                        "decode_elapsed_seconds": elapsed,
                        "phase": "tool_call",
                    }
                    return chunk

                def _tool_delta_chunk(
                    call: dict[str, Any], tokens: int, elapsed: float
                ):
                    """One incremental tool_calls delta, OpenAI streaming shape.

                    The Anthropic translator already turns these into a
                    tool_use content block plus input_json_delta events, so
                    emitting them is what puts real content on the wire while a
                    long call is still being written.
                    """
                    chunk = self._chat_chunk(
                        completion_id, created, {"tool_calls": [call]}
                    )
                    chunk["colibri"] = {
                        "generated_tokens": tokens,
                        "decode_elapsed_seconds": elapsed,
                        "phase": "tool_call",
                    }
                    return chunk

                with self._generation_guard():
                    steps = self.generator.stream_messages(
                        request.messages,
                        prepared_prompt_ids=request.prompt_ids,
                        max_new_tokens=request.max_new_tokens,
                        sampling=request.sampling,
                        enable_thinking=request.enable_thinking,
                        progress=progress,
                    )
                    try:
                        for step in steps:
                            final_step = step
                            if step.finished:
                                continue
                            if step.token_id is None:
                                continue
                            delta_text = step.text_delta or ""
                            text_parts.append(delta_text)
                            now = time.perf_counter()
                            if decode_started is None:
                                decode_started = now
                            if tool_channels is not None:
                                # Hide the reasoning channel before the tool
                                # marker scan sees it: text_parts keeps the raw
                                # turn, which is what the tool parser needs.
                                delta_text, channel_reasoning = tool_channels.feed(
                                    delta_text
                                )
                                if channel_reasoning:
                                    yield self._chat_chunk(
                                        completion_id,
                                        created,
                                        {
                                            _reasoning_delta_field(
                                                request.separate_reasoning
                                            ): channel_reasoning
                                        },
                                    )
                            marker_window = ""
                            if tool_start is None:
                                # Stream clean text, but stop at (and never leak)
                                # the native marker so clients receive a structured
                                # tool call rather than raw XML.
                                pending += delta_text
                                found = pending.find(marker)
                                if found != -1:
                                    delta = pending[:found]
                                    pending = pending[found:]
                                    tool_start = 0
                                    tool_start_tokens = len(step.generated_ids)
                                    marker_window = pending
                                    tool_body = pending[len(marker) :]
                                    tool_end_tail = marker_window[
                                        -len(TOOL_CALL_END_MARKER) :
                                    ]
                                else:
                                    # Hold back a possible partial marker at the tail.
                                    safe = max(0, len(pending) - holdback)
                                    delta = pending[:safe]
                                    pending = pending[safe:]
                                if delta:
                                    yield _content_chunk(
                                        delta,
                                        len(step.generated_ids),
                                        now - decode_started,
                                    )
                            else:
                                marker_window = tool_end_tail + delta_text
                                tool_body += delta_text
                                tool_end_tail = marker_window[
                                    -len(TOOL_CALL_END_MARKER) :
                                ]
                            if tool_start is not None and (
                                TOOL_CALL_END_MARKER in marker_window
                            ):
                                accumulated = "".join(text_parts)
                                _, tool_calls = _parse_tool_calls(
                                    accumulated, tools=request.tools
                                )
                                if tool_calls:
                                    break
                            if tool_start is not None and not tool_calls:
                                token_count = len(step.generated_ids)
                                elapsed = now - decode_started
                                if (
                                    self.max_tool_call_tokens
                                    and token_count - tool_start_tokens
                                    > self.max_tool_call_tokens
                                ):
                                    # A call that never closes would otherwise
                                    # run to max_new_tokens. Stop here and let
                                    # the reconciliation below close whatever
                                    # was streamed.
                                    break
                                # Stream the call itself rather than only a
                                # liveness ping: a ping is not content, and a
                                # client that waits minutes for content while a
                                # large parameter is written times the stream
                                # out no matter how many pings it carries.
                                if tool_streamer is None:
                                    tool_streamer = _ToolCallStreamer()
                                was_started = tool_streamer.started
                                fragments = tool_streamer.feed(tool_body)
                                if not was_started and tool_streamer.started:
                                    # The schema is only resolvable once the
                                    # name is, and it decides which values may
                                    # be streamed before they are complete.
                                    tool_streamer.bind_schema(
                                        _tool_argument_schema(
                                            request.tools, tool_streamer.name or ""
                                        )
                                    )
                                    fragments = tool_streamer.feed(tool_body)
                                    tool_stream_id = f"call_{uuid.uuid4().hex}"
                                    yield _tool_delta_chunk(
                                        {
                                            "index": 0,
                                            "id": tool_stream_id,
                                            "type": "function",
                                            "function": {
                                                "name": tool_streamer.name,
                                                "arguments": "",
                                            },
                                        },
                                        token_count,
                                        elapsed,
                                    )
                                    last_tool_progress_at = now
                                    last_tool_progress_tokens = token_count
                                if fragments:
                                    yield _tool_delta_chunk(
                                        {
                                            "index": 0,
                                            "function": {
                                                "arguments": "".join(fragments)
                                            },
                                        },
                                        token_count,
                                        elapsed,
                                    )
                                    last_tool_progress_at = now
                                    last_tool_progress_tokens = token_count
                                elif (
                                    last_tool_progress_tokens == 0
                                    or token_count - last_tool_progress_tokens >= 32
                                    or now - last_tool_progress_at >= 1.0
                                ):
                                    # Nothing settled yet -- the name is still
                                    # arriving, or the open value is of a type
                                    # that cannot be emitted until it closes.
                                    yield _tool_progress_chunk(token_count, elapsed)
                                    last_tool_progress_tokens = token_count
                                    last_tool_progress_at = now
                    finally:
                        close = getattr(steps, "close", None)
                        if close is not None:
                            close()
                if final_step is None:
                    raise RuntimeError("generation stream ended without a final result")
                if tool_channels is not None:
                    # Release what the channel filter was withholding as a
                    # possible marker. A turn cut short by max_tokens can end on
                    # a character that begins one -- without this the answer
                    # loses its last few characters, and only when tools are
                    # enabled. Routed through `pending` so the tool-marker tail
                    # flush below is the single place that emits it.
                    tail_visible, tail_reasoning = tool_channels.flush()
                    if tail_reasoning:
                        yield self._chat_chunk(
                            completion_id,
                            created,
                            {
                                _reasoning_delta_field(
                                    request.separate_reasoning
                                ): tail_reasoning
                            },
                        )
                    if tail_visible and tool_start is None:
                        pending += tail_visible
                accumulated = "".join(text_parts)
                if not tool_calls:
                    _, tool_calls = _parse_tool_calls(accumulated, tools=request.tools)
                streamed_tail: list[str] = []
                produced_tool_call = bool(tool_calls)
                if tool_streamer is not None and tool_streamer.started:
                    # Reconcile: whatever was streamed is a prefix of the
                    # authoritative arguments, and this closes the difference.
                    # A call that failed to parse still gets its JSON closed so
                    # the client is never left holding a truncated object.
                    first = tool_calls[0] if tool_calls else None
                    streamed_tail = tool_streamer.finish(
                        json.loads(first["function"]["arguments"])
                        if first is not None
                        else None
                    )
                    # The streamed block owns index 0 and its own id; only the
                    # calls after it still need emitting in full.
                    tool_calls = tool_calls[1:] if tool_calls else []
                if streamed_tail or tool_calls:
                    calls: list[dict[str, Any]] = []
                    if streamed_tail:
                        calls.append(
                            {
                                "index": 0,
                                "function": {"arguments": "".join(streamed_tail)},
                            }
                        )
                    offset = 1 if tool_streamer is not None and tool_streamer.started else 0
                    calls.extend(
                        {"index": index + offset, **call}
                        for index, call in enumerate(tool_calls)
                    )
                    chunk = self._chat_chunk(
                        completion_id, created, {"tool_calls": calls}
                    )
                    if decode_started is not None:
                        chunk["colibri"] = {
                            "generated_tokens": len(final_step.generated_ids),
                            "decode_elapsed_seconds": (
                                time.perf_counter() - decode_started
                            ),
                            "phase": "tool_call",
                        }
                    yield chunk
                elif tool_start is None and pending:
                    # No tool-call marker: flush the tail we held back in case it
                    # was a partial marker, so the turn is never silently empty.
                    yield self._chat_chunk(
                        completion_id, created, {"content": pending}
                    )
                # else: a <tool_call> marker was seen but produced no parseable
                # call (truncated by max_tokens or malformed). Suppress the raw
                # markup from streamed[..] -- the finish_reason below reports
                # "length"/"stop" so the client knows the tool call was cut off.
                finish_reason = (
                    "tool_calls"
                    if produced_tool_call
                    else ("stop" if final_step.stopped_on_eos else "length")
                )
                prompt_count = len(final_step.prompt_ids)
                completion_count = len(final_step.generated_ids)
            else:
                plain_final_step: GenerationStep | None = None
                plain_decode_started: float | None = None
                # Muse Glimmer always reasons, so the raw stream carries its
                # chain-of-thought and framing; route them to reasoning_content
                # instead of letting them through as assistant text. Keyed on
                # the architecture rather than on spotting markup mid-stream,
                # which would forward the opening "to=self" as content before
                # there was enough text to recognize it.
                channels = (
                    MuseChannelStream()
                    if getattr(self.generator.tokenizer, "architecture", None)
                    == "muse-glimmer"
                    else ThinkingPrefixStream() if request.thinking_open
                    else None
                )
                with self._generation_guard():
                    for step in self.generator.stream_messages(
                        request.messages,
                        prepared_prompt_ids=request.prompt_ids,
                        max_new_tokens=request.max_new_tokens,
                        sampling=request.sampling,
                        enable_thinking=request.enable_thinking,
                        progress=progress,
                    ):
                        if step.finished:
                            plain_final_step = step
                        elif step.token_id is not None:
                            now = time.perf_counter()
                            if plain_decode_started is None:
                                plain_decode_started = now
                            delta = _channel_delta(
                                channels,
                                step.text_delta,
                                separate_reasoning=request.separate_reasoning,
                            )
                            chunk = self._chat_chunk(
                                completion_id,
                                created,
                                delta,
                            )
                            # Keep the standard OpenAI chunk shape while exposing
                            # a small provider extension for live UI metrics.
                            chunk["colibri"] = {
                                "generated_tokens": len(step.generated_ids),
                                "decode_elapsed_seconds": now - plain_decode_started,
                            }
                            yield chunk
                if channels is not None:
                    # Whatever the marker holdback was still withholding.
                    tail_visible, tail_reasoning = channels.flush()
                    tail = {}
                    if tail_reasoning:
                        field = _reasoning_delta_field(request.separate_reasoning)
                        tail[field] = tail_reasoning
                    if tail_visible:
                        tail["content"] = tail.get("content", "") + tail_visible
                    if tail:
                        yield self._chat_chunk(completion_id, created, tail)
                if plain_final_step is None:
                    raise RuntimeError("generation stream ended without a final result")
                finish_reason = (
                    "stop" if plain_final_step.stopped_on_eos else "length"
                )
                prompt_count = len(plain_final_step.prompt_ids)
                completion_count = len(plain_final_step.generated_ids)
            yield self._chat_chunk(
                completion_id,
                created,
                {},
                finish_reason=finish_reason,
            )
            if include_usage:
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_name,
                    "choices": [],
                    "usage": _usage(prompt_count, completion_count),
                }
            yield "[DONE]"

        return events()

    def completion(
        self,
        payload: Mapping[str, Any],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        if _boolean_option(payload, "stream", False):
            raise APIError(400, "use the streaming response path", parameter="stream")
        request = self._prepare_text(payload)
        with self._generation_guard():
            result = self.generator.generate_text(
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                sampling=request.sampling,
                progress=progress,
            )
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        return self._completion_response(completion_id, result)

    def stream_completion(
        self,
        payload: Mapping[str, Any],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[dict[str, Any] | str]:
        request = self._prepare_text(payload)
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        def events() -> Iterator[dict[str, Any] | str]:
            final_step: GenerationStep | None = None
            with self._generation_guard():
                for step in self.generator.stream_text(
                    request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    sampling=request.sampling,
                    progress=progress,
                ):
                    if step.finished:
                        final_step = step
                    elif step.text_delta:
                        yield self._completion_chunk(
                            completion_id, created, step.text_delta, None
                        )
            if final_step is None:
                raise RuntimeError("generation stream ended without a final result")
            finish_reason = "stop" if final_step.stopped_on_eos else "length"
            yield self._completion_chunk(completion_id, created, "", finish_reason)
            yield "[DONE]"

        return events()

    def tokenize(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        content = payload.get("content", payload.get("prompt"))
        if not isinstance(content, str):
            raise APIError(400, "content must be text", parameter="content")
        tokens = self.generator.tokenizer.encode(content)
        return {"tokens": tokens, "count": len(tokens)}

    def detokenize(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        tokens = payload.get("tokens")
        if not isinstance(tokens, list) or any(
            isinstance(token, bool) or not isinstance(token, int) for token in tokens
        ):
            raise APIError(
                400, "tokens must be an array of integers", parameter="tokens"
            )
        return {
            "content": self.generator.tokenizer.decode(
                tokens, skip_special_tokens=False
            )
        }

    def count_response_input(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        messages = _response_messages(payload)
        tools = _response_tools(payload)
        if tools:
            _prepend_tool_prompt(messages, tools, payload.get("tool_choice"))
        tokens = self.generator.tokenizer.encode_messages(
            messages,
            enable_thinking=_boolean_option(payload, "enable_thinking", None),
        )
        return {"object": "response.input_tokens", "input_tokens": len(tokens)}

    def count_anthropic_input(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Count the exact prompt produced by the Anthropic compatibility path."""
        chat_payload = _anthropic_to_chat_payload(
            {**payload, "max_tokens": payload.get("max_tokens", 1)}
        )
        messages, _ = _chat_messages(
            chat_payload, architecture=getattr(self.generator.tokenizer, "architecture", None)
        )
        tokens = self.generator.tokenizer.encode_messages(
            messages,
            enable_thinking=_boolean_option(
                chat_payload, "enable_thinking", None
            ),
        )
        return {"input_tokens": len(tokens)}

    def properties(self) -> dict[str, Any]:
        return {
            "model_path": self.model_name,
            "model_alias": self.model_name,
            "total_slots": 1,
            "max_output_tokens": self.max_new_tokens,
            "context_window": self.context_window,
            "chat_template": "qwen-chatml",
            "generation_defaults": dict(self.generation_defaults),
            "capabilities": [
                "completion",
                "chat",
                "chat_streaming",
                "responses",
                "responses_streaming",
                "function_tools",
                "tokenize",
            ],
        }

    def slots(self) -> list[dict[str, Any]]:
        return [
            {
                "id": 0,
                "is_processing": self._generation_lock.locked(),
                "model": self.model_name,
            }
        ]

    def response(
        self,
        payload: Mapping[str, Any],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        if _boolean_option(payload, "stream", False):
            raise APIError(400, "use the streaming response path", parameter="stream")
        messages = self._response_messages_with_history(payload)
        history_messages = [dict(message) for message in messages]
        tools = _response_tools(payload)
        if tools:
            _prepend_tool_prompt(messages, tools, payload.get("tool_choice"))
        request = self._prepare_generation(
            payload,
            messages,
            max_key="max_output_tokens",
            tools_enabled=bool(tools),
        )
        with self._generation_guard():
            result = self._generate_request(request, tools=tools, progress=progress)
        response = self._response_object(payload, result)
        self._store_response(response, history_messages, result)
        return response

    def stream_response(
        self,
        payload: Mapping[str, Any],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[dict[str, Any]]:
        messages = self._response_messages_with_history(payload)
        history_messages = [dict(message) for message in messages]
        tools = _response_tools(payload)
        if tools:
            _prepend_tool_prompt(messages, tools, payload.get("tool_choice"))
        request = self._prepare_generation(
            payload,
            messages,
            max_key="max_output_tokens",
            tools_enabled=bool(tools),
        )
        response_id = f"resp_{uuid.uuid4().hex}"
        message_id = f"msg_{uuid.uuid4().hex}"
        created_at = int(time.time())

        def events() -> Iterator[dict[str, Any]]:
            sequence = 0
            created_response = self._response_shell(
                payload, response_id, created_at, status="in_progress"
            )
            yield {
                "type": "response.created",
                "sequence_number": sequence,
                "response": created_response,
            }
            sequence += 1

            if request.tools_enabled:
                with self._generation_guard():
                    result = self._generate_request(
                        request, tools=tools, progress=progress
                    )
                completed_response = self._response_object(
                    payload,
                    result,
                    response_id=response_id,
                    message_id=message_id,
                    created_at=created_at,
                )
                for event in _response_output_events(
                    completed_response["output"], sequence
                ):
                    yield event
                    sequence = event["sequence_number"] + 1
                self._store_response(completed_response, history_messages, result)
                yield {
                    "type": (
                        "response.incomplete"
                        if completed_response["status"] == "incomplete"
                        else "response.completed"
                    ),
                    "sequence_number": sequence,
                    "response": completed_response,
                }
                return

            output_item = {
                "id": message_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            }
            yield {
                "type": "response.output_item.added",
                "sequence_number": sequence,
                "output_index": 0,
                "item": output_item,
            }
            sequence += 1
            content_part = {
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": "",
            }
            yield {
                "type": "response.content_part.added",
                "sequence_number": sequence,
                "item_id": message_id,
                "output_index": 0,
                "content_index": 0,
                "part": content_part,
            }
            sequence += 1
            final_step: GenerationStep | None = None
            with self._generation_guard():
                for step in self.generator.stream_messages(
                    request.messages,
                    prepared_prompt_ids=request.prompt_ids,
                    max_new_tokens=request.max_new_tokens,
                    sampling=request.sampling,
                    enable_thinking=request.enable_thinking,
                    progress=progress,
                ):
                    if step.finished:
                        final_step = step
                    elif step.text_delta:
                        yield {
                            "type": "response.output_text.delta",
                            "sequence_number": sequence,
                            "item_id": message_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": step.text_delta,
                            "logprobs": [],
                        }
                        sequence += 1
            if final_step is None:
                raise RuntimeError("generation stream ended without a final result")
            result = GenerationResult(
                prompt_ids=tuple(final_step.prompt_ids),
                generated_ids=tuple(final_step.generated_ids),
                text=final_step.text,
                stopped_on_eos=final_step.stopped_on_eos,
                state_tokens=final_step.state_tokens,
            )
            yield {
                "type": "response.output_text.done",
                "sequence_number": sequence,
                "item_id": message_id,
                "output_index": 0,
                "content_index": 0,
                "text": result.text,
                "logprobs": [],
            }
            sequence += 1
            done_part = {**content_part, "text": result.text}
            yield {
                "type": "response.content_part.done",
                "sequence_number": sequence,
                "item_id": message_id,
                "output_index": 0,
                "content_index": 0,
                "part": done_part,
            }
            sequence += 1
            done_item = {
                **output_item,
                "status": (
                    "completed" if result.stopped_on_eos else "incomplete"
                ),
                "content": [done_part],
            }
            yield {
                "type": "response.output_item.done",
                "sequence_number": sequence,
                "output_index": 0,
                "item": done_item,
            }
            sequence += 1
            completed_response = self._response_object(
                payload,
                result,
                response_id=response_id,
                message_id=message_id,
                created_at=created_at,
            )
            self._store_response(completed_response, history_messages, result)
            yield {
                "type": (
                    "response.incomplete"
                    if completed_response["status"] == "incomplete"
                    else "response.completed"
                ),
                "sequence_number": sequence,
                "response": completed_response,
            }

        return events()

    def anthropic_message(
        self,
        payload: Mapping[str, Any],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        chat_payload = _anthropic_to_chat_payload(payload)
        response = self.chat_completion(chat_payload, progress=progress)
        choice = response["choices"][0]
        message = choice["message"]
        content: list[dict[str, Any]] = []
        if message.get("content"):
            content.append({"type": "text", "text": message["content"]})
        for call in message.get("tool_calls", []):
            try:
                input_value = json.loads(call["function"]["arguments"])
            except (KeyError, json.JSONDecodeError):
                input_value = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": call.get("id", f"toolu_{uuid.uuid4().hex}"),
                    "name": call["function"]["name"],
                    "input": input_value,
                }
            )
        finish_reason = choice.get("finish_reason")
        stop_reason = {
            "tool_calls": "tool_use",
            "length": "max_tokens",
        }.get(finish_reason, "end_turn")
        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": payload.get("model", self.model_name),
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": response["usage"]["prompt_tokens"],
                "output_tokens": response["usage"]["completion_tokens"],
            },
        }

    def stream_anthropic_message(
        self,
        payload: Mapping[str, Any],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[dict[str, Any]]:
        chat_payload = _anthropic_to_chat_payload(payload)
        chat_payload["stream_options"] = {"include_usage": True}
        prepared_request = self._prepare_chat(chat_payload)
        input_tokens = len(prepared_request.prompt_ids)
        chat_events = self.stream_chat_completion(
            chat_payload,
            progress=progress,
            _prepared_request=prepared_request,
        )

        def events() -> Iterator[dict[str, Any]]:
            message_id = f"msg_{uuid.uuid4().hex}"
            yield {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": payload.get("model", self.model_name),
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            }
            # Anthropic's SDK/CLI expect periodic pings on the stream; emit one
            # up front so clients see liveness before the first token arrives.
            yield {"type": "ping"}
            block_index = 0
            output_tokens = 0
            stop_reason = "end_turn"
            usage: dict[str, Any] | None = None
            text_block_index = None
            tool_blocks: dict[str, int] = {}
            last_colibri: dict[str, Any] | None = None
            for event in chat_events:
                if event == "[DONE]":
                    for open_index in sorted(
                        i
                        for i in [text_block_index, *tool_blocks.values()]
                        if i is not None
                    ):
                        yield {"type": "content_block_stop", "index": open_index}
                    yield {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                        "usage": {"output_tokens": output_tokens},
                        **(
                            {"colibri": last_colibri}
                            if last_colibri is not None
                            else {}
                        ),
                    }
                    yield {"type": "message_stop"}
                    return
                if not isinstance(event, dict):
                    continue
                colibri = event.get("colibri")
                if isinstance(colibri, dict):
                    last_colibri = colibri
                    if colibri.get("phase") == "tool_call" and not _is_decode_event(
                        event
                    ):
                        # Keep Claude Code's stream alive and expose local
                        # progress while raw tool syntax remains intentionally
                        # hidden until the complete call can be parsed.
                        yield {"type": "ping", "colibri": colibri}
                        continue
                if event.get("usage"):
                    usage = event["usage"]
                    output_tokens = usage.get("completion_tokens", output_tokens)
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})
                if choice.get("finish_reason"):
                    stop_reason = {
                        "tool_calls": "tool_use",
                        "length": "max_tokens",
                    }.get(choice["finish_reason"], "end_turn")
                text = delta.get("content")
                if text:
                    if text_block_index is None:
                        text_block_index = block_index
                        yield {
                            "type": "content_block_start",
                            "index": text_block_index,
                            "content_block": {"type": "text", "text": ""},
                        }
                        block_index += 1
                    yield {
                        "type": "content_block_delta",
                        "index": text_block_index,
                        "delta": {"type": "text_delta", "text": text},
                        **(
                            {"colibri": last_colibri}
                            if last_colibri is not None
                            else {}
                        ),
                    }
                for position, call in enumerate(delta.get("tool_calls", [])):
                    stop_reason = "tool_use"
                    function = call.get("function", {})
                    # Keyed by the call's index, not its id: a streamed call
                    # carries its id only on the opening delta, so keying by id
                    # opened a fresh content block for every argument fragment.
                    key = call.get("index", position)
                    if key not in tool_blocks:
                        tool_blocks[key] = block_index
                        yield {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": call.get("id", f"toolu_{uuid.uuid4().hex}"),
                                "name": function.get("name", "tool"),
                                "input": {},
                            },
                        }
                        block_index += 1
                    arguments = function.get("arguments", "")
                    if arguments:
                        yield {
                            "type": "content_block_delta",
                            "index": tool_blocks[key],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": arguments,
                            },
                            **(
                                {"colibri": last_colibri}
                                if last_colibri is not None
                                else {}
                            ),
                        }
            if usage is not None:
                output_tokens = usage.get("completion_tokens", output_tokens)

        return events()

    def retrieve_response(self, response_id: str) -> dict[str, Any]:
        with self._response_lock:
            record = self._response_records.get(response_id)
            if record is None:
                raise APIError(
                    404, f"response '{response_id}' was not found", "not_found_error"
                )
            self._response_records.move_to_end(response_id)
            return record[0]

    def delete_response(self, response_id: str) -> dict[str, Any]:
        with self._response_lock:
            if self._response_records.pop(response_id, None) is None:
                raise APIError(
                    404, f"response '{response_id}' was not found", "not_found_error"
                )
        return {"id": response_id, "object": "response.deleted", "deleted": True}

    def _response_messages_with_history(
        self, payload: Mapping[str, Any]
    ) -> list[dict[str, str]]:
        current = _response_messages(payload)
        previous_id = payload.get("previous_response_id")
        if previous_id is None:
            return current
        if not isinstance(previous_id, str):
            raise APIError(
                400,
                "previous_response_id must be text",
                parameter="previous_response_id",
            )
        with self._response_lock:
            record = self._response_records.get(previous_id)
            if record is None:
                raise APIError(
                    404, f"response '{previous_id}' was not found", "not_found_error"
                )
            history = [dict(message) for message in record[1]]
        if current and current[0]["role"] == "system":
            history = [message for message in history if message["role"] != "system"]
            history.insert(0, current.pop(0))
        history.extend(current)
        return history

    def _store_response(
        self,
        response: dict[str, Any],
        messages: list[dict[str, str]],
        result: GenerationResult,
    ) -> None:
        if not response["store"]:
            return
        history = [dict(message) for message in messages]
        history.append({"role": "assistant", "content": result.text})
        with self._response_lock:
            self._response_records[response["id"]] = (response, history)
            self._response_records.move_to_end(response["id"])
            while len(self._response_records) > 128:
                self._response_records.popitem(last=False)

    def _prepare_text(self, payload: Mapping[str, Any]) -> _TextRequest:
        self._validate_model(payload.get("model"))
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise APIError(400, "prompt must be text", parameter="prompt")
        if payload.get("logprobs") is not None:
            raise APIError(
                400, "logprobs are not implemented yet", parameter="logprobs"
            )
        requested_max = _integer_option(
            payload,
            "max_tokens",
            default=int(self.generation_defaults["max_new_tokens"]),
        )
        max_new_tokens = self._fit_max_new_tokens(
            requested_max,
            len(self.generator.tokenizer.encode(prompt)),
            parameter="max_tokens",
        )
        try:
            sampling = SamplingConfig(
                temperature=_float_option(
                    payload, "temperature", float(self.generation_defaults["temperature"])
                ),
                top_k=_integer_option(
                    payload, "top_k", default=int(self.generation_defaults["top_k"])
                ),
                top_p=_float_option(
                    payload, "top_p", float(self.generation_defaults["top_p"])
                ),
                seed=_optional_integer(payload, "seed"),
            )
        except ValueError as error:
            raise APIError(400, str(error)) from error
        return _TextRequest(prompt, max_new_tokens, sampling)

    def _prepare_chat(self, payload: Mapping[str, Any]) -> _GenerationRequest:
        messages, tools_enabled = _chat_messages(
            payload, architecture=getattr(self.generator.tokenizer, "architecture", None)
        )
        tools = tuple(_selected_tools(payload)) if tools_enabled else ()
        return self._prepare_generation(
            payload,
            messages,
            max_key="max_completion_tokens",
            fallback_max_key="max_tokens",
            tools_enabled=tools_enabled,
            tools=tools,
        )

    def _prepare_generation(
        self,
        payload: Mapping[str, Any],
        messages: list[dict[str, str]],
        *,
        max_key: str,
        fallback_max_key: str | None = None,
        tools_enabled: bool = False,
        tools: tuple[dict[str, Any], ...] = (),
    ) -> _GenerationRequest:
        self._validate_model(payload.get("model"))
        _reject_unsupported_generation_options(payload)
        requested_max = _integer_option(
            payload,
            max_key,
            fallback_key=fallback_max_key,
            default=int(self.generation_defaults["max_new_tokens"]),
        )
        try:
            sampling = SamplingConfig(
                temperature=_float_option(
                    payload, "temperature", float(self.generation_defaults["temperature"])
                ),
                top_k=_integer_option(
                    payload, "top_k", default=int(self.generation_defaults["top_k"])
                ),
                top_p=_float_option(
                    payload, "top_p", float(self.generation_defaults["top_p"])
                ),
                seed=_optional_integer(payload, "seed"),
            )
        except ValueError as error:
            raise APIError(400, str(error)) from error
        enable_thinking = _boolean_option(payload, "enable_thinking", None)
        separate_reasoning = _boolean_option(payload, "separate_reasoning", False)
        try:
            prepare_messages = getattr(self.generator, "prepare_messages", None)
            prompt_ids = tuple(
                prepare_messages(messages, enable_thinking=enable_thinking)
                if callable(prepare_messages)
                else self.generator.tokenizer.encode_messages(
                    messages, enable_thinking=enable_thinking
                )
            )
        except Exception as error:
            raise APIError(
                400,
                f"unable to tokenize the formatted prompt: {error}",
                parameter="messages",
            ) from error
        max_new_tokens = self._fit_max_new_tokens(
            requested_max, len(prompt_ids), parameter=max_key
        )
        return _GenerationRequest(
            messages,
            prompt_ids,
            max_new_tokens,
            sampling,
            enable_thinking,
            tools_enabled,
            tools,
            separate_reasoning,
            # Read off the prompt itself rather than inferred from the flag: a
            # template decides this, and it may reason by default with the flag
            # never set. Only the tail can carry the marker.
            self._prompt_opens_thinking(prompt_ids),
        )

    def _prompt_opens_thinking(self, prompt_ids: Sequence[int]) -> bool:
        tokenizer = self.generator.tokenizer
        decode = getattr(tokenizer, "decode", None)
        if decode is None or not prompt_ids:
            return False
        try:
            tail = decode(list(prompt_ids[-8:]), skip_special_tokens=False)
        except Exception:
            return False
        return tail.rstrip().endswith("<think>")

    def _fit_max_new_tokens(
        self, requested: int, prompt_tokens: int, *, parameter: str
    ) -> int:
        """Clamp requested output tokens to the service limit and context room.

        Agentic clients (Claude Code, Codex, ...) routinely request far more
        output tokens than a small local deployment allows, and treat that as
        an upper bound, not an exact demand. Clamp instead of rejecting so those
        requests still run; only error when the prompt alone leaves no room.
        """
        if requested <= 0:
            raise APIError(400, f"{parameter} must be positive", parameter=parameter)
        room = self.context_window - prompt_tokens
        if room <= 0:
            raise APIError(
                400,
                f"prompt has {prompt_tokens} tokens, filling the "
                f"{self.context_window}-token context window",
                parameter=parameter,
                code="context_length_exceeded",
            )
        return max(1, min(requested, self.max_new_tokens, room))

    def _validate_model(self, requested_model: Any) -> None:
        if (
            self.strict_model
            and requested_model is not None
            and requested_model != self.model_name
        ):
            raise APIError(
                404,
                f"model '{requested_model}' is not loaded",
                error_type="not_found_error",
                parameter="model",
            )

    def _chat_response(
        self,
        result: GenerationResult,
        *,
        tools: tuple[dict[str, Any], ...] = (),
    ) -> dict[str, Any]:
        visible, reasoning = _split_reasoning_content(result.text)
        content, tool_calls = _parse_tool_calls(visible, tools=tools)
        finish_reason = (
            "tool_calls"
            if tool_calls
            else ("stop" if result.stopped_on_eos else "length")
        )
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
            "refusal": None,
            "annotations": [],
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": _usage(len(result.prompt_ids), len(result.generated_ids)),
            "service_tier": "default",
            "system_fingerprint": None,
        }

    def _chat_chunk(
        self,
        completion_id: str,
        created: int,
        delta: Mapping[str, Any],
        *,
        finish_reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": dict(delta),
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "system_fingerprint": None,
        }

    def _completion_response(
        self, completion_id: str, result: GenerationResult
    ) -> dict[str, Any]:
        finish_reason = "stop" if result.stopped_on_eos else "length"
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "text": result.text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": _usage(len(result.prompt_ids), len(result.generated_ids)),
        }

    def _completion_chunk(
        self,
        completion_id: str,
        created: int,
        text: str,
        finish_reason: str | None,
    ) -> dict[str, Any]:
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": self.model_name,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }

    def _response_shell(
        self,
        payload: Mapping[str, Any],
        response_id: str,
        created_at: int,
        *,
        status: str,
    ) -> dict[str, Any]:
        return {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": status,
            "background": False,
            "error": None,
            "incomplete_details": None,
            "instructions": payload.get("instructions"),
            "max_output_tokens": payload.get("max_output_tokens"),
            "model": self.model_name,
            "output": [],
            "parallel_tool_calls": False,
            "previous_response_id": payload.get("previous_response_id"),
            "store": _boolean_option(payload, "store", True),
            "temperature": _float_option(
                payload, "temperature", float(self.generation_defaults["temperature"])
            ),
            "text": {"format": {"type": "text"}},
            "tool_choice": payload.get(
                "tool_choice", "auto" if payload.get("tools") else "none"
            ),
            "tools": payload.get("tools") or [],
            "top_p": _float_option(
                payload, "top_p", float(self.generation_defaults["top_p"])
            ),
            "truncation": "disabled",
            "usage": None,
        }

    def _response_object(
        self,
        payload: Mapping[str, Any],
        result: GenerationResult,
        *,
        response_id: str | None = None,
        message_id: str | None = None,
        created_at: int | None = None,
    ) -> dict[str, Any]:
        response_id = response_id or f"resp_{uuid.uuid4().hex}"
        message_id = message_id or f"msg_{uuid.uuid4().hex}"
        created_at = created_at or int(time.time())
        tools = _response_tools(payload)
        _, tool_calls = (
            _parse_tool_calls(result.text, tools=tools) if tools else (result.text, [])
        )
        incomplete = not result.stopped_on_eos and not tool_calls
        return {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "incomplete" if incomplete else "completed",
            "background": False,
            "error": None,
            "incomplete_details": (
                {"reason": "max_output_tokens"} if incomplete else None
            ),
            "instructions": payload.get("instructions"),
            "max_output_tokens": payload.get("max_output_tokens"),
            "model": self.model_name,
            "output": _response_output(result, message_id, tools=tools),
            "parallel_tool_calls": False,
            "previous_response_id": payload.get("previous_response_id"),
            "store": _boolean_option(payload, "store", True),
            "temperature": _float_option(
                payload, "temperature", float(self.generation_defaults["temperature"])
            ),
            "text": {"format": {"type": "text"}},
            "tool_choice": payload.get(
                "tool_choice", "auto" if payload.get("tools") else "none"
            ),
            "tools": payload.get("tools") or [],
            "top_p": _float_option(
                payload, "top_p", float(self.generation_defaults["top_p"])
            ),
            "truncation": "disabled",
            "usage": {
                "input_tokens": len(result.prompt_ids),
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": len(result.generated_ids),
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": len(result.prompt_ids) + len(result.generated_ids),
            },
        }


class ColibriHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler: type[BaseHTTPRequestHandler],
        bind_and_activate: bool = True,
        *,
        max_connections: int = 128,
    ):
        if max_connections <= 0:
            raise ValueError("max_connections must be positive")
        self._connection_slots = threading.BoundedSemaphore(max_connections)
        super().__init__(server_address, request_handler, bind_and_activate)

    def process_request(
        self,
        request: socket.socket | tuple[bytes, socket.socket],
        client_address: object,
    ) -> None:
        if not self._connection_slots.acquire(blocking=False):
            (request[1] if isinstance(request, tuple) else request).close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connection_slots.release()
            raise

    def process_request_thread(
        self,
        request: socket.socket | tuple[bytes, socket.socket],
        client_address: object,
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connection_slots.release()


_SSE_KEEPALIVE = object()


def _deferred_stream(
    factory: Callable[[], Iterator[dict[str, Any] | str]],
) -> Iterator[dict[str, Any] | str]:
    """Defer request preparation until after SSE headers are sent."""
    yield from factory()


def _sse_with_keepalive(
    events: Iterator[dict[str, Any] | str], interval: float
) -> Iterator[object]:
    """Yield the upstream events, injecting _SSE_KEEPALIVE while they stall.

    Evaluating a long prompt holds the generator for minutes before the first
    token, and pooled HTTP clients (Claude Code, Cline, ...) abandon a stream
    that produces no bytes for that long. Draining the upstream iterator on a
    worker thread lets the caller keep writing liveness markers meanwhile.
    """
    pending: queue.Queue[object] = queue.Queue(maxsize=1024)
    finished = object()
    stop = threading.Event()
    failure: list[BaseException] = []

    def pump() -> None:
        try:
            for event in events:
                while not stop.is_set():
                    try:
                        pending.put(event, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    break
        except BaseException as error:  # re-raised on the consuming thread
            failure.append(error)
        finally:
            # The pump is the only thread inside the generator frame, so it is
            # also the only one that may close it.
            close = getattr(events, "close", None)
            if close is not None:
                with contextlib.suppress(Exception):
                    close()
            with contextlib.suppress(queue.Full):
                pending.put_nowait(finished)

    worker = threading.Thread(target=pump, daemon=True)
    worker.start()
    try:
        while True:
            try:
                item = pending.get(timeout=interval)
            except queue.Empty:
                yield _SSE_KEEPALIVE
                continue
            if item is finished:
                break
            yield item
        if failure:
            raise failure[0]
    finally:
        stop.set()
        # Drain briefly so a pump blocked on a full queue notices `stop`. A
        # pump still inside a long native call cannot be interrupted; it is a
        # daemon thread, so leave it rather than block the connection.
        deadline = time.monotonic() + 1.0
        while worker.is_alive() and time.monotonic() < deadline:
            with contextlib.suppress(queue.Empty):
                pending.get_nowait()
            worker.join(timeout=0.05)


def create_handler(
    service: InferenceService,
) -> type[BaseHTTPRequestHandler]:
    class ColibriRequestHandler(BaseHTTPRequestHandler):
        server_version = "colibri-next/0.2"
        protocol_version = "HTTP/1.1"

        def setup(self) -> None:
            super().setup()
            # Waiting for a request is not the same as servicing one; see
            # parse_request, which switches to the shorter request timeout as
            # soon as a request line arrives.
            self.connection.settimeout(service.keepalive_timeout_seconds)

        def parse_request(self) -> bool:
            parsed = super().parse_request()
            self.connection.settimeout(service.request_timeout_seconds)
            return parsed

        def handle_one_request(self) -> None:
            super().handle_one_request()
            # Back to idle: this connection may now wait a long time for its
            # next request while another one occupies the runtime.
            with contextlib.suppress(OSError):
                self.connection.settimeout(service.keepalive_timeout_seconds)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._send_cors_headers()
            self.send_header(
                "Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"
            )
            self.send_header(
                "Access-Control-Allow-Headers",
                "Authorization, Content-Type, x-api-key, anthropic-version",
            )
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path in UI_ASSETS:
                self._send_static(path)
                return
            try:
                self._authenticate()
            except APIError as error:
                self._send_error(error)
                return
            if path == "/health":
                self._send_json(200, service.health())
                return
            if path == "/v1/me":
                self._send_json(200, service.me())
                return
            if path == "/v1/models":
                self._send_json(200, service.models())
                return
            if path == "/props":
                self._send_json(200, service.properties())
                return
            if path == "/slots":
                self._send_json(200, {"slots": service.slots()})
                return
            if path.startswith("/v1/responses/"):
                response_id = unquote(path[len("/v1/responses/") :])
                try:
                    self._send_json(200, service.retrieve_response(response_id))
                except OSError as error:
                    # Native runtime failures (e.g. C++ exceptions escaping
                    # ctypes) surface as OSError with a Windows error code.
                    # Log the detail so operators can diagnose the root cause.
                    self.log_error("native runtime error: %s", error)
                    self.close_connection = True
                except APIError as error:
                    self._send_error(error)
                return
            if path.startswith("/v1/models/"):
                model_name = unquote(path[len("/v1/models/") :])
                if service.strict_model and model_name != service.model_name:
                    self._send_error(
                        APIError(
                            404,
                            f"model '{model_name}' is not loaded",
                            "not_found_error",
                        )
                    )
                    return
                self._send_json(200, service.model())
                return
            self._send_error(APIError(404, "endpoint not found", "not_found_error"))

        def do_HEAD(self) -> None:
            path = urlsplit(self.path).path
            if path in UI_ASSETS:
                self._send_static(path, head_only=True)
                return
            try:
                self._authenticate()
                if path == "/v1/me":
                    self._send_json(200, service.me(), head_only=True)
                    return
                if path == "/health":
                    self._send_json(200, service.health(), head_only=True)
                    return
                if path == "/v1/models":
                    self._send_json(200, service.models(), head_only=True)
                    return
                self._send_error(APIError(404, "endpoint not found", "not_found_error"))
            except APIError as error:
                self._send_error(error)

        def do_DELETE(self) -> None:
            try:
                self._authenticate()
                path = urlsplit(self.path).path
                if not path.startswith("/v1/responses/"):
                    raise APIError(404, "endpoint not found", "not_found_error")
                response_id = unquote(path[len("/v1/responses/") :])
                self._send_json(200, service.delete_response(response_id))
            except APIError as error:
                self._send_error(error)

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            try:
                self._authenticate()
                if path == "/v1/me":
                    self._discard_request_body()
                    self._send_json(200, service.me())
                    self.log_message("request completed: %s", path)
                    return
                payload = self._read_json()
                self.log_message("request processing: %s", path)
                progress = self._prompt_progress(path)
                if path == "/v1/chat/completions":
                    if _boolean_option(payload, "stream", False):
                        self._send_sse(
                            _deferred_stream(
                                lambda: service.stream_chat_completion(
                                    payload, progress=progress
                                )
                            )
                        )
                    else:
                        self._send_json(
                            200, service.chat_completion(payload, progress=progress)
                        )
                        self.log_message("request completed: %s", path)
                    return
                if path == "/v1/completions":
                    if _boolean_option(payload, "stream", False):
                        self._send_sse(
                            _deferred_stream(
                                lambda: service.stream_completion(
                                    payload, progress=progress
                                )
                            )
                        )
                    else:
                        self._send_json(
                            200, service.completion(payload, progress=progress)
                        )
                        self.log_message("request completed: %s", path)
                    return
                if path == "/tokenize":
                    self._send_json(200, service.tokenize(payload))
                    self.log_message("request completed: %s", path)
                    return
                if path == "/detokenize":
                    self._send_json(200, service.detokenize(payload))
                    self.log_message("request completed: %s", path)
                    return
                if path == "/v1/messages/count_tokens":
                    self._send_json(200, service.count_anthropic_input(payload))
                    self.log_message("request completed: %s", path)
                    return
                if path == "/v1/messages":
                    if _boolean_option(payload, "stream", False):
                        self._send_sse(
                            _deferred_stream(
                                lambda: service.stream_anthropic_message(
                                    payload, progress=progress
                                )
                            ),
                            keepalive={"type": "ping"},
                        )
                    else:
                        self._send_json(
                            200,
                            service.anthropic_message(payload, progress=progress),
                        )
                        self.log_message("request completed: %s", path)
                    return
                if path == "/v1/responses/input_tokens":
                    self._send_json(200, service.count_response_input(payload))
                    self.log_message("request completed: %s", path)
                    return
                if path == "/v1/responses":
                    if _boolean_option(payload, "stream", False):
                        self._send_sse(
                            service.stream_response(payload, progress=progress)
                        )
                    else:
                        self._send_json(
                            200, service.response(payload, progress=progress)
                        )
                        self.log_message("request completed: %s", path)
                    return
                raise APIError(404, "endpoint not found", "not_found_error")
            except APIError as error:
                self._send_error(error)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception:
                # The client only ever sees "internal server error", so the
                # traceback has to reach the log or the failure is undebuggable.
                self.log_error("unhandled request error: %s", traceback.format_exc())
                self._send_error(APIError(500, "internal server error", "server_error"))

        def _authenticate(self) -> None:
            if service.api_key is None:
                return
            authorization = self.headers.get("Authorization", "")
            api_key = self.headers.get("x-api-key", "")
            expected = service.api_key
            valid_bearer = hmac.compare_digest(authorization, f"Bearer {expected}")
            valid_api_key = hmac.compare_digest(api_key, expected)
            if not (valid_bearer or valid_api_key):
                raise APIError(
                    401,
                    "invalid or missing API key",
                    error_type="authentication_error",
                )

        def _prompt_progress(self, path: str) -> Callable[[int, int], None]:
            label = f"[prefill {uuid.uuid4().hex[:8]}] {path}"
            last_seen = -1
            last_logged = -1
            started = time.perf_counter()
            last_logged_at = started
            first_report = True
            reused = 0

            def report(processed: int, total: int) -> None:
                nonlocal first_report, last_logged, last_logged_at, last_seen
                nonlocal reused, started
                if processed == last_seen:
                    return
                last_seen = processed
                now = time.perf_counter()
                if first_report:
                    first_report = False
                    reused = min(max(0, processed), total)
                    remaining = max(0, total - reused)
                    cached = f", {reused} cached/reused" if reused else ""
                    sys.stderr.write(
                        f"{label}: starting {total} prompt tokens"
                        f"{cached}, {remaining} to evaluate\n"
                    )
                    sys.stderr.flush()
                    # The clock is NOT restarted here, and that is deliberate.
                    #
                    # It used to be, on the reasoning that the first callback
                    # reports cache reuse, which costs nothing and should not
                    # drag the rate down. But a runtime that emits no
                    # incremental prefill events gets a compatibility fallback
                    # in v2_server, which synthesizes both callbacks back to
                    # back once the first token arrives. The clock then started
                    # and stopped microseconds apart and the rate came out as
                    # "1227 tokens at 48521041 tok/s" -- a number with no
                    # relationship to anything.
                    #
                    # `started` is set when this reporter is built, just after
                    # the request is read, so it brackets tokenization, prefill
                    # and (on the fallback path) the first token. That
                    # over-counts the setup slightly. Over-counting by a
                    # constant beats under-counting by three orders of
                    # magnitude, and unlike the reset it cannot produce a figure
                    # that is obviously false yet easy to quote.
                    last_logged = processed
                    last_logged_at = now
                    if processed < total:
                        return
                minimum_step = max(1, total // 20)
                if (
                    processed < total
                    and processed - last_logged < minimum_step
                    and now - last_logged_at < 1.0
                ):
                    return
                pct = (100 * processed / total) if total else 100
                elapsed = now - started
                evaluated = max(0, processed - reused)
                rate = (evaluated / elapsed) if elapsed > 0 else 0.0
                sys.stderr.write(
                    f"{label}: {processed}/{total} tokens "
                    f"({pct:5.1f}%) {rate:6.1f} tok/s"
                    "\n"
                )
                sys.stderr.flush()
                last_logged = processed
                last_logged_at = now
                if processed >= total:
                    sys.stderr.write(
                        f"{label}: ready in {elapsed:5.2f}s "
                        f"({evaluated} evaluated, {reused} reused, "
                        f"{rate:6.1f} tok/s)\n"
                    )
                    sys.stderr.flush()

            return report

        def _read_json(self) -> Mapping[str, Any]:
            content_type = self.headers.get_content_type()
            if content_type != "application/json":
                raise APIError(415, "content type must be application/json")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise APIError(400, "invalid Content-Length header") from error
            if length <= 0:
                raise APIError(400, "request body must not be empty")
            if length > MAX_REQUEST_BYTES:
                raise APIError(413, "request body is too large")
            try:
                payload = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise APIError(400, "request body must be valid JSON") from error
            if not isinstance(payload, dict):
                raise APIError(400, "request body must be a JSON object")
            return payload

        def _discard_request_body(self) -> None:
            """Consume an optional probe body before keeping the connection alive."""
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise APIError(400, "invalid Content-Length header") from error
            if length < 0 or length > MAX_REQUEST_BYTES:
                raise APIError(413, "request body is too large")
            if length:
                self.rfile.read(length)

        def _send_sse(
            self,
            events: Iterator[dict[str, Any] | str],
            *,
            keepalive: Mapping[str, Any] | None = None,
        ) -> None:
            # HTTP/1.1 chunked transfer-encoding rather than close-delimited: each
            # SSE event is a self-framed chunk, so the stream stays intact and the
            # keep-alive connection is cleanly reusable. Close-delimiting (Connection:
            # close) races with pooled HTTP clients (httpx/undici in Claude Code,
            # Cline, Codex): the next request reuses the socket across the close and
            # the leftover stream bytes corrupt it ("streams then garbled").
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")  # disable proxy buffering
            self._send_cors_headers()
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            stream_ok = True
            decoding_logged = False
            event_count = 0
            decode_tokens = 0
            # Span from the first token to the most recent one, as reported by
            # the generator. N tokens span N-1 intervals, so the rate below
            # divides by intervals rather than tokens -- the first token ends
            # prompt evaluation and is not part of decode. This has to match
            # formatGenerationMetrics() in ui/app.js, which the browser shows
            # for the very same generation.
            decode_elapsed = 0.0
            decode_phase: str | None = None
            last_stat = 0.0
            stream = _sse_with_keepalive(events, service.sse_keepalive_seconds)
            try:
                for event in stream:
                    if self._client_is_gone():
                        # Stop generating for a client that has gone. The
                        # `finally` below closes the wrapper, which closes the
                        # upstream generator and releases the model.
                        self.log_message(
                            "request abandoned by client: streaming events=%d",
                            event_count,
                        )
                        stream_ok = False
                        self.close_connection = True
                        break
                    if event is _SSE_KEEPALIVE:
                        self._write_sse_keepalive(keepalive)
                        continue
                    data = (
                        event
                        if isinstance(event, str)
                        else json.dumps(event, ensure_ascii=False)
                    )
                    event_count += 1
                    if not decoding_logged and _is_decode_event(event):
                        self.log_message("request decoding")
                        decoding_logged = True
                    if isinstance(event, dict):
                        colibri = event.get("colibri")
                        if isinstance(colibri, dict):
                            tokens = colibri.get("generated_tokens")
                            elapsed = colibri.get("decode_elapsed_seconds")
                            phase = colibri.get("phase")
                            if isinstance(tokens, int) and isinstance(elapsed, float):
                                decode_tokens = tokens
                                decode_elapsed = elapsed
                                if isinstance(phase, str):
                                    decode_phase = phase
                                now = time.perf_counter()
                                if now - last_stat >= 1.0:
                                    last_stat = now
                                    intervals = tokens - 1
                                    rate = (
                                        intervals / elapsed
                                        if intervals > 0 and elapsed > 0
                                        else 0.0
                                    )
                                    sys.stderr.write(
                                        f"\r[gen ] {tokens} tokens "
                                        f"{rate:6.1f} tok/s {elapsed:5.1f}s"
                                        + (
                                            " [building tool call]"
                                            if decode_phase == "tool_call"
                                            else ""
                                        )
                                    )
                                    sys.stderr.flush()
                    self._write_sse_event(
                        data, event if isinstance(event, dict) else None
                    )
                if decode_tokens:
                    # The last chunk's own span, not the wall time to the end of
                    # the stream: the tail after the final token is teardown, not
                    # decode, and counting it understated the rate.
                    intervals = decode_tokens - 1
                    rate = (
                        intervals / decode_elapsed
                        if intervals > 0 and decode_elapsed > 0
                        else 0.0
                    )
                    sys.stderr.write(
                        f"\n[gen ] done {decode_tokens} tokens "
                        f"in {decode_elapsed:5.2f}s ({rate:6.1f} tok/s)"
                        + (
                            " [tool call ready]"
                            if decode_phase == "tool_call"
                            else ""
                        )
                        + "\n"
                    )
                    sys.stderr.flush()
                self.log_message("request completed: streaming events=%d", event_count)
            except (BrokenPipeError, ConnectionResetError):
                stream_ok = False
                self.close_connection = True
            except OSError as error:
                # Native runtime failures (e.g. C++ exceptions escaping
                # ctypes) surface as OSError with a Windows error code.
                # Surface the detail so operators can diagnose it.
                self.log_error("native runtime stream error: %s", error)
                stream_ok = False
                self.close_connection = True
            except APIError as error:
                # Once SSE headers are sent, a JSON error response is no longer
                # possible. Send the error as the final SSE event instead.
                self.log_error("stream request failed: %s", error.message)
                try:
                    self._write_sse_event(json.dumps(self._error_body(error)))
                except (BrokenPipeError, ConnectionResetError):
                    stream_ok = False
                    self.close_connection = True
            except Exception as error:
                # This catches genuine Python bugs (AttributeError,
                # TypeError, ...) that occur while the iterator is being
                # consumed after the HTTP status line has already been
                # written. Keep clients such as Cline and OpenCode from
                # waiting on a silent/truncated stream.
                self.log_error("unhandled stream error: %s", error)
                try:
                    self._write_sse_event(
                        json.dumps(
                            self._error_body(
                                APIError(500, "internal server error", "server_error")
                            )
                        )
                    )
                except (BrokenPipeError, ConnectionResetError):
                    stream_ok = False
                    self.close_connection = True
            finally:
                # Closing the wrapper stops its worker, which owns closing the
                # upstream generator.
                stream.close()
                if stream_ok:
                    try:
                        self.wfile.write(b"0\r\n\r\n")  # terminating chunk
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        self.close_connection = True

        def _client_is_gone(self) -> bool:
            """Whether the peer has closed its end of a stream still in flight.

            A client that abandons a stream stops reading, but writes into it
            keep succeeding until the socket buffer fills -- so a long
            generation runs to completion for nobody. Watching the *read* side
            catches it: mid-response the client owes us no bytes, so a readable
            socket means either EOF or an early next request, and EOF is the
            one that says the response has no reader left.
            """
            connection = getattr(self, "connection", None)
            if connection is None:
                return False
            try:
                readable, _, _ = select.select([connection], [], [], 0)
                if not readable:
                    return False
                return connection.recv(1, socket.MSG_PEEK) == b""
            except (BlockingIOError, InterruptedError):
                return False
            except (OSError, ValueError):
                # A closed or unusable socket has no reader either way.
                return True

        def _write_sse_keepalive(self, event: Mapping[str, Any] | None) -> None:
            if event is not None:
                self._write_sse_event(json.dumps(event, ensure_ascii=False), event)
                return
            # An SSE comment: valid framing that every compliant client
            # ignores, so it is safe on the OpenAI-shaped endpoints where no
            # keepalive event type is defined.
            payload = b": keepalive\n\n"
            self.wfile.write(f"{len(payload):X}\r\n".encode("ascii"))
            self.wfile.write(payload)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

        def _write_sse_event(
            self, data: str, event: Mapping[str, Any] | None = None
        ) -> None:
            event_name = event.get("type") if event is not None else None
            prefix = f"event: {event_name}\n" if event_name else ""
            payload = f"{prefix}data: {data}\n\n".encode("utf-8")
            # One HTTP chunk per SSE event: "<hex length>\r\n<payload>\r\n".
            self.wfile.write(f"{len(payload):X}\r\n".encode("ascii"))
            self.wfile.write(payload)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

        def _send_cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", service.cors_origin)
            self.send_header("Vary", "Origin")

        def _send_error(self, error: APIError) -> None:
            self._send_json(error.status, self._error_body(error))

        def _error_body(self, error: APIError) -> dict[str, Any]:
            if urlsplit(self.path).path in {
                "/v1/messages",
                "/v1/messages/count_tokens",
            }:
                return {
                    "type": "error",
                    "error": {
                        "type": error.error_type,
                        "message": error.message,
                    },
                }
            return _error_payload(error)

        def _send_static(self, path: str, *, head_only: bool = False) -> None:
            filename, content_type = UI_ASSETS[path]
            try:
                body = (UI_DIRECTORY / filename).read_bytes()
            except OSError:
                self._send_error(APIError(404, "UI asset not found", "not_found_error"))
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header(
                "Content-Security-Policy",
                PREVIEW_CSP if filename == "preview.html" else UI_CSP,
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not head_only:
                self.wfile.write(body)

        def _send_json(
            self,
            status: int,
            payload: Mapping[str, Any],
            *,
            head_only: bool = False,
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_cors_headers()
            if status == 401:
                self.send_header("WWW-Authenticate", "Bearer")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not head_only:
                self.wfile.write(body)

    return ColibriRequestHandler


def serve(
    service: InferenceService,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    max_connections: int = 128,
) -> None:
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    server = ColibriHTTPServer(
        (host, port), create_handler(service), max_connections=max_connections
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


# Architectures whose own chat template renders tools: the schemas, the call
# markup and the result blocks. Anything else gets the generic Hermes prompt and
# our own rendering, which is what a checkpoint without tool support needs.
NATIVE_TOOL_ARCHITECTURES = ("deepseek4", "bailingmoe3")


def _template_tool_calls(
    tool_calls: Sequence[Mapping[str, Any]], architecture: str | None
) -> list[dict[str, Any]]:
    """Tool calls in the shape this architecture's template reads them.

    The wire format carries `arguments` as a JSON *string*. Most templates
    iterate it as a mapping (`arguments.items()`), which is a TypeError against
    a string; DeepSeek-V4's applies `from_json` and so wants the string kept.
    Decode for everyone except the one template that decodes for itself.
    """
    keep_encoded = architecture == "deepseek4"
    out: list[dict[str, Any]] = []
    for call in tool_calls:
        function = dict(call.get("function") or {})
        arguments = function.get("arguments")
        if isinstance(arguments, str) and not keep_encoded:
            try:
                function["arguments"] = json.loads(arguments)
            except json.JSONDecodeError:
                # A template that cannot iterate it would raise; an empty
                # mapping renders an argument-less call, which at least keeps
                # the turn structure the conversation depends on.
                function["arguments"] = {}
        out.append({**call, "function": function})
    return out


def _chat_messages(
    payload: Mapping[str, Any], *, architecture: str | None = None
) -> tuple[list[dict[str, Any]], bool]:
    value = payload.get("messages")
    if not isinstance(value, list) or not value:
        raise APIError(400, "messages must be a non-empty array", parameter="messages")
    tools = _selected_tools(payload)
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise APIError(
                400, f"messages[{index}] must be an object", parameter="messages"
            )
        role = message.get("role")
        if role not in VALID_ROLES:
            raise APIError(
                400, f"messages[{index}].role is invalid", parameter="messages"
            )
        if role in ("system", "developer"):
            system_text = _optional_text_content(message.get("content"), index)
            if system_text:
                system_parts.append(system_text)
            continue
        if role == "tool":
            # A tool may legitimately produce no output; keep the (possibly
            # empty) tool_response block so the turn structure is preserved.
            content = _optional_text_content(message.get("content"), index)
            messages.append(
                {"role": "tool", "content": content}
                if architecture in NATIVE_TOOL_ARCHITECTURES
                else {
                    "role": "user",
                    "content": f"<tool_response>\n{content}\n</tool_response>",
                }
            )
            continue
        content = _optional_text_content(message.get("content"), index)
        if role == "assistant" and message.get("tool_calls"):
            if architecture not in NATIVE_TOOL_ARCHITECTURES:
                content = _render_tool_calls(content, message["tool_calls"], index)
        if not content and not (role == "assistant" and message.get("tool_calls")):
            raise APIError(
                400, f"messages[{index}].content must be text", parameter="messages"
            )
        normalized: dict[str, Any] = {"role": role, "content": content}
        if (role == "assistant" and message.get("tool_calls")
                and architecture in NATIVE_TOOL_ARCHITECTURES):
            normalized["tool_calls"] = _template_tool_calls(
                message["tool_calls"], architecture
            )
        reasoning = message.get("reasoning_content")
        if role == "assistant" and reasoning is not None:
            if not isinstance(reasoning, str):
                raise APIError(
                    400,
                    f"messages[{index}].reasoning_content must be text",
                    parameter="messages",
                )
            normalized["reasoning_content"] = reasoning
        messages.append(normalized)
    if tools and architecture not in NATIVE_TOOL_ARCHITECTURES:
        system_parts.insert(0, _tool_prompt(tools, payload.get("tool_choice")))
    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    if tools and architecture in NATIVE_TOOL_ARCHITECTURES:
        # The checkpoint template owns the tool instructions and the schema
        # rendering, so the schemas ride on the first message rather than being
        # injected as a generic prompt.
        #
        # After the system turn is inserted, not before: attaching first put
        # them on what was the first message and the system turn then went in
        # ahead of it, so every request that carried a system prompt -- which
        # is every request a coding harness sends -- lost its tools entirely.
        if messages:
            messages[0]["tools"] = list(tools)
    # OpenAI-compatible clients may end a request with an assistant message
    # when they are asking the model to continue/prefill that turn. The chat
    # formatter adds the assistant generation marker after the supplied
    # history, so this is a valid prompt for the local runtime as well.
    # "tool" belongs here for an architecture whose template renders tool
    # results as their own turn. Elsewhere a tool result was rewritten into a
    # user turn above and never reached this check, which is why the list did
    # not match the message it raises.
    if not messages or messages[-1]["role"] not in ("user", "assistant", "tool"):
        raise APIError(
            400,
            "the last message must have role 'user', 'assistant', or 'tool'",
            parameter="messages",
        )
    return messages, bool(tools)


def _anthropic_to_chat_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("max_tokens") is None:
        raise APIError(400, "max_tokens is required", parameter="max_tokens")
    if payload.get("stop_sequences") not in (None, [], ""):
        raise APIError(
            400,
            "stop_sequences is not supported",
            parameter="stop_sequences",
        )
    thinking = payload.get("thinking")
    if thinking is not None and not isinstance(thinking, dict):
        raise APIError(400, "thinking must be an object", parameter="thinking")
    messages: list[dict[str, Any]] = []
    system = _anthropic_text_strict(payload.get("system"), "system")
    if system:
        messages.append({"role": "system", "content": system})
    value = payload.get("messages")
    if not isinstance(value, list) or not value:
        raise APIError(400, "messages must be a non-empty array", parameter="messages")
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise APIError(400, f"messages[{index}] is invalid", parameter="messages")
        role = message.get("role")
        if role in {"system", "developer"}:
            system_text = _anthropic_text_strict(
                message.get("content"), f"messages[{index}].content"
            )
            if system_text:
                messages.append({"role": "system", "content": system_text})
            continue
        if role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "content": _anthropic_text_strict(
                        message.get("content"), f"messages[{index}].content"
                    ),
                }
            )
            continue
        if role not in {"user", "assistant"}:
            raise APIError(400, f"messages[{index}] is invalid", parameter="messages")
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise APIError(
                400, f"messages[{index}].content is invalid", parameter="messages"
            )
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, str]] = []
        for block in content:
            if not isinstance(block, dict):
                raise APIError(
                    400, f"messages[{index}].content is invalid", parameter="messages"
                )
            block_type = block.get("type")
            if block_type in {"text", "input_text", "output_text"}:
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            elif block_type == "tool_use" and role == "assistant":
                tool_calls.append(
                    {
                        "id": block.get("id", f"call_{uuid.uuid4().hex}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", "tool"),
                            "arguments": json.dumps(
                                block.get("input", {}), ensure_ascii=False
                            ),
                        },
                    }
                )
            elif block_type == "tool_result" and role == "user":
                tool_results.append(
                    {
                        "role": "tool",
                        "content": _anthropic_text_strict(
                            block.get("content"),
                            f"messages[{index}].content",
                        ),
                    }
                )
            else:
                raise APIError(
                    400,
                    f"messages[{index}].content contains an unsupported block",
                    parameter="messages",
                )
        # A tool result answers the preceding assistant turn, so it must reach
        # the prompt before any new user text carried by the same message.
        messages.extend(tool_results)
        if text_parts or tool_calls:
            item: dict[str, Any] = {
                "role": role,
                "content": "".join(text_parts) or None,
            }
            if tool_calls:
                item["tool_calls"] = tool_calls
            messages.append(item)
        elif not tool_results:
            messages.append({"role": role, "content": ""})
    result: dict[str, Any] = {
        "model": payload.get("model"),
        "messages": messages,
        "max_completion_tokens": payload.get("max_tokens"),
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "top_k": payload.get("top_k"),
        # Absent means unstated, which leaves the checkpoint's own default in
        # place. Answering "false" here turned reasoning off for every Claude
        # Code request, since that client does not send a thinking block.
        "enable_thinking": (
            None if thinking is None
            else bool(isinstance(thinking, dict) and thinking.get("type") == "enabled")
        ),
    }
    tools = payload.get("tools") or []
    if not isinstance(tools, list):
        raise APIError(400, "tools must be an array", parameter="tools")
    converted_tools: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise APIError(400, f"tools[{index}] is invalid", parameter="tools")
        # Anthropic's canonical tool format has name/description/input_schema
        # directly on the object; OpenAI-compatible clients may wrap it as a
        # function tool. Accept both representations.
        if isinstance(tool.get("function"), dict):
            function = tool["function"]
            name = function.get("name")
            parameters = function.get("parameters", {})
            description = function.get("description", "")
        else:
            name = tool.get("name")
            parameters = tool.get("input_schema", {})
            description = tool.get("description", "")
        if not isinstance(name, str) or not name or not isinstance(parameters, dict):
            raise APIError(400, f"tools[{index}] is invalid", parameter="tools")
        converted_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    result["tools"] = converted_tools
    choice = payload.get("tool_choice")
    if isinstance(choice, dict):
        choice_type = choice.get("type")
        if choice_type == "any":
            result["tool_choice"] = "required"
        elif choice_type == "none":
            result["tool_choice"] = "none"
        elif choice_type == "tool":
            result["tool_choice"] = {"function": {"name": choice.get("name")}}
        elif choice_type == "auto":
            result["tool_choice"] = "auto"
        else:
            raise APIError(400, "tool_choice is invalid", parameter="tool_choice")
    elif choice is not None:
        result["tool_choice"] = choice
    return result


def _anthropic_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                parts.append(_anthropic_text(block.get("content")))
        return "".join(parts)
    return ""


def _anthropic_text_strict(value: Any, parameter: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise APIError(400, f"{parameter} must be text blocks", parameter=parameter)
    parts: list[str] = []
    for block in value:
        if (
            not isinstance(block, dict)
            or block.get("type") not in {"text", "input_text", "output_text"}
            or not isinstance(block.get("text"), str)
        ):
            raise APIError(
                400,
                f"{parameter} contains an unsupported content block",
                parameter=parameter,
            )
        parts.append(block["text"])
    return "".join(parts)


def _selected_tools(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("tools") or []
    if not isinstance(value, list):
        raise APIError(400, "tools must be an array", parameter="tools")
    tools: list[dict[str, Any]] = []
    for index, tool in enumerate(value):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise APIError(400, f"tools[{index}] must be a function", parameter="tools")
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise APIError(
                400, f"tools[{index}].function is invalid", parameter="tools"
            )
        tools.append(tool)
    choice = payload.get("tool_choice", "auto" if tools else "none")
    if choice == "none":
        return []
    if choice in ("auto", "required"):
        return tools
    if isinstance(choice, dict):
        function = choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        selected = [tool for tool in tools if tool["function"]["name"] == name]
        if not selected:
            raise APIError(
                400, "tool_choice names an unknown tool", parameter="tool_choice"
            )
        return selected
    raise APIError(400, "tool_choice is invalid", parameter="tool_choice")


def _tool_prompt(tools: list[dict[str, Any]], tool_choice: Any) -> str:
    serialized = "\n".join(
        json.dumps(tool, ensure_ascii=False, separators=(",", ":")) for tool in tools
    )
    requirement = (
        "\n\nYou must call one of the available functions."
        if tool_choice == "required" or isinstance(tool_choice, dict)
        else ""
    )
    names = {
        tool.get("function", {}).get("name")
        for tool in tools
        if isinstance(tool.get("function"), dict)
    }
    edit_guidance = (
        "\n\nFor the Edit tool, copy old_string byte-for-byte from the latest "
        "Read result, including indentation and line endings. If an edit "
        "fails, Read the target again and do not repeat identical arguments."
        if "Edit" in names
        else ""
    )
    return (
        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
        f"{serialized}\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format "
        "with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
        "<parameter=example_parameter>\nvalue\n</parameter>\n</function>\n"
        "</tool_call>\n\nRequired parameters MUST be specified. You may provide "
        "reasoning before the tool call, but nothing after it."
        f"{edit_guidance}"
        f"{requirement}"
    )


def _render_tool_calls(content: str, value: Any, message_index: int) -> str:
    if not isinstance(value, list):
        raise APIError(400, "tool_calls must be an array", parameter="messages")
    rendered: list[str] = [content] if content else []
    for call in value:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise APIError(
                400,
                f"messages[{message_index}].tool_calls is invalid",
                parameter="messages",
            )
        function = call["function"]
        name = function.get("name")
        arguments = function.get("arguments", "{}")
        if not isinstance(name, str) or not isinstance(arguments, str):
            raise APIError(400, "tool call function is invalid", parameter="messages")
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise APIError(
                400, "tool call arguments must be JSON", parameter="messages"
            ) from error
        if not isinstance(parsed, dict):
            raise APIError(
                400, "tool call arguments must be an object", parameter="messages"
            )
        parameters = "".join(
            f"<parameter={key}>\n{_tool_value(value)}\n</parameter>\n"
            for key, value in parsed.items()
        )
        rendered.append(
            f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>"
        )
    return "\n\n".join(rendered)


def _tool_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse_tool_calls(
    text: str,
    *,
    tools: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
) -> tuple[str | None, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    # Each block is carried with the text that follows it, up to the next
    # <tool_call>, so parameters the model emitted *after* closing the block
    # can be recovered when -- and only when -- the call turns out incomplete.
    # See _recover_trailing_parameters for why they land there.
    decoded: list[tuple[str | None, dict[str, Any], str]] = []
    blocks = list(TOOL_CALL_BLOCK_PATTERN.finditer(text))
    for index, match in enumerate(blocks):
        following = (
            text[match.end() : blocks[index + 1].start()]
            if index + 1 < len(blocks)
            else text[match.end() :]
        )
        name, arguments = _decode_tool_call_body(match.group(1))
        decoded.append((name, arguments, following))
    for block in DSML_TOOL_CALL_BLOCK_PATTERN.finditer(text):
        for invoke in DSML_INVOKE_PATTERN.finditer(block.group(1)):
            arguments: dict[str, Any] = {}
            for parameter in DSML_PARAMETER_PATTERN.finditer(invoke.group(2)):
                key, string_flag, raw = parameter.groups()
                if string_flag == "true":
                    arguments[key] = raw
                else:
                    try:
                        arguments[key] = json.loads(raw.strip())
                    except json.JSONDecodeError:
                        arguments[key] = raw
            decoded.append((invoke.group(1), arguments, ""))
    for name, arguments, following in decoded:
        if name is None:
            continue
        schema = _tool_argument_schema(tools, name)
        if schema is not None:
            arguments = _normalize_schema_value(arguments, schema)
            required = schema.get("required")

            def _missing(values: dict[str, Any]) -> bool:
                return isinstance(required, list) and any(
                    isinstance(key, str) and key not in values for key in required
                )

            if _missing(arguments) and following:
                # Only now, with the schema in hand, is it clear the block came
                # out short -- so a well-formed call never has trailing text
                # folded into it.
                recovered = _normalize_schema_value(
                    _recover_trailing_parameters(arguments, following), schema
                )
                if not _missing(recovered):
                    arguments = recovered
            if _missing(arguments):
                # Do not hand an incomplete call to the client/tool runner.
                # Models sometimes close a tool block before emitting all
                # parameters; exposing it produces confusing downstream schema
                # errors (for example write missing content and filePath).
                continue
        elif isinstance(arguments, dict):
            # Undeclared tool: no required list to test against, so the only
            # safe signal that the block came out short is that it decoded
            # nothing at all. A well-formed call always has its arguments.
            if not arguments and following:
                arguments = _recover_trailing_parameters(arguments, following)
            # Nothing describes these values, so fall back to inferring each.
            arguments = {
                key: _infer_tool_value(item) for key, item in arguments.items()
            }
        encoded_arguments = json.dumps(arguments, ensure_ascii=False)
        signature = (name, encoded_arguments)
        if signature in seen:
            continue
        seen.add(signature)
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {"name": name, "arguments": encoded_arguments},
            }
        )
    # Content is whatever precedes the first tool-call marker. Bound it on the
    # literal marker rather than on successfully-parsed blocks so a truncated
    # (no closing </tool_call>) or malformed tool call never leaks its raw
    # markup into assistant content -- the caller sees empty content plus a
    # "length"/"stop" finish reason instead of a wall of <tool_call> tags.
    markers = [
        position for position in (
            text.find(TOOL_CALL_MARKER), text.find(DSML_TOOL_CALL_MARKER)
        ) if position != -1
    ]
    marker = min(markers) if markers else -1
    content = (text[:marker] if marker != -1 else text).strip()
    return (content or None), calls


def _split_muse_channels(text: str) -> tuple[str, str | None] | None:
    """Separate Muse Glimmer's recipient-tagged messages.

    An assistant turn is one or more messages, each ``to=<recipient>`` and
    terminated by ``<|eom|>`` (more follow) or ``<|eot|>`` (turn over).
    Messages addressed to ``self`` are chain-of-thought; everything else --
    ``to=user``, or a tool recipient carrying ATEM markup -- is the visible
    turn. Returns None when the text carries no such markup, so the caller can
    fall through to the ``<think>`` convention.
    """
    if "<|message|>" not in text:
        return None
    reasoning: list[str] = []
    visible: list[str] = []
    for header in MUSE_MESSAGE_HEADER.finditer(text):
        start = header.end()
        terminator = MUSE_MESSAGE_END.search(text, start)
        body = text[start : terminator.start()] if terminator else text[start:]
        # An absent recipient is the final answer; only "self" is hidden.
        target = reasoning if header.group("recipient") == "self" else visible
        target.append(body)
    if not reasoning and not visible:
        return None
    return "".join(visible).strip(), "\n\n".join(reasoning).strip() or None


def _reasoning_delta_field(separate_reasoning: bool) -> str:
    """Where a streamed reasoning delta belongs: never in the answer.

    This used to forward reasoning as ordinary content so that a client which
    ignores `reasoning_content` still saw a live stream rather than a long
    silence. That trade is wrong for a model asked to DO something: told to
    write a file, the model drafts it while thinking, and streaming the draft
    as content made a coding harness render the file instead of writing it --
    the tool call, when it came, arrived after the answer had already been
    shown.

    Reasoning is the model's notes, content is its answer, and the two are not
    interchangeable at any level of client sophistication. `separate_reasoning`
    is kept for callers that pass it, but both branches now agree that notes
    are not the answer.

    Either way the protocol framing is stripped: `<|message|>` and friends are
    never content by any reading.
    """
    return "reasoning_content"


def _channel_delta(
    channels: "MuseChannelStream | ThinkingPrefixStream | None",
    text_delta: str | None,
    *,
    separate_reasoning: bool = False,
) -> dict[str, str]:
    """Shape one streaming delta, separating reasoning from the answer.

    The splitter is whichever one the turn needs -- Muse Glimmer's channels or
    a thinking block the prompt opened -- and a turn that has neither is
    forwarded untouched, so every other architecture streams exactly as before.
    """
    if not text_delta:
        return {}
    if channels is None:
        return {"content": text_delta}
    visible, reasoning = channels.feed(text_delta)
    delta: dict[str, str] = {}
    if reasoning:
        field = _reasoning_delta_field(separate_reasoning)
        delta[field] = reasoning
    if visible:
        delta["content"] = delta.get("content", "") + visible
    return delta


def _without_partial_marker(text: str) -> str:
    """Drop a trailing run that could still grow into a message marker."""
    for size in range(min(len(text), _MUSE_LONGEST_MARKER - 1), 0, -1):
        tail = text[-size:]
        if any(marker.startswith(tail) for marker in _MUSE_MARKERS):
            return text[:-size]
    return text


class ThinkingPrefixStream:
    """Splits a turn whose reasoning block was opened by the PROMPT.

    A reasoning checkpoint is asked to think by ending the prompt with an open
    ``<think>``, so the model emits its reasoning and then only the CLOSING
    tag. Streaming that raw sends the private reasoning to the client as the
    assistant's answer -- which reads as the model restating its plan before
    doing it, then doing it.

    Everything up to ``</think>`` is reasoning and everything after is visible.
    The tag is withheld while it may still be arriving character by character,
    for the same reason the Muse stream withholds its markers.
    """

    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._closed = False

    def feed(self, delta: str) -> tuple[str, str]:
        """Return (visible, reasoning) newly settled by this delta."""
        if self._closed:
            return delta, ""
        self._buffer += delta
        end = self._buffer.find(self._CLOSE)
        if end == -1:
            # Hold back a tail that could still become the closing tag.
            settled = self._buffer
            for size in range(min(len(settled), len(self._CLOSE) - 1), 0, -1):
                if self._CLOSE.startswith(settled[-size:]):
                    settled = settled[:-size]
                    break
            self._buffer = self._buffer[len(settled):]
            return "", settled
        reasoning = self._buffer[:end]
        visible = self._buffer[end + len(self._CLOSE):]
        self._buffer = ""
        self._closed = True
        return visible.lstrip(), reasoning

    def flush(self) -> tuple[str, str]:
        """Release whatever was being withheld as a possible closing tag."""
        if self._closed or not self._buffer:
            return "", ""
        held, self._buffer = self._buffer, ""
        return "", held


class MuseChannelStream:
    """Incremental form of :func:`_split_muse_channels` for streaming.

    Muse Glimmer has no setting that stops it reasoning, so a stream that
    forwarded raw text would show every client the chain-of-thought and the
    <|eom|>/<|start|> framing around it. This re-splits the accumulated text on
    each delta and returns only what is newly settled, which keeps one
    implementation of the channel rules rather than a second incremental one.

    Text is emitted only once it cannot still turn out to be the start of a
    marker, so a ``<|eom|>`` arriving one character at a time never leaks a
    stray ``<|`` into the output. What has already been emitted is tracked by
    length, which means emission has to be monotonic -- nothing may be sent
    that a later delta would retract.
    """

    def __init__(self) -> None:
        self._raw = ""
        self._visible = 0
        self._reasoning = 0
        self._saw_header = False

    def feed(self, delta: str) -> tuple[str, str]:
        """Return (visible, reasoning) text newly settled by this delta."""
        self._raw += delta
        return self._advance(final=False)

    def flush(self) -> tuple[str, str]:
        """Release the tail that was being withheld as a possible marker.

        A turn that never produced a header carries no channels to separate --
        it is forwarded whole rather than swallowed.
        """
        visible, reasoning = self._advance(final=True)
        if not self._saw_header and not self._visible and not self._reasoning:
            return self._raw, ""
        return visible, reasoning

    def _advance(self, *, final: bool) -> tuple[str, str]:
        text = self._raw
        visible: list[str] = []
        reasoning: list[str] = []
        for header in MUSE_MESSAGE_HEADER.finditer(text):
            self._saw_header = True
            start = header.end()
            terminator = MUSE_MESSAGE_END.search(text, start)
            if terminator:
                body = text[start : terminator.start()]
            else:
                # The open message: its tail may still be a partial marker.
                body = text[start:]
                if not final:
                    body = _without_partial_marker(body)
            (reasoning if header.group("recipient") == "self" else visible).append(body)
        # Joined the way the batch splitter joins them, so a stream and a
        # non-streamed response of the same turn read identically.
        joined_visible = "".join(visible)
        joined_reasoning = "\n\n".join(reasoning)
        fresh_visible = joined_visible[self._visible :]
        fresh_reasoning = joined_reasoning[self._reasoning :]
        self._visible = len(joined_visible)
        self._reasoning = len(joined_reasoning)
        return fresh_visible, fresh_reasoning


def _split_reasoning_content(text: str) -> tuple[str, str | None]:
    """Separate DeepSeek's leading thinking block from visible assistant text."""
    muse = _split_muse_channels(text)
    if muse is not None:
        return muse
    match = THINKING_BLOCK_PATTERN.match(text)
    if match:
        return text[match.end():], match.group(1)
    # A prompt that ends with an open <think> -- which is how every reasoning
    # checkpoint here asks for reasoning -- means the model's output carries
    # only the CLOSING tag: the thinking is everything before it. Left in
    # place, a turn's private reasoning was served as its visible answer,
    # restating the plan before doing it and reading exactly like repetition.
    #
    # A closing tag with nothing before it is the same case with no reasoning
    # produced, which is what a non-thinking DeepSeek prompt emits.
    closing = text.find("</think>")
    if closing != -1 and "<think>" not in text[:closing]:
        return text[closing + len("</think>"):].lstrip(), text[:closing].strip() or None
    return text, None


def _has_complete_tool_call(text: str, *, tools: Sequence[dict[str, Any]]) -> bool:
    """Return true only for closed native markup that parses as a tool call."""
    if TOOL_CALL_END_MARKER not in text and DSML_TOOL_CALL_END_MARKER not in text:
        return False
    _, calls = _parse_tool_calls(text, tools=list(tools))
    return bool(calls)


def _generation_result(step: GenerationStep) -> GenerationResult:
    return GenerationResult(
        prompt_ids=tuple(step.prompt_ids),
        generated_ids=tuple(step.generated_ids),
        text=step.text,
        stopped_on_eos=step.stopped_on_eos,
        state_tokens=step.state_tokens,
    )


def _tool_argument_schema(
    tools: tuple[dict[str, Any], ...] | list[dict[str, Any]], name: str
) -> dict[str, Any] | None:
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, dict) or function.get("name") != name:
            continue
        parameters = function.get("parameters")
        return parameters if isinstance(parameters, dict) else None
    return None


def _normalize_schema_value(value: Any, schema: Mapping[str, Any]) -> Any:
    """Normalize an inferred tool value using its JSON schema.

    Hermes parameters carry only text, so ``<parameter=taskId>1`` is
    ambiguous.  Parsing that text as JSON produces an integer even when the
    client declared ``taskId`` as a string.  The declared schema is the source
    of truth; recursively restore unambiguous scalar types before returning the
    tool call to the client.
    """
    schema_type = schema.get("type")
    allowed_types = (
        tuple(item for item in schema_type if isinstance(item, str))
        if isinstance(schema_type, list)
        else ((schema_type,) if isinstance(schema_type, str) else ())
    )
    if not allowed_types:
        variants = schema.get("anyOf", schema.get("oneOf"))
        if isinstance(variants, list):
            candidates = [item for item in variants if isinstance(item, dict)]
            for candidate in candidates:
                candidate_type = candidate.get("type")
                if (
                    (candidate_type == "string" and not isinstance(value, (dict, list)))
                    or (candidate_type == "object" and isinstance(value, dict))
                    or (candidate_type == "array" and isinstance(value, list))
                ):
                    return _normalize_schema_value(value, candidate)
        enum = schema.get("enum")
        if (
            isinstance(enum, list)
            and enum
            and all(isinstance(item, str) for item in enum)
        ):
            allowed_types = ("string",)

    if value is None and "null" in allowed_types:
        return None

    if "string" in allowed_types and not isinstance(value, (dict, list)):
        if isinstance(value, str):
            return value
        if value is True:
            return "true"
        if value is False:
            return "false"
        if value is None:
            return "null"
        return str(value)

    # Hermes parameters arrive as text, so a declared object/array has to be
    # decoded here -- after the string branch above, which must keep its text.
    if isinstance(value, str) and (
        "object" in allowed_types or "array" in allowed_types
    ):
        value = _infer_tool_value(value)

    if "object" in allowed_types and isinstance(value, dict):
        properties = schema.get("properties")
        property_schemas = properties if isinstance(properties, dict) else {}
        additional = schema.get("additionalProperties")
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            item_schema = property_schemas.get(key)
            if not isinstance(item_schema, dict) and isinstance(additional, dict):
                item_schema = additional
            normalized[key] = (
                _normalize_schema_value(item, item_schema)
                if isinstance(item_schema, dict)
                else _infer_tool_value(item)
            )
        return normalized

    if "array" in allowed_types and isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            return [_normalize_schema_value(item, items) for item in value]
        return value

    if isinstance(value, str):
        if "integer" in allowed_types and re.fullmatch(r"[+-]?\d+", value):
            return int(value)
        if "number" in allowed_types:
            try:
                return float(value)
            except ValueError:
                pass
        if "boolean" in allowed_types and value.lower() in {"true", "false"}:
            return value.lower() == "true"
    return value


def _trim_parameter_text(raw: str) -> str:
    """Drop only the framing newlines around a Hermes parameter value.

    _tool_prompt() shows the model ``<parameter=name>\\nvalue\\n</parameter>``,
    so exactly one newline per side is layout. Stripping all whitespace instead
    removed the first line's indentation, so every Edit against indented code
    failed its exact-match check and the model fell back to shell edits.
    """
    raw = TOOL_PARAMETER_LEAD.sub("", raw, count=1)
    return TOOL_PARAMETER_TAIL.sub("", raw, count=1)


def _infer_tool_value(value: Any) -> Any:
    """Type a parameter the tool schema does not describe."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value.strip())
    except json.JSONDecodeError:
        return value


class _ToolCallStreamer:
    """Turn an accumulating ``<tool_call>`` body into incremental JSON deltas.

    Withholding the whole call until ``</tool_call>`` arrives means a client
    receives no content for as long as the call takes to write -- minutes, for a
    tool whose argument is a file. Clients time such a stream out ("no chunks
    received") no matter how many SSE pings it carries, because a ping is not
    content. This emits the tool name as soon as it is decodable and then the
    arguments as JSON fragments, so the wire carries real progress.

    Only the two tag-shaped bodies stream: their values have an unambiguous
    close, so a value can be emitted before the call ends. A JSON body is
    unparseable until its last brace, so it is emitted whole at the end.

    The fragments concatenate into exactly ``json.dumps(arguments)`` for the
    arguments the authoritative parse produces -- ``finish()`` reconciles
    against it, so a client never assembles something the parser would reject.
    """

    # A value may be emitted only up to the point where no suffix of what is
    # held back could still turn out to be the closing tag or the framing
    # newline that _trim_parameter_text() removes.
    _FORMATS = (
        ("<arg_value>", "</arg_value>"),
        (None, "</parameter>"),
    )

    def __init__(self, schema: Mapping[str, Any] | None = None) -> None:
        self._schema = schema
        self._name: str | None = None
        self._opened = False
        self._done: dict[str, Any] = {}
        self._streaming_key: str | None = None
        self._streamed = ""  # raw value text already emitted for _streaming_key
        self._finished = False

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def started(self) -> bool:
        return self._name is not None

    def bind_schema(self, schema: Mapping[str, Any] | None) -> None:
        """Attach the declared schema, once the name has identified the tool.

        Nothing may have been emitted yet: the schema decides both which values
        stream early and how each one is typed.
        """
        if self._opened:
            raise RuntimeError("the tool call is already being streamed")
        self._schema = schema

    def _property_schema(self, key: str) -> Mapping[str, Any] | None:
        if not isinstance(self._schema, Mapping):
            return None
        properties = self._schema.get("properties")
        if not isinstance(properties, Mapping):
            return None
        candidate = properties.get(key)
        return candidate if isinstance(candidate, Mapping) else None

    def _streams_as_string(self, key: str) -> bool:
        """Whether `key` may be emitted before its value is complete.

        Only strings: any other type is decided by _normalize_schema_value once
        the whole text is known, and a partial number or object cannot be walked
        back after it is on the wire.
        """
        schema = self._property_schema(key)
        if schema is None:
            return False
        declared = schema.get("type")
        if isinstance(declared, list):
            return declared == ["string"] or declared == ["string", "null"]
        return declared == "string"

    def _encode(self, key: str, value: Any) -> str:
        schema = self._property_schema(key)
        if schema is not None:
            value = _normalize_schema_value(value, schema)
        elif self._schema is None:
            value = _infer_tool_value(value)
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _escape(text: str) -> str:
        """Escape `text` as JSON string content, without the framing quotes."""
        return json.dumps(text, ensure_ascii=False)[1:-1]

    def _separator(self) -> str:
        if not self._opened:
            self._opened = True
            return "{"
        return ","

    def feed(self, body: str) -> list[str]:
        """Emit fragments for everything `body` newly settles. Never blocks."""
        if self._finished:
            return []
        fragments: list[str] = []
        if self._name is None:
            self._name = _decode_tool_call_name(body)
            # Emit nothing on the call that discovers the name: the caller has
            # to bind the schema first, and the schema decides how the very
            # first value is typed and whether it may stream at all.
            return fragments
        for key, value, closed in _iter_tool_call_arguments(body):
            if key in self._done:
                continue
            if closed:
                if self._streaming_key == key:
                    # Close the string this value was being streamed into. The
                    # tail trim only applies once the value is known to be over.
                    remainder = _trim_parameter_text(value)[len(self._streamed):]
                    fragments.append(self._escape(remainder) + '"')
                    self._streaming_key = None
                    self._streamed = ""
                else:
                    fragments.append(
                        self._separator() + json.dumps(key, ensure_ascii=False)
                        + ":" + self._encode(key, _trim_parameter_text(value))
                    )
                self._done[key] = True
                continue
            # An open value: stream it only if its declared type makes the text
            # final, and only the part no closing tag could still claim.
            if not self._streams_as_string(key):
                break
            if self._streaming_key != key:
                if self._streaming_key is not None:
                    break
                fragments.append(
                    self._separator() + json.dumps(key, ensure_ascii=False) + ':"'
                )
                self._streaming_key = key
                self._streamed = ""
            settled = _settled_prefix(_trim_parameter_text(value))
            if len(settled) > len(self._streamed):
                fragments.append(self._escape(settled[len(self._streamed):]))
                self._streamed = settled
            break
        return fragments

    def finish(self, arguments: Mapping[str, Any] | None) -> list[str]:
        """Emit whatever `arguments` still owes the client, and close the object.

        `arguments` is the authoritative parse. Anything streamed already is a
        prefix of it; anything left is emitted here.
        """
        if self._finished:
            return []
        self._finished = True
        fragments: list[str] = []
        if arguments is None:
            # The call did not survive parsing. Close what is already on the
            # wire so the client sees valid JSON rather than a truncated object.
            if self._streaming_key is not None:
                fragments.append('"')
            if self._opened:
                fragments.append("}")
            return fragments
        for key, value in arguments.items():
            encoded = json.dumps(value, ensure_ascii=False)
            if self._streaming_key == key:
                # Close the streamed string against the authoritative value
                # rather than the raw text, so normalization still decides it.
                streamed = self._escape(self._streamed)
                tail = encoded[1:-1] if encoded.startswith('"') else ""
                fragments.append(
                    (tail[len(streamed):] if tail.startswith(streamed) else "") + '"'
                )
                self._streaming_key = None
            elif key not in self._done:
                fragments.append(
                    self._separator() + json.dumps(key, ensure_ascii=False)
                    + ":" + encoded
                )
            self._done[key] = True
        if self._streaming_key is not None:
            fragments.append('"')
        if not self._opened:
            self._opened = True
            fragments.append("{")
        fragments.append("}")
        return fragments


def _settled_prefix(value: str) -> str:
    """The part of an open tag value no closing tag or framing could reclaim.

    Holds back the longest suffix that is a prefix of any closing tag, plus a
    trailing newline run, which _trim_parameter_text() drops if the value ends
    there.
    """
    limit = len(value)
    for open_tag, close_tag in _ToolCallStreamer._FORMATS:
        del open_tag
        for length in range(min(len(close_tag) - 1, len(value)), 0, -1):
            if value.endswith(close_tag[:length]):
                limit = min(limit, len(value) - length)
                break
    settled = value[:limit]
    trailing = TOOL_PARAMETER_TAIL.search(settled)
    return settled[: trailing.start()] if trailing else settled


def _decode_tool_call_name(body: str) -> str | None:
    """The function name from a possibly-incomplete tool-call body."""
    body = body.strip()
    key_match = TOOL_ARG_KEY_PATTERN.search(body)
    if key_match:
        head = body[: key_match.start()].strip()
        return head.splitlines()[0].strip() if head else None
    function_match = TOOL_FUNCTION_PATTERN.search(body)
    if function_match:
        return function_match.group(1).strip()
    return None


def _iter_tool_call_arguments(body: str):
    """Yield ``(key, raw_value, closed)`` for a possibly-incomplete body.

    The trailing entry may be open -- its value is still being generated -- which
    is what lets a long parameter stream instead of landing in one piece.
    """
    body = body.strip()
    if TOOL_ARG_KEY_PATTERN.search(body):
        pattern, open_pattern = TOOL_ARG_PAIR_PATTERN, TOOL_ARG_OPEN_PATTERN
    elif TOOL_FUNCTION_PATTERN.search(body):
        pattern, open_pattern = TOOL_PARAMETER_PATTERN, TOOL_PARAMETER_OPEN_PATTERN
    else:
        return
    end = 0
    for match in pattern.finditer(body):
        yield match.group(1).strip(), match.group(2), True
        end = match.end()
    trailing = open_pattern.search(body, end)
    if trailing:
        yield trailing.group(1).strip(), trailing.group(2), False


def _recover_trailing_parameters(
    arguments: dict[str, Any], following: str
) -> dict[str, Any]:
    """Fold in parameters the model emitted after </tool_call>.

    Qwen3.5's format nests three levels deep, and only the outermost pair is
    atomic in its vocabulary: <tool_call> and </tool_call> are single tokens,
    while </function> and </parameter> are spelled out in ordinary subwords. So
    the closers the model has to compose by hand are exactly the ones it
    misorders, and the common shape is a block that closes early with the
    arguments trailing behind it:

        <tool_call>
        <function=list_files>
        </function>
        </tool_call>
        <parameter=execute>
        true
        </parameter>

    The block itself parses -- there is a name -- but it decodes to no
    arguments at all, so the required-parameter check drops the call and the
    agent loop stalls with nothing saying why. The model's own template warns
    against this shape ("ONLY reply in the following format with NO suffix",
    "reasoning ... BEFORE the function call, but NOT after"), which is a fair
    sign of how often it happens.

    Recovery is deliberately additive: anything the block decoded itself wins,
    and a well-formed call has no trailing parameters to find, so it is
    unaffected. Text between calls that is not a <parameter=...> tag is ignored.
    """
    if not arguments and TOOL_CALL_MARKER in following:
        # Defensive: never reach across into a block we did not match.
        following = following[: following.index(TOOL_CALL_MARKER)]
    recovered = dict(arguments)
    for key, value in TOOL_PARAMETER_LOOSE_PATTERN.findall(following):
        recovered.setdefault(key.strip(), _trim_parameter_text(value))
    return recovered


def _decode_tool_call_body(body: str) -> tuple[str | None, dict[str, Any]]:
    """Decode one <tool_call> body in the Hermes, BailingMoE3 or JSON format."""
    body = body.strip()
    # BailingMoE3 puts the function name on the opening line and each argument
    # in its own tagged pair:
    #     <tool_call>read_file
    #     <arg_key>path</arg_key>
    #     <arg_value>/etc/hosts</arg_value>
    #     </tool_call>
    # Checked before the others because a name followed by tags parses as
    # neither of them -- which is what used to happen, and a block that decodes
    # to no name is dropped along with the content preceding it, so the model
    # appeared to answer with nothing at all.
    key_match = TOOL_ARG_KEY_PATTERN.search(body)
    if key_match:
        name = body[: key_match.start()].strip().splitlines()[0].strip()
        arguments: dict[str, Any] = {}
        for key, value in TOOL_ARG_PAIR_PATTERN.findall(body):
            # Text, for the same reason the Hermes branch keeps it as text: the
            # declared schema decides the type in _normalize_schema_value.
            arguments[key.strip()] = _trim_parameter_text(value)
        return (name or None), arguments
    function_match = TOOL_FUNCTION_PATTERN.search(body)
    if function_match:
        arguments: dict[str, Any] = {}
        for parameter in TOOL_PARAMETER_PATTERN.finditer(body):
            # Kept as text. Typing it here would have to guess, and guessing
            # JSON turns file content that happens to parse (a .json edit, a
            # bare number) into a value the tool never asked for. The declared
            # schema decides in _normalize_schema_value instead.
            arguments[parameter.group(1).strip()] = _trim_parameter_text(
                parameter.group(2)
            )
        # A dropped </parameter> otherwise costs the entire call: the arguments
        # come out empty, the required-parameter check in _parse_tool_calls
        # rejects it, and the caller is handed prose with no tool call at all --
        # an agent loop just stalls, with nothing saying why. The format is
        # Qwen3.5's own (its template mandates the nested function/parameter
        # tags), and a low-bit quantization of it drops the closing tag often
        # enough to matter, so recover those parameters by ending them at the
        # next tag. Strictly-parsed values win, leaving well-formed bodies
        # decoding exactly as before.
        if len(arguments) != len(TOOL_PARAMETER_OPEN_TAG_PATTERN.findall(body)):
            for key, value in TOOL_PARAMETER_LOOSE_PATTERN.findall(body):
                arguments.setdefault(key.strip(), _trim_parameter_text(value))
        return function_match.group(1).strip(), arguments
    # JSON style: {"name": "fn", "arguments": {...}} (arguments may be a string).
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None, {}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("name"), str):
        return None, {}
    raw_arguments = parsed.get("arguments", parsed.get("parameters", {}))
    if isinstance(raw_arguments, str):
        try:
            raw_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            raw_arguments = {}
    return parsed["name"], raw_arguments if isinstance(raw_arguments, dict) else {}


def _response_tools(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("tools") or []
    if not isinstance(value, list):
        raise APIError(400, "tools must be an array", parameter="tools")
    tools: list[dict[str, Any]] = []
    for index, tool in enumerate(value):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise APIError(
                400,
                f"tools[{index}] must be a function",
                parameter="tools",
            )
        name = tool.get("name")
        parameters = tool.get("parameters", {})
        if not isinstance(name, str) or not name or not isinstance(parameters, dict):
            raise APIError(400, f"tools[{index}] is invalid", parameter="tools")
        function = {
            "name": name,
            "description": tool.get("description", ""),
            "parameters": parameters,
        }
        tools.append({"type": "function", "function": function})
    choice = payload.get("tool_choice", "auto" if tools else "none")
    if choice == "none":
        return []
    if choice in ("auto", "required"):
        return tools
    if isinstance(choice, dict) and choice.get("type") == "function":
        name = choice.get("name")
        selected = [tool for tool in tools if tool["function"]["name"] == name]
        if not selected:
            raise APIError(
                400,
                "tool_choice names an unknown tool",
                parameter="tool_choice",
            )
        return selected
    raise APIError(400, "tool_choice is invalid", parameter="tool_choice")


def _prepend_tool_prompt(
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    tool_choice: Any,
) -> None:
    prompt = _tool_prompt(tools, tool_choice)
    if messages and messages[0]["role"] == "system":
        messages[0] = {
            "role": "system",
            "content": f"{prompt}\n\n{messages[0]['content']}",
        }
        return
    messages.insert(0, {"role": "system", "content": prompt})


def _response_output(
    result: GenerationResult,
    message_id: str,
    *,
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    content, tool_calls = (
        _parse_tool_calls(result.text, tools=tools) if tools else (result.text, [])
    )
    incomplete = not result.stopped_on_eos and not tool_calls
    output: list[dict[str, Any]] = []
    if content is not None or not tool_calls:
        output.append(
            {
                "id": message_id,
                "type": "message",
                "status": "incomplete" if incomplete else "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": content or "",
                    }
                ],
            }
        )
    for call in tool_calls:
        output.append(
            {
                "id": f"fc_{uuid.uuid4().hex}",
                "type": "function_call",
                "status": "completed",
                "arguments": call["function"]["arguments"],
                "call_id": call["id"],
                "name": call["function"]["name"],
            }
        )
    return output


def _response_output_events(
    output: list[dict[str, Any]], sequence: int
) -> Iterator[dict[str, Any]]:
    for output_index, item in enumerate(output):
        if item["type"] == "function_call":
            pending_item = {**item, "status": "in_progress", "arguments": ""}
            yield {
                "type": "response.output_item.added",
                "sequence_number": sequence,
                "output_index": output_index,
                "item": pending_item,
            }
            sequence += 1
            yield {
                "type": "response.function_call_arguments.delta",
                "sequence_number": sequence,
                "item_id": item["id"],
                "output_index": output_index,
                "delta": item["arguments"],
            }
            sequence += 1
            yield {
                "type": "response.function_call_arguments.done",
                "sequence_number": sequence,
                "item_id": item["id"],
                "output_index": output_index,
                "arguments": item["arguments"],
            }
            sequence += 1
            yield {
                "type": "response.output_item.done",
                "sequence_number": sequence,
                "output_index": output_index,
                "item": item,
            }
            sequence += 1
            continue

        text = item["content"][0]["text"]
        pending_item = {**item, "status": "in_progress", "content": []}
        yield {
            "type": "response.output_item.added",
            "sequence_number": sequence,
            "output_index": output_index,
            "item": pending_item,
        }
        sequence += 1
        part = {**item["content"][0], "text": ""}
        yield {
            "type": "response.content_part.added",
            "sequence_number": sequence,
            "item_id": item["id"],
            "output_index": output_index,
            "content_index": 0,
            "part": part,
        }
        sequence += 1
        if text:
            yield {
                "type": "response.output_text.delta",
                "sequence_number": sequence,
                "item_id": item["id"],
                "output_index": output_index,
                "content_index": 0,
                "delta": text,
                "logprobs": [],
            }
            sequence += 1
        yield {
            "type": "response.output_text.done",
            "sequence_number": sequence,
            "item_id": item["id"],
            "output_index": output_index,
            "content_index": 0,
            "text": text,
            "logprobs": [],
        }
        sequence += 1
        yield {
            "type": "response.content_part.done",
            "sequence_number": sequence,
            "item_id": item["id"],
            "output_index": output_index,
            "content_index": 0,
            "part": item["content"][0],
        }
        sequence += 1
        yield {
            "type": "response.output_item.done",
            "sequence_number": sequence,
            "output_index": output_index,
            "item": item,
        }
        sequence += 1


def _validate_response_input(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise APIError(400, "input must be a non-empty array", parameter="input")
    messages: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise APIError(400, f"input[{index}] must be an object", parameter="input")
        item_type = item.get("type")
        if item_type == "function_call_output":
            output = item.get("output")
            if not isinstance(output, str):
                raise APIError(400, "function output must be text", parameter="input")
            messages.append(
                {
                    "role": "user",
                    "content": f"<tool_response>\n{output}\n</tool_response>",
                }
            )
            continue
        if item_type == "function_call":
            name = item.get("name")
            arguments = item.get("arguments", "{}")
            rendered = _render_tool_calls(
                "",
                [{"function": {"name": name, "arguments": arguments}}],
                index,
            )
            messages.append({"role": "assistant", "content": rendered})
            continue
        role = item.get("role")
        if item_type not in (None, "message") or role not in VALID_ROLES:
            raise APIError(400, f"input[{index}] is invalid", parameter="input")
        content = _text_content(item.get("content"), index)
        messages.append(
            {"role": "system" if role == "developer" else role, "content": content}
        )
    return messages


def _response_messages(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    instructions = payload.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str) or not instructions.strip():
            raise APIError(400, "instructions must be text", parameter="instructions")
        messages.append({"role": "system", "content": instructions})
    input_value = payload.get("input")
    if isinstance(input_value, str):
        if not input_value.strip():
            raise APIError(400, "input must not be empty", parameter="input")
        messages.append({"role": "user", "content": input_value})
    elif isinstance(input_value, list):
        messages.extend(_validate_response_input(input_value))
    else:
        raise APIError(
            400, "input must be text or an array of messages", parameter="input"
        )
    if messages[-1]["role"] != "user":
        raise APIError(
            400, "the last input message must have role 'user'", parameter="input"
        )
    return messages


def _validate_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise APIError(400, "messages must be a non-empty array", parameter="messages")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise APIError(
                400, f"messages[{index}] must be an object", parameter="messages"
            )
        role = message.get("role")
        if role not in VALID_ROLES:
            raise APIError(
                400, f"messages[{index}].role is invalid", parameter="messages"
            )
        content = _text_content(message.get("content"), index)
        normalized_role = "system" if role == "developer" else role
        messages.append({"role": normalized_role, "content": content})
    if messages[-1]["role"] != "user":
        raise APIError(
            400, "the last message must have role 'user'", parameter="messages"
        )
    return messages


def _text_content(value: Any, message_index: int) -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list) and value:
        parts: list[str] = []
        for part_index, part in enumerate(value):
            if not isinstance(part, dict) or part.get("type") not in TEXT_PART_TYPES:
                raise APIError(
                    400,
                    f"messages[{message_index}].content[{part_index}] must be a text part",
                    parameter="messages",
                )
            text = part.get("text")
            if not isinstance(text, str) or not text:
                raise APIError(
                    400, "text content parts must contain text", parameter="messages"
                )
            parts.append(text)
        return "".join(parts)
    raise APIError(
        400,
        f"messages[{message_index}].content must be text",
        parameter="messages",
    )


def _optional_text_content(value: Any, message_index: int) -> str:
    """Like _text_content but tolerates missing/empty content (returns "").

    Agentic clients (opencode, Cline, ...) send empty content on assistant
    tool-call turns ({"content": "", "tool_calls": [...]}) and for tools that
    produce no output; those are valid and must not 400 the whole request.
    A non-empty but non-text payload still raises via _text_content.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value if value.strip() else ""
    if isinstance(value, list) and not value:
        return ""
    return _text_content(value, message_index)


def _reject_unsupported_generation_options(payload: Mapping[str, Any]) -> None:
    if payload.get("n", 1) != 1:
        raise APIError(400, "only n=1 is supported", parameter="n")


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    }


@overload
def _boolean_option(payload: Mapping[str, Any], key: str, default: bool) -> bool: ...
@overload
def _boolean_option(
    payload: Mapping[str, Any], key: str, default: None
) -> bool | None: ...
def _boolean_option(
    payload: Mapping[str, Any], key: str, default: bool | None
) -> bool | None:
    """The flag as the caller set it. A None default means "unstated".

    Whether a request is silent about a setting is information: a chat template
    may have its own default, and answering "false" on its behalf overrides a
    choice the checkpoint made.
    """
    value = payload.get(key, default)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise APIError(400, f"{key} must be a boolean", parameter=key)
    return value


def _integer_option(
    payload: Mapping[str, Any],
    key: str,
    *,
    fallback_key: str | None = None,
    default: int,
) -> int:
    value = payload.get(key)
    if value is None and fallback_key is not None:
        value = payload.get(fallback_key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise APIError(400, f"{key} must be an integer", parameter=key)
    return value


def _optional_integer(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise APIError(400, f"{key} must be an integer", parameter=key)
    return value


def _float_option(payload: Mapping[str, Any], key: str, default: float) -> float:
    # OpenAI-compatible clients commonly serialize unset optional values as
    # JSON null. Treat that the same as an omitted option.
    value = payload.get(key)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise APIError(400, f"{key} must be a number", parameter=key)
    return float(value)


def _error_payload(error: APIError) -> dict[str, Any]:
    return {
        "error": {
            "message": error.message,
            "type": error.error_type,
            "param": error.parameter,
            "code": error.code,
        }
    }


def _is_decode_event(event: object) -> bool:
    """Return whether an SSE event contains generated output."""
    if not isinstance(event, dict):
        return False
    event_type = event.get("type")
    if event_type in {
        "response.output_text.delta",
        "response.function_call_arguments.delta",
    }:
        return bool(event.get("delta"))
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    choice = choices[0]
    if not isinstance(choice, dict):
        return False
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return False
    return bool(delta.get("content") or delta.get("tool_calls"))
