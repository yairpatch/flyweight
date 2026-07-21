from __future__ import annotations

import hmac
import json
import re
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import unquote, urlsplit

from .causal_lm import QwenForCausalLM
from .cuda import active_cuda
from .generation import GenerationResult, GenerationStep, TextGenerator
from .hardware import available_ram_bytes as probe_available_ram_bytes
from .native import active_native
from .sampling import SamplingConfig
from .tokenizer import HuggingFaceTokenizer


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
TOOL_CALL_BLOCK_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_FUNCTION_PATTERN = re.compile(r"<function=([^>\n]+)>", re.DOTALL)
TOOL_PARAMETER_PATTERN = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL
)


@dataclass(frozen=True, slots=True)
class APIError(Exception):
    status: int
    message: str
    error_type: str = "invalid_request_error"
    parameter: str | None = None


@dataclass(frozen=True, slots=True)
class _GenerationRequest:
    messages: list[dict[str, str]]
    max_new_tokens: int
    sampling: SamplingConfig
    enable_thinking: bool
    tools_enabled: bool = False


@dataclass(frozen=True, slots=True)
class _TextRequest:
    prompt: str
    max_new_tokens: int
    sampling: SamplingConfig


class InferenceService:
    """Persistent model service with serialized access to one accelerator."""

    def __init__(
        self,
        model_name: str,
        generator: TextGenerator,
        *,
        max_new_tokens: int = 64,
        context_window: int = 4096,
        api_key: str | None = None,
        cors_origin: str = "*",
        cpu_moe_layers: int = 0,
        strict_model: bool = False,
    ):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if context_window <= 0:
            raise ValueError("context_window must be positive")
        if max_new_tokens > context_window:
            raise ValueError("max_new_tokens must not exceed context_window")
        if cpu_moe_layers < 0:
            raise ValueError("cpu_moe_layers must be non-negative")
        self.model_name = model_name
        self.generator = generator
        self.max_new_tokens = max_new_tokens
        self.context_window = context_window
        self.api_key = api_key
        self.cors_origin = cors_origin
        self.cpu_moe_layers = cpu_moe_layers
        self.strict_model = strict_model
        self.loaded_at = int(time.time())
        self.preloaded_experts = 0
        self.expert_storage_bytes = 0
        self._generation_lock = threading.Lock()
        self._response_lock = threading.Lock()
        self._response_records: OrderedDict[
            str, tuple[dict[str, Any], list[dict[str, str]]]
        ] = OrderedDict()

    @classmethod
    def from_model_directory(
        cls,
        root: Path | str,
        *,
        model_name: str | None = None,
        rows_per_chunk: int = 4096,
        max_new_tokens: int = 64,
        context_window: int = 4096,
        api_key: str | None = None,
        cors_origin: str = "*",
        expert_preload: str = "none",
        cpu_moe_layers: int = 0,
        strict_model: bool = False,
        available_ram_bytes: int | None = None,
        expert_preload_progress: Callable[[int, int], None] | None = None,
    ) -> "InferenceService":
        root_path = Path(root)
        tokenizer = HuggingFaceTokenizer.from_model_directory(root_path)
        model = QwenForCausalLM.from_model_directory(
            root_path, rows_per_chunk=rows_per_chunk
        )
        model.configure_moe_placement(cpu_moe_layers)
        if expert_preload == "auto" and available_ram_bytes is None:
            available_ram_bytes = probe_available_ram_bytes()
        preloaded_experts = model.preload_experts(
            mode=expert_preload,
            available_ram_bytes=available_ram_bytes,
            progress=expert_preload_progress,
        )
        service = cls(
            model_name or root_path.name,
            TextGenerator(model, tokenizer),
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            api_key=api_key,
            cors_origin=cors_origin,
            cpu_moe_layers=model.cpu_moe_layers,
            strict_model=strict_model,
        )
        service.preloaded_experts = preloaded_experts
        service.expert_storage_bytes = model.estimated_expert_storage_bytes
        return service

    def health(self) -> dict[str, Any]:
        accelerator = active_cuda()
        execution = (
            accelerator.stats() if accelerator is not None else {"device": "cpu"}
        )
        native = active_native()
        if native is not None:
            execution["native_cpu_features"] = list(native.features)
        return {
            "status": "ok",
            "model": self.model_name,
            "loaded_at": self.loaded_at,
            "busy": self._generation_lock.locked(),
            "preloaded_experts": self.preloaded_experts,
            "expert_storage_bytes": self.expert_storage_bytes,
            "cpu_moe_layers": self.cpu_moe_layers,
            "context_window": self.context_window,
            "prefix_cache": (
                self.generator.prefix_cache_stats()
                if hasattr(self.generator, "prefix_cache_stats")
                else {"entries": 0, "capacity": 0}
            ),
            "execution": execution,
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
        with self._generation_lock:
            result = self.generator.generate_messages(
                request.messages,
                max_new_tokens=request.max_new_tokens,
                sampling=request.sampling,
                enable_thinking=request.enable_thinking,
                progress=progress,
            )
        return self._chat_response(result)

    def stream_chat_completion(
        self,
        payload: Mapping[str, Any],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[dict[str, Any] | str]:
        request = self._prepare_chat(payload)
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
                # first token. Tool calls can only be parsed once the full
                # generation is known, so their tool_use block is emitted at
                # the end from the accumulated text.
                final_step: GenerationStep | None = None
                accumulated = ""
                decode_started: float | None = None
                with self._generation_lock:
                    for step in self.generator.stream_messages(
                        request.messages,
                        max_new_tokens=request.max_new_tokens,
                        sampling=request.sampling,
                        enable_thinking=request.enable_thinking,
                        progress=progress,
                    ):
                        if step.finished:
                            final_step = step
                            continue
                        if step.token_id is None:
                            continue
                        accumulated += step.text_delta or ""
                        now = time.perf_counter()
                        if decode_started is None:
                            decode_started = now
                        chunk = self._chat_chunk(
                            completion_id,
                            created,
                            {"content": step.text_delta} if step.text_delta else {},
                        )
                        chunk["colibri"] = {
                            "generated_tokens": len(step.generated_ids),
                            "decode_elapsed_seconds": now - decode_started,
                        }
                        yield chunk
                if final_step is None:
                    raise RuntimeError("generation stream ended without a final result")
                _, tool_calls = _parse_tool_calls(accumulated)
                if tool_calls:
                    yield self._chat_chunk(
                        completion_id,
                        created,
                        {
                            "tool_calls": [
                                {"index": index, **call}
                                for index, call in enumerate(tool_calls)
                            ]
                        },
                    )
                finish_reason = (
                    "tool_calls"
                    if tool_calls
                    else ("stop" if final_step.stopped_on_eos else "length")
                )
                prompt_count = len(final_step.prompt_ids)
                completion_count = len(final_step.generated_ids)
            else:
                final_step: GenerationStep | None = None
                decode_started: float | None = None
                with self._generation_lock:
                    for step in self.generator.stream_messages(
                        request.messages,
                        max_new_tokens=request.max_new_tokens,
                        sampling=request.sampling,
                        enable_thinking=request.enable_thinking,
                        progress=progress,
                    ):
                        if step.finished:
                            final_step = step
                        elif step.token_id is not None:
                            now = time.perf_counter()
                            if decode_started is None:
                                decode_started = now
                            chunk = self._chat_chunk(
                                completion_id,
                                created,
                                {"content": step.text_delta} if step.text_delta else {},
                            )
                            # Keep the standard OpenAI chunk shape while exposing
                            # a small provider extension for live UI metrics.
                            chunk["colibri"] = {
                                "generated_tokens": len(step.generated_ids),
                                "decode_elapsed_seconds": now - decode_started,
                            }
                            yield chunk
                if final_step is None:
                    raise RuntimeError("generation stream ended without a final result")
                finish_reason = "stop" if final_step.stopped_on_eos else "length"
                prompt_count = len(final_step.prompt_ids)
                completion_count = len(final_step.generated_ids)
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
        with self._generation_lock:
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
            with self._generation_lock:
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
            enable_thinking=_boolean_option(payload, "enable_thinking", False),
        )
        return {"object": "response.input_tokens", "input_tokens": len(tokens)}

    def properties(self) -> dict[str, Any]:
        return {
            "model_path": self.model_name,
            "model_alias": self.model_name,
            "total_slots": 1,
            "max_output_tokens": self.max_new_tokens,
            "context_window": self.context_window,
            "chat_template": "qwen-chatml",
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
        with self._generation_lock:
            result = self.generator.generate_messages(
                request.messages,
                max_new_tokens=request.max_new_tokens,
                sampling=request.sampling,
                enable_thinking=request.enable_thinking,
                progress=progress,
            )
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
                with self._generation_lock:
                    result = self.generator.generate_messages(
                        request.messages,
                        max_new_tokens=request.max_new_tokens,
                        sampling=request.sampling,
                        enable_thinking=request.enable_thinking,
                        progress=progress,
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
                    "type": "response.completed",
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
            with self._generation_lock:
                for step in self.generator.stream_messages(
                    request.messages,
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
                prompt_ids=final_step.prompt_ids,
                generated_ids=final_step.generated_ids,
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
                "status": "completed",
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
                "type": "response.completed",
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
        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": payload.get("model", self.model_name),
            "content": content,
            "stop_reason": "tool_use" if finish_reason == "tool_calls" else "end_turn",
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
        input_tokens = self._estimate_prompt_tokens(chat_payload)
        chat_events = self.stream_chat_completion(chat_payload, progress=progress)

        def events() -> Iterator[dict[str, Any]]:
            message_id = f"msg_{uuid.uuid4().hex}"
            sequence = 0
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
            text_block_open = False
            text_block_index = None
            tool_blocks: dict[str, int] = {}
            last_colibri: dict[str, Any] | None = None
            for event in chat_events:
                if event == "[DONE]":
                    for open_index in sorted(
                        i for i in [text_block_index, *tool_blocks.values()]
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
                    stop_reason = (
                        "tool_use"
                        if choice["finish_reason"] == "tool_calls"
                        else "end_turn"
                    )
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
                for call in delta.get("tool_calls", []):
                    stop_reason = "tool_use"
                    function = call.get("function", {})
                    call_id = call.get("id", f"toolu_{uuid.uuid4().hex}")
                    if call_id not in tool_blocks:
                        tool_blocks[call_id] = block_index
                        yield {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": call_id,
                                "name": function.get("name", "tool"),
                                "input": {},
                            },
                        }
                        block_index += 1
                    arguments = function.get("arguments", "")
                    if arguments:
                        yield {
                            "type": "content_block_delta",
                            "index": tool_blocks[call_id],
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
            payload, "max_tokens", default=min(16, self.max_new_tokens)
        )
        max_new_tokens = self._fit_max_new_tokens(
            requested_max,
            len(self.generator.tokenizer.encode(prompt)),
            parameter="max_tokens",
        )
        try:
            sampling = SamplingConfig(
                temperature=_float_option(payload, "temperature", 0.0),
                top_k=_integer_option(payload, "top_k", default=20),
                top_p=_float_option(payload, "top_p", 0.95),
                seed=_optional_integer(payload, "seed"),
            )
        except ValueError as error:
            raise APIError(400, str(error)) from error
        return _TextRequest(prompt, max_new_tokens, sampling)

    def _prepare_chat(self, payload: Mapping[str, Any]) -> _GenerationRequest:
        messages, tools_enabled = _chat_messages(payload)
        return self._prepare_generation(
            payload,
            messages,
            max_key="max_completion_tokens",
            fallback_max_key="max_tokens",
            tools_enabled=tools_enabled,
        )

    def _prepare_generation(
        self,
        payload: Mapping[str, Any],
        messages: list[dict[str, str]],
        *,
        max_key: str,
        fallback_max_key: str | None = None,
        tools_enabled: bool = False,
    ) -> _GenerationRequest:
        self._validate_model(payload.get("model"))
        _reject_unsupported_generation_options(payload)
        requested_max = _integer_option(
            payload,
            max_key,
            fallback_key=fallback_max_key,
            default=min(16, self.max_new_tokens),
        )
        try:
            sampling = SamplingConfig(
                temperature=_float_option(payload, "temperature", 0.0),
                top_k=_integer_option(payload, "top_k", default=20),
                top_p=_float_option(payload, "top_p", 0.95),
                seed=_optional_integer(payload, "seed"),
            )
        except ValueError as error:
            raise APIError(400, str(error)) from error
        enable_thinking = _boolean_option(payload, "enable_thinking", False)
        try:
            prompt_tokens = len(
                self.generator.tokenizer.encode_messages(
                    messages, enable_thinking=enable_thinking
                )
            )
        except Exception:
            # Tokenization may fail on exotic/partial prompts or a backend
            # tokenizer that rejects certain control sequences. Never let that
            # block the request; fall back to a zero prompt count so clamping
            # simply uses the whole context window.
            prompt_tokens = 0
        max_new_tokens = self._fit_max_new_tokens(
            requested_max, prompt_tokens, parameter=max_key
        )
        return _GenerationRequest(
            messages, max_new_tokens, sampling, enable_thinking, tools_enabled
        )

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
            )
        return max(1, min(requested, self.max_new_tokens, room))

    def _estimate_prompt_tokens(self, chat_payload: Mapping[str, Any]) -> int:
        """Best-effort prompt token count for streaming usage reporting.

        Mirrors the generation path (tool-prompt injection + chat formatting)
        so the ``input_tokens`` reported in an Anthropic ``message_start`` block
        matches what generation actually consumes. Returns 0 on any error since
        usage reporting must never break the stream.
        """
        try:
            messages, _ = _chat_messages(chat_payload)
            return len(
                self.generator.tokenizer.encode_messages(
                    messages,
                    enable_thinking=_boolean_option(
                        chat_payload, "enable_thinking", False
                    ),
                )
            )
        except Exception:
            return 0

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

    def _chat_response(self, result: GenerationResult) -> dict[str, Any]:
        content, tool_calls = _parse_tool_calls(result.text)
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
            "temperature": _float_option(payload, "temperature", 0.0),
            "text": {"format": {"type": "text"}},
            "tool_choice": payload.get(
                "tool_choice", "auto" if payload.get("tools") else "none"
            ),
            "tools": payload.get("tools") or [],
            "top_p": _float_option(payload, "top_p", 0.95),
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
        return {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "completed",
            "background": False,
            "error": None,
            "incomplete_details": None,
            "instructions": payload.get("instructions"),
            "max_output_tokens": payload.get("max_output_tokens"),
            "model": self.model_name,
            "output": _response_output(result, message_id, parse_tools=bool(tools)),
            "parallel_tool_calls": False,
            "previous_response_id": payload.get("previous_response_id"),
            "store": _boolean_option(payload, "store", True),
            "temperature": _float_option(payload, "temperature", 0.0),
            "text": {"format": {"type": "text"}},
            "tool_choice": payload.get(
                "tool_choice", "auto" if payload.get("tools") else "none"
            ),
            "tools": payload.get("tools") or [],
            "top_p": _float_option(payload, "top_p", 0.95),
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


def create_handler(
    service: InferenceService,
) -> type[BaseHTTPRequestHandler]:
    class ColibriRequestHandler(BaseHTTPRequestHandler):
        server_version = "colibri-next/0.2"
        protocol_version = "HTTP/1.1"

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._send_cors_headers()
            self.send_header(
                "Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"
            )
            self.send_header(
                "Access-Control-Allow-Headers", "Authorization, Content-Type"
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
                            service.stream_chat_completion(payload, progress=progress)
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
                            service.stream_completion(payload, progress=progress)
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
                    messages_payload = _anthropic_to_chat_payload(payload)
                    self._send_json(
                        200,
                        service.count_response_input(
                            {
                                "input": messages_payload["messages"],
                                "model": messages_payload.get("model"),
                            }
                        ),
                    )
                    self.log_message("request completed: %s", path)
                    return
                if path == "/v1/messages":
                    if _boolean_option(payload, "stream", False):
                        self._send_sse(
                            service.stream_anthropic_message(payload, progress=progress)
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
                self.log_error("unhandled request error")
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
            last_reported = -1
            started = time.perf_counter()

            def report(processed: int, total: int) -> None:
                nonlocal last_reported
                if processed == last_reported:
                    return
                last_reported = processed
                pct = (100 * processed / total) if total else 100
                elapsed = time.perf_counter() - started
                rate = (processed / elapsed) if elapsed > 0 else 0.0
                sys.stderr.write(
                    f"\r[load] prompt {processed}/{total} tokens "
                    f"({pct:5.1f}%) {rate:6.1f} tok/s"
                )
                sys.stderr.flush()
                if processed >= total:
                    sys.stderr.write(
                        f"\n[load] prompt ready in {elapsed:5.2f}s "
                        f"({total} tokens, {rate:6.1f} tok/s)\n"
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

        def _send_sse(self, events: Iterator[dict[str, Any] | str]) -> None:
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
            decode_started: float | None = None
            last_stat = 0.0
            try:
                for event in events:
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
                            if isinstance(tokens, int) and isinstance(elapsed, float):
                                decode_tokens = tokens
                                if decode_started is None:
                                    decode_started = time.perf_counter() - elapsed
                                now = time.perf_counter()
                                if now - last_stat >= 1.0:
                                    last_stat = now
                                    rate = (tokens / elapsed) if elapsed > 0 else 0.0
                                    sys.stderr.write(
                                        f"\r[gen ] {tokens} tokens "
                                        f"{rate:6.1f} tok/s {elapsed:5.1f}s"
                                    )
                                    sys.stderr.flush()
                    self._write_sse_event(
                        data, event if isinstance(event, dict) else None
                    )
                if decode_tokens:
                    total = (
                        time.perf_counter() - decode_started
                        if decode_started is not None
                        else 0.0
                    )
                    rate = (decode_tokens / total) if total > 0 else 0.0
                    sys.stderr.write(
                        f"\n[gen ] done {decode_tokens} tokens "
                        f"in {total:5.2f}s ({rate:6.1f} tok/s)\n"
                    )
                    sys.stderr.flush()
                self.log_message("request completed: streaming events=%d", event_count)
            except (BrokenPipeError, ConnectionResetError):
                stream_ok = False
                self.close_connection = True
            except APIError as error:
                # Once SSE headers are sent, a JSON error response is no longer
                # possible. Send the error as the final SSE event instead.
                self.log_error("stream request failed: %s", error.message)
                try:
                    self._write_sse_event(json.dumps(_error_payload(error)))
                except (BrokenPipeError, ConnectionResetError):
                    stream_ok = False
                    self.close_connection = True
            except Exception as error:
                # Generation happens while the iterator is consumed, after the
                # HTTP status line has already been written. Keep clients such
                # as Cline and OpenCode from waiting on a silent/truncated stream.
                self.log_error("unhandled stream error: %s", error)
                try:
                    self._write_sse_event(
                        json.dumps(
                            _error_payload(
                                APIError(500, "internal server error", "server_error")
                            )
                        )
                    )
                except (BrokenPipeError, ConnectionResetError):
                    stream_ok = False
                    self.close_connection = True
            finally:
                close = getattr(events, "close", None)
                if close is not None:
                    close()
                if stream_ok:
                    try:
                        self.wfile.write(b"0\r\n\r\n")  # terminating chunk
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        self.close_connection = True

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
            self._send_json(error.status, _error_payload(error))

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
) -> None:
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    server = ColibriHTTPServer((host, port), create_handler(service))
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _chat_messages(payload: Mapping[str, Any]) -> tuple[list[dict[str, str]], bool]:
    value = payload.get("messages")
    if not isinstance(value, list) or not value:
        raise APIError(400, "messages must be a non-empty array", parameter="messages")
    tools = _selected_tools(payload)
    system_parts: list[str] = []
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
        if role in ("system", "developer"):
            system_parts.append(_text_content(message.get("content"), index))
            continue
        if role == "tool":
            content = _text_content(message.get("content"), index)
            messages.append(
                {
                    "role": "user",
                    "content": f"<tool_response>\n{content}\n</tool_response>",
                }
            )
            continue
        content_value = message.get("content")
        content = "" if content_value is None else _text_content(content_value, index)
        if role == "assistant" and message.get("tool_calls"):
            content = _render_tool_calls(content, message["tool_calls"], index)
        if not content:
            raise APIError(
                400, f"messages[{index}].content must be text", parameter="messages"
            )
        messages.append({"role": role, "content": content})
    if tools:
        system_parts.insert(0, _tool_prompt(tools, payload.get("tool_choice")))
    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    # OpenAI-compatible clients may end a request with an assistant message
    # when they are asking the model to continue/prefill that turn. The chat
    # formatter adds the assistant generation marker after the supplied
    # history, so this is a valid prompt for the local runtime as well.
    if not messages or messages[-1]["role"] not in ("user", "assistant"):
        raise APIError(
            400,
            "the last message must have role 'user', 'assistant', or 'tool'",
            parameter="messages",
        )
    return messages, bool(tools)


def _anthropic_to_chat_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system = _anthropic_text(payload.get("system"))
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
            system_text = _anthropic_text(message.get("content"))
            if system_text:
                messages.append({"role": "system", "content": system_text})
            continue
        if role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "content": _anthropic_text(message.get("content")),
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
                    {"role": "tool", "content": _anthropic_text(block.get("content"))}
                )
        if text_parts or tool_calls:
            item: dict[str, Any] = {
                "role": role,
                "content": "".join(text_parts) or None,
            }
            if tool_calls:
                item["tool_calls"] = tool_calls
            messages.append(item)
        messages.extend(tool_results)
    result: dict[str, Any] = {
        "model": payload.get("model"),
        "messages": messages,
        "max_completion_tokens": payload.get("max_tokens"),
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "top_k": payload.get("top_k"),
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
        elif choice_type == "tool":
            result["tool_choice"] = {"function": {"name": choice.get("name")}}
        else:
            result["tool_choice"] = "auto"
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
    return (
        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
        f"{serialized}\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format "
        "with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
        "<parameter=example_parameter>\nvalue\n</parameter>\n</function>\n"
        "</tool_call>\n\nRequired parameters MUST be specified. You may provide "
        "reasoning before the tool call, but nothing after it."
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


def _parse_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    first_start: int | None = None
    for match in TOOL_CALL_BLOCK_PATTERN.finditer(text):
        name, arguments = _decode_tool_call_body(match.group(1))
        if name is None:
            continue
        if first_start is None:
            first_start = match.start()
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    content = text[:first_start].strip() if first_start is not None else text
    return (content or None), calls


def _decode_tool_call_body(body: str) -> tuple[str | None, dict[str, Any]]:
    """Decode one <tool_call> body in either the Hermes or JSON tool format."""
    body = body.strip()
    function_match = TOOL_FUNCTION_PATTERN.search(body)
    if function_match:
        arguments: dict[str, Any] = {}
        for parameter in TOOL_PARAMETER_PATTERN.finditer(body):
            raw_value = parameter.group(2).strip()
            try:
                arguments[parameter.group(1)] = json.loads(raw_value)
            except json.JSONDecodeError:
                arguments[parameter.group(1)] = raw_value
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
    parse_tools: bool,
) -> list[dict[str, Any]]:
    content, tool_calls = (
        _parse_tool_calls(result.text) if parse_tools else (result.text, [])
    )
    output: list[dict[str, Any]] = []
    if content is not None or not tool_calls:
        output.append(
            {
                "id": message_id,
                "type": "message",
                "status": "completed",
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


def _boolean_option(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
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
            "code": None,
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
