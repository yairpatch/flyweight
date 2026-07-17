from __future__ import annotations

import hmac
import json
import re
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
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
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
        api_key: str | None = None,
        cors_origin: str = "*",
        cpu_moe_layers: int = 0,
    ):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if cpu_moe_layers < 0:
            raise ValueError("cpu_moe_layers must be non-negative")
        self.model_name = model_name
        self.generator = generator
        self.max_new_tokens = max_new_tokens
        self.api_key = api_key
        self.cors_origin = cors_origin
        self.cpu_moe_layers = cpu_moe_layers
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
        api_key: str | None = None,
        cors_origin: str = "*",
        expert_preload: str = "none",
        cpu_moe_layers: int = 0,
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
            api_key=api_key,
            cors_origin=cors_origin,
            cpu_moe_layers=model.cpu_moe_layers,
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

    def chat_completion(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if _boolean_option(payload, "stream", False):
            raise APIError(400, "use the streaming response path", parameter="stream")
        request = self._prepare_chat(payload)
        with self._generation_lock:
            result = self.generator.generate_messages(
                request.messages,
                max_new_tokens=request.max_new_tokens,
                sampling=request.sampling,
                enable_thinking=request.enable_thinking,
            )
        return self._chat_response(result)

    def stream_chat_completion(
        self, payload: Mapping[str, Any]
    ) -> Iterator[dict[str, Any] | str]:
        request = self._prepare_chat(payload)
        stream_options = payload.get("stream_options") or {}
        if not isinstance(stream_options, dict):
            raise APIError(400, "stream_options must be an object", parameter="stream_options")
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
                with self._generation_lock:
                    result = self.generator.generate_messages(
                        request.messages,
                        max_new_tokens=request.max_new_tokens,
                        sampling=request.sampling,
                        enable_thinking=request.enable_thinking,
                    )
                content, tool_calls = _parse_tool_calls(result.text)
                if content:
                    yield self._chat_chunk(
                        completion_id, created, {"content": content}
                    )
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
                    else ("stop" if result.stopped_on_eos else "length")
                )
                prompt_count = len(result.prompt_ids)
                completion_count = len(result.generated_ids)
            else:
                final_step: GenerationStep | None = None
                with self._generation_lock:
                    for step in self.generator.stream_messages(
                        request.messages,
                        max_new_tokens=request.max_new_tokens,
                        sampling=request.sampling,
                        enable_thinking=request.enable_thinking,
                    ):
                        if step.finished:
                            final_step = step
                        elif step.text_delta:
                            yield self._chat_chunk(
                                completion_id,
                                created,
                                {"content": step.text_delta},
                            )
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
    def completion(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if _boolean_option(payload, "stream", False):
            raise APIError(400, "use the streaming response path", parameter="stream")
        request = self._prepare_text(payload)
        with self._generation_lock:
            result = self.generator.generate_text(
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                sampling=request.sampling,
            )
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        return self._completion_response(completion_id, result)

    def stream_completion(
        self, payload: Mapping[str, Any]
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
        if (
            not isinstance(tokens, list)
            or any(isinstance(token, bool) or not isinstance(token, int) for token in tokens)
        ):
            raise APIError(400, "tokens must be an array of integers", parameter="tokens")
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
    def response(self, payload: Mapping[str, Any]) -> dict[str, Any]:
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
            )
        response = self._response_object(payload, result)
        self._store_response(response, history_messages, result)
        return response

    def stream_response(
        self, payload: Mapping[str, Any]
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
    def retrieve_response(self, response_id: str) -> dict[str, Any]:
        with self._response_lock:
            record = self._response_records.get(response_id)
            if record is None:
                raise APIError(404, f"response '{response_id}' was not found", "not_found_error")
            self._response_records.move_to_end(response_id)
            return record[0]

    def delete_response(self, response_id: str) -> dict[str, Any]:
        with self._response_lock:
            if self._response_records.pop(response_id, None) is None:
                raise APIError(404, f"response '{response_id}' was not found", "not_found_error")
        return {"id": response_id, "object": "response.deleted", "deleted": True}

    def _response_messages_with_history(
        self, payload: Mapping[str, Any]
    ) -> list[dict[str, str]]:
        current = _response_messages(payload)
        previous_id = payload.get("previous_response_id")
        if previous_id is None:
            return current
        if not isinstance(previous_id, str):
            raise APIError(400, "previous_response_id must be text", parameter="previous_response_id")
        with self._response_lock:
            record = self._response_records.get(previous_id)
            if record is None:
                raise APIError(404, f"response '{previous_id}' was not found", "not_found_error")
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
            raise APIError(400, "logprobs are not implemented yet", parameter="logprobs")
        max_new_tokens = _integer_option(
            payload, "max_tokens", default=min(16, self.max_new_tokens)
        )
        if max_new_tokens <= 0 or max_new_tokens > self.max_new_tokens:
            raise APIError(
                400,
                f"max_tokens must be between 1 and {self.max_new_tokens}",
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
        max_new_tokens = _integer_option(
            payload,
            max_key,
            fallback_key=fallback_max_key,
            default=min(16, self.max_new_tokens),
        )
        if max_new_tokens <= 0 or max_new_tokens > self.max_new_tokens:
            raise APIError(
                400,
                f"max tokens must be between 1 and {self.max_new_tokens}",
                parameter=max_key,
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
        return _GenerationRequest(
            messages, max_new_tokens, sampling, enable_thinking, tools_enabled
        )

    def _validate_model(self, requested_model: Any) -> None:
        if requested_model is not None and requested_model != self.model_name:
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
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
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
                if model_name != service.model_name:
                    self._send_error(
                        APIError(404, f"model '{model_name}' is not loaded", "not_found_error")
                    )
                    return
                self._send_json(200, service.model())
                return
            self._send_error(APIError(404, "endpoint not found", "not_found_error"))

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
                payload = self._read_json()
                if path == "/v1/chat/completions":
                    if _boolean_option(payload, "stream", False):
                        self._send_sse(service.stream_chat_completion(payload))
                    else:
                        self._send_json(200, service.chat_completion(payload))
                    return
                if path == "/v1/completions":
                    if _boolean_option(payload, "stream", False):
                        self._send_sse(service.stream_completion(payload))
                    else:
                        self._send_json(200, service.completion(payload))
                    return
                if path == "/tokenize":
                    self._send_json(200, service.tokenize(payload))
                    return
                if path == "/detokenize":
                    self._send_json(200, service.detokenize(payload))
                    return
                if path == "/v1/responses/input_tokens":
                    self._send_json(200, service.count_response_input(payload))
                    return
                if path == "/v1/responses":
                    if _boolean_option(payload, "stream", False):
                        self._send_sse(service.stream_response(payload))
                    else:
                        self._send_json(200, service.response(payload))
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
            expected = f"Bearer {service.api_key}"
            if not hmac.compare_digest(authorization, expected):
                raise APIError(
                    401,
                    "invalid or missing API key",
                    error_type="authentication_error",
                )
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

        def _send_sse(self, events: Iterator[dict[str, Any] | str]) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self._send_cors_headers()
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                for event in events:
                    data = event if isinstance(event, str) else json.dumps(event, ensure_ascii=False)
                    event_name = (
                        event.get("type") if isinstance(event, dict) else None
                    )
                    prefix = f"event: {event_name}\n" if event_name else ""
                    self.wfile.write(
                        f"{prefix}data: {data}\n\n".encode("utf-8")
                    )
                    self.wfile.flush()
            finally:
                close = getattr(events, "close", None)
                if close is not None:
                    close()

        def _send_cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", service.cors_origin)
            self.send_header("Vary", "Origin")
        def _send_error(self, error: APIError) -> None:
            self._send_json(
                error.status,
                {
                    "error": {
                        "message": error.message,
                        "type": error.error_type,
                        "param": error.parameter,
                        "code": None,
                    }
                },
            )

        def _send_static(self, path: str) -> None:
            filename, content_type = UI_ASSETS[path]
            try:
                body = (UI_DIRECTORY / filename).read_bytes()
            except OSError:
                self._send_error(
                    APIError(404, "UI asset not found", "not_found_error")
                )
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
            self.wfile.write(body)

        def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_cors_headers()
            if status == 401:
                self.send_header("WWW-Authenticate", "Bearer")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
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


def _chat_messages(
    payload: Mapping[str, Any]
) -> tuple[list[dict[str, str]], bool]:
    value = payload.get("messages")
    if not isinstance(value, list) or not value:
        raise APIError(400, "messages must be a non-empty array", parameter="messages")
    tools = _selected_tools(payload)
    system_parts: list[str] = []
    messages: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise APIError(400, f"messages[{index}] must be an object", parameter="messages")
        role = message.get("role")
        if role not in VALID_ROLES:
            raise APIError(400, f"messages[{index}].role is invalid", parameter="messages")
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
            raise APIError(400, f"messages[{index}].content must be text", parameter="messages")
        messages.append({"role": role, "content": content})
    if tools:
        system_parts.insert(0, _tool_prompt(tools, payload.get("tool_choice")))
    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    if not messages or messages[-1]["role"] != "user":
        raise APIError(400, "the last message must have role 'user' or 'tool'", parameter="messages")
    return messages, bool(tools)


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
            raise APIError(400, f"tools[{index}].function is invalid", parameter="tools")
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
            raise APIError(400, "tool_choice names an unknown tool", parameter="tool_choice")
        return selected
    raise APIError(400, "tool_choice is invalid", parameter="tool_choice")


def _tool_prompt(tools: list[dict[str, Any]], tool_choice: Any) -> str:
    serialized = "\n".join(
        json.dumps(tool, ensure_ascii=False, separators=(",", ":"))
        for tool in tools
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
            raise APIError(400, f"messages[{message_index}].tool_calls is invalid", parameter="messages")
        function = call["function"]
        name = function.get("name")
        arguments = function.get("arguments", "{}")
        if not isinstance(name, str) or not isinstance(arguments, str):
            raise APIError(400, "tool call function is invalid", parameter="messages")
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise APIError(400, "tool call arguments must be JSON", parameter="messages") from error
        if not isinstance(parsed, dict):
            raise APIError(400, "tool call arguments must be an object", parameter="messages")
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
    for match in TOOL_CALL_PATTERN.finditer(text):
        if first_start is None:
            first_start = match.start()
        arguments: dict[str, Any] = {}
        for parameter in TOOL_PARAMETER_PATTERN.finditer(match.group(2)):
            raw_value = parameter.group(2).strip()
            try:
                arguments[parameter.group(1)] = json.loads(raw_value)
            except json.JSONDecodeError:
                arguments[parameter.group(1)] = raw_value
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": match.group(1).strip(),
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    content = text[:first_start].strip() if first_start is not None else text
    return (content or None), calls


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
        raise APIError(400, "input must be text or an array of messages", parameter="input")
    if messages[-1]["role"] != "user":
        raise APIError(400, "the last input message must have role 'user'", parameter="input")
    return messages


def _validate_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise APIError(400, "messages must be a non-empty array", parameter="messages")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise APIError(400, f"messages[{index}] must be an object", parameter="messages")
        role = message.get("role")
        if role not in VALID_ROLES:
            raise APIError(400, f"messages[{index}].role is invalid", parameter="messages")
        content = _text_content(message.get("content"), index)
        normalized_role = "system" if role == "developer" else role
        messages.append({"role": normalized_role, "content": content})
    if messages[-1]["role"] != "user":
        raise APIError(400, "the last message must have role 'user'", parameter="messages")
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
                raise APIError(400, "text content parts must contain text", parameter="messages")
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
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise APIError(400, f"{key} must be a number", parameter=key)
    return float(value)
