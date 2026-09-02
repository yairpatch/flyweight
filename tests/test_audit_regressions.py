"""Regressions pinned by the 2026-09 runtime audit.

Each test names the defect it guards: an option the server accepted and
ignored, a handler that could drop a connection without a response, a cache
key blind to the tool set, and a ctypes binding whose arity drifted from the
C signature so the call failed before it reached the library.
"""

from __future__ import annotations

import http.client
import json
import re
import threading
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from flyweight import v2
from flyweight.generation import GenerationResult
from flyweight.server import (
    APIError,
    FlyweightHTTPServer,
    InferenceService,
    _finished_turn,
    _responses_payload_with_format,
    _tool_turn,
    create_handler,
)
from flyweight.v2_server import _chat_key

from .test_server import StubGenerator, ToolStubGenerator

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


class TwoCallToolStubGenerator(ToolStubGenerator):
    TOOL_TEXT = (
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nParis\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nRome\n</parameter>\n"
        "</function>\n</tool_call>"
    )


class RejectedOptionsTests(unittest.TestCase):
    """Options the chat endpoint accepted and silently ignored."""

    def setUp(self) -> None:
        self.service = InferenceService("qwen-local", StubGenerator(), max_new_tokens=32)

    def _chat(self, **extra):
        return self.service.chat_completion(
            {
                "model": "qwen-local",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
                **extra,
            }
        )

    def test_logprobs_are_rejected_not_ignored(self) -> None:
        with self.assertRaises(APIError) as caught:
            self._chat(logprobs=True)
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(caught.exception.parameter, "logprobs")
        with self.assertRaises(APIError):
            self._chat(top_logprobs=5)
        # The falsy spellings clients send by default still pass.
        self._chat(logprobs=False)
        self._chat(logprobs=None)

    def test_logit_bias_is_rejected_when_non_empty(self) -> None:
        with self.assertRaises(APIError) as caught:
            self._chat(logit_bias={"50256": -100})
        self.assertEqual(caught.exception.parameter, "logit_bias")
        self._chat(logit_bias={})


class SingleToolCallTests(unittest.TestCase):
    """`parallel_tool_calls: false` must cap the turn at one call."""

    def test_finished_turn_truncates_when_asked(self) -> None:
        result = GenerationResult(
            prompt_ids=(1,),
            generated_ids=(2,),
            text=TwoCallToolStubGenerator.TOOL_TEXT,
            stopped_on_eos=True,
            state_tokens=2,
        )
        _, _, calls, reason, _ = _finished_turn(result, (GET_WEATHER_TOOL,))
        self.assertEqual(len(calls), 2)
        _, _, calls, reason, _ = _finished_turn(
            result, (GET_WEATHER_TOOL,), single_tool_call=True
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(reason, "tool_calls")
        self.assertIn("Paris", calls[0]["function"]["arguments"])

    def test_chat_completion_honours_parallel_tool_calls_false(self) -> None:
        service = InferenceService("qwen-local", TwoCallToolStubGenerator())
        request = {
            "model": "qwen-local",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [GET_WEATHER_TOOL],
        }
        # The service stops the generation at the first complete call on this
        # path already; the option must still be accepted and hold to one.
        one = service.chat_completion({**request, "parallel_tool_calls": False})
        self.assertEqual(len(one["choices"][0]["message"]["tool_calls"]), 1)
        self.assertEqual(one["choices"][0]["finish_reason"], "tool_calls")

    def test_chat_stream_honours_parallel_tool_calls_false(self) -> None:
        service = InferenceService("qwen-local", TwoCallToolStubGenerator())
        chunks = list(
            service.stream_chat_completion(
                {
                    "model": "qwen-local",
                    "messages": [{"role": "user", "content": "weather"}],
                    "tools": [GET_WEATHER_TOOL],
                    "parallel_tool_calls": False,
                }
            )
        )
        indices = {
            call["index"]
            for chunk in chunks
            if isinstance(chunk, dict)
            for choice in chunk.get("choices", [])
            for call in (choice.get("delta") or {}).get("tool_calls", []) or []
        }
        self.assertEqual(indices, {0})

    def test_anthropic_disable_parallel_tool_use(self) -> None:
        service = InferenceService("qwen-local", TwoCallToolStubGenerator())
        request = {
            "model": "qwen-local",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [
                {
                    "name": "get_weather",
                    "input_schema": GET_WEATHER_TOOL["function"]["parameters"],
                }
            ],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        }
        response = service.anthropic_message(request)
        uses = [block for block in response["content"] if block["type"] == "tool_use"]
        self.assertEqual(len(uses), 1)


class ToolResultErrorTests(unittest.TestCase):
    def test_is_error_is_spelled_out_in_the_prompt(self) -> None:
        turn = _tool_turn("disk full", architecture=None, is_error=True)
        self.assertIn("Error: disk full", turn["content"])
        native = _tool_turn("disk full", architecture="deepseek4", is_error=True,
                            tool_call_id="c1")
        self.assertEqual(native["role"], "tool")
        self.assertTrue(native["content"].startswith("Error: "))
        # Already-labelled output is not doubled, and a plain result is untouched.
        self.assertEqual(
            _tool_turn("error: x", architecture=None, is_error=True)["content"],
            "<tool_response>\nerror: x\n</tool_response>",
        )
        self.assertEqual(
            _tool_turn("ok", architecture=None)["content"],
            "<tool_response>\nok\n</tool_response>",
        )


class ResponsesTextFormatTests(unittest.TestCase):
    def test_json_object_translates_to_response_format(self) -> None:
        payload = {"text": {"format": {"type": "json_object"}}}
        self.assertEqual(
            _responses_payload_with_format(payload)["response_format"],
            {"type": "json_object"},
        )

    def test_json_schema_keeps_name_schema_and_strict(self) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        payload = {
            "text": {"format": {"type": "json_schema", "name": "n",
                                "schema": schema, "strict": True}}
        }
        translated = _responses_payload_with_format(payload)["response_format"]
        self.assertEqual(translated["type"], "json_schema")
        self.assertEqual(translated["json_schema"]["schema"], schema)
        self.assertEqual(translated["json_schema"]["name"], "n")
        self.assertTrue(translated["json_schema"]["strict"])

    def test_plain_text_and_absent_are_left_alone(self) -> None:
        for payload in ({}, {"text": {}}, {"text": {"format": {"type": "text"}}}):
            self.assertIs(_responses_payload_with_format(payload), payload)
        explicit = {"text": {"format": {"type": "json_object"}},
                    "response_format": {"type": "json_object"}}
        self.assertIs(_responses_payload_with_format(explicit), explicit)

    def test_unknown_format_is_a_400(self) -> None:
        with self.assertRaises(APIError):
            _responses_payload_with_format({"text": {"format": {"type": "xml"}}})

    def test_response_echoes_the_requested_format(self) -> None:
        service = InferenceService("qwen-local", StubGenerator(), max_new_tokens=32)
        response = service.response(
            {
                "model": "qwen-local",
                "input": "say {}",
                "max_output_tokens": 4,
                "text": {"format": {"type": "json_object"}},
            }
        )
        self.assertEqual(response["text"]["format"], {"type": "json_object"})


class ContinuationKeyTests(unittest.TestCase):
    def test_tool_declarations_change_the_key(self) -> None:
        base = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        with_tools = [dict(base[0], tools=[{"name": "read"}]), base[1]]
        other_tools = [dict(base[0], tools=[{"name": "write"}]), base[1]]
        self.assertNotEqual(_chat_key(base), _chat_key(with_tools))
        self.assertNotEqual(_chat_key(with_tools), _chat_key(other_tools))
        self.assertEqual(_chat_key(with_tools), _chat_key([dict(m) for m in with_tools]))
        # Role and content stay at index 0 and 1: the prefix matcher reads them.
        self.assertEqual(_chat_key(base)[0][:2], ("system", "s"))
        # A plain message keeps the three-field key existing callers build.
        self.assertEqual(len(_chat_key(base)[0]), 3)
        self.assertEqual(len(_chat_key(with_tools)[0]), 4)


class GuardedHandlerTests(unittest.TestCase):
    """A failing GET must answer 500, not drop the connection."""

    def setUp(self) -> None:
        self.service = InferenceService("qwen-local", StubGenerator())
        self.server = FlyweightHTTPServer(("127.0.0.1", 0), create_handler(self.service))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=5
        )

    def tearDown(self) -> None:
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_health_failure_is_a_500_response(self) -> None:
        errors = StringIO()
        with patch.object(
            InferenceService, "health", side_effect=RuntimeError("runtime went away")
        ), redirect_stderr(errors):
            self.connection.request("GET", "/health")
            response = self.connection.getresponse()
            body = json.loads(response.read())
        self.assertEqual(response.status, 500)
        self.assertEqual(body["error"]["type"], "server_error")
        self.assertIn("runtime went away", errors.getvalue())
        # And the server is still serving afterwards.
        self.connection.request("GET", "/health")
        response = self.connection.getresponse()
        self.assertEqual(response.status, 200)
        response.read()

    def test_delete_of_an_unknown_route_is_a_404(self) -> None:
        self.connection.request("DELETE", "/nope")
        response = self.connection.getresponse()
        self.assertEqual(response.status, 404)
        response.read()


class NativeBindingArityTests(unittest.TestCase):
    """Every ctypes argtypes list must match the C parameter count.

    An argtypes list one entry longer than the C signature (which is how
    `flyweight_v2_gpu_decoder_attention_cached` shipped) makes ctypes reject
    every call with a TypeError before it reaches the library; one entry
    shorter passes garbage. The symbol-presence check cannot see either.
    """

    def test_argtypes_match_the_header(self) -> None:
        header = (
            Path(v2.__file__).resolve().parents[2]
            / "native" / "include" / "flyweight_v2.h"
        )
        if not header.is_file():
            self.skipTest("no checkout beside this install")
        text = header.read_text(encoding="utf-8")
        declared: dict[str, int] = {}
        for match in re.finditer(
            r"FLYWEIGHT_V2_API\s+[\w\s\*]+?\b(flyweight_v2_\w+)\s*\((.*?)\)\s*;",
            text, re.S,
        ):
            name, params = match.group(1), match.group(2).strip()
            if params in ("", "void"):
                declared[name] = 0
                continue
            # Function-pointer parameters carry their own parenthesised list.
            depth, count = 0, 1
            for char in params:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                elif char == "," and depth == 0:
                    count += 1
            declared[name] = count
        self.assertGreater(len(declared), 100)

        library = v2._library()
        mismatched = []
        for name, count in declared.items():
            function = getattr(library, name, None)
            if function is None:
                continue
            argtypes = getattr(function, "argtypes", None)
            if argtypes is None:
                continue
            if len(argtypes) != count:
                mismatched.append((name, len(argtypes), count))
        self.assertEqual(mismatched, [], "argtypes length differs from the C signature")


class StreamingStatusTests(unittest.TestCase):
    """A streaming request's 429 and 400 must be HTTP statuses, not SSE events."""

    def setUp(self) -> None:
        self.service = InferenceService(
            "qwen-local", StubGenerator(), max_concurrent_requests=1
        )
        self.server = FlyweightHTTPServer(("127.0.0.1", 0), create_handler(self.service))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=5
        )

    def tearDown(self) -> None:
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _post(self, path: str, payload: dict) -> http.client.HTTPResponse:
        self.connection.request(
            "POST", path, body=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        return self.connection.getresponse()

    def test_queue_full_is_a_429_on_a_streaming_request(self) -> None:
        payload = {
            "model": "qwen-local",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        with self.service._admission():  # the one slot is taken
            response = self._post("/v1/chat/completions", payload)
            body = json.loads(response.read())
        self.assertEqual(response.status, 429)
        self.assertEqual(response.getheader("Retry-After"), "1")
        self.assertEqual(body["error"]["type"], "rate_limit_error")
        # And the slot is free again afterwards: the stream works.
        response = self._post("/v1/chat/completions", payload)
        self.assertEqual(response.status, 200)
        self.assertTrue(response.getheader("Content-Type").startswith("text/event-stream"))
        response.read()

    def test_invalid_streaming_request_is_a_400(self) -> None:
        for path, payload in (
            ("/v1/chat/completions",
             {"model": "qwen-local", "messages": [{"role": "user", "content": "hi"}],
              "stream": True, "n": 2}),
            ("/v1/completions",
             {"model": "qwen-local", "prompt": "hi", "stream": True, "logprobs": 1}),
            ("/v1/messages",
             {"model": "qwen-local", "messages": [], "max_tokens": 4, "stream": True}),
            ("/v1/responses",
             {"model": "qwen-local", "input": "hi", "stream": True,
              "text": {"format": {"type": "xml"}}}),
        ):
            with self.subTest(path=path):
                response = self._post(path, payload)
                body = json.loads(response.read())
                self.assertEqual(response.status, 400, body)
                self.assertNotEqual(
                    response.getheader("Content-Type"), "text/event-stream; charset=utf-8"
                )


class UsageStubGenerator(StubGenerator):
    """Reports two of three prompt tokens reused, and thinks before answering."""

    TEXT = "<think>plan ahead</think>Hello!"

    def generate_messages(self, messages, **options) -> GenerationResult:
        progress = options.get("progress")
        if progress is not None:
            progress(2, 3)
            progress(3, 3)
        return GenerationResult(
            prompt_ids=(1, 2, 3), generated_ids=(4, 5, 6), text=self.TEXT,
            stopped_on_eos=True, state_tokens=6,
        )

    def stream_messages(self, messages, **options):
        from flyweight.generation import GenerationStep

        progress = options.get("progress")
        if progress is not None:
            progress(2, 3)
            progress(3, 3)
        generated: list[int] = []
        for token, delta in ((4, "<think>plan ahead"), (5, "</think>"), (6, "Hello!")):
            generated.append(token)
            yield GenerationStep(
                token_id=token, text_delta=delta, prompt_ids=(1, 2, 3),
                generated_ids=tuple(generated), text="", stopped_on_eos=False,
                finished=False, state_tokens=3 + len(generated),
            )
        yield GenerationStep(
            token_id=None, text_delta="", prompt_ids=(1, 2, 3),
            generated_ids=tuple(generated), text=self.TEXT, stopped_on_eos=True,
            finished=True, state_tokens=6,
        )


class UsageAccountingTests(unittest.TestCase):
    """cached_tokens and reasoning_tokens were hard-coded to zero."""

    # The stub tokenizer encodes one token per character, so the reasoning
    # "plan ahead" is ten tokens.
    REASONING = 10
    CACHED = 2

    def setUp(self) -> None:
        self.service = InferenceService("qwen-local", UsageStubGenerator(), max_new_tokens=32)
        self.chat = {
            "model": "qwen-local",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        }

    def test_chat_completion_usage_details(self) -> None:
        usage = self.service.chat_completion(self.chat)["usage"]
        self.assertEqual(usage["prompt_tokens"], 3)
        self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], self.CACHED)
        self.assertEqual(
            usage["completion_tokens_details"]["reasoning_tokens"], self.REASONING
        )

    def test_streamed_chat_usage_chunk_details(self) -> None:
        chunks = list(
            self.service.stream_chat_completion(
                {**self.chat, "stream": True, "stream_options": {"include_usage": True}}
            )
        )
        usage = [c["usage"] for c in chunks if isinstance(c, dict) and c.get("usage")]
        self.assertEqual(len(usage), 1)
        self.assertEqual(usage[0]["prompt_tokens_details"]["cached_tokens"], self.CACHED)
        self.assertEqual(
            usage[0]["completion_tokens_details"]["reasoning_tokens"], self.REASONING
        )

    def test_anthropic_usage_splits_cached_out_of_input(self) -> None:
        request = {"model": "qwen-local", "max_tokens": 8,
                   "messages": [{"role": "user", "content": "hi"}]}
        usage = self.service.anthropic_message(request)["usage"]
        self.assertEqual(usage["cache_read_input_tokens"], self.CACHED)
        self.assertEqual(usage["input_tokens"], 3 - self.CACHED)
        self.assertEqual(usage["output_tokens"], 3)
        deltas = [
            event for event in self.service.stream_anthropic_message(request)
            if event["type"] == "message_delta"
        ]
        self.assertEqual(deltas[-1]["usage"]["cache_read_input_tokens"], self.CACHED)
        self.assertEqual(deltas[-1]["usage"]["input_tokens"], 3 - self.CACHED)
        self.assertEqual(deltas[-1]["usage"]["output_tokens"], 3)

    def test_responses_usage_details(self) -> None:
        usage = self.service.response(
            {"model": "qwen-local", "input": "hi", "max_output_tokens": 8}
        )["usage"]
        self.assertEqual(usage["input_tokens_details"]["cached_tokens"], self.CACHED)
        self.assertEqual(usage["output_tokens_details"]["reasoning_tokens"], self.REASONING)

    def test_a_generator_without_progress_or_thinking_reports_zero(self) -> None:
        service = InferenceService("qwen-local", StubGenerator(), max_new_tokens=32)
        usage = service.chat_completion(self.chat)["usage"]
        self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], 0)
        self.assertEqual(usage["completion_tokens_details"]["reasoning_tokens"], 0)


class ToolCallPenaltyTests(unittest.TestCase):
    """Penalties pause inside an open tool call, where arguments are verbatim."""

    def test_sampler_gates_penalties_on_the_tool_grammar(self) -> None:
        root = Path(v2.__file__).resolve().parents[2]
        source = root / "native" / "src" / "v2_runtime.cpp"
        if not source.is_file():
            self.skipTest("no checkout beside this install")
        text = source.read_text(encoding="utf-8")
        self.assertIn(
            "sampling.penalizes()&&!sampling.recent.empty()&&!inside_tool_call", text
        )
        self.assertIn("sampling.grammar.armed()&&!penalize_tool_calls", text)
        self.assertIn("FLYWEIGHT_TOOL_CALL_PENALTY", text)
