import http.client
import json
import threading
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import Mock, patch

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from colibri_next.cli import main
from colibri_next.generation import GenerationResult, GenerationStep
from colibri_next.server import (
    APIError,
    ColibriHTTPServer,
    InferenceService,
    create_handler,
    _is_decode_event,
    _parse_tool_calls,
)


class StubTokenizer:
    eos_token_ids = (99,)

    def encode(self, text):
        return [ord(character) for character in text]

    def encode_messages(self, messages, *, enable_thinking=False):
        return self.encode("".join(message["content"] for message in messages))

    def decode(self, tokens, *, skip_special_tokens=True):
        return "".join(chr(token) for token in tokens)


class StubGenerator:
    def __init__(self) -> None:
        self.calls = []
        self.tokenizer = StubTokenizer()

    def generate_messages(self, messages, **options) -> GenerationResult:
        self.calls.append((messages, options))
        return GenerationResult(
            prompt_ids=(1, 2, 3),
            generated_ids=(4, 5),
            text="Hello!",
            stopped_on_eos=True,
            state_tokens=4,
        )

    def generate_text(self, prompt, **options) -> GenerationResult:
        return self.generate_messages([{"role": "user", "content": prompt}], **options)

    def stream_text(self, prompt, **options):
        return self.stream_messages([{"role": "user", "content": prompt}], **options)

    def stream_messages(self, messages, **options):
        self.calls.append((messages, options))
        yield GenerationStep(
            token_id=4,
            text_delta="Hello",
            prompt_ids=(1, 2, 3),
            generated_ids=(4,),
            text="Hello",
            stopped_on_eos=False,
            finished=False,
            state_tokens=3,
        )
        yield GenerationStep(
            token_id=5,
            text_delta="!",
            prompt_ids=(1, 2, 3),
            generated_ids=(4, 5),
            text="Hello!",
            stopped_on_eos=True,
            finished=False,
            state_tokens=4,
        )
        yield GenerationStep(
            token_id=None,
            text_delta="",
            prompt_ids=(1, 2, 3),
            generated_ids=(4, 5),
            text="Hello!",
            stopped_on_eos=True,
            finished=True,
            state_tokens=4,
        )


class ToolStubGenerator(StubGenerator):
    def generate_messages(self, messages, **options) -> GenerationResult:
        self.calls.append((messages, options))
        return GenerationResult(
            prompt_ids=(1, 2, 3),
            generated_ids=(4, 5),
            text=(
                "<tool_call>\n<function=get_weather>\n"
                "<parameter=city>\nParis\n</parameter>\n"
                "</function>\n</tool_call>"
            ),
            stopped_on_eos=True,
            state_tokens=4,
        )

    def stream_messages(self, messages, **options):
        text = (
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nParis\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        for index, char in enumerate(text):
            yield GenerationStep(
                token_id=index,
                text_delta=char,
                prompt_ids=(1, 2, 3),
                generated_ids=tuple(range(index + 1)),
                text=text[: index + 1],
                stopped_on_eos=False,
                finished=False,
                state_tokens=index + 1,
            )
        yield GenerationStep(
            token_id=None,
            text_delta="",
            prompt_ids=(1, 2, 3),
            generated_ids=tuple(range(len(text) + 1)),
            text=text,
            stopped_on_eos=True,
            finished=True,
            state_tokens=len(text) + 1,
        )


class BareStringIdToolStubGenerator(ToolStubGenerator):
    def generate_messages(self, messages, **options) -> GenerationResult:
        self.calls.append((messages, options))
        return GenerationResult(
            prompt_ids=(1, 2, 3),
            generated_ids=(4, 5),
            text=(
                "<tool_call>\n<function=TaskUpdate>\n"
                "<parameter=status>\ncompleted\n</parameter>\n"
                "<parameter=taskId>\n1\n</parameter>\n"
                "</function>\n</tool_call>"
            ),
            stopped_on_eos=True,
            state_tokens=4,
        )


class TruncatedToolStubGenerator(StubGenerator):
    """Emits a <tool_call> that never closes and stops without EOS, mimicking a
    tool call cut off by the output-token ceiling (the AskUserQuestion leak)."""

    TEXT = "I can help.\n<tool_call>\n<function=question>\n<parameter=questions>\n[{"

    def stream_messages(self, messages, **options):
        self.calls.append((messages, options))
        for index, char in enumerate(self.TEXT):
            yield GenerationStep(
                token_id=index,
                text_delta=char,
                prompt_ids=(1, 2, 3),
                generated_ids=tuple(range(index + 1)),
                text=self.TEXT[: index + 1],
                stopped_on_eos=False,
                finished=False,
                state_tokens=index + 1,
            )
        yield GenerationStep(
            token_id=None,
            text_delta="",
            prompt_ids=(1, 2, 3),
            generated_ids=tuple(range(len(self.TEXT) + 1)),
            text=self.TEXT,
            stopped_on_eos=False,  # truncated, not a clean stop
            finished=True,
            state_tokens=len(self.TEXT) + 1,
        )


class ToolCallParsingTests(unittest.TestCase):
    def test_parses_hermes_xml_format(self) -> None:
        content, calls = _parse_tool_calls(
            "sure\n<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
        )
        self.assertEqual(content, "sure")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"city": "Paris"}
        )

    def test_parses_json_format(self) -> None:
        # Qwen3/DeepSeek/GLM style: JSON object inside the <tool_call> block.
        content, calls = _parse_tool_calls(
            'thinking\n<tool_call>\n{"name": "get_weather", '
            '"arguments": {"city": "Paris"}}\n</tool_call>'
        )
        self.assertEqual(content, "thinking")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"city": "Paris"}
        )

    def test_schema_restores_string_ids_from_unquoted_hermes_values(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "TaskUpdate",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "taskId": {"type": "string"},
                            "status": {"type": "string"},
                        },
                        "required": ["taskId", "status"],
                    },
                },
            }
        ]
        _, calls = _parse_tool_calls(
            "<tool_call>\n<function=TaskUpdate>\n"
            "<parameter=status>\ncompleted\n</parameter>\n"
            "<parameter=taskId>\n1\n</parameter>\n"
            "</function>\n</tool_call>",
            tools=tools,
        )
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"status": "completed", "taskId": "1"},
        )

    def test_schema_normalization_recurses_through_arrays_and_objects(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "configure",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "enabled": {"type": "boolean"},
                                    },
                                },
                            }
                        },
                    },
                },
            }
        ]
        _, calls = _parse_tool_calls(
            '<tool_call>{"name":"configure","arguments":'
            '{"items":[{"id":7,"enabled":"true"}]}}</tool_call>',
            tools=tools,
        )
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"items": [{"id": "7", "enabled": True}]},
        )

    def test_plain_text_has_no_tool_calls(self) -> None:
        content, calls = _parse_tool_calls("just a normal answer")
        self.assertEqual(content, "just a normal answer")
        self.assertEqual(calls, [])

    def test_truncated_tool_call_does_not_leak_markup(self) -> None:
        # Model hit the token ceiling mid tool-call: the <tool_call> block never
        # closes. It must not leak raw markup as content (the AskUserQuestion
        # leak); content is only the clean prose before the marker.
        content, calls = _parse_tool_calls(
            "I can help.\n<tool_call>\n<function=question>\n"
            '<parameter=questions>\n[{"question": "Which framework?"'
        )
        self.assertEqual(content, "I can help.")
        self.assertEqual(calls, [])

    def test_malformed_closed_tool_call_does_not_leak_markup(self) -> None:
        # A complete but unparseable block (invalid JSON body) is also suppressed
        # rather than surfaced as assistant text.
        content, calls = _parse_tool_calls(
            "here goes\n<tool_call>\n{not valid json}\n</tool_call>"
        )
        self.assertEqual(content, "here goes")
        self.assertEqual(calls, [])

    def test_tool_call_only_output_yields_empty_content(self) -> None:
        content, calls = _parse_tool_calls("<tool_call>\n<function=question>\n")
        self.assertIsNone(content)
        self.assertEqual(calls, [])


class InferenceServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = StubGenerator()
        self.service = InferenceService("qwen-local", self.generator, max_new_tokens=32)

    def test_chat_completion_supports_developer_and_text_parts(self) -> None:
        response = self.service.chat_completion(
            {
                "model": "qwen-local",
                "messages": [
                    {"role": "developer", "content": "Be concise"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Con"},
                            {"type": "text", "text": "tinue"},
                        ],
                    },
                ],
                "max_tokens": 7,
                "temperature": 0,
            }
        )
        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["choices"][0]["message"]["content"], "Hello!")
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(response["usage"]["total_tokens"], 5)
        self.assertEqual(
            self.generator.calls[0][0],
            [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Continue"},
            ],
        )
        self.assertEqual(self.generator.calls[0][1]["max_new_tokens"], 7)

    def test_chat_completion_accepts_null_optional_sampling_options(self) -> None:
        response = self.service.chat_completion(
            {
                "model": "qwen-local",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": None,
                "top_p": None,
            }
        )
        self.assertEqual(response["choices"][0]["message"]["content"], "Hello!")

    def test_chat_completion_accepts_assistant_prefill(self) -> None:
        response = self.service.chat_completion(
            {
                "model": "qwen-local",
                "messages": [
                    {"role": "user", "content": "Complete this answer"},
                    {"role": "assistant", "content": "The answer is"},
                ],
            }
        )
        self.assertEqual(response["choices"][0]["message"]["content"], "Hello!")
        self.assertEqual(
            self.generator.calls[-1][0][-1],
            {"role": "assistant", "content": "The answer is"},
        )

    def test_client_model_id_is_a_routing_hint_by_default(self) -> None:
        response = self.service.chat_completion(
            {
                "model": "claude-fable-5",
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )
        self.assertEqual(response["choices"][0]["message"]["content"], "Hello!")
        strict = InferenceService("qwen-local", self.generator, strict_model=True)
        with self.assertRaises(APIError):
            strict.chat_completion(
                {
                    "model": "claude-fable-5",
                    "messages": [{"role": "user", "content": "Hi"}],
                }
            )

    def test_health_reports_cpu_moe_placement(self) -> None:
        service = InferenceService("qwen-local", self.generator, cpu_moe_layers=12)
        self.assertEqual(service.health()["cpu_moe_layers"], 12)

    def test_context_window_limits_combined_prompt_and_output(self) -> None:
        service = InferenceService(
            "qwen-local",
            self.generator,
            max_new_tokens=10,
            context_window=10,
        )
        accepted = service.chat_completion(
            {
                "messages": [{"role": "user", "content": "123456"}],
                "max_tokens": 4,
            }
        )
        self.assertEqual(accepted["object"], "chat.completion")
        # A 6-token prompt in a 10-token window leaves room for 4 output tokens;
        # requesting more is clamped rather than rejected (agentic clients treat
        # max_tokens as an upper bound).
        clamped = service.chat_completion(
            {
                "messages": [{"role": "user", "content": "123456"}],
                "max_tokens": 5,
            }
        )
        self.assertEqual(clamped["object"], "chat.completion")
        self.assertEqual(self.generator.calls[-1][1]["max_new_tokens"], 4)
        self.assertEqual(service.properties()["context_window"], 10)

    def test_context_window_clamps_legacy_text_completion(self) -> None:
        service = InferenceService(
            "qwen-local",
            self.generator,
            max_new_tokens=8,
            context_window=8,
        )
        # 5-token prompt in an 8-token window leaves room for 3 output tokens.
        service.completion({"prompt": "12345", "max_tokens": 4})
        self.assertEqual(self.generator.calls[-1][1]["max_new_tokens"], 3)

    def test_prompt_filling_context_window_is_rejected(self) -> None:
        service = InferenceService(
            "qwen-local",
            self.generator,
            max_new_tokens=8,
            context_window=4,
        )
        with self.assertRaisesRegex(APIError, "filling the"):
            service.completion({"prompt": "12345", "max_tokens": 1})

    def test_chat_stream_emits_deltas_usage_and_done(self) -> None:
        events = list(
            self.service.stream_chat_completion(
                {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            )
        )
        self.assertEqual(events[0]["object"], "chat.completion.chunk")
        self.assertEqual(events[0]["choices"][0]["delta"]["role"], "assistant")
        live_chunks = [
            event
            for event in events
            if isinstance(event, dict) and event.get("colibri")
        ]
        self.assertEqual(
            [event["colibri"]["generated_tokens"] for event in live_chunks],
            [1, 2],
        )
        decode_elapsed = [
            event["colibri"]["decode_elapsed_seconds"] for event in live_chunks
        ]
        self.assertEqual(decode_elapsed[0], 0.0)
        self.assertGreaterEqual(decode_elapsed[1], decode_elapsed[0])
        deltas = [
            event["choices"][0]["delta"].get("content", "")
            for event in events
            if isinstance(event, dict) and event.get("choices")
        ]
        self.assertEqual("".join(deltas), "Hello!")
        self.assertEqual(events[-2]["usage"]["total_tokens"], 5)
        self.assertEqual(events[-1], "[DONE]")

    def test_responses_api_returns_output_text_shape(self) -> None:
        response = self.service.response(
            {
                "model": "qwen-local",
                "instructions": "Be concise",
                "input": "Say hi",
                "max_output_tokens": 4,
            }
        )
        self.assertEqual(response["object"], "response")
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["output"][0]["content"][0]["text"], "Hello!")
        self.assertEqual(response["usage"]["total_tokens"], 5)

    def test_anthropic_messages_api_returns_anthropic_shape(self) -> None:
        response = self.service.anthropic_message(
            {
                "model": "claude-fable-5",
                "system": "Be concise",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 4,
            }
        )
        self.assertEqual(response["type"], "message")
        self.assertEqual(response["role"], "assistant")
        self.assertEqual(response["content"][0]["type"], "text")
        self.assertEqual(response["content"][0]["text"], "Hello!")

        events = list(
            self.service.stream_anthropic_message(
                {
                    "model": "claude-fable-5",
                    "messages": [{"role": "user", "content": "Say hi"}],
                    "max_tokens": 4,
                }
            )
        )
        self.assertEqual(events[0]["type"], "message_start")
        self.assertTrue(any(event["type"] == "content_block_delta" for event in events))
        self.assertEqual(events[-1]["type"], "message_stop")

    def test_anthropic_stream_text_then_tool_use_gets_separate_blocks(self) -> None:
        chat_events = [
            {"choices": [{"index": 0, "delta": {"content": "Let me check."}}]},
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "toolu_abc",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":',
                                    },
                                }
                            ]
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "toolu_abc",
                                    "function": {"arguments": ' "Paris"}'},
                                }
                            ]
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            "[DONE]",
        ]

        def fake_stream(payload, **kwargs):
            for ev in chat_events:
                yield ev

        self.service.stream_chat_completion = fake_stream
        events = list(
            self.service.stream_anthropic_message(
                {"model": "qwen-local", "messages": [{"role": "user", "content": "hi"}]}
            )
        )

        starts = [e for e in events if e["type"] == "content_block_start"]
        self.assertEqual(len(starts), 2)
        self.assertEqual(starts[0]["index"], 0)
        self.assertEqual(starts[0]["content_block"]["type"], "text")
        self.assertEqual(starts[1]["index"], 1)
        self.assertEqual(starts[1]["content_block"]["type"], "tool_use")
        self.assertEqual(starts[1]["content_block"]["id"], "toolu_abc")
        self.assertEqual(starts[1]["content_block"]["name"], "get_weather")

        tool_deltas = [
            e for e in events if e["type"] == "content_block_delta" and e["index"] == 1
        ]
        self.assertEqual(tool_deltas[0]["delta"]["type"], "input_json_delta")
        self.assertEqual(
            "".join(d["delta"]["partial_json"] for d in tool_deltas),
            '{"city": "Paris"}',
        )

        text_deltas = [
            e for e in events if e["type"] == "content_block_delta" and e["index"] == 0
        ]
        self.assertEqual(
            "".join(d["delta"]["text"] for d in text_deltas), "Let me check."
        )

        stops = [e for e in events if e["type"] == "content_block_stop"]
        self.assertEqual([e["index"] for e in stops], [0, 1])

    def test_anthropic_messages_accepts_embedded_system_and_tool_roles(self) -> None:
        response = self.service.anthropic_message(
            {
                "model": "claude-fable-5",
                "messages": [
                    {"role": "system", "content": "Be concise"},
                    {"role": "user", "content": "Use the tool"},
                    {"role": "tool", "content": "Tool result"},
                ],
                "max_tokens": 4,
            }
        )
        self.assertEqual(response["type"], "message")

    def test_anthropic_canonical_tools_are_forwarded(self) -> None:
        service = InferenceService("qwen-local", ToolStubGenerator())
        response = service.anthropic_message(
            {
                "model": "claude-fable-5",
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
                "tool_choice": {"type": "any"},
                "max_tokens": 4,
            }
        )
        self.assertEqual(response["stop_reason"], "tool_use")
        self.assertEqual(response["content"][0]["type"], "tool_use")

    def test_anthropic_tool_arguments_follow_input_schema(self) -> None:
        service = InferenceService("qwen-local", BareStringIdToolStubGenerator())
        response = service.anthropic_message(
            {
                "model": "claude-fable-5",
                "messages": [{"role": "user", "content": "Complete task 1"}],
                "tools": [
                    {
                        "name": "TaskUpdate",
                        "description": "Update a task",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "taskId": {"type": "string"},
                                "status": {"type": "string"},
                            },
                            "required": ["taskId", "status"],
                        },
                    }
                ],
                "max_tokens": 16,
            }
        )
        self.assertEqual(response["stop_reason"], "tool_use")
        self.assertEqual(
            response["content"][0]["input"],
            {"status": "completed", "taskId": "1"},
        )

    def test_function_calling_uses_native_qwen_format(self) -> None:
        service = InferenceService("qwen-local", ToolStubGenerator())
        payload = {
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        }
        response = service.chat_completion(payload)
        choice = response["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        call = choice["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "get_weather")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Paris"})

        stream_events = list(
            service.stream_chat_completion({**payload, "stream": True})
        )
        tool_chunks = [
            event
            for event in stream_events
            if isinstance(event, dict)
            and event.get("choices")
            and event["choices"][0]["delta"].get("tool_calls")
        ]
        self.assertEqual(
            tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "get_weather",
        )

    def test_empty_content_tool_call_history_is_accepted(self) -> None:
        # opencode/Cline send assistant tool-call turns as {"content": "",
        # "tool_calls": [...]} and empty tool outputs; neither must 400.
        response = self.service.chat_completion(
            {
                "model": "qwen-local",
                "messages": [
                    {"role": "user", "content": "weather in Paris?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "content": "", "tool_call_id": "call_1"},
                    {"role": "user", "content": "thanks"},
                ],
            }
        )
        self.assertEqual(response["choices"][0]["message"]["content"], "Hello!")
        # The empty-content assistant turn is rendered as its tool call, not dropped.
        history = self.generator.calls[-1][0]
        assistant_turn = next(m for m in history if m["role"] == "assistant")
        self.assertIn("<tool_call>", assistant_turn["content"])
        self.assertIn("get_weather", assistant_turn["content"])

    def test_truncated_tool_call_stream_suppresses_markup(self) -> None:
        service = InferenceService("qwen-local", TruncatedToolStubGenerator())
        payload = {
            "messages": [{"role": "user", "content": "Build me an OS"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "question",
                        "description": "Ask the user",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "stream": True,
        }
        events = list(service.stream_chat_completion(payload))
        content = "".join(
            event["choices"][0]["delta"].get("content", "")
            for event in events
            if isinstance(event, dict) and event.get("choices")
        )
        # Clean prose before the marker is fine; raw tool-call markup must never
        # reach the client as assistant text.
        self.assertNotIn("<tool_call>", content)
        self.assertNotIn("<function=", content)
        self.assertNotIn("<parameter=", content)
        self.assertEqual(content.strip(), "I can help.")
        finish_reasons = [
            event["choices"][0].get("finish_reason")
            for event in events
            if isinstance(event, dict) and event.get("choices")
        ]
        # Truncated (no EOS) tool call reports "length", signalling the cutoff.
        self.assertIn("length", finish_reasons)

    def test_responses_function_tools_stream_and_continue(self) -> None:
        service = InferenceService("qwen-local", ToolStubGenerator())
        tool = {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
        payload = {
            "input": "Weather in Paris?",
            "tools": [tool],
            "tool_choice": "required",
            "max_output_tokens": 8,
        }

        response = service.response(payload)
        call = response["output"][0]
        self.assertEqual(call["type"], "function_call")
        self.assertEqual(call["name"], "get_weather")
        self.assertEqual(json.loads(call["arguments"]), {"city": "Paris"})

        events = list(service.stream_response({**payload, "stream": True}))
        event_types = [event["type"] for event in events]
        self.assertIn("response.function_call_arguments.delta", event_types)
        self.assertIn("response.function_call_arguments.done", event_types)
        self.assertEqual(events[-1]["response"]["output"][0]["type"], "function_call")

        service.response(
            {
                "previous_response_id": response["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": '{"temperature": 24}',
                    }
                ],
                "max_output_tokens": 8,
            }
        )
        continued_messages = service.generator.calls[-1][0]
        self.assertIn("<tool_call>", continued_messages[-2]["content"])
        self.assertIn("<tool_response>", continued_messages[-1]["content"])

    def test_previous_response_state_retrieve_and_delete(self) -> None:
        first = self.service.response({"input": "First", "max_output_tokens": 2})
        second = self.service.response(
            {
                "input": "Second",
                "previous_response_id": first["id"],
                "max_output_tokens": 2,
            }
        )
        second_messages = self.generator.calls[1][0]
        self.assertEqual(second["previous_response_id"], first["id"])
        self.assertEqual(
            second_messages[-2], {"role": "assistant", "content": "Hello!"}
        )
        self.assertEqual(self.service.retrieve_response(first["id"])["id"], first["id"])
        deleted = self.service.delete_response(first["id"])
        self.assertTrue(deleted["deleted"])
        with self.assertRaises(APIError):
            self.service.retrieve_response(first["id"])

    def test_rejects_invalid_history_and_tools(self) -> None:
        with self.assertRaises(APIError):
            self.service.chat_completion(
                {"messages": [{"role": "system", "content": "Hi"}]}
            )
        with self.assertRaises(APIError) as tools_error:
            self.service.chat_completion(
                {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "tools": [{"type": "function"}],
                }
            )
        self.assertEqual(tools_error.exception.parameter, "tools")

    def test_decode_event_detection_ignores_protocol_events(self) -> None:
        self.assertFalse(
            _is_decode_event(
                {"object": "chat.completion.chunk", "choices": [{"delta": {}}]}
            )
        )
        self.assertTrue(
            _is_decode_event(
                {
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": "Hi"}}],
                }
            )
        )
        self.assertTrue(
            _is_decode_event({"type": "response.output_text.delta", "delta": "Hi"})
        )


class HTTPServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = StubGenerator()
        self.service = InferenceService("qwen-local", self.generator)
        self.server = ColibriHTTPServer(("127.0.0.1", 0), create_handler(self.service))
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

    def request_json(self, method: str, path: str, payload=None):
        body = None if payload is None else json.dumps(payload)
        headers = {} if payload is None else {"Content-Type": "application/json"}
        self.connection.request(method, path, body=body, headers=headers)
        response = self.connection.getresponse()
        return response, json.loads(response.read())

    def test_chat_ui_static_assets(self) -> None:
        self.connection.request("GET", "/")
        response = self.connection.getresponse()
        html = response.read().decode("utf-8")
        self.assertEqual(response.status, 200)
        self.assertTrue(response.getheader("Content-Type").startswith("text/html"))
        self.assertIn(
            "default-src 'self'", response.getheader("Content-Security-Policy")
        )
        self.assertIn("Colibri Chat", html)

        self.connection.request("GET", "/app.js")
        response = self.connection.getresponse()
        javascript = response.read().decode("utf-8")
        self.assertEqual(response.status, 200)
        self.assertTrue(
            response.getheader("Content-Type").startswith("text/javascript")
        )
        self.assertIn("/v1/chat/completions", javascript)
        self.assertIn("decodeIntervals", javascript)

        self.connection.request("GET", "/preview.html")
        response = self.connection.getresponse()
        preview = response.read().decode("utf-8")
        self.assertEqual(response.status, 200)
        self.assertTrue(response.getheader("Content-Type").startswith("text/html"))
        policy = response.getheader("Content-Security-Policy")
        self.assertIn("sandbox allow-scripts", policy)
        self.assertIn("frame-ancestors 'self'", policy)
        self.assertIn("location.hash", preview)

    def test_health_models_chat_and_responses_endpoints(self) -> None:
        response, health = self.request_json("GET", "/health")
        self.assertEqual(response.status, 200)
        self.assertEqual(health["status"], "ok")

        _, models = self.request_json("GET", "/v1/models")
        self.assertEqual(models["data"][0]["id"], "qwen-local")
        _, model = self.request_json("GET", "/v1/models/qwen-local")
        self.assertEqual(model["object"], "model")

        _, completion = self.request_json(
            "POST",
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 2},
        )
        self.assertEqual(completion["choices"][0]["message"]["role"], "assistant")

        _, response_object = self.request_json(
            "POST",
            "/v1/responses",
            {"model": "qwen-local", "input": "Hi", "max_output_tokens": 2},
        )
        self.assertEqual(response_object["object"], "response")
        _, tokenized = self.request_json("POST", "/tokenize", {"content": "Hi"})
        self.assertEqual(tokenized["count"], 2)
        _, token_count = self.request_json(
            "POST", "/v1/responses/input_tokens", {"input": "Hi"}
        )
        self.assertEqual(token_count["input_tokens"], 2)
        _, properties = self.request_json("GET", "/props")
        self.assertEqual(properties["max_output_tokens"], 64)

    def test_cli_compatibility_probes(self) -> None:
        self.connection.request("HEAD", "/")
        response = self.connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.read(), b"")

        _, profile = self.request_json("GET", "/v1/me")
        self.assertEqual(profile["object"], "user")

        _, profile = self.request_json("POST", "/v1/me", {})
        self.assertEqual(profile["id"], "local")

    def test_chat_stream_uses_sse_framing(self) -> None:
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        self.connection.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = self.connection.getresponse()
        stream_body = response.read().decode("utf-8")
        self.assertEqual(response.status, 200)
        self.assertTrue(
            response.getheader("Content-Type").startswith("text/event-stream")
        )
        self.assertIn('"object": "chat.completion.chunk"', stream_body)
        self.assertIn("data: [DONE]", stream_body)


class AuthenticationTests(unittest.TestCase):
    def test_bearer_auth_and_cors_preflight(self) -> None:
        server = ColibriHTTPServer(
            ("127.0.0.1", 0),
            create_handler(
                InferenceService(
                    "qwen-local",
                    StubGenerator(),
                    api_key="secret",
                    cors_origin="http://localhost:3000",
                )
            ),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=5
        )
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 200)

            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 401)
            self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")

            connection.request(
                "GET", "/v1/models", headers={"Authorization": "Bearer secret"}
            )
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 200)

            connection.request("OPTIONS", "/v1/chat/completions")
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 204)
            self.assertEqual(
                response.getheader("Access-Control-Allow-Origin"),
                "http://localhost:3000",
            )
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


@unittest.skipIf(OpenAI is None, "OpenAI SDK is not installed")
class OpenAISDKTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ColibriHTTPServer(
            ("127.0.0.1", 0),
            create_handler(InferenceService("qwen-local", StubGenerator())),
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = OpenAI(
            base_url=f"http://127.0.0.1:{self.server.server_port}/v1",
            api_key="local",
        )

    def tearDown(self) -> None:
        self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_chat_stream_and_responses_clients(self) -> None:
        chat = self.client.chat.completions.create(
            model="qwen-local",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=2,
        )
        self.assertEqual(chat.choices[0].message.content, "Hello!")

        stream = self.client.chat.completions.create(
            model="qwen-local",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=2,
            stream=True,
        )
        streamed_text = "".join(
            chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices
        )
        self.assertEqual(streamed_text, "Hello!")

        response = self.client.responses.create(
            model="qwen-local", input="Hi", max_output_tokens=2
        )
        self.assertEqual(response.output_text, "Hello!")
        response_stream = self.client.responses.create(
            model="qwen-local", input="Hi", max_output_tokens=2, stream=True
        )
        response_text = "".join(
            event.delta
            for event in response_stream
            if event.type == "response.output_text.delta"
        )
        self.assertEqual(response_text, "Hello!")

        completion = self.client.completions.create(
            model="qwen-local", prompt="Hi", max_tokens=2
        )
        self.assertEqual(completion.choices[0].text, "Hello!")


class ServerCLITests(unittest.TestCase):
    @patch("colibri_next.cli.serve_http")
    @patch("colibri_next.cli.InferenceService.from_model_directory")
    def test_serve_command_loads_once_and_starts_http(
        self, load_model, serve_http
    ) -> None:
        load_model.return_value = Mock(model_name="demo-model")
        with redirect_stderr(StringIO()):
            result = main(
                [
                    "serve",
                    "model-root",
                    "--port",
                    "9012",
                    "--max-new-tokens",
                    "24",
                    "--cpu-moe-layers",
                    "10",
                ]
            )
        self.assertEqual(result, 0)
        load_model.assert_called_once()
        self.assertEqual(load_model.call_args.kwargs["cpu_moe_layers"], 10)
        serve_http.assert_called_once_with(
            load_model.return_value, host="127.0.0.1", port=9012
        )


if __name__ == "__main__":
    unittest.main()
