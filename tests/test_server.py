import http.client
import itertools
import json
import re
import socket
import threading
import time
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from io import StringIO
from unittest.mock import Mock, patch

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from colibri_next.cli import main
from colibri_next.generation import GenerationResult, GenerationStep
from colibri_next.sampling import SamplingConfig
from colibri_next.server import (
    APIError,
    ColibriHTTPServer,
    InferenceService,
    create_handler,
    _is_decode_event,
    _parse_tool_calls,
    _split_reasoning_content,
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
        self.prepare_calls = []
        self.tokenizer = StubTokenizer()

    def prepare_messages(self, messages, **options):
        self.prepare_calls.append((messages, options))
        return self.tokenizer.encode_messages(
            messages,
            enable_thinking=bool(options.get("enable_thinking", False)),
        )

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


class LengthStubGenerator(StubGenerator):
    def generate_messages(self, messages, **options) -> GenerationResult:
        return replace(
            super().generate_messages(messages, **options), stopped_on_eos=False
        )

    def stream_messages(self, messages, **options):
        for step in super().stream_messages(messages, **options):
            yield replace(step, stopped_on_eos=False)


class ToolStubGenerator(StubGenerator):
    TOOL_TEXT = (
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nParis\n</parameter>\n"
        "</function>\n</tool_call>"
    )

    def generate_messages(self, messages, **options) -> GenerationResult:
        self.calls.append((messages, options))
        return GenerationResult(
            prompt_ids=(1, 2, 3),
            generated_ids=(4, 5),
            text=self.TOOL_TEXT,
            stopped_on_eos=True,
            state_tokens=4,
        )

    def stream_messages(self, messages, **options):
        self.calls.append((messages, options))
        text = self.TOOL_TEXT
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


class RunawayToolStubGenerator(ToolStubGenerator):
    """Keeps generating after a valid tool call unless its iterator is closed."""

    TOOL_TEXT = (
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nParis\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    RUNAWAY_SUFFIX = " this output must never be consumed" * 20

    def __init__(self) -> None:
        super().__init__()
        self.consumed = 0
        self.closed = False

    def stream_messages(self, messages, **options):
        self.calls.append((messages, options))
        text = self.TOOL_TEXT + self.RUNAWAY_SUFFIX
        try:
            for index, char in enumerate(text):
                self.consumed += 1
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
                generated_ids=tuple(range(len(text))),
                text=text,
                stopped_on_eos=False,
                finished=True,
                state_tokens=len(text),
            )
        finally:
            self.closed = True


class BareStringIdToolStubGenerator(ToolStubGenerator):
    TOOL_TEXT = (
        "<tool_call>\n<function=TaskUpdate>\n"
        "<parameter=status>\ncompleted\n</parameter>\n"
        "<parameter=taskId>\n1\n</parameter>\n"
        "</function>\n</tool_call>"
    )

    def generate_messages(self, messages, **options) -> GenerationResult:
        self.calls.append((messages, options))
        return GenerationResult(
            prompt_ids=(1, 2, 3),
            generated_ids=(4, 5),
            text=self.TOOL_TEXT,
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
    def test_splits_deepseek_reasoning_from_visible_content(self) -> None:
        visible, reasoning = _split_reasoning_content(
            "<think>considering options</think>final answer"
        )
        self.assertEqual(visible, "final answer")
        self.assertEqual(reasoning, "considering options")

    def test_strips_non_thinking_protocol_close(self) -> None:
        self.assertEqual(_split_reasoning_content("</think>answer"), ("answer", None))

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

    def test_hermes_recovers_a_parameter_missing_its_closing_tag(self) -> None:
        # Qwen3.5 at a low-bit quantization drops </parameter> often enough to
        # stall an agent loop: the call used to decode to no arguments, fail the
        # required-parameter check, and reach the client as prose with no tool
        # call at all.
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                },
            }
        ]
        content, calls = _parse_tool_calls(
            "exploring\n<tool_call>\n<function=read>\n"
            "<parameter=file_path>\n/app/package.json\n</function>\n</tool_call>",
            tools=tools,
        )
        self.assertEqual(content, "exploring")
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"file_path": "/app/package.json"},
        )

    def test_unclosed_recovery_leaves_a_closed_value_verbatim(self) -> None:
        # The loose pass must not truncate a well-formed value at text that
        # merely looks like a closing tag.
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "Write",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            }
        ]
        _, calls = _parse_tool_calls(
            "<tool_call>\n<function=Write>\n"
            "<parameter=path>\n/a.txt\n</parameter>\n"
            "<parameter=content>\nline1\n</function>\nline2\n</parameter>\n"
            "</function>\n</tool_call>",
            tools=tools,
        )
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"path": "/a.txt", "content": "line1\n</function>\nline2"},
        )

    EDIT_TOOL = [
        {
            "type": "function",
            "function": {
                "name": "Edit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                    },
                },
            },
        }
    ]

    @staticmethod
    def _hermes(name: str, **parameters: str) -> str:
        body = "".join(
            f"<parameter={key}>\n{value}\n</parameter>\n"
            for key, value in parameters.items()
        )
        return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"

    def test_hermes_parameter_keeps_leading_indentation(self) -> None:
        # Edit matches old_string byte-for-byte. Stripping the value's
        # surrounding whitespace ate the first line's indentation, so every
        # edit to indented code failed and the model resorted to shell edits.
        old = "    def foo(self):\n        return 1"
        _, calls = _parse_tool_calls(
            self._hermes("Edit", file_path="/tmp/a.py", old_string=old),
            tools=self.EDIT_TOOL,
        )
        arguments = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(arguments["old_string"], old)

    def test_hermes_string_parameter_is_not_reinterpreted_as_json(self) -> None:
        # Editing a JSON file sends content that happens to parse. Inferring a
        # type from it replaced the declared string with a dict, and
        # re-serializing changed the very whitespace the edit had to match.
        old = '{"a": 1,\n "b": 2}'
        _, calls = _parse_tool_calls(
            self._hermes("Edit", file_path="/tmp/a.json", old_string=old),
            tools=self.EDIT_TOOL,
        )
        arguments = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(arguments["old_string"], old)

    def test_hermes_declared_scalars_are_still_typed(self) -> None:
        _, calls = _parse_tool_calls(
            self._hermes("Edit", file_path="/tmp/a.py", replace_all="true"),
            tools=self.EDIT_TOOL,
        )
        arguments = json.loads(calls[0]["function"]["arguments"])
        self.assertIs(arguments["replace_all"], True)

    def test_hermes_declared_object_parameter_is_decoded(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "Configure",
                    "parameters": {
                        "type": "object",
                        "properties": {"options": {"type": "object"}},
                    },
                },
            }
        ]
        _, calls = _parse_tool_calls(
            self._hermes("Configure", options='{"depth": 2}'), tools=tools
        )
        arguments = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(arguments["options"], {"depth": 2})

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

    def test_parses_deepseek_dsml_with_multiple_invocations(self) -> None:
        content, calls = _parse_tool_calls(
            "thinking\n<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="weather">\n'
            '<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>\n'
            '</｜DSML｜invoke>\n'
            '<｜DSML｜invoke name="notify">\n'
            '<｜DSML｜parameter name="urgent" string="false">true</｜DSML｜parameter>\n'
            '</｜DSML｜invoke>\n</｜DSML｜tool_calls>'
        )
        self.assertEqual(content, "thinking")
        self.assertEqual([call["function"]["name"] for call in calls], ["weather", "notify"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Paris", "days": 3})
        self.assertEqual(json.loads(calls[1]["function"]["arguments"]), {"urgent": True})

    def test_truncated_deepseek_dsml_does_not_leak_markup(self) -> None:
        content, calls = _parse_tool_calls(
            'clean\n<｜DSML｜tool_calls><｜DSML｜invoke name="weather">'
        )
        self.assertEqual(content, "clean")
        self.assertEqual(calls, [])

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

    def test_native_tool_architecture_keeps_the_round_trip_structured(self) -> None:
        # What a harness sends on the turn after a tool call: the call it got
        # back, then the result. For an architecture whose template renders
        # both, they must survive as structure -- and a tool result is a legal
        # final message, which the role check used to reject even though its
        # own error message said otherwise.
        from colibri_next.server import _chat_messages

        payload = {
            "messages": [
                {"role": "user", "content": "Read it"},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "read_file",
                                 "arguments": '{"path": "/etc/hostname"}'},
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "colibri-dev"},
            ],
            "tools": [{"type": "function", "function": {
                "name": "read_file",
                "parameters": {"type": "object",
                               "properties": {"path": {"type": "string"}}},
            }}],
        }
        messages, enabled = _chat_messages(payload, architecture="bailingmoe3")

        self.assertTrue(enabled)
        # Schemas ride on the first message, where the template picks them up.
        self.assertEqual(messages[0]["tools"][0]["function"]["name"], "read_file")
        # Arguments arrive as a JSON string and templates iterate them as a
        # mapping, so they are decoded rather than passed through.
        self.assertEqual(
            messages[1]["tool_calls"][0]["function"]["arguments"],
            {"path": "/etc/hostname"},
        )
        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[2]["content"], "colibri-dev")

    def test_tools_survive_a_system_prompt_on_a_native_architecture(self) -> None:
        # The shape every coding harness sends: a system prompt AND tools. The
        # schemas were attached to the first message and the system turn was
        # then inserted ahead of it, so the message the template reads them
        # from no longer had any -- the model was told about no tools at all.
        from colibri_next.server import _chat_messages

        payload = {
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Read it"},
            ],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
        }
        messages, enabled = _chat_messages(payload, architecture="bailingmoe3")

        self.assertTrue(enabled)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["tools"][0]["function"]["name"], "read_file")

    def test_anthropic_tool_round_trip_reaches_a_native_template(self) -> None:
        # Claude Code's wire shape: system as its own field, tool_use blocks on
        # the assistant turn, tool_result blocks inside the following user turn.
        from colibri_next.server import _anthropic_to_chat_payload, _chat_messages

        payload = _anthropic_to_chat_payload({
            "model": "m",
            "max_tokens": 64,
            "system": "You are a coding assistant.",
            "tools": [{"name": "read_file", "description": "Read a file.",
                       "input_schema": {"type": "object",
                                        "properties": {"path": {"type": "string"}}}}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Read it"}]},
                {"role": "assistant", "content": [{
                    "type": "tool_use", "id": "toolu_1", "name": "read_file",
                    "input": {"path": "/app/main.py"}}]},
                {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": "toolu_1",
                    "content": "import server"}]},
            ],
        })
        messages, enabled = _chat_messages(payload, architecture="bailingmoe3")

        self.assertTrue(enabled)
        self.assertEqual(messages[0]["tools"][0]["function"]["name"], "read_file")
        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user", "assistant", "tool"],
        )
        self.assertEqual(
            messages[2]["tool_calls"][0]["function"]["arguments"],
            {"path": "/app/main.py"},
        )

    def test_generic_architecture_still_renders_tools_into_content(self) -> None:
        # The Hermes prompt and the rendered history remain for checkpoints
        # whose template knows nothing about tools.
        from colibri_next.server import _chat_messages

        payload = {
            "messages": [
                {"role": "user", "content": "Read it"},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "colibri-dev"},
            ],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
        }
        messages, _ = _chat_messages(payload, architecture="qwen3")

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("read_file", messages[0]["content"])
        self.assertNotIn("tool_calls", messages[2])
        self.assertEqual(messages[3]["role"], "user")
        self.assertIn("<tool_response>", messages[3]["content"])

    def test_reasoning_opened_by_the_prompt_is_not_visible_content(self) -> None:
        # A reasoning checkpoint is asked to think by ending the PROMPT with an
        # open <think>, so the turn carries only the closing tag. Treated as
        # content, the model's private plan was served as its answer -- it read
        # as the model saying what it was about to do, then doing it.
        from colibri_next.server import _split_reasoning_content

        visible, reasoning = _split_reasoning_content(
            "I should call the tool.\n</think>Here is the answer."
        )
        self.assertEqual(visible, "Here is the answer.")
        self.assertEqual(reasoning, "I should call the tool.")

    def test_a_turn_without_thinking_is_untouched(self) -> None:
        from colibri_next.server import _split_reasoning_content

        visible, reasoning = _split_reasoning_content("Just an answer.")
        self.assertEqual(visible, "Just an answer.")
        self.assertIsNone(reasoning)

    def test_streaming_reasoning_is_split_from_the_answer(self) -> None:
        # The closing tag may arrive in pieces, and nothing before it may be
        # streamed to the client as content.
        from colibri_next.server import ThinkingPrefixStream

        stream = ThinkingPrefixStream()
        deltas = [stream.feed(part)
                  for part in ("Plan: ", "call it.", "</th", "ink>", "Done.")]
        visible = "".join(v for v, _ in deltas)
        reasoning = "".join(r for _, r in deltas)
        self.assertEqual(visible, "Done.")
        self.assertEqual(reasoning, "Plan: call it.")
        self.assertEqual(stream.flush(), ("", ""))

    def test_bailing_tagged_tool_call_is_decoded(self) -> None:
        # BailingMoE3 names the function on the opening line and tags each
        # argument. Parsed as neither Hermes nor JSON, the call decoded to no
        # name and was dropped -- along with the content before it, so a model
        # that had just called a tool appeared to answer with nothing.
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["path"],
                    },
                },
            }
        ]
        content, calls = _parse_tool_calls(
            "Let me look.\n<tool_call>read_file\n"
            "<arg_key>path</arg_key>\n<arg_value>/etc/hostname</arg_value>\n"
            "<arg_key>limit</arg_key>\n<arg_value>10</arg_value>\n"
            "</tool_call>",
            tools=tools,
        )
        self.assertEqual(content, "Let me look.")
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            # limit follows the declared schema, as it does for every format.
            {"path": "/etc/hostname", "limit": 10},
        )

    def test_bailing_tagged_tool_value_keeps_its_whitespace(self) -> None:
        # The value may be file content, where leading indentation is data.
        _, calls = _parse_tool_calls(
            "<tool_call>write\n<arg_key>text</arg_key>\n"
            "<arg_value>\n    indented\n</arg_value>\n</tool_call>"
        )
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"])["text"], "    indented"
        )

    def test_incomplete_required_tool_call_is_not_exposed(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "filePath": {"type": "string"},
                        },
                        "required": ["content", "filePath"],
                    },
                },
            }
        ]
        _, calls = _parse_tool_calls(
            "<tool_call><function=write>\n"
            "<parameter=content>\nhello\n</parameter>\n"
            "</function></tool_call>",
            tools=tools,
        )
        self.assertEqual(calls, [])

    def test_a_finished_call_missing_a_parameter_still_reaches_the_client(self) -> None:
        """Dropping is for truncation, not for a model that left a field out.

        Qwen omits the parameters that read as optional to it -- a
        `description` beside a `command` is the common one -- and returning
        nothing for those leaves the caller with an empty assistant turn and no
        way to tell a bad call from a bug. The call goes out and the caller's
        own validator says what is missing, which an agent can retry from.
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["command", "description"],
                    },
                },
            }
        ]
        emitted = ("<tool_call><function=bash>\n"
                   "<parameter=command>\nls -la\n</parameter>\n"
                   "</function></tool_call>")
        # The generation ended on its own: the model had its chance.
        _, finished = _parse_tool_calls(emitted, tools=tools, keep_incomplete=True)
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0]["function"]["name"], "bash")
        self.assertEqual(
            json.loads(finished[0]["function"]["arguments"]), {"command": "ls -la"}
        )
        # Cut short by the token budget: the parameters were never written, so
        # there is no call to make and finish_reason carries the reason.
        _, truncated = _parse_tool_calls(emitted, tools=tools)
        self.assertEqual(truncated, [])

    def test_parameters_after_the_closing_tag_are_recovered(self) -> None:
        # The shape a low-bit Qwen3.5 actually emits: <tool_call> and
        # </tool_call> are single tokens and come out right, while </function>
        # and </parameter> are spelled from ordinary subwords and get
        # misordered, closing the block before the arguments are written.
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "parameters": {
                        "type": "object",
                        "properties": {"execute": {"type": "boolean"}},
                        "required": ["execute"],
                    },
                },
            }
        ]
        content, calls = _parse_tool_calls(
            "I'll explore the project structure.\n"
            "<tool_call>\n<function=list_files>\n</function>\n</tool_call>\n"
            "<parameter=execute>\ntrue\n</parameter>\n</function>\n</tool_call>",
            tools=tools,
        )
        self.assertEqual(content, "I'll explore the project structure.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "list_files")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"execute": True}
        )

    def test_trailing_recovery_does_not_cross_into_the_next_call(self) -> None:
        # Two well-formed calls: the first must not absorb the second's
        # parameters just because they follow its closing tag.
        _, calls = _parse_tool_calls(
            "<tool_call><function=get_weather>\n"
            "<parameter=city>\nParis\n</parameter>\n"
            "</function></tool_call>\n"
            "<tool_call><function=get_time>\n"
            "<parameter=zone>\nCET\n</parameter>\n"
            "</function></tool_call>"
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"city": "Paris"}
        )
        self.assertEqual(
            json.loads(calls[1]["function"]["arguments"]), {"zone": "CET"}
        )

    def test_trailing_parameter_tag_does_not_join_a_complete_call(self) -> None:
        # A call that already decoded its arguments is complete, so a
        # <parameter=...> tag in the prose after it -- the model explaining the
        # format, say -- must not be folded into the call.
        _, calls = _parse_tool_calls(
            "<tool_call><function=get_weather>\n"
            "<parameter=city>\nParis\n</parameter>\n"
            "</function></tool_call>\n"
            "you can also pass <parameter=verbose>\ntrue\n</parameter>"
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"city": "Paris"}
        )

    def test_undeclared_tool_recovers_parameters_after_the_block(self) -> None:
        # With no schema there is no required list to test against, so recovery
        # keys off the block having decoded nothing at all.
        _, calls = _parse_tool_calls(
            "<tool_call>\n<function=list_files>\n</function>\n</tool_call>\n"
            "<parameter=execute>\ntrue\n</parameter>"
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"execute": True}
        )

    def test_block_parameters_win_over_trailing_ones(self) -> None:
        _, calls = _parse_tool_calls(
            "<tool_call><function=get_weather>\n"
            "<parameter=city>\nParis\n</parameter>\n"
            "</function></tool_call>\n"
            "<parameter=city>\nBerlin\n</parameter>"
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"city": "Paris"}
        )

    def test_duplicate_tool_calls_are_collapsed(self) -> None:
        block = (
            "<tool_call><function=get_weather>\n"
            "<parameter=city>\nParis\n</parameter>\n"
            "</function></tool_call>"
        )
        _, calls = _parse_tool_calls(block + block)
        self.assertEqual(len(calls), 1)

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

    def test_omitted_output_limits_use_service_ceiling(self) -> None:
        self.service.chat_completion(
            {"messages": [{"role": "user", "content": "Hi"}]}
        )
        self.assertEqual(self.generator.calls[-1][1]["max_new_tokens"], 32)

        self.service.response({"input": "Hi"})
        self.assertEqual(self.generator.calls[-1][1]["max_new_tokens"], 32)

        self.service.completion({"prompt": "Hi"})
        self.assertEqual(self.generator.calls[-1][1]["max_new_tokens"], 32)

    def test_model_generation_defaults_apply_only_when_request_omits_values(self) -> None:
        service = InferenceService(
            "qwen-local",
            self.generator,
            max_new_tokens=32,
            generation_defaults={
                "temperature": 0.6,
                "top_k": 7,
                "top_p": 0.8,
                "max_new_tokens": 12,
            },
        )

        service.chat_completion({"messages": [{"role": "user", "content": "Hi"}]})
        options = self.generator.calls[-1][1]
        self.assertEqual(options["max_new_tokens"], 12)
        self.assertEqual(options["sampling"], SamplingConfig(0.6, 7, 0.8))

        service.chat_completion(
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 3,
                "temperature": 0,
                "top_k": 2,
                "top_p": 1,
            }
        )
        options = self.generator.calls[-1][1]
        self.assertEqual(options["max_new_tokens"], 3)
        self.assertEqual(options["sampling"], SamplingConfig(0, 2, 1))

    def test_responses_api_reports_token_limit_as_incomplete(self) -> None:
        service = InferenceService(
            "qwen-local", LengthStubGenerator(), max_new_tokens=32
        )
        response = service.response(
            {"input": "Hi", "max_output_tokens": 2}
        )
        self.assertEqual(response["status"], "incomplete")
        self.assertEqual(
            response["incomplete_details"], {"reason": "max_output_tokens"}
        )
        self.assertEqual(response["output"][0]["status"], "incomplete")

        events = list(
            service.stream_response(
                {"input": "Hi", "max_output_tokens": 2, "stream": True}
            )
        )
        self.assertEqual(events[-1]["type"], "response.incomplete")
        self.assertEqual(events[-1]["response"]["status"], "incomplete")

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

    def test_health_reports_native_backend(self) -> None:
        service = InferenceService("qwen-local", self.generator)
        self.assertEqual(service.health()["execution"]["backend"], "native-v2")

    def test_chat_prompt_is_prepared_once_and_reused_for_generation(self) -> None:
        self.service.chat_completion(
            {"messages": [{"role": "user", "content": "Hi"}]}
        )
        self.assertEqual(len(self.generator.prepare_calls), 1)
        _, options = self.generator.calls[-1]
        self.assertEqual(options["prepared_prompt_ids"], (72, 105))

    def test_anthropic_stream_prepares_prompt_only_once(self) -> None:
        list(
            self.service.stream_anthropic_message(
                {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 4,
                }
            )
        )
        self.assertEqual(len(self.generator.prepare_calls), 1)

    def test_generation_capacity_rejects_excess_work_without_queueing(self) -> None:
        service = InferenceService(
            "qwen-local", self.generator, max_concurrent_requests=1
        )
        entered = threading.Event()
        release = threading.Event()

        def hold_slot() -> None:
            with service._generation_guard():
                entered.set()
                release.wait(2)

        worker = threading.Thread(target=hold_slot)
        worker.start()
        self.assertTrue(entered.wait(1))
        try:
            self.assertEqual(service.health()["active_requests"], 1)
            with self.assertRaises(APIError) as caught:
                with service._generation_guard():
                    pass
            self.assertEqual(caught.exception.status, 429)
        finally:
            release.set()
            worker.join()
        self.assertEqual(service.health()["active_requests"], 0)

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
                "choices": [{"index": 0, "delta": {}}],
                "colibri": {
                    "generated_tokens": 32,
                    "decode_elapsed_seconds": 0.75,
                    "phase": "tool_call",
                },
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
                {
                    "model": "qwen-local",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 4,
                }
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
        progress_pings = [
            event
            for event in events
            if event.get("type") == "ping"
            and event.get("colibri", {}).get("phase") == "tool_call"
        ]
        self.assertEqual(progress_pings[0]["colibri"]["generated_tokens"], 32)

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

    def test_anthropic_tool_results_precede_same_turn_user_text(self) -> None:
        self.service.anthropic_message(
            {
                "messages": [
                    {"role": "user", "content": "Run it"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "run",
                                "input": {},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "done",
                            },
                            {"type": "text", "text": "What next?"},
                        ],
                    },
                ],
                "max_tokens": 4,
            }
        )
        translated = self.generator.calls[-1][0]
        self.assertIn(
            "<tool_response>\ndone\n</tool_response>", translated[-2]["content"]
        )
        self.assertEqual(translated[-1]["content"], "What next?")

    def test_anthropic_protocol_validation_and_options(self) -> None:
        with self.assertRaisesRegex(APIError, "max_tokens"):
            self.service.anthropic_message(
                {"messages": [{"role": "user", "content": "Hi"}]}
            )
        with self.assertRaisesRegex(APIError, "unsupported block"):
            self.service.anthropic_message(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "image", "source": {}}],
                        }
                    ],
                    "max_tokens": 4,
                }
            )
        self.service.anthropic_message(
            {
                "messages": [{"role": "user", "content": "Think"}],
                "thinking": {"type": "enabled"},
                "tools": [{"name": "run", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "none"},
                "max_tokens": 4,
            }
        )
        _, options = self.generator.calls[-1]
        self.assertTrue(options["enable_thinking"])

    def test_anthropic_length_finish_maps_to_max_tokens(self) -> None:
        generator = StubGenerator()
        generator.generate_messages = Mock(
            return_value=GenerationResult(
                prompt_ids=(1,),
                generated_ids=(2,),
                text="unfinished",
                stopped_on_eos=False,
                state_tokens=2,
            )
        )
        service = InferenceService("qwen-local", generator)
        response = service.anthropic_message(
            {
                "messages": [{"role": "user", "content": "Continue"}],
                "max_tokens": 1,
            }
        )
        self.assertEqual(response["stop_reason"], "max_tokens")

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

    def test_complete_tool_call_stops_runaway_openai_generation(self) -> None:
        generator = RunawayToolStubGenerator()
        service = InferenceService("qwen-local", generator)
        payload = {
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            "stream": True,
        }

        events = list(service.stream_chat_completion(payload))

        self.assertTrue(generator.closed)
        self.assertEqual(generator.consumed, len(generator.TOOL_TEXT))
        progress = [
            event["colibri"]
            for event in events
            if isinstance(event, dict)
            and event.get("colibri", {}).get("phase") == "tool_call"
        ]
        self.assertTrue(progress)
        self.assertGreaterEqual(progress[-1]["generated_tokens"], len("<tool_call>"))
        finish = next(
            event["choices"][0]["finish_reason"]
            for event in events
            if isinstance(event, dict)
            and event.get("choices")
            and event["choices"][0].get("finish_reason")
        )
        self.assertEqual(finish, "tool_calls")

    def test_complete_tool_call_stops_runaway_non_stream_generation(self) -> None:
        generator = RunawayToolStubGenerator()
        service = InferenceService("qwen-local", generator)

        response = service.chat_completion(
            {
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            }
        )

        self.assertTrue(generator.closed)
        self.assertEqual(generator.consumed, len(generator.TOOL_TEXT))
        self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")

    def test_complete_tool_call_stops_runaway_anthropic_generation(self) -> None:
        generator = RunawayToolStubGenerator()
        service = InferenceService("qwen-local", generator)

        events = list(
            service.stream_anthropic_message(
                {
                    "model": "qwen-local",
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                    "tools": [
                        {
                            "name": "get_weather",
                            "input_schema": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        }
                    ],
                    "max_tokens": 4096,
                    "stream": True,
                }
            )
        )

        self.assertTrue(generator.closed)
        self.assertEqual(generator.consumed, len(generator.TOOL_TEXT))
        self.assertTrue(
            any(
                event.get("type") == "content_block_start"
                and event.get("content_block", {}).get("type") == "tool_use"
                for event in events
            )
        )

    def test_complete_tool_call_stops_runaway_responses_generation(self) -> None:
        generator = RunawayToolStubGenerator()
        service = InferenceService("qwen-local", generator)

        events = list(
            service.stream_response(
                {
                    "model": "qwen-local",
                    "input": "Weather in Paris?",
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        }
                    ],
                    "max_output_tokens": 4096,
                    "stream": True,
                }
            )
        )

        self.assertTrue(generator.closed)
        self.assertEqual(generator.consumed, len(generator.TOOL_TEXT))
        self.assertTrue(
            any(
                event["type"] == "response.function_call_arguments.done"
                for event in events
            )
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

    def test_idle_pooled_connection_outlives_the_request_timeout(self) -> None:
        # A harness keeps connections pooled and the runtime serves one request
        # at a time, so a connection routinely sits idle for longer than a
        # request may take. Applying the request timeout to that wait closed
        # working connections -- "Request timed out" in the log, a reconnect for
        # the client -- so the idle wait gets its own, longer limit.
        service = InferenceService(
            "qwen-local", StubGenerator(), request_timeout_seconds=0.2
        )
        server = ColibriHTTPServer(("127.0.0.1", 0), create_handler(service))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=5
        )
        try:
            connection.request("GET", "/health")
            self.assertEqual(connection.getresponse().read() and 200, 200)
            time.sleep(0.5)  # longer than the request timeout, idle throughout
            connection.request("GET", "/health")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            response.read()
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

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
        _, anthropic_count = self.request_json(
            "POST",
            "/v1/messages/count_tokens",
            {
                "model": "qwen-local",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        self.assertEqual(anthropic_count, {"input_tokens": 2})
        _, anthropic_tool_count = self.request_json(
            "POST",
            "/v1/messages/count_tokens",
            {
                "model": "qwen-local",
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [
                    {
                        "name": "Edit",
                        "description": "Replace exact text in a file.",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "old_string": {"type": "string"},
                                "new_string": {"type": "string"},
                            },
                        },
                    }
                ],
            },
        )
        self.assertGreater(
            anthropic_tool_count["input_tokens"],
            anthropic_count["input_tokens"],
        )
        self.service.anthropic_message(
            {
                "model": "qwen-local",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 2,
                "tools": [
                    {
                        "name": "Edit",
                        "input_schema": {"type": "object"},
                    }
                ],
            }
        )
        self.assertIn(
            "copy old_string byte-for-byte",
            self.generator.calls[-1][0][0]["content"],
        )
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

    def test_stream_console_rate_counts_intervals_not_tokens(self) -> None:
        # N streamed tokens span N-1 intervals: the first token ends prompt
        # evaluation and is not part of decode. Dividing tokens by that span
        # instead overstates the rate (2x at the stub's two tokens) and makes
        # the console disagree with the browser, which divides by intervals in
        # formatGenerationMetrics(). A 0.5 s tick keeps the printed elapsed
        # exact at two decimals so the two can be compared.
        ticks = itertools.count(1000.0, 0.5)
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            }
        )
        captured = StringIO()
        with patch("time.perf_counter", lambda: next(ticks)):
            with redirect_stderr(captured):
                self.connection.request(
                    "POST",
                    "/v1/chat/completions",
                    body=body,
                    headers={"Content-Type": "application/json"},
                )
                self.connection.getresponse().read()
        done = re.search(
            r"\[gen \] done (\d+) tokens in\s*([\d.]+)s \(\s*([\d.]+) tok/s\)",
            captured.getvalue(),
        )
        self.assertIsNotNone(done, captured.getvalue())
        tokens, elapsed, rate = (
            int(done.group(1)), float(done.group(2)), float(done.group(3))
        )
        self.assertGreater(tokens, 1)
        self.assertGreater(elapsed, 0.0)
        self.assertAlmostEqual(rate, (tokens - 1) / elapsed, delta=0.05)

    def test_anthropic_errors_use_anthropic_envelope(self) -> None:
        response, payload = self.request_json(
            "POST",
            "/v1/messages",
            {"messages": [{"role": "user", "content": "Hi"}]},
        )
        self.assertEqual(response.status, 400)
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertNotIn("param", payload["error"])


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
            allowed_headers = response.getheader("Access-Control-Allow-Headers")
            self.assertIn("x-api-key", allowed_headers)
            self.assertIn("anthropic-version", allowed_headers)
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
    @patch("colibri_next.v2_server.NativeV2InferenceService")
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
                ]
            )
        self.assertEqual(result, 0)
        load_model.assert_called_once()
        self.assertEqual(load_model.call_args.kwargs["max_new_tokens"], 24)
        serve_http.assert_called_once_with(
            load_model.return_value,
            host="127.0.0.1",
            port=9012,
            max_connections=128,
        )
        load_model.return_value.close.assert_called_once()


class StreamKeepaliveTests(unittest.TestCase):
    """A stalled generator must keep producing bytes on the wire.

    Prompt evaluation holds the generator for minutes on a long prompt. With
    no traffic in between, pooled clients (Claude Code, Cline) abandon the
    request long before the first token arrives.
    """

    def setUp(self) -> None:
        self.service = InferenceService(
            "qwen-local", StubGenerator(), sse_keepalive_seconds=0.05
        )
        self.server = ColibriHTTPServer(
            ("127.0.0.1", 0), create_handler(self.service)
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=10
        )

    def tearDown(self) -> None:
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _stalling(self, events):
        def stream(payload, *, progress=None):
            time.sleep(0.3)  # stands in for a long prompt evaluation
            yield from events

        return stream

    def _post(self, path: str) -> str:
        self.connection.request(
            "POST",
            path,
            body=json.dumps(
                {"messages": [{"role": "user", "content": "Hi"}], "stream": True}
            ),
            headers={"Content-Type": "application/json"},
        )
        response = self.connection.getresponse()
        self.assertEqual(response.status, 200)
        return response.read().decode("utf-8")

    def test_anthropic_stream_pings_while_the_prompt_is_evaluated(self) -> None:
        self.service.stream_anthropic_message = self._stalling(
            [
                {"type": "message_start", "message": {"id": "msg_1"}},
                {"type": "message_stop"},
            ]
        )
        body = self._post("/v1/messages")
        head = body.split("message_start", 1)[0]
        # 0.3 s of stall at a 0.05 s interval: several pings, and they have to
        # land before the first real event rather than after it.
        self.assertGreaterEqual(head.count('"type": "ping"'), 2, body)
        self.assertIn("event: ping", head)
        self.assertIn("message_stop", body)

    def test_openai_stream_keepalive_uses_sse_comments(self) -> None:
        self.service.stream_chat_completion = self._stalling(
            [{"object": "chat.completion.chunk"}, "[DONE]"]
        )
        body = self._post("/v1/chat/completions")
        head = body.split("chat.completion.chunk", 1)[0]
        # A comment is inert framing for OpenAI-shaped clients, which have no
        # ping event to parse.
        self.assertGreaterEqual(head.count(": keepalive"), 2, body)
        self.assertIn("data: [DONE]", body)

    def test_stream_errors_still_propagate_through_the_keepalive_pump(self) -> None:
        def failing(payload, *, progress=None):
            yield {"object": "chat.completion.chunk"}
            raise APIError(500, "boom", "server_error")

        self.service.stream_chat_completion = failing
        body = self._post("/v1/chat/completions")
        self.assertIn("boom", body)


class EndlessStubGenerator(StubGenerator):
    """Generates until its iterator is closed, recording how far it got."""

    def __init__(self) -> None:
        super().__init__()
        self.produced = 0
        self.closed = threading.Event()

    def stream_messages(self, messages, **options):
        self.calls.append((messages, options))
        try:
            index = 0
            while True:
                index += 1
                self.produced = index
                yield GenerationStep(
                    token_id=index,
                    text_delta="word ",
                    prompt_ids=(1, 2, 3),
                    generated_ids=tuple(range(index)),
                    text="word " * index,
                    stopped_on_eos=False,
                    finished=False,
                    state_tokens=index,
                )
                time.sleep(0.002)
        finally:
            self.closed.set()


class AbandonedStreamTests(unittest.TestCase):
    """A client that walks away must not leave the model generating.

    Writes into an abandoned stream keep succeeding until the socket buffer
    fills, so a long generation used to run to completion for nobody -- one
    observed request kept going to 7205 tokens after its client gave up.
    """

    def setUp(self) -> None:
        self.generator = EndlessStubGenerator()
        self.service = InferenceService(
            "qwen-local", self.generator, max_new_tokens=100000
        )
        self.server = ColibriHTTPServer(
            ("127.0.0.1", 0), create_handler(self.service)
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _abandon_mid_stream(self) -> socket.socket:
        """Start a stream, then half-close: FIN sent, socket still writable.

        This is the case that used to run forever. A full close makes the next
        write fail on its own, so it would pass with or without the fix; a
        half-close leaves the server's writes succeeding, which is exactly what
        let one request keep generating for a minute after its client gave up.
        """
        client = socket.create_connection(("127.0.0.1", self.port))
        body = json.dumps(
            {
                "model": "qwen-local",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            }
        ).encode()
        client.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\nContent-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        )
        deadline = time.monotonic() + 10
        seen = b""
        while b"\r\n\r\n" not in seen and time.monotonic() < deadline:
            seen += client.recv(4096)
        self.assertIn(b"200", seen)
        client.shutdown(socket.SHUT_WR)
        return client

    def test_generation_stops_when_the_client_abandons_the_stream(self) -> None:
        client = self._abandon_mid_stream()
        try:
            self.assertTrue(
                self.generator.closed.wait(timeout=10),
                "the generator kept running after the client abandoned the stream",
            )
            stopped_at = self.generator.produced
            time.sleep(0.3)
            self.assertEqual(
                self.generator.produced,
                stopped_at,
                "generation continued after the stream was closed",
            )
        finally:
            client.close()


class EndlessToolStubGenerator(StubGenerator):
    """Opens a <tool_call> and never closes it, as a looping model does."""

    HEAD = "<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/a\n</parameter>\n<parameter=content>\n"

    def __init__(self) -> None:
        super().__init__()
        self.produced = 0

    def stream_messages(self, messages, **options):
        self.calls.append((messages, options))
        text = self.HEAD
        index = 0
        yield GenerationStep(
            token_id=0, text_delta=text, prompt_ids=(1, 2, 3), generated_ids=(0,),
            text=text, stopped_on_eos=False, finished=False, state_tokens=1,
        )
        while index < 1000:  # a bound only so a failing test terminates
            index += 1
            self.produced = index
            text += "loop "
            yield GenerationStep(
                token_id=index,
                text_delta="loop ",
                prompt_ids=(1, 2, 3),
                generated_ids=tuple(range(index + 1)),
                text=text,
                stopped_on_eos=False,
                finished=False,
                state_tokens=index + 1,
            )
        yield GenerationStep(
            token_id=None, text_delta="", prompt_ids=(1, 2, 3),
            generated_ids=tuple(range(index + 1)), text=text,
            stopped_on_eos=False, finished=True, state_tokens=index + 1,
        )


class SlowToolStubGenerator(StubGenerator):
    """Writes a large parameter one piece at a time, closing the call at the end.

    This is the shape that timed a real client out: the call is valid, but its
    value takes minutes to write, and nothing about it reached the wire until
    the closing tag arrived.
    """

    HEAD = "<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/a\n</parameter>\n"
    BODY_PIECES = [f"line {index}\n" for index in range(40)]
    TAIL = "\n</parameter>\n</function>\n</tool_call>"

    def _pieces(self):
        yield self.HEAD
        yield "<parameter=content>\n"
        yield from self.BODY_PIECES
        yield self.TAIL

    def stream_messages(self, messages, **options):
        self.calls.append((messages, options))
        text = ""
        for index, piece in enumerate(self._pieces()):
            text += piece
            yield GenerationStep(
                token_id=index,
                text_delta=piece,
                prompt_ids=(1, 2, 3),
                generated_ids=tuple(range(index + 1)),
                text=text,
                stopped_on_eos=False,
                finished=False,
                state_tokens=index + 1,
            )
        yield GenerationStep(
            token_id=None,
            text_delta="",
            prompt_ids=(1, 2, 3),
            generated_ids=tuple(range(60)),
            text=text,
            stopped_on_eos=True,
            finished=True,
            state_tokens=60,
        )


WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
}


class ToolCallStreamingTests(unittest.TestCase):
    """A tool call must reach the client while it is still being written.

    Holding it back until </tool_call> meant a long call produced no content
    for minutes. Clients abandon such a stream ("no chunks received") however
    many SSE pings it carries, because a ping is not content.
    """

    def _anthropic_events(self, generator):
        service = InferenceService("qwen-local", generator, max_new_tokens=4096)
        return list(
            service.stream_anthropic_message(
                {
                    "model": "qwen-local",
                    "messages": [{"role": "user", "content": "write it"}],
                    "max_tokens": 4096,
                    "tools": [
                        {
                            "name": "write_file",
                            "input_schema": WRITE_FILE_TOOL["function"]["parameters"],
                        }
                    ],
                }
            )
        )

    def test_tool_use_block_opens_before_the_call_is_complete(self) -> None:
        events = self._anthropic_events(SlowToolStubGenerator())
        types = [event["type"] for event in events]
        start = types.index("content_block_start")
        stop = len(types) - 1 - types[::-1].index("content_block_stop")
        deltas = [
            index
            for index, event in enumerate(events)
            if event["type"] == "content_block_delta"
            and event["delta"]["type"] == "input_json_delta"
        ]
        self.assertTrue(deltas, "the tool call produced no input_json_delta events")
        # Many deltas, spread between the block opening and closing -- not one
        # lump at the end, which is what the client timed out waiting for.
        self.assertGreater(len(deltas), 5)
        self.assertLess(start, deltas[0])
        self.assertLess(deltas[-1], stop)

    def test_streamed_tool_call_uses_exactly_one_block(self) -> None:
        events = self._anthropic_events(SlowToolStubGenerator())
        starts = [e for e in events if e["type"] == "content_block_start"]
        tool_starts = [
            e for e in starts if e["content_block"]["type"] == "tool_use"
        ]
        self.assertEqual(len(tool_starts), 1)
        self.assertEqual(tool_starts[0]["content_block"]["name"], "write_file")

    def test_streamed_fragments_assemble_into_the_parsed_arguments(self) -> None:
        events = self._anthropic_events(SlowToolStubGenerator())
        partial = "".join(
            event["delta"]["partial_json"]
            for event in events
            if event["type"] == "content_block_delta"
            and event["delta"]["type"] == "input_json_delta"
        )
        # The invariant that matters: streaming the call must deliver exactly
        # what the non-streaming parse of the same text produces.
        generator = SlowToolStubGenerator()
        whole = "".join(generator._pieces())
        _, calls = _parse_tool_calls(whole, tools=[WRITE_FILE_TOOL])
        self.assertEqual(
            json.loads(partial), json.loads(calls[0]["function"]["arguments"])
        )
        self.assertEqual(json.loads(partial)["path"], "/tmp/a")

    def test_stop_reason_is_tool_use_when_the_call_was_streamed(self) -> None:
        events = self._anthropic_events(SlowToolStubGenerator())
        deltas = [e for e in events if e["type"] == "message_delta"]
        self.assertEqual(deltas[-1]["delta"]["stop_reason"], "tool_use")

    def test_openai_stream_reports_tool_calls_finish_reason(self) -> None:
        service = InferenceService(
            "qwen-local", SlowToolStubGenerator(), max_new_tokens=4096
        )
        chunks = list(
            service.stream_chat_completion(
                {
                    "model": "qwen-local",
                    "messages": [{"role": "user", "content": "write it"}],
                    "tools": [WRITE_FILE_TOOL],
                }
            )
        )
        reasons = [
            choice.get("finish_reason")
            for chunk in chunks
            if isinstance(chunk, dict)
            for choice in chunk.get("choices", [])
        ]
        self.assertIn("tool_calls", reasons)
        arguments = "".join(
            call.get("function", {}).get("arguments", "")
            for chunk in chunks
            if isinstance(chunk, dict)
            for choice in chunk.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
        )
        self.assertEqual(json.loads(arguments)["path"], "/tmp/a")

    def test_truncated_tool_call_still_closes_valid_json(self) -> None:
        events = self._anthropic_events(TruncatedToolStubGenerator())
        partial = "".join(
            event["delta"]["partial_json"]
            for event in events
            if event["type"] == "content_block_delta"
            and event["delta"]["type"] == "input_json_delta"
        )
        if partial:
            json.loads(partial)  # must not raise
        self.assertEqual(
            len([e for e in events if e["type"] == "message_stop"]), 1
        )

    def test_max_tool_call_tokens_abandons_a_call_that_never_closes(self) -> None:
        generator = EndlessToolStubGenerator()
        service = InferenceService(
            "qwen-local",
            generator,
            max_new_tokens=100000,
            max_tool_call_tokens=50,
        )
        events = list(
            service.stream_anthropic_message(
                {
                    "model": "qwen-local",
                    "messages": [{"role": "user", "content": "write it"}],
                    "max_tokens": 100000,
                    "tools": [
                        {
                            "name": "write_file",
                            "input_schema": WRITE_FILE_TOOL["function"]["parameters"],
                        }
                    ],
                }
            )
        )
        self.assertLess(generator.produced, 400, "the tool call was not bounded")
        partial = "".join(
            event["delta"]["partial_json"]
            for event in events
            if event["type"] == "content_block_delta"
            and event["delta"]["type"] == "input_json_delta"
        )
        if partial:
            json.loads(partial)  # closed into valid JSON despite the cut
        self.assertEqual(len([e for e in events if e["type"] == "message_stop"]), 1)

    def test_unbounded_by_default(self) -> None:
        self.assertEqual(
            InferenceService("qwen-local", StubGenerator()).max_tool_call_tokens, 0
        )

    def test_short_tool_call_is_unchanged(self) -> None:
        events = self._anthropic_events(ToolStubGenerator())
        partial = "".join(
            event["delta"]["partial_json"]
            for event in events
            if event["type"] == "content_block_delta"
            and event["delta"]["type"] == "input_json_delta"
        )
        self.assertEqual(json.loads(partial), {"city": "Paris"})


if __name__ == "__main__":
    unittest.main()
