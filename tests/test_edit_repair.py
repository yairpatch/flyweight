import json
import os
import unittest
from unittest.mock import patch

from flyweight.generation import GenerationResult, GenerationStep
from flyweight.server import InferenceService
from tests.test_server import StubGenerator

FILE = """\
function calculateTotal(items) {
    let total = 0;
    for (const item of items) {
        total += foo(item);
    }
    return total;
}
"""

EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "Edit",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
}


def listing(text):
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return "\n".join(f"{index + 1:6d}\t{line}" for index, line in enumerate(lines))


def tool_call_text(name, **parameters):
    """A Hermes tool call, laid out the way _tool_prompt asks for one."""
    body = "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n"
        for key, value in parameters.items()
    )
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


class FixedGenerator(StubGenerator):
    """A generator that says one thing, so a test can choose what was sampled.

    Both entry points say it: a request carrying tools is served through the
    streaming one even when the client did not ask to stream, so that a turn
    can be cut the moment a complete tool call exists.
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def generate_messages(self, messages, **options) -> GenerationResult:
        self.calls.append((messages, options))
        return GenerationResult(
            prompt_ids=(1, 2, 3),
            generated_ids=(4, 5),
            text=self.text,
            stopped_on_eos=True,
            state_tokens=4,
        )

    def stream_messages(self, messages, **options):
        self.calls.append((messages, options))
        yield GenerationStep(
            token_id=1,
            text_delta=self.text,
            prompt_ids=(1, 2, 3),
            generated_ids=(1,),
            text=self.text,
            stopped_on_eos=False,
            finished=False,
            state_tokens=1,
        )
        yield GenerationStep(
            token_id=None,
            text_delta="",
            prompt_ids=(1, 2, 3),
            generated_ids=(1,),
            text=self.text,
            stopped_on_eos=True,
            finished=True,
            state_tokens=1,
        )


def history(read_result=None, path="/w/app.py"):
    """A transcript in which the model has read the file it is about to edit."""
    return [
        {"role": "user", "content": "rename foo to bar"},
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "r1",
                "type": "function",
                "function": {
                    "name": "Read",
                    "arguments": json.dumps({"file_path": path}),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "r1",
            "content": listing(FILE) if read_result is None else read_result,
        },
    ]


def edit_call(old, new="total += bar(item);", **extra):
    return tool_call_text(
        "Edit", file_path="/w/app.py", old_string=old, new_string=new, **extra
    )


def arguments_of(response):
    calls = response["choices"][0]["message"]["tool_calls"]
    return json.loads(calls[0]["function"]["arguments"])


def run(generated, messages=None, tools=(EDIT_TOOL,)):
    service = InferenceService("qwen-local", FixedGenerator(generated))
    return service.chat_completion({
        "messages": messages if messages is not None else history(),
        "tools": list(tools),
    })


class RepairTest(unittest.TestCase):
    def test_a_locator_that_widened_a_run_of_spaces_is_put_right(self):
        response = run(edit_call("total  +=  foo(item);"))
        self.assertEqual(
            arguments_of(response)["old_string"], "total += foo(item);"
        )

    def test_a_locator_whose_indentation_was_guessed_is_put_right(self):
        response = run(edit_call(
            "for (const item of items) {\n  total += foo(item);\n}"
        ))
        repaired = arguments_of(response)["old_string"]
        self.assertIn(repaired, FILE)
        self.assertTrue(repaired.startswith("for (const item"))

    def test_the_replacement_is_never_touched(self):
        response = run(edit_call("total  +=  foo(item);", new="total += bar(item);"))
        self.assertEqual(
            arguments_of(response)["new_string"], "total += bar(item);"
        )

    def test_a_locator_that_was_already_exact_goes_through_unchanged(self):
        response = run(edit_call("total += foo(item);"))
        self.assertEqual(
            arguments_of(response)["old_string"], "total += foo(item);"
        )

    def test_the_repair_reaches_the_anthropic_path_too(self):
        service = InferenceService(
            "qwen-local", FixedGenerator(edit_call("total  +=  foo(item);"))
        )
        response = service.anthropic_message({
            "model": "qwen-local",
            "max_tokens": 256,
            "messages": [
                {"role": "user", "content": "rename foo to bar"},
                {"role": "assistant", "content": [{
                    "type": "tool_use", "id": "r1", "name": "Read",
                    "input": {"file_path": "/w/app.py"},
                }]},
                {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": "r1",
                    "content": listing(FILE),
                }]},
            ],
            "tools": [{
                "name": "Edit",
                "input_schema": EDIT_TOOL["function"]["parameters"],
            }],
        })
        blocks = [item for item in response["content"] if item["type"] == "tool_use"]
        self.assertEqual(blocks[0]["input"]["old_string"], "total += foo(item);")

    def test_the_repair_reaches_the_streaming_path(self):
        service = InferenceService(
            "qwen-local", FixedGenerator(edit_call("total  +=  foo(item);"))
        )
        emitted = []
        for event in service.stream_chat_completion({
            "messages": history(),
            "tools": [EDIT_TOOL],
            "stream": True,
        }):
            if not isinstance(event, dict):
                continue
            for call in event["choices"][0]["delta"].get("tool_calls", []):
                emitted.append(json.loads(call["function"]["arguments"]))
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["old_string"], "total += foo(item);")


class AbstainTest(unittest.TestCase):
    """Every one of these is a case where guessing would be the wrong answer."""

    def test_an_ambiguous_locator_is_left_for_the_harness_to_refuse(self):
        response = run(edit_call("foo("))
        self.assertEqual(arguments_of(response)["old_string"], "foo(")

    def test_a_locator_that_is_nowhere_in_the_file_is_left_alone(self):
        response = run(edit_call("total += quux(item);"))
        self.assertEqual(
            arguments_of(response)["old_string"], "total += quux(item);"
        )

    def test_a_truncated_read_cannot_prove_uniqueness_so_nothing_moves(self):
        messages = history(
            read_result=listing(FILE) + "\n[truncated: showing first 6 lines]"
        )
        response = run(edit_call("total  +=  foo(item);"), messages=messages)
        self.assertEqual(
            arguments_of(response)["old_string"], "total  +=  foo(item);"
        )

    def test_replace_all_is_left_alone_because_the_set_would_change(self):
        response = run(edit_call("total  +=  foo(item);", replace_all="true"))
        self.assertEqual(
            arguments_of(response)["old_string"], "total  +=  foo(item);"
        )

    def test_a_file_never_read_is_not_invented(self):
        messages = [{"role": "user", "content": "rename foo to bar"}]
        response = run(edit_call("total  +=  foo(item);"), messages=messages)
        self.assertEqual(
            arguments_of(response)["old_string"], "total  +=  foo(item);"
        )

    def test_an_edit_of_a_different_file_is_not_repaired_from_this_one(self):
        generated = tool_call_text(
            "Edit",
            file_path="/w/other.js",
            old_string="total  +=  foo(item);",
            new_string="x",
        )
        response = run(generated)
        self.assertEqual(
            arguments_of(response)["old_string"], "total  +=  foo(item);"
        )


class GateTest(unittest.TestCase):
    def test_the_env_switch_turns_the_whole_thing_off(self):
        with patch.dict(os.environ, {"FLYWEIGHT_EDIT_REPAIR": "0"}):
            response = run(edit_call("total  +=  foo(item);"))
        self.assertEqual(
            arguments_of(response)["old_string"], "total  +=  foo(item);"
        )

    def test_a_request_with_no_edit_tool_never_builds_a_transcript(self):
        weather = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
        service = InferenceService("qwen-local", FixedGenerator("Hello!"))
        request = service._prepare_chat({
            "messages": history(),
            "tools": [weather],
        })
        self.assertEqual(request.transcript, ())

    def test_a_request_with_an_edit_tool_does_build_one(self):
        service = InferenceService("qwen-local", FixedGenerator("Hello!"))
        request = service._prepare_chat({
            "messages": history(),
            "tools": [EDIT_TOOL],
        })
        self.assertTrue(request.transcript)


if __name__ == "__main__":
    unittest.main()
