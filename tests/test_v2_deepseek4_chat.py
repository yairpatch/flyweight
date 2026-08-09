"""DeepSeek-V4's chat markup, and the rules that are not guessable from it.

Three of them cost measurement to establish, so they are pinned here rather
than left to a rereading of the template:

- the assistant role token is a *transition* emitted on the preceding
  user-side turn, so an assistant message that follows a system message or
  another assistant carries no role token at all;
- a tool result renders inside the user block it answers, and consecutive
  user-side turns join with a blank line instead of repeating the token;
- reasoning is dropped from every assistant turn the conversation has moved
  past, which is what decides between `<think></think>` and a bare `</think>`.

The real checkpoint's own template is the authority; where it is available the
last test in this file compares against it directly over every role sequence up
to length four.
"""

from __future__ import annotations

import itertools
import os
import unittest

from colibri_next.v2_server import NativeV2Tokenizer

_CHECKPOINT_PATH = os.environ.get("DEEPSEEK4_GGUF")
# A stale path is as good as no path: the variable often outlives the file it
# named, and treating that as "configured" turns a missing checkpoint into a
# wall of errors instead of a skip.
CHECKPOINT = _CHECKPOINT_PATH if _CHECKPOINT_PATH and os.path.exists(_CHECKPOINT_PATH) else None

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"


def render(messages, *, enable_thinking: bool = False) -> str:
    return NativeV2Tokenizer._format_deepseek4(
        messages, enable_thinking=enable_thinking
    )


class Deepseek4TemplateTests(unittest.TestCase):
    def test_a_single_user_turn(self):
        self.assertEqual(
            render([{"role": "user", "content": "Hi"}]),
            f"{BOS}<｜User｜>Hi<｜Assistant｜></think>",
        )

    def test_the_system_prompt_leads_with_no_markup_of_its_own(self):
        self.assertEqual(
            render([
                {"role": "system", "content": "A"},
                {"role": "system", "content": "B"},
                {"role": "user", "content": "Hi"},
            ]),
            f"{BOS}A\n\nB<｜User｜>Hi<｜Assistant｜></think>",
        )

    def test_thinking_opens_the_generation_prompt(self):
        self.assertTrue(
            render([{"role": "user", "content": "Hi"}], enable_thinking=True)
            .endswith("<｜Assistant｜><think>")
        )

    def test_consecutive_user_turns_join_rather_than_repeat_the_token(self):
        self.assertEqual(
            render([
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
            ]),
            f"{BOS}<｜User｜>one\n\ntwo<｜Assistant｜></think>",
        )

    def test_a_tool_result_lands_inside_the_user_block(self):
        self.assertEqual(
            render([
                {"role": "user", "content": "call"},
                {"role": "assistant", "content": "ok"},
                {"role": "tool", "content": "42"},
            ]),
            f"{BOS}<｜User｜>call<｜Assistant｜></think>ok{EOS}"
            f"<｜User｜><tool_result>42</tool_result><｜Assistant｜></think>",
        )

    def test_an_assistant_without_a_user_predecessor_has_no_role_token(self):
        # The encoder emits the role token as a transition on the user turn,
        # so there is none to emit here.
        self.assertEqual(
            render([
                {"role": "system", "content": "sys"},
                {"role": "assistant", "content": "hello"},
            ]),
            f"{BOS}syshello{EOS}<｜Assistant｜></think>",
        )

    def test_a_past_assistant_turn_keeps_no_open_think_block(self):
        rendered = render([
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ], enable_thinking=True)
        self.assertIn(f"<｜Assistant｜></think>two{EOS}", rendered)
        self.assertTrue(rendered.endswith("<｜Assistant｜><think>"))

    def test_a_trailing_assistant_turn_keeps_its_reasoning_block(self):
        # It sits after the last user message, so the conversation has not
        # moved past it and the encoder keeps the (here empty) block.
        self.assertIn(
            "<｜Assistant｜><think></think>two",
            render([
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
            ], enable_thinking=True),
        )

    def test_reasoning_content_is_preserved_on_the_live_assistant_turn(self):
        self.assertIn(
            "<think>why</think>answer",
            render([
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer", "reasoning_content": "why"},
            ], enable_thinking=True),
        )

    def test_structured_tool_history_renders_as_dsml(self):
        rendered = render([
            {"role": "user", "content": "weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "weather",
                        "arguments": '{"city":"Paris","days":3}',
                    }
                }],
            },
        ])
        self.assertIn('<｜DSML｜invoke name="weather">', rendered)
        self.assertIn('name="city" string="true">Paris', rendered)
        self.assertIn('name="days" string="false">3', rendered)

    def test_thinking_drops_a_developer_turn_the_conversation_moved_past(self):
        rendered = render([
            {"role": "developer", "content": "gone"},
            {"role": "user", "content": "kept"},
        ], enable_thinking=True)
        self.assertNotIn("gone", rendered)
        self.assertIn("kept", rendered)

    def test_the_same_developer_turn_survives_without_thinking(self):
        self.assertIn("gone", render([
            {"role": "developer", "content": "gone"},
            {"role": "user", "content": "kept"},
        ]))

    def test_an_empty_message_is_refused(self):
        with self.assertRaises(ValueError):
            render([{"role": "user", "content": "   "}])

    def test_an_unknown_role_is_refused(self):
        with self.assertRaises(ValueError):
            render([{"role": "captain", "content": "x"}])


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class TemplateEquivalenceTests(unittest.TestCase):
    """The fallback must be the checkpoint's template, not an approximation.

    It is only reached when a GGUF ships without one, which the real file does
    not -- but a rendering that disagrees with training would be invisible
    until the output quality dropped, so it is compared exhaustively here.
    """

    @classmethod
    def setUpClass(cls):
        from colibri_next.v2 import V2Model
        cls.model = V2Model(CHECKPOINT)
        cls.tokenizer = NativeV2Tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_it_matches_the_checkpoint_template_on_every_short_conversation(self):
        roles = ("system", "user", "assistant", "tool", "developer")
        compared = 0
        for size in (1, 2, 3, 4):
            for combination in itertools.product(roles, repeat=size):
                messages = [
                    {"role": role, "content": f"{role}{index}"}
                    for index, role in enumerate(combination)
                ]
                for thinking in (False, True):
                    expected = self.tokenizer.format_messages(
                        messages, enable_thinking=thinking
                    )
                    self.assertEqual(
                        render(messages, enable_thinking=thinking),
                        expected,
                        msg=f"{combination} thinking={thinking}",
                    )
                    compared += 1
        self.assertGreater(compared, 1000)

    def test_the_turn_separator_is_the_end_of_sentence_token(self):
        self.assertEqual(self.tokenizer.turn_separator, EOS)
        self.assertEqual(
            self.tokenizer.encode(EOS), [self.tokenizer.eos_token_ids[0]]
        )


if __name__ == "__main__":
    unittest.main()
