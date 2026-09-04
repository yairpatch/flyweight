"""Replayed chain-of-thought: whose default, and what overrides it.

A client that receives thinking blocks sends them back on the next turn. What
reaches the prompt is the checkpoint's convention -- keep the reasoning of the
tool-call loop the model is still inside, drop what the conversation has moved
past -- and `preserve_thinking` is the only thing that overrides it.
"""

from __future__ import annotations

import unittest

from flyweight.v2_server import NativeV2Tokenizer, keeps_replayed_reasoning


class ReplayRuleTests(unittest.TestCase):
    def test_the_default_keeps_the_turn_the_model_is_still_inside(self) -> None:
        # last_user = 2; index 3 is the assistant turn that called a tool and
        # is about to read its result, so its reasoning justifies the call.
        self.assertTrue(keeps_replayed_reasoning(None, True, 3, 2))
        # index 1 is an assistant turn the conversation has moved past.
        self.assertFalse(keeps_replayed_reasoning(None, True, 1, 2))

    def test_thinking_off_keeps_none_of_it(self) -> None:
        self.assertFalse(keeps_replayed_reasoning(None, False, 3, 2))
        self.assertFalse(keeps_replayed_reasoning(None, None, 3, 2))

    def test_an_explicit_choice_wins_in_both_directions(self) -> None:
        # True reaches back past the last user turn, which the default will not.
        self.assertTrue(keeps_replayed_reasoning(True, False, 1, 2))
        # False drops the current loop's reasoning, which the default keeps.
        self.assertFalse(keeps_replayed_reasoning(False, True, 3, 2))


class FallbackFormatterTests(unittest.TestCase):
    """The ChatML fallback, for a checkpoint carrying no jinja template."""

    def setUp(self) -> None:
        self.tokenizer = NativeV2Tokenizer.__new__(NativeV2Tokenizer)
        self.tokenizer.architecture = "qwen3"

    def _render(self, **kwargs: object) -> str:
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Older answer",
             "reasoning_content": "OLD THOUGHT"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Calling a tool",
             "reasoning_content": "CURRENT THOUGHT"},
            {"role": "tool", "content": "42"},
        ]
        return self.tokenizer.format_messages(messages, **kwargs)  # type: ignore[arg-type]

    def test_by_default_the_current_loop_is_kept_and_the_rest_dropped(self) -> None:
        rendered = self._render(enable_thinking=True)
        self.assertIn("CURRENT THOUGHT", rendered)
        self.assertNotIn("OLD THOUGHT", rendered)

    def test_preserve_true_replays_the_whole_conversation(self) -> None:
        rendered = self._render(enable_thinking=True, preserve_thinking=True)
        self.assertIn("CURRENT THOUGHT", rendered)
        self.assertIn("OLD THOUGHT", rendered)

    def test_preserve_false_drops_all_of_it(self) -> None:
        rendered = self._render(enable_thinking=True, preserve_thinking=False)
        self.assertNotIn("CURRENT THOUGHT", rendered)
        self.assertNotIn("OLD THOUGHT", rendered)

    def test_a_history_turn_never_opens_a_block_it_does_not_close(self) -> None:
        # enable_thinking governs the turn about to be generated. Applying it to
        # history left an unterminated <think> mid-conversation, which no
        # training example contains.
        rendered = self._render(enable_thinking=True)
        history = rendered.rsplit("<|im_start|>assistant", 1)[0]
        self.assertEqual(history.count("<think>"), history.count("</think>"))

    def test_a_turn_that_already_carries_its_block_is_left_alone(self) -> None:
        rendered = self.tokenizer.format_messages(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "<think>inline</think>Answer"},
            ],
            enable_thinking=True,
        )
        self.assertIn("<think>inline</think>Answer", rendered)


class ContinuationKeyTests(unittest.TestCase):
    def test_replayed_reasoning_is_part_of_the_cache_key(self) -> None:
        # Reducing to (role, content) matched a turn rendered from a different
        # chain-of-thought and reused its tokens.
        from flyweight.v2_server import _chat_key

        base = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Same", "reasoning_content": "A"},
        ]
        other = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Same", "reasoning_content": "B"},
        ]
        self.assertNotEqual(_chat_key(base), _chat_key(other))
        self.assertEqual(_chat_key(base), _chat_key(list(base)))


class ProtocolTests(unittest.TestCase):
    """The switch has to survive the trip from the wire to the renderer."""

    def setUp(self) -> None:
        from tests.test_server import StubGenerator  # noqa: PLC0415
        from flyweight.server import InferenceService  # noqa: PLC0415

        self.generator = StubGenerator()
        self.service = InferenceService("qwen-local", self.generator)

    def _sent(self) -> object:
        return self.generator.calls[-1][1].get("preserve_thinking")

    def test_unstated_stays_unstated(self) -> None:
        # None is not False: it means the architecture decides, and coercing it
        # is the same mistake enable_thinking documents at length.
        self.service.chat_completion(
            {"messages": [{"role": "user", "content": "Hi"}]}
        )
        self.assertIsNone(self._sent())

    def test_the_flat_field_reaches_the_generator(self) -> None:
        for asked in (True, False):
            with self.subTest(asked=asked):
                self.service.chat_completion(
                    {
                        "messages": [{"role": "user", "content": "Hi"}],
                        "preserve_thinking": asked,
                    }
                )
                self.assertIs(self._sent(), asked)

    def test_the_vllm_spelling_reaches_it_too_and_loses_to_the_flat_field(self) -> None:
        self.service.chat_completion(
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "chat_template_kwargs": {"preserve_thinking": True},
            }
        )
        self.assertIs(self._sent(), True)
        self.service.chat_completion(
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "preserve_thinking": False,
                "chat_template_kwargs": {"preserve_thinking": True},
            }
        )
        self.assertIs(self._sent(), False)

    def test_it_reaches_the_anthropic_path(self) -> None:
        self.service.anthropic_message(
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 4,
                "preserve_thinking": True,
            }
        )
        self.assertIs(self._sent(), True)


if __name__ == "__main__":
    unittest.main()
