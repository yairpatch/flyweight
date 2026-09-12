"""A requested effort level meets the vocabulary its checkpoint was trained on.

Every protocol normalizes onto one ladder -- low, medium, high, xhigh -- but a
checkpoint's chat template reads whichever subset its authors chose, and some
of them raise on the rest. Qwen3.8-Flash-Next is the case that motivated this:
its template takes xhigh / medium / low and calls raise_exception on "high",
which is exactly what an OpenAI-shaped client sends. The server asks the
template which levels it renders, then clamps to the nearest one it named.
"""

from __future__ import annotations

import unittest

from jinja2 import Undefined
from jinja2.sandbox import ImmutableSandboxedEnvironment

from flyweight.v2_server import NativeV2Tokenizer


# The shape of Flash-Next's template, reduced to the part under test: a fixed
# vocabulary, an exception on anything outside it, and a visible marker of the
# level that was rendered.
FLASH_NEXT_TEMPLATE = """
{%- set effort = reasoning_effort|default('xhigh') %}
{%- if effort not in ('xhigh', 'medium', 'low') %}
    {{- raise_exception('Unexpected reasoning effort ' ~ effort) }}
{%- endif %}
EFFORT={{ effort }}
{%- for message in messages %}
{{ message.role }}: {{ message.content }}
{%- endfor %}
"""

# A template that takes the whole ladder, as Qwen3.5's does by folding high
# into xhigh itself.
PERMISSIVE_TEMPLATE = "EFFORT={{ reasoning_effort|default('none') }}"

# One that cannot render the probe conversation at all, for reasons that have
# nothing to do with the effort variable.
BROKEN_TEMPLATE = "{{ raise_exception('this template is broken') }}"


def _tokenizer(template: str) -> NativeV2Tokenizer:
    """A tokenizer with nothing but the template machinery wired up."""
    tokenizer = NativeV2Tokenizer.__new__(NativeV2Tokenizer)
    tokenizer.architecture = "qwen4exp"
    tokenizer._template_tokens = {"bos_token": "", "eos_token": ""}
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, undefined=Undefined
    )
    environment.globals["raise_exception"] = tokenizer._raise_template_exception
    tokenizer._compiled_chat_template = environment.from_string(template)
    return tokenizer


class ProbeTests(unittest.TestCase):
    def test_it_reports_the_levels_the_template_renders(self) -> None:
        tokenizer = _tokenizer(FLASH_NEXT_TEMPLATE)
        self.assertEqual(
            tokenizer._accepted_reasoning_efforts(), ("low", "medium", "xhigh")
        )

    def test_the_probe_runs_once(self) -> None:
        tokenizer = _tokenizer(FLASH_NEXT_TEMPLATE)
        first = tokenizer._accepted_reasoning_efforts()
        # Swapping the template out afterwards must not re-open the question:
        # the second call has to come from the cache, not from a fresh probe.
        tokenizer._compiled_chat_template = None
        self.assertEqual(tokenizer._accepted_reasoning_efforts(), first)

    def test_a_template_that_takes_everything_has_no_opinion(self) -> None:
        # Indistinguishable from one that ignores the variable, and treated the
        # same way: nothing to clamp, so the request passes through untouched.
        self.assertIsNone(
            _tokenizer(PERMISSIVE_TEMPLATE)._accepted_reasoning_efforts()
        )

    def test_a_template_that_renders_at_no_level_has_no_opinion(self) -> None:
        # Failing everywhere says something about the probe conversation, not
        # about the effort vocabulary. Claiming "this checkpoint accepts no
        # effort at all" from it would clamp every request on a template whose
        # only problem is that it needs a system turn.
        self.assertIsNone(
            _tokenizer(BROKEN_TEMPLATE)._accepted_reasoning_efforts()
        )

    def test_no_template_has_no_opinion(self) -> None:
        tokenizer = NativeV2Tokenizer.__new__(NativeV2Tokenizer)
        tokenizer.architecture = "qwen3"
        self.assertIsNone(tokenizer._accepted_reasoning_efforts())


class ClampTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = _tokenizer(FLASH_NEXT_TEMPLATE)

    def _render(self, effort: str | None) -> str:
        return self.tokenizer.format_messages(
            [{"role": "user", "content": "hi"}], reasoning_effort=effort
        )

    def test_high_becomes_xhigh_rather_than_failing(self) -> None:
        self.assertIn("EFFORT=xhigh", self._render("high"))

    def test_a_level_the_template_names_is_passed_through(self) -> None:
        for effort in ("low", "medium", "xhigh"):
            with self.subTest(effort=effort):
                self.assertIn(f"EFFORT={effort}", self._render(effort))

    def test_no_effort_leaves_the_checkpoint_its_default(self) -> None:
        self.assertIn("EFFORT=xhigh", self._render(None))

    def test_the_stronger_level_wins_a_tie(self) -> None:
        # "high" sits one step from medium and one from xhigh. A checkpoint
        # that does not name it does not distinguish it either, and rounding an
        # explicit request for more reasoning down to less is the worse miss.
        self.assertEqual(self.tokenizer._supported_reasoning_effort("high"), "xhigh")

    def test_a_vocabulary_off_the_ladder_is_left_alone(self) -> None:
        # Not ours to place: pass it to the template and let it decide.
        self.assertEqual(
            self.tokenizer._supported_reasoning_effort("turbo"), "turbo"
        )


class RenderFallbackTests(unittest.TestCase):
    """A level the probe cleared can still be refused by a real conversation."""

    # Accepts every level on the one-turn probe, then refuses xhigh as soon as
    # the conversation carries a tool turn -- the shape the probe cannot see.
    CONDITIONAL_TEMPLATE = """
{%- for message in messages %}
{%- if message.role == 'tool' and reasoning_effort is defined
       and reasoning_effort == 'xhigh' %}
{{- raise_exception('no xhigh with tool results') }}
{%- endif %}
{%- endfor %}
EFFORT={{ reasoning_effort|default('none') }}
"""

    def test_the_request_is_answered_at_the_default(self) -> None:
        tokenizer = _tokenizer(self.CONDITIONAL_TEMPLATE)
        rendered = tokenizer.format_messages(
            [{"role": "user", "content": "hi"}, {"role": "tool", "content": "42"}],
            reasoning_effort="xhigh",
        )
        self.assertIn("EFFORT=none", rendered)

    def test_a_failure_with_no_effort_to_drop_still_raises(self) -> None:
        # The retry exists for the effort variable alone; every other template
        # fault has to keep reaching the caller.
        tokenizer = _tokenizer(BROKEN_TEMPLATE)
        with self.assertRaises(ValueError):
            tokenizer.format_messages([{"role": "user", "content": "hi"}])


if __name__ == "__main__":
    unittest.main()
