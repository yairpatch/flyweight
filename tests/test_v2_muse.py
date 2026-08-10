"""Muse Glimmer: GGUF metadata parsing and the `llama4` pre-tokenizer.

The pre-tokenizer is a hand transcription of the reference's GPT-4o regex --
`llama4` maps onto that pre-type -- so it is checked against that regex run
through a real Unicode-aware engine rather than against pinned expectations.
`regex` (not `re`) is required for the \\p{...} classes and the lookahead.
"""

from __future__ import annotations

import itertools
import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from colibri_next.generation import GenerationStep
from colibri_next.server import (
    MuseChannelStream,
    _split_muse_channels,
    _split_reasoning_content,
)
from colibri_next.v2 import V2Model

from tests.muse_gguf_fixture import MuseSpec, build_muse_gguf

try:
    import regex
except ImportError:  # pragma: no cover - exercised only where regex is absent
    regex = None


# Verbatim from the reference's LLAMA_VOCAB_PRE_TYPE_GPT4O, which is what the
# `llama4` pre-tokenizer name resolves to. Note the case classes are ASCII, so
# non-ASCII letters satisfy both sides of the split.
GPT4O_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+"
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
    r"|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*"
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

CORPUS = [
    "Hello world",
    "  leading and trailing  ",
    "one two  three   four",
    "line one\nline two\n\n\nline three",
    "trailing whitespace at end   ",
    "tabs\tand\r\ncarriage returns",
    "1 12 123 1234 12345 1234567890",
    "version 3.14.159 and 2026-08-10",
    "snake_case camelCase kebab-case",
    "CamelCase ALLCAPS mixedUP XMLHttpRequest",
    "punctuation!!! ??? ...ellipsis",
    "don't can't we're I've I'm they'll would'd",
    "DON'T CAN'T WE'RE",
    "@mention #hashtag $USD 50%",
    "'quoted' \"double\" `backtick`",
    "path/to/file.txt and C:\\Windows\\System32",
    "https://example.com/a/b?c=d",
    "e=mc^2 + 5*3 >= 7",
    "def f(x): return x ** 2  # comment",
    '{"key": "value", "n": [1, 2, 3]}',
    "",
    " ",
    "\n",
    "\n\n\n",
    "a",
    "   ",
]


def _reference_split(text: str) -> tuple[str, ...]:
    """Every match is a piece, and so is every gap between matches."""
    if not text:
        return ()
    compiled = regex.compile(GPT4O_PATTERN)
    pieces: list[str] = []
    at = 0
    for match in compiled.finditer(text):
        start, end = match.span()
        if end == start:
            continue
        if start > at:
            pieces.append(text[at:start])
        pieces.append(text[start:end])
        at = end
    if at < len(text):
        pieces.append(text[at:])
    return tuple(pieces)


class MuseModelTests(unittest.TestCase):
    """Metadata the runtime has to read correctly to build a plan at all."""

    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix="colibri-muse-")
        cls.spec = MuseSpec()
        cls.path = Path(cls.workspace.name) / "muse.gguf"
        build_muse_gguf(cls.path, cls.spec)
        cls.model = V2Model(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.model.close()
        cls.workspace.cleanup()

    def test_architecture_and_geometry(self):
        self.assertEqual(self.model.info["architecture"], "muse-glimmer")
        self.assertEqual(self.model.config["layer_count"], self.spec.layers)
        self.assertEqual(self.model.config["hidden_size"], self.spec.hidden)
        self.assertEqual(self.model.config["attention_heads"], self.spec.heads)
        self.assertEqual(self.model.config["attention_kv_heads"], self.spec.kv_heads)

    def test_scalar_sliding_window_period_expands_per_layer(self):
        # The GGUF carries the period, not the pattern. The cycle has to end on
        # the full-attention layer: sliding on 0, 1, 2 and full on 3.
        for layer in range(self.spec.layers):
            with self.subTest(layer=layer):
                window = self.model.config["attention_windows"][layer]
                if self.spec.is_sliding(layer):
                    self.assertEqual(window, self.spec.sliding_window)
                else:
                    self.assertEqual(window, 0)

    def test_full_attention_lands_every_fourth_layer(self):
        full = [
            layer
            for layer in range(self.spec.layers)
            if self.model.config["attention_windows"][layer] == 0
        ]
        self.assertEqual(full, [3, 7])

    def test_logit_scale_and_softcap_are_read(self):
        self.assertAlmostEqual(
            self.model.config["logit_scale"], self.spec.logit_scale, places=6
        )
        self.assertAlmostEqual(
            self.model.config["final_logit_softcap"], self.spec.logit_softcap, places=5
        )


@unittest.skipIf(regex is None, "the regex module is required for the oracle")
class MusePretokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix="colibri-muse-tok-")
        path = Path(cls.workspace.name) / "muse.gguf"
        build_muse_gguf(path)
        cls.model = V2Model(path)

    @classmethod
    def tearDownClass(cls):
        cls.model.close()
        cls.workspace.cleanup()

    def test_pretokenizer_matches_the_reference_pattern(self):
        for text in CORPUS:
            with self.subTest(text=text):
                self.assertEqual(self.model.pretokenize(text), _reference_split(text))

    def test_pieces_always_reconstruct_the_input(self):
        for text in CORPUS:
            with self.subTest(text=text):
                self.assertEqual("".join(self.model.pretokenize(text)), text)

    def test_digits_group_in_threes(self):
        # The Laguna pre-tokenizer stands every digit alone; this one does not.
        self.assertEqual(self.model.pretokenize("1234567"), ("123", "456", "7"))

    def test_a_space_stays_with_the_word_that_follows(self):
        # "\s+(?!\S)" hands the last space of a run to the next word.
        self.assertEqual(self.model.pretokenize("a  b"), ("a", " ", " b"))

    def test_case_boundaries_split_words(self):
        self.assertEqual(
            self.model.pretokenize("camelCase"), _reference_split("camelCase")
        )
        self.assertEqual(
            self.model.pretokenize("ALLCAPSthen"), _reference_split("ALLCAPSthen")
        )

    def test_contractions_stay_attached(self):
        self.assertEqual(self.model.pretokenize("don't"), ("don't",))
        self.assertEqual(self.model.pretokenize("we're"), ("we're",))


class MuseChannelSplitTests(unittest.TestCase):
    """Recipient-tagged messages split into reasoning and visible content.

    Without this the whole raw turn -- chain-of-thought and <|eom|>/<|start|>
    framing included -- lands in the OpenAI `content` field.
    """

    def test_reasoning_is_separated_from_the_answer(self):
        raw = (
            "to=self<|message|>We need Paris.<|eom|>"
            "<|start|>assistant to=user<|message|>The capital of France is Paris."
        )
        self.assertEqual(
            _split_reasoning_content(raw),
            ("The capital of France is Paris.", "We need Paris."),
        )

    def test_final_message_may_omit_its_recipient(self):
        raw = "to=self<|message|>think<|eom|><|start|>assistant<|message|>Answer."
        self.assertEqual(_split_reasoning_content(raw), ("Answer.", "think"))

    def test_a_turn_without_reasoning_has_no_reasoning_content(self):
        self.assertEqual(
            _split_reasoning_content("to=user<|message|>Just the answer.<|eot|>"),
            ("Just the answer.", None),
        )

    def test_several_analysis_messages_join(self):
        raw = (
            "to=self<|message|>first<|eom|>"
            "<|start|>assistant to=self<|message|>second<|eom|>"
            "<|start|>assistant to=user<|message|>Done."
        )
        self.assertEqual(_split_reasoning_content(raw), ("Done.", "first\n\nsecond"))

    def test_the_think_convention_still_works(self):
        # Muse markup is recognized by its own markers, so the architectures
        # that wrap reasoning in <think> must be untouched.
        self.assertEqual(
            _split_reasoning_content("<think>\nhidden\n</think>\n\nvisible"),
            ("visible", "\nhidden\n"),
        )
        self.assertEqual(
            _split_reasoning_content("</think>\n\nvisible"), ("visible", None)
        )
        self.assertEqual(_split_reasoning_content("just text"), ("just text", None))


class MuseChannelStreamTests(unittest.TestCase):
    """The streaming filter must agree with the batch splitter, always.

    Markers arrive split across token boundaries, so the risk is emitting a
    fragment like ``<|`` as content -- emission is tracked by length and cannot
    be retracted once sent.
    """

    TURNS = [
        "to=self<|message|>We need Paris.<|eom|>"
        "<|start|>assistant to=user<|message|>The capital of France is Paris.<|eot|>",
        "to=self<|message|>a<|eom|><|start|>assistant to=self<|message|>b<|eom|>"
        "<|start|>assistant<|message|>Final answer here.",
        "to=user<|message|>Straight to the answer, no reasoning at all.<|eot|>",
        "to=self<|message|>edge < case with <| bare marks |> inside<|eom|>"
        "<|start|>assistant to=user<|message|>Done <|eot|>",
    ]

    def _drive(self, raw: str, sizes):
        stream = MuseChannelStream()
        visible = reasoning = ""
        at = 0
        for size in itertools.cycle(sizes):
            if at >= len(raw):
                break
            piece, at = raw[at : at + size], at + size
            fresh_visible, fresh_reasoning = stream.feed(piece)
            visible += fresh_visible
            reasoning += fresh_reasoning
        fresh_visible, fresh_reasoning = stream.flush()
        return visible + fresh_visible, reasoning + fresh_reasoning

    def test_matches_the_batch_split_for_every_chunking(self):
        for raw in self.TURNS:
            expected = _split_muse_channels(raw)
            for sizes in ([1], [2], [3], [7], [1, 5, 2], [len(raw)]):
                with self.subTest(raw=raw[:30], sizes=sizes):
                    visible, reasoning = self._drive(raw, sizes)
                    self.assertEqual(visible.strip(), expected[0])
                    self.assertEqual(reasoning.strip() or None, expected[1])

    def test_a_marker_split_across_deltas_never_leaks(self):
        # One character at a time is the worst case: every marker is torn apart.
        visible, reasoning = self._drive(self.TURNS[0], [1])
        self.assertNotIn("<|", visible)
        self.assertNotIn("<|", reasoning)
        self.assertEqual(visible.strip(), "The capital of France is Paris.")

    def test_text_without_markup_is_forwarded_whole(self):
        stream = MuseChannelStream()
        stream.feed("plain text, no channels")
        self.assertEqual(stream.flush(), ("plain text, no channels", ""))


class MuseStreamingEndpointTests(unittest.TestCase):
    """The streaming endpoint must not leak reasoning into `content`.

    Driven through the real endpoint with a stub generator, because the filter
    being correct in isolation says nothing about it being wired up.
    """

    RAW = (
        "to=self<|message|>We need Paris.<|eom|>"
        "<|start|>assistant to=user<|message|>The capital of France is Paris.<|eot|>"
    )

    def _events(self, architecture: str, raw: str | None = None, **payload):
        from colibri_next.server import InferenceService

        from tests.test_server import StubGenerator

        generator = StubGenerator()
        generator.tokenizer.architecture = architecture

        text = self.RAW if raw is None else raw

        def stream_messages(messages, **options):
            generated: list[int] = []
            # One character per step, so every marker straddles a delta.
            for index, character in enumerate(text):
                generated.append(index)
                yield GenerationStep(
                    token_id=index,
                    text_delta=character,
                    prompt_ids=(1, 2, 3),
                    generated_ids=tuple(generated),
                    text=text[: index + 1],
                    stopped_on_eos=False,
                    finished=False,
                    state_tokens=3 + len(generated),
                )
            yield GenerationStep(
                token_id=None,
                text_delta="",
                prompt_ids=(1, 2, 3),
                generated_ids=tuple(generated),
                text=text,
                stopped_on_eos=raw is None,
                finished=True,
                state_tokens=3 + len(generated),
            )

        generator.stream_messages = stream_messages
        service = InferenceService("muse-local", generator, max_new_tokens=512)
        return list(
            service.stream_chat_completion(
                {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                    **payload,
                }
            )
        )

    def _joined(self, events, field):
        return "".join(
            event["choices"][0]["delta"].get(field, "")
            for event in events
            if isinstance(event, dict) and event.get("choices")
        )

    def test_by_default_reasoning_streams_live_as_content(self):
        # This model cannot be told to stop reasoning, and the reasoning comes
        # first. Withholding it would leave the stream silent until the answer
        # began, so by default it flows as ordinary content -- minus the
        # protocol framing, which is never content.
        events = self._events("muse-glimmer")
        content = self._joined(events, "content")
        self.assertIn("We need Paris.", content)
        self.assertIn("The capital of France is Paris.", content)
        self.assertNotIn("<|", content)
        self.assertNotIn("to=self", content)
        self.assertEqual(self._joined(events, "reasoning_content"), "")

    def test_the_first_delta_arrives_before_the_answer(self):
        # The liveness property itself: something is emitted well before the
        # reasoning ends, rather than one burst at the end.
        events = [
            event
            for event in self._events("muse-glimmer")
            if isinstance(event, dict)
            and event.get("choices")
            and event["choices"][0]["delta"].get("content")
        ]
        self.assertGreater(len(events), 1)
        first = events[0]["choices"][0]["delta"]["content"]
        self.assertTrue(first.strip())
        self.assertNotIn("Paris is", first)

    def test_separate_reasoning_moves_it_off_the_content_stream(self):
        events = self._events("muse-glimmer", separate_reasoning=True)
        content = self._joined(events, "content")
        reasoning = self._joined(events, "reasoning_content")
        self.assertEqual(content.strip(), "The capital of France is Paris.")
        self.assertEqual(reasoning.strip(), "We need Paris.")
        self.assertNotIn("We need Paris", content)
        self.assertNotIn("<|", content + reasoning)

    def test_a_truncated_turn_keeps_its_last_characters(self):
        """A stream cut short by max_tokens must still release the tail.

        The filter withholds text that could still grow into a marker. If the
        turn simply stops -- no <|eot|> -- that tail has to be flushed, or the
        answer silently loses its last characters. The tool-enabled stream is a
        separate code path and regressed here independently.
        """
        truncated = (
            "to=self<|message|>t<|eom|>"
            "<|start|>assistant to=user<|message|>Use the tag <"
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "w",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        for label, payload in (("plain", {}), ("tools", {"tools": tools})):
            with self.subTest(stream=label):
                events = self._events("muse-glimmer", raw=truncated, **payload)
                self.assertEqual(
                    self._joined(events, "content").strip(), "tUse the tag <"
                )

    def test_other_architectures_stream_unchanged(self):
        # The same raw text from a non-Muse model is forwarded verbatim.
        content = self._joined(self._events("qwen35"), "content")
        self.assertEqual(content, self.RAW)


class MuseNativeTests(unittest.TestCase):
    """The plan and the decode path, on a fixture small enough to run here."""

    def _runtime(self):
        if not V2Model.gpu_info()["available"]:
            raise unittest.SkipTest("native CUDA runtime is unavailable")
        workspace = tempfile.TemporaryDirectory(prefix="colibri-muse-native-")
        self.addCleanup(workspace.cleanup)
        path = Path(workspace.name) / "muse.gguf"
        spec = build_muse_gguf(path)
        model = V2Model(path)
        self.addCleanup(model.close)
        runtime = model.native_runtime(context_limit=64, mtp_drafts=0)
        self.addCleanup(runtime.close)
        runtime.prepare()
        return runtime, spec

    def test_runtime_plans_and_decodes(self):
        runtime, spec = self._runtime()
        info = runtime.info
        # Three of every four layers slide; the fourth closes the cycle.
        self.assertEqual(
            info["swa_layers"],
            sum(1 for layer in range(spec.layers) if spec.is_sliding(layer)),
        )
        token = runtime.decode(5)
        self.assertGreaterEqual(token, 0)
        self.assertLess(token, spec.vocabulary)

    def test_decode_is_deterministic(self):
        first, _ = self._runtime()
        produced = [first.decode(token) for token in (5, 6, 7, 8)]
        second, _ = self._runtime()
        self.assertEqual(produced, [second.decode(token) for token in (5, 6, 7, 8)])

    def test_batched_prefill_matches_token_by_token(self):
        """The rows forward must agree with the one-token path.

        This is the load-bearing check on batched prefill. It drives the
        interleaved RoPE, the NoPE full-attention layers, the per-channel gate
        and both post-norms through a second, independently written code path,
        and any disagreement shows up as a different continuation.
        """
        prompt = [7, 11, 3, 29, 5, 17, 23, 2, 13, 19, 31, 37]
        results = {}
        for rows in ("1", "8"):
            with unittest.mock.patch.dict(os.environ, {"COLIBRI_PREFILL_ROWS": rows}):
                runtime, _ = self._runtime()
                produced: list[int] = []
                runtime.generate(prompt, 6, produced.append)
                results[rows] = produced
        self.assertEqual(results["1"], results["8"])


@unittest.skipUnless(
    os.environ.get("COLIBRI_TEST_MUSE_MODEL")
    and Path(os.environ["COLIBRI_TEST_MUSE_MODEL"]).is_file(),
    "set COLIBRI_TEST_MUSE_MODEL to a Muse Glimmer GGUF checkpoint",
)
class MuseRealCheckpointTests(unittest.TestCase):
    """Batched prefill against the one-token path, on real quantized weights.

    The synthetic fixture is f32, so it never reaches the DP4A projections --
    which is exactly where batched prefill broke by reusing a stale Q8
    activation cache, producing fluent-looking garbage. Only a quantized
    checkpoint exercises it.
    """

    def _continuation(self, rows: str) -> list[int]:
        # Closed before the next configuration runs: two runtimes of a
        # multi-gigabyte checkpoint do not fit on the card at once, and the
        # second would spill every block and answer a different question.
        with unittest.mock.patch.dict(os.environ, {"COLIBRI_PREFILL_ROWS": rows}):
            model = V2Model(os.environ["COLIBRI_TEST_MUSE_MODEL"])
            try:
                runtime = model.native_runtime(context_limit=2048, mtp_drafts=0)
                try:
                    runtime.prepare()
                    body = "Paris is the capital of France. " * 40
                    prompt = model.tokenize(
                        f"<|start|>user<|message|>{body}\nWhat is the capital of "
                        "France? Reply with just the city name.<|eot|>"
                        "<|start|>assistant to=user<|message|>"
                    )
                    produced: list[int] = []
                    runtime.generate(prompt, 10, produced.append)
                    return produced
                finally:
                    runtime.close()
            finally:
                model.close()

    def test_batched_prefill_matches_token_by_token(self):
        if not V2Model.gpu_info()["available"]:
            self.skipTest("native CUDA runtime is unavailable")
        self.assertEqual(self._continuation("1"), self._continuation("64"))


if __name__ == "__main__":
    unittest.main()
