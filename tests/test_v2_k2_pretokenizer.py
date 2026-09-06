"""The K2-Horizon pre-tokenizer against the regex in its tokenizer.json.

K2 used to ride on the GPT-4o transcription, which is close and wrong in
three places the reference is not: GPT-4o splits a letter run at an
upper-to-lower transition, attaches a contraction to its word, and lets a
punctuation run swallow trailing slashes. The pattern is the llama3 shape
with marks, zero-width joiners and three-digit groups, and these tests pin
the transcription to it through the `regex` module.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from flyweight.v2 import V2Model
from tests.k2_horizon_gguf_fixture import build_k2_horizon_gguf
from tests.test_v2_qwen35_pretokenizer import CORPUS

try:
    import regex
except ImportError:  # pragma: no cover - exercised only where regex is absent
    regex = None

# Verbatim from IFM/K2-Horizon-3.7B tokenizer.json (pre_tokenizer.pretokenizers[0]).
K2_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|‌|‍)+"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

K2_CORPUS = list(CORPUS) + [
    "import { Game } from './src/core/Game';\n\n/**\n * Entry\n */\n",
    "zero‍width‌joiners and मराठी हिन्दी",
]


def _reference_split(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    compiled = regex.compile(K2_PATTERN)
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


@unittest.skipIf(regex is None, "the regex module is required for the oracle")
class K2HorizonPretokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory(prefix="flyweight-k2-tok-")
        path = Path(cls._directory.name) / "k2_horizon.gguf"
        build_k2_horizon_gguf(path)
        cls.model = V2Model(path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.model.close()
        cls._directory.cleanup()

    def test_pretokenizer_matches_the_reference_pattern(self) -> None:
        for text in K2_CORPUS:
            with self.subTest(text=text):
                self.assertEqual(self.model.pretokenize(text), _reference_split(text))

    def test_pieces_always_reconstruct_the_input(self) -> None:
        for text in K2_CORPUS:
            with self.subTest(text=text):
                self.assertEqual("".join(self.model.pretokenize(text)), text)

    def test_letter_runs_are_not_split_on_case(self) -> None:
        self.assertEqual(self.model.pretokenize("camelCase mV"), ("camelCase", " mV"))

    def test_a_contraction_is_its_own_piece(self) -> None:
        # Alternative 1 only matches at the apostrophe; the letter run before
        # it stops there. GPT-4o glues the two together.
        self.assertEqual(self.model.pretokenize("don't"), ("don", "'t"))

    def test_a_slash_opens_the_next_piece(self) -> None:
        # GPT-4o's punctuation tail is [\r\n/]*, K2's is [\r\n]*: the slash
        # after a blank line belongs to the comment it opens.
        self.assertEqual(self.model.pretokenize("';\n\n/**\n"), ("';\n\n", "/**\n"))

    def test_digits_group_in_threes_and_joiners_stay_in_the_word(self) -> None:
        self.assertEqual(self.model.pretokenize("1234567"), ("123", "456", "7"))
        self.assertEqual(self.model.pretokenize("x‍y"), ("x‍y",))


if __name__ == "__main__":
    unittest.main()
