"""The Qwen pre-tokenizer against the regex in Qwen3.6's tokenizer.json.

A Qwen checkpoint used to get no pre-tokenization at all: BPE ran over the
whole text, and its merges turned "\\n    private" into "\\n", "    ",
"private" where the reference makes "\\n", "   ", " private". Every indented
line of every prompt reached the model in a shape it was never trained on,
and it answered with the whitespace token it was shown plus the
space-prefixed word it knows -- one space too many per line, on every file an
agent wrote. These tests pin the transcription to the reference pattern, run
through the `regex` module, which knows the \\p{..} classes and the lookahead.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from flyweight.v2 import V2Model
from tests import qwen35_hf_fixture as fixture

try:
    import regex
except ImportError:  # pragma: no cover - exercised only where regex is absent
    regex = None

# Verbatim from Qwen/Qwen3.6-35B-A3B tokenizer.json (pre_tokenizer.pretokenizers[0]).
QWEN36_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

CORPUS = [
    "class A {\n    private _speed = 0;\n        this._speed = this._track.trackSpeed;\n    }\n}\n",
    "def f(x):\n    return x  # ok\n\n\n    y = [1, 2, 3]\n",
    "if (this._track) {\r\n       this._gForce = 1;\r\n     }\r\n",
    "\tindented\twith\ttabs\n\t\tdeeper\n",
    "one two  three   four    five",
    "  leading and trailing  ",
    "1 12 123 1234 12345 1234567890",
    "version 3.14.159 and 2026-08-10",
    "snake_case camelCase kebab-case XMLHttpRequest",
    "don't can't we're I've I'm they'll would'd DON'T",
    "punctuation!!! ??? ...ellipsis @mention #hashtag $USD 50%",
    "path/to/file.txt and C:\\Windows\\System32 https://example.com/a?b=c",
    '{"key": "value", "n": [1, 2, 3]}',
    "café cafe\u0301 na\u00efve \u0301x x\u0301",  # precomposed and combining marks
    "日本語のテキスト and 中文 mixed with English",
    "emoji 🎢 and symbols ≥ ≤ ± ×",
    "",
    " ",
    "\n",
    "\n\n\n",
    "   ",
    "a",
]


def _reference_split(text: str) -> tuple[str, ...]:
    """Every match is a piece, and so is every gap between matches."""
    if not text:
        return ()
    compiled = regex.compile(QWEN36_PATTERN)
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
class Qwen35PretokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory(prefix="flyweight-qwen35-tok-")
        cls.path = fixture.build(Path(cls._directory.name) / "qwen35")
        # The fixture ships the llama3 split pattern. Swapping in Qwen3.6's is
        # what makes the HF loader select the `qwen35` pre-tokenizer -- the
        # same name the family's GGUFs carry, so both loaders share the split.
        tokenizer_path = cls.path / "tokenizer.json"
        document = json.loads(tokenizer_path.read_text())
        document["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = QWEN36_PATTERN
        tokenizer_path.write_text(json.dumps(document))
        cls.model = V2Model(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.model.close()
        cls._directory.cleanup()

    def test_pretokenizer_matches_the_reference_pattern(self) -> None:
        for text in CORPUS:
            with self.subTest(text=text):
                self.assertEqual(self.model.pretokenize(text), _reference_split(text))

    def test_pieces_always_reconstruct_the_input(self) -> None:
        for text in CORPUS:
            with self.subTest(text=text):
                self.assertEqual("".join(self.model.pretokenize(text)), text)

    def test_an_indent_hands_its_last_space_to_the_word(self) -> None:
        # The bug this file exists for: "\s+(?!\S)" stops one short of the
        # word, so the word carries its usual leading space. Without the
        # split, BPE produced "    " + "private", and the model wrote five
        # spaces back.
        self.assertEqual(
            self.model.pretokenize("\n    private _speed;"),
            ("\n", "   ", " private", " _", "speed", ";"),
        )
        self.assertEqual(
            self.model.pretokenize("        this._speed"),
            ("       ", " this", "._", "speed"),
        )

    def test_digits_stand_alone(self) -> None:
        self.assertEqual(self.model.pretokenize("1234"), ("1", "2", "3", "4"))

    def test_letter_runs_keep_their_case_boundaries(self) -> None:
        # No case machinery, unlike GPT-4o: "mV" is one piece.
        self.assertEqual(self.model.pretokenize("mV camelCase"), ("mV", " camelCase"))

    def test_a_combining_mark_travels_with_its_letter(self) -> None:
        # [\p{L}\p{M}]+ keeps "e" and U+0301 together; the llama3 pattern
        # would split the accent off as punctuation.
        self.assertEqual(self.model.pretokenize("cafe\u0301 x"), ("cafe\u0301", " x"))


if __name__ == "__main__":
    unittest.main()
