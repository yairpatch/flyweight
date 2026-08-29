"""The `joyai-llm` pre-tokenizer that DeepSeek-V4-Flash ships with.

The native splitter is a hand transcription of three regexes, so it is checked
against those regexes run through an actual Unicode-aware engine. `regex` (not
`re`) is required because the patterns use \\p{...} classes and a negative
lookahead.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from flyweight.v2 import V2Model

from tests.deepseek4_gguf_fixture import build_deepseek4_gguf

try:
    import regex
except ImportError:  # pragma: no cover - exercised only where regex is absent
    regex = None


# Verbatim from the reference tokenizer's LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM.
JOYAI_PATTERNS = [
    r"\p{N}{1,3}",
    r"[一-龥぀-ゟ゠-ヿ]+",
    r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+"
    r"|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+"
    r"| ?[\p{P}\p{S}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+",
]

CORPUS = [
    "Hello world",
    "  leading and trailing  ",
    "one two  three   four",
    "line one\nline two\n\n\nline three",
    "trailing whitespace at end   ",
    "tabs\tand\r\ncarriage returns",
    "1 12 123 1234 12345 1234567890",
    "version 3.14.159 and 2026-08-04",
    "snake_case camelCase kebab-case",
    "punctuation!!! ??? ...ellipsis",
    # '~' is the one ASCII character the reference puts in neither the
    # punctuation nor the symbol class, so it splits unlike every other symbol.
    "+++ ~~~ ===",
    " ~",
    "a~b ~~~ ~",
    "~/path ~user cmd~",
    "@mention #hashtag $USD 50%",
    "'quoted' \"double\" `backtick`",
    "path/to/file.txt and C:\\Windows\\System32",
    "e=mc^2 + 5*3 >= 7",
    "emoji 🙂🎉 mixed with text",
    "café naïve résumé Zoë",
    "Ω≈ç√∫˜µ≤≥÷",
    "日本語のテキストです",
    "カタカナとひらがなの混在テスト",
    "中文文本测试内容",
    "한국어 텍스트",
    "Русский текст здесь",
    "العربية نص",
    "mixed 日本語 and English 123 text",
    "中文123English日本語",
    "<｜begin▁of▁sentence｜>",
    "<think>reasoning</think>",
    "",
    " ",
    "\n",
    "\n\n\n",
    "a",
    "。、！？「」",
    "def f(x): return x ** 2  # comment",
    '{"key": "value", "n": [1, 2, 3]}',
]


# The reference does not match \p{...} against real Unicode for ASCII. It
# collapses the text -- ASCII kept as itself, each non-ASCII codepoint replaced
# by one marker character for its category -- and rewrites each category in the
# pattern as an explicit ASCII class plus that marker. These are llama.cpp's
# lists verbatim; note '~' is in neither, which is a real behavioural
# difference, not a transcription slip.
ASCII_CLASS = {
    r"\p{N}": "0-9",
    r"\p{L}": "A-Za-z",
    r"\p{P}": r"!-#%-*,-/:-;?-@\[-\]_{}",
    r"\p{M}": "",
    r"\p{S}": r"$+<->^`|",
}
MARKER = {r"\p{N}": "Ñ", r"\p{L}": "Ò", r"\p{P}": "Ó",
          r"\p{M}": "Ô", r"\p{S}": "Õ"}


def _collapse(text: str) -> str:
    out = []
    for char in text:
        cpt = ord(char)
        if cpt < 128:
            out.append(char)
        elif regex.match(r"\s", char):
            out.append("\v")  # the reference's whitespace stand-in
        else:
            category = regex.match(r"\p{N}", char) and r"\p{N}" \
                or regex.match(r"\p{L}", char) and r"\p{L}" \
                or regex.match(r"\p{P}", char) and r"\p{P}" \
                or regex.match(r"\p{M}", char) and r"\p{M}" \
                or regex.match(r"\p{S}", char) and r"\p{S}"
            out.append(MARKER[category] if category else "Ð")
    return "".join(out)


def _collapse_pattern(pattern: str) -> str:
    """Rewrite each category as its marker plus ASCII class.

    A category standing on its own becomes a bracket class; one already inside
    a class is spliced in bare, since classes cannot nest.
    """
    out: list[str] = []
    inside = False
    at = 0
    while at < len(pattern):
        char = pattern[at]
        if char == "[" and (at == 0 or pattern[at - 1] != "\\"):
            out.append("[")
            inside = True
            at += 1
            continue
        if inside and char == "]" and pattern[at - 1] != "\\":
            out.append("]")
            inside = False
            at += 1
            continue
        name = next((n for n in ASCII_CLASS if pattern.startswith(n, at)), None)
        if name is not None:
            body = MARKER[name] + ASCII_CLASS[name]
            out.append(body if inside else f"[{body}]")
            at += len(name)
            continue
        out.append(char)
        at += 1
    return "".join(out)


def _reference_split(text: str) -> tuple[str, ...]:
    """Split the way the reference does: each pattern refines the last.

    Every match becomes a piece and so does every gap between matches, with the
    scan confined to one piece at a time. Patterns that mention a Unicode
    category run against the collapsed text; the rest run against the real
    codepoints, which is exactly how llama.cpp dispatches them.
    """
    pieces = [text] if text else []
    for pattern in JOYAI_PATTERNS:
        collapsed = any(name in pattern for name in ASCII_CLASS)
        compiled = regex.compile(_collapse_pattern(pattern) if collapsed else pattern)
        refined: list[str] = []
        for piece in pieces:
            # Collapsing is codepoint-for-codepoint, so match offsets carry
            # straight back to the original piece.
            subject = _collapse(piece) if collapsed else piece
            at = 0
            for match in compiled.finditer(subject):
                start, end = match.span()
                if end == start:
                    continue
                if start > at:
                    refined.append(piece[at:start])
                refined.append(piece[start:end])
                at = end
            if at < len(piece):
                refined.append(piece[at:])
        pieces = refined
    return tuple(pieces)


@unittest.skipIf(regex is None, "the regex module is required for the oracle")
class JoyaiPretokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix="flyweight-joyai-")
        path = Path(cls.workspace.name) / "deepseek4.gguf"
        build_deepseek4_gguf(path)
        cls.model = V2Model(path)

    @classmethod
    def tearDownClass(cls):
        cls.model.close()
        cls.workspace.cleanup()

    def test_pretokenizer_matches_the_reference_patterns(self):
        for text in CORPUS:
            with self.subTest(text=text):
                self.assertEqual(
                    self.model.pretokenize(text),
                    _reference_split(text),
                )

    def test_pieces_always_reconstruct_the_input(self):
        for text in CORPUS:
            with self.subTest(text=text):
                self.assertEqual("".join(self.model.pretokenize(text)), text)

    def test_digits_group_in_threes(self):
        self.assertEqual(self.model.pretokenize("1234567"), ("123", "456", "7"))

    def test_cjk_runs_split_from_latin(self):
        self.assertEqual(
            self.model.pretokenize("abc日本語def"),
            _reference_split("abc日本語def"),
        )

    def test_tilde_belongs_to_no_category(self):
        # Every other ASCII symbol keeps its leading space in one piece; '~'
        # does not, because the reference's ASCII class lists omit it. Pinned
        # against the real tokenizer, which produces [223, 95548] for " ~~~".
        self.assertEqual(self.model.pretokenize(" +++"), (" +++",))
        self.assertEqual(self.model.pretokenize(" ~~~"), (" ", "~~~"))

    def test_a_space_stays_with_the_word_that_follows(self):
        # "\s+(?!\S)" hands the last space of a run to the next word, which is
        # what keeps " word" a single piece.
        self.assertEqual(self.model.pretokenize("a  b"), _reference_split("a  b"))


if __name__ == "__main__":
    unittest.main()
