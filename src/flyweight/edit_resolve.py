"""Locating an edit from a description of it, rather than a copy of it.

A search/replace edit tool asks the model for the exact bytes it wants
replaced. That is a strong guarantee -- an edit that applies is an edit that
hit what the model was looking at -- and it is paid for twice. Once in output
tokens, because changing one line inside a forty-line function costs the model
the forty lines needed to name that line unambiguously; and once in failures,
because the copy has to be byte-perfect and the indentation of a line the
model never wrote is the easiest thing in a file to get wrong.

The second cost is the worse one, and not for the reason it looks. The call
that fails carries the replacement too, so a mistyped space throws away the
part of the generation that was actually reasoning about the code, and the
retry regenerates all of it.

This module is the other half of that contract. The model says what it wants
changed and, optionally, which block to look in -- `find="foo()"`,
`scope="calculateTotal"` -- and the search happens here, against the file, in
offsets that are right by construction because they came out of the file.

Three rules keep it from trading a loud failure for a silent wrong edit:

  * A match is never guessed. When `find` lands in more than one place the
    answer is the list of places with their line numbers, and the caller asks
    the model to pick one by number -- a repair that costs a token instead of
    a regeneration.
  * Whitespace is elastic only where it was ambiguous to begin with: the
    indentation a line carries, and runs of spaces and tabs between things
    that are not whitespace. Every other byte is literal, and the elastic
    match is only consulted when the literal one found nothing at all.
  * `scope` narrows, it never selects. A block boundary read wrongly can only
    lose a match, never move one, because the match is still `find`'s and
    nothing else is ever replaced.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import Literal, Sequence

# How many places an ambiguous `find` reports back. A `find` of "}" in a large
# file has hundreds, and listing them would cost more tokens than the copy
# this module exists to avoid -- past this the model is told to narrow the
# scope instead, which is the answer it would have to reach anyway.
_MAX_CANDIDATES = 12

# A candidate line is shown so the model can tell two matches apart, not so it
# can read the code again; it already has the file.
_PREVIEW_CHARS = 160

Problem = Literal["empty_find", "no_scope", "not_in_scope", "not_found", "no_such_occurrence"]


@dataclass(frozen=True, slots=True)
class Match:
    """One place in the file: what to replace, and something readable."""

    start: int
    end: int
    line: int  # 1-based, of the line `start` falls on
    preview: str


@dataclass(frozen=True, slots=True)
class Outcome:
    """Resolved, ambiguous, or missing -- never a guess.

    `matches` is the ranges to replace when resolved and the places to choose
    between when ambiguous, so a caller that only wants to report back can
    treat the two the same way.
    """

    kind: Literal["located", "ambiguous", "missing"]
    matches: tuple[Match, ...] = ()
    problem: Problem | None = None
    detail: str = ""

    @property
    def located(self) -> bool:
        return self.kind == "located"


def locate(
    text: str,
    *,
    find: str,
    scope: str | None = None,
    occurrence: int | Literal["all"] | None = None,
) -> Outcome:
    """Where in `text` the described edit goes.

    `occurrence` is how the model answers an earlier ambiguous outcome: the
    1-based number of the candidate it meant, or "all" to take every one of
    them. Leaving it out means the edit must resolve on its own.
    """
    if not find or not find.strip():
        return Outcome(
            "missing",
            problem="empty_find",
            detail="find must be a non-empty piece of the file's text",
        )

    lines = _Lines(text)
    if scope is None:
        spans: list[tuple[int, int]] = [(0, len(text))]
    else:
        spans = _scope_spans(text, lines, scope)
        if not spans:
            return Outcome(
                "missing",
                problem="no_scope",
                detail=f"no block matching scope {scope!r} appears in the file",
            )

    matches = _search_spans(text, lines, find, spans)
    if not matches:
        return _nothing_found(text, lines, find, scope, spans)

    if occurrence == "all":
        return Outcome(
            "located",
            tuple(matches),
            detail=f"{len(matches)} occurrences",
        )
    if isinstance(occurrence, int):
        if occurrence < 1 or occurrence > len(matches):
            return Outcome(
                "missing",
                tuple(matches[:_MAX_CANDIDATES]),
                problem="no_such_occurrence",
                detail=(
                    f"occurrence {occurrence} was asked for but find matches "
                    f"{len(matches)} place{'' if len(matches) == 1 else 's'} here"
                ),
            )
        return Outcome(
            "located",
            (matches[occurrence - 1],),
            detail=f"occurrence {occurrence} of {len(matches)}",
        )

    if len(matches) > 1:
        return Outcome(
            "ambiguous",
            tuple(matches[:_MAX_CANDIDATES]),
            detail=_ambiguous_detail(len(matches), scope),
        )
    return Outcome("located", (matches[0],), detail=f"line {matches[0].line}")


def apply(text: str, matches: Sequence[Match], replacement: str) -> str:
    """`text` with every match replaced, right to left so offsets hold."""
    out = text
    for match in sorted(matches, key=lambda found: found.start, reverse=True):
        out = out[: match.start] + replacement + out[match.end :]
    return out


# --------------------------------------------------------------------------
# Lines
# --------------------------------------------------------------------------


class _Lines:
    """Line offsets for a file, so a match can name where it landed."""

    def __init__(self, text: str) -> None:
        self.text = text
        starts = [0]
        index = text.find("\n")
        while index != -1:
            starts.append(index + 1)
            index = text.find("\n", index + 1)
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def number(self, offset: int) -> int:
        """The 1-based line an offset falls on."""
        return bisect.bisect_right(self.starts, offset)

    def span(self, index: int) -> tuple[int, int]:
        """The 0-based line's offsets, newline excluded."""
        start = self.starts[index]
        end = (
            self.starts[index + 1] - 1
            if index + 1 < len(self.starts)
            else len(self.text)
        )
        return start, end

    def body(self, index: int) -> str:
        start, end = self.span(index)
        return self.text[start:end].rstrip("\r")

    def end_of(self, index: int) -> int:
        """Just past the line, its newline included -- where a block stops."""
        return (
            self.starts[index + 1]
            if index + 1 < len(self.starts)
            else len(self.text)
        )


def _match_at(lines: _Lines, start: int, end: int) -> Match:
    number = lines.number(start)
    preview = lines.body(number - 1).strip()
    if len(preview) > _PREVIEW_CHARS:
        preview = preview[: _PREVIEW_CHARS - 1] + "…"
    return Match(start, end, number, preview)


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def _elastic(needle: str) -> re.Pattern[str] | None:
    """`needle` with its whitespace made elastic and every other byte literal.

    Splitting on whitespace and rejoining with `\\s+` keeps the distinction
    that matters: where the needle had whitespace the file must have some, and
    where it had none the file must have none. A needle stripped at both ends
    is what makes the indentation the model guessed at irrelevant.
    """
    parts = [re.escape(part) for part in needle.strip().split() if part]
    if not parts:
        return None
    return re.compile(r"\s+".join(parts))


def _exact_spans(text: str, needle: str, lo: int, hi: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    index = text.find(needle, lo, hi)
    while index != -1:
        spans.append((index, index + len(needle)))
        # Non-overlapping: two overlapping hits are one place to a reader, and
        # replacing both would corrupt the text between them.
        index = text.find(needle, index + len(needle), hi)
    return spans


def _spans_in(text: str, needle: str, lo: int, hi: int) -> list[tuple[int, int]]:
    """Literal hits, or elastic ones when the literal search found nothing."""
    spans = _exact_spans(text, needle, lo, hi)
    if spans:
        return spans
    pattern = _elastic(needle)
    if pattern is None:
        return []
    return [(found.start(), found.end()) for found in pattern.finditer(text, lo, hi)]


def _search_spans(
    text: str, lines: _Lines, find: str, spans: Sequence[tuple[int, int]]
) -> list[Match]:
    """Every place `find` lands inside any of the search ranges.

    The ranges can overlap -- two scope blocks where one nests in the other --
    so hits are deduplicated by offset rather than trusted to be distinct.
    """
    seen: dict[tuple[int, int], Match] = {}
    for lo, hi in spans:
        for start, end in _spans_in(text, find, lo, hi):
            seen.setdefault((start, end), _match_at(lines, start, end))
    return [seen[key] for key in sorted(seen)]


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _scope_spans(text: str, lines: _Lines, scope: str) -> list[tuple[int, int]]:
    """The block each header line matching `scope` opens.

    Two shapes cover what models write. A header that leaves a brace open owns
    everything up to the brace that closes it; one that does not -- a Python
    `def`, a YAML key -- owns the lines indented past it. Neither is a parser
    and neither has to be: the range only narrows where `find` is searched, so
    a boundary read wrongly loses a match and shows the model nothing it did
    not ask for.
    """
    needle = scope.strip()
    if not needle:
        return []
    pattern = _elastic(needle)
    headers: list[int] = []
    for index in range(len(lines)):
        body = lines.body(index)
        hit = needle in body or (pattern is not None and pattern.search(body) is not None)
        if hit:
            headers.append(index)
    return [(lines.starts[index], _block_end(text, lines, index)) for index in headers]


def _block_end(text: str, lines: _Lines, index: int) -> int:
    header = lines.body(index)
    depth = header.count("{") - header.count("}")
    if depth > 0:
        return _brace_end(text, lines.end_of(index), depth)
    return _indent_end(lines, index)


def _brace_end(text: str, start: int, depth: int) -> int:
    """Just past the brace that closes the header's, counting naively.

    Braces inside strings and comments are counted too. Getting that right
    means a lexer per language; getting it wrong stretches or clips a search
    range, which is recoverable, so the naive count is the right trade here.
    """
    for offset in range(start, len(text)):
        char = text[offset]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return offset + 1
    return len(text)


def _indent_end(lines: _Lines, index: int) -> int:
    base = _indent(lines.body(index))
    last = index
    for probe in range(index + 1, len(lines)):
        body = lines.body(probe)
        if not body.strip():
            continue  # a blank line inside a block does not end it
        if _indent(body) <= base:
            break
        last = probe
    return lines.end_of(last)


# --------------------------------------------------------------------------
# Failure
# --------------------------------------------------------------------------


def _nothing_found(
    text: str,
    lines: _Lines,
    find: str,
    scope: str | None,
    spans: Sequence[tuple[int, int]],
) -> Outcome:
    """Why there was no match, in the terms the model can act on.

    "Not in that block, but here it is at line 210" is a different instruction
    from "not in this file at all": the first is answered by changing `scope`,
    the second by reading the file again. Telling them apart costs one more
    search and saves a turn.
    """
    if scope is not None:
        elsewhere = _search_spans(text, lines, find, [(0, len(text))])
        if elsewhere:
            shown = elsewhere[:_MAX_CANDIDATES]
            where = ", ".join(str(found.line) for found in shown)
            more = "" if len(elsewhere) == len(shown) else ", …"
            return Outcome(
                "missing",
                tuple(shown),
                problem="not_in_scope",
                detail=(
                    f"find does not appear inside scope {scope!r}, but it does "
                    f"appear elsewhere in the file at line {where}{more}"
                ),
            )
    return Outcome(
        "missing",
        problem="not_found",
        detail=(
            "find does not appear in the file, even ignoring indentation and "
            "the width of runs of spaces; read the file again"
        ),
    )


def _ambiguous_detail(count: int, scope: str | None) -> str:
    where = f" inside scope {scope!r}" if scope is not None else ""
    if count > _MAX_CANDIDATES:
        return (
            f"find matches {count} places{where}; the first {_MAX_CANDIDATES} "
            "are listed -- narrow it with a longer find or a scope"
        )
    return (
        f"find matches {count} places{where}; re-send with occurrence set to "
        "the number of the one you meant, or replace_all to take every one"
    )
