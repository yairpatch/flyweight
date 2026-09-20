"""The file a model is editing, reconstructed from the transcript.

When flyweight serves its own agent the file is on disk and there is nothing
to reconstruct. When it serves someone else's -- pi agent, opencode, anything
speaking either wire protocol -- the workspace is on the client's machine and
the runtime never sees it. All it has is what the harness put in the prompt.

That turns out to be enough, because a model that is about to edit a file has
by construction just been shown it: the harness's own contract makes it read
before it writes. So the file's text is in the transcript, and the runtime can
recover it the same way the model does -- by reading the last read of that
path and replaying every edit since.

`transcript_audit` already asks a weaker version of this question offline, to
decide whether a bad edit was the model's fault or the runtime's. This asks it
online, and has to be stricter about it: the audit only needs to know whether
some text was *present*, while a caller here wants to hand the harness bytes
that will match, which means knowing where the file's text stops and the
harness's own framing around it begins.

Two flags carry that. `text` is the best reconstruction available and is worth
searching. `exact` says the bytes can be used verbatim -- the read was
recognisably a whole file, nothing was truncated, and every edit since then
replayed cleanly. When `exact` is false a caller must not synthesise an
`old_string` from it; there is no shame in abstaining and letting the model
use the contract it already has.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import transcript_audit
from .transcript_audit import classify_call, same_path

# What a harness says when it truncates a file it was asked to read. A corpus
# built from a clipped read is a prefix of the file, which is fine to search
# and not fine to prove uniqueness against -- the rest of the file was never
# seen, and the second occurrence may well be in it.
_TRUNCATION_MARKERS = (
    "truncated",
    "clipped",
    "file is too large",
    "showing first",
    "showing only",
    "lines omitted",
    "... (rest of",
    "[snip]",
)

# A read result is usually the file with a line number glued to every line,
# which is not what is in the file. The separator varies by harness; the
# number is what identifies the format, and the arrow is Claude Code's.
_NUMBERED_LINE = re.compile(r"^[ \t]{0,8}(\d+)[\t→|](.*)$")

# How many numbered lines in a row before the prefix is taken to be the
# harness's rather than the file's. Two consecutive ones happen by accident;
# the run also has to count up by one throughout, which is the real test.
_NUMBERED_MINIMUM = 3


@dataclass(frozen=True, slots=True)
class Corpus:
    """One path's text as the model should currently believe it to be."""

    path: str
    text: str
    exact: bool
    reason: str

    def __bool__(self) -> bool:
        return bool(self.text)


def rebuild(turns: Sequence[Mapping[str, Any]], path: str) -> Corpus | None:
    """`path` as of the end of `turns`, or None when it was never established.

    The walk is deliberately forgetful. An edit that cannot be replayed drops
    the reconstruction entirely rather than carrying a text that has silently
    diverged from the file, and the next read of that path starts it over --
    which is exactly what the harness makes the model do too.
    """
    text: str | None = None
    exact = False
    reason = "no read of this path appears in the transcript"

    for call, result in _paired(turns):
        if call.path is None or not same_path(call.path, path):
            continue
        failed = result is not None and _failed(result)

        if call.kind == "read":
            if result is None:
                text, exact, reason = None, False, "a read of this path never returned"
                continue
            if failed:
                continue  # a read that errored says nothing about the file
            text, exact, reason = _from_read(result)
            continue

        if call.kind == "write":
            if failed or call.new is None:
                continue
            # The model supplied the whole file, so this is the one case where
            # the text is known without having read anything.
            text, exact, reason = call.new, True, "written in full by the model"
            continue

        if call.kind == "edit":
            if failed:
                continue  # the harness refused it; the file did not move
            if text is None:
                continue  # nothing to apply it to; a later read will resync
            if call.old is None or call.new is None:
                text, exact = None, False
                reason = "an edit of this path could not be read back"
                continue
            if call.old not in text:
                # Either the reconstruction was already wrong, or the harness
                # matched something this text does not have. Both mean the
                # same thing here, and neither is worth guessing through.
                text, exact = None, False
                reason = "an edit applied to text this reconstruction does not have"
                continue
            text = text.replace(call.old, call.new, 1)

    if text is None:
        return None
    return Corpus(path, text, exact, reason)


def paths(turns: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every path the transcript touched, most recently used last."""
    seen: dict[str, None] = {}
    for call, _ in _paired(turns):
        if call.path is not None and call.kind in ("read", "edit", "write"):
            seen.pop(call.path, None)
            seen[call.path] = None
    return list(seen)


# --------------------------------------------------------------------------
# Reading a read
# --------------------------------------------------------------------------


def _from_read(result: str) -> tuple[str, bool, str]:
    """A read result as file text, and whether it is the file's own bytes."""
    listing = _strip_line_numbers(result)
    if listing is None:
        # Not a recognisable numbered listing. It may still be the file
        # verbatim -- plenty of harnesses send exactly that -- but nothing
        # here can tell that from a file wrapped in a sentence, so the text
        # is offered for searching and not for quoting.
        return result, False, "the read result is not a recognisable file listing"
    if _truncated(result):
        return listing, False, "the read result was truncated by the harness"
    return listing, True, "read in full"


def _strip_line_numbers(result: str) -> str | None:
    """The file out of a numbered listing, or None if it is not one.

    Two conditions have to hold together. The numbered lines must be one
    unbroken run, so that whatever framing the harness wrapped the content in
    -- a path header, a code fence -- can be dropped without dropping content
    with it; and the numbers must count up by one across the whole run, which
    is what a listing does and what a file with a numeric first column does
    not, unless that column happens to be a consecutive index.

    That last case is genuinely indistinguishable from here, and it is worth
    being clear about what it costs. Such a file is shredded into its later
    columns, a caller quotes bytes from it, and the harness refuses the edit
    because they are not in the file. The failure is loud, which is the only
    property that matters: nothing wrong is written.
    """
    numbered = [
        (index, found)
        for index, line in enumerate(result.split("\n"))
        if (found := _NUMBERED_LINE.match(line)) is not None
    ]
    if len(numbered) < _NUMBERED_MINIMUM:
        return None
    if len(numbered) != numbered[-1][0] - numbered[0][0] + 1:
        # Something inside the run is not a numbered line, so the run is not
        # the file and dropping the odd line out would silently lose content.
        return None
    counts = [int(found.group(1)) for _, found in numbered]
    if any(later != earlier + 1 for earlier, later in zip(counts, counts[1:])):
        # Gaps mean elided regions. The text would be a splice of parts of the
        # file that are not adjacent in it, and a snippet spanning the seam
        # would be quotable and wrong, so this is not treated as a listing.
        return None
    return "\n".join(found.group(2) for _, found in numbered)


def _truncated(result: str) -> bool:
    lowered = result.lower()
    return any(marker in lowered for marker in _TRUNCATION_MARKERS)


def _failed(result: str) -> bool:
    lowered = result.lower()
    return any(marker in lowered for marker in transcript_audit._FAILURE_MARKERS)


# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------


def _paired(
    turns: Sequence[Mapping[str, Any]],
) -> list[tuple[transcript_audit.ToolCall, str | None]]:
    """Every tool call in order, with the text the harness answered it with.

    Ids are used when the harness sends them and order when it does not; a
    call still in flight at the end of the transcript -- the one the model is
    about to make -- pairs with nothing, which is the correct answer for it.
    """
    calls: list[transcript_audit.ToolCall] = []
    results: list[str | None] = []
    pending: list[int] = []

    for turn in turns:
        if not isinstance(turn, Mapping):
            continue
        result = turn.get("tool_result")
        if isinstance(result, Mapping):
            text = result.get("text") or ""
            index = _claim(calls, pending, result.get("id"))
            if index is not None:
                results[index] = text
        for raw in turn.get("tool_calls") or []:
            if not isinstance(raw, Mapping):
                continue
            arguments = raw.get("arguments")
            classified = classify_call(
                str(raw.get("name") or ""),
                arguments if isinstance(arguments, Mapping) else {},
            )
            if classified is None:
                continue
            calls.append(
                transcript_audit.ToolCall(
                    0, raw.get("id"), classified.name, classified.kind,
                    classified.path, classified.old, classified.new,
                )
            )
            results.append(None)
            pending.append(len(calls) - 1)
    return list(zip(calls, results))


def _claim(
    calls: Sequence[transcript_audit.ToolCall],
    pending: list[int],
    identifier: Any,
) -> int | None:
    if not pending:
        return None
    if identifier is not None:
        for position, index in enumerate(pending):
            if calls[index].id == identifier:
                return pending.pop(position)
        # An id that matches nothing outstanding is a result for a call from
        # before the transcript window; it must not consume a pending one.
        return None
    return pending.pop(0)
