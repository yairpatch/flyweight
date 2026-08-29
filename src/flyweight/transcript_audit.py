"""Per-request transcript dumps, and the read-before-edit audit over them.

A harness that edits a file it never read produces a bad edit, and from the
outside there is no way to tell whose fault it is: the model may have guessed,
or the read result may have reached the client and then been lost on the way
to the model -- by a prompt-rendering bug, a truncation, or a prefix cache
serving a turn that no longer matches. The two have opposite fixes, so the
diagnostic has to separate them before anything is changed.

It separates them by recording both sides of the boundary for every request:
the transcript the client sent, and the prompt text the model actually saw.
An edit call in request N+1's transcript was produced by request N, so the
audit can ask of each edit, in order:

  1. was the text it is replacing anywhere in the client's transcript?
  2. was it in the rendered prompt of the request that produced the call?

No to (1) is the model editing blind, and no fix of ours reaches it. Yes to
(1) and no to (2) is ours, and says exactly which request dropped it.

Recording is off unless FLYWEIGHT_TRANSCRIPT_DUMP names a directory. It writes
one JSON file per request and holds nothing in memory between them, so a
session can run as long as it needs to; the prompt text dominates the size
(roughly 4 bytes per prompt token per request), and FLYWEIGHT_TRANSCRIPT_PROMPT=0
drops it, leaving a digest and the answer to (1) only.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

# Argument spellings for the same idea across harnesses. Matching on argument
# shape rather than tool name is deliberate: the name is whatever the harness
# chose -- Claude Code's `Edit`, opencode's `edit`, `str_replace_editor`,
# `apply_patch` -- while the arguments of a search/replace edit are the same
# three fields everywhere, under four spellings.
_PATH_KEYS = ("file_path", "filePath", "path", "file", "filename", "target_file")
_OLD_KEYS = ("old_string", "oldString", "old_str", "search", "find")
_NEW_KEYS = ("new_string", "newString", "new_str", "replace")
_CONTENT_KEYS = ("content", "contents", "file_text", "new_text", "text")

# Names that read a file, for the weaker check: an edit whose replaced text is
# absent may still have been preceded by a read of the same path, which means
# the model had the file and mis-transcribed it rather than inventing it.
_READ_NAMES = ("read", "view", "cat", "open", "readfile", "read_file", "get_file")

# What a harness says when it rejects or fails an edit. Used only to mark
# which edits actually went wrong, so a session's findings can be ranked by
# real damage instead of by suspicion.
_FAILURE_MARKERS = (
    "has not been read",
    "not been read yet",
    "must read",
    "read the file first",
    "string to replace not found",
    "old_string not found",
    "oldstring not found",
    "did not match",
    "no changes to make",
    "found 0 matches",
    "file has been modified since",
    "no replacement was performed",
)

# Enough of a replaced string to be a meaningful search key. A one-line
# old_string of a few characters ("}" , "pass") occurs in any file by chance,
# so a containment hit on it proves nothing; below this the check abstains
# rather than reporting a result it cannot stand behind.
_MIN_MATCH_CHARS = 24


def recorder_from_env() -> "Recorder | None":
    """The recorder this process was started with, or None when it is off."""
    directory = os.environ.get("FLYWEIGHT_TRANSCRIPT_DUMP")
    if not directory:
        return None
    keep_prompt = os.environ.get("FLYWEIGHT_TRANSCRIPT_PROMPT", "1") not in (
        "0", "false", "no",
    )
    return Recorder(Path(directory), keep_prompt=keep_prompt)


class Recorder:
    """Writes one dump file per request, and never fails a request.

    A diagnostic that can 500 a live session is worse than no diagnostic, so
    every entry point swallows its own errors. The sequence counter is what
    orders the session -- file mtimes do not, at the sub-millisecond spacing
    of a streaming harness -- and it is handed out under a lock because the
    server serves requests on a thread each.
    """

    def __init__(self, directory: Path, *, keep_prompt: bool = True) -> None:
        self.directory = directory
        self.keep_prompt = keep_prompt
        self._lock = threading.Lock()
        self._sequence = 0
        self._broken = False
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._broken = True

    def record(
        self,
        *,
        endpoint: str,
        payload: Mapping[str, Any],
        prompt_ids: Sequence[int],
        prompt_text: str | None,
        tools: Sequence[Mapping[str, Any]],
    ) -> None:
        if self._broken:
            return
        try:
            with self._lock:
                self._sequence += 1
                sequence = self._sequence
            turns = normalize_transcript(endpoint, payload)
            record: dict[str, Any] = {
                "sequence": sequence,
                "endpoint": endpoint,
                "model": payload.get("model"),
                "tools": _tool_names(tools),
                "tool_choice": _plain(payload.get("tool_choice")),
                "prompt_tokens": len(prompt_ids),
                "prompt_sha256": (
                    hashlib.sha256(prompt_text.encode("utf-8", "replace")).hexdigest()
                    if prompt_text is not None
                    else None
                ),
                "turns": turns,
                # The per-turn digests are how the audit pairs an edit call
                # with the request that produced it: that request's transcript
                # is this one's, one turn shorter. Comparing digests rather
                # than sequence numbers survives retries, parallel slots and a
                # dump that started mid-session.
                "turn_digests": [_turn_digest(turn) for turn in turns],
            }
            if self.keep_prompt and prompt_text is not None:
                record["prompt_text"] = prompt_text
            path = self.directory / f"{sequence:06d}-{endpoint}.json"
            path.write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            # Including anything the payload does that json cannot serialize.
            # One lost dump is a gap in the audit; a raised exception here is
            # a failed generation.
            return


def normalize_transcript(
    endpoint: str, payload: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Both wire protocols reduced to one ordered list of turns.

    The audit runs over the client's own structure rather than the rendered
    prompt, because that is the side where a tool call is still structured
    data; the rendered form differs per architecture and would make every
    check a template-specific regex.
    """
    if endpoint == "anthropic":
        return _anthropic_turns(payload)
    return _openai_turns(payload)


def _openai_turns(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return turns
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", ""))
        turn: dict[str, Any] = {
            "role": role, "text": _flatten_text(message.get("content")),
        }
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            parsed = [_openai_call(call) for call in calls]
            calls_out = [call for call in parsed if call is not None]
            if calls_out:
                turn["tool_calls"] = calls_out
        if role == "tool":
            turn["tool_result"] = {
                "id": _plain(message.get("tool_call_id")),
                "text": turn["text"],
            }
        turns.append(turn)
    return turns


def _openai_call(call: Any) -> dict[str, Any] | None:
    if not isinstance(call, Mapping):
        return None
    function = call.get("function")
    if not isinstance(function, Mapping):
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError):
            arguments = {}
    if not isinstance(arguments, Mapping):
        arguments = {}
    return {
        "id": _plain(call.get("id")),
        "name": str(function.get("name") or ""),
        "arguments": dict(arguments),
    }


def _anthropic_turns(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    system = payload.get("system")
    if system is not None:
        turns.append({"role": "system", "text": _flatten_text(system)})
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return turns
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", ""))
        content = message.get("content")
        turn: dict[str, Any] = {"role": role, "text": _flatten_text(content)}
        calls: list[dict[str, Any]] = []
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, Mapping):
                    continue
                kind = block.get("type")
                if kind == "tool_use":
                    arguments = block.get("input")
                    calls.append({
                        "id": _plain(block.get("id")),
                        "name": str(block.get("name") or ""),
                        "arguments": dict(arguments)
                        if isinstance(arguments, Mapping)
                        else {},
                    })
                elif kind == "tool_result":
                    # A result block rides inside a user turn on this
                    # protocol, so the turn is split out into the same shape
                    # the OpenAI side produces rather than left nested.
                    turns.append({
                        "role": "tool",
                        "text": _flatten_text(block.get("content")),
                        "tool_result": {
                            "id": _plain(block.get("tool_use_id")),
                            "text": _flatten_text(block.get("content")),
                        },
                    })
        if calls:
            turn["tool_calls"] = calls
        if turn["text"] or calls or not isinstance(content, list):
            turns.append(turn)
    return turns


def _flatten_text(content: Any) -> str:
    """Every content shape either protocol accepts, as one string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        for key in ("text", "content"):
            if key in content:
                return _flatten_text(content[key])
        return ""
    if isinstance(content, list):
        # Tool blocks are split into turns of their own, so leaving them in
        # the surrounding text too would double every file the audit searches
        # for -- and make a user turn that carries nothing but a tool result
        # look like the user pasting the file in.
        return "".join(
            _flatten_text(part)
            for part in content
            if not (
                isinstance(part, Mapping)
                and part.get("type") in ("tool_use", "tool_result")
            )
        )
    return ""


def _tool_names(tools: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            names.append(function["name"])
    return names


def _plain(value: Any) -> Any:
    """Anything not JSON-native reduced to a string, so a dump always writes."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _turn_digest(turn: Mapping[str, Any]) -> str:
    payload = json.dumps(turn, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8", "replace")).hexdigest()[:16]


# --------------------------------------------------------------------------
# The audit
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolCall:
    turn: int
    id: Any
    name: str
    kind: str  # "read" | "edit" | "write" | "other"
    path: str | None
    old: str | None


@dataclass(frozen=True, slots=True)
class Finding:
    sequence: int
    turn: int
    tool: str
    path: str | None
    verdict: str
    read_before: bool
    old_in_transcript: bool | None
    old_in_prompt: bool | None
    producing_sequence: int | None
    failed: bool
    detail: str


@dataclass
class Report:
    records: int = 0
    edits: int = 0
    findings: list[Finding] = field(default_factory=list)
    unpaired: int = 0

    def by_verdict(self, verdict: str) -> list[Finding]:
        return [finding for finding in self.findings if finding.verdict == verdict]


def load(directory: Path) -> list[dict[str, Any]]:
    """Every dump in the directory, in the order the server wrote them."""
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and "turns" in record:
            records.append(record)
    records.sort(key=lambda record: record.get("sequence", 0))
    return records


def classify_call(name: str, arguments: Mapping[str, Any]) -> ToolCall | None:
    """What a call does to a file, read off its arguments.

    `str_replace_editor` is the one tool whose behaviour lives in an argument
    rather than in its shape, so its command is consulted; everything else is
    decided by which of the four field families it carries.
    """
    path = _first_string(arguments, _PATH_KEYS)
    old = _first_string(arguments, _OLD_KEYS)
    has_new = any(key in arguments for key in _NEW_KEYS)
    lowered = name.lower()
    command = arguments.get("command")
    if isinstance(command, str):
        if command == "view":
            return ToolCall(0, None, name, "read", path, None)
        if command in ("create", "insert"):
            return ToolCall(0, None, name, "write", path, None)
    if old is not None and (has_new or "replace" in lowered or "edit" in lowered):
        return ToolCall(0, None, name, "edit", path, old)
    if path is not None and any(key in arguments for key in _CONTENT_KEYS):
        # A whole-file write. Nothing to match against the file's prior text,
        # so only the weaker "was it ever read" question applies.
        return ToolCall(0, None, name, "write", path, None)
    if path is not None and any(word in lowered for word in _READ_NAMES):
        return ToolCall(0, None, name, "read", path, None)
    return ToolCall(0, None, name, "other", path, None)


def _first_string(arguments: Mapping[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _calls(record: Mapping[str, Any]) -> Iterator[ToolCall]:
    for index, turn in enumerate(record.get("turns", [])):
        for call in turn.get("tool_calls", []) if isinstance(turn, Mapping) else []:
            if not isinstance(call, Mapping):
                continue
            arguments = call.get("arguments")
            classified = classify_call(
                str(call.get("name") or ""),
                arguments if isinstance(arguments, Mapping) else {},
            )
            if classified is None:
                continue
            yield ToolCall(
                index, call.get("id"), classified.name, classified.kind,
                classified.path, classified.old,
            )


def audit(records: Sequence[Mapping[str, Any]]) -> Report:
    """Every edit in the session, with the side of the boundary that lost it.

    Each edit is judged from the last record that contains it, because only a
    later request carries the harness's answer to the call -- whether the edit
    was rejected or applied.
    """
    report = Report(records=len(records))
    # A record is keyed by its transcript prefix, so the request that produced
    # a call at turn i can be found by the digest list ending exactly there.
    by_prefix: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for record in records:
        digests = tuple(record.get("turn_digests", []))
        # First writer wins: a retried request re-renders the same prefix, and
        # the first one is the one whose prompt produced the call.
        by_prefix.setdefault(digests, record)

    seen: set[tuple[Any, int, str]] = set()
    for record in records:
        turns = record.get("turns", [])
        digests = record.get("turn_digests", [])
        for call in _calls(record):
            if call.kind not in ("edit", "write"):
                continue
            key = (call.id, call.turn, call.name)
            if call.id is not None and key in seen:
                continue
            seen.add(key)
            report.edits += 1
            report.findings.append(
                _judge(record, turns, digests, call, by_prefix, report)
            )
    order = {"runtime-dropped": 0, "blind": 1, "unverified": 2, "model": 3}
    report.findings.sort(
        key=lambda finding: (
            order.get(finding.verdict, 9), not finding.failed, finding.sequence
        )
    )
    return report


def _judge(
    record: Mapping[str, Any],
    turns: Sequence[Mapping[str, Any]],
    digests: Sequence[str],
    call: ToolCall,
    by_prefix: Mapping[tuple[str, ...], Mapping[str, Any]],
    report: Report,
) -> Finding:
    before = turns[: call.turn]
    read_before = any(
        prior.kind == "read"
        and prior.path is not None
        and call.path is not None
        and _same_path(prior.path, call.path)
        for prior in _calls({"turns": before})
    )
    failed = _call_failed(turns, call)

    producer = by_prefix.get(tuple(digests[: call.turn]))
    producing_sequence = producer.get("sequence") if producer else None
    if producer is None:
        report.unpaired += 1

    if call.old is None or len(call.old) < _MIN_MATCH_CHARS:
        # A whole-file write, or a replaced string too short to search for.
        verdict = "blind" if not read_before else "model"
        detail = (
            "no read of this path precedes the call"
            if not read_before
            else "path was read earlier in the session"
        )
        if call.old is not None:
            detail += f"; replaced text too short to verify ({len(call.old)} chars)"
        return Finding(
            record.get("sequence", 0), call.turn, call.name, call.path, verdict,
            read_before, None, None, producing_sequence, failed, detail,
        )

    old_in_transcript = any(
        call.old in (turn.get("text") or "") for turn in before
    )
    prompt_text = producer.get("prompt_text") if producer else None
    old_in_prompt: bool | None = None
    if isinstance(prompt_text, str):
        old_in_prompt = call.old in prompt_text

    if not old_in_transcript:
        return Finding(
            record.get("sequence", 0), call.turn, call.name, call.path, "blind",
            read_before, False, old_in_prompt, producing_sequence, failed,
            "the replaced text appears nowhere earlier in the client's own "
            "transcript" + ("" if not read_before else ", though the path was read"),
        )
    if old_in_prompt is False:
        return Finding(
            record.get("sequence", 0), call.turn, call.name, call.path,
            "runtime-dropped", read_before, True, False, producing_sequence, failed,
            "the client sent the text but the rendered prompt for request "
            f"{producing_sequence} did not contain it",
        )
    if old_in_prompt is None:
        return Finding(
            record.get("sequence", 0), call.turn, call.name, call.path,
            "unverified", read_before, True, None, producing_sequence, failed,
            "the client sent the text; no rendered prompt was recorded for the "
            "request that produced the call"
            + ("" if producer is not None else " (no matching request in the dump)"),
        )
    return Finding(
        record.get("sequence", 0), call.turn, call.name, call.path, "model",
        read_before, True, True, producing_sequence, failed,
        "the model had the text in its prompt",
    )


def _same_path(left: str, right: str) -> bool:
    if left == right:
        return True
    # Harnesses mix absolute and workspace-relative spellings of one file
    # within a session, so a suffix match is what actually pairs them.
    return left.endswith(right) or right.endswith(left)


def _call_failed(turns: Sequence[Mapping[str, Any]], call: ToolCall) -> bool:
    """Whether the harness rejected or failed this specific call."""
    for turn in turns[call.turn + 1 :]:
        result = turn.get("tool_result")
        if not isinstance(result, Mapping):
            continue
        if call.id is not None and result.get("id") not in (None, call.id):
            continue
        text = (result.get("text") or "").lower()
        return any(marker in text for marker in _FAILURE_MARKERS)
    return False


def format_report(report: Report, *, verbose: bool = False) -> str:
    lines = [
        f"{report.records} requests, {report.edits} edit/write calls",
    ]
    if not report.edits:
        lines.append("")
        lines.append(
            "No edits in this dump. Run the harness through a change to a file "
            "before auditing."
        )
        return "\n".join(lines)
    counts = {
        verdict: len(report.by_verdict(verdict))
        for verdict in ("runtime-dropped", "blind", "unverified", "model")
    }
    failed = sum(1 for finding in report.findings if finding.failed)
    lines.append(
        f"  runtime-dropped {counts['runtime-dropped']}   "
        f"blind {counts['blind']}   "
        f"model {counts['model']}   "
        f"unverified {counts['unverified']}   "
        f"({failed} rejected by the harness)"
    )
    if report.unpaired:
        lines.append(
            f"  {report.unpaired} calls had no matching producing request in "
            "the dump -- expected for calls made before recording started"
        )
    lines.append("")
    if counts["runtime-dropped"]:
        lines.append(
            "runtime-dropped is ours: the client sent the file text and the "
            "prompt we rendered did not carry it."
        )
    elif counts["blind"] and not counts["runtime-dropped"]:
        lines.append(
            "Nothing was dropped between client and prompt. The blind edits "
            "are the model or the harness's tool contract, not the runtime."
        )
    lines.append("")
    shown = report.findings if verbose else report.findings[:20]
    for finding in shown:
        mark = "FAILED" if finding.failed else "ok"
        lines.append(
            f"  [{finding.verdict}] request {finding.sequence} turn "
            f"{finding.turn} {finding.tool}({finding.path or '?'}) -> {mark}"
        )
        lines.append(f"      {finding.detail}")
    if not verbose and len(report.findings) > len(shown):
        lines.append(f"  ... {len(report.findings) - len(shown)} more (--verbose)")
    return "\n".join(lines)
