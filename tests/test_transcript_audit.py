import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flyweight import transcript_audit
from flyweight.cli import main
from flyweight.server import InferenceService
from tests.test_server import StubGenerator


def _openai_record(sequence, turns, prompt_text):
    """A dump the way the recorder writes one, from already-normalized turns."""
    return {
        "sequence": sequence,
        "endpoint": "chat",
        "model": "local",
        "tools": ["Read", "Edit"],
        "prompt_tokens": len(prompt_text),
        "turns": turns,
        "turn_digests": [transcript_audit._turn_digest(turn) for turn in turns],
        "prompt_text": prompt_text,
    }


def _read_call(path):
    return {
        "role": "assistant",
        "text": "",
        "tool_calls": [
            {"id": "r1", "name": "Read", "arguments": {"file_path": path}}
        ],
    }


def _read_result(text, call_id="r1"):
    return {
        "role": "tool",
        "text": text,
        "tool_result": {"id": call_id, "text": text},
    }


def _edit_call(path, old, call_id="e1"):
    return {
        "role": "assistant",
        "text": "",
        "tool_calls": [
            {
                "id": call_id,
                "name": "Edit",
                "arguments": {
                    "file_path": path,
                    "old_string": old,
                    "new_string": "replacement",
                },
            }
        ],
    }


FILE_TEXT = "def widen(value):\n    return value * 2  # the line an edit targets\n"
OLD = "    return value * 2  # the line an edit targets"


class ClassifyCallTest(unittest.TestCase):
    def test_recognizes_each_harness_spelling_of_an_edit(self) -> None:
        for name, arguments in (
            ("Edit", {"file_path": "a.py", "old_string": "x", "new_string": "y"}),
            ("edit", {"filePath": "a.py", "oldString": "x", "newString": "y"}),
            ("str_replace_editor",
             {"path": "a.py", "old_str": "x", "new_str": "y"}),
        ):
            with self.subTest(name=name):
                call = transcript_audit.classify_call(name, arguments)
                self.assertEqual(call.kind, "edit")
                self.assertEqual(call.path, "a.py")
                self.assertEqual(call.old, "x")

    def test_recognizes_reads_writes_and_ignores_the_rest(self) -> None:
        cases = {
            "read": [
                ("Read", {"file_path": "a.py"}),
                ("read", {"filePath": "a.py"}),
                ("str_replace_editor", {"path": "a.py", "command": "view"}),
            ],
            "write": [
                ("Write", {"file_path": "a.py", "content": "body"}),
                ("write", {"filePath": "a.py", "content": "body"}),
                ("str_replace_editor",
                 {"path": "a.py", "command": "create", "file_text": "body"}),
            ],
            "other": [
                ("Bash", {"command": "ls"}),
                ("Grep", {"pattern": "x", "path": "."}),
            ],
        }
        for kind, calls in cases.items():
            for name, arguments in calls:
                with self.subTest(kind=kind, name=name):
                    self.assertEqual(
                        transcript_audit.classify_call(name, arguments).kind, kind
                    )


class NormalizeTranscriptTest(unittest.TestCase):
    def test_anthropic_tool_blocks_become_calls_and_results(self) -> None:
        turns = transcript_audit.normalize_transcript(
            "anthropic",
            {
                "system": "be brief",
                "messages": [
                    {"role": "user", "content": "widen it"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "reading"},
                            {"type": "tool_use", "id": "t1", "name": "Read",
                             "input": {"file_path": "a.py"}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "t1",
                             "content": [{"type": "text", "text": FILE_TEXT}]},
                        ],
                    },
                ],
            },
        )
        self.assertEqual([turn["role"] for turn in turns],
                         ["system", "user", "assistant", "tool"])
        self.assertEqual(turns[2]["tool_calls"][0]["name"], "Read")
        # The tool_use block must not leak into the assistant turn's text, or
        # every later containment check matches against the call rather than
        # the result it is looking for.
        self.assertEqual(turns[2]["text"], "reading")
        self.assertEqual(turns[3]["tool_result"]["text"], FILE_TEXT)

    def test_openai_string_arguments_are_parsed(self) -> None:
        turns = transcript_audit.normalize_transcript(
            "chat",
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "c1", "type": "function", "function": {
                                "name": "Edit",
                                "arguments": json.dumps({"file_path": "a.py"}),
                            }},
                        ],
                    },
                ]
            },
        )
        self.assertEqual(turns[0]["tool_calls"][0]["arguments"], {"file_path": "a.py"})

    def test_unparseable_arguments_do_not_lose_the_turn(self) -> None:
        turns = transcript_audit.normalize_transcript(
            "chat",
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "c1", "function": {
                                "name": "Edit", "arguments": "{not json"},
                             },
                        ],
                    },
                ]
            },
        )
        self.assertEqual(turns[0]["tool_calls"][0]["arguments"], {})


class AuditTest(unittest.TestCase):
    def test_edit_with_the_text_in_prompt_and_transcript_is_the_model(self) -> None:
        producing = [
            {"role": "user", "text": "widen it"},
            _read_call("a.py"),
            _read_result(FILE_TEXT),
        ]
        answering = producing + [
            _edit_call("a.py", OLD),
            _read_result("Error: String to replace not found", "e1"),
        ]
        report = transcript_audit.audit([
            _openai_record(1, producing, "prompt " + FILE_TEXT),
            _openai_record(2, answering, "prompt " + FILE_TEXT + " edit"),
        ])
        self.assertEqual(report.edits, 1)
        finding = report.findings[0]
        self.assertEqual(finding.verdict, "model")
        self.assertEqual(finding.producing_sequence, 1)
        self.assertTrue(finding.read_before)
        self.assertTrue(finding.failed)

    def test_text_in_transcript_but_missing_from_prompt_is_ours(self) -> None:
        producing = [
            {"role": "user", "text": "widen it"},
            _read_call("a.py"),
            _read_result(FILE_TEXT),
        ]
        answering = producing + [_edit_call("a.py", OLD)]
        report = transcript_audit.audit([
            # The prompt the model was given never carried the read result.
            _openai_record(1, producing, "prompt without the file"),
            _openai_record(2, answering, "prompt without the file"),
        ])
        finding = report.findings[0]
        self.assertEqual(finding.verdict, "runtime-dropped")
        self.assertTrue(finding.old_in_transcript)
        self.assertFalse(finding.old_in_prompt)
        self.assertEqual(finding.producing_sequence, 1)
        self.assertIn("request 1", finding.detail)

    def test_edit_with_no_preceding_read_is_blind(self) -> None:
        turns = [{"role": "user", "text": "widen it"}, _edit_call("a.py", OLD)]
        report = transcript_audit.audit([_openai_record(1, turns, "prompt")])
        finding = report.findings[0]
        self.assertEqual(finding.verdict, "blind")
        self.assertFalse(finding.read_before)
        self.assertFalse(finding.old_in_transcript)

    def test_a_missing_producing_request_is_unverified_not_ours(self) -> None:
        """A dump that started mid-session must not accuse the server."""
        turns = [
            {"role": "user", "text": "widen it"},
            _read_result(FILE_TEXT),
            _edit_call("a.py", OLD),
        ]
        report = transcript_audit.audit([_openai_record(9, turns, "prompt")])
        finding = report.findings[0]
        self.assertEqual(finding.verdict, "unverified")
        self.assertIsNone(finding.producing_sequence)
        self.assertEqual(report.unpaired, 1)

    def test_a_short_replaced_string_abstains_rather_than_guessing(self) -> None:
        turns = [
            {"role": "user", "text": "widen it"},
            _read_call("a.py"),
            _read_result(FILE_TEXT),
            _edit_call("a.py", "* 2"),
        ]
        report = transcript_audit.audit([_openai_record(1, turns, "prompt")])
        finding = report.findings[0]
        self.assertEqual(finding.verdict, "model")
        self.assertIsNone(finding.old_in_transcript)
        self.assertIn("too short", finding.detail)

    def test_one_edit_replayed_across_requests_is_counted_once(self) -> None:
        turns = [{"role": "user", "text": "go"}, _edit_call("a.py", OLD)]
        later = turns + [{"role": "user", "text": "again"}]
        report = transcript_audit.audit([
            _openai_record(1, turns, "prompt"),
            _openai_record(2, later, "prompt"),
        ])
        self.assertEqual(report.edits, 1)

    def test_findings_lead_with_the_runtime_and_report_reads_cleanly(self) -> None:
        blind = [{"role": "user", "text": "go"}, _edit_call("b.py", OLD, "e2")]
        dropped_producing = [
            {"role": "user", "text": "go"}, _read_call("a.py"), _read_result(FILE_TEXT),
        ]
        report = transcript_audit.audit([
            _openai_record(1, blind, "prompt"),
            _openai_record(2, dropped_producing, "no file here"),
            _openai_record(
                3, dropped_producing + [_edit_call("a.py", OLD)], "no file here"
            ),
        ])
        self.assertEqual(
            [finding.verdict for finding in report.findings],
            ["runtime-dropped", "blind"],
        )
        text = transcript_audit.format_report(report)
        self.assertIn("runtime-dropped 1", text)
        self.assertIn("runtime-dropped is ours", text)

    def test_a_clean_session_says_the_runtime_is_not_at_fault(self) -> None:
        turns = [{"role": "user", "text": "go"}, _edit_call("b.py", OLD)]
        report = transcript_audit.audit([_openai_record(1, turns, "prompt")])
        self.assertIn(
            "not the runtime", transcript_audit.format_report(report)
        )


class RecorderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())

    def _service(self, **environ):
        with patch.dict(
            "os.environ",
            {"FLYWEIGHT_TRANSCRIPT_DUMP": str(self.directory), **environ},
            clear=False,
        ):
            return InferenceService("local", StubGenerator(), max_new_tokens=32)

    def _dumps(self):
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.directory.glob("*.json"))
        ]

    def test_a_chat_request_lands_as_one_auditable_dump(self) -> None:
        service = self._service()
        service.chat_completion({
            "model": "local",
            "messages": [
                {"role": "user", "content": "widen it"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "r1", "type": "function", "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "a.py"}),
                    }},
                ]},
                {"role": "tool", "tool_call_id": "r1", "content": FILE_TEXT},
            ],
            "tools": [{"type": "function", "function": {
                "name": "Read",
                "parameters": {"type": "object", "properties": {}},
            }}],
        })
        dumps = self._dumps()
        self.assertEqual(len(dumps), 1)
        record = dumps[0]
        self.assertEqual(record["endpoint"], "chat")
        self.assertEqual(record["tools"], ["Read"])
        self.assertEqual(record["turns"][1]["tool_calls"][0]["name"], "Read")
        self.assertIn(FILE_TEXT, record["prompt_text"])
        self.assertEqual(len(record["turn_digests"]), len(record["turns"]))

    def test_prompt_text_can_be_left_out(self) -> None:
        service = self._service(FLYWEIGHT_TRANSCRIPT_PROMPT="0")
        service.chat_completion(
            {"model": "local", "messages": [{"role": "user", "content": "hi"}]}
        )
        record = self._dumps()[0]
        self.assertNotIn("prompt_text", record)
        self.assertTrue(record["prompt_sha256"])

    def test_recording_is_off_without_the_variable(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            service = InferenceService("local", StubGenerator(), max_new_tokens=32)
        service.chat_completion(
            {"model": "local", "messages": [{"role": "user", "content": "hi"}]}
        )
        self.assertEqual(self._dumps(), [])

    def test_a_broken_dump_directory_never_fails_the_request(self) -> None:
        blocker = self.directory / "not-a-directory"
        blocker.write_text("", encoding="utf-8")
        with patch.dict(
            "os.environ", {"FLYWEIGHT_TRANSCRIPT_DUMP": str(blocker / "under")},
        ):
            service = InferenceService("local", StubGenerator(), max_new_tokens=32)
        response = service.chat_completion(
            {"model": "local", "messages": [{"role": "user", "content": "hi"}]}
        )
        self.assertEqual(response["object"], "chat.completion")

    def test_an_undecodable_prompt_still_dumps_the_client_side(self) -> None:
        service = self._service()

        def explode(tokens, *, skip_special_tokens=True):
            raise RuntimeError("no round trip")

        with patch.object(service.generator.tokenizer, "decode", explode):
            service.chat_completion(
                {"model": "local", "messages": [{"role": "user", "content": "hi"}]}
            )
        record = self._dumps()[0]
        self.assertIsNone(record["prompt_sha256"])
        self.assertEqual(record["turns"][0]["text"], "hi")


class TranscriptAuditCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())

    def _write(self, records) -> None:
        for record in records:
            path = self.directory / f"{record['sequence']:06d}-chat.json"
            path.write_text(json.dumps(record), encoding="utf-8")

    def test_exits_non_zero_only_when_the_server_dropped_something(self) -> None:
        producing = [
            {"role": "user", "text": "go"}, _read_call("a.py"), _read_result(FILE_TEXT),
        ]
        self._write([
            _openai_record(1, producing, "prompt " + FILE_TEXT),
            _openai_record(2, producing + [_edit_call("a.py", OLD)], "prompt"),
        ])
        self.assertEqual(main(["transcript-audit", str(self.directory)]), 0)

        self._write([_openai_record(1, producing, "dropped")])
        self.assertEqual(main(["transcript-audit", str(self.directory)]), 1)

    def test_an_empty_directory_is_an_error_not_a_clean_bill(self) -> None:
        self.assertEqual(main(["transcript-audit", str(self.directory)]), 2)


if __name__ == "__main__":
    unittest.main()
