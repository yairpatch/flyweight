import unittest

from flyweight import edit_corpus
from flyweight.edit_corpus import rebuild
from flyweight.transcript_audit import normalize_transcript

FILE = """\
def calculate_total(items):
    total = 0
    for item in items:
        total += foo(item)
    return total
"""


def listing(text, *, start=1, step=1):
    """A read result the way a harness renders one: numbers, tab, line."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return "\n".join(
        f"{start + index * step:6d}\t{line}" for index, line in enumerate(lines)
    )


def call(identifier, name, arguments):
    return {
        "role": "assistant",
        "text": "",
        "tool_calls": [{"id": identifier, "name": name, "arguments": arguments}],
    }


def result(identifier, text):
    return {
        "role": "tool",
        "text": text,
        "tool_result": {"id": identifier, "text": text},
    }


def read_turns(path=  "/w/app.py", text=FILE, identifier="r1"):
    return [
        call(identifier, "Read", {"file_path": path}),
        result(identifier, listing(text)),
    ]


class ReadTest(unittest.TestCase):
    def test_a_numbered_listing_becomes_the_file_s_own_bytes(self):
        corpus = rebuild(read_turns(), "/w/app.py")
        self.assertIsNotNone(corpus)
        self.assertTrue(corpus.exact)
        self.assertEqual(corpus.text, FILE.rstrip("\n"))

    def test_a_path_never_read_is_not_reconstructed(self):
        self.assertIsNone(rebuild(read_turns(), "/w/other.py"))

    def test_an_absolute_and_a_relative_spelling_are_the_same_file(self):
        corpus = rebuild(read_turns(path="/w/app.py"), "app.py")
        self.assertIsNotNone(corpus)

    def test_a_truncated_read_is_searchable_but_not_quotable(self):
        turns = read_turns()
        turns[1]["tool_result"]["text"] += "\n[File truncated: showing first 5 lines]"
        turns[1]["text"] = turns[1]["tool_result"]["text"]
        corpus = rebuild(turns, "/w/app.py")
        self.assertFalse(corpus.exact)
        self.assertIn("truncated", corpus.reason)
        self.assertIn("total += foo(item)", corpus.text)

    def test_a_listing_with_skipped_numbers_is_not_spliced_together(self):
        # Gapped numbers mean elided regions, so the lines that survive are
        # not adjacent in the file and a snippet spanning the seam would be
        # quotable and wrong. The whole result is kept unparsed instead.
        turns = read_turns()
        turns[1]["tool_result"]["text"] = listing(FILE, step=2)
        corpus = rebuild(turns, "/w/app.py")
        self.assertFalse(corpus.exact)
        self.assertIn("\t", corpus.text)

    def test_an_unnumbered_read_is_kept_but_not_trusted_verbatim(self):
        turns = read_turns()
        turns[1]["tool_result"]["text"] = FILE
        corpus = rebuild(turns, "/w/app.py")
        self.assertFalse(corpus.exact)
        self.assertEqual(corpus.text, FILE)

    def test_a_data_file_whose_first_column_jumps_is_not_a_listing(self):
        rows = "1\t2\t3\n4\t5\t6\n7\t8\t9\n"
        turns = read_turns(text=rows)
        turns[1]["tool_result"]["text"] = rows
        corpus = rebuild(turns, "/w/app.py")
        # 1, 4, 7 does not count up by one, so the shape is refused and the
        # text is kept whole rather than shredded into its later columns.
        self.assertFalse(corpus.exact)
        self.assertEqual(corpus.text, rows)

    def test_framing_around_the_listing_is_dropped(self):
        turns = read_turns()
        turns[1]["tool_result"]["text"] = (
            "Here is /w/app.py:\n```python\n" + listing(FILE) + "\n```"
        )
        corpus = rebuild(turns, "/w/app.py")
        self.assertTrue(corpus.exact)
        self.assertEqual(corpus.text, FILE.rstrip("\n"))

    def test_a_read_that_errored_leaves_the_file_unknown(self):
        turns = [
            call("r1", "Read", {"file_path": "/w/app.py"}),
            result("r1", "Error: file has not been read yet"),
        ]
        self.assertIsNone(rebuild(turns, "/w/app.py"))


class ReplayTest(unittest.TestCase):
    def test_an_applied_edit_moves_the_reconstruction_with_it(self):
        turns = read_turns() + [
            call("e1", "Edit", {
                "file_path": "/w/app.py",
                "old_string": "total += foo(item)",
                "new_string": "total += bar(item)",
            }),
            result("e1", "Edited /w/app.py"),
        ]
        corpus = rebuild(turns, "/w/app.py")
        self.assertTrue(corpus.exact)
        self.assertIn("total += bar(item)", corpus.text)
        self.assertNotIn("foo(item)", corpus.text)

    def test_a_rejected_edit_leaves_the_reconstruction_alone(self):
        turns = read_turns() + [
            call("e1", "Edit", {
                "file_path": "/w/app.py",
                "old_string": "total +=  foo(item)",
                "new_string": "total += bar(item)",
            }),
            result("e1", "Error: String to replace not found in file"),
        ]
        corpus = rebuild(turns, "/w/app.py")
        self.assertTrue(corpus.exact)
        self.assertIn("total += foo(item)", corpus.text)

    def test_a_deletion_replays_as_an_empty_replacement(self):
        turns = read_turns() + [
            call("e1", "Edit", {
                "file_path": "/w/app.py",
                "old_string": "    total = 0\n",
                "new_string": "",
            }),
            result("e1", "Edited /w/app.py"),
        ]
        corpus = rebuild(turns, "/w/app.py")
        self.assertNotIn("total = 0", corpus.text)

    def test_an_edit_of_another_file_does_not_touch_this_one(self):
        turns = read_turns() + [
            call("e1", "Edit", {
                "file_path": "/w/other.py",
                "old_string": "total += foo(item)",
                "new_string": "gone",
            }),
            result("e1", "Edited /w/other.py"),
        ]
        self.assertIn("total += foo(item)", rebuild(turns, "/w/app.py").text)

    def test_an_edit_the_reconstruction_cannot_account_for_abandons_it(self):
        # The harness applied something this text does not contain, so the
        # text is no longer the file and guessing on is the one thing that
        # would produce a wrong edit rather than a refused one.
        turns = read_turns() + [
            call("e1", "Edit", {
                "file_path": "/w/app.py",
                "old_string": "something never in the file",
                "new_string": "x",
            }),
            result("e1", "Edited /w/app.py"),
        ]
        self.assertIsNone(rebuild(turns, "/w/app.py"))

    def test_a_later_read_restarts_a_reconstruction_that_was_lost(self):
        turns = read_turns() + [
            call("e1", "Edit", {
                "file_path": "/w/app.py",
                "old_string": "never there",
                "new_string": "x",
            }),
            result("e1", "Edited /w/app.py"),
        ] + read_turns(identifier="r2", text="rebuilt = 1\nsecond = 2\nthird = 3\n")
        corpus = rebuild(turns, "/w/app.py")
        self.assertTrue(corpus.exact)
        self.assertIn("rebuilt = 1", corpus.text)

    def test_a_whole_file_write_needs_no_read_at_all(self):
        turns = [
            call("w1", "Write", {"file_path": "/w/new.py", "content": FILE}),
            result("w1", "Wrote /w/new.py"),
        ]
        corpus = rebuild(turns, "/w/new.py")
        self.assertTrue(corpus.exact)
        self.assertEqual(corpus.text, FILE)


class PairingTest(unittest.TestCase):
    def test_results_are_claimed_by_id_not_by_arrival_order(self):
        turns = [
            call("r1", "Read", {"file_path": "/w/app.py"}),
            call("r2", "Read", {"file_path": "/w/other.py"}),
            result("r2", listing("other = 1\nother = 2\nother = 3\n")),
            result("r1", listing(FILE)),
        ]
        self.assertIn("calculate_total", rebuild(turns, "/w/app.py").text)
        self.assertIn("other = 1", rebuild(turns, "/w/other.py").text)

    def test_a_harness_without_ids_pairs_in_order(self):
        turns = [
            call(None, "Read", {"file_path": "/w/app.py"}),
            result(None, listing(FILE)),
        ]
        self.assertIn("calculate_total", rebuild(turns, "/w/app.py").text)

    def test_paths_lists_what_the_session_touched_most_recent_last(self):
        turns = read_turns() + [
            call("r2", "Read", {"file_path": "/w/other.py"}),
            result("r2", listing("a = 1\nb = 2\nc = 3\n")),
        ]
        self.assertEqual(edit_corpus.paths(turns), ["/w/app.py", "/w/other.py"])


class ProtocolTest(unittest.TestCase):
    """The corpus is built from normalized turns, so both wires must reach it."""

    def test_an_anthropic_payload_reconstructs_the_same_file(self):
        payload = {
            "messages": [
                {"role": "user", "content": "fix it"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "r1", "name": "Read",
                     "input": {"file_path": "/w/app.py"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "r1",
                     "content": listing(FILE)},
                ]},
            ]
        }
        turns = normalize_transcript("anthropic", payload)
        corpus = rebuild(turns, "/w/app.py")
        self.assertTrue(corpus.exact)
        self.assertEqual(corpus.text, FILE.rstrip("\n"))

    def test_an_openai_payload_reconstructs_the_same_file(self):
        payload = {
            "messages": [
                {"role": "user", "content": "fix it"},
                {"role": "assistant", "tool_calls": [{
                    "id": "r1", "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/w/app.py"}',
                    },
                }]},
                {"role": "tool", "tool_call_id": "r1", "content": listing(FILE)},
            ]
        }
        turns = normalize_transcript("chat", payload)
        corpus = rebuild(turns, "/w/app.py")
        self.assertTrue(corpus.exact)
        self.assertEqual(corpus.text, FILE.rstrip("\n"))


if __name__ == "__main__":
    unittest.main()
