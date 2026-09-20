import unittest

from flyweight import edit_resolve
from flyweight.edit_resolve import apply, locate

JS = """\
function calculateTotal(items) {
    let total = 0;
    for (const item of items) {
        total += foo(item);
    }
    return foo(total);
}

function calculateTax(amount) {
    return foo(amount) * 0.2;
}
"""

PY = """\
def calculate_total(items):
    total = 0
    for item in items:
        total += foo(item)
    return foo(total)


def calculate_tax(amount):
    return foo(amount) * 0.2
"""


class LocateTest(unittest.TestCase):
    def test_a_unique_find_resolves_to_its_line(self):
        outcome = locate(JS, find="foo(item)")
        self.assertEqual(outcome.kind, "located")
        self.assertEqual(len(outcome.matches), 1)
        self.assertEqual(outcome.matches[0].line, 4)
        self.assertEqual(
            JS[outcome.matches[0].start : outcome.matches[0].end], "foo(item)"
        )

    def test_several_places_are_listed_rather_than_guessed(self):
        outcome = locate(JS, find="foo(")
        self.assertEqual(outcome.kind, "ambiguous")
        self.assertEqual([found.line for found in outcome.matches], [4, 6, 10])
        # The preview is what lets the model tell them apart in one glance.
        self.assertEqual(outcome.matches[0].preview, "total += foo(item);")

    def test_occurrence_picks_one_of_the_listed_places(self):
        outcome = locate(JS, find="foo(", occurrence=2)
        self.assertEqual(outcome.kind, "located")
        self.assertEqual(outcome.matches[0].line, 6)

    def test_occurrence_all_takes_every_place(self):
        outcome = locate(JS, find="foo(", occurrence="all")
        self.assertEqual(outcome.kind, "located")
        self.assertEqual([found.line for found in outcome.matches], [4, 6, 10])

    def test_an_occurrence_past_the_end_is_refused_with_the_count(self):
        outcome = locate(JS, find="foo(", occurrence=9)
        self.assertEqual(outcome.kind, "missing")
        self.assertEqual(outcome.problem, "no_such_occurrence")
        self.assertIn("3 places", outcome.detail)

    def test_an_empty_find_is_refused(self):
        self.assertEqual(locate(JS, find="   ").problem, "empty_find")


class WhitespaceTest(unittest.TestCase):
    """The failure this module exists for: the copy is right, a space is not."""

    def test_a_run_of_spaces_the_model_widened_still_matches(self):
        outcome = locate(JS, find="total  +=  foo(item);")
        self.assertEqual(outcome.kind, "located")
        self.assertEqual(outcome.matches[0].line, 4)
        self.assertEqual(
            JS[outcome.matches[0].start : outcome.matches[0].end],
            "total += foo(item);",
        )

    def test_a_multi_line_find_indented_wrongly_still_matches(self):
        # What a model writes when it is reconstructing a block from memory
        # rather than copying it: the right code, the wrong indentation.
        outcome = locate(
            JS,
            find="for (const item of items) {\n  total += foo(item);\n}",
        )
        self.assertEqual(outcome.kind, "located")
        self.assertEqual(outcome.matches[0].line, 3)
        replaced = JS[outcome.matches[0].start : outcome.matches[0].end]
        self.assertTrue(replaced.startswith("for (const item"))
        self.assertTrue(replaced.endswith("}"))

    def test_lf_from_the_model_matches_a_crlf_file(self):
        outcome = locate(
            JS.replace("\n", "\r\n"), find="let total = 0;\nfor (const item"
        )
        self.assertEqual(outcome.kind, "located")
        self.assertEqual(outcome.matches[0].line, 2)

    def test_a_literal_hit_wins_over_an_elastic_one(self):
        # "a  b" exists verbatim; the elastic form would also match "a b" two
        # lines down, and letting it would make a resolvable edit ambiguous.
        text = "x = a  b\ny = 1\nz = a b\n"
        outcome = locate(text, find="a  b")
        self.assertEqual(outcome.kind, "located")
        self.assertEqual(outcome.matches[0].line, 1)

    def test_whitespace_the_file_has_and_the_find_does_not_is_not_invented(self):
        # Elastic means "some whitespace here", never "whitespace may appear
        # here" -- otherwise `foo()` would match `foo ()` and `f oo()`.
        self.assertEqual(locate(JS, find="total+= foo(item);").kind, "missing")


class ScopeTest(unittest.TestCase):
    def test_a_brace_block_narrows_an_otherwise_ambiguous_find(self):
        outcome = locate(JS, find="foo(", scope="function calculateTax")
        self.assertEqual(outcome.kind, "located")
        self.assertEqual(outcome.matches[0].line, 10)

    def test_an_indented_block_narrows_it_too(self):
        outcome = locate(PY, find="foo(", scope="def calculate_tax")
        self.assertEqual(outcome.kind, "located")
        self.assertEqual(outcome.matches[0].line, 9)

    def test_a_block_stops_before_the_next_one_at_its_own_indent(self):
        outcome = locate(PY, find="foo(", scope="def calculate_total")
        self.assertEqual(outcome.kind, "ambiguous")
        self.assertEqual([found.line for found in outcome.matches], [4, 5])

    def test_a_scope_still_leaves_a_genuinely_repeated_find_ambiguous(self):
        outcome = locate(JS, find="foo(", scope="function calculateTotal")
        self.assertEqual(outcome.kind, "ambiguous")
        self.assertEqual([found.line for found in outcome.matches], [4, 6])
        self.assertIn("occurrence", outcome.detail)

    def test_overlapping_blocks_report_one_place_not_two(self):
        # "items" heads both the function and the loop nested inside it, so
        # the same hit is found under both spans.
        outcome = locate(JS, find="foo(item)", scope="items")
        self.assertEqual(outcome.kind, "located")
        self.assertEqual(outcome.matches[0].line, 4)

    def test_an_unknown_scope_says_so_rather_than_searching_the_file(self):
        outcome = locate(JS, find="foo(", scope="calculateDiscount")
        self.assertEqual(outcome.problem, "no_scope")

    def test_a_find_outside_the_scope_is_told_where_it_actually_is(self):
        outcome = locate(JS, find="foo(amount)", scope="function calculateTotal")
        self.assertEqual(outcome.problem, "not_in_scope")
        self.assertIn("line 10", outcome.detail)
        self.assertEqual([found.line for found in outcome.matches], [10])

    def test_a_find_in_no_block_at_all_says_to_read_again(self):
        outcome = locate(JS, find="bar(quux)", scope="function calculateTotal")
        self.assertEqual(outcome.problem, "not_found")


class CandidateCapTest(unittest.TestCase):
    def test_a_find_matching_everything_is_capped_and_says_to_narrow(self):
        text = "".join(f"    total += foo({index});\n" for index in range(40))
        outcome = locate(text, find="foo(")
        self.assertEqual(outcome.kind, "ambiguous")
        self.assertEqual(len(outcome.matches), edit_resolve._MAX_CANDIDATES)
        self.assertIn("40 places", outcome.detail)
        self.assertIn("narrow", outcome.detail)


class ApplyTest(unittest.TestCase):
    def test_one_match_changes_only_that_match(self):
        outcome = locate(JS, find="foo(item)")
        updated = apply(JS, outcome.matches, "bar(item)")
        self.assertEqual(updated, JS.replace("foo(item)", "bar(item)"))
        self.assertEqual(updated.count("foo("), 2)

    def test_every_match_changes_without_disturbing_the_offsets(self):
        outcome = locate(JS, find="foo(", occurrence="all")
        updated = apply(JS, outcome.matches, "bar(")
        self.assertEqual(updated, JS.replace("foo(", "bar("))

    def test_an_elastic_match_replaces_the_file_s_bytes_not_the_model_s(self):
        outcome = locate(JS, find="total  +=  foo(item);")
        updated = apply(JS, outcome.matches, "total += bar(item);")
        self.assertIn("        total += bar(item);\n", updated)


if __name__ == "__main__":
    unittest.main()
