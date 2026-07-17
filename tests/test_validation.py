from __future__ import annotations

import unittest

from colibri_next.validation import (
    compare_logit_vectors,
    validate_against_reference,
)


class _State:
    pass


class _Result:
    def __init__(self, logits: list[float]):
        self.logits = logits


class _Model:
    def new_state(self) -> _State:
        return _State()

    def prefill(self, token_ids: list[int], state: _State) -> _Result:
        return _Result(_logits(token_ids))

    def forward_token(self, token_id: int, state: _State) -> _Result:
        return _Result(_logits([token_id]))


class _Reference:
    def logits(self, token_ids: list[int]) -> list[float]:
        return _logits([token_ids[-1]])


def _logits(token_ids: list[int]) -> list[float]:
    token = token_ids[-1]
    return [float(token), float(token + 2), float(token - 1)]


class ValidationTests(unittest.TestCase):
    def test_compare_logit_vectors_reports_parity_metrics(self) -> None:
        result = compare_logit_vectors(
            [1.0, 3.0, 2.0],
            [1.5, 3.0, 1.0],
            position=2,
            input_token=7,
            top_k=2,
        )

        self.assertEqual(result.colibri_greedy_token, 1)
        self.assertEqual(result.reference_greedy_token, 1)
        self.assertTrue(result.greedy_match)
        self.assertEqual(result.max_absolute_error, 1.0)
        self.assertEqual(result.mean_absolute_error, 0.5)
        self.assertEqual(result.top_k_overlap, 0.5)

    def test_teacher_forced_validation_uses_reference_tokens(self) -> None:
        report = validate_against_reference(
            _Model(), _Reference(), [4, 5], generate_tokens=2, top_k=2
        )

        self.assertEqual(report.generated_tokens, (1, 1))
        self.assertEqual(len(report.comparisons), 3)
        self.assertTrue(report.all_greedy_tokens_match)
        self.assertEqual(report.to_dict()["greedy_matches"], 3)

    def test_validation_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            validate_against_reference(_Model(), _Reference(), [])
        with self.assertRaisesRegex(ValueError, "vocabulary sizes differ"):
            compare_logit_vectors(
                [1.0], [1.0, 2.0], position=0, input_token=0
            )


if __name__ == "__main__":
    unittest.main()
