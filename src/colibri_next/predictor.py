from __future__ import annotations

from collections import Counter, defaultdict

from .expert import ExpertKey


class TransitionPredictor:
    """Learns layer-to-layer expert co-occurrence for lightweight lookahead."""

    def __init__(self):
        self._transitions: dict[ExpertKey, Counter[int]] = defaultdict(Counter)

    def observe(self, current: list[ExpertKey], following: list[ExpertKey]) -> None:
        for source in current:
            counts = self._transitions[source]
            for destination in following:
                counts[destination.expert] += 1

    def predict(
        self, current: list[ExpertKey], next_layer: int, budget: int
    ) -> list[ExpertKey]:
        combined: Counter[int] = Counter()
        for source in current:
            combined.update(self._transitions.get(source, {}))
        return [
            ExpertKey(next_layer, expert_id)
            for expert_id, _ in combined.most_common(budget)
        ]
