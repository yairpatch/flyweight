from __future__ import annotations

from math import cos, sin

from .expert import ExpertKey
from .predictor import TransitionPredictor
from .residency import ResidencyManager


class ToyMoERuntime:
    """Deterministic sparse runtime that exercises residency, not LLM quality."""

    def __init__(
        self,
        residency: ResidencyManager,
        predictor: TransitionPredictor,
        *,
        top_k: int = 2,
        prefetch_budget: int = 2,
    ):
        self.residency = residency
        self.predictor = predictor
        self.top_k = top_k
        self.prefetch_budget = prefetch_budget

    def run_token(self, token_id: int) -> list[float]:
        width = self.residency.store.width
        hidden = [sin(token_id * 0.17 + index * 0.31) for index in range(width)]
        previous_route: list[ExpertKey] | None = None

        for layer in range(self.residency.store.layers):
            route = self._route(hidden, layer)
            if previous_route is not None:
                self.predictor.observe(previous_route, route)

            if layer + 1 < self.residency.store.layers:
                predicted = self.predictor.predict(
                    route, layer + 1, self.prefetch_budget
                )
                self.residency.prefetch(predicted)

            weighted_outputs: list[list[float]] = []
            for rank, key in enumerate(route):
                expert_output = self.residency.get(key).forward(hidden)
                gate_weight = 1.0 / (rank + 1)
                weighted_outputs.append(
                    [gate_weight * value for value in expert_output]
                )
            normalizer = sum(1.0 / (rank + 1) for rank in range(len(route)))
            hidden = [sum(values) / normalizer for values in zip(*weighted_outputs)]
            previous_route = route
        return hidden

    def run(self, tokens: list[int]) -> list[list[float]]:
        return [self.run_token(token_id) for token_id in tokens]

    def _route(self, hidden: list[float], layer: int) -> list[ExpertKey]:
        expert_count = self.residency.store.experts_per_layer
        scores = []
        summary = sum(hidden[index] * (index + 1) for index in range(len(hidden)))
        for expert_id in range(expert_count):
            score = sin(summary * 0.13 + expert_id * 1.91 + layer * 0.77)
            score += cos(summary * 0.07 - expert_id * 0.37)
            scores.append((score, expert_id))
        scores.sort(reverse=True)
        return [ExpertKey(layer, expert_id) for _, expert_id in scores[: self.top_k]]
