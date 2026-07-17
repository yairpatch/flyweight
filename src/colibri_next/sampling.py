from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    temperature: float = 0.0
    top_k: int = 20
    top_p: float = 0.95
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


class LogitsSampler:
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.random = random.Random(config.seed)

    def sample_device(self, logits: object, accelerator: object) -> int:
        cp = accelerator.cp
        if logits.size == 0:
            raise ValueError("logits must not be empty")
        if self.config.temperature == 0:
            return int(cp.argmax(logits).item())
        candidate_count = (
            min(self.config.top_k, int(logits.size))
            if self.config.top_k
            else int(logits.size)
        )
        if candidate_count < int(logits.size):
            start = int(logits.size) - candidate_count
            candidate_ids = cp.argpartition(logits, start)[start:]
            candidate_logits = logits[candidate_ids]
            order = cp.argsort(candidate_logits)[::-1]
            candidate_ids = candidate_ids[order]
            candidate_logits = candidate_logits[order]
        else:
            candidate_ids = cp.argsort(logits)[::-1]
            candidate_logits = logits[candidate_ids]
        scaled = candidate_logits / cp.float32(self.config.temperature)
        probabilities = cp.exp(scaled - scaled[0])
        probabilities /= cp.sum(probabilities)
        if self.config.top_p < 1.0:
            cumulative = cp.cumsum(probabilities)
            keep = int(cp.searchsorted(cumulative, self.config.top_p).item()) + 1
            candidate_ids = candidate_ids[:keep]
            probabilities = probabilities[:keep]
            probabilities /= cp.sum(probabilities)
        threshold = self.random.random()
        selected = int(
            cp.searchsorted(cp.cumsum(probabilities), threshold).item()
        )
        selected = min(selected, int(candidate_ids.size) - 1)
        return int(candidate_ids[selected].item())

    def sample(self, logits: list[float]) -> int:
        if not logits:
            raise ValueError("logits must not be empty")
        candidate_count = (
            min(self.config.top_k, len(logits))
            if self.config.top_k
            else len(logits)
        )
        candidates = heapq.nlargest(
            candidate_count,
            enumerate(logits),
            key=lambda item: item[1],
        )
        if self.config.temperature == 0:
            return candidates[0][0]
        if self.config.temperature == 0:
            return candidates[0][0]
        maximum = candidates[0][1] / self.config.temperature
        probabilities = [
            math.exp(logit / self.config.temperature - maximum)
            for _, logit in candidates
        ]
        total = sum(probabilities)
        probabilities = [value / total for value in probabilities]
        if self.config.top_p < 1.0:
            cumulative = 0.0
            keep = 0
            for probability in probabilities:
                cumulative += probability
                keep += 1
                if cumulative >= self.config.top_p:
                    break
            candidates = candidates[:keep]
            probabilities = probabilities[:keep]
            total = sum(probabilities)
            probabilities = [value / total for value in probabilities]
        threshold = self.random.random()
        cumulative = 0.0
        for (token_id, _), probability in zip(candidates, probabilities):
            cumulative += probability
            if threshold <= cumulative:
                return token_id
        return candidates[-1][0]
