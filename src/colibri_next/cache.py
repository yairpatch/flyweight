from __future__ import annotations

from collections import OrderedDict
from threading import RLock

from .expert import Expert, ExpertKey


class LayeredExpertCache:
    """Thread-safe per-layer LRU with an optional non-evictable hot tier."""

    def __init__(self, capacity_bytes_per_layer: int):
        if capacity_bytes_per_layer < 0:
            raise ValueError("cache capacity cannot be negative")
        self.capacity_bytes_per_layer = capacity_bytes_per_layer
        self._layers: dict[int, OrderedDict[int, Expert]] = {}
        self._pinned: dict[ExpertKey, Expert] = {}
        self._sizes: dict[int, int] = {}
        self._lock = RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: ExpertKey) -> Expert | None:
        with self._lock:
            pinned = self._pinned.get(key)
            if pinned is not None:
                self.hits += 1
                return pinned
            layer = self._layers.get(key.layer)
            if layer is None or key.expert not in layer:
                self.misses += 1
                return None
            expert = layer.pop(key.expert)
            layer[key.expert] = expert
            self.hits += 1
            return expert

    def peek(self, key: ExpertKey) -> Expert | None:
        with self._lock:
            pinned = self._pinned.get(key)
            if pinned is not None:
                return pinned
            return self._layers.get(key.layer, {}).get(key.expert)

    def put(self, expert: Expert, *, pinned: bool = False) -> None:
        key = expert.key
        with self._lock:
            if pinned:
                self._remove_lru_entry(key)
                self._pinned[key] = expert
                return
            if key in self._pinned or self.capacity_bytes_per_layer == 0:
                return

            layer = self._layers.setdefault(key.layer, OrderedDict())
            existing = layer.pop(key.expert, None)
            if existing is not None:
                self._sizes[key.layer] -= existing.byte_size
            layer[key.expert] = expert
            self._sizes[key.layer] = self._sizes.get(key.layer, 0) + expert.byte_size
            self._evict_layer(key.layer)

    def pin(self, expert: Expert) -> None:
        self.put(expert, pinned=True)

    def resident_keys(self) -> set[ExpertKey]:
        with self._lock:
            keys = set(self._pinned)
            for layer_id, experts in self._layers.items():
                keys.update(ExpertKey(layer_id, expert_id) for expert_id in experts)
            return keys

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            accesses = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "resident_experts": len(self.resident_keys()),
                "pinned_experts": len(self._pinned),
                "hit_rate": self.hits / accesses if accesses else 0.0,
            }

    def _remove_lru_entry(self, key: ExpertKey) -> None:
        layer = self._layers.get(key.layer)
        if layer is None:
            return
        existing = layer.pop(key.expert, None)
        if existing is not None:
            self._sizes[key.layer] -= existing.byte_size

    def _evict_layer(self, layer_id: int) -> None:
        layer = self._layers[layer_id]
        while self._sizes[layer_id] > self.capacity_bytes_per_layer and layer:
            _, evicted = layer.popitem(last=False)
            self._sizes[layer_id] -= evicted.byte_size
            self.evictions += 1
