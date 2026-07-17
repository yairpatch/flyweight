import tempfile
import unittest
from pathlib import Path

from colibri_next.cache import LayeredExpertCache
from colibri_next.expert import ExpertKey
from colibri_next.storage import ExpertStore


class LayeredExpertCacheTests(unittest.TestCase):
    def test_lru_eviction_is_isolated_per_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ExpertStore.create_demo(
                Path(directory), layers=2, experts_per_layer=3, width=4
            )
            cache = LayeredExpertCache(store.expert_byte_size() * 2)
            first = store.load(ExpertKey(0, 0))
            second = store.load(ExpertKey(0, 1))
            third = store.load(ExpertKey(0, 2))
            other_layer = store.load(ExpertKey(1, 0))

            cache.put(first)
            cache.put(second)
            self.assertIs(cache.get(first.key), first)
            cache.put(third)
            cache.put(other_layer)

            self.assertIsNone(cache.peek(second.key))
            self.assertIs(cache.peek(first.key), first)
            self.assertIs(cache.peek(other_layer.key), other_layer)

    def test_pinned_expert_survives_eviction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ExpertStore.create_demo(
                Path(directory), layers=1, experts_per_layer=3, width=4
            )
            cache = LayeredExpertCache(store.expert_byte_size())
            pinned = store.load(ExpertKey(0, 0))
            cache.pin(pinned)
            cache.put(store.load(ExpertKey(0, 1)))
            cache.put(store.load(ExpertKey(0, 2)))
            self.assertIs(cache.peek(pinned.key), pinned)


if __name__ == "__main__":
    unittest.main()
