import tempfile
import unittest
from pathlib import Path

from colibri_next.cache import LayeredExpertCache
from colibri_next.predictor import TransitionPredictor
from colibri_next.residency import ResidencyManager
from colibri_next.runtime import ToyMoERuntime
from colibri_next.storage import ExpertStore


class RuntimeTests(unittest.TestCase):
    def test_runtime_is_deterministic_and_learns_prefetches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ExpertStore.create_demo(
                Path(directory), layers=4, experts_per_layer=8, width=6
            )
            cache = LayeredExpertCache(store.expert_byte_size())
            predictor = TransitionPredictor()
            with ResidencyManager(store, cache, io_workers=2) as residency:
                runtime = ToyMoERuntime(residency, predictor, top_k=2, prefetch_budget=2)
                first = runtime.run_token(11)
                second = runtime.run_token(11)
                self.assertEqual(first, second)
                self.assertGreater(residency.prefetch_requests, 0)
                self.assertGreater(residency.prefetch_hits, 0)


if __name__ == "__main__":
    unittest.main()
