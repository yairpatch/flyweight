import unittest
from flyweight.v2 import V2KvCache


class V2KvCacheTests(unittest.TestCase):
    def test_native_cache_tracks_and_resets_position(self):
        with V2KvCache(0x1000, 0x2000, 16, 2, 8) as cache:
            self.assertEqual(cache.position, 0)
            cache.reset()
            self.assertEqual(cache.position, 0)

if __name__ == "__main__":
    unittest.main()
