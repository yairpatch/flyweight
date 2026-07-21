import unittest
import struct
import tempfile
from pathlib import Path

from colibri_next.v2 import V2KvCache, V2Model, V2Error


def _string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _model_file():
    body = b"GGUF" + struct.pack("<IQQ", 3, 0, 0) + b"\0" * 8
    handle = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
    handle.write(body)
    handle.close()
    return Path(handle.name)


class V2KvCacheTests(unittest.TestCase):
    def test_native_cache_tracks_and_resets_position(self):
        with V2KvCache(0x1000, 0x2000, 16, 2, 8) as cache:
            self.assertEqual(cache.position, 0)
            cache.reset()
            self.assertEqual(cache.position, 0)

    def test_session_accepts_matching_cache_and_detaches(self):
        path = _model_file()
        try:
            with V2Model(path) as model, model.session(8) as session, V2KvCache(0x1000, 0x2000, 16, 2, 8) as cache:
                session.prompt([1, 2])
                with self.assertRaises(V2Error):
                    session.attach_kv_cache(cache)
                cache.reset()
                session.detach_kv_cache()
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
