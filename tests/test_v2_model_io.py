import struct
import tempfile
import unittest
from pathlib import Path

from colibri_next.v2 import V2Model


def string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


class V2ModelIOTests(unittest.TestCase):
    def test_qwen_embedding_and_lm_head_cpu_reference(self):
        metadata = b"".join((
            string("general.architecture") + struct.pack("<I", 8) + string("qwen3moe"),
            string("qwen3moe.embedding_length") + struct.pack("<II", 4, 3),
            string("qwen3moe.block_count") + struct.pack("<II", 4, 1),
            string("qwen3moe.attention.head_count") + struct.pack("<II", 4, 1),
        ))
        tensor_infos = b"".join((
            string("token_embd.weight") + struct.pack("<IQQIQ", 2, 2, 3, 0, 0),
            string("lm_head.weight") + struct.pack("<IQQIQ", 2, 2, 3, 0, 24),
        ))
        header = b"GGUF" + struct.pack("<IQQ", 3, 2, 4)
        body = header + metadata + tensor_infos
        body += b"\0" * ((32 - len(body) % 32) % 32)
        payload = struct.pack("<6f", 1, 2, 3, 4, 5, 6)
        payload += struct.pack("<6f", 1, 0, 0, 0, 1, 0)
        handle = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        handle.write(body + payload)
        handle.close()
        path = Path(handle.name)
        try:
            with V2Model(path) as model:
                model.validate_qwen()
                self.assertEqual(model.qwen_embedding(1, 3), [4.0, 5.0, 6.0])
                self.assertEqual(model.qwen_lm_head([2.0, 3.0, 4.0], 2), [2.0, 3.0])
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
