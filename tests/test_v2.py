import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from colibri_next.v2 import V2Model, V2Error


def gguf_string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


class V2RuntimeTests(unittest.TestCase):
    def make_model(self) -> tuple[Path, bytes]:
        metadata = b"".join((
            gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string("qwen3moe"),
            gguf_string("general.name") + struct.pack("<I", 8) + gguf_string("test"),
            gguf_string("qwen3moe.embedding_length") + struct.pack("<II", 4, 16),
            gguf_string("qwen3moe.block_count") + struct.pack("<II", 4, 1),
            gguf_string("qwen3moe.attention.head_count") + struct.pack("<II", 4, 2),
            gguf_string("qwen3moe.rope.dimension_count") + struct.pack("<II", 4, 64),
            gguf_string("qwen3moe.full_attention_interval") + struct.pack("<II", 4, 4),
            gguf_string("qwen3moe.attention.layer_norm_rms_epsilon") + struct.pack("<If", 6, 1e-6),
            gguf_string("qwen3moe.rope.freq_base") + struct.pack("<If", 6, 10_000_000.0),
        ))
        tensor = gguf_string("token_embd.weight") + struct.pack("<IQQIQ", 2, 2, 2, 0, 0)
        header = b"GGUF" + struct.pack("<IQQ", 3, 1, 9)
        body = header + metadata + tensor
        body += b"\0" * ((32 - len(body) % 32) % 32)
        payload = b"\x01\x02\x03\x04"
        handle = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        handle.write(body + payload)
        handle.close()
        return Path(handle.name), payload

    def test_reads_metadata_offsets_and_session_stats(self):
        path, payload = self.make_model()
        try:
            with V2Model(path) as model:
                self.assertEqual(model.info["architecture"], "qwen3moe")
                self.assertEqual(model.info["format"], "gguf")
                self.assertEqual(model.config["rotary_dimension"], 64)
                self.assertEqual(model.config["full_attention_interval"], 4)
                self.assertAlmostEqual(model.config["rms_norm_epsilon"], 1e-6)
                self.assertEqual(model.config["rope_freq_base"], 10_000_000.0)
                model.validate_qwen()
                self.assertEqual(model.qwen_tensor("token_embeddings")["name"], "token_embd.weight")
                tensor = model.tensor("token_embd.weight")
                self.assertEqual(tensor["size"], len(payload))
                self.assertEqual(
                    model.read_tensor_slice("token_embd.weight", 1, 2),
                    payload[1:3],
                )
                self.assertEqual(
                    model.view_tensor_slice("token_embd.weight", 1, 2),
                    payload[1:3],
                )
                with model.session(8) as session:
                    session.prompt([1, 2])
                    self.assertIsInstance(session.decode(), int)
                    self.assertEqual(session.stats["prompt_tokens"], 2)
        finally:
            path.unlink()

    def test_context_limit_and_cancellation_are_reported(self):
        path, _ = self.make_model()
        try:
            with V2Model(path) as model, model.session(1) as session:
                with self.assertRaises(V2Error):
                    session.prompt([1, 2])
                session.cancel()
                with self.assertRaises(V2Error):
                    session.decode()
        finally:
            path.unlink()

    def test_tokenize_recognizes_non_pipe_qwen_control_tokens(self):
        model = object.__new__(V2Model)

        def token_id(_model, text):
            if text == "<think>":
                return 248068
            raise V2Error("not a control token")

        with patch.object(V2Model, "token_id", token_id), patch.object(
            V2Model, "_tokenize_plain", lambda _model, text, _capacity: [len(text)]
        ):
            self.assertEqual(
                model.tokenize("a<think>b<div>c"),
                [1, 248068, 1, 5, 1],
            )
