import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

from colibri_next.v2 import V2Model, V2Error


def gguf_string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


class V2RuntimeTests(unittest.TestCase):
    def make_model(
        self,
        sliding_pattern: tuple[bool, ...] | None = None,
        architecture: str = "qwen3moe",
        kv_heads: tuple[int, ...] | None = None,
    ) -> tuple[Path, bytes]:
        prefix = architecture
        metadata_items = [
            gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string(architecture),
            gguf_string("general.name") + struct.pack("<I", 8) + gguf_string("test"),
            gguf_string(f"{prefix}.embedding_length") + struct.pack("<II", 4, 16),
            gguf_string(f"{prefix}.block_count") + struct.pack("<II", 4, len(sliding_pattern) if sliding_pattern else 1),
            gguf_string(f"{prefix}.attention.head_count") + struct.pack("<II", 4, 2),
            gguf_string(f"{prefix}.rope.dimension_count") + struct.pack("<II", 4, 64),
            gguf_string(f"{prefix}.full_attention_interval") + struct.pack("<II", 4, 4),
            gguf_string(f"{prefix}.attention.layer_norm_rms_epsilon") + struct.pack("<If", 6, 1e-6),
            gguf_string(f"{prefix}.rope.freq_base") + struct.pack("<If", 6, 10_000_000.0),
        ]
        if kv_heads is not None:
            metadata_items.append(
                gguf_string(f"{prefix}.attention.head_count_kv")
                + struct.pack("<IIQ", 9, 5, len(kv_heads))
                + struct.pack(f"<{len(kv_heads)}i", *kv_heads)
            )
        if sliding_pattern is not None:
            metadata_items.extend((
                gguf_string(f"{prefix}.attention.sliding_window") + struct.pack("<II", 4, 128),
                gguf_string(f"{prefix}.attention.sliding_window_pattern")
                + struct.pack("<IIQ", 9, 7, len(sliding_pattern))
                + bytes(sliding_pattern),
            ))
        metadata = b"".join(metadata_items)
        tensor = gguf_string("token_embd.weight") + struct.pack("<IQQIQ", 2, 2, 2, 0, 0)
        header = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata_items))
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

    def test_parses_generic_interleaved_sliding_window_pattern(self):
        path, _ = self.make_model((True, True, False, True, False, False))
        try:
            with V2Model(path) as model:
                config = model.config
                self.assertEqual(config["sliding_window"], 128)
                self.assertEqual(config["sliding_window_pattern_length"], 6)
                self.assertEqual(
                    config["attention_windows"], (128, 128, 0, 128, 0, 0)
                )
                self.assertEqual(
                    config["sliding_window_pattern"],
                    (True, True, False, True, False, False),
                )
        finally:
            path.unlink()

    def test_parses_gemma4_signed_per_layer_kv_head_array(self):
        pattern = (True, True, False)
        path, _ = self.make_model(pattern, architecture="gemma4", kv_heads=(8, 8, 2))
        try:
            with V2Model(path) as model:
                self.assertEqual(model.info["architecture"], "gemma4")
                self.assertEqual(model.config["attention_windows"], (128, 128, 0))
        finally:
            path.unlink()

    def test_generic_runtime_selects_cpu_experts_for_gemma4(self):
        model = object.__new__(V2Model)
        with patch.object(
            V2Model, "info", new_callable=PropertyMock,
            return_value={"architecture": "gemma4"},
        ), patch.object(V2Model, "native_qwen_runtime", return_value="runtime") as create:
            self.assertEqual(model.native_runtime(context_limit=32), "runtime")
            create.assert_called_once_with(context_limit=32, moe_device="cpu")

    def test_gemma4_decode_converts_word_boundary_markers_to_spaces(self):
        model = object.__new__(V2Model)
        model._architecture = "gemma4"
        pieces = {1: "Experts", 2: "▁(MoE)", 3: "▁works", 4: "."}
        with patch.object(V2Model, "token_text", lambda _model, token: pieces[token]):
            self.assertEqual(
                model.decode_tokens([1, 2, 3, 4]),
                "Experts (MoE) works.",
            )

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
