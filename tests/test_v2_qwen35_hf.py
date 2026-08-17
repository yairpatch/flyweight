"""Loading a Qwen3.5 HuggingFace checkpoint.

The GGUFs of this family and the HF release describe the same 866 language
tensors under different names, so what these tests pin is the translation
between them -- plus the three things about this checkpoint that a
single-architecture fixture would not catch: the decoder config nests under
``text_config``, the vision tower must be dropped rather than mapped, and the
end-of-turn token lives in generation_config.json where config.json carries
end-of-document.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from colibri_next.v2 import V2Model
from tests import qwen35_hf_fixture as fixture


class Qwen35HuggingFaceLoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The path is gated while the numeric transforms are unimplemented; see
        # require_qwen3_5_opt_in. These tests cover the name and config
        # translation, which is the part that is finished.
        os.environ["COLIBRI_HF_QWEN35_INCOMPLETE"] = "1"
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = str(fixture.build(Path(cls._directory.name) / "qwen35"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()
        os.environ.pop("COLIBRI_HF_QWEN35_INCOMPLETE", None)

    def names(self) -> set[str]:
        with V2Model(self.path) as model:
            return {str(tensor["name"]) for tensor in model.tensors()}

    def test_it_refuses_without_the_opt_in(self) -> None:
        os.environ.pop("COLIBRI_HF_QWEN35_INCOMPLETE", None)
        try:
            with self.assertRaises(Exception) as caught:
                V2Model(self.path).close()
            self.assertIn("incomplete", str(caught.exception))
        finally:
            os.environ["COLIBRI_HF_QWEN35_INCOMPLETE"] = "1"

    def test_it_opens_the_directory_as_safetensors(self) -> None:
        with V2Model(self.path) as model:
            info = model.info
        self.assertEqual(info["format"], "safetensors")
        self.assertEqual(info["architecture"], "qwen35")

    def test_config_comes_from_the_nested_text_config(self) -> None:
        with V2Model(self.path) as model:
            config = model.config
        self.assertEqual(config["architecture"], "qwen35")
        self.assertEqual(config["hidden_size"], fixture.HIDDEN)
        self.assertEqual(config["layer_count"], fixture.LAYERS)
        self.assertEqual(config["attention_heads"], fixture.HEADS)
        self.assertEqual(config["attention_kv_heads"], fixture.KV_HEADS)
        self.assertEqual(config["intermediate_size"], fixture.FFN)
        self.assertEqual(config["vocabulary_size"], fixture.VOCAB)
        self.assertEqual(config["full_attention_interval"], fixture.FULL_INTERVAL)

    def test_rope_is_read_from_the_nested_rope_parameters(self) -> None:
        with V2Model(self.path) as model:
            config = model.config
        self.assertAlmostEqual(config["rope_freq_base"], 10000000.0, places=0)
        # partial_rotary_factor 0.25 of a 64-wide head.
        self.assertEqual(config["rotary_dimension"], fixture.HEAD_DIM // 4)

    def test_generation_config_supplies_the_end_of_turn_token(self) -> None:
        # text_config says 5, generation_config lists 7 first. Taking the
        # config.json value would leave generation unable to stop.
        with V2Model(self.path) as model:
            self.assertEqual(model.config["eos_token_id"], 7)

    def test_every_language_tensor_is_translated(self) -> None:
        self.assertEqual(self.names(), fixture.language_tensor_names())

    def test_the_vision_tower_is_dropped(self) -> None:
        names = self.names()
        self.assertFalse([name for name in names if "visual" in name])
        # Dropped, not merely unmapped: an unrecognised tensor is a load error,
        # so opening the model at all is the other half of this assertion.
        self.assertIn("token_embd.weight", names)

    def test_the_mtp_block_lands_past_the_decoder_stack(self) -> None:
        names = self.names()
        block = f"blk.{fixture.LAYERS}."
        self.assertIn(block + "nextn.eh_proj.weight", names)
        # Its transformer half takes ordinary names on the same index, which is
        # what makes the runtime treat the block as a layer.
        self.assertIn(block + "attn_q.weight", names)
        self.assertIn(block + "ffn_gate.weight", names)

    def test_delta_and_attention_layers_get_different_tensors(self) -> None:
        names = self.names()
        # Layer 3 closes the first group of four, so it is full attention;
        # layer 0 is gated-delta.
        self.assertIn("blk.3.attn_q.weight", names)
        self.assertNotIn("blk.3.attn_qkv.weight", names)
        self.assertIn("blk.0.attn_qkv.weight", names)
        self.assertNotIn("blk.0.attn_q.weight", names)

    def test_the_conv1d_keeps_the_kernel_as_its_leading_extent(self) -> None:
        with V2Model(self.path) as model:
            conv = next(t for t in model.tensors()
                        if str(t["name"]) == "blk.0.ssm_conv1d.weight")
            shape = list(conv["shape"])
        # [channels, 1, kernel] on disk reverses to [kernel, 1, channels]. The
        # kernel width has to lead, because that is where the layer plan reads
        # it from.
        self.assertEqual(shape[0], fixture.CONV_KERNEL)

    def test_quantization_keeps_the_norms_in_full_precision(self) -> None:
        with V2Model(self.path) as model:
            types = {str(t["name"]): int(t["ggml_type"]) for t in model.tensors()}
        self.assertEqual(types["blk.0.attn_norm.weight"], 0)  # F32
        self.assertEqual(types["output_norm.weight"], 0)
        # The 2-D bulk is quantized; which target is policy, but it must not
        # have been left at f32.
        self.assertNotEqual(types["blk.0.ffn_gate.weight"], 0)


if __name__ == "__main__":
    unittest.main()
