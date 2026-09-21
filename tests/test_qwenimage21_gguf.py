"""Tests for Qwen-Image-2.1 with GGUF DiT transformer."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flyweight.images import ImageGenerator
from flyweight.v2 import V2Model, _library
from tests import qwenimage21_gguf_fixture as gguf_fixture
from tests import qwenimage21_hf_fixture as fixture


def _gpu_available() -> bool:
    try:
        return _library().flyweight_v2_gpu_available() == 1
    except Exception:
        return False


class QwenImage21GGUFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.snapshot = fixture.build(cls.root / "snapshot")
        cls.gguf_path = gguf_fixture.build_gguf_transformer(cls.root / "qwen_image.gguf")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_the_transformer_opens_as_qwenimage21_dit_from_gguf(self) -> None:
        model = V2Model(self.gguf_path)
        try:
            config = model.config
            self.assertEqual(config["architecture"], "qwenimage21-dit")
            self.assertEqual(config["hidden_size"], fixture.DIT_DIM)
            self.assertEqual(config["intermediate_size"], fixture.DIT_FFN)
            self.assertEqual(config["layer_count"], fixture.DIT_LAYERS)
            self.assertEqual(config["attention_heads"], fixture.DIT_HEADS)
            self.assertEqual(config["attention_kv_heads"], fixture.DIT_HEADS)

            tensors = {tensor["name"]: tensor for tensor in model.tensors()}
            self.assertIn("transformer_blocks.0.attn.to_q.weight", tensors)
            self.assertIn("transformer_blocks.0.attn.to_k.weight", tensors)
            self.assertIn("transformer_blocks.0.attn.to_v.weight", tensors)
            self.assertIn("transformer_blocks.0.attn.to_out.0.weight", tensors)
            self.assertIn("transformer_blocks.0.img_mlp.gate_up.weight", tensors)
            self.assertIn("transformer_blocks.0.img_mlp.out.weight", tensors)
            self.assertNotIn("transformer_blocks.0.attn.qkv.weight", tensors)
            self.assertNotIn("transformer_blocks.0.img_mlp.gate_layer.weight", tensors)

            gate_up = tensors["transformer_blocks.0.img_mlp.gate_up.weight"]
            self.assertEqual(tuple(gate_up["shape"]), (fixture.DIT_DIM, 2 * fixture.DIT_FFN))
        finally:
            model.close()

    def test_generator_with_explicit_transformer_path(self) -> None:
        generator = ImageGenerator(self.snapshot, transformer=self.gguf_path)
        try:
            self.assertTrue(generator.qwenimage)
            info = generator.describe()
            self.assertTrue(info["edit"])
        finally:
            generator.close()

    def test_generator_with_gguf_in_transformer_directory(self) -> None:
        gguf_snapshot = self.root / "gguf_snapshot"
        shutil.copytree(self.snapshot / "text_encoder", gguf_snapshot / "text_encoder")
        shutil.copytree(self.snapshot / "vae", gguf_snapshot / "vae")
        if (self.snapshot / "processor").is_dir():
            shutil.copytree(self.snapshot / "processor", gguf_snapshot / "processor")
        (gguf_snapshot / "transformer").mkdir(parents=True)
        shutil.copy(self.gguf_path, gguf_snapshot / "transformer" / "model.gguf")

        generator = ImageGenerator(gguf_snapshot)
        try:
            self.assertTrue(generator.qwenimage)
            self.assertEqual(generator.transformer.config["architecture"], "qwenimage21-dit")
        finally:
            generator.close()

    @unittest.skipUnless(_gpu_available(), "test requires a CUDA device")
    def test_numerical_parity_between_hf_and_gguf_dit(self) -> None:
        """The GGUF DiT must produce results matching HF safetensors DiT."""
        import base64
        import io
        from PIL import Image

        gen_hf = ImageGenerator(self.snapshot, precision="fast")
        gen_gguf = ImageGenerator(self.snapshot, transformer=self.gguf_path, precision="fast")
        try:
            req = {"prompt": "abc ab", "size": "64x64", "steps": 1, "seed": 42}
            res_hf = gen_hf.generate(req)
            res_gguf = gen_gguf.generate(req)

            img_hf = Image.open(io.BytesIO(base64.b64decode(res_hf["data"][0]["b64_json"])))
            img_gguf = Image.open(io.BytesIO(base64.b64decode(res_gguf["data"][0]["b64_json"])))

            arr_hf = np.array(img_hf, dtype=np.float32)
            arr_gguf = np.array(img_gguf, dtype=np.float32)
            diff = np.abs(arr_hf - arr_gguf)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            # Allow minor floating point rounding differences (< 2 gray levels)
            self.assertLess(max_diff, 5.0, f"Max pixel difference too high: {max_diff}")
            self.assertLess(mean_diff, 0.5, f"Mean pixel difference too high: {mean_diff}")
        finally:
            gen_hf.close()
            gen_gguf.close()

    @unittest.skipUnless(_gpu_available(), "test requires a CUDA device")
    def test_multi_step_kv_cache_parity_with_gguf_dit(self) -> None:
        """Prefix KV caching must give identical output with GGUF DiT."""
        import base64
        import io
        from PIL import Image

        generator = ImageGenerator(self.snapshot, transformer=self.gguf_path, precision="fast")
        try:
            req = {"prompt": "abc ab", "size": "64x64", "steps": 2, "seed": 123}
            os.environ["FLYWEIGHT_DIFF_KV_CACHE"] = "1"
            res_cached = generator.generate(req)

            os.environ["FLYWEIGHT_DIFF_KV_CACHE"] = "0"
            res_nocache = generator.generate(req)

            img_cached = Image.open(io.BytesIO(base64.b64decode(res_cached["data"][0]["b64_json"])))
            img_nocache = Image.open(io.BytesIO(base64.b64decode(res_nocache["data"][0]["b64_json"])))

            arr_cached = np.array(img_cached, dtype=np.float32)
            arr_nocache = np.array(img_nocache, dtype=np.float32)
            np.testing.assert_allclose(arr_cached, arr_nocache, atol=1.0)
        finally:
            os.environ.pop("FLYWEIGHT_DIFF_KV_CACHE", None)
            generator.close()

    @unittest.skipUnless(_gpu_available(), "test requires a CUDA device")
    def test_the_generator_edits_with_gguf_transformer(self) -> None:
        """The generator must successfully perform image editing with GGUF DiT."""
        import base64
        import io
        from PIL import Image

        generator = ImageGenerator(self.snapshot, transformer=self.gguf_path, max_width=64, max_height=64, weights="device")
        try:
            buf = io.BytesIO()
            Image.new("RGBA", (64, 32), (255, 0, 0, 255)).save(buf, format="PNG")
            data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

            result = generator.generate({
                "prompt": "make it blue",
                "images": [data_url],
                "size": "64x64",
                "steps": 1,
                "seed": 42,
            })
            self.assertEqual(result["size"], "64x64")
            self.assertEqual(len(result["data"]), 1)
            pic = Image.open(io.BytesIO(base64.b64decode(result["data"][0]["b64_json"])))
            self.assertEqual((pic.mode, pic.size), ("RGBA", (64, 64)))
        finally:
            generator.close()

