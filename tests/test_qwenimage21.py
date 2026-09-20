"""The Qwen-Image-2.1 diffusion path: loader, tower, and the request shaping."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from flyweight.images import ImageGenerator, parse_size
from flyweight.server import APIError
from flyweight.v2 import V2Diffusion, V2Error, V2Model, _library
from tests import qwenimage21_hf_fixture as fixture


def _gpu_available() -> bool:
    try:
        return _library().flyweight_v2_gpu_available() == 1
    except Exception:
        return False


class QwenImage21LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.snapshot = fixture.build(Path(cls.tmp.name) / "snapshot")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_the_text_encoder_opens_as_qwen3_vl_without_its_vision_tower(self) -> None:
        model = V2Model(self.snapshot / "text_encoder")
        try:
            config = model.config
            self.assertEqual(config["architecture"], "qwen3vl")
            # The decoder's dimensions come out of text_config.
            self.assertEqual(config["hidden_size"], fixture.ENC_HIDDEN)
            self.assertEqual(config["layer_count"], fixture.ENC_LAYERS)
            self.assertEqual(config["attention_kv_heads"], fixture.ENC_KV_HEADS)
            names = {tensor["name"] for tensor in model.tensors()}
            self.assertIn("token_embd.weight", names)
            self.assertIn("blk.0.attn_q_norm.weight", names)
            self.assertIn("blk.2.post_attention_norm.weight", names)
            self.assertIn("output.weight", names)
            # The vision tower is kept, under the mmproj names.
            self.assertFalse([name for name in names if "visual" in name])
            self.assertIn("v.patch_embd.weight", names)
            self.assertIn("v.blk.1.attn_qkv.weight", names)
            self.assertIn("v.deepstack_list.0.fc2.weight", names)
            self.assertIn("mm.2.weight", names)
            # The tokenizer came from ../processor/, and the chat template with it.
            self.assertTrue(model.tokenize("abc ab"))
            self.assertIn("<|im_start|>", model.chat_template)
            # Like plain Qwen3, a diffusion text encoder only.
            with self.assertRaisesRegex(V2Error, "diffusion text encoder"):
                model.native_runtime(device=0)
        finally:
            model.close()

    def test_the_transformer_stacks_qkv_and_derives_its_widths(self) -> None:
        model = V2Model(self.snapshot / "transformer")
        try:
            config = model.config
            self.assertEqual(config["architecture"], "qwenimage21-dit")
            self.assertEqual(config["hidden_size"], fixture.DIT_DIM)
            self.assertEqual(config["intermediate_size"], fixture.DIT_FFN)
            self.assertEqual(config["layer_count"], fixture.DIT_LAYERS)
            tensors = {tensor["name"]: tensor for tensor in model.tensors()}
            qkv = tensors["transformer_blocks.0.attn.qkv.weight"]
            self.assertEqual(tuple(qkv["shape"]), (fixture.DIT_DIM, fixture.DIT_DIM, 3))
            self.assertEqual(qkv["ggml_type"], 8, "the DiT packs as Q8_0 by default")
            self.assertNotIn("transformer_blocks.0.attn.to_q.weight", tensors)
            self.assertEqual(tensors["txt_in.text_norm.weight"]["ggml_type"], 0)
            self.assertIn("modulation.1.weight", tensors)
            # No per-block modulation weights exist to be found.
            self.assertFalse([name for name in tensors if "adaLN" in name])
        finally:
            model.close()

    def test_the_autoencoder_keeps_both_halves_for_a_single_frame(self) -> None:
        model = V2Model(self.snapshot / "vae")
        try:
            self.assertEqual(model.config["architecture"], "qwenimage21-vae")
            tensors = {tensor["name"]: tensor for tensor in model.tensors()}
            self.assertIn("encoder.conv_in.weight", tensors)
            self.assertIn("encoder.down_blocks.1.downsampler.resample.1.weight", tensors)
            self.assertIn("quant_conv.weight", tensors)
            # The temporal resample never runs on one frame, in either half.
            self.assertFalse([name for name in tensors if "time_conv" in name])
            self.assertIn("post_quant_conv.weight", tensors)
            self.assertIn("decoder.up_blocks.0.upsampler.resample.1.weight", tensors)
            self.assertNotIn("decoder.up_blocks.4.upsampler.resample.1.weight", tensors)
            # Level 3 narrows 16 -> 8 and carries the shortcut.
            conv = tensors["decoder.up_blocks.3.resnets.0.conv_shortcut.weight"]
            self.assertEqual(tuple(conv["shape"]), (1, 1, fixture.VAE_LEVELS[2], fixture.VAE_LEVELS[1]))
            self.assertEqual(tensors["decoder.norm_out.gamma"]["ggml_type"], 0)
        finally:
            model.close()

    def test_mixing_the_two_pipelines_components_is_refused(self) -> None:
        if not _gpu_available():
            self.skipTest("needs a CUDA device")
        from tests import zimage_hf_fixture
        with tempfile.TemporaryDirectory() as other:
            zimage = zimage_hf_fixture.build(Path(other) / "zimage")
            encoder = V2Model(zimage / "text_encoder")
            transformer = V2Model(self.snapshot / "transformer")
            vae = V2Model(self.snapshot / "vae")
            try:
                with self.assertRaisesRegex(V2Error, "Qwen3-VL"):
                    V2Diffusion(encoder, transformer, vae, max_width=64, max_height=64,
                                max_prompt_tokens=32, weights="device")
            finally:
                vae.close()
                transformer.close()
                encoder.close()

    @unittest.skipUnless(_gpu_available(), "needs a CUDA device")
    def test_the_tower_renders_rgba_on_a_32_pixel_grid(self) -> None:
        encoder = V2Model(self.snapshot / "text_encoder")
        transformer = V2Model(self.snapshot / "transformer")
        vae = V2Model(self.snapshot / "vae")
        tower = V2Diffusion(encoder, transformer, vae, max_width=96, max_height=64,
                            max_prompt_tokens=64, weights="device")
        try:
            info = tower.info
            self.assertEqual((info["latent_stride"], info["size_multiple"]), (16, 32))
            self.assertEqual((info["output_channels"], info["latent_channels"]), (4, fixture.CHANNELS))
            self.assertEqual((info["default_steps"], info["default_shift"]), (40, 0.0))
            tokens = encoder.tokenize("abc ab abc")
            caption = tower.encode_text(tokens)
            self.assertEqual(len(caption), len(tokens) * fixture.ENC_HIDDEN)
            pixels = tower.generate(tokens, 96, 64, steps=2, seed=3, caption_drop=2)
            self.assertEqual(len(pixels), 96 * 64 * 4)
            self.assertEqual(pixels, tower.generate(tokens, 96, 64, steps=2, seed=3, caption_drop=2))
            self.assertNotEqual(pixels, tower.generate(tokens, 96, 64, steps=2, seed=4, caption_drop=2))
            # The dropped prefix changes the conditioning, so the picture.
            self.assertNotEqual(pixels, tower.generate(tokens, 96, 64, steps=2, seed=3, caption_drop=1))
            with self.assertRaisesRegex(V2Error, "multiples of 32"):
                tower.generate(tokens, 80, 64, steps=1)
            with self.assertRaisesRegex(V2Error, "whole prompt"):
                tower.generate(tokens, 96, 64, steps=1, caption_drop=len(tokens))
            # A constant shift is the same formula at exp(mu) = shift.
            self.assertEqual(len(tower.generate(tokens, 64, 64, steps=2, seed=3, shift=3.0)), 64 * 64 * 4)
            # Editing: a 64x32 reference image is 2 vision tokens; its run sits
            # at token 1 of a prompt that holds those pads.
            pad = encoder.tokenize("<|image_pad|>")[0]
            prompt = [tokens[0], pad, pad] + list(tokens[1:])
            rgba = bytes(64 * 32 * 4)
            planes = tower.encode_vision(rgba, 64, 32)
            self.assertEqual(len(planes), 1 + len(fixture.VIT_DEEPSTACK))
            self.assertEqual(len(planes[0]), 2 * fixture.ENC_HIDDEN)
            latents = tower.encode_image(rgba, 64, 32)
            self.assertEqual(len(latents), fixture.CHANNELS * 2 * 4)
            rows = tower.encode_prompt(prompt, [(rgba, 64, 32, 1)])
            self.assertEqual(len(rows), len(prompt) * fixture.ENC_HIDDEN)
            edited = tower.edit(prompt, [(rgba, 64, 32, 1)], 96, 64, steps=2, seed=3, caption_drop=1)
            self.assertEqual(len(edited), 96 * 64 * 4)
            self.assertEqual(edited, tower.edit(prompt, [(rgba, 64, 32, 1)], 96, 64, steps=2, seed=3, caption_drop=1))
            with self.assertRaisesRegex(V2Error, "token run"):
                tower.edit(prompt, [(rgba, 64, 32, len(prompt) - 1)], 96, 64, steps=1)
        finally:
            tower.close()
        # Streamed from host memory, the same kernels see the same bytes.
        tower = V2Diffusion(encoder, transformer, vae, max_width=96, max_height=64,
                            max_prompt_tokens=64, weights="host")
        try:
            self.assertTrue(tower.info["host_weights"])
            self.assertEqual(pixels, tower.generate(tokens, 96, 64, steps=2, seed=3, caption_drop=2))
        finally:
            tower.close()
        for precision in ("balanced", "exact"):
            tower = V2Diffusion(encoder, transformer, vae, max_width=96, max_height=64,
                                max_prompt_tokens=64, weights="device", precision=precision)
            try:
                self.assertEqual(len(tower.generate(tokens, 96, 64, steps=2, seed=3, caption_drop=2)),
                                 96 * 64 * 4)
            finally:
                tower.close()
        # Past 32 latents a side the decoder tiles; the picture is still whole.
        tower = V2Diffusion(encoder, transformer, vae, max_width=640, max_height=544,
                            max_prompt_tokens=64, weights="device")
        try:
            latents = [0.1] * (fixture.CHANNELS * 34 * 40)
            self.assertEqual(len(tower.decode_latents(latents, 34, 40)), 640 * 544 * 4)
        finally:
            tower.close()
            vae.close()
            transformer.close()
            encoder.close()

    @unittest.skipUnless(_gpu_available(), "needs a CUDA device")
    def test_the_workspace_is_a_cap_that_grows_on_demand(self) -> None:
        encoder = V2Model(self.snapshot / "text_encoder")
        transformer = V2Model(self.snapshot / "transformer")
        vae = V2Model(self.snapshot / "vae")
        tokens = encoder.tokenize("abc ab")
        # Zero is the model's own maximum, and the workspace starts at 1024.
        tower = V2Diffusion(encoder, transformer, vae, max_prompt_tokens=32, weights="device")
        try:
            self.assertEqual((tower.max_width, tower.max_height), (2048, 2048))
            before = tower.info["device_bytes"]
            latents = [0.1] * (fixture.CHANNELS * 68 * 68)
            self.assertEqual(len(tower.decode_latents(latents, 68, 68)), 1088 * 1088 * 4)
            # The decoder tiles, so only the transformer's rows can outgrow the
            # arena; at this fixture's width that takes the full 2048.
            self.assertEqual(len(tower.generate(tokens, 2048, 2048, steps=1, caption_drop=1)),
                             2048 * 2048 * 4)
            self.assertGreater(tower.info["device_bytes"], before, "the arena grew for 2048")
            with self.assertRaisesRegex(V2Error, "image-max-size"):
                tower.generate(tokens, 2080, 32, steps=1)
        finally:
            tower.close()
        # Reserved: the whole cap is held from the start and nothing grows.
        tower = V2Diffusion(encoder, transformer, vae, max_width=1088, max_height=1088,
                            max_prompt_tokens=32, weights="device", reserve=True)
        try:
            before = tower.info["device_bytes"]
            tower.generate(tokens, 1088, 1088, steps=1, caption_drop=1)
            self.assertEqual(tower.info["device_bytes"], before)
        finally:
            tower.close()
            vae.close()
            transformer.close()
            encoder.close()

    @unittest.skipUnless(_gpu_available(), "needs a CUDA device")
    def test_the_generator_shapes_requests_for_the_model(self) -> None:
        generator = ImageGenerator(self.snapshot, max_width=64, max_height=64, weights="device")
        try:
            self.assertTrue(generator.qwenimage)
            self.assertEqual(generator.pixel_mode, "RGBA")
            self.assertEqual(generator.size_multiple, 32)
            self.assertEqual(generator.default_steps, 40)
            self.assertGreater(generator.caption_drop, 0)
            tokens = generator.tokenize("abc")
            # The system turn is in front of the prompt and is what gets dropped.
            self.assertEqual(tokens[:generator.caption_drop],
                             list(generator.encoder.tokenize(
                                 "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n")))
            described = generator.describe()
            self.assertEqual(described["default_steps"], 40)
            self.assertTrue(described["alpha"])
            result = generator.generate({"prompt": "abc ab", "size": "64x64", "steps": 1, "seed": 1})
            self.assertEqual(result["size"], "64x64")
            import base64
            import io
            from PIL import Image
            picture = Image.open(io.BytesIO(base64.b64decode(result["data"][0]["b64_json"])))
            self.assertEqual((picture.mode, picture.size), ("RGBA", (64, 64)))
            with self.assertRaises(APIError):
                generator.generate({"prompt": "abc", "size": "48x64", "steps": 1})
        finally:
            generator.close()


class RequestShapingTests(unittest.TestCase):
    def test_sizes_follow_the_models_grid(self) -> None:
        self.assertEqual(parse_size("256x384", (512, 512), (512, 512), 32), (256, 384))
        with self.assertRaises(APIError):
            parse_size("272x384", (512, 512), (512, 512), 32)
        # The default grid is Z-Image's.
        self.assertEqual(parse_size("272x384", (512, 512), (512, 512)), (272, 384))


if __name__ == "__main__":
    unittest.main()


class EditsRouteTests(unittest.TestCase):
    """`/v1/images/edits` in OpenAI's multipart shape lands on the image service as
    a generation request carrying `images`."""

    def setUp(self) -> None:
        import threading
        from flyweight.server import FlyweightHTTPServer, InferenceService, create_handler
        from tests.test_server import StubGenerator
        self.service = InferenceService("qwen-local", StubGenerator())
        self.server = FlyweightHTTPServer(("127.0.0.1", 0), create_handler(self.service))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _post(self, path: str, body: bytes, content_type: str) -> tuple[int, dict]:
        import http.client
        import json
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        try:
            connection.request("POST", path, body=body, headers={"Content-Type": content_type})
            response = connection.getresponse()
            return response.status, json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()

    def test_multipart_images_become_data_urls(self) -> None:
        seen = {}

        def images_generations(payload):
            seen.update(payload)
            return {"created": 1, "data": [{"b64_json": "AAAA", "seed": 7}]}

        self.service.images_generations = images_generations
        boundary = "b0undary"
        parts = [
            ("prompt", None, b"make the bicycle blue"),
            ("size", None, b"512x512"),
            ("n", None, b"1"),
            ("image", "a.png", b"\x89PNG\r\nfake"),
            ("image[]", "b.png", b"\x89PNG\r\nother"),
        ]
        body = b""
        for name, filename, data in parts:
            body += f"--{boundary}\r\n".encode()
            disposition = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
            body += f"Content-Disposition: {disposition}\r\n".encode()
            if filename:
                body += b"Content-Type: image/png\r\n"
            body += b"\r\n" + data + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        status, payload = self._post("/v1/images/edits", body, f"multipart/form-data; boundary={boundary}")
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["data"][0]["seed"], 7)
        self.assertEqual(seen["prompt"], "make the bicycle blue")
        self.assertEqual(seen["size"], "512x512")
        self.assertEqual(seen["n"], 1)
        self.assertEqual(len(seen["images"]), 2)
        self.assertTrue(all(url.startswith("data:image/png;base64,") for url in seen["images"]))

    def test_json_edits_without_images_are_refused(self) -> None:
        import json
        status, payload = self._post("/v1/images/edits", json.dumps({"prompt": "x"}).encode(), "application/json")
        self.assertEqual(status, 400)
        self.assertIn("image", payload["error"]["message"])
