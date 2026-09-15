"""The Z-Image diffusion path: loader, tower, and the images endpoint."""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from flyweight.images import ImageGenerator, parse_size, snapshot_model_name
from flyweight.server import APIError, FlyweightHTTPServer, InferenceService, create_handler
from flyweight.v2 import V2Diffusion, V2Error, V2Model, _library
from tests import zimage_hf_fixture as fixture
from tests.test_server import StubGenerator


def _gpu_available() -> bool:
    try:
        return _library().flyweight_v2_gpu_available() == 1
    except Exception:
        return False


class ZImageLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.snapshot = fixture.build(Path(cls.tmp.name) / "snapshot")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_the_text_encoder_opens_as_plain_qwen3_with_the_sibling_tokenizer(self) -> None:
        model = V2Model(self.snapshot / "text_encoder")
        try:
            self.assertEqual(model.config["architecture"], "qwen3")
            names = {tensor["name"] for tensor in model.tensors()}
            self.assertIn("token_embd.weight", names)
            self.assertIn("blk.0.attn_q_norm.weight", names)
            self.assertIn("blk.2.post_attention_norm.weight", names)
            # tie_word_embeddings: no head of its own.
            self.assertNotIn("output.weight", names)
            # The tokenizer came from ../tokenizer/, and the chat template with it.
            self.assertTrue(model.tokenize("abc ab"))
            self.assertIn("<|im_start|>", model.chat_template)
            # The Qwen decode runtime has no plan for a gate-less Qwen3.
            with self.assertRaisesRegex(V2Error, "diffusion text encoder"):
                model.native_runtime(device=0)
        finally:
            model.close()

    def test_the_transformer_stacks_qkv_and_keeps_pad_tokens_f32(self) -> None:
        model = V2Model(self.snapshot / "transformer")
        try:
            config = model.config
            self.assertEqual(config["architecture"], "zimage-dit")
            self.assertEqual(config["hidden_size"], fixture.DIT_DIM)
            self.assertEqual(config["layer_count"], fixture.DIT_LAYERS)
            tensors = {tensor["name"]: tensor for tensor in model.tensors()}
            qkv = tensors["layers.0.attention.qkv.weight"]
            # GGUF order [in, out, piece]: q, k and v rows stacked into one matrix.
            self.assertEqual(tuple(qkv["shape"]), (fixture.DIT_DIM, fixture.DIT_DIM, 3))
            self.assertEqual(qkv["ggml_type"], 8, "the DiT packs as Q8_0 by default")
            self.assertNotIn("layers.0.attention.to_q.weight", tensors)
            # A [1][dim] parameter is a vector, not a matrix to quantize.
            self.assertEqual(tensors["x_pad_token"]["ggml_type"], 0)
            self.assertEqual(tensors["layers.0.attention_norm1.weight"]["ggml_type"], 0)
            self.assertIn("context_refiner.0.feed_forward.w1.weight", tensors)
            self.assertNotIn("context_refiner.0.adaLN_modulation.0.weight", tensors)
        finally:
            model.close()

    def test_the_autoencoder_keeps_only_its_decoder_in_f32(self) -> None:
        model = V2Model(self.snapshot / "vae")
        try:
            self.assertEqual(model.config["architecture"], "autoencoder-kl")
            tensors = {tensor["name"]: tensor for tensor in model.tensors()}
            self.assertNotIn("encoder.conv_in.weight", tensors)
            self.assertNotIn("quant_conv.weight", tensors)
            conv = tensors["decoder.up_blocks.2.resnets.0.conv_shortcut.weight"]
            self.assertEqual(conv["ggml_type"], 0)
            # Level 2 narrows from the second-widest level to the second-narrowest.
            self.assertEqual(tuple(conv["shape"]), (1, 1, fixture.VAE_CHANNELS[2], fixture.VAE_CHANNELS[1]))
            self.assertEqual(tensors["decoder.mid_block.attentions.0.to_q.weight"]["ggml_type"], 0)
        finally:
            model.close()

    def test_the_generator_refuses_a_directory_that_is_not_a_snapshot(self) -> None:
        with self.assertRaises(FileNotFoundError):
            ImageGenerator(self.snapshot / "vae")

    @unittest.skipUnless(_gpu_available(), "needs a CUDA device")
    def test_the_tower_renders_an_image_of_the_requested_size(self) -> None:
        encoder = V2Model(self.snapshot / "text_encoder")
        transformer = V2Model(self.snapshot / "transformer")
        vae = V2Model(self.snapshot / "vae")
        tower = V2Diffusion(encoder, transformer, vae, max_width=64, max_height=48,
                            max_prompt_tokens=64, weights="device")
        try:
            self.assertFalse(tower.info["host_weights"])
            tokens = encoder.tokenize("abc ab")
            caption = tower.encode_text(tokens)
            self.assertEqual(len(caption), len(tokens) * fixture.ENC_HIDDEN)
            rgb = tower.generate(tokens, 64, 48, steps=2, seed=3)
            self.assertEqual(len(rgb), 64 * 48 * 3)
            # Deterministic for a seed, and the seed matters.
            self.assertEqual(rgb, tower.generate(tokens, 64, 48, steps=2, seed=3))
            self.assertNotEqual(rgb, tower.generate(tokens, 64, 48, steps=2, seed=4))
            with self.assertRaisesRegex(V2Error, "exceeds"):
                tower.generate(tokens, 128, 48, steps=1)
        finally:
            tower.close()
        # Streamed from host memory, the same kernels see the same bytes.
        tower = V2Diffusion(encoder, transformer, vae, max_width=64, max_height=48,
                            max_prompt_tokens=64, weights="host")
        try:
            info = tower.info
            self.assertTrue(info["host_weights"])
            self.assertGreater(info["host_bytes"], 0)
            self.assertEqual(rgb, tower.generate(tokens, 64, 48, steps=2, seed=3))
        finally:
            tower.close()
        # Past 64 latents a side the decoder tiles; the picture is still whole.
        tower = V2Diffusion(encoder, transformer, vae, max_width=640, max_height=528,
                            max_prompt_tokens=64, weights="device")
        try:
            latents = [0.1] * (16 * 66 * 80)
            self.assertEqual(len(tower.decode_latents(latents, 66, 80)), 640 * 528 * 3)
        finally:
            tower.close()
            vae.close()
            transformer.close()
            encoder.close()


class ImageRequestTests(unittest.TestCase):
    def test_sizes_are_parsed_and_bounded(self) -> None:
        self.assertEqual(parse_size(None, (512, 512), (512, 512)), (512, 512))
        self.assertEqual(parse_size("256x384", (512, 512), (512, 512)), (256, 384))
        with self.assertRaises(APIError):
            parse_size("300x300", (512, 512), (512, 512))
        with self.assertRaises(APIError):
            parse_size("1024x1024", (512, 512), (512, 512))
        with self.assertRaises(APIError):
            parse_size(512, (512, 512), (512, 512))

    def test_a_hub_snapshot_reports_its_repository_name(self) -> None:
        root = Path("/cache/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/f332072a")
        self.assertEqual(snapshot_model_name(root), "Tongyi-MAI/Z-Image-Turbo")
        self.assertEqual(snapshot_model_name(Path("/models/zimage")), "zimage")


class ImagesEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = InferenceService("qwen-local", StubGenerator())
        self.server = FlyweightHTTPServer(("127.0.0.1", 0), create_handler(self.service))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)

    def tearDown(self) -> None:
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _post(self, body: dict) -> tuple[int, dict]:
        self.connection.request(
            "POST", "/v1/images/generations", body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        response = self.connection.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))

    def test_a_server_without_an_image_model_says_so(self) -> None:
        status, payload = self._post({"prompt": "a boat"})
        self.assertEqual(status, 404)
        self.assertIn("--image-model", payload["error"]["message"])

    def test_the_route_returns_what_the_service_renders(self) -> None:
        seen = {}

        def images_generations(payload):
            seen.update(payload)
            return {"created": 1, "data": [{"b64_json": "AAAA", "seed": 7}]}

        self.service.images_generations = images_generations
        status, payload = self._post({"prompt": "a boat", "size": "512x512"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"][0]["seed"], 7)
        self.assertEqual(seen["size"], "512x512")


if __name__ == "__main__":
    unittest.main()
