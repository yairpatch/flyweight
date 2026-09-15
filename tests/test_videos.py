"""The MiniMax-H3 video path: request parsing, the schedule, and the videos endpoint."""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from flyweight.server import APIError, FlyweightHTTPServer, InferenceService, create_handler
from flyweight.videos import (
    VideoGenerator, align_frames, audio_latents, latent_frames, parse_video_size,
    resolve_model_dir, sigma_schedule,
)
from tests.test_server import StubGenerator


class VideoRequestTests(unittest.TestCase):
    def test_frames_snap_to_what_the_vae_decodes(self) -> None:
        # 17n + 5 frames become 5n + 2 latent frames; anything else rounds up.
        self.assertEqual(align_frames(5), 5)
        self.assertEqual(align_frames(6), 22)
        self.assertEqual(align_frames(22), 22)
        self.assertEqual(align_frames(120), 124)
        self.assertEqual(align_frames(124), 124)
        self.assertEqual(latent_frames(5), 2)
        self.assertEqual(latent_frames(22), 7)
        self.assertEqual(latent_frames(124), 37)
        # 40 audio latents per second at 24 fps.
        self.assertEqual(audio_latents(124), 207)
        self.assertEqual(audio_latents(22), 37)

    def test_the_schedule_is_the_shifted_linspace_ending_at_zero(self) -> None:
        sigmas = sigma_schedule(8, 12.0)
        self.assertEqual(len(sigmas), 8)
        self.assertEqual(sigmas[0], 1.0)
        self.assertEqual(sigmas[-1], 0.0)
        self.assertTrue(all(a > b for a, b in zip(sigmas, sigmas[1:])))
        # sigma' = s*sigma / (1 + (s-1)*sigma) at the midpoint.
        base = 1.0 - 3 / 7
        self.assertAlmostEqual(sigmas[3], 12.0 * base / (1.0 + 11.0 * base), places=6)
        # A shift of one leaves the grid alone.
        self.assertAlmostEqual(sigma_schedule(5, 1.0)[1], 0.75, places=6)

    def test_sizes_are_parsed_and_bounded(self) -> None:
        self.assertEqual(parse_video_size(None, (640, 384), (640, 384)), (640, 384))
        self.assertEqual(parse_video_size("auto", (640, 384), (640, 384)), (640, 384))
        self.assertEqual(parse_video_size("384x640", (640, 384), (640, 640)), (384, 640))
        with self.assertRaises(APIError):
            parse_video_size("640x400", (640, 384), (640, 384))  # not a multiple of 32
        with self.assertRaises(APIError):
            parse_video_size("1024x576", (640, 384), (640, 384))  # past the tower's canvas
        with self.assertRaises(APIError):
            parse_video_size(640, (640, 384), (640, 384))

    def test_a_model_directory_names_what_it_lacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(FileNotFoundError, "transformer.gguf"):
                resolve_model_dir(root)
            with self.assertRaisesRegex(FileNotFoundError, "MiniMax-H3 model directory"):
                VideoGenerator(root)
            for name in ("text_encoder.gguf", "transformer.gguf"):
                (root / name).write_bytes(b"")
            for part in ("vae", "text_encoder", "transformer"):
                (root / part).mkdir()
                (root / part / "config.json").write_text("{}")
            (root / "tokenizer").mkdir()
            (root / "tokenizer" / "tokenizer.json").write_text("{}")
            paths = resolve_model_dir(root)
            self.assertEqual(paths["vae"], root / "vae")
            self.assertEqual(paths["tokenizer"], root / "tokenizer")


class VideosEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = InferenceService("qwen-local", StubGenerator())
        self.server = FlyweightHTTPServer(("127.0.0.1", 0), create_handler(self.service))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection(*self.server.server_address)

    def tearDown(self) -> None:
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()

    def _post(self, body: dict) -> tuple[int, dict]:
        self.connection.request(
            "POST", "/v1/videos/generations", body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        response = self.connection.getresponse()
        return response.status, json.loads(response.read())

    def test_a_server_without_a_video_model_says_so(self) -> None:
        status, payload = self._post({"prompt": "a boat"})
        self.assertEqual(status, 404)
        self.assertIn("--video-model", payload["error"]["message"])

    def test_the_streaming_route_relays_progress_and_the_clip(self) -> None:
        def stream_videos_generations(payload):
            yield {"type": "progress", "step": 0, "steps": 8, "stage": "encode"}
            yield {"type": "video", "index": 0, "count": 1, "b64_json": "AAAA", "seed": 3, "seconds": 1.5}
            yield {"type": "done", "frames": 22}
            yield "[DONE]"

        self.service.stream_videos_generations = stream_videos_generations
        self.connection.request(
            "POST", "/v1/videos/generations", body=json.dumps({"prompt": "a boat", "stream": True}),
            headers={"Content-Type": "application/json"},
        )
        response = self.connection.getresponse()
        self.assertEqual(response.status, 200)
        body = response.read().decode()
        self.assertIn('"stage": "encode"', body)
        self.assertIn('"b64_json": "AAAA"', body)
        self.assertIn("data: [DONE]", body)

    def test_the_route_returns_what_the_service_renders(self) -> None:
        seen = {}

        def videos_generations(payload):
            seen.update(payload)
            return {"created": 1, "data": [{"b64_json": "AAAA", "seed": 3}], "frames": 22}

        self.service.videos_generations = videos_generations
        status, payload = self._post({"prompt": "a boat", "size": "256x256", "frames": 22})
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"][0]["seed"], 3)
        self.assertEqual(seen["frames"], 22)


if __name__ == "__main__":
    unittest.main()
