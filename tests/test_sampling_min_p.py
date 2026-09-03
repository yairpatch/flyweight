"""min_p sampling and the llama.cpp-aligned built-in defaults."""

from __future__ import annotations

import unittest

import numpy as np

from flyweight import sampling
from flyweight.sampling import SamplingConfig
from flyweight.server import InferenceService

from tests.test_server import StubGenerator


class DefaultsTests(unittest.TestCase):
    def test_built_in_defaults_follow_llama_cpp(self) -> None:
        defaults = sampling.defaults()
        self.assertEqual(defaults["temperature"], 0.8)
        self.assertEqual(defaults["top_k"], 40)
        self.assertEqual(defaults["top_p"], 0.95)
        self.assertEqual(defaults["min_p"], 0.05)
        self.assertEqual(defaults["repetition_penalty"], 1.0)
        self.assertEqual(defaults["penalty_window"], 64)

    def test_min_p_is_a_server_setting_and_validated(self) -> None:
        self.assertIn("min_p", {setting.name for setting in sampling.SERVER_SETTINGS})
        SamplingConfig(min_p=0.0)
        SamplingConfig(min_p=1.0)
        with self.assertRaises(ValueError):
            SamplingConfig(min_p=1.5)
        with self.assertRaises(ValueError):
            SamplingConfig(min_p=-0.1)

    def test_service_defaults_and_requests_carry_min_p(self) -> None:
        generator = StubGenerator()
        service = InferenceService("qwen-local", generator, max_new_tokens=32)
        self.assertEqual(service.generation_defaults["min_p"], 0.05)
        service.chat_completion({"messages": [{"role": "user", "content": "Hi"}]})
        self.assertEqual(generator.calls[-1][1]["sampling"].min_p, 0.05)
        service.chat_completion(
            {"messages": [{"role": "user", "content": "Hi"}], "min_p": 0.3}
        )
        self.assertEqual(generator.calls[-1][1]["sampling"].min_p, 0.3)
        service.anthropic_message(
            {"model": "m", "max_tokens": 4, "min_p": 0.2,
             "messages": [{"role": "user", "content": "Hi"}]}
        )
        self.assertEqual(generator.calls[-1][1]["sampling"].min_p, 0.2)
        with self.assertRaises(Exception):
            service.chat_completion(
                {"messages": [{"role": "user", "content": "Hi"}], "min_p": 2}
            )


class NumpySamplerTests(unittest.TestCase):
    """The host-side sampler shared by the DeepSeek path applies min_p after
    top_p, relative to the best candidate, and always keeps one token."""

    def test_min_p_drops_the_tail(self) -> None:
        from flyweight.deepseek4_server import sample_token

        # Probabilities at temperature 1: ~[0.64, 0.24, 0.09, 0.03].
        logits = np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32)
        rng = np.random.default_rng(0)
        loose = {sample_token(logits, SamplingConfig(temperature=1.0, top_k=0, top_p=1.0, min_p=0.0), rng) for _ in range(400)}
        self.assertEqual(loose, {0, 1, 2, 3})
        tight = {sample_token(logits, SamplingConfig(temperature=1.0, top_k=0, top_p=1.0, min_p=0.2), rng) for _ in range(400)}
        self.assertEqual(tight, {0, 1})
        only = {sample_token(logits, SamplingConfig(temperature=1.0, top_k=0, top_p=1.0, min_p=1.0), rng) for _ in range(50)}
        self.assertEqual(only, {0})


if __name__ == "__main__":
    unittest.main()
