"""Tests for Raon-1 (raon1) architecture and runtime support in Flyweight."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from tests import raon1_gguf_fixture
from flyweight.v2 import CACHE_TYPE_CODES, V2Model, RaonRuntime
from flyweight.v2_server import NativeV2InferenceService
from flyweight.sampling import SamplingConfig


@pytest.fixture(scope="module")
def raon_model_path(tmp_path_factory) -> Path:
    dir_path = tmp_path_factory.mktemp("raon1_fixture")
    return raon1_gguf_fixture.build(dir_path)


def test_raon_pretokenizer(raon_model_path: Path):
    """Test custom Raon pre-tokenization with CJK, numbers, punctuation, and words."""
    with V2Model(raon_model_path) as model:
        # 1. CJK characters should split individually
        text = "안녕하세요"
        tokens = model.tokenize(text)
        assert len(tokens) > 0

        # 2. Digits in chunks of up to 3
        text = "1234567"
        tokens = model.tokenize(text)
        assert len(tokens) >= 3

        # 3. Simple ASCII words
        text = "hello world"
        tokens = model.tokenize(text)
        assert len(tokens) > 0


def test_raon_model_open(raon_model_path: Path):
    """Test opening the synthetic Raon-1 GGUF model and inspecting geometry."""
    with V2Model(raon_model_path) as model:
        config = model.config
        assert config["architecture"] == "raon1"
        assert config["hidden_size"] == 64
        assert config["layer_count"] == 2
        assert config["expert_count"] == 4
        assert config["expert_used_count"] == 2
        assert config["expert_shared_count"] == 1


def test_raon_runtime_eval(raon_model_path: Path):
    """Test forward pass through both RDA and MLA layers with multi-stream mixing."""
    with V2Model(raon_model_path) as model:
        with RaonRuntime(model, capacity=128, slots=2) as runtime:
            assert runtime.slot_count == 2
            assert isinstance(runtime.uses_gpu, bool)

            # Single token eval
            logits = runtime.eval([5], slot=0)
            assert isinstance(logits, np.ndarray)
            assert logits.shape == (raon1_gguf_fixture.VOCAB,)
            assert np.all(np.isfinite(logits))

            # Multi-token prompt eval
            logits_multi = runtime.eval([5, 6, 7], slot=0)
            assert logits_multi.shape == (raon1_gguf_fixture.VOCAB,)
            assert np.all(np.isfinite(logits_multi))


def test_raon_batched_prefill_matches_token_steps(raon_model_path: Path):
    """A prompt evaluated in one call (the chunked, layer-major prefill) must
    land on the same state as the same prompt fed one token at a time (the
    decode step): same logits now and after a further token. 80 tokens spans
    the prefill's streaming threshold, and on a GPU build reaches the device
    MLA attention and the streamed weights."""
    prompt = [(7 * i + 3) % raon1_gguf_fixture.VOCAB for i in range(80)]
    with V2Model(raon_model_path) as model:
        with RaonRuntime(model, capacity=128, slots=2) as runtime:
            batched = runtime.eval(prompt, slot=0).copy()
            for token in prompt:
                stepped = runtime.eval([token], slot=1)
            np.testing.assert_allclose(batched, stepped, rtol=2e-3, atol=2e-3)
            np.testing.assert_allclose(runtime.eval([5], slot=0), runtime.eval([5], slot=1),
                                       rtol=2e-3, atol=2e-3)


def test_raon_sampling_and_generation(raon_model_path: Path):
    """Test token sampling and generation loop."""
    with V2Model(raon_model_path) as model:
        with RaonRuntime(model, capacity=64, slots=1) as runtime:
            runtime.eval([5, 6])
            config = SamplingConfig(temperature=0.7, top_k=5, top_p=0.9, seed=123)
            token = runtime.sample(config)
            assert isinstance(token, int)
            assert 0 <= token < raon1_gguf_fixture.VOCAB

            # Test generation
            runtime.reset()
            tokens = runtime.generate([5], max_tokens=4)
            assert len(tokens) == 4
            for t in tokens:
                assert 0 <= t < raon1_gguf_fixture.VOCAB


def test_raon_slot_isolation(raon_model_path: Path):
    """Test that concurrent slots maintain isolated recurrent states and caches."""
    with V2Model(raon_model_path) as model:
        with RaonRuntime(model, capacity=64, slots=2) as runtime:
            # Advance slot 0 with sequence A
            logits0 = runtime.eval([5, 6], slot=0)
            # Advance slot 1 with sequence B
            logits1 = runtime.eval([7, 8], slot=1)

            assert not np.allclose(logits0, logits1)

            # Resetting slot 0 should not disturb slot 1
            runtime.reset(slot=0)
            # Advance slot 1 with another token
            logits1_next = runtime.eval([5], slot=1)
            assert np.all(np.isfinite(logits1_next))

            # Slot 0 from fresh start with [5, 6] should produce logits0 again
            logits0_fresh = runtime.eval([5, 6], slot=0)
            np.testing.assert_allclose(logits0, logits0_fresh, atol=1e-5)


def test_raon_cache_save_and_load(raon_model_path: Path):
    """Test saving and restoring recurrent delta and MLA KV cache state."""
    with V2Model(raon_model_path) as model:
        with RaonRuntime(model, capacity=64, slots=2) as runtime:
            # Run prompt on slot 0
            runtime.eval([5, 6, 7], slot=0)
            saved_state = runtime.save_state(slot=0)
            assert isinstance(saved_state, bytes)
            assert len(saved_state) > 0

            # Step one token further to get target continuation logits
            target_logits = runtime.eval([8], slot=0).copy()

            # Corrupt / change slot 0 with other tokens
            runtime.eval([9, 10, 11], slot=0)
            corrupted_logits = runtime.eval([8], slot=0)
            assert not np.allclose(target_logits, corrupted_logits)

            # Restore saved state into slot 0
            runtime.load_state(saved_state, slot=0)
            restored_logits = runtime.eval([8], slot=0)
            np.testing.assert_allclose(target_logits, restored_logits, atol=1e-5)

            # Restore into slot 1 and verify identical logits
            runtime.load_state(saved_state, slot=1)
            slot1_logits = runtime.eval([8], slot=1)
            np.testing.assert_allclose(target_logits, slot1_logits, atol=1e-5)


def test_raon_server_service(raon_model_path: Path):
    """Test NativeV2InferenceService integration with Raon-1."""
    service = NativeV2InferenceService(
        str(raon_model_path),
        context_window=128,
        max_new_tokens=4,
    )
    try:
        assert service.architecture == "raon1"
        health = service.health()
        assert health["execution"]["backend"] == "native-v2-raon1"
        assert health["execution"]["architecture"] == "raon1"
        assert health["execution"]["device"] in ("cpu", "gpu")

        # Chat completion
        response = service.chat_completion({
            "model": "raon1-synthetic-test",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 3,
            "temperature": 0.0,
        })
        assert "choices" in response
        assert len(response["choices"]) > 0
    finally:
        service.close()


def test_raon_cli(raon_model_path: Path):
    """Test CLI architecture detection and benchmark command for Raon-1."""
    from flyweight.cli import _architecture, _parser, _benchmark

    arch = _architecture(raon_model_path)
    assert arch == "raon1"

    parser = _parser()
    args = parser.parse_args([
        "benchmark", str(raon_model_path),
        "--iterations", "3",
        "--warmup", "1",
        "--context", "64",
        "--prompt", "hello",
    ])
    code = _benchmark(args)
    assert code == 0


@pytest.mark.parametrize("cache_type", ["f16", "bf16"])
def test_raon_half_cache(raon_model_path: Path, cache_type: str):
    """A 16-bit MLA cache: batched prefill and token steps agree with each
    other, and stay close to the f32 cache (the latent is RMS-normalized, so
    16 bits keep it well)."""
    prompt = [(5 * i + 1) % raon1_gguf_fixture.VOCAB for i in range(80)]
    with V2Model(raon_model_path) as model:
        with RaonRuntime(model, capacity=128, slots=2, cache_type=cache_type) as runtime:
            assert runtime.placement["cache_type_k"] == CACHE_TYPE_CODES[cache_type]
            batched = runtime.eval(prompt, slot=0).copy()
            for token in prompt:
                stepped = runtime.eval([token], slot=1)
            np.testing.assert_allclose(batched, stepped, rtol=2e-3, atol=2e-3)
        with RaonRuntime(model, capacity=128, cache_type="f32") as reference:
            exact = reference.eval(prompt).copy()
        np.testing.assert_allclose(batched, exact, rtol=0.05, atol=0.05)


def test_raon_cache_snapshot_crosses_types(raon_model_path: Path):
    """A snapshot keeps the type it was taken in and loads into any other."""
    prompt = [3, 9, 4, 1, 5, 9, 2, 6]
    with V2Model(raon_model_path) as model:
        with RaonRuntime(model, capacity=64, cache_type="f16") as half:
            half.eval(prompt)
            saved = half.save_state()
            expected = half.eval([7]).copy()
        with RaonRuntime(model, capacity=64, cache_type="f32") as full:
            full.load_state(saved)
            np.testing.assert_allclose(full.eval([7]), expected, rtol=1e-4, atol=1e-4)


def test_raon_cache_type_validation():
    from flyweight.v2 import raon_cache_type

    assert raon_cache_type("auto", "auto") == "f16"
    assert raon_cache_type("bf16", "bf16") == "bf16"
    with pytest.raises(ValueError, match="must match"):
        raon_cache_type("f16", "f32")
    with pytest.raises(ValueError, match="f32, f16 or bf16"):
        raon_cache_type("q8_0", "q8_0")


def test_raon_server_accepts_a_system_prompt(raon_model_path: Path):
    """A leading system message gives the shared streaming path an opening
    anchor, which it hands to the engine's submit() along with any images; the
    Raon engine must take both (opencode sends a system prompt on every
    request, and the tiny fixture tokenizer never yields an anchor itself)."""
    service = NativeV2InferenceService(str(raon_model_path), context_window=128, max_new_tokens=4)
    try:
        engine = service.generator.engine
        task, events = engine.submit([5, 6, 7], 2, (), anchor=2, images=None)
        kinds = []
        while True:
            kind, _ = events.get(timeout=60)
            kinds.append(kind)
            if kind in ("done", "error"):
                break
        assert kinds[-1] == "done", kinds
    finally:
        service.close()
