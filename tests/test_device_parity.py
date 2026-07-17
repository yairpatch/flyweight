import os
import tempfile
import unittest
from pathlib import Path

from colibri_next.attention_converter import QwenAttentionConverter
from colibri_next.causal_lm import QwenForCausalLM
from colibri_next.converter import QwenCheckpointConverter
from colibri_next.cuda import (
    CudaUnavailableError,
    active_cuda,
    configure_cuda,
    disable_cuda,
)
from colibri_next.gated_delta_converter import QwenGatedDeltaConverter
from colibri_next.model_io_converter import QwenModelIOConverter
from colibri_next.moe_converter import QwenMoELayerConverter
from colibri_next.sampling import LogitsSampler, SamplingConfig

from tests.test_decoder import create_decoder_checkpoint


PROMPT_IDS = [0, 1, 2]
DECODE_STEPS = 3
TOLERANCE = 2e-3


def convert_model(source: Path, output: Path, quantization: str) -> None:
    QwenCheckpointConverter(source).convert(
        output, extract_experts=True, quantization="q4"
    )
    QwenMoELayerConverter(source).convert(output)
    QwenGatedDeltaConverter(source).convert(output, quantization=quantization)
    QwenAttentionConverter(source).convert(output, quantization=quantization)
    QwenModelIOConverter(source).convert(output, quantization=quantization)


class DeviceDecodeParityTests(unittest.TestCase):
    """The portable and device-resident decode paths must not drift apart."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            configure_cuda(cache_mib=64)
        except CudaUnavailableError as error:
            raise unittest.SkipTest(str(error)) from error
        disable_cuda()

    def test_bf16_static_decode_matches_portable_path(self) -> None:
        self._assert_device_parity("bf16")

    def test_q4_static_decode_matches_portable_path(self) -> None:
        self._assert_device_parity("q4")

    def test_hybrid_cpu_moe_offload_matches_portable_path(self) -> None:
        self._assert_device_parity("q4", cpu_moe_layers=1)

    def test_seeded_device_sampler_matches_host_sampler(self) -> None:
        accelerator = configure_cuda(cache_mib=64)
        try:
            logits = [0.1, 2.5, -1.0, 0.7, 1.3, -0.4, 0.9, 2.1]
            device_logits = accelerator.device_vector(logits)
            greedy = LogitsSampler(SamplingConfig(temperature=0))
            self.assertEqual(
                greedy.sample(logits),
                greedy.sample_device(device_logits, accelerator),
            )
            config = SamplingConfig(
                temperature=0.8, top_k=5, top_p=0.9, seed=7
            )
            host_tokens = [
                LogitsSampler(config).sample(logits) for _ in range(4)
            ]
            device_tokens = [
                LogitsSampler(config).sample_device(device_logits, accelerator)
                for _ in range(4)
            ]
            self.assertEqual(host_tokens, device_tokens)
        finally:
            disable_cuda()

    def test_fused_delta_prefill_matches_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_decoder_checkpoint(source)
            convert_model(source, output, "bf16")
            model = QwenForCausalLM.from_model_directory(output)
            accelerator = configure_cuda(cache_mib=64)
            try:
                os.environ["COLIBRI_FUSED_DELTA_PREFILL"] = "0"
                fallback_state = model.new_state()
                fallback = model.prefill_device(PROMPT_IDS, fallback_state)
                fallback_logits = accelerator.device_to_host(fallback.logits)

                os.environ["COLIBRI_FUSED_DELTA_PREFILL"] = "1"
                fused_state = model.new_state()
                fused = model.prefill_device(PROMPT_IDS, fused_state)
                fused_logits = accelerator.device_to_host(fused.logits)
                for expected, actual in zip(fallback_logits, fused_logits):
                    self.assertAlmostEqual(actual, expected, delta=TOLERANCE)
                self.assertEqual(fused_state.tokens, fallback_state.tokens)
            finally:
                os.environ.pop("COLIBRI_FUSED_DELTA_PREFILL", None)
                disable_cuda()

    def _assert_device_parity(
        self, quantization: str, *, cpu_moe_layers: int = 0
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            create_decoder_checkpoint(source)
            convert_model(source, output, quantization)
            model = QwenForCausalLM.from_model_directory(output)

            self.assertIsNone(active_cuda())
            reference_steps = self._reference_decode(model)

            accelerator = configure_cuda(cache_mib=64)
            try:
                model.configure_moe_placement(cpu_moe_layers)
                self.assertEqual(model.cpu_moe_layers, cpu_moe_layers)
                # Offloading MoE experts to CPU must not disable the GPU path.
                self.assertTrue(model.device_decode_available)
                state = model.new_state()
                result = model.prefill_device(PROMPT_IDS, state)
                for index, (token_id, expected_logits) in enumerate(
                    reference_steps
                ):
                    logits = accelerator.device_to_host(result.logits)
                    self.assertEqual(len(logits), len(expected_logits))
                    for expected, actual in zip(expected_logits, logits):
                        self.assertAlmostEqual(
                            actual, expected, delta=TOLERANCE
                        )
                    ranked = sorted(expected_logits, reverse=True)
                    if ranked[0] - ranked[1] > 2 * TOLERANCE:
                        greedy = max(
                            range(len(logits)), key=logits.__getitem__
                        )
                        self.assertEqual(greedy, token_id)
                    if index + 1 < len(reference_steps):
                        result = model.forward_token_device(token_id, state)
                self.assertEqual(
                    state.tokens, len(PROMPT_IDS) + DECODE_STEPS - 1
                )
            finally:
                disable_cuda()

    def _reference_decode(self, model) -> list[tuple[int, list[float]]]:
        state = model.new_state()
        result = model.prefill(PROMPT_IDS, state)
        steps: list[tuple[int, list[float]]] = []
        for index in range(DECODE_STEPS):
            token_id = result.greedy_token
            steps.append((token_id, result.logits))
            if index + 1 < DECODE_STEPS:
                result = model.forward_token(token_id, state)
        self.assertEqual(state.tokens, len(PROMPT_IDS) + DECODE_STEPS - 1)
        return steps


if __name__ == "__main__":
    unittest.main()
