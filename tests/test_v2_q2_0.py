"""Q2_0 (GGML type 42) experts and shared experts.

ISTA DASLab's GSQ-RCO builds of Qwen3.8-Flash-Next store the routed expert
down projection in Q2_0 on most layers and the shared-expert down projection
in it on a few. The type is 18 bytes per 64 values: an f16 scale then sixteen
bytes of two-bit codes, element j at bits 2*(j%4) of byte j/4, code q meaning
(q-1)*d. Prefill can take q20_q8_mmq_routed (64-wide unit, so 640-wide downs
divide). Decode still uses the CPU expert path / grouped octet kernels, and a
Q2_0 static tensor is requantized to Q8_0 at prepare.

The fixture packs the same random weights two ways -- Q2_0 bytes, and the
decoded values as f32 -- so the runtime's Q2_0 path is checked against its own
f32 path on identical weights, in the batched prefill (row dequant) and in
decode (row dot) alike.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from flyweight.deepseek4 import expert_matvec, matvec
from flyweight.v2 import V2Model

from tests.qwen4exp_gguf_fixture import (
    GGML_BF16, GGML_IQ2_S, GGML_IQ4_NL, GGML_Q2_0, Qwen4ExpSpec, build_qwen4exp_gguf,
)

# Expert rows are `expert_intermediate` wide, and Q2_0 blocks 64 values; the
# IQ2_S gate/up rows are `hidden` wide and need a 256-value super-block.
SPEC = dict(expert_intermediate=64, hidden=256, ple_head_dim=64)
PROMPT = [(t * 37 + 11) % 96 for t in range(96)]
CONTINUATION = 8
CONTEXT = 256


def _gpu_present() -> bool:
    try:
        return bool(V2Model.gpu_info()["available"])
    except Exception:
        return False


def _generate_on_gpu(path: Path) -> list[int]:
    """Routed experts resident on the device. IQ-family experts only reach the
    GPU behind an explicit seed (prepare otherwise keeps them on the CPU to
    avoid the per-layer round trip), so seed the whole fixture's expert set."""
    out: list[int] = []
    V2Model.select_backend("auto")
    try:
        with V2Model(str(path)) as model:
            with model.native_qwen_runtime(
                    context_limit=CONTEXT, expert_mode="resident",
                    prefill_cache_seed=8) as runtime:
                runtime.prepare()
                info = runtime.info
                assert info["expert_mode"] != "cpu", info["expert_mode"]
                runtime.generate(
                    PROMPT, CONTINUATION,
                    lambda t: (out.append(t) or len(out) < CONTINUATION))
    finally:
        V2Model.select_backend("cpu")
    return out


def _generate(path: Path, **options) -> list[int]:
    out: list[int] = []
    with V2Model(str(path)) as model:
        with model.native_qwen_runtime(context_limit=CONTEXT, **options) as runtime:
            runtime.prepare()
            runtime.generate(
                PROMPT, CONTINUATION,
                lambda t: (out.append(t) or len(out) < CONTINUATION))
    return out


class Q20Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        experts = {"ffn_down_exps.weight": GGML_Q2_0}
        both = dict(experts, **{"ffn_down_shexp.weight": GGML_Q2_0})
        cls.experts_path = root / "experts.gguf"
        cls.experts_spec = build_qwen4exp_gguf(
            cls.experts_path, Qwen4ExpSpec(**SPEC), quantize=experts)
        cls.twin_path = root / "twin.gguf"
        build_qwen4exp_gguf(
            cls.twin_path, Qwen4ExpSpec(**SPEC), quantize=experts,
            quantize_in_f32=True)
        cls.both_path = root / "both.gguf"
        cls.both_spec = build_qwen4exp_gguf(
            cls.both_path, Qwen4ExpSpec(**SPEC), quantize=both)
        # The GSQ-RCO layout on its other layers: IQ4_NL shared-expert down,
        # a format with expert kernels but no dense matvec.
        cls.iq4nl_path = root / "iq4nl.gguf"
        cls.iq4nl_spec = build_qwen4exp_gguf(
            cls.iq4nl_path, Qwen4ExpSpec(**SPEC),
            quantize=dict(experts, **{"ffn_down_shexp.weight": GGML_IQ4_NL}))
        # And its bf16 shared-expert gate, which the dense requant must leave
        # alone: the gate kernels read f32 or bf16, never Q8_0.
        gate = {"ffn_gate_inp_shexp.weight": GGML_BF16}
        cls.bf16_gate_path = root / "bf16_gate.gguf"
        build_qwen4exp_gguf(cls.bf16_gate_path, Qwen4ExpSpec(**SPEC), quantize=gate)
        cls.bf16_gate_twin_path = root / "bf16_gate_twin.gguf"
        build_qwen4exp_gguf(
            cls.bf16_gate_twin_path, Qwen4ExpSpec(**SPEC), quantize=gate,
            quantize_in_f32=True)
        # IQ2_S gate/up experts, the other GSQ-RCO format that had only a
        # scalar row dequant on the batched prefill path.
        iq2s = dict(experts, **{"ffn_gate_exps.weight": GGML_IQ2_S,
                                "ffn_up_exps.weight": GGML_IQ2_S})
        cls.iq2s_path = root / "iq2s.gguf"
        cls.iq2s_spec = build_qwen4exp_gguf(
            cls.iq2s_path, Qwen4ExpSpec(**SPEC), quantize=iq2s)
        cls.iq2s_twin_path = root / "iq2s_twin.gguf"
        build_qwen4exp_gguf(
            cls.iq2s_twin_path, Qwen4ExpSpec(**SPEC), quantize=iq2s,
            quantize_in_f32=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def setUp(self) -> None:
        V2Model.select_backend("cpu")

    def tearDown(self) -> None:
        V2Model.select_backend("auto")

    def test_the_type_is_decodable(self) -> None:
        with V2Model(str(self.both_path)) as model:
            self.assertEqual(model.unsupported_quant_types(), {})

    def test_expert_rows_decode_to_the_packed_values(self) -> None:
        # The dot kernel against the fixture's own decode of the bytes it
        # wrote, on a routed expert (3D, one expert's slice) and on the
        # shared expert (2D).
        rng = np.random.default_rng(3)
        spec = self.both_spec
        with V2Model(str(self.both_path)) as model:
            for layer, expert in ((2, 0), (2, 5), (7, 3)):
                name = f"blk.{layer}.ffn_down_exps.weight"
                weights = spec.tensors[name][expert]  # (hidden, intermediate)
                x = rng.standard_normal(weights.shape[1]).astype(np.float32)
                got = expert_matvec(model, name, expert, x, weights.shape[0])
                np.testing.assert_allclose(got, weights @ x, rtol=1e-5, atol=1e-5)
            name = "blk.2.ffn_down_shexp.weight"
            weights = spec.tensors[name]
            x = rng.standard_normal(weights.shape[1]).astype(np.float32)
            got = matvec(model, name, x, weights.shape[0])
            np.testing.assert_allclose(got, weights @ x, rtol=1e-5, atol=1e-5)

    def test_prefill_and_decode_match_the_f32_twin(self) -> None:
        # Same weights, two encodings. A 96-token prompt goes through the
        # batched CPU MoE (row dequant), the continuation through the
        # per-token dot; both must agree with the f32 file to a token.
        self.assertEqual(_generate(self.experts_path), _generate(self.twin_path))

    def test_a_q2_0_shared_expert_runs(self) -> None:
        # The shared expert is a static tensor: prepare requantizes it to
        # Q8_0, since no device kernel reads type 42.
        tokens = _generate(self.both_path)
        self.assertEqual(len(tokens), CONTINUATION)
        self.assertTrue(all(0 <= t < 96 for t in tokens), tokens)

    def test_a_bf16_shared_expert_gate_survives_the_dense_requant(self) -> None:
        # Forced Q8_0 requant of every bf16 static tensor. The gate has to be
        # exempt: with it converted, qwen_shared_scale read Q8_0 bytes as f32
        # and the GSQ-RCO checkpoint produced noise in every mode whose
        # budget chose to requant.
        self.assertEqual(
            _generate(self.bf16_gate_path, dense_requant="q8"),
            _generate(self.bf16_gate_twin_path))

    def test_iq2_s_experts_match_the_f32_twin(self) -> None:
        # Decode reads IQ2_S through the vectorized dot; prefill through the
        # row dequant that was scalar until 2026-09-16. Check both against
        # the fixture's decode of its own bytes, then against the twin file.
        rng = np.random.default_rng(7)
        name = "blk.2.ffn_gate_exps.weight"
        weights = self.iq2s_spec.tensors[name][1]
        x = rng.standard_normal(weights.shape[1]).astype(np.float32)
        with V2Model(str(self.iq2s_path)) as model:
            got = expert_matvec(model, name, 1, x, weights.shape[0])
        np.testing.assert_allclose(got, weights @ x, rtol=1e-4, atol=1e-4)
        self.assertEqual(_generate(self.iq2s_path), _generate(self.iq2s_twin_path))

    @unittest.skipUnless(_gpu_present(), "no CUDA device available")
    def test_q2_0_experts_on_the_gpu_match_the_cpu(self) -> None:
        # The grouped q20 family (decode) and q20_matmul_rows (streamed
        # prefill) against the CPU expert path on the same bytes.
        self.assertEqual(_generate_on_gpu(self.experts_path), _generate(self.experts_path))

    @unittest.skipUnless(_gpu_present(), "no CUDA device available")
    def test_iq2_s_experts_on_the_gpu_match_the_cpu(self) -> None:
        self.assertEqual(_generate_on_gpu(self.iq2s_path), _generate(self.iq2s_path))

    def test_an_iq4_nl_shared_expert_runs(self) -> None:
        # Same requant for IQ4_NL, which until now only ever appeared on the
        # routed experts and the PLE table.
        rng = np.random.default_rng(5)
        name = "blk.2.ffn_down_shexp.weight"
        weights = self.iq4nl_spec.tensors[name]
        x = rng.standard_normal(weights.shape[1]).astype(np.float32)
        with V2Model(str(self.iq4nl_path)) as model:
            self.assertEqual(model.unsupported_quant_types(), {})
            got = matvec(model, name, x, weights.shape[0])
        np.testing.assert_allclose(got, weights @ x, rtol=1e-5, atol=1e-5)
        tokens = _generate(self.iq4nl_path)
        self.assertEqual(len(tokens), CONTINUATION)
        self.assertTrue(all(0 <= t < 96 for t in tokens), tokens)


if __name__ == "__main__":
    unittest.main()
