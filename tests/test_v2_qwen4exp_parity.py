"""qwen4exp forward parity against transformers' own Qwen4Exp decoder.

The synthetic GGUF fixture is loaded into this runtime (CPU backend, f32) and
its raw arrays are mapped back into a `Qwen4ExpTextForCausalLM`; the greedy
token at every position must agree. This pins everything the oracle module
checks cannot: the gated-residual bookends threaded through the real decode
path, the PLE staging + dilated conv state, the partial rope on the gated
attention, the MoE routing, and the head collapse standing in for a final norm.

The mapping back to HF inverts the two conversion-side transforms
(plans/qwen4exp-semantics.md): the +1 baked into every norm weight, and the
grouped->tiled reorder of DeltaNet value heads (llama.cpp
`_LinearAttentionVReorderBase`; colibri's kernels index `head % key_heads`,
the tiled order).

Skipped without torch and a transformers with the qwen4_exp model (merged
upstream 2026-08-26).
"""

from __future__ import annotations

import importlib.machinery
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from colibri_next.v2 import V2Model
from tests.qwen4exp_gguf_fixture import Qwen4ExpSpec, build_qwen4exp_gguf

# 64 positions, every one greedy-checked; includes the ple eos token (5) so
# the n-gram segmentation reset is exercised mid-sequence.
TOKENS = [(t * 37 + 11) % 96 for t in range(60)] + [5, 3, 9, 17]


def _load_transformers():
    if "torchaudio" not in sys.modules:
        stub = types.ModuleType("torchaudio")
        stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
        stub.__version__ = "0"
        sys.modules["torchaudio"] = stub
    try:
        import torch  # noqa: F401
        from transformers.models.qwen4_exp import modeling_qwen4_exp
    except Exception:  # pragma: no cover - depends on the environment
        return None
    # The hub/fla kernels refuse float32 and are not the reference; the plain
    # torch implementations the decorators wrap are.
    for name in ("torch_chunk_gated_delta_rule",
                 "torch_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update"):
        wrapped = getattr(modeling_qwen4_exp, name, None)
        if wrapped is not None and hasattr(wrapped, "__wrapped__"):
            setattr(modeling_qwen4_exp, name, wrapped.__wrapped__)
    return modeling_qwen4_exp


def _untile(array: np.ndarray, axis: int, key_heads: int, v_per_k: int,
            head_dim: int) -> np.ndarray:
    """Inverse of the conversion's grouped->tiled value-head reorder."""
    shape = list(array.shape)
    if axis < 0:
        axis += len(shape)
    tiled = array.reshape(
        shape[:axis] + [v_per_k, key_heads, head_dim] + shape[axis + 1:])
    grouped = np.swapaxes(tiled, axis, axis + 1)
    return np.ascontiguousarray(grouped).reshape(shape)


class Qwen4ExpForwardParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.modeling = _load_transformers()
        if cls.modeling is None:
            raise unittest.SkipTest(
                "torch and a qwen4_exp-capable transformers are needed")
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls._directory.name) / "qwen4exp.gguf"
        cls.spec = build_qwen4exp_gguf(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def _config(self):
        from transformers.models.qwen4_exp.configuration_qwen4_exp import (
            Qwen4ExpTextConfig,
        )
        spec: Qwen4ExpSpec = self.spec
        return Qwen4ExpTextConfig(
            vocab_size=spec.vocabulary,
            hidden_size=spec.hidden,
            num_hidden_layers=spec.layers,
            num_attention_heads=spec.heads,
            num_key_value_heads=spec.kv_heads,
            head_dim=spec.head_dim,
            full_attention_interval=spec.attention_every,
            linear_num_key_heads=spec.key_heads,
            linear_num_value_heads=spec.value_heads,
            linear_key_head_dim=spec.ssm_head_dim,
            linear_value_head_dim=spec.ssm_head_dim,
            linear_conv_kernel_dim=spec.conv_kernel,
            hc_count=spec.hc_count,
            hc_lowrank=spec.hc_low_rank,
            num_experts=spec.experts,
            num_experts_per_tok=spec.experts_used,
            moe_intermediate_size=spec.expert_intermediate,
            shared_expert_intermediate_size=spec.expert_intermediate,
            norm_topk_prob=True,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            output_gate_type="sigmoid",
            ple_layer_ids=[layer + 1 for layer in spec.ple_layers],
            ngram_size=spec.ngram_size,
            heads_per_ngram=spec.heads_per_ngram,
            ple_embed_dim=spec.hidden,
            ngram_vocab_size_base=96,
            make_ngram_vocab_size_divisible_by=1,
            ple_conv_kernel_size=spec.ple_conv_kernel,
            indexer_n_heads=spec.indexer_heads,
            indexer_head_dim=spec.indexer_key_dim,
            indexer_kv_heads=1,
            indexer_budget=spec.indexer_top_k,
            indexer_compress_ratio=4,
            max_position_embeddings=512,
            eos_token_id=spec.ple_eos_token_id,
            pad_token_id=None,
            tie_word_embeddings=False,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": spec.rotary_dim / spec.head_dim,
                # Sections sum to rotary_dim/2; for text all position streams
                # are equal, so the split is arbitrary but must be present.
                "mrope_section": [
                    spec.rotary_dim // 2 - 2 * (spec.rotary_dim // 6),
                    spec.rotary_dim // 6,
                    spec.rotary_dim // 6,
                ],
                "mrope_interleaved": True,
            },
        )

    def _state_dict(self):
        import torch

        spec: Qwen4ExpSpec = self.spec
        raw = self.spec.tensors
        kh, vh, hd = spec.key_heads, spec.value_heads, spec.ssm_head_dim
        vpk = vh // kh

        def t(name):
            return torch.from_numpy(np.ascontiguousarray(raw[name]))

        def norm(name):
            return torch.from_numpy(raw[name] - 1.0)

        out = {
            "model.embed_tokens.weight": t("token_embd.weight"),
            "lm_head.weight": t("output.weight"),
            "model.hyper_connection_mixer.hc_norm.weight":
                norm("output_hc_norm.weight"),
            "model.hyper_connection_mixer.input_mix_weight_down.weight":
                t("output_hc_down.weight"),
            "model.hyper_connection_mixer.input_mix_weight_up.weight":
                t("output_hc_up.weight"),
        }
        for layer in range(spec.layers):
            g = f"blk.{layer}."
            h = f"model.layers.{layer}."
            for kind, module in (("attn", "attn_hyper_connection"),
                                 ("ffn", "mlp_hyper_connection")):
                out[h + module + ".hc_norm.weight"] = norm(
                    g + f"hc_{kind}_norm.weight")
                out[h + module + ".input_mix_weight_down.weight"] = t(
                    g + f"hc_{kind}_down.weight")
                out[h + module + ".input_mix_weight_up.weight"] = t(
                    g + f"hc_{kind}_up.weight")
                out[h + module + ".block_inject_weight.weight"] = t(
                    g + f"hc_{kind}_inject.weight")
            if spec.is_attention(layer):
                a = h + "self_attn."
                out[a + "q_proj.weight"] = t(g + "attn_q.weight")
                out[a + "k_proj.weight"] = t(g + "attn_k.weight")
                out[a + "v_proj.weight"] = t(g + "attn_v.weight")
                out[a + "o_proj.weight"] = t(g + "attn_output.weight")
                out[a + "q_norm.weight"] = norm(g + "attn_q_norm.weight")
                out[a + "k_norm.weight"] = norm(g + "attn_k_norm.weight")
                # HF fuses the indexer q/k projections; the conversion splits
                # them with q rows first.
                out[a + "indexer.index_qk_proj.weight"] = torch.from_numpy(
                    np.concatenate([raw[g + "indexer.q_proj.weight"],
                                    raw[g + "indexer.k_proj.weight"]], axis=0))
                out[a + "indexer.q_layernorm.weight"] = norm(
                    g + "indexer.q_norm.weight")
                out[a + "indexer.k_layernorm.weight"] = norm(
                    g + "indexer.k_norm.weight")
            else:
                d = h + "linear_attn."
                qkv = raw[g + "attn_qkv.weight"].copy()
                key_dim = kh * hd
                qkv[2 * key_dim:] = _untile(qkv[2 * key_dim:], 0, kh, vpk, hd)
                out[d + "in_proj_qkv.weight"] = torch.from_numpy(qkv)
                out[d + "in_proj_z.weight"] = torch.from_numpy(
                    _untile(raw[g + "attn_gate.weight"], 0, kh, vpk, hd))
                out[d + "in_proj_a.weight"] = torch.from_numpy(
                    _untile(raw[g + "ssm_alpha.weight"], 0, kh, vpk, 1))
                out[d + "in_proj_b.weight"] = torch.from_numpy(
                    _untile(raw[g + "ssm_beta.weight"], 0, kh, vpk, 1))
                conv = raw[g + "ssm_conv1d.weight"].copy()  # [channels][tap]
                conv[2 * key_dim:] = _untile(conv[2 * key_dim:], 0, kh, vpk, hd)
                out[d + "conv1d.weight"] = torch.from_numpy(
                    conv[:, None, :].copy())
                out[d + "dt_bias"] = torch.from_numpy(
                    _untile(raw[g + "ssm_dt.bias"], 0, kh, vpk, 1))
                out[d + "A_log"] = torch.from_numpy(
                    np.log(-_untile(raw[g + "ssm_a"], 0, kh, vpk, 1)))
                # RMSNormGated is ones-initialized and applies its weight
                # directly -- the only norm class WITHOUT the (1+w) form, so
                # the conversion does not bake +1 into ssm_norm.
                out[d + "norm.weight"] = t(g + "ssm_norm.weight")
                out[d + "out_proj.weight"] = torch.from_numpy(
                    _untile(raw[g + "ssm_out.weight"], 1, kh, vpk, hd))
            if layer in spec.ple_layers:
                p = h + "ple."
                out[p + "ple_embedding.ngram_embedding.weight"] = t(
                    "per_layer_token_embd.weight")
                out[p + "key_proj.weight"] = t(g + "ple_key.weight")
                out[p + "value_proj.weight"] = t(g + "ple_value.weight")
                out[p + "norm_key.weight"] = norm(g + "ple_norm_key.weight")
                out[p + "norm_query.weight"] = norm(g + "ple_norm_query.weight")
                out[p + "norm_conv.weight"] = norm(g + "ple_norm_conv.weight")
                out[p + "conv1d.weight"] = torch.from_numpy(
                    raw[g + "ple_conv1d.weight"][:, None, :].copy())
            m = h + "mlp."
            out[m + "gate.weight"] = t(g + "ffn_gate_inp.weight")
            out[m + "experts.gate_up_proj"] = torch.from_numpy(
                np.concatenate([raw[g + "ffn_gate_exps.weight"],
                                raw[g + "ffn_up_exps.weight"]], axis=1))
            out[m + "experts.down_proj"] = t(g + "ffn_down_exps.weight")
            out[m + "shared_expert.gate_proj.weight"] = t(
                g + "ffn_gate_shexp.weight")
            out[m + "shared_expert.up_proj.weight"] = t(g + "ffn_up_shexp.weight")
            out[m + "shared_expert.down_proj.weight"] = t(
                g + "ffn_down_shexp.weight")
            out[m + "shared_expert_gate.weight"] = t(
                g + "ffn_gate_inp_shexp.weight").reshape(1, -1)
        return out

    def reference_tokens(self) -> list[int]:
        import torch

        config = self._config()
        config._attn_implementation = "eager"
        model = self.modeling.Qwen4ExpForCausalLM(config)
        model = model.to(torch.float32).eval()
        state = self._state_dict()
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Buffers (ngram multipliers) are derived from the config seed and must
        # already equal the fixture's metadata; anything else missing is a
        # mapping bug, not an acceptable skip.
        real_missing = [name for name in missing
                        if "layer_multipliers" not in name
                        and "ngram_heads" not in name]
        self.assertEqual(real_missing, [], f"unmapped parameters: {real_missing}")
        self.assertEqual(list(unexpected), [])
        ple = model.model.layers[self.spec.ple_layers[0]].ple
        np.testing.assert_array_equal(
            ple.ple_embedding.layer_multipliers.numpy().astype(np.uint64),
            np.array(self.spec.ple_multipliers, dtype=np.uint64))
        with torch.no_grad():
            logits = model(torch.tensor([TOKENS]), use_cache=False).logits[0]
        return logits.argmax(-1).tolist()

    def runtime_tokens(self) -> list[int]:
        V2Model.select_backend("cpu")
        try:
            with V2Model(str(self.path)) as model:
                with model.native_qwen_runtime(context_limit=256) as runtime:
                    runtime.prepare()
                    return [runtime.decode(token) for token in TOKENS]
        finally:
            V2Model.select_backend("auto")

    def test_greedy_tokens_match_transformers(self) -> None:
        self.assertEqual(self.runtime_tokens(), self.reference_tokens())


class Qwen4ExpQsaParityTest(Qwen4ExpForwardParityTest):
    """Phase 3: the same parity with the QSA selection ACTIVE.

    The release checkpoint's indexer budget (2048) never prunes inside a test,
    so this fixture shrinks it to 8 tokens at ratio 4: from position 11 on,
    every full-attention query keeps only its top 2 blocks plus the incomplete
    tail. Transformers applies the identical selection per row, so greedy
    parity now pins the whole indexer: raw-key caching, fp32 block pooling,
    k_norm + block-anchored rope, relu-sum scoring, deterministic top-k, and
    the sparse slot-list attention. The base class covers the dense-fallback
    side (its budget of 2048 never fires in 64 positions) -- together they are
    the phase-3 gate: bit-exact below the budget, reference-exact above it.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # QSA is opt-in (it costs 7-10% decode and a per-slot block-key arena),
        # so these tests must ask for it explicitly.
        cls._qsa_previous = os.environ.get("COLIBRI_QSA")
        os.environ["COLIBRI_QSA"] = "1"
        cls.modeling = _load_transformers()
        if cls.modeling is None:
            raise unittest.SkipTest(
                "torch and a qwen4_exp-capable transformers are needed")
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls._directory.name) / "qwen4exp_qsa.gguf"
        cls.spec = build_qwen4exp_gguf(
            cls.path, Qwen4ExpSpec(indexer_top_k=8))

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._qsa_previous is None:
            os.environ.pop("COLIBRI_QSA", None)
        else:
            os.environ["COLIBRI_QSA"] = cls._qsa_previous
        super().tearDownClass()

    def test_qsa_prefill_matches_decode(self) -> None:
        """Chunked rows prefill (16-row chunks straddling the pruning onset)
        must continue exactly like the token-by-token decode path."""
        import os

        continuation = 6
        V2Model.select_backend("cpu")
        os.environ["COLIBRI_PREFILL_ROWS"] = "16"
        try:
            generated: list[int] = []
            with V2Model(str(self.path)) as model:
                with model.native_qwen_runtime(context_limit=256) as runtime:
                    runtime.prepare()
                    runtime.generate(
                        TOKENS, continuation,
                        lambda token: (generated.append(token)
                                       or len(generated) < continuation))
            with V2Model(str(self.path)) as model:
                with model.native_qwen_runtime(context_limit=256) as runtime:
                    runtime.prepare()
                    outputs = [runtime.decode(token) for token in TOKENS]
                    decoded = [outputs[-1]]
                    for _ in range(continuation - 1):
                        decoded.append(runtime.decode(decoded[-1]))
        finally:
            del os.environ["COLIBRI_PREFILL_ROWS"]
            V2Model.select_backend("auto")
        self.assertEqual(generated, decoded)


class Qwen4ExpQsaPathTest(unittest.TestCase):
    """QSA on the paths transformers cannot referee: CUDA and multi-sequence.

    The CPU parity above fixes what the answer IS; these fix that every other
    execution path produces the same one. Same shrunken indexer budget (8 at
    ratio 4), so selection is active for most of the run.
    """

    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._qsa_previous = os.environ.get("COLIBRI_QSA")
        os.environ["COLIBRI_QSA"] = "1"
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls._directory.name) / "qwen4exp_qsa.gguf"
        cls.spec = build_qwen4exp_gguf(
            cls.path, Qwen4ExpSpec(indexer_top_k=8))

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._qsa_previous is None:
            os.environ.pop("COLIBRI_QSA", None)
        else:
            os.environ["COLIBRI_QSA"] = cls._qsa_previous
        cls._directory.cleanup()

    def _decode(self, backend: str, tokens: list[int]) -> list[int]:
        V2Model.select_backend(backend)
        try:
            with V2Model(str(self.path)) as model:
                with model.native_qwen_runtime(context_limit=256) as runtime:
                    runtime.prepare()
                    return [runtime.decode(token) for token in tokens]
        finally:
            V2Model.select_backend("auto")

    def test_cuda_matches_cpu(self) -> None:
        if not V2Model.gpu_info()["available"]:
            raise unittest.SkipTest("native CUDA runtime is unavailable")
        self.assertEqual(self._decode("cuda", TOKENS), self._decode("cpu", TOKENS))

    def test_interleaved_matches_solo(self) -> None:
        """Two sequences batched through qwen_decode_multi must each decode
        exactly as they do alone: the block-key store and raw-key ring live in
        per-slot arenas, and the parked-slot rule governs which position each
        slot's indexer reads."""
        prompt_a, prompt_b = TOKENS[:40], TOKENS[8:41]
        continuation = 6

        def solo(prompt: list[int]) -> list[int]:
            with V2Model(str(self.path)) as model:
                with model.native_qwen_runtime(context_limit=256) as runtime:
                    runtime.prepare()
                    out: list[int] = []
                    runtime.generate(
                        prompt, continuation,
                        lambda t: (out.append(t) or len(out) < continuation))
                    return out

        V2Model.select_backend("cpu")
        try:
            expected_a, expected_b = solo(prompt_a), solo(prompt_b)
            with V2Model(str(self.path)) as model:
                with model.native_qwen_runtime(
                        context_limit=256, parallel_sequences=2) as runtime:
                    runtime.prepare()
                    task_a = runtime.task_submit(prompt_a, continuation)
                    task_b = runtime.task_submit(prompt_b, continuation)
                    collected: dict[int, list[int]] = {task_a: [], task_b: []}
                    finished: set[int] = set()
                    for _ in range(64 + 8 * continuation):
                        for task, token, kind in runtime.engine_step():
                            if kind == 0:
                                collected[task].append(token)
                            elif kind == 1:
                                finished.add(task)
                            elif kind == 2:
                                raise AssertionError("engine task failed")
                        if finished == {task_a, task_b}:
                            break
                    self.assertEqual(finished, {task_a, task_b},
                                     "tasks did not finish")
        finally:
            V2Model.select_backend("auto")
        self.assertEqual(collected[task_a], expected_a, "interleaved task A")
        self.assertEqual(collected[task_b], expected_b, "interleaved task B")

    def test_dense_fallback_is_bit_exact_below_the_budget(self) -> None:
        """The phase-3 gate: with COLIBRI_QSA=0 the runtime must produce the
        identical tokens, because below the budget the selection keeps every
        block and the sparse path is mathematically the dense one. Any drift
        here means the indexer perturbed a path it should not touch."""
        import os

        short = TOKENS[:11]  # 11 visible tokens: 2 complete blocks, budget 2
        with_qsa = self._decode("cpu", short)
        os.environ["COLIBRI_QSA"] = "0"
        try:
            without = self._decode("cpu", short)
        finally:
            del os.environ["COLIBRI_QSA"]
        self.assertEqual(with_qsa, without)


if __name__ == "__main__":
    unittest.main()
