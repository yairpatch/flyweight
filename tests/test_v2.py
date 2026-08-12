import ctypes
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

from colibri_next.v2 import (
    V2Model,
    V2Error,
    V2QwenRuntime,
    _normalize_prefill_cache_seed,
    _QwenRuntimeOptions,
    _resolve_expert_mode,
)


def gguf_string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


class V2RuntimeTests(unittest.TestCase):
    def test_expert_mode_aliases_resolve_to_canonical_policies(self):
        self.assertEqual(_resolve_expert_mode("cpu"), ("cpu", 1, False))
        self.assertEqual(_resolve_expert_mode("auto"), ("auto", 2, False))
        self.assertEqual(
            _resolve_expert_mode("hybrid"), ("legacy-hybrid", 2, False)
        )
        self.assertEqual(
            _resolve_expert_mode("resident"), ("resident", 0, True)
        )
        self.assertEqual(
            _resolve_expert_mode("gpu"), ("legacy-paging", 0, False)
        )
        self.assertEqual(
            _resolve_expert_mode("legacy-paging"),
            ("legacy-paging", 0, False),
        )
        with self.assertRaisesRegex(ValueError, "expert_mode"):
            _resolve_expert_mode("accelerator")

    def test_expert_mode_and_legacy_name_must_agree(self):
        with self.assertRaisesRegex(ValueError, "different policies"):
            V2QwenRuntime(
                object(), expert_mode="auto", moe_device="cpu"
            )

    def test_expert_mode_aliases_build_identical_native_options(self):
        class FakeLibrary:
            def __init__(self):
                self.options = b""

            def colibri_v2_qwen_runtime_create(
                self, _model, options, _runtime
            ):
                self.options = ctypes.string_at(
                    options, ctypes.sizeof(_QwenRuntimeOptions)
                )
                return 0

        class FakeModel:
            def __init__(self):
                self._lib = FakeLibrary()
                self._handle = ctypes.c_void_p(1)

            def _check(self, status):
                self.assert_status = status

        def native_options(mode):
            model = FakeModel()
            runtime = V2QwenRuntime(model, expert_mode=mode)
            return model._lib.options, runtime

        auto_options, auto = native_options("auto")
        hybrid_options, hybrid = native_options("hybrid")
        legacy_hybrid_options, legacy_hybrid = native_options("legacy-hybrid")
        self.assertEqual(hybrid_options, legacy_hybrid_options)
        self.assertEqual(auto.expert_mode, "auto")
        self.assertEqual(hybrid.expert_mode, "legacy-hybrid")
        self.assertEqual(legacy_hybrid.expert_mode, "legacy-hybrid")

        resident_options, resident = native_options("resident")
        gpu_options, gpu = native_options("gpu")
        paging_options, paging = native_options("legacy-paging")
        self.assertEqual(gpu_options, paging_options)
        self.assertEqual(resident.expert_mode, "resident")
        self.assertEqual(gpu.expert_mode, "legacy-paging")
        self.assertEqual(paging.expert_mode, "legacy-paging")

        auto_native = _QwenRuntimeOptions.from_buffer_copy(auto_options)
        hybrid_native = _QwenRuntimeOptions.from_buffer_copy(hybrid_options)
        resident_native = _QwenRuntimeOptions.from_buffer_copy(resident_options)
        self.assertEqual(auto_native.moe_device, 2)
        self.assertEqual(auto_native.hybrid_prefill_cpu, 1)
        self.assertEqual(auto_native.immutable_residency, 1)
        self.assertEqual(auto_native.strict_resident, 0)
        self.assertEqual(hybrid_native.moe_device, 2)
        self.assertEqual(hybrid_native.prefill_cache_seed_auto, 0)
        self.assertEqual(hybrid_native.immutable_residency, 0)
        self.assertEqual(resident_native.moe_device, 0)
        self.assertEqual(resident_native.strict_resident, 1)

    def test_prefill_cache_seed_normalization(self):
        self.assertEqual(_normalize_prefill_cache_seed("auto"), (0, True))
        self.assertEqual(_normalize_prefill_cache_seed("off"), (0, False))
        self.assertEqual(_normalize_prefill_cache_seed(0), (0, False))
        self.assertEqual(_normalize_prefill_cache_seed(4), (4, False))
        for invalid in (-1, 257, "4", True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "prefill_cache_seed"):
                    _normalize_prefill_cache_seed(invalid)

    def test_hybrid_policies_are_validated_and_forwarded(self):
        model = object.__new__(V2Model)
        with patch("colibri_next.v2.V2QwenRuntime", return_value="runtime") as create:
            self.assertEqual(
                model.native_qwen_runtime(
                    hybrid_prefill="cpu", expert_residency="immutable"
                ),
                "runtime",
            )
            self.assertEqual(create.call_args.kwargs["hybrid_prefill"], "cpu")
            self.assertEqual(
                create.call_args.kwargs["expert_residency"], "immutable"
            )

        with self.assertRaisesRegex(ValueError, "hybrid_prefill"):
            V2QwenRuntime(object(), hybrid_prefill="invalid")
        with self.assertRaisesRegex(ValueError, "expert_residency"):
            V2QwenRuntime(object(), expert_residency="invalid")

    def test_nvfp4_prefill_has_blackwell_tensor_core_dispatch_and_fallback(self):
        root = Path(__file__).resolve().parents[1]
        kernels = (
            root / "native/include/colibri_v2_qwen_kernels.hpp"
        ).read_text()
        driver = (root / "native/src/gpu_driver.cpp").read_text()
        verifier = (root / "native/src/v2_mtp_verifier.inc").read_text()
        runtime = (root / "native/src/v2_runtime.cpp").read_text()

        self.assertIn("nvfp4_repack_cublaslt", kernels)
        self.assertIn("nvfp4_quantize_cublaslt", kernels)
        self.assertIn("nvfp4_repack_stacked_moe_cublaslt", kernels)
        self.assertIn("nvfp4_repack_concat_down_cublaslt", kernels)
        self.assertIn("nvfp4_copy_gguf_values_cublaslt", kernels)
        self.assertIn("blockIdx.x * 8 + warp", kernels)
        self.assertIn("COLIBRI_NVFP4_TILED", runtime)
        self.assertIn("CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3", kernels)
        self.assertIn("cublasLtMatmulAlgoGetHeuristic", driver)
        self.assertIn("colibri_gpu_nvfp4_matmul_cublas", driver)
        self.assertIn("colibri_gpu_nvfp4_moe_cublas", driver)
        self.assertIn("COLIBRI_NVFP4_TENSOR_CORE_PROFILE", driver)
        self.assertIn("COLIBRI_NVFP4_PERSISTENT", runtime)
        self.assertIn("colibri_gpu_nvfp4_moe_persistent", runtime)
        self.assertIn("COLIBRI_NVFP4_PERSISTENT_GROUPED", driver)
        self.assertIn("kPreferenceMaxWorkspace = 1", driver)
        self.assertIn("tc_env&&tc_env[0]=='1'", runtime)
        self.assertIn("COLIBRI_NVFP4_TENSOR_CORES", verifier)
        self.assertIn('launch("nvfp4_matmul_rows"', verifier)

    def test_q8_routed_experts_have_explicit_gpu_swiglu_dispatch(self):
        root = Path(__file__).resolve().parents[1]
        kernels = (
            root / "native/include/colibri_v2_qwen_kernels.hpp"
        ).read_text()
        driver = (root / "native/src/gpu_driver.cpp").read_text()
        runtime = (root / "native/src/v2_runtime.cpp").read_text()

        self.assertIn("void q8_grouped_swiglu(", kernels)
        self.assertIn("void q8_grouped_swiglu_rows(", kernels)
        self.assertIn('"q8_grouped_swiglu"', driver)
        self.assertIn('"q8_grouped_swiglu_rows"', driver)
        self.assertIn(
            'if(type==8)return rows?"q8_grouped_swiglu_rows":"q8_grouped_swiglu";',
            runtime,
        )

    def test_mtp_adaptive_trial_uses_warm_baseline_and_decisive_margin(self):
        """The gate must compare like with like, and reach a verdict.

        This test used to assert only the constants, so it passed while the
        gate was measuring its sequential baseline on the first 16 tokens after
        prefill -- the slowest tokens there are -- and MTP afterwards. That
        inflated the baseline enough to keep MTP while it ran ~30% slower.
        """
        root = Path(__file__).resolve().parents[1]
        runtime = (root / "native/src/v2_runtime.cpp").read_text()

        self.assertIn("kQwenMtpKeepPercent=80", runtime)
        self.assertIn(
            "mtp_per_token*100<baseline_per_token*kQwenMtpKeepPercent",
            runtime,
        )

        # The post-prefill ramp is discarded before either arm is timed.
        self.assertIn("kQwenMtpWarmupTokens=8", runtime)
        self.assertIn(
            "runtime.mtp_calibration_warmup_tokens<kQwenMtpWarmupTokens",
            runtime,
        )

        # Both arms are then sampled alternately, in the same thermal state.
        self.assertIn(
            "if(need_decode&&need_rounds)return runtime.mtp_calibration_draft_turn;",
            runtime,
        )

        # The verdict must be reachable from whichever recorder fills the last
        # slot; deciding only inside record_round wedged should_draft at false
        # with the verdict never reached.
        self.assertIn("qwen_mtp_finish_calibration", runtime)
        record_decode = runtime.split("static void qwen_mtp_record_decode")[1]
        record_decode = record_decode.split("\n}")[0]
        self.assertIn("qwen_mtp_finish_calibration(runtime);", record_decode)
        record_round = runtime.split("static void qwen_mtp_record_round")[1]
        record_round = record_round.split("\n}")[0]
        self.assertIn("qwen_mtp_finish_calibration(runtime);", record_round)

        # Both verdicts are logged: a gate that wrongly keeps MTP used to be
        # completely silent, which is the case that costs throughput.
        self.assertIn('keep?"keeping MTP":"falling back', runtime)

    def test_mtp_small_q8_batches_use_decode_matvecs(self):
        root = Path(__file__).resolve().parents[1]
        verifier = (root / "native/src/v2_mtp_verifier.inc").read_text()
        kernels = (
            root / "native/include/colibri_v2_qwen_kernels.hpp"
        ).read_text()

        self.assertIn("rows<=8&&type==8", verifier)
        self.assertIn('launch("q8_matvec_transposed_pair"', verifier)
        self.assertIn('launch("q8_matvec_transposed_triple"', verifier)
        self.assertIn("colibri_gpu_q8_matvec_transposed(", verifier)
        self.assertIn("q8_matvec_transposed_pair", kernels)
        self.assertIn("q8_matvec_transposed_triple", kernels)

    def test_mtp_prompt_prefill_keeps_the_batched_target_path(self):
        root = Path(__file__).resolve().parents[1]
        runtime = (root / "native/src/v2_runtime.cpp").read_text()
        verifier = (root / "native/src/v2_mtp_verifier.inc").read_text()
        workspace = (
            root / "native/include/colibri_v2_workspace.hpp"
        ).read_text()

        self.assertIn("if(runtime->prefill_rows>1&&prompt_count>1", runtime)
        self.assertNotIn(
            "runtime->prefill_rows>1&&!runtime->options.mtp_drafts", runtime
        )
        self.assertIn("mtp_prompt_hidden", workspace)
        self.assertIn("preserve_mtp_prompt_hidden", verifier)
        self.assertIn("qwen_mtp_append_pair(", verifier)

    def test_cuda_waits_default_to_blocking_context_scheduling(self):
        root = Path(__file__).resolve().parents[1]
        driver = (root / "native/src/gpu_driver.cpp").read_text()
        runtime = (root / "native/src/v2_runtime.cpp").read_text()

        self.assertIn("cuDevicePrimaryCtxSetFlags", driver)
        self.assertIn("cuCtxSetFlags", driver)
        self.assertIn("kCtxSchedBlockingSync = 0x04", driver)
        self.assertIn("COLIBRI_CUDA_SPIN_WAIT", driver)
        # gpu_info() retains the primary context before runtime initialization,
        # so the probe must set the same scheduling policy first.
        self.assertIn("cuDevicePrimaryCtxSetFlags", runtime)
        self.assertIn("set_flags(device,0x04)", runtime)

    def test_native_tiled_attention_covers_f16_bf16_and_q8(self):
        root = Path(__file__).resolve().parents[1]
        kernels = (
            root / "native/include/colibri_v2_qwen_kernels.hpp"
        ).read_text()
        driver = (root / "native/src/gpu_driver.cpp").read_text()
        runtime = (root / "native/src/v2_runtime.cpp").read_text()
        policy = (
            root / "native/include/colibri_v2_attention_policy.hpp"
        ).read_text()

        for precision in ("f16", "bf16", "q8"):
            symbol = f"kv_attention_fused_{precision}_tiles"
            self.assertIn(symbol, kernels)
            self.assertIn(symbol, driver)
            self.assertIn(symbol, runtime)
        self.assertIn("kv_fused_tiles_kernel(*runtime)", runtime)
        self.assertIn("runtime->fused_attention&&fused_tiles", runtime)
        self.assertIn("kDefaultCublasMinTokens = 128", policy)
        self.assertGreaterEqual(
            runtime.count("qwen_cublas_attention_eligible("), 3
        )
        self.assertNotIn("tokens>=4096", runtime)

    def test_sampled_topk_stays_on_gpu_until_candidates_are_reduced(self):
        root = Path(__file__).resolve().parents[1]
        kernels = (
            root / "native/include/colibri_v2_qwen_kernels.hpp"
        ).read_text()
        driver = (root / "native/src/gpu_driver.cpp").read_text()
        runtime = (root / "native/src/v2_runtime.cpp").read_text()
        self.assertIn("void sampling_block_topk_logits(", kernels)
        self.assertIn("void sampling_block_topk_pairs(", kernels)
        self.assertIn("cub::BlockRadixSort", kernels)
        self.assertIn("colibri_gpu_sampling_topk", driver)
        self.assertIn("sampling_gpu_topk_bytes", runtime)
        self.assertIn("sampling_nanoseconds", runtime)
        self.assertIn("COLIBRI_SAMPLING_GPU_TOPK", runtime)

    def test_turbo_cache_types_reach_the_native_options(self):
        class FakeLibrary:
            def __init__(self):
                self.options = b""

            def colibri_v2_qwen_runtime_create(
                self, _model, options, _runtime
            ):
                self.options = ctypes.string_at(
                    options, ctypes.sizeof(_QwenRuntimeOptions)
                )
                return 0

        class FakeModel:
            def __init__(self):
                self._lib = FakeLibrary()
                self._handle = ctypes.c_void_p(1)

            def _check(self, status):
                self.assert_status = status

        # The native side keys every KV kernel off these codes, so a rename or a
        # reordering here silently sends the runtime to the wrong codec.
        for name, code in (
            ("f32", 0), ("f16", 1), ("bf16", 2),
            ("q8_0", 3), ("turbo3", 4), ("turbo4", 5), ("auto", 6),
        ):
            with self.subTest(cache_type=name):
                model = FakeModel()
                V2QwenRuntime(model, cache_type_k=name, cache_type_v=name)
                native = _QwenRuntimeOptions.from_buffer_copy(model._lib.options)
                self.assertEqual(native.cache_type_k, code)
                self.assertEqual(native.cache_type_v, code)

        # K and V are independent, and turbo3 keys with turbo4 values is a
        # configuration the harness measurements pointed at.
        model = FakeModel()
        V2QwenRuntime(model, cache_type_k="turbo3", cache_type_v="turbo4")
        mixed = _QwenRuntimeOptions.from_buffer_copy(model._lib.options)
        self.assertEqual((mixed.cache_type_k, mixed.cache_type_v), (4, 5))

        for invalid in ("turbo", "turbo2", "turbo5", "fp8"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "cache_type"):
                    V2QwenRuntime(FakeModel(), cache_type_k=invalid)

        # The default is f16, NOT "auto". "auto" selects turbo4 above 32K, which
        # changes generated tokens in exactly the regime -- long context -- where
        # KV quantization damage concentrates, and no long-context quality
        # benchmark has been run. Flipping this default is a deliberate call that
        # needs that evidence first; llama.cpp and vLLM both ship unquantized KV.
        model = FakeModel()
        V2QwenRuntime(model)
        default = _QwenRuntimeOptions.from_buffer_copy(model._lib.options)
        self.assertEqual((default.cache_type_k, default.cache_type_v), (1, 1))

    def test_dense_requant_policy_reaches_native_options(self):
        class FakeLibrary:
            def __init__(self):
                self.options = b""

            def colibri_v2_qwen_runtime_create(
                self, _model, options, _runtime
            ):
                self.options = ctypes.string_at(
                    options, ctypes.sizeof(_QwenRuntimeOptions)
                )
                return 0

        class FakeModel:
            def __init__(self):
                self._lib = FakeLibrary()
                self._handle = ctypes.c_void_p(1)

            def _check(self, status):
                self.assert_status = status

        for name, code in (("auto", 0), ("q8", 1), ("off", 2)):
            with self.subTest(dense_requant=name):
                model = FakeModel()
                V2QwenRuntime(model, dense_requant=name)
                native = _QwenRuntimeOptions.from_buffer_copy(model._lib.options)
                self.assertEqual(native.dense_requant, code)

        with self.assertRaisesRegex(ValueError, "dense_requant"):
            V2QwenRuntime(FakeModel(), dense_requant="q4")

    def test_native_turbo_kv_path_is_wired_end_to_end(self):
        root = Path(__file__).resolve().parents[1]
        kernels = (
            root / "native/include/colibri_v2_qwen_kernels.hpp"
        ).read_text()
        driver = (root / "native/src/gpu_driver.cpp").read_text()
        runtime = (root / "native/src/v2_runtime.cpp").read_text()
        rows = (root / "native/src/v2_mtp_verifier.inc").read_text()

        # Every turbo kernel has to be defined, registered by name in the driver
        # lookup table, and reachable from a runtime dispatch. Missing the
        # driver table is the easy one to forget: it fails only at launch.
        for width in ("turbo3", "turbo4"):
            for symbol in (
                f"kv_store_{width}_k",
                f"kv_store_{width}_v",
                f"kv_attention_scores_{width}",
                f"kv_attention_scores_{width}_ring",
                f"kv_attention_values_{width}",
                f"kv_attention_values_{width}_ring",
                f"kv_dequant_{width}_f16",
            ):
                with self.subTest(symbol=symbol):
                    self.assertIn(symbol, kernels)
                    self.assertIn(symbol, driver)
                    self.assertIn(symbol, runtime)
        for symbol in ("turbo_rotate_rows", "turbo_unrotate_rows"):
            self.assertIn(symbol, kernels)
            self.assertIn(symbol, driver)
            self.assertIn(symbol, runtime)

        # Keys rotate under sign stream 0 and values under 1. If the two ever
        # agree, scores and the value fold silently use the wrong basis.
        self.assertIn("kv_store_turbo_impl<3>(current, cache, kv_heads, head_dim,"
                      " position, capacity, 0u)", kernels)
        self.assertIn("kv_store_turbo_impl<3>(current, cache, kv_heads, head_dim,"
                      " position, capacity, 1u)", kernels)
        self.assertIn("turbo_sign_d(d, 0u)", kernels)
        self.assertIn("turbo_sign_d(d, 1u)", kernels)

        # The staged prefill must take the ungated cuBLAS unpack: turbo values
        # are still rotated there, and the gate is a sigmoid, so gating before
        # the inverse rotation would be wrong.
        self.assertIn("qwen_attention_prefill_unpack", kernels)
        self.assertIn("qwen_attention_prefill_unpack_gate", kernels)
        self.assertIn("apply_gate", driver)
        self.assertIn("qwen_turbo_prefill_stage", runtime)
        self.assertIn("qwen_turbo_prefill_stage(", rows)
        self.assertIn("qwen_turbo_cublas_attention", runtime)

        # The Walsh-Hadamard butterfly needs a power-of-two head_dim, and the
        # rotated row is staged in a fixed shared-memory scratch.
        self.assertIn("TURBO_MAX_DIM 512", kernels)
        self.assertIn("head_dim&(head_dim-1)", runtime)
        self.assertIn("kv_type_is_turbo", runtime)

    def test_turbo_block_sizes_match_the_cpu_reference(self):
        root = Path(__file__).resolve().parents[1]
        reference = (root / "native/src/turboquant.h").read_text()
        runtime = (root / "native/src/v2_runtime.cpp").read_text()
        # 2 bytes of f16 scale plus 32 packed indices: 3.5 and 4.5 bits/value.
        # The arena sizing and the codec must agree or the cache overruns.
        self.assertIn("return 2 + turbo_bits(type) * kTurboBlock / 8;", reference)
        self.assertIn("kTurbo3BlockSize = 14", runtime)
        self.assertIn("kTurbo4BlockSize = 18", runtime)

    def make_model(
        self,
        sliding_pattern: tuple[bool, ...] | None = None,
        architecture: str = "qwen3moe",
        kv_heads: tuple[int, ...] | None = None,
        chat_template: str | None = None,
    ) -> tuple[Path, bytes]:
        prefix = architecture
        metadata_items = [
            gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string(architecture),
            gguf_string("general.name") + struct.pack("<I", 8) + gguf_string("test"),
            gguf_string(f"{prefix}.embedding_length") + struct.pack("<II", 4, 16),
            gguf_string(f"{prefix}.block_count") + struct.pack("<II", 4, len(sliding_pattern) if sliding_pattern else 1),
            gguf_string(f"{prefix}.attention.head_count") + struct.pack("<II", 4, 2),
            gguf_string(f"{prefix}.rope.dimension_count") + struct.pack("<II", 4, 64),
            gguf_string(f"{prefix}.full_attention_interval") + struct.pack("<II", 4, 4),
            gguf_string(f"{prefix}.attention.layer_norm_rms_epsilon") + struct.pack("<If", 6, 1e-6),
            gguf_string(f"{prefix}.rope.freq_base") + struct.pack("<If", 6, 10_000_000.0),
        ]
        if kv_heads is not None:
            metadata_items.append(
                gguf_string(f"{prefix}.attention.head_count_kv")
                + struct.pack("<IIQ", 9, 5, len(kv_heads))
                + struct.pack(f"<{len(kv_heads)}i", *kv_heads)
            )
        if chat_template is not None:
            metadata_items.append(
                gguf_string("tokenizer.chat_template")
                + struct.pack("<I", 8)
                + gguf_string(chat_template)
            )
        if sliding_pattern is not None:
            metadata_items.extend((
                gguf_string(f"{prefix}.attention.sliding_window") + struct.pack("<II", 4, 128),
                gguf_string(f"{prefix}.attention.sliding_window_pattern")
                + struct.pack("<IIQ", 9, 7, len(sliding_pattern))
                + bytes(sliding_pattern),
            ))
        metadata = b"".join(metadata_items)
        tensor = gguf_string("token_embd.weight") + struct.pack("<IQQIQ", 2, 2, 2, 0, 0)
        header = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata_items))
        body = header + metadata + tensor
        body += b"\0" * ((32 - len(body) % 32) % 32)
        payload = b"\x01\x02\x03\x04"
        handle = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        handle.write(body + payload)
        handle.close()
        return Path(handle.name), payload

    def test_reads_metadata_and_tensor_offsets(self):
        template = "{% for message in messages %}{{ message.content }}{% endfor %}"
        path, payload = self.make_model(chat_template=template)
        try:
            with V2Model(path) as model:
                self.assertEqual(model.info["architecture"], "qwen3moe")
                self.assertEqual(model.info["format"], "gguf")
                self.assertEqual(model.config["rotary_dimension"], 64)
                self.assertEqual(model.config["full_attention_interval"], 4)
                self.assertAlmostEqual(model.config["rms_norm_epsilon"], 1e-6)
                self.assertEqual(model.config["rope_freq_base"], 10_000_000.0)
                self.assertEqual(model.chat_template, template)
                model.validate_qwen()
                self.assertEqual(model.qwen_tensor("token_embeddings")["name"], "token_embd.weight")
                tensor = model.tensor("token_embd.weight")
                self.assertEqual(tensor["size"], len(payload))
                self.assertEqual(
                    model.read_tensor_slice("token_embd.weight", 1, 2),
                    payload[1:3],
                )
                self.assertEqual(
                    model.view_tensor_slice("token_embd.weight", 1, 2),
                    payload[1:3],
                )
        finally:
            path.unlink()

    def test_missing_chat_template_is_none(self):
        path, _ = self.make_model()
        try:
            with V2Model(path) as model:
                self.assertIsNone(model.chat_template)
        finally:
            path.unlink()

    @unittest.skipUnless(Path("/proc/self/fd").is_dir(), "requires procfs")
    def test_invalid_model_open_does_not_leak_file_descriptors(self):
        handle = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        handle.write(b"not a GGUF file")
        handle.close()
        path = Path(handle.name)
        try:
            before = len(os.listdir("/proc/self/fd"))
            for _ in range(32):
                with self.assertRaises(V2Error):
                    V2Model(path)
            after = len(os.listdir("/proc/self/fd"))
            self.assertLessEqual(after, before + 1)
        finally:
            path.unlink()

    def test_parses_generic_interleaved_sliding_window_pattern(self):
        path, _ = self.make_model((True, True, False, True, False, False))
        try:
            with V2Model(path) as model:
                config = model.config
                self.assertEqual(config["sliding_window"], 128)
                self.assertEqual(config["sliding_window_pattern_length"], 6)
                self.assertEqual(
                    config["attention_windows"], (128, 128, 0, 128, 0, 0)
                )
                self.assertEqual(
                    config["sliding_window_pattern"],
                    (True, True, False, True, False, False),
                )
        finally:
            path.unlink()

    def test_parses_gemma4_signed_per_layer_kv_head_array(self):
        pattern = (True, True, False)
        path, _ = self.make_model(pattern, architecture="gemma4", kv_heads=(8, 8, 2))
        try:
            with V2Model(path) as model:
                self.assertEqual(model.info["architecture"], "gemma4")
                self.assertEqual(model.config["attention_windows"], (128, 128, 0))
        finally:
            path.unlink()

    def test_generic_runtime_defaults_to_auto_for_gemma4(self):
        model = object.__new__(V2Model)
        with patch.object(
            V2Model, "info", new_callable=PropertyMock,
            return_value={"architecture": "gemma4"},
        ), patch.object(V2Model, "native_qwen_runtime", return_value="runtime") as create:
            self.assertEqual(model.native_runtime(context_limit=32), "runtime")
            create.assert_called_once_with(context_limit=32)

        with patch.object(
            V2Model, "info", new_callable=PropertyMock,
            return_value={"architecture": "gemma4"},
        ), patch.object(V2Model, "native_qwen_runtime", return_value="runtime") as create:
            self.assertEqual(
                model.native_runtime(
                    context_limit=32,
                    moe_device="hybrid",
                    parallel_sequences=2,
                    prompt_cache_mib=64,
                ),
                "runtime",
            )
            create.assert_called_once_with(
                context_limit=32,
                moe_device="hybrid",
                parallel_sequences=2,
                prompt_cache_mib=64,
            )

    def test_gemma4_decode_converts_word_boundary_markers_to_spaces(self):
        model = object.__new__(V2Model)
        model._architecture = "gemma4"
        pieces = {1: "Experts", 2: "▁(MoE)", 3: "▁works", 4: "."}
        with patch.object(V2Model, "token_text", lambda _model, token: pieces[token]):
            self.assertEqual(
                model.decode_tokens([1, 2, 3, 4]),
                "Experts (MoE) works.",
            )

    def test_tokenize_recognizes_non_pipe_qwen_control_tokens(self):
        model = object.__new__(V2Model)

        def token_id(_model, text):
            if text == "<think>":
                return 248068
            raise V2Error("not a control token")

        with patch.object(V2Model, "token_id", token_id), patch.object(
            V2Model, "_tokenize_plain", lambda _model, text, _capacity: [len(text)]
        ):
            # ``<think>`` is in the vocabulary and splits the run. ``<div>`` is
            # not, so it stays inside the surrounding plain run rather than
            # becoming a piece of its own: "b<div>c" is one run of 7.
            #
            # This previously asserted [1, 248068, 1, 5, 1], i.e. that an
            # unrecognised candidate was tokenized alone. That invents a piece
            # boundary the pre-tokenizer never produces and blocks merges
            # across it. Checked against the HF reference tokenizer on 3000
            # randomized strings: the old behaviour disagreed on 27, the new
            # one on none.
            self.assertEqual(
                model.tokenize("a<think>b<div>c"),
                [1, 248068, 7],
            )
