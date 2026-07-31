import math
import struct
import unittest
from pathlib import Path

from colibri_next.q4 import Q4BlockTensor
from colibri_next.v2 import V2Error, V2Model
from colibri_next.v2_qwen import QwenMoELayer


def bf16_bytes(values):
    output = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        output.extend(struct.pack("<H", bits >> 16))
    return bytes(output)


class V2CudaParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() == 0:
                raise RuntimeError("no CUDA device")
            from colibri_next.cuda import _KERNEL_SOURCE
            V2Model.gpu_prepare(_KERNEL_SOURCE)
            cls.cp = cp
        except (ImportError, RuntimeError, V2Error) as error:
            raise unittest.SkipTest(str(error)) from error

    def test_v2_rms_norm_matches_cupy(self):
        cp = self.cp
        size = 37
        values = cp.asarray([math.sin(i * 0.07) for i in range(size)], dtype=cp.float32)
        weights = cp.asarray([math.cos(i * 0.03) * 0.1 for i in range(size)], dtype=cp.float32)
        output = cp.empty_like(values)
        V2Model.gpu_rms_norm(values.data.ptr, weights.data.ptr, output.data.ptr, size, 1e-5)
        cp.cuda.runtime.deviceSynchronize()
        expected = values / cp.sqrt(cp.mean(values * values) + cp.float32(1e-5)) * (1.0 + weights)
        cp.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)

    def test_v2_q4_matvec_matches_portable_reference(self):
        cp = self.cp
        shape = (7, 37)
        tensor = Q4BlockTensor.from_bf16(
            bf16_bytes([math.cos(index * 0.019) for index in range(math.prod(shape))]),
            shape,
            prefer_numpy=False,
        )
        vector = [math.sin(index * 0.11) for index in range(shape[1])]
        packed = cp.asarray(memoryview(tensor.packed), dtype=cp.uint8)
        scales = cp.asarray(memoryview(tensor.scales), dtype=cp.uint8).view(cp.float16)
        input_vector = cp.asarray(vector, dtype=cp.float32)
        output = cp.empty(shape[0], dtype=cp.float32)
        V2Model.gpu_q4_matvec(packed.data.ptr, scales.data.ptr, input_vector.data.ptr,
                              output.data.ptr, shape[0], shape[1])
        cp.cuda.runtime.deviceSynchronize()
        expected = cp.asarray(tensor.matvec(vector, prefer_numpy=False), dtype=cp.float32)
        cp.testing.assert_allclose(output, expected, rtol=2e-4, atol=2e-4)

    def test_v2_dense_projection_composes_norm_and_q4(self):
        cp = self.cp
        rows, columns = 7, 37
        tensor = Q4BlockTensor.from_bf16(
            bf16_bytes([math.cos(index * 0.019) for index in range(rows * columns)]),
            (rows, columns), prefer_numpy=False,
        )
        values = cp.asarray([math.sin(index * 0.07) for index in range(columns)], dtype=cp.float32)
        norm_weights = cp.asarray([math.cos(index * 0.03) * 0.1 for index in range(columns)], dtype=cp.float32)
        normalized = cp.empty_like(values)
        packed = cp.asarray(memoryview(tensor.packed), dtype=cp.uint8)
        scales = cp.asarray(memoryview(tensor.scales), dtype=cp.uint8).view(cp.float16)
        projection = cp.empty(rows, dtype=cp.float32)
        V2Model.gpu_dense_projection(
            values.data.ptr, norm_weights.data.ptr, normalized.data.ptr,
            packed.data.ptr, scales.data.ptr, projection.data.ptr,
            rows, columns, 1e-5,
        )
        cp.cuda.runtime.deviceSynchronize()
        expected_normalized = values / cp.sqrt(cp.mean(values * values) + cp.float32(1e-5)) * (1.0 + norm_weights)
        expected = cp.asarray(tensor.matvec(expected_normalized.tolist(), prefer_numpy=False), dtype=cp.float32)
        cp.testing.assert_allclose(normalized, expected_normalized, rtol=2e-5, atol=2e-5)
        cp.testing.assert_allclose(projection, expected, rtol=2e-4, atol=2e-4)

    def test_route_topk_matches_cupy(self):
        from colibri_next.cuda import CudaAccelerator

        accelerator = CudaAccelerator(cache_mib=64)
        try:
            logits = self.cp.asarray(
                [math.sin(index * 0.37) * 3.0 for index in range(256)],
                dtype=self.cp.float32,
            )
            selected, weights = accelerator.route_topk_device(logits, 8)
            expected_ids = self.cp.argsort(logits)[-8:][::-1]
            expected_logits = logits[expected_ids]
            expected_weights = self.cp.exp(
                expected_logits - self.cp.max(expected_logits)
            )
            expected_weights /= self.cp.sum(expected_weights)
            self.cp.testing.assert_array_equal(selected, expected_ids)
            self.cp.testing.assert_allclose(
                weights, expected_weights, rtol=2e-6, atol=2e-6
            )
        finally:
            accelerator.clear()

    def test_v2_dense_residual_adds_on_device(self):
        cp = self.cp
        rows = columns = 37
        tensor = Q4BlockTensor.from_bf16(
            bf16_bytes([math.cos(index * 0.019) for index in range(rows * columns)]),
            (rows, columns), prefer_numpy=False,
        )
        values = cp.asarray([math.sin(index * 0.07) for index in range(columns)], dtype=cp.float32)
        norm_weights = cp.asarray([math.cos(index * 0.03) * 0.1 for index in range(columns)], dtype=cp.float32)
        normalized = cp.empty_like(values)
        packed = cp.asarray(memoryview(tensor.packed), dtype=cp.uint8)
        scales = cp.asarray(memoryview(tensor.scales), dtype=cp.uint8).view(cp.float16)
        output = cp.empty(rows, dtype=cp.float32)
        V2Model.gpu_dense_residual(
            values.data.ptr, norm_weights.data.ptr, normalized.data.ptr,
            packed.data.ptr, scales.data.ptr, output.data.ptr,
            rows, columns, 1e-5,
        )
        cp.cuda.runtime.deviceSynchronize()
        expected_normalized = values / cp.sqrt(cp.mean(values * values) + cp.float32(1e-5)) * (1.0 + norm_weights)
        expected = cp.asarray(tensor.matvec(expected_normalized.tolist(), prefer_numpy=False), dtype=cp.float32)
        expected += values
        cp.testing.assert_allclose(output, expected, rtol=2e-4, atol=2e-4)

    def test_v2_grouped_attention_matches_reference(self):
        cp = self.cp
        heads, kv_heads, head_dim, tokens = 4, 2, 8, 5
        query = cp.asarray([math.sin(i * 0.13) for i in range(heads * head_dim)], dtype=cp.float32)
        keys = cp.asarray([math.cos(i * 0.07) for i in range(kv_heads * tokens * head_dim)], dtype=cp.float32)
        values = cp.asarray([math.sin(i * 0.11) for i in range(kv_heads * tokens * head_dim)], dtype=cp.float32)
        output = cp.empty(heads * head_dim, dtype=cp.float32)
        scale = 1.0 / math.sqrt(head_dim)
        V2Model.gpu_attention(query.data.ptr, keys.data.ptr, values.data.ptr, output.data.ptr,
                              heads, kv_heads, head_dim, tokens, scale)
        cp.cuda.runtime.deviceSynchronize()
        q = query.reshape(heads, head_dim)
        k = keys.reshape(kv_heads, tokens, head_dim)
        v = values.reshape(kv_heads, tokens, head_dim)
        expected_parts = []
        for head in range(heads):
            kv_head = head // (heads // kv_heads)
            scores = cp.sum(q[head][None, :] * k[kv_head], axis=1) * scale
            probabilities = cp.exp(scores - cp.max(scores))
            probabilities /= cp.sum(probabilities)
            expected_parts.append(cp.sum(probabilities[:, None] * v[kv_head], axis=0))
        expected = cp.stack(expected_parts).reshape(-1)
        cp.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)

    def test_v2_decoder_attention_step_updates_gpu_kv_cache(self):
        cp = self.cp
        hidden, heads, kv_heads, head_dim, capacity, position = 16, 4, 2, 4, 5, 2
        qkv_rows = (heads + 2 * kv_heads) * head_dim
        qkv_tensor = Q4BlockTensor.from_bf16(
            bf16_bytes([math.sin(i * 0.017) * 0.2 for i in range(qkv_rows * hidden)]),
            (qkv_rows, hidden), prefer_numpy=False,
        )
        out_tensor = Q4BlockTensor.from_bf16(
            bf16_bytes([math.cos(i * 0.013) * 0.15 for i in range(hidden * hidden)]),
            (hidden, hidden), prefer_numpy=False,
        )
        input_values = [math.sin(i * 0.07) for i in range(hidden)]
        norm_values = [math.cos(i * 0.03) * 0.1 for i in range(hidden)]
        input_device = cp.asarray(input_values, dtype=cp.float32)
        norm_device = cp.asarray(norm_values, dtype=cp.float32)
        normalized = cp.empty_like(input_device)
        qkv = cp.empty(qkv_rows, dtype=cp.float32)
        cache_keys = cp.zeros(kv_heads * capacity * head_dim, dtype=cp.float32)
        cache_values = cp.zeros_like(cache_keys)
        cache_keys[:2 * kv_heads * head_dim] = cp.asarray(
            [math.cos(i * 0.021) for i in range(2 * kv_heads * head_dim)], dtype=cp.float32
        )
        cache_values[:2 * kv_heads * head_dim] = cp.asarray(
            [math.sin(i * 0.023) for i in range(2 * kv_heads * head_dim)], dtype=cp.float32
        )
        attention_output = cp.empty(heads * head_dim, dtype=cp.float32)
        output = cp.empty(hidden, dtype=cp.float32)
        qkv_packed = cp.asarray(memoryview(qkv_tensor.packed), dtype=cp.uint8)
        qkv_scales = cp.asarray(memoryview(qkv_tensor.scales), dtype=cp.uint8).view(cp.float16)
        out_packed = cp.asarray(memoryview(out_tensor.packed), dtype=cp.uint8)
        out_scales = cp.asarray(memoryview(out_tensor.scales), dtype=cp.uint8).view(cp.float16)
        V2Model.gpu_decoder_attention_step(
            input_device.data.ptr, norm_device.data.ptr, normalized.data.ptr,
            qkv_packed.data.ptr, qkv_scales.data.ptr, qkv.data.ptr,
            cache_keys.data.ptr, cache_values.data.ptr, attention_output.data.ptr,
            out_packed.data.ptr, out_scales.data.ptr, output.data.ptr,
            hidden, heads, kv_heads, head_dim, position, capacity, 1e-5,
        )
        cp.cuda.runtime.deviceSynchronize()

        normalized_expected = cp.asarray(input_values, dtype=cp.float32)
        normalized_expected /= cp.sqrt(cp.mean(normalized_expected ** 2) + cp.float32(1e-5))
        normalized_expected *= 1.0 + cp.asarray(norm_values, dtype=cp.float32)
        qkv_expected = cp.asarray(qkv_tensor.matvec(normalized_expected.tolist(), prefer_numpy=False), dtype=cp.float32)
        q_rows = heads * head_dim
        k_rows = kv_heads * head_dim
        expected_keys = cache_keys.copy()
        expected_values = cache_values.copy()
        for kv_head in range(kv_heads):
            cache_offset = (kv_head * capacity + position) * head_dim
            current_offset = kv_head * head_dim
            expected_keys[cache_offset : cache_offset + head_dim] = qkv_expected[q_rows + current_offset : q_rows + current_offset + head_dim]
            expected_values[cache_offset : cache_offset + head_dim] = qkv_expected[q_rows + k_rows + current_offset : q_rows + k_rows + current_offset + head_dim]
        q = qkv_expected[:q_rows].reshape(heads, head_dim)
        k = expected_keys.reshape(kv_heads, capacity, head_dim)
        v = expected_values.reshape(kv_heads, capacity, head_dim)
        attention_parts = []
        for head in range(heads):
            kv_head = head // (heads // kv_heads)
            scores = cp.sum(q[head][None, :] * k[kv_head, : position + 1], axis=1) / math.sqrt(head_dim)
            probabilities = cp.exp(scores - cp.max(scores))
            probabilities /= cp.sum(probabilities)
            attention_parts.append(cp.sum(probabilities[:, None] * v[kv_head, : position + 1], axis=0))
        attention_expected = cp.stack(attention_parts).reshape(-1)
        output_expected = cp.asarray(out_tensor.matvec(attention_expected.tolist(), prefer_numpy=False), dtype=cp.float32) + input_device
        cp.testing.assert_allclose(cache_keys, expected_keys, rtol=2e-5, atol=2e-5)
        cp.testing.assert_allclose(cache_values, expected_values, rtol=2e-5, atol=2e-5)
        cp.testing.assert_allclose(attention_output, attention_expected, rtol=2e-4, atol=2e-4)
        cp.testing.assert_allclose(output, output_expected, rtol=2e-4, atol=2e-4)

    def test_grouped_q5_q6_experts_match_separate_kernels(self):
        model_path = Path("/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf")
        if not model_path.is_file():
            self.skipTest("real Qwen GGUF is unavailable")
        from colibri_next.cuda import CudaAccelerator

        accelerator = CudaAccelerator(cache_mib=512)
        try:
            with V2Model(model_path) as model:
                layer = QwenMoELayer(model, 0, cuda_only=True)
                vector = self.cp.linspace(-0.25, 0.25, layer.hidden, dtype=self.cp.float32)
                groups = []
                expected = self.cp.zeros(layer.hidden, dtype=self.cp.float32)
                weights = self.cp.asarray([0.37, 0.63], dtype=self.cp.float32)
                for index, expert in enumerate((0, 1)):
                    gate_raw = layer._expert_bytes(layer.expert_gate_info, expert)
                    up_raw = layer._expert_bytes(layer.expert_up_info, expert)
                    down_raw = layer._expert_bytes(layer.expert_down_info, expert)
                    groups.append((gate_raw, up_raw, down_raw))
                    gate = accelerator.q5k_matvec_transposed(
                        gate_raw, layer.hidden,
                        int(layer.expert_gate_info["shape"][1]), vector,
                        return_device=True,
                    )
                    up = accelerator.q5k_matvec_transposed(
                        up_raw, layer.hidden,
                        int(layer.expert_up_info["shape"][1]), vector,
                        return_device=True,
                    )
                    activated = gate / (
                        1.0 + self.cp.exp(-self.cp.clip(gate, -80.0, 80.0))
                    ) * up
                    down = accelerator.q6k_matvec_transposed(
                        down_raw, int(layer.expert_down_info["shape"][0]),
                        layer.hidden, activated, return_device=True,
                    )
                    expected += weights[index] * down
                output = self.cp.zeros(layer.hidden, dtype=self.cp.float32)
                accelerator.q5k_q6k_grouped_swiglu_accumulate(
                    groups, layer.hidden,
                    int(layer.expert_gate_info["shape"][1]), layer.hidden,
                    vector, output, weights,
                )
                self.cp.cuda.runtime.deviceSynchronize()
                self.cp.testing.assert_allclose(
                    output, expected, rtol=3e-4, atol=3e-4
                )
        finally:
            accelerator.clear()

    def test_grouped_q5_q8_experts_match_separate_kernels(self):
        model_path = Path("/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf")
        if not model_path.is_file():
            self.skipTest("real Qwen GGUF is unavailable")
        from colibri_next.cuda import CudaAccelerator

        accelerator = CudaAccelerator(cache_mib=512)
        try:
            with V2Model(model_path) as model:
                layer = QwenMoELayer(model, 34, cuda_only=True)
                vector = self.cp.linspace(
                    -0.25, 0.25, layer.hidden, dtype=self.cp.float32
                )
                groups = []
                expected = self.cp.zeros(layer.hidden, dtype=self.cp.float32)
                weights = self.cp.asarray([0.37, 0.63], dtype=self.cp.float32)
                for index, expert in enumerate((0, 1)):
                    gate_raw = layer._expert_bytes(layer.expert_gate_info, expert)
                    up_raw = layer._expert_bytes(layer.expert_up_info, expert)
                    down_raw = layer._expert_bytes(layer.expert_down_info, expert)
                    groups.append((gate_raw, up_raw, down_raw))
                    gate = accelerator.q5k_matvec_transposed(
                        gate_raw, layer.hidden,
                        int(layer.expert_gate_info["shape"][1]), vector,
                        return_device=True,
                    )
                    up = accelerator.q5k_matvec_transposed(
                        up_raw, layer.hidden,
                        int(layer.expert_up_info["shape"][1]), vector,
                        return_device=True,
                    )
                    activated = gate / (
                        1.0 + self.cp.exp(-self.cp.clip(gate, -80.0, 80.0))
                    ) * up
                    down = accelerator.q8_matvec_transposed(
                        down_raw, int(layer.expert_down_info["shape"][0]),
                        layer.hidden, activated, return_device=True,
                    )
                    expected += weights[index] * down
                output = self.cp.zeros(layer.hidden, dtype=self.cp.float32)
                accelerator.q5k_q6k_grouped_swiglu_accumulate(
                    groups, layer.hidden,
                    int(layer.expert_gate_info["shape"][1]), layer.hidden,
                    vector, output, weights, down_ggml_type=8,
                )
                self.cp.cuda.runtime.deviceSynchronize()
                self.cp.testing.assert_allclose(
                    output, expected, rtol=3e-4, atol=3e-4
                )
        finally:
            accelerator.clear()


if __name__ == "__main__":
    unittest.main()
