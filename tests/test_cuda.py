import math
import struct
import unittest

from colibri_next.bf16 import BF16Tensor
from colibri_next.cuda import (
    CudaUnavailableError,
    _group_selected_experts,
    _attention_prefill_chunk_size,
    _use_batched_attention_prefill,
    _use_expert_major_prefill,
    configure_cuda,
    disable_cuda,
)
from colibri_next.kernels import Q4SwiGLUExpert
from colibri_next.q4 import Q4BlockTensor


def bf16_bytes(values: list[float]) -> bytes:
    output = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        output.extend(struct.pack("<H", bits >> 16))
    return bytes(output)


class ExpertGroupingTests(unittest.TestCase):
    def test_groups_token_routes_by_expert(self) -> None:
        self.assertEqual(
            _group_selected_experts([[3, 7], [7, 2], [3, 2]]),
            {
                3: [(0, 0), (2, 0)],
                7: [(0, 1), (1, 0)],
                2: [(1, 1), (2, 1)],
            },
        )

    def test_adaptive_dispatch_uses_long_prompt_batching(self) -> None:
        self.assertFalse(_use_expert_major_prefill(64))
        self.assertTrue(_use_expert_major_prefill(128))

    def test_long_attention_prefill_avoids_quadratic_batch(self) -> None:
        self.assertTrue(_use_batched_attention_prefill(512))
        self.assertFalse(_use_batched_attention_prefill(513))
        self.assertEqual(_attention_prefill_chunk_size(), 256)


class CudaKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.accelerator = configure_cuda(cache_mib=64)
        except CudaUnavailableError as error:
            raise unittest.SkipTest(str(error)) from error

    @classmethod
    def tearDownClass(cls) -> None:
        disable_cuda()

    def test_bf16_matvec_matches_cpu(self) -> None:
        shape = (5, 37)
        values = [math.sin(index * 0.031) for index in range(math.prod(shape))]
        vector = [math.cos(index * 0.07) for index in range(shape[1])]
        tensor = BF16Tensor(shape, bf16_bytes(values))
        expected = tensor.matvec(vector, prefer_numpy=False)
        actual = tensor.matvec(vector)
        for expected_value, actual_value in zip(expected, actual):
            self.assertAlmostEqual(expected_value, actual_value, places=4)

    def test_q8_transposed_matvec_matches_gguf_layout(self) -> None:
        input_size, output_size = 5, 7
        quantized = [((index * 3) % 17) - 8 for index in range(input_size * output_size)]
        packed = bytearray()
        for start in range(0, len(quantized), 32):
            block = quantized[start : start + 32]
            packed.extend(struct.pack("<e", 1.0))
            packed.extend(struct.pack("<32b", *(block + [0] * (32 - len(block)))))
        vector = [math.sin(index * 0.17) for index in range(input_size)]
        actual = self.accelerator.q8_matvec_transposed(
            bytes(packed), input_size, output_size, vector
        )
        expected = [
            sum(quantized[output * input_size + input] * vector[input] for input in range(input_size))
            for output in range(output_size)
        ]
        for expected_value, actual_value in zip(expected, actual):
            self.assertAlmostEqual(expected_value, actual_value, places=4)

    def test_batched_q4_moe_matches_weighted_portable_experts(self) -> None:
        hidden_size = 32
        intermediate_size = 32
        gate_shape = (intermediate_size * 2, hidden_size)
        down_shape = (hidden_size, intermediate_size)
        gate_values = [
            math.sin(index * 0.021) * 0.16
            for index in range(math.prod(gate_shape))
        ]
        down_values = [
            math.cos(index * 0.015) * 0.13
            for index in range(math.prod(down_shape))
        ]
        expert = Q4SwiGLUExpert(
            Q4BlockTensor.from_bf16(bf16_bytes(gate_values), gate_shape),
            Q4BlockTensor.from_bf16(bf16_bytes(down_values), down_shape),
        )
        hidden = [math.cos(index * 0.08) for index in range(hidden_size)]
        portable = expert.forward(hidden, prefer_numpy=False)
        previous_q8 = self.accelerator.q8_moe_enabled
        self.accelerator.q8_moe_enabled = True
        actual = self.accelerator.q4_moe(
            [expert, expert],
            [0.3, 0.7],
            expert,
            0.4,
            hidden,
        )
        self.accelerator.q8_moe_enabled = previous_q8
        expected = [value * 1.4 for value in portable]
        for expected_value, actual_value in zip(expected, actual):
            self.assertAlmostEqual(expected_value, actual_value, delta=0.01)
        self.assertGreater(self.accelerator.q8_grouped_moe_calls, 0)

    def test_q4_expert_sequence_matches_individual_execution(self) -> None:
        hidden_size = 32
        intermediate_size = 32
        gate_shape = (intermediate_size * 2, hidden_size)
        down_shape = (hidden_size, intermediate_size)
        expert = Q4SwiGLUExpert(
            Q4BlockTensor.from_bf16(
                bf16_bytes(
                    [
                        math.sin(index * 0.017) * 0.2
                        for index in range(math.prod(gate_shape))
                    ]
                ),
                gate_shape,
            ),
            Q4BlockTensor.from_bf16(
                bf16_bytes(
                    [
                        math.cos(index * 0.023) * 0.15
                        for index in range(math.prod(down_shape))
                    ]
                ),
                down_shape,
            ),
        )
        vectors = [
            [math.cos((token + 1) * (index + 1) * 0.03) for index in range(hidden_size)]
            for token in range(3)
        ]
        actual = self.accelerator._q4_swiglu_sequence_device(
            expert,
            self.accelerator.cp.asarray(vectors, dtype=self.accelerator.cp.float32),
            protect_weights=False,
        ).get()
        for token, vector in enumerate(vectors):
            expected = expert.forward(vector, prefer_numpy=False)
            for expected_value, actual_value in zip(expected, actual[token]):
                self.assertAlmostEqual(expected_value, float(actual_value), places=3)

    def test_packed_q4_matvec_matches_cpu(self) -> None:
        shape = (7, 37)
        values = [math.cos(index * 0.019) for index in range(math.prod(shape))]
        vector = [math.sin(index * 0.11) for index in range(shape[1])]
        tensor = Q4BlockTensor.from_bf16(bf16_bytes(values), shape)
        expected = tensor.matvec(vector, prefer_numpy=False)
        actual = tensor.matvec(vector)
        for expected_value, actual_value in zip(expected, actual):
            self.assertAlmostEqual(expected_value, actual_value, places=4)

    def test_async_q4_prefetch_matches_cpu(self) -> None:
        self.accelerator.clear()
        shape = (7, 37)
        values = [math.sin(index * 0.029) for index in range(math.prod(shape))]
        vector = [math.cos(index * 0.13) for index in range(shape[1])]
        tensor = Q4BlockTensor.from_bf16(bf16_bytes(values), shape)
        expected = tensor.matvec(vector, prefer_numpy=False)
        requests = self.accelerator.expert_prefetch_requests
        previous = self.accelerator.expert_prefetch_enabled
        self.accelerator.expert_prefetch_enabled = True
        try:
            self.assertTrue(self.accelerator.prefetch_q4(tensor))
            actual = tensor.matvec(vector)
            self.assertEqual(
                self.accelerator.expert_prefetch_requests, requests + 1
            )
            self.assertFalse(self.accelerator.prefetch_q4(tensor))
        finally:
            self.accelerator.expert_prefetch_enabled = previous
        for expected_value, actual_value in zip(expected, actual):
            self.assertAlmostEqual(expected_value, actual_value, places=4)

    def test_bf16_matvec_many_matches_individual_cpu_results(self) -> None:
        columns = 37
        tensors = []
        for rows, frequency in ((3, 0.017), (5, 0.023), (2, 0.031)):
            values = [
                math.sin(index * frequency)
                for index in range(rows * columns)
            ]
            tensors.append(BF16Tensor((rows, columns), bf16_bytes(values)))
        vector = [math.cos(index * 0.07) for index in range(columns)]
        expected = [
            tensor.matvec(vector, prefer_numpy=False) for tensor in tensors
        ]
        actual = self.accelerator.bf16_matvec_many(tensors, vector)
        for expected_output, actual_output in zip(expected, actual):
            for expected_value, actual_value in zip(
                expected_output, actual_output
            ):
                self.assertAlmostEqual(expected_value, actual_value, places=4)

    def test_fused_q4_swiglu_matches_portable_path(self) -> None:
        hidden_size = 32
        intermediate_size = 32
        gate_shape = (intermediate_size * 2, hidden_size)
        down_shape = (hidden_size, intermediate_size)
        gate_values = [
            math.sin(index * 0.017) * 0.2
            for index in range(math.prod(gate_shape))
        ]
        down_values = [
            math.cos(index * 0.013) * 0.15
            for index in range(math.prod(down_shape))
        ]
        expert = Q4SwiGLUExpert(
            Q4BlockTensor.from_bf16(bf16_bytes(gate_values), gate_shape),
            Q4BlockTensor.from_bf16(bf16_bytes(down_values), down_shape),
        )
        hidden = [math.sin(index * 0.09) for index in range(hidden_size)]
        expected = expert.forward(hidden, prefer_numpy=False)
        actual = expert.forward(hidden)
        for expected_value, actual_value in zip(expected, actual):
            self.assertAlmostEqual(expected_value, actual_value, places=4)

    def test_protected_weights_survive_dynamic_eviction(self) -> None:
        original_limit = self.accelerator.cache_limit_bytes
        try:
            self.accelerator.clear()
            self.accelerator.cache_limit_bytes = 128
            protected_owner = object()
            first_dynamic_owner = object()
            second_dynamic_owner = object()
            upload = lambda: (self.accelerator.cp.empty(1),)
            self.accelerator._cached_arrays(
                "static", protected_owner, 64, upload, protected=True
            )
            self.accelerator._cached_arrays(
                "dynamic", first_dynamic_owner, 64, upload
            )
            self.accelerator._cached_arrays(
                "dynamic", second_dynamic_owner, 64, upload
            )
            keys = set(self.accelerator._cache)
            self.assertIn(("static", id(protected_owner)), keys)
            self.assertNotIn(("dynamic", id(first_dynamic_owner)), keys)
            self.assertIn(("dynamic", id(second_dynamic_owner)), keys)
        finally:
            self.accelerator.clear()
            self.accelerator.cache_limit_bytes = original_limit

    def test_evicted_weight_storage_is_reused_safely(self) -> None:
        original_limit = self.accelerator.cache_limit_bytes
        try:
            self.accelerator.clear()
            self.accelerator.cache_limit_bytes = 128 * 1024
            shape = (128, 512)
            vector = [math.sin(index * 0.03) for index in range(shape[1])]
            tensors = [
                BF16Tensor(
                    shape,
                    bf16_bytes([
                        math.cos((index + offset) * 0.007)
                        for index in range(math.prod(shape))
                    ]),
                )
                for offset in range(3)
            ]
            expected = [tensor.matvec(vector, prefer_numpy=False) for tensor in tensors]
            actual = [tensor.matvec(vector) for tensor in tensors]
            self.assertGreaterEqual(self.accelerator.cache_evictions, 2)
            for expected_output, actual_output in zip(expected, actual):
                for expected_value, actual_value in zip(expected_output, actual_output):
                    self.assertAlmostEqual(expected_value, actual_value, places=3)
        finally:
            self.accelerator.clear()
            self.accelerator.cache_limit_bytes = original_limit

    def test_weight_uploads_are_cached(self) -> None:
        tensor = BF16Tensor((2, 2), bf16_bytes([1.0, 2.0, 3.0, 4.0]))
        misses = self.accelerator.cache_misses
        hits = self.accelerator.cache_hits
        tensor.matvec([1.0, 1.0])
        tensor.matvec([1.0, 1.0])
        self.assertEqual(self.accelerator.cache_misses, misses + 1)
        self.assertEqual(self.accelerator.cache_hits, hits + 1)


if __name__ == "__main__":
    unittest.main()
