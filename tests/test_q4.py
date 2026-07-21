import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from colibri_next.kernels import Q4SwiGLUExpert
from colibri_next.q4 import BLOCK_SIZE, Q4BlockTensor
from colibri_next.tensor_container import write_coli_tensor_file


def bf16_bytes(values: list[float]) -> bytes:
    output = bytearray()
    for value in values:
        float_bits = struct.unpack("<I", struct.pack("<f", value))[0]
        output.extend(struct.pack("<H", float_bits >> 16))
    return bytes(output)


def reference_matvec(matrix: list[float], shape: tuple[int, int], vector: list[float]):
    rows, columns = shape
    return [
        sum(matrix[row * columns + column] * vector[column] for column in range(columns))
        for row in range(rows)
    ]


def silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))


class Q4TensorTests(unittest.TestCase):
    def test_q4_compresses_bf16_and_supports_both_implementations(self) -> None:
        values = [math.sin(index * 0.19) for index in range(BLOCK_SIZE * 2)]
        source = bf16_bytes(values)
        numpy_tensor = Q4BlockTensor.from_bf16(source, (2, BLOCK_SIZE))
        python_tensor = Q4BlockTensor.from_bf16(
            source, (2, BLOCK_SIZE), prefer_numpy=False
        )

        self.assertLess(len(numpy_tensor.packed) + len(numpy_tensor.scales), len(source))
        self.assertEqual(numpy_tensor.packed, python_tensor.packed)
        for numpy_value, python_value in zip(
            numpy_tensor.dequantize(), python_tensor.dequantize(prefer_numpy=False)
        ):
            self.assertAlmostEqual(numpy_value, python_value, places=6)

    def test_matvec_numpy_and_python_paths_match(self) -> None:
        values = [math.cos(index * 0.07) * 0.25 for index in range(3 * BLOCK_SIZE)]
        tensor = Q4BlockTensor.from_bf16(bf16_bytes(values), (3, BLOCK_SIZE))
        vector = [math.sin(index * 0.11) for index in range(BLOCK_SIZE)]
        accelerated = tensor.matvec(vector)
        portable = tensor.matvec(vector, prefer_numpy=False)
        for accelerated_value, portable_value in zip(accelerated, portable):
            self.assertAlmostEqual(accelerated_value, portable_value, places=5)


class Q4SwiGLUKernelTests(unittest.TestCase):
    def test_cpu_override_does_not_dispatch_to_active_cuda(self) -> None:
        gate = Q4BlockTensor.from_bf16(
            bf16_bytes([0.1] * (BLOCK_SIZE * 2 * BLOCK_SIZE)),
            (BLOCK_SIZE * 2, BLOCK_SIZE),
        )
        down = Q4BlockTensor.from_bf16(
            bf16_bytes([0.2] * (BLOCK_SIZE * BLOCK_SIZE)),
            (BLOCK_SIZE, BLOCK_SIZE),
        )
        expert = Q4SwiGLUExpert(gate, down)
        hidden = [0.1] * BLOCK_SIZE

        class UnexpectedCudaDispatch:
            def q4_swiglu(self, *_args):
                raise AssertionError("CPU override dispatched to CUDA")

        with patch(
            "colibri_next.cuda.active_cuda",
            return_value=UnexpectedCudaDispatch(),
        ):
            actual = expert.forward(hidden, allow_cuda=False)

        expected = expert.forward(hidden, prefer_numpy=False)
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=6)

    def test_container_executes_quantized_swiglu_expert(self) -> None:
        hidden_size = BLOCK_SIZE
        intermediate_size = BLOCK_SIZE
        gate_shape = (intermediate_size * 2, hidden_size)
        down_shape = (hidden_size, intermediate_size)
        gate_values = [
            math.sin(index * 0.013) * 0.15
            for index in range(gate_shape[0] * gate_shape[1])
        ]
        down_values = [
            math.cos(index * 0.017) * 0.12
            for index in range(down_shape[0] * down_shape[1])
        ]
        gate = Q4BlockTensor.from_bf16(bf16_bytes(gate_values), gate_shape)
        down = Q4BlockTensor.from_bf16(bf16_bytes(down_values), down_shape)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expert.coli"
            write_coli_tensor_file(
                path,
                [*gate.payloads("gate_up_proj"), *down.payloads("down_proj")],
                metadata={
                    "quantization": {
                        "scheme": "q4_symmetric",
                        "block_size": BLOCK_SIZE,
                        "tensors": {
                            "gate_up_proj": gate.metadata(),
                            "down_proj": down.metadata(),
                        },
                    }
                },
            )
            expert = Q4SwiGLUExpert.from_file(path)
            hidden = [math.sin(index * 0.09) for index in range(hidden_size)]
            accelerated = expert.forward(hidden)
            portable = expert.forward(hidden, prefer_numpy=False)

        gate_output = reference_matvec(gate.dequantize(), gate_shape, hidden)
        activated = [
            silu(gate_output[index]) * gate_output[intermediate_size + index]
            for index in range(intermediate_size)
        ]
        expected = reference_matvec(down.dequantize(), down_shape, activated)
        for accelerated_value, portable_value, expected_value in zip(
            accelerated, portable, expected
        ):
            self.assertAlmostEqual(accelerated_value, portable_value, places=5)
            self.assertAlmostEqual(accelerated_value, expected_value, places=5)


if __name__ == "__main__":
    unittest.main()
