"""The RDNA3 WMMA prefill kernels (<format>_q8_mmq_wmma) against numpy.

Each kernel runs on random weights and activations and is compared with the
format dequantized on the host, the ggml reference layouts written out below.
One format per kernel shape: Q4_K (one scale per 32, sub-block minimum), Q2_K
(two scales and the minimum), Q6_K (two scales) and Q8_0 (one scale), on
shapes that leave partial row and token tiles.

The corpus is compiled with hipRTC and launched through libamdhip64 directly,
in a subprocess like test_rocm_backend's device tests: the GPU platform is
process-global. Skips without a gfx11 GPU.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r'''
import ctypes as C, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path("tools").resolve()))
import check_hip_corpus

def load(name):
    root = Path(os.environ.get("ROCM_PATH") or "/opt/rocm") / "lib"
    for candidate in (name, str(root / name)):
        try:
            return C.CDLL(candidate)
        except OSError:
            pass
    return None

hip, rtc = load("libamdhip64.so"), load("libhiprtc.so")
if hip is None or rtc is None or hip.hipSetDevice(0) != 0:
    print(json.dumps({"skip": "no HIP runtime or device"})); raise SystemExit
arch = C.create_string_buffer(1472)
hip.hipGetDevicePropertiesR0600(arch, 0)
name = arch.raw[1160:1416].split(b"\0")[0].split(b":")[0].decode()
if not name.startswith("gfx11"):
    print(json.dumps({"skip": f"{name or 'device'} is not RDNA3"})); raise SystemExit

source = check_hip_corpus.translation_unit().encode()
program = C.c_void_p()
rtc.hiprtcCreateProgram(C.byref(program), source, b"k.cu", 0, None, None)
options = (C.c_char_p * 2)(f"--offload-arch={name}".encode(), b"-O3")
assert rtc.hiprtcCompileProgram(program, 2, options) == 0, "hipRTC failed"
size = C.c_size_t(); rtc.hiprtcGetCodeSize(program, C.byref(size))
code = C.create_string_buffer(size.value); rtc.hiprtcGetCode(program, code)
module = C.c_void_p(); assert hip.hipModuleLoadData(C.byref(module), code) == 0

def device(array):
    pointer = C.c_void_p(); assert hip.hipMalloc(C.byref(pointer), max(array.nbytes, 4)) == 0
    hip.hipMemcpy(pointer, array.ctypes.data, array.nbytes, 1)
    return pointer

def half_bytes(value, shape):
    return np.frombuffer(np.full(shape, value, np.float16).tobytes(), np.uint8).reshape(*shape, 2)

def q4k(rng, rows, blocks):
    raw = rng.integers(0, 256, (rows, blocks, 144), dtype=np.uint8)
    raw[..., 0:2] = half_bytes(0.01, (rows, blocks)); raw[..., 2:4] = half_bytes(0.005, (rows, blocks))
    d = raw[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float64)
    dmin = raw[..., 2:4].copy().view(np.float16)[..., 0].astype(np.float64)
    sc = raw[..., 4:16].astype(np.int64); qs = raw[..., 16:].astype(np.int64)
    def scale_min(j):
        if j < 4:
            return sc[..., j] & 63, sc[..., j + 4] & 63
        return ((sc[..., j + 4] & 15) | ((sc[..., j - 4] >> 6) << 4),
                (sc[..., j + 4] >> 4) | ((sc[..., j] >> 6) << 4))
    out = np.zeros((rows, blocks, 256))
    for chunk in range(4):
        q = qs[..., chunk * 32:(chunk + 1) * 32]
        for half, values in ((0, q & 15), (1, q >> 4)):
            s, m = scale_min(2 * chunk + half)
            base = chunk * 64 + half * 32
            out[..., base:base + 32] = (d * s)[..., None] * values - (dmin * m)[..., None]
    return raw, out

def q2k(rng, rows, blocks):
    raw = rng.integers(0, 256, (rows, blocks, 84), dtype=np.uint8)
    raw[..., 80:82] = half_bytes(0.02, (rows, blocks)); raw[..., 82:84] = half_bytes(0.01, (rows, blocks))
    d = raw[..., 80:82].copy().view(np.float16)[..., 0].astype(np.float64)
    dmin = raw[..., 82:84].copy().view(np.float16)[..., 0].astype(np.float64)
    sc = raw[..., 0:16].astype(np.int64); qs = raw[..., 16:80].astype(np.int64)
    out = np.zeros((rows, blocks, 256)); index = 0
    for n in range(2):
        q = qs[..., n * 32:(n + 1) * 32]
        for shift in (0, 2, 4, 6):
            for part in range(2):
                s = sc[..., index]; index += 1
                values = (q[..., part * 16:(part + 1) * 16] >> shift) & 3
                base = n * 128 + shift // 2 * 32 + part * 16
                out[..., base:base + 16] = (d * (s & 15))[..., None] * values - (dmin * (s >> 4))[..., None]
    return raw, out

def q6k(rng, rows, blocks):
    raw = rng.integers(0, 256, (rows, blocks, 210), dtype=np.uint8)
    raw[..., 208:210] = half_bytes(0.001, (rows, blocks))
    d = raw[..., 208:210].copy().view(np.float16)[..., 0].astype(np.float64)
    ql = raw[..., 0:128].astype(np.int64); qh = raw[..., 128:192].astype(np.int64)
    sc = raw[..., 192:208].view(np.int8).astype(np.int64)
    out = np.zeros((rows, blocks, 256))
    for n in range(2):
        low, high, s = ql[..., n * 64:(n + 1) * 64], qh[..., n * 32:(n + 1) * 32], sc[..., n * 8:(n + 1) * 8]
        for l in range(32):
            i = l // 16
            for k, (nibbles, shift) in enumerate(((low[..., l] & 15, 0), (low[..., l + 32] & 15, 2),
                                                 (low[..., l] >> 4, 4), (low[..., l + 32] >> 4, 6))):
                q = (nibbles | (((high[..., l] >> shift) & 3) << 4)) - 32
                out[..., n * 128 + l + 32 * k] = d * s[..., i + 2 * k] * q
    return raw, out

def q80(rng, rows, blocks):
    raw = rng.integers(0, 256, (rows, blocks * 8, 34), dtype=np.uint8)
    raw[..., 0:2] = half_bytes(0.003, (rows, blocks * 8))
    d = raw[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float64)
    out = d[..., None] * raw[..., 2:].view(np.int8).astype(np.float64)
    return raw, out

results = {}
for fmt, build in (("q4k", q4k), ("q2k", q2k), ("q6k", q6k), ("q80", q80)):
    function = C.c_void_p()
    assert hip.hipModuleGetFunction(C.byref(function), module, f"{fmt}_q8_mmq_wmma".encode()) == 0
    worst = 0.0
    for seed, (width, outputs, tokens) in enumerate(((256, 16, 9), (1024, 200, 70), (2560, 384, 255))):
        rng = np.random.default_rng(seed)
        raw, weights = build(rng, outputs, width // 256)
        weights = weights.reshape(outputs, width)
        vectors = rng.integers(-127, 128, (tokens, width), dtype=np.int8)
        scale_stride = (width // 32) * 2
        scales = np.zeros((tokens, scale_stride), np.float16)
        scales[:, :width // 32] = rng.uniform(0.001, 0.02, (tokens, width // 32))
        activations = (vectors.reshape(tokens, -1, 32).astype(np.float64)
                       * scales[:, :width // 32].astype(np.float64)[..., None]).reshape(tokens, width)
        expected = activations @ weights.T
        result = np.full((tokens, outputs), np.nan, np.float32)
        buffers = [device(np.ascontiguousarray(raw)), device(vectors), device(scales), device(result)]
        values = buffers + [C.c_int(v) for v in (width, outputs, tokens, scale_stride)]
        arguments = (C.c_void_p * 8)(*[C.addressof(v) for v in values])
        assert hip.hipModuleLaunchKernel(function, (outputs + 127) // 128, (tokens + 63) // 64, 1,
                                         256, 1, 1, 0, None, arguments, None) == 0
        assert hip.hipDeviceSynchronize() == 0
        hip.hipMemcpy(result.ctypes.data, buffers[3], result.nbytes, 2)
        for pointer in buffers:
            hip.hipFree(pointer)
        worst = max(worst, float(np.nanmax(np.abs(result - expected)) / np.abs(expected).max())
                    if not np.isnan(result).any() else float("inf"))
    results[fmt] = worst
print(json.dumps({"errors": results}))
'''


def _run() -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(SCRIPT)], capture_output=True, text=True,
        timeout=900, cwd=ROOT)
    if completed.returncode != 0:
        raise AssertionError(f"subprocess failed:\n{completed.stderr[-4000:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


class RocmWmmaKernelTests(unittest.TestCase):
    def test_wmma_kernels_match_the_dequantized_product(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy is not installed")
        result = _run()
        if "skip" in result:
            self.skipTest(result["skip"])
        for fmt, error in result["errors"].items():
            with self.subTest(format=fmt):
                # Integer dots are exact; what is left is float32 scale folding.
                self.assertLess(error, 1e-5)


if __name__ == "__main__":
    unittest.main()
