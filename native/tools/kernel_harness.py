"""Compile and launch individual native CUDA kernels from Python.

The production runtime concatenates the two kernel headers and hands the result
to NVRTC through the driver API (see v2_runtime.cpp: `qwen_cuda_source +
qwen_native_cuda_source`).  This module reproduces that exact assembly and
compiles it with CuPy instead, so a single kernel can be run, checked against a
NumPy reference, and timed without standing up a model.

Development utility only: nothing in the serving path imports it, and CuPy stays
an optional dependency.
"""
from __future__ import annotations

import functools
from pathlib import Path

import cupy as cp

INCLUDE = Path(__file__).resolve().parent.parent / "include"
HEADERS = ("flyweight_v2_qwen_kernels.hpp", "flyweight_v2_native_kernels.hpp")

_OPEN = 'R"FLYWEIGHT_CUDA('
_CLOSE = ')FLYWEIGHT_CUDA"'


def _extract(header: Path) -> str:
    """Recover the CUDA text from a header of concatenated raw-string chunks.

    The headers split their source across several literals because MSVC caps a
    single string literal at 64 KiB.  The chunks are adjacent literals that the
    C++ compiler joins with nothing between them, so the split points must not
    introduce whitespace here either.
    """
    text = header.read_text()
    chunks = []
    position = text.index(_OPEN) + len(_OPEN)
    while True:
        end = text.index(_CLOSE, position)
        chunks.append(text[position:end])
        following = text.find(_OPEN, end)
        if following == -1:
            break
        position = following + len(_OPEN)
    return "".join(chunks)


@functools.lru_cache(maxsize=1)
def source() -> str:
    """The full kernel translation unit, in the runtime's concatenation order."""
    return "".join(_extract(INCLUDE / name) for name in HEADERS)


@functools.lru_cache(maxsize=2)
def module(extra: str = "") -> cp.RawModule:
    """Compile the kernel corpus, optionally with `extra` source appended.

    `extra` lets a kernel be prototyped against the compiled corpus before it is
    committed to a header.  NVRTC options mirror gpu_driver.cpp, minus the
    Windows CUDA_PATH probing.
    """
    device = cp.cuda.Device()
    major, minor = device.compute_capability[0], device.compute_capability[1:]
    options = (f"--gpu-architecture=compute_{major}{minor}", "--std=c++17")
    return cp.RawModule(code=source() + extra, options=options,
                        name_expressions=None, backend="nvrtc")


def kernel(name: str, extra: str = "") -> cp.RawKernel:
    return module(extra).get_function(name)


def _batch(fn, args, grid, block, shared, repeat):
    samples = []
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    for _ in range(repeat):
        start.record()
        fn(grid, block, args, shared_mem=shared)
        stop.record()
        stop.synchronize()
        samples.append(cp.cuda.get_elapsed_time(start, stop))
    samples.sort()
    return samples[len(samples) // 2]


def time_kernel(fn, *args, grid, block, shared=0, repeat=20, settle=0.03,
                budget=8.0) -> float:
    """Median milliseconds for one launch, measured at settled clocks.

    A idle laptop GPU boots this workload at roughly a sixth of its sustained
    clock and takes a second or more of continuous load to ramp, which is long
    enough to make a handful of warmup launches report a kernel as 6x slower
    than it is.  So batches run until two consecutive medians agree within
    `settle`, and the last one is reported.  `budget` caps the wall seconds
    spent ramping in case the clocks never settle.
    """
    import time

    deadline = time.monotonic() + budget
    previous = _batch(fn, args, grid, block, shared, repeat)
    while time.monotonic() < deadline:
        current = _batch(fn, args, grid, block, shared, repeat)
        if abs(current - previous) <= settle * max(current, previous):
            return current
        previous = current
    return previous


def settle_clocks(seconds=3.0) -> None:
    """Hold the GPU busy so a following measurement starts at boost clocks."""
    import time

    size = 2048
    a = cp.random.random((size, size), dtype=cp.float32)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for _ in range(10):
            a = a @ a * 1e-6
        cp.cuda.runtime.deviceSynchronize()
