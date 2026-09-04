"""Phase 0 of plans/decode-device-dispatch.md: can the GPU read expert weights
straight out of registered host memory fast enough to make the miss path
device-side?

The proposed design lets a grouped expert kernel fall back to the CUDA-host-
registered mmap when an expert is not resident in the VRAM cache, which is what
removes the per-layer host round-trip. That is only worth building if the kernel
running against host memory is not catastrophically slower than the two things it
would replace: the same kernel against VRAM, and today's CPU expert path.

This drives the real `iq1s_grouped_swiglu` / `iq4nl_grouped_swiglu` kernels at the
checkpoint's real expert shape (2560 -> 640) through ctypes, with the weight
pointers aimed at either VRAM or a large registered host arena. The host arena is
deliberately much larger than the working set and the experts are scattered
through it, so the reads are as cache-hostile as the real thing.

    python bench_host_expert_reads.py
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

# Real qwen4exp expert geometry.
INPUT_SIZE = 2560
OUTPUT_SIZE = 640
TOP_K = 10           # experts dispatched per layer
HOST_ARENA_BYTES = 3 << 30   # 3 GiB, far past any cache
ITERATIONS = 200

# IQ block sizes: (bytes per 256-element super-block).
IQ_BLOCK_BYTES = {"iq1s": 50, "iq4nl": 144}


def _load():
    here = Path(__file__).resolve().parent
    lib = ctypes.CDLL(str(here / "src/flyweight/_native/flyweight_v2.so"))
    lib.flyweight_gpu_init.restype = ctypes.c_int
    lib.flyweight_gpu_init.argtypes = [ctypes.c_int]
    for name, args in (
        ("flyweight_gpu_alloc", [ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]),
        ("flyweight_gpu_free", [ctypes.c_uint64]),
        ("flyweight_gpu_host_register", [ctypes.c_void_p, ctypes.c_uint64]),
        ("flyweight_gpu_upload_sync", [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]),
        ("flyweight_gpu_stream_create", [ctypes.POINTER(ctypes.c_uint64)]),
        ("flyweight_gpu_stream_sync", [ctypes.c_uint64]),
        ("flyweight_gpu_event_create", [ctypes.POINTER(ctypes.c_uint64)]),
        ("flyweight_gpu_timed_event_create", [ctypes.POINTER(ctypes.c_uint64)]),
        ("flyweight_gpu_event_record", [ctypes.c_uint64, ctypes.c_uint64]),
        ("flyweight_gpu_event_sync", [ctypes.c_uint64]),
        ("flyweight_gpu_event_elapsed",
         [ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_float)]),
        ("flyweight_gpu_launch_named",
         [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
          ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p)]),
    ):
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = ctypes.c_int
    return lib


def _compile_kernels():
    """The kernel module is built by NVRTC when a runtime prepares, so borrow a
    tiny fixture to get it loaded; `flyweight_gpu_launch_named` then resolves the
    IQ kernels by name in this same process."""
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from flyweight.v2 import V2Model
    from tests.qwen4exp_gguf_fixture import build_qwen4exp_gguf

    holder = tempfile.TemporaryDirectory()
    path = Path(holder.name) / "tiny.gguf"
    build_qwen4exp_gguf(path)
    if V2Model.select_backend("cuda") != "cuda":
        return None
    model = V2Model(str(path))
    runtime = model.native_qwen_runtime(context_limit=64)
    runtime.prepare()
    return (holder, model, runtime)


def main() -> int:
    lib = _load()
    if lib.flyweight_gpu_init(0) != 0:
        print("skipped: no CUDA device")
        return 0
    keepalive = _compile_kernels()
    if keepalive is None:
        print("skipped: CUDA backend unavailable")
        return 0

    stream = ctypes.c_uint64()
    if lib.flyweight_gpu_stream_create(ctypes.byref(stream)) != 0:
        print("stream create failed")
        return 1
    start, stop = ctypes.c_uint64(), ctypes.c_uint64()
    lib.flyweight_gpu_timed_event_create(ctypes.byref(start))
    lib.flyweight_gpu_timed_event_create(ctypes.byref(stop))

    rng = np.random.default_rng(7)

    # Activation vector and the per-expert output, both device-side always.
    vector = np.ascontiguousarray(rng.standard_normal(INPUT_SIZE), dtype=np.float32)
    d_vector = ctypes.c_uint64()
    lib.flyweight_gpu_alloc(vector.nbytes, ctypes.byref(d_vector))
    lib.flyweight_gpu_upload_sync(d_vector, vector.ctypes.data, vector.nbytes)
    d_activated = ctypes.c_uint64()
    lib.flyweight_gpu_alloc(TOP_K * OUTPUT_SIZE * 4, ctypes.byref(d_activated))

    # One big registered host arena; experts are scattered through it so the
    # reads cannot be served by any cache that a compact layout would enjoy.
    host = np.zeros(HOST_ARENA_BYTES, dtype=np.uint8)
    host[:] = rng.integers(0, 256, size=(1 << 20,), dtype=np.uint8)[
        np.arange(HOST_ARENA_BYTES) % (1 << 20)]
    host_addr = host.ctypes.data
    reg = lib.flyweight_gpu_host_register(ctypes.c_void_p(host_addr), HOST_ARENA_BYTES)
    if reg != 0:
        print(f"host registration of {HOST_ARENA_BYTES>>30} GiB failed ({reg})")
        return 1
    print(f"registered {HOST_ARENA_BYTES>>30} GiB of host memory for device reads\n")

    print(f"{'kernel':8s} {'weights':10s} {'ms/launch':>10s} {'GB/s':>8s} {'vs VRAM':>9s}")
    print("-" * 50)

    results = {}
    for kernel, block_bytes in IQ_BLOCK_BYTES.items():
        matrix_bytes = INPUT_SIZE * OUTPUT_SIZE // 256 * block_bytes
        # gate+up per expert, read once per launch
        traffic = 2 * TOP_K * matrix_bytes

        # VRAM copy of the same weights.
        d_weights = ctypes.c_uint64()
        lib.flyweight_gpu_alloc(2 * TOP_K * matrix_bytes, ctypes.byref(d_weights))
        blob = np.frombuffer(host[: 2 * TOP_K * matrix_bytes], dtype=np.uint8)
        lib.flyweight_gpu_upload_sync(d_weights, blob.ctypes.data, blob.nbytes)

        for label in ("VRAM", "host mmap"):
            if label == "VRAM":
                ptrs = [int(d_weights.value) + i * matrix_bytes
                        for i in range(2 * TOP_K)]
            else:
                # Scatter across the whole arena, page-aligned, non-overlapping.
                span = HOST_ARENA_BYTES - matrix_bytes
                offsets = sorted(rng.choice(span // 4096, size=2 * TOP_K,
                                            replace=False) * 4096)
                ptrs = [host_addr + int(o) for o in offsets]

            table = np.array(ptrs, dtype=np.uint64)
            d_table = ctypes.c_uint64()
            lib.flyweight_gpu_alloc(table.nbytes, ctypes.byref(d_table))
            lib.flyweight_gpu_upload_sync(d_table, table.ctypes.data, table.nbytes)
            gate_table = ctypes.c_uint64(d_table.value)
            up_table = ctypes.c_uint64(d_table.value + TOP_K * 8)

            c_in = ctypes.c_int(INPUT_SIZE)
            c_out = ctypes.c_int(OUTPUT_SIZE)
            c_experts = ctypes.c_int(TOP_K)
            args = (ctypes.c_void_p * 7)(
                ctypes.cast(ctypes.byref(gate_table), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(up_table), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(d_vector), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(d_activated), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(c_in), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(c_out), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(c_experts), ctypes.c_void_p),
            )
            name = f"{kernel}_grouped_swiglu".encode()

            def launch():
                return lib.flyweight_gpu_launch_named(
                    name, OUTPUT_SIZE, TOP_K, 256, 0, stream, args)

            for _ in range(10):
                if launch() != 0:
                    print(f"  {kernel}: kernel launch failed (name not registered?)")
                    return 1
            lib.flyweight_gpu_stream_sync(stream)

            lib.flyweight_gpu_event_record(start, stream)
            for _ in range(ITERATIONS):
                launch()
            lib.flyweight_gpu_event_record(stop, stream)
            lib.flyweight_gpu_event_sync(stop)
            ms = ctypes.c_float()
            lib.flyweight_gpu_event_elapsed(start, stop, ctypes.byref(ms))
            per = ms.value / ITERATIONS
            gbs = traffic / (per / 1000) / 1e9
            results[(kernel, label)] = per
            ratio = (f"{per / results[(kernel, 'VRAM')]:.1f}x"
                     if label != "VRAM" else "-")
            print(f"{kernel:8s} {label:10s} {per:10.3f} {gbs:8.1f} {ratio:>9s}")
            lib.flyweight_gpu_free(d_table)
        lib.flyweight_gpu_free(d_weights)
        print()

    # What the design is competing against: today's CPU expert path measured on
    # the real model was 6.4-8.1 ms/token for ~30% of 480 routed experts, i.e.
    # roughly 0.13-0.17 ms per layer's miss set.
    print("reference: CPU expert path today ~0.13-0.17 ms per layer's miss set")
    print("           (6.4-8.1 ms/token over 48 layers, ~3 misses per layer)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
