"""Diagnose host->device expert paging bandwidth.

Measures achieved H2D throughput from the registered model mmap and compares it
against a truly-pinned buffer (cuMemAllocHost) and a pageable malloc buffer.
If the mmap tracks the pageable baseline rather than the pinned one, then
cuMemHostRegister is not actually page-locking the mapping on this platform.
"""

import ctypes
import mmap
import os
import sys
import time
from ctypes import POINTER, c_int32, c_uint64, c_void_p, byref

import numpy as np

DLL = r"C:\Users\thegr\OneDrive\Documents\flyweight\src\flyweight\_native\flyweight_v2.dll"
MODEL = r"C:\Users\thegr\Downloads\Qwen3.6-35B-A3B-UD-Q6_K.gguf"

MIB = 1024 * 1024
EXPERT = 2580480          # one Q6_K expert (gate+up+down)
ROLE = EXPERT // 3        # one role tensor, what the pager actually uploads
BULK = 256 * MIB          # large contiguous transfer -> link ceiling
WARM_WINDOW = 8 << 30     # pre-touched region for the random-access test


def load_lib():
    lib = ctypes.CDLL(DLL)
    lib.flyweight_gpu_init.argtypes = [c_int32]
    lib.flyweight_gpu_init.restype = c_int32
    lib.flyweight_gpu_alloc.argtypes = [c_uint64, POINTER(c_uint64)]
    lib.flyweight_gpu_host_alloc.argtypes = [c_uint64, POINTER(c_void_p)]
    lib.flyweight_gpu_host_register.argtypes = [c_void_p, c_uint64]
    lib.flyweight_gpu_upload.argtypes = [c_uint64, c_void_p, c_uint64, c_uint64]
    lib.flyweight_gpu_stream_create.argtypes = [POINTER(c_uint64)]
    lib.flyweight_gpu_stream_sync.argtypes = [c_uint64]
    return lib


def bandwidth(lib, dst, src_addr, chunk, count, stream, offsets=None, span=None):
    """Issue `count` uploads of `chunk` bytes and return GB/s.

    Reads walk the source region and wrap once they reach `span`, so a small
    source buffer can still serve an arbitrary number of chunks.
    """
    if offsets is None:
        room = max(1, (span or chunk) - chunk + 1)
        offsets = [((i * chunk) % room) & ~4095 for i in range(count)]
    lib.flyweight_gpu_stream_sync(stream)
    started = time.perf_counter()
    for i in range(count):
        off = int(offsets[i])
        rc = lib.flyweight_gpu_upload(dst, c_void_p(src_addr + off), chunk, stream)
        if rc != 0:
            raise RuntimeError(f"upload failed rc={rc}")
    lib.flyweight_gpu_stream_sync(stream)
    elapsed = time.perf_counter() - started
    return (chunk * count) / elapsed / 1e9, elapsed


def main():
    lib = load_lib()
    if lib.flyweight_gpu_init(0) != 0:
        sys.exit("flyweight_gpu_init failed")

    stream = c_uint64(0)
    lib.flyweight_gpu_stream_create(byref(stream))
    stream = stream.value

    dev = c_uint64(0)
    if lib.flyweight_gpu_alloc(BULK, byref(dev)) != 0:
        sys.exit("device alloc failed")
    dev = dev.value

    print("=== baselines ===")

    # True pinned memory, allocated by CUDA itself.
    pinned = c_void_p(0)
    if lib.flyweight_gpu_host_alloc(BULK, byref(pinned)) == 0:
        ctypes.memset(pinned, 1, BULK)
        gbs, _ = bandwidth(lib, dev, pinned.value, BULK, 4, stream, span=BULK)
        print(f"pinned   bulk {BULK//MIB:4d} MiB : {gbs:7.2f} GB/s")
        gbs, _ = bandwidth(lib, dev, pinned.value, EXPERT, 200, stream, span=BULK)
        print(f"pinned   expert  2.46 MiB : {gbs:7.2f} GB/s")
        gbs, _ = bandwidth(lib, dev, pinned.value, ROLE, 600, stream, span=BULK)
        print(f"pinned   role    0.82 MiB : {gbs:7.2f} GB/s")
    else:
        print("pinned host alloc failed")

    # Pageable malloc.
    pageable = ctypes.create_string_buffer(BULK)
    pg_addr = ctypes.addressof(pageable)
    gbs, _ = bandwidth(lib, dev, pg_addr, BULK, 4, stream, span=BULK)
    print(f"pageable bulk {BULK//MIB:4d} MiB : {gbs:7.2f} GB/s")
    gbs, _ = bandwidth(lib, dev, pg_addr, EXPERT, 200, stream, span=BULK)
    print(f"pageable expert  2.46 MiB : {gbs:7.2f} GB/s")

    # The real model mapping.
    print("\n=== model mmap ===")
    size = os.path.getsize(MODEL)
    print(f"model: {size/2**30:.1f} GiB")
    fh = open(MODEL, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    arr = np.frombuffer(mm, dtype=np.uint8)
    base = arr.ctypes.data
    print(f"mapped at 0x{base:x}")

    started = time.perf_counter()
    rc = lib.flyweight_gpu_host_register(c_void_p(base), size)
    reg_s = time.perf_counter() - started
    print(f"flyweight_gpu_host_register -> rc={rc} in {reg_s*1e3:.3f} ms")

    # Which flag actually succeeds? Probe the driver directly.
    try:
        cuda = ctypes.WinDLL("nvcuda.dll")
        cuda.cuMemHostRegister_v2.argtypes = [c_void_p, ctypes.c_size_t, ctypes.c_uint]
        cuda.cuMemHostUnregister.argtypes = [c_void_p]
        probe_len = 64 * MIB
        probe_at = base + (16 << 30)   # well inside the tensor region
        for flag, name in ((0x08, "READ_ONLY"), (0x01, "PORTABLE"), (0x00, "DEFAULT")):
            t = time.perf_counter()
            r = cuda.cuMemHostRegister_v2(c_void_p(probe_at), probe_len, flag)
            dt = time.perf_counter() - t
            print(f"  cuMemHostRegister(64 MiB, {name:9s}) -> CUresult={r} in {dt*1e3:8.3f} ms")
            if r == 0:
                cuda.cuMemHostUnregister(c_void_p(probe_at))
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"  direct driver probe unavailable: {exc}")

    # Warm the window so we measure PCIe, not disk.
    print(f"\nwarming {WARM_WINDOW>>30} GiB of the mapping...", flush=True)
    t = time.perf_counter()
    touched = int(arr[:WARM_WINDOW:4096].sum(dtype=np.uint64))
    print(f"warmed in {time.perf_counter()-t:.1f} s (checksum {touched})")

    gbs, _ = bandwidth(lib, dev, base, BULK, 4, stream, span=BULK)
    print(f"mmap     bulk {BULK//MIB:4d} MiB : {gbs:7.2f} GB/s")

    n = 600
    gbs, _ = bandwidth(lib, dev, base, ROLE, n, stream, span=WARM_WINDOW)
    print(f"mmap seq role    0.82 MiB : {gbs:7.2f} GB/s")

    # The decisive test: is the cost paid on the *first* DMA touch of a page?
    # Each region is disjoint, so "first" really is first. Same offsets are then
    # replayed to separate one-time locking cost from steady-state bandwidth.
    print("\n=== first DMA touch vs re-touch (random 2.46 MiB experts) ===")
    rng = np.random.default_rng(0)
    region = 4 << 30
    for label, lo in (("region A", 8 << 30), ("region B", 12 << 30), ("region C", 16 << 30)):
        off = (rng.integers(lo, lo + region - EXPERT, size=300) & ~4095).astype(np.int64)
        first, t1 = bandwidth(lib, dev, base, EXPERT, 300, stream, offsets=off)
        again, t2 = bandwidth(lib, dev, base, EXPERT, 300, stream, offsets=off)
        print(f"{label}: first touch {first:6.2f} GB/s ({t1*1e3:7.1f} ms)"
              f"   re-touch {again:6.2f} GB/s ({t2*1e3:6.1f} ms)"
              f"   ratio {again/first:5.1f}x")

    # What this implies for decode.
    print("\n=== implied ===")
    per_token = 23.05 * EXPERT
    print(f"decode streams 23.05 misses/token x 2.46 MiB = {per_token/1e6:.1f} MB/token")
    print(f"  at 27 GB/s (re-touch)   : {per_token/27e9*1e3:6.2f} ms/token")
    print(f"  observed expert bucket  :  15.80 ms/token")


if __name__ == "__main__":
    main()
