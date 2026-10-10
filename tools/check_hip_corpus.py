"""Compile the kernel corpus with hipRTC for any AMD architecture, offline.

The ROCm counterpart of check_cuda_corpus.py. The runtime compiles for the
GPU it finds, so a change to the corpus or to flyweight_hip_prelude.hpp is
only proven on the card it was tested on -- and the prelude branches per
architecture (RDNA2's sdot4 against RDNA3/4's sudot4). hipRTC needs no GPU
to compile, so this checks every RDNA generation from one machine:

    python tools/check_hip_corpus.py                  # gfx1030 gfx1100 gfx1201
    python tools/check_hip_corpus.py gfx1036 gfx1151

Each target gets the same translation unit the runtime builds: the prelude,
then the qwen + native + diffusion corpus. hipRTC comes from the loader path
or ROCM_PATH (default /opt/rocm).
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADERS = (
    ROOT / "native/include/flyweight_v2_qwen_kernels.hpp",
    ROOT / "native/include/flyweight_v2_native_kernels.hpp",
    ROOT / "native/include/flyweight_v2_diffusion_kernels.hpp",
)
PRELUDE = ROOT / "native/include/flyweight_hip_prelude.hpp"
CHUNK = re.compile(r'R"FLYWEIGHT_CUDA\((.*?)\)FLYWEIGHT_CUDA"', re.DOTALL)
PRELUDE_CHUNK = re.compile(r'R"FLYWEIGHT_HIP\((.*?)\)FLYWEIGHT_HIP"', re.DOTALL)


def translation_unit() -> str:
    prelude = "".join(PRELUDE_CHUNK.findall(PRELUDE.read_text(encoding="utf-8")))
    return prelude + "".join(
        chunk for header in HEADERS for chunk in CHUNK.findall(header.read_text(encoding="utf-8"))
    )


def load_hiprtc() -> ctypes.CDLL:
    root = Path(os.environ.get("ROCM_PATH") or "/opt/rocm") / "lib"
    for name in ("libhiprtc.so.7", "libhiprtc.so.6", "libhiprtc.so", str(root / "libhiprtc.so")):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise SystemExit("no hipRTC found; install the HIP runtime or set ROCM_PATH")


def main(argv: list[str]) -> int:
    archs = argv or ["gfx1030", "gfx1100", "gfx1201"]
    hiprtc = load_hiprtc()
    hiprtc.hiprtcGetErrorString.restype = ctypes.c_char_p
    major, minor = ctypes.c_int(), ctypes.c_int()
    hiprtc.hiprtcVersion(ctypes.byref(major), ctypes.byref(minor))
    print(f"hipRTC {major.value}.{minor.value}")
    source = translation_unit().encode()
    failures = 0
    for arch in archs:
        program = ctypes.c_void_p()
        hiprtc.hiprtcCreateProgram(ctypes.byref(program), source, b"flyweight_kernels.cu",
                                   0, None, None)
        options = [f"--offload-arch={arch}", "-O3"]
        encoded = (ctypes.c_char_p * len(options))(*(o.encode() for o in options))
        result = hiprtc.hiprtcCompileProgram(program, len(options), encoded)
        size = ctypes.c_size_t()
        hiprtc.hiprtcGetProgramLogSize(program, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value + 1)
        hiprtc.hiprtcGetProgramLog(program, log)
        errors = [line for line in log.value.decode(errors="replace").splitlines()
                  if "error" in line]
        if result == 0:
            hiprtc.hiprtcGetCodeSize(program, ctypes.byref(size))
            print(f"[ok  ] {arch}: code object {size.value} bytes")
        else:
            failures += 1
            print(f"[FAIL] {arch}: {hiprtc.hiprtcGetErrorString(result).decode()}")
            for line in errors[:10]:
                print("       " + line)
        hiprtc.hiprtcDestroyProgram(ctypes.byref(program))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
