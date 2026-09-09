"""Compile the CUDA kernel corpus with NVRTC for any architecture, offline.

The runtime compiles the kernels at model open, for the GPU it finds. That
proves nothing about other cards: a Blackwell driver accepts what Ada rejects,
and a newer NVRTC knows headers an older one lacks. This assembles the same
source the runtime does, from the raw-string chunks in the kernel headers, and
runs NVRTC on it for each requested architecture, so a kernel change can be
checked for sm_89 or compute_90 without owning either.

    python tools/check_cuda_corpus.py               # sm_89 sm_120 compute_90
    python tools/check_cuda_corpus.py sm_86 compute_80

`sm_XY` asks for a cubin (what the runtime loads when NVRTC knows the device);
`compute_XY` asks for PTX (what it falls back to when NVRTC predates the GPU).
The CUDA headers come from CUDA_PATH, CUDA_HOME, /opt/cuda or /usr/local/cuda.
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
)
CHUNK = re.compile(r'R"FLYWEIGHT_CUDA\((.*?)\)FLYWEIGHT_CUDA"', re.DOTALL)


def corpus() -> str:
    return "".join(
        chunk for header in HEADERS for chunk in CHUNK.findall(header.read_text(encoding="utf-8"))
    )


def cuda_root() -> Path | None:
    for candidate in (os.environ.get("CUDA_PATH"), os.environ.get("CUDA_HOME"),
                      "/opt/cuda", "/usr/local/cuda"):
        if candidate and (Path(candidate) / "include" / "cuda_fp16.h").is_file():
            return Path(candidate)
    return None


def load_nvrtc(root: Path) -> ctypes.CDLL:
    if sys.platform == "win32":
        names = sorted((root / "bin").glob("nvrtc64_*_0.dll"), reverse=True)
    else:
        names = [root / "lib64" / "libnvrtc.so", root / "lib" / "libnvrtc.so"]
        names += sorted((root / "lib64").glob("libnvrtc.so.*"), reverse=True)
    for name in names:
        if name.is_file():
            return ctypes.CDLL(str(name))
    return ctypes.CDLL("libnvrtc.so" if sys.platform != "win32" else "nvrtc64_120_0.dll")


def main(argv: list[str]) -> int:
    archs = argv or ["sm_89", "sm_120", "compute_90"]
    root = cuda_root()
    if root is None:
        print("no CUDA headers found; set CUDA_PATH")
        return 2
    nvrtc = load_nvrtc(root)
    nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
    major, minor = ctypes.c_int(), ctypes.c_int()
    nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    print(f"NVRTC {major.value}.{minor.value}, headers {root / 'include'}")
    source = corpus().encode()
    failures = 0
    for arch in archs:
        program = ctypes.c_void_p()
        nvrtc.nvrtcCreateProgram(ctypes.byref(program), source, b"flyweight_kernels.cu",
                                 0, None, None)
        options = [f"--gpu-architecture={arch}", f"-I{root / 'include'}",
                   f"-I{root / 'include' / 'cccl'}"]
        encoded = (ctypes.c_char_p * len(options))(*(o.encode() for o in options))
        result = nvrtc.nvrtcCompileProgram(program, len(options), encoded)
        size = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value + 1)
        nvrtc.nvrtcGetProgramLog(program, log)
        text = log.value.decode(errors="replace")
        errors = [line for line in text.splitlines() if "error" in line]
        if result == 0:
            getter = "CUBIN" if arch.startswith("sm_") else "PTX"
            getattr(nvrtc, f"nvrtcGet{getter}Size")(program, ctypes.byref(size))
            print(f"[ok  ] {arch}: {getter} {size.value} bytes")
        else:
            failures += 1
            name = nvrtc.nvrtcGetErrorString(result).decode()
            print(f"[FAIL] {arch}: {name}")
            for line in errors[:10]:
                print("       " + line)
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
