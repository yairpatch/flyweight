"""Generate the native CUDA source header from the parity kernel corpus.

This is a source-generation utility, not a runtime dependency.  The emitted
header is compiled into colibri_v2 so production initialization and decode do
not import Python or CuPy.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: embed_qwen_kernels.py CUDA.PY OUTPUT.HPP")
    source_path, output_path = map(Path, sys.argv[1:])
    source = source_path.read_text()
    marker = '_KERNEL_SOURCE = r"""'
    start = source.index(marker) + len(marker)
    end = source.index('\n"""', start)
    cuda = source[start:end]
    # C++ raw-string delimiters are limited to 16 characters.
    delimiter = "COLIBRI_CUDA"
    if delimiter in cuda:
        raise RuntimeError("CUDA source contains the C++ raw-string delimiter")
    output_path.write_text(
        "#pragma once\n\n"
        "namespace colibri::v2 {\n"
        f"inline constexpr char qwen_cuda_source[] = R\"{delimiter}(\n"
        f"{cuda}\n"
        f"){delimiter}\";\n"
        "}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
