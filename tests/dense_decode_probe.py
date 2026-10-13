"""Greedy-decode a dense hybrid fixture of a given head geometry; print JSON.

The dense counterpart of k2_decode_probe: run as a subprocess by the decode
attention parity tests, because the MMA, cuBLAS and fused switches latch once
per process. Arguments are heads, kv_heads and head_dim, so one probe covers
every grouped-query shape the tensor-core kernels are instantiated for;
FLYWEIGHT_PROBE_KV picks the cache type and FLYWEIGHT_ATTENTION_DIAG on stderr
says which path ran.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from flyweight.v2 import V2Model
from tests.dense_gguf_fixture import DenseQwenSpec, build_dense_qwen35_gguf


def main() -> int:
    heads, kv_heads, head_dim = (int(value) for value in sys.argv[1:4])
    prompt_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    context = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
    steps = int(sys.argv[6]) if len(sys.argv) > 6 else 24
    if not V2Model.gpu_info()["available"]:
        print("null")
        return 0
    directory = tempfile.mkdtemp(prefix="flyweight-dense-probe-")
    path = Path(directory) / "dense.gguf"
    build_dense_qwen35_gguf(
        path, DenseQwenSpec(heads=heads, kv_heads=kv_heads, head_dim=head_dim),
        seed=5,
    )
    cache = os.environ.get("FLYWEIGHT_PROBE_KV", "f16")
    model = V2Model(path)
    runtime = model.native_runtime(
        context_limit=context, mtp_drafts=0,
        cache_type_k=cache, cache_type_v=cache,
    )
    runtime.prepare()
    try:
        produced: list[int] = []
        prompt = [3 + (i * 7) % 60 for i in range(prompt_tokens)]
        runtime.generate(prompt, steps, produced.append)
        print(json.dumps(produced))
    finally:
        runtime.close()
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
