"""Greedy-decode a 128-dim K2-Horizon fixture and print the tokens as JSON.

Run as a subprocess by the attention-path parity tests: the MMA attention
switch (FLYWEIGHT_MMA_ATTENTION) is latched once per process, so every path
under test needs an interpreter of its own. Environment selects the path;
stderr carries the FLYWEIGHT_ATTENTION_DIAG announcement when asked for.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from flyweight.v2 import V2Model
from tests.k2_horizon_gguf_fixture import K2HorizonSpec, build_k2_horizon_gguf


def main() -> int:
    prompt_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    context = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    if not V2Model.gpu_info()["available"]:
        print("null")
        return 0
    directory = tempfile.mkdtemp(prefix="flyweight-k2-probe-")
    path = Path(directory) / "k2_horizon.gguf"
    build_k2_horizon_gguf(
        path, K2HorizonSpec(heads=8, kv_heads=2, head_dim=128, rotary_dim=64), seed=5
    )
    model = V2Model(path)
    runtime = model.native_runtime(
        context_limit=context, mtp_drafts=0, expert_mode="cpu",
        cache_type_k=os.environ.get("FLYWEIGHT_PROBE_KV", "f16"),
        cache_type_v=os.environ.get("FLYWEIGHT_PROBE_KV", "f16"),
    )
    runtime.prepare()
    try:
        produced: list[int] = []
        prompt = [8 + (i * 7) % 80 for i in range(prompt_tokens)]
        runtime.generate(prompt, steps, produced.append)
        print(json.dumps(produced))
    finally:
        runtime.close()
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
