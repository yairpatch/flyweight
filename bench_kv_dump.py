"""Dump real KV caches and score TurboQuant bit widths against them.

Fills the cache with a real prompt, dumps each attention layer's live window
via ColibriV2QwenRuntime.dump_kv, then runs native/tools/bench_turboquant over
each dump. What matters in the output is the K/V norm ratio (which decides
whether keys need a wider allocation than values) and the score/output error
per bit width.

Attention layers are found by probing: dump_kv rejects Gated DeltaNet layers,
which carry recurrent state and no KV cache at all, so those are skipped.

Run:
    PYTHONPATH=src python3 bench_kv_dump.py
Env knobs:
    COLIBRI_MODEL      GGUF path (default Qwen3.6-35B-A3B-UD-Q5_K_M.gguf)
    COLIBRI_MOE_DEVICE gpu|cpu|hybrid (default hybrid)
    COLIBRI_GPU_CACHE_MIB  total GPU budget in MiB (default 8192)
    COLIBRI_CONTEXT    context window (default 4096)
    COLIBRI_CACHE_TYPE f32|f16|bf16|q8_0 (default f16, the runtime default)
    COLIBRI_PROMPT_TOKENS  how much cache to fill (default 1024)
    COLIBRI_KV_LAYERS  comma list of layers to dump (default: first 3 found)
    COLIBRI_KV_BENCH   path to the bench_turboquant binary
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from colibri_next.v2 import V2Model, V2Error

MODEL = os.environ.get(
    "COLIBRI_MODEL",
    "/home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf",
)
MOE_DEVICE = os.environ.get("COLIBRI_MOE_DEVICE", "hybrid")
GPU_CACHE_MIB = int(os.environ.get("COLIBRI_GPU_CACHE_MIB", "8192"))
CONTEXT = int(os.environ.get("COLIBRI_CONTEXT", "4096"))
CACHE_TYPE = os.environ.get("COLIBRI_CACHE_TYPE", "f16")
PROMPT_TOKENS = int(os.environ.get("COLIBRI_PROMPT_TOKENS", "1024"))
LAYERS = [int(x) for x in os.environ.get("COLIBRI_KV_LAYERS", "").split(",") if x]
BASELINE_BITS = {"f32": "32", "f16": "16", "bf16": "16", "q8_0": "8.5"}

SENTENCE = (
    "Mixture-of-experts models route each token to a small subset of experts, "
    "which saves compute compared with a dense network of equal capacity. "
)


def find_bench() -> str | None:
    override = os.environ.get("COLIBRI_KV_BENCH")
    if override:
        return override if Path(override).exists() else None
    found = shutil.which("colibri_turboquant_bench")
    if found:
        return found
    for candidate in Path(__file__).parent.rglob("colibri_turboquant_bench"):
        if candidate.is_file():
            return str(candidate)
    return None


def main() -> int:
    bench = find_bench()
    if bench is None:
        print(
            "colibri_turboquant_bench not found. Build it with:\n"
            "  cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build build/native --target colibri_turboquant_bench",
            file=sys.stderr,
        )
        return 1

    model = V2Model(MODEL)
    try:
        print(f"model={MODEL}", file=sys.stderr, flush=True)
        print(
            f"moe_device={MOE_DEVICE} gpu_cache_mib={GPU_CACHE_MIB} "
            f"context={CONTEXT} cache_type={CACHE_TYPE}",
            file=sys.stderr,
            flush=True,
        )
        rt = model.native_qwen_runtime(
            context_limit=CONTEXT,
            gpu_cache_bytes=GPU_CACHE_MIB * 1024 * 1024,
            moe_device=MOE_DEVICE,
            cache_type_k=CACHE_TYPE,
            cache_type_v=CACHE_TYPE,
        )
        rt.prepare()

        info = rt.info
        total = int(info["layers"])
        print(
            f"layers={total} attention={info['attention_layers']} "
            f"deltanet={info['deltanet_layers']} swa={info['swa_layers']}",
            file=sys.stderr,
            flush=True,
        )

        # Fill the cache with real content: the norm statistics that drive the
        # bit allocation are a property of the activations, so a synthetic or
        # repeated-token prompt would not measure the right thing.
        base = model.tokenize(SENTENCE)
        prompt = (base * (PROMPT_TOKENS // max(1, len(base)) + 1))[:PROMPT_TOKENS]
        produced = []
        rt.reset()
        rt.generate(prompt, 8, produced.append)
        print(
            f"filled cache with {len(prompt)} prompt + {len(produced)} decoded tokens",
            file=sys.stderr,
            flush=True,
        )

        with tempfile.TemporaryDirectory(prefix="colibri-kv-") as work:
            wanted = LAYERS if LAYERS else list(range(total))
            dumped = 0
            for layer in wanted:
                path = Path(work) / f"kv_layer_{layer}.bin"
                try:
                    rt.dump_kv(layer, str(path))
                except V2Error as error:
                    if not LAYERS and "attention layer" in str(error):
                        continue  # a Gated DeltaNet block: no KV cache to dump
                    print(f"layer {layer}: {error}", file=sys.stderr, flush=True)
                    continue

                print(f"\n===== layer {layer} =====", flush=True)
                subprocess.run(
                    [
                        bench,
                        "--dump", str(path),
                        "--queries", "32",
                        "--baseline-bits", BASELINE_BITS.get(CACHE_TYPE, "16"),
                    ],
                    check=False,
                )
                dumped += 1
                if not LAYERS and dumped >= 3:
                    break

            if dumped == 0:
                print("no attention layers dumped", file=sys.stderr)
                return 1
    finally:
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
