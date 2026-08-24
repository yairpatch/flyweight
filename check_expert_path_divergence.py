#!/usr/bin/env python3
"""Quantify how far the CPU and GPU expert execution paths disagree.

check_greedy_determinism.py establishes *that* the paths disagree (forcing one
path makes greedy decode reproducible; changing only gpu_cache_bytes changes the
output). This measures *how much*, which is what decides whether the fix is
matching reduction order or repairing one of the two paths.

Two things are measured over an identical forced token sequence:

  * argmax agreement -- teacher forcing means both configurations consume the
    same tokens no matter what they predict, so predictions stay comparable
    past the first disagreement. Free-running generation cannot measure this:
    once the paths diverge they are conditioning on different prefixes and any
    later difference is uninterpretable.
  * per-layer KV divergence -- dump_kv decodes the live attention window, and
    an f32 cache is forced so the f16 store rounding (~1e-3 relative) does not
    sit on top of the effect being measured. This localises the divergence to a
    layer and shows whether it amplifies with depth.

Interpretation: a relative divergence around f32 accumulation noise (~1e-6) is
reduction-order drift and is fixable by pinning the order. Divergence around
int8 activation-quantisation noise (~1e-2) is the expected consequence of the
GPU path quantising activations while the CPU path keeps them in f32 -- that is
a precision *choice* to standardise, not a bug. Anything much larger than that
is a genuine defect in one path.

    ./check_expert_path_divergence.py MODEL.gguf
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

PROMPT = "Explain how a turbocharger works, step by step."

CHILD = """
import json, os
from colibri_next.v2 import V2Model, V2QwenRuntime

options = json.loads(os.environ["COLIBRI_DIVERGENCE_OPTIONS"])
outdir = os.environ["COLIBRI_DIVERGENCE_OUTDIR"]

model = V2Model(os.environ["COLIBRI_DIVERGENCE_MODEL"])
runtime = V2QwenRuntime(
    model, cache_type_k="f32", cache_type_v="f32", **options)
runtime.prepare()

tokens = model.tokenize(os.environ["COLIBRI_DIVERGENCE_PROMPT"])
forced = json.loads(os.environ["COLIBRI_DIVERGENCE_FORCED"])
extend = int(os.environ["COLIBRI_DIVERGENCE_EXTEND"])

# Teacher forcing: feed the fixed sequence regardless of what is predicted, so
# both configurations stay on the same context and their argmaxes stay
# comparable at every position. Without it, the two runs condition on different
# prefixes after the first disagreement and nothing later is interpretable.
if forced:
    predictions = [runtime.decode(token) for token in forced]
else:
    # Reference side: free-run to build the sequence the other side is forced
    # through. A free run is teacher forcing on its own output, so these
    # predictions are already the forced-path answers for this configuration.
    # `forced` records exactly what was consumed, so predictions[i] is always
    # the argmax after consuming forced[i] on both sides.
    forced = list(tokens)
    predictions = [runtime.decode(token) for token in tokens]
    for _ in range(extend):
        forced.append(predictions[-1])
        predictions.append(runtime.decode(predictions[-1]))

layers = []
for index in range(int(os.environ["COLIBRI_DIVERGENCE_MAXLAYERS"])):
    path = os.path.join(outdir, f"layer{index:03d}.bin")
    try:
        runtime.dump_kv(index, path)
    except Exception:
        continue          # DeltaNet layer, or past the end of the stack
    layers.append(index)

print(json.dumps({
    "tokens": tokens,
    "forced": forced,
    "predictions": predictions,
    "layers": layers,
    "info": {k: v for k, v in runtime.info.items()
             if k in ("expert_cache_hits", "expert_cache_misses")},
}))
"""


def run(model, options, outdir, max_layers, forced=(), extend=0):
    environment = dict(
        os.environ,
        PYTHONPATH="src",
        COLIBRI_DIVERGENCE_MODEL=model,
        COLIBRI_DIVERGENCE_PROMPT=PROMPT,
        COLIBRI_DIVERGENCE_OPTIONS=json.dumps(options),
        COLIBRI_DIVERGENCE_OUTDIR=outdir,
        COLIBRI_DIVERGENCE_MAXLAYERS=str(max_layers),
        COLIBRI_DIVERGENCE_FORCED=json.dumps(list(forced)),
        COLIBRI_DIVERGENCE_EXTEND=str(extend),
    )
    result = subprocess.run(
        [sys.executable, "-c", CHILD], capture_output=True, text=True,
        env=environment,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr[-2000:])
        raise SystemExit(f"run failed with status {result.returncode}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def parse_pairs(values):
    pairs = {}
    for value in values or ():
        if "=" not in value:
            raise SystemExit(f"expected KEY=VALUE, got {value!r}")
        key, _, raw = value.partition("=")
        pairs[key] = raw
    return pairs


def coerce(raw):
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    if raw in ("true", "false"):
        return raw == "true"
    return raw


def load_dump(path):
    """Read a dump_kv file: int32 count, int32 head_dim, then keys, values."""
    with open(path, "rb") as handle:
        raw = handle.read()
    count, head_dim = struct.unpack_from("<ii", raw, 0)
    span = count * head_dim
    floats = struct.unpack_from(f"<{span * 2}f", raw, 8)
    return floats[:span], floats[span:]


def compare(left, right):
    """max|delta|, rms delta, and rms delta relative to the rms magnitude."""
    peak = 0.0
    total = 0.0
    scale = 0.0
    for a, b in zip(left, right):
        delta = a - b
        peak = max(peak, abs(delta))
        total += delta * delta
        scale += a * a
    count = max(1, len(left))
    rms = (total / count) ** 0.5
    magnitude = (scale / count) ** 0.5
    return peak, rms, (rms / magnitude if magnitude else 0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--max-layers", type=int, default=80)
    parser.add_argument("--gpu-cache-mib", type=int, default=8000,
                        help="cache budget for the GPU-path side")
    parser.add_argument("--extend", type=int, default=64,
                        help="generated tokens to append to the forced sequence; "
                             "prompt tokens alone have margins too wide to flip")
    parser.add_argument("--left-cache-mib", type=int, default=3000,
                        help="cache budget for the reference side")
    parser.add_argument("--left", action="append", metavar="KEY=VALUE",
                        help="V2QwenRuntime keyword for the reference side")
    parser.add_argument("--right", action="append", metavar="KEY=VALUE",
                        help="V2QwenRuntime keyword for the compared side")
    arguments = parser.parse_args()

    # Default is the clean A/B: one code path (the hybrid branch), differing
    # only in how much of the expert working set stays resident. Comparing
    # expert_mode=cpu against auto is NOT clean -- those are two different MoE
    # implementations (qwen_cpu_moe vs the hybrid route split), so a difference
    # there confounds "where experts run" with "which routine ran".
    def side(spec, fallback_mib):
        options = {k: coerce(v) for k, v in parse_pairs(spec).items()}
        options.setdefault("context_limit", 2048)
        if "gpu_cache_bytes" not in options and "expert_mode" not in options:
            options["gpu_cache_bytes"] = fallback_mib * 1024 * 1024
        return options

    configurations = {
        "cpu": side(arguments.left, arguments.left_cache_mib),
        "gpu": side(arguments.right, arguments.gpu_cache_mib),
    }

    results = {}
    with tempfile.TemporaryDirectory() as root:
        # The CPU side free-runs first to build the forced sequence; the GPU
        # side is then driven through exactly those tokens.
        for name, options in configurations.items():
            outdir = os.path.join(root, name)
            os.makedirs(outdir)
            results[name] = run(
                arguments.model, options, outdir, arguments.max_layers,
                forced=results["cpu"]["forced"] if results else (),
                extend=0 if results else arguments.extend,
            )
            results[name]["dir"] = outdir
            print(f"{name}: {len(results[name]['layers'])} attention layers, "
                  f"{results[name]['info']}")

        left, right = results["cpu"], results["gpu"]
        if left["forced"] != right["forced"]:
            raise SystemExit("forced sequences differed between runs")

        agree = sum(a == b for a, b in zip(
            left["predictions"], right["predictions"]))
        total = len(left["predictions"])
        first = next((i for i, (a, b) in enumerate(
            zip(left["predictions"], right["predictions"])) if a != b), None)
        print()
        print(f"argmax agreement: {agree}/{total} forced positions "
              f"({100.0 * agree / total:.1f}%)"
              + (f", first disagreement at {first}" if first is not None
                 else ", identical"))
        print()

        shared = sorted(set(left["layers"]) & set(right["layers"]))
        print(f"{'layer':>5}  {'keys max|d|':>12} {'keys rel':>10}"
              f"  {'values max|d|':>13} {'values rel':>10}")
        worst = 0.0
        for index in shared:
            name = f"layer{index:03d}.bin"
            keys_a, values_a = load_dump(os.path.join(left["dir"], name))
            keys_b, values_b = load_dump(os.path.join(right["dir"], name))
            k_peak, _, k_rel = compare(keys_a, keys_b)
            v_peak, _, v_rel = compare(values_a, values_b)
            worst = max(worst, k_rel, v_rel)
            print(f"{index:>5}  {k_peak:>12.3e} {k_rel:>10.2e}"
                  f"  {v_peak:>13.3e} {v_rel:>10.2e}")

        print()
        print(f"worst relative divergence: {worst:.2e}")
        # Only the bottom band is a safe call. Everything above it is a
        # precision-class difference rather than reduction-order drift, but
        # accumulated KV divergence cannot separate "int8 activations
        # amplifying over 40 layers" from "int8 activations plus a defect" --
        # per-layer amplification puts plausible quantisation error well into
        # the 1e-2 range. Settling that needs a single expert's output compared
        # between the two paths, not the accumulated state.
        if worst < 1e-5:
            print("  -> f32 accumulation-order drift; pin the reduction order")
        else:
            print("  -> NOT accumulation drift: this is a precision-class "
                  "difference,\n     consistent with the GPU path quantising "
                  "activations to int8 while\n     the CPU path keeps f32. "
                  "Whether a defect rides on top of that\n     is not decidable "
                  "from accumulated KV -- compare one expert's\n     output "
                  "between the paths to separate them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
