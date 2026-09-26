#!/usr/bin/env python3
"""Quality bench for DeltaNet / attention kernel variants.

Greedy md5-equality answers "did it change" but not "did it get worse": a
synonym and a forgotten fact both read DIFFERENT. The DeltaNet parallel
rewrite and the tensor-core attention work both change summation order (and,
in fp16, precision), so they need a quality oracle rather than the greedy
one. This file is that oracle, in two halves:

* `kernel` (synthetic, GPU, no weights): sequential vs chunked DeltaNet on
  random inputs *and* on `FLYWEIGHT_DELTA_DUMP` replays, with distributional
  metrics -- max/RMS relative error, cosine distance -- plus a rollout that
  carries state across consecutive chunks. Max-rel-err hides slow state
  drift; the rollout does not. A future fp16/tensor-core variant plugs in as
  a third runner next to `sequential`/`chunked`.

* `model` (needs MODEL.gguf): diverse raw prompts x env-flag arms
  (delta-sequential vs WY, warp vs cuBLAS attention, tile rows, QSA), each
  arm in its own subprocess because several flags are read once per process
  (chunked floor, tile rows). Metrics per arm: free-greedy first-divergence
  vs the reference arm, drift-free teacher-forced top-1 agreement, and a
  control arm (reference config, run last) as the noise floor -- the same
  discipline as bench_kv_quality.py, applied to kernel paths instead of KV
  types.

Usage:

    python bench/bench_kernel_quality.py kernel [--rows 256,512,1024]
    python bench/bench_kernel_quality.py model MODEL.gguf [--arms default,delta-sequential,attn-warp]
    python bench/bench_kernel_quality.py model MODEL.gguf --preview
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
NATIVE_TOOLS = BENCH.parent / "native" / "tools"
sys.path.insert(0, str(NATIVE_TOOLS))

CHUNK, DIM = 64, 128
EPSILON = 1e-6

PROMPTS = {
    "code": "Write a Python function that returns the nth Fibonacci number using memoization:\n",
    "math": "If a train travels 60 km in 45 minutes, what is its average speed in km/h? Show your reasoning.\n",
    "spanish": "Explica brevemente en español por qué el cielo es azul.\n",
    "chinese": "用中文简要解释什么是机器学习。\n",
    "factual": "List the first five prime numbers and explain what makes a number prime.\n",
    "creative": "Write the opening paragraph of a mystery novel set in a lighthouse.\n",
}

# Long-form prompt (~1100 tokens) that actually engages the chunked DeltaNet
# path (floor 512 rows) and the tensor-core attention crossover (>=512
# visible tokens). Short prompts take the sequential/warp path under every
# arm, which makes a kernel comparison vacuous -- this prompt exists so the
# arms differ by something.
LONG_PARAGRAPH = (
    "The harbor master kept a lantern lit through every storm, not because "
    "the ships needed it -- they had their own instruments -- but because "
    "the light reminded the town that someone was watching. Each evening "
    "she climbed the narrow stair with oil and wick, logged the wind in a "
    "ledger bound with twine, and noted which vessels had come home. "
)
LONG_TARGET_TOKENS = 1100

# Env-flag arms. Values are applied in the worker subprocess; every arm runs
# the same weights, context, budget and prompts, so only the kernel path
# differs. Flags read once per process (tile rows, chunked floor) are the
# reason arms are subprocesses rather than runtimes in one process.
ARMS: dict[str, dict[str, str]] = {
    "default": {},
    "delta-sequential": {"FLYWEIGHT_DELTA_SEQUENTIAL": "1"},
    "attn-warp": {"FLYWEIGHT_CUBLAS_PREFILL_ATTENTION": "0"},
    "attn-tensor": {"FLYWEIGHT_CUBLAS_PREFILL_ATTENTION": "1"},
    "tile16": {"FLYWEIGHT_CUBLAS_TILE_ROWS": "16"},
    "tile32": {"FLYWEIGHT_CUBLAS_TILE_ROWS": "32"},
    "tile64": {"FLYWEIGHT_CUBLAS_TILE_ROWS": "64"},
    "qsa": {"FLYWEIGHT_QSA": "1"},
    # Direct-from-packed int16 fold is the default for IQ2/IQ3 gate/up.
    # This arm is the dequant-then-fold path it replaced.
    "rows-i16": {"FLYWEIGHT_ROWS_Q8_IQ3S": "0"},
}


# --------------------------------------------------------------------------
# kernel mode (synthetic)


def _gpu_modules():
    import cupy as cp

    import deltanet_reference as ref
    import kernel_harness as kh

    return cp, ref, kh


def kernel_report(rows, key_heads, value_heads, gate_sigmoid, seed=7, rollout_chunks=1):
    """Distributional comparison of sequential vs chunked over `rollout_chunks`
    consecutive chunks with carried state. Returns a dict of metrics."""
    import numpy as np

    cp, ref, kh = _gpu_modules()
    kh.settle_clocks(0.5)
    metrics: dict[str, object] = {
        "rows": rows, "key_heads": key_heads, "value_heads": value_heads,
        "gate": int(gate_sigmoid), "rollout_chunks": rollout_chunks,
    }
    per_chunk_err: list[float] = []
    state = None
    for chunk_idx in range(rollout_chunks):
        data = ref.random_inputs(rows, key_heads, value_heads, DIM,
                                 seed=seed + chunk_idx)
        if state is None:
            state = data["state"]
        else:
            data["state"] = state
        want_out, want_state = ref.reference(
            data["convolved"], data["gates"], data["beta_logits"],
            data["decay_logits"], data["a_log"], data["dt_bias"],
            data["norm"], data["state"], key_heads, value_heads, DIM,
            EPSILON, gate_sigmoid=bool(gate_sigmoid))
        scale_out = float(np.abs(want_out).max())
        g = {k: cp.asarray(v) for k, v in data.items()}
        chunks = (rows + CHUNK - 1) // CHUNK

        def zeros(shape):
            return cp.zeros(shape, dtype=cp.float32)

        buffers = (
            zeros((chunks, value_heads, CHUNK, CHUNK)),
            zeros((chunks, value_heads, CHUNK, CHUNK)),
            zeros((rows, value_heads)), zeros((rows, value_heads)),
            zeros((rows, value_heads)), zeros((rows, value_heads)),
            zeros((rows, value_heads, DIM)), zeros((rows, value_heads, DIM)),
            zeros((rows, value_heads, DIM)),
            zeros((rows, value_heads * DIM)))
        r, k, v = np.int32(rows), np.int32(key_heads), np.int32(value_heads)
        got = {}
        for name in ("sequential", "chunked"):
            st = cp.asarray(data["state"]).copy()
            if name == "sequential":
                out = zeros((rows, value_heads * DIM))
                kh.kernel("qwen_delta_recurrent_chunk")(
                    (value_heads, 1, 1), (128, 1, 1),
                    (g["convolved"], g["gates"], g["beta_logits"],
                     g["decay_logits"], g["a_log"], g["dt_bias"], g["norm"],
                     st, out, r, k, v, np.int32(DIM), np.float32(EPSILON),
                     np.int32(gate_sigmoid)))
            else:
                attn, pmat, gcum, beta, qinv, kinv, w_rows, u_rows, core, out = buffers

                def launch(nm, grid, block, args):
                    kh.kernel(nm)(grid, block, args)

                launch("qwen_delta_wy_scores", (chunks, value_heads, 1), (256, 1, 1),
                       (g["convolved"], g["beta_logits"], g["decay_logits"],
                        g["a_log"], g["dt_bias"], attn, pmat, gcum, beta,
                        qinv, kinv, r, k, v))
                launch("qwen_delta_wy_solve", (chunks, value_heads, 1), (256, 1, 1),
                       (g["convolved"], attn, gcum, beta, kinv, w_rows, u_rows,
                        r, k, v))
                launch("qwen_delta_state_pass", (value_heads, DIM // 32, 1),
                       (256, 1, 1),
                       (g["convolved"], pmat, gcum, qinv, kinv, w_rows, u_rows,
                        st, core, r, k, v))
                launch("qwen_delta_norm_gate", (rows, value_heads, 1), (DIM, 1, 1),
                       (core, g["gates"], g["norm"], out, v,
                        np.float32(EPSILON), np.int32(gate_sigmoid)))
            cp.cuda.runtime.deviceSynchronize()
            arr = cp.asnumpy(out).reshape(rows, value_heads * DIM)
            got[name] = arr
        diff = got["chunked"] - got["sequential"]
        denom = float(np.abs(got["sequential"]).max())
        per_chunk_err.append(float(np.abs(diff).max() / max(denom, 1e-30)))
        if chunk_idx == 0:
            flat_want = want_out.reshape(-1)
            for name, arr in got.items():
                flat = arr.reshape(-1)
                err = np.abs(arr - want_out)
                cos = float(np.dot(flat, flat_want) / max(
                    np.linalg.norm(flat) * np.linalg.norm(flat_want), 1e-30))
                metrics[f"{name}.max_rel"] = float(err.max() / max(scale_out, 1e-30))
                metrics[f"{name}.rms_rel"] = float(
                    np.sqrt((err ** 2).mean()) / max(scale_out, 1e-30))
                metrics[f"{name}.cos_dist"] = float(max(0.0, 1.0 - cos))
        # Carry the sequential state: drift in the reference path is a bug in
        # the test, not the kernel, so the rollout tracks the trusted path.
        _, state = ref.reference(
            data["convolved"], data["gates"], data["beta_logits"],
            data["decay_logits"], data["a_log"], data["dt_bias"],
            data["norm"], data["state"], key_heads, value_heads, DIM,
            EPSILON, gate_sigmoid=bool(gate_sigmoid))
    metrics["cross.max_rel_per_chunk"] = per_chunk_err
    metrics["cross.max_rel_worst"] = max(per_chunk_err)
    return metrics


def cmd_kernel(args) -> int:
    geometry = {"dense-27b": (16, 48), "moe-35b-a3b": (16, 32),
                "qwen4exp": (16, 48)}[args.geometry]
    key_heads, value_heads = geometry
    gate = 1 if args.geometry == "qwen4exp" else 0
    rows_list = [int(r) for r in args.rows.split(",") if r.strip()]
    worst = 0.0
    for rows in rows_list:
        m = kernel_report(rows, key_heads, value_heads, gate,
                          rollout_chunks=args.rollout)
        worst = max(worst, float(m["cross.max_rel_worst"]))
        print(f"rows={rows} rollout={args.rollout}")
        for name in ("sequential", "chunked"):
            print(f"  {name:10s} max_rel {m[f'{name}.max_rel']:.2e}"
                  f"  rms_rel {m[f'{name}.rms_rel']:.2e}"
                  f"  cos_dist {m[f'{name}.cos_dist']:.2e}")
        print(f"  cross      worst-chunk max_rel {m['cross.max_rel_worst']:.2e}")
        if args.jsonl:
            with open(args.jsonl, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(
                    {"mode": "kernel", "geometry": args.geometry, **m,
                     "cross.max_rel_per_chunk": list(
                         m["cross.max_rel_per_chunk"])}) + "\n")
    print(f"\nworst cross-path drift: {worst:.2e}")
    # Gate: chunked must track sequential an order of magnitude below the
    # reference tolerance bench_deltanet.py already holds (~1e-6); anything
    # coarser is a rewrite-candidate rejection, not a measurement.
    return 0 if worst < 1e-5 else 2


# --------------------------------------------------------------------------
# model mode (real weights, subprocess arms)


def run_worker(args) -> int:
    """One arm: greedy continuations + teacher-forced agreement as JSON."""
    os.environ["FLYWEIGHT_EXPERT_HISTORY"] = "0"
    from flyweight.v2 import V2Model

    model = V2Model(args.model)
    try:
        only = os.environ.get("FLYWEIGHT_KQ_ONLY", "")
        names = [n for n in PROMPTS] if not only else [only] if only in PROMPTS else []
        prompts = {name: model.tokenize(PROMPTS[name]) for name in names}
        if not only or only == "long":
            long_unit = model.tokenize(LONG_PARAGRAPH)
            prompts["long"] = (long_unit * (LONG_TARGET_TOKENS // len(long_unit) + 1))[
                :LONG_TARGET_TOKENS]
        out: dict[str, object] = {"prompts": {}, "info": {}}
        with model.native_runtime(
            context_limit=args.context, gpu_cache_bytes=args.gpu_cache_mib * 1024 * 1024,
            cache_type_k="f16", cache_type_v="f16", context_explicit=True,
            prefill_checkpoint_interval=args.checkpoint_interval,
        ) as runtime:
            runtime.prepare()
            out["info"] = {k: int(runtime.info[k]) for k in
                           ("expert_cache_bytes", "expert_cache_slots",
                            "kv_reserved_bytes") if k in runtime.info}
            for name, prompt in prompts.items():
                tokens: list[int] = []
                runtime.reset()
                runtime.generate(list(prompt), args.tokens, tokens.append)
                out["prompts"][name] = {"tokens": tokens}
        print(json.dumps({"ok": True, **out}))
        return 0
    finally:
        model.close()


def cmd_model(args) -> int:
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown} (known: {sorted(ARMS)})")
    if args.preview:
        print(f"arms: {arms}")
        print(f"prompts: {sorted(PROMPTS) + ['long']}  tokens={args.tokens} context={args.context}")
        for name, env in ARMS.items():
            if name in arms:
                print(f"  {name:18s} {env or '(defaults)'}")
        return 0

    full_arms = list(arms)
    control = None
    if args.control and "default" in arms:
        control = "default'"
        full_arms.append(control)

    results: dict[str, dict] = {}
    for label in full_arms:
        base = "default" if label == "default'" else label
        env = dict(os.environ)
        env.update(ARMS.get(base, {}))
        env["FLYWEIGHT_EXPERT_HISTORY"] = "0"
        # Workers must import the checkout under test, never the installed
        # copy: an env flag the installed library does not know would be
        # silently ignored and the comparison vacuous.
        env["PYTHONPATH"] = str(BENCH.parent / "src") + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        cmd = [sys.executable, str(BENCH / "bench_kernel_quality.py"),
               "worker", args.model, "--context", str(args.context),
               "--gpu-cache-mib", str(args.gpu_cache_mib),
               "--tokens", str(args.tokens),
               "--checkpoint-interval", str(args.checkpoint_interval)]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=args.timeout)
        if proc.returncode != 0:
            print(f"arm {label} failed:\n{proc.stderr[-2000:]}")
            return 1
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            print(f"arm {label}: unparseable worker output:\n{proc.stdout[-2000:]}")
            return 1
        results[label] = payload
        n = len(payload["prompts"])
        print(f"  {label:18s} {n} prompts, expert slots "
              f"{payload['info'].get('expert_cache_slots')}", flush=True)

    slots = {json.dumps(r["info"].get("expert_cache_slots")) for r in results.values()}
    if len(slots) > 1:
        print("  !! expert cache differs across arms: placement, not just "
              "kernel path, is in the measurement")

    ref_prompts = results["default"]["prompts"]
    print(f"\n  {'arm':>18}  {'identical':>9}  {'diverge@':>8}  {'agree':>6}")
    summaries = {}
    for label, payload in results.items():
        if label == "default":
            continue
        agreements = _agree_all(args, label, ref_prompts)
        divs = []
        agrees = []
        identical = 0
        for name, ref in ref_prompts.items():
            cand = payload["prompts"][name]["tokens"]
            first = next((i for i, (a, b) in
                          enumerate(zip(cand, ref["tokens"])) if a != b), None)
            if first is None and len(cand) != len(ref["tokens"]):
                first = min(len(cand), len(ref["tokens"]))
            if first is None:
                identical += 1
            else:
                divs.append(first)
            agrees.append(agreements.get(name, float("nan")))
        summaries[label] = (identical, divs, agrees)
        div_med = f"{statistics.median(divs):8.0f}" if divs else "       -"
        agr = f"{statistics.mean(agrees):6.1%}" if agrees else "     -"
        print(f"  {label:>18}  {identical:9d}  {div_med}  {agr}")

    if control is not None and control in summaries:
        ident, divs, _ = summaries[control]
        total = len(ref_prompts)
        print(f"\n  noise floor (default rerun): {ident}/{total} identical"
              + (f", median diverge@{statistics.median(divs):.0f}" if divs else ""))
        for label in results:
            if label in ("default", control):
                continue
            ident_c, _, _ = summaries[label]
            verdict = ("INSIDE the noise floor -- shows nothing either way"
                       if ident_c >= ident else
                       f"{total - ident_c}/{total} continuations moved "
                       f"vs {total - ident}/{total} for the control")
            print(f"  {label:>18}: {verdict}")
    else:
        print("\n  no control arm: rerun with default in --arms to get the noise floor")
    if args.jsonl:
        with open(args.jsonl, "a", encoding="utf-8") as stream:
            for label, payload in results.items():
                stream.write(json.dumps(
                    {"arm": label, "model": args.model, "context": args.context,
                     "identical": summaries.get(label, (None, None, None))[0],
                     "info": payload["info"]}) + "\n")
    return 0


_agree_cache: dict[tuple[str, str], dict[str, float]] = {}


def _agree_all(args, label, ref_prompts) -> dict[str, float]:
    """Teacher-forced top-1 agreement for every prompt, one worker prepare."""
    base = "default" if label == "default'" else label
    key = (label, ",".join(
        ",".join(map(str, ref_prompts[n]["tokens"][:4])) for n in sorted(ref_prompts)))
    if key in _agree_cache:
        return _agree_cache[key]
    env = dict(os.environ)
    env.update(ARMS.get(base, {}))
    env["FLYWEIGHT_EXPERT_HISTORY"] = "0"
    env["PYTHONPATH"] = str(BENCH.parent / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    refs_blob = json.dumps({n: ref_prompts[n]["tokens"] for n in ref_prompts})
    cmd = [sys.executable, str(BENCH / "bench_kernel_quality.py"),
           "agreebatch", args.model, "--context", str(args.context),
           "--gpu-cache-mib", str(args.gpu_cache_mib),
           "--checkpoint-interval", str(args.checkpoint_interval),
           "--agree-refs", refs_blob]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=args.timeout)
    try:
        values = {n: float(v) for n, v in
                  json.loads(proc.stdout.strip().splitlines()[-1])["agreements"].items()}
    except (json.JSONDecodeError, IndexError, KeyError, ValueError):
        print(f"  !! agreebatch worker failed for {label}:\n"
              f"{proc.stderr[-1000:]}")
        values = {n: float("nan") for n in ref_prompts}
    _agree_cache[key] = values
    return values


def run_agreebatch(args) -> int:
    """Worker: teacher-forced agreement for all refs with a single prepare."""
    from flyweight.v2 import V2Model

    os.environ["FLYWEIGHT_EXPERT_HISTORY"] = "0"
    model = V2Model(args.model)
    try:
        refs = json.loads(args.agree_refs or "{}")
        prompts = {name: model.tokenize(PROMPTS[name]) for name in refs if name in PROMPTS}
        if "long" in refs:
            long_unit = model.tokenize(LONG_PARAGRAPH)
            prompts["long"] = (long_unit * (LONG_TARGET_TOKENS // len(long_unit) + 1))[
                :LONG_TARGET_TOKENS]
        agreements: dict[str, float] = {}
        with model.native_runtime(
            context_limit=args.context, gpu_cache_bytes=args.gpu_cache_mib * 1024 * 1024,
            cache_type_k="f16", cache_type_v="f16", context_explicit=True,
            prefill_checkpoint_interval=args.checkpoint_interval,
        ) as runtime:
            runtime.prepare()
            for name, reference in refs.items():
                if not reference:
                    agreements[name] = 0.0
                    continue
                first: list[int] = []
                runtime.reset()
                runtime.generate(list(prompts[name]), 1, first.append)
                agree = int(bool(first) and first[0] == reference[0])
                context = reference[0]
                for expected in reference[1:]:
                    agree += int(runtime.decode(context) == expected)
                    context = expected
                agreements[name] = agree / len(reference)
        print(json.dumps({"agreements": agreements}))
        return 0
    finally:
        model.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    pk = sub.add_parser("kernel", help="synthetic DeltaNet distributional check")
    pk.add_argument("--rows", default="256,512,1024")
    pk.add_argument("--geometry", default="qwen4exp",
                    choices=["dense-27b", "moe-35b-a3b", "qwen4exp"])
    pk.add_argument("--rollout", type=int, default=4,
                    help="consecutive chunks with carried state")
    pk.add_argument("--jsonl", help="append metrics here")

    pm = sub.add_parser("model", help="prompt x env-arm quality comparison")
    pm.add_argument("model")
    pm.add_argument("--arms", default="default,delta-sequential,attn-warp,attn-tensor",
                    help=f"comma list of {sorted(ARMS)}")
    pm.add_argument("--context", type=int, default=4096)
    pm.add_argument("--gpu-cache-mib", type=int, default=9000)
    pm.add_argument("--tokens", type=int, default=64)
    pm.add_argument("--checkpoint-interval", type=int, default=0,
                    help="mid-prefill checkpoint spacing; 0 (end-only) keeps full "
                         "1024-row chunks so the 512-row WY floor and the 512-token "
                         "attention crossover actually engage. Checkpoints do not "
                         "change greedy output (each item resets first).")
    pm.add_argument("--timeout", type=int, default=1200)
    pm.add_argument("--control", action="store_true", default=True)
    pm.add_argument("--no-control", dest="control", action="store_false")
    pm.add_argument("--preview", action="store_true")
    pm.add_argument("--jsonl", help="append per-arm summary here")

    pw = sub.add_parser("worker", help=argparse.SUPPRESS)
    pw.add_argument("model")
    pw.add_argument("--context", type=int, default=4096)
    pw.add_argument("--gpu-cache-mib", type=int, default=9000)
    pw.add_argument("--tokens", type=int, default=64)
    pw.add_argument("--checkpoint-interval", type=int, default=0)

    pa = sub.add_parser("agreebatch", help=argparse.SUPPRESS)
    pa.add_argument("model")
    pa.add_argument("--context", type=int, default=4096)
    pa.add_argument("--gpu-cache-mib", type=int, default=9000)
    pa.add_argument("--checkpoint-interval", type=int, default=0)
    pa.add_argument("--agree-refs", default="{}")

    args = parser.parse_args()
    if args.mode == "kernel":
        return cmd_kernel(args)
    if args.mode == "model":
        return cmd_model(args)
    if args.mode == "worker":
        return run_worker(args)
    if args.mode == "agreebatch":
        return run_agreebatch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
