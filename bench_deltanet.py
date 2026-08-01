"""Chunked WY DeltaNet: correctness against the sequential kernel, and speed.

The prefill path runs the gated delta rule one token at a time
(`qwen_delta_recurrent_chunk`), so a 1024-token chunk is 1024 serial steps
spread over only `value_heads` blocks.  The chunked WY form does the same
recurrence as matrix work inside 64-token chunks and only hands the state
between chunks, so the serial depth drops to rows/64 and the grid widens.

Both paths are checked against native/tools/deltanet_reference.py, which is
itself checked against the sequential kernel.

    python bench_deltanet.py [rows ...]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "native" / "tools"))

import cupy as cp  # noqa: E402

import deltanet_reference as ref  # noqa: E402
import kernel_harness as kh  # noqa: E402

CHUNK, DIM, TILE = 64, 128, 32
# Head counts come from the checkpoint: ssm_a gives the value heads, ssm_norm the
# head dim, and ssm_conv1d's channels leave (channels - v*dim) / (2*dim) key heads.
#   Qwen3.6-27B      (dense): 48 value / 16 key / 128
#   Qwen3.6-35B-A3B  (MoE):   32 value / 16 key / 128
# head_dim is 128 in both, which is what these kernels require.
GEOMETRY = {"dense-27b": (16, 48), "moe-35b-a3b": (16, 32)}
KEY_HEADS, VALUE_HEADS = GEOMETRY[os.environ.get("COLIBRI_GEOMETRY", "dense-27b")]
EPSILON = 1e-6
# Once the kernels are embedded in colibri_v2_native_kernels.hpp the corpus
# already defines them; appending the prototype again would be a redefinition.
_PROTOTYPE_PATH = Path(__file__).parent / "native" / "tools" / "deltanet_chunked.cu"
PROTOTYPE = ("" if "qwen_delta_wy_scores" in kh.source()
             else _PROTOTYPE_PATH.read_text())


def chunked(g, rows, key_heads, value_heads, state, buffers):
    """Run the four chunked kernels; returns (output, state) device arrays."""
    chunks = (rows + CHUNK - 1) // CHUNK
    attn, pmat, gcum, beta, qinv, kinv, w_rows, u_rows, core, out = buffers
    launch = lambda name, grid, block, args: kh.kernel(name, PROTOTYPE)(grid, block, args)
    r, k, v = np.int32(rows), np.int32(key_heads), np.int32(value_heads)
    launch("qwen_delta_wy_scores", (chunks, value_heads, 1), (256, 1, 1),
           (g["convolved"], g["beta_logits"], g["decay_logits"], g["a_log"],
            g["dt_bias"], attn, pmat, gcum, beta, qinv, kinv, r, k, v))
    launch("qwen_delta_wy_solve", (chunks, value_heads, 1), (256, 1, 1),
           (g["convolved"], attn, gcum, beta, kinv, w_rows, u_rows, r, k, v))
    launch("qwen_delta_state_pass", (value_heads, DIM // TILE, 1), (256, 1, 1),
           (g["convolved"], pmat, gcum, qinv, kinv, w_rows, u_rows, state, core, r, k, v))
    launch("qwen_delta_norm_gate", (rows, value_heads, 1), (DIM, 1, 1),
           (core, g["gates"], g["norm"], out, v, np.float32(EPSILON)))
    return out, state


def sequential(g, rows, key_heads, value_heads, state, out):
    kh.kernel("qwen_delta_recurrent_chunk")(
        (value_heads, 1, 1), (128, 1, 1),
        (g["convolved"], g["gates"], g["beta_logits"], g["decay_logits"],
         g["a_log"], g["dt_bias"], g["norm"], state, out,
         np.int32(rows), np.int32(key_heads), np.int32(value_heads),
         np.int32(DIM), np.float32(EPSILON)))
    return out, state


def allocate(rows, value_heads):
    chunks = (rows + CHUNK - 1) // CHUNK
    zeros = lambda shape: cp.zeros(shape, dtype=cp.float32)
    return (zeros((chunks, value_heads, CHUNK, CHUNK)),
            zeros((chunks, value_heads, CHUNK, CHUNK)),
            zeros((rows, value_heads)), zeros((rows, value_heads)),
            zeros((rows, value_heads)), zeros((rows, value_heads)),
            zeros((rows, value_heads, DIM)), zeros((rows, value_heads, DIM)),
            zeros((rows, value_heads, DIM)), zeros((rows, value_heads * DIM)))


CHECKPOINT = os.environ.get("COLIBRI_MODEL")


def report(rows, seed=7):
    kh.settle_clocks(0.5)
    decay = None
    if CHECKPOINT:
        layer, a_log, dt_bias = ref.checkpoint_decay(CHECKPOINT, VALUE_HEADS, seed=seed)
        decay = (a_log, dt_bias)
        print(f"decay weights from {CHECKPOINT.split('/')[-1]} blk.{layer}: "
              f"a_log [{a_log.min():+.2f}, {a_log.max():+.2f}]")
    data = ref.random_inputs(rows, KEY_HEADS, VALUE_HEADS, DIM, seed=seed, decay=decay)
    want_out, want_state = ref.reference(
        data["convolved"], data["gates"], data["beta_logits"], data["decay_logits"],
        data["a_log"], data["dt_bias"], data["norm"], data["state"],
        KEY_HEADS, VALUE_HEADS, DIM, EPSILON)
    scale_out = np.abs(want_out).max()
    scale_state = np.abs(want_state).max()

    g = {k: cp.asarray(v) for k, v in data.items()}
    buffers = allocate(rows, VALUE_HEADS)
    print(f"rows={rows}")
    results = {}
    for name, run in (("sequential", sequential), ("chunked", chunked)):
        state = cp.asarray(data["state"]).copy()
        if name == "sequential":
            out, state = run(g, rows, KEY_HEADS, VALUE_HEADS, state,
                             cp.zeros((rows, VALUE_HEADS * DIM), dtype=cp.float32))
        else:
            out, state = run(g, rows, KEY_HEADS, VALUE_HEADS, state, buffers)
        cp.cuda.runtime.deviceSynchronize()
        got_out = cp.asnumpy(out).reshape(rows, VALUE_HEADS * DIM)
        got_state = cp.asnumpy(state).reshape(want_state.shape)
        results[name] = (np.abs(got_out - want_out).max() / scale_out,
                         np.abs(got_state - want_state).max() / scale_state)
        print(f"  {name:10s} rel err  output {results[name][0]:.2e}"
              f"  state {results[name][1]:.2e}")

    state = cp.asarray(data["state"]).copy()
    out = cp.zeros((rows, VALUE_HEADS * DIM), dtype=cp.float32)
    serial_ms = kh.time_kernel(
        kh.kernel("qwen_delta_recurrent_chunk"),
        g["convolved"], g["gates"], g["beta_logits"], g["decay_logits"], g["a_log"],
        g["dt_bias"], g["norm"], state, out, np.int32(rows), np.int32(KEY_HEADS),
        np.int32(VALUE_HEADS), np.int32(DIM), np.float32(EPSILON),
        grid=(VALUE_HEADS, 1, 1), block=(128, 1, 1))

    # The four-kernel sequence is timed as a whole, under the same settling rule
    # time_kernel applies to a single launch.
    state = cp.asarray(data["state"]).copy()
    start, stop = cp.cuda.Event(), cp.cuda.Event()

    def batch():
        samples = []
        for _ in range(20):
            start.record()
            chunked(g, rows, KEY_HEADS, VALUE_HEADS, state, buffers)
            stop.record()
            stop.synchronize()
            samples.append(cp.cuda.get_elapsed_time(start, stop))
        samples.sort()
        return samples[len(samples) // 2]

    import time as _time
    deadline = _time.monotonic() + 8.0
    chunk_ms = batch()
    while _time.monotonic() < deadline:
        current = batch()
        if abs(current - chunk_ms) <= 0.03 * max(current, chunk_ms):
            chunk_ms = current
            break
        chunk_ms = current
    print(f"  sequential {serial_ms:8.3f} ms   chunked {chunk_ms:8.3f} ms"
          f"   speedup {serial_ms / chunk_ms:5.2f}x")
    return results


def replay(directory):
    """Re-run both kernels on activations dumped by COLIBRI_DELTA_DUMP.

    The synthetic inputs are Gaussian; real projections are not, so this is the
    check that the chunked form holds on the distribution it will actually see.
    """
    root = Path(directory)
    meta = dict(line.split() for line in
                (root / "delta_meta.txt").read_text().strip().splitlines())
    rows, key_heads = int(meta["rows"]), int(meta["key_heads"])
    value_heads, head_dim = int(meta["value_heads"]), int(meta["head_dim"])
    epsilon = float(meta["epsilon"])
    if head_dim != DIM:
        raise SystemExit(f"dump has head_dim {head_dim}; these kernels require {DIM}")
    load = lambda name: np.fromfile(root / f"delta_{name}.f32", dtype=np.float32)
    data = dict(
        convolved=load("convolved").reshape(rows, int(meta["channels"])),
        gates=load("gates").reshape(rows, value_heads * head_dim),
        beta_logits=load("beta_logits").reshape(rows, value_heads),
        decay_logits=load("decay_logits").reshape(rows, value_heads),
        a_log=load("a_log"), dt_bias=load("dt_bias"), norm=load("norm"),
        state=load("state").reshape(value_heads, head_dim, head_dim))

    conv = data["convolved"]
    decay = np.exp(data["a_log"][None, :] * ref.softplus(
        data["decay_logits"] + data["dt_bias"][None, :]))
    print(f"replay layer {meta['layer']}: rows={rows} key_heads={key_heads} "
          f"value_heads={value_heads}")
    print(f"  convolved  absmax {np.abs(conv).max():.3f}  std {conv.std():.3f}  "
          f"kurtosis {((conv - conv.mean())**4).mean() / conv.var()**2:.1f}")
    print(f"  decay      min {decay.min():.3e}  max {decay.max():.6f}  "
          f"zero-decay {100 * (decay == 0).mean():.2f}%")
    print(f"  state      absmax {np.abs(data['state']).max():.3f}")

    want_out, want_state = ref.reference(
        data["convolved"], data["gates"], data["beta_logits"], data["decay_logits"],
        data["a_log"], data["dt_bias"], data["norm"], data["state"],
        key_heads, value_heads, head_dim, epsilon)
    scale_out, scale_state = np.abs(want_out).max(), np.abs(want_state).max()

    g = {k: cp.asarray(v) for k, v in data.items()}
    buffers = allocate(rows, value_heads)
    for name, run in (("sequential", sequential), ("chunked", chunked)):
        state = cp.asarray(data["state"]).copy()
        target = (cp.zeros((rows, value_heads * DIM), dtype=cp.float32)
                  if name == "sequential" else buffers)
        out, state = run(g, rows, key_heads, value_heads, state, target)
        cp.cuda.runtime.deviceSynchronize()
        got_out = cp.asnumpy(out).reshape(rows, value_heads * DIM)
        got_state = cp.asnumpy(state).reshape(want_state.shape)
        print(f"  {name:10s} rel err  output "
              f"{np.abs(got_out - want_out).max() / scale_out:.2e}"
              f"  state {np.abs(got_state - want_state).max() / scale_state:.2e}"
              f"  nonfinite {int(np.isnan(got_out).sum() + np.isinf(got_out).sum())}")
    return 0


def main() -> int:
    dump = os.environ.get("COLIBRI_DELTA_DUMP")
    if dump:
        return replay(dump)
    row_counts = [int(a) for a in sys.argv[1:]] or [64, 200, 512, 1024, 2048]
    worst = 0.0
    for rows in row_counts:
        for name, (out_err, state_err) in report(rows).items():
            worst = max(worst, out_err, state_err)
    print(f"\nworst relative error across all runs: {worst:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
