"""Native vision tower against the NumPy reference, on a real mmproj.

The tower needs a prepared Qwen runtime for its stream and kernels but not
a real language model, so a synthetic qwen35 GGUF sized to the mmproj's
projection width stands in for one. Pixels are random; the check is the
arithmetic, not the picture.

    python check_vision_parity.py [--mmproj PATH] [--side 128] [--backend cuda|cpu]

Exit status is non-zero when the largest deviation exceeds the tolerance
(bf16 weights with f32 accumulation put the native path within ~1e-3 of the
all-f32 reference on the 27-block Ornith tower).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "native" / "tools"))

from tests.dense_gguf_fixture import DenseQwenSpec, build_dense_qwen35_gguf  # noqa: E402
from flyweight.v2 import V2Model  # noqa: E402
import qwen_vision_reference as reference  # noqa: E402

DEFAULT_MMPROJ = Path.home() / "Downloads/gguf/mmproj-Ornith-1.5-35B-BF16.gguf"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmproj", type=Path, default=DEFAULT_MMPROJ)
    parser.add_argument("--side", type=int, default=128, help="square image side in pixels")
    parser.add_argument("--width", type=int, default=0, help="override width (multiple of 32)")
    parser.add_argument("--height", type=int, default=0, help="override height (multiple of 32)")
    parser.add_argument("--backend", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--tolerance", type=float, default=2e-2)
    parser.add_argument("--skip-reference", action="store_true", help="time the native tower only")
    args = parser.parse_args()

    gguf = reference.read_gguf(args.mmproj)
    config = reference.VisionConfig.from_gguf(gguf)
    width = args.width or args.side
    height = args.height or args.side
    rng = np.random.default_rng(20260903)
    pixels = rng.uniform(-1, 1, size=(height, width, 3)).astype(np.float32)

    V2Model.select_backend(args.backend)
    with tempfile.TemporaryDirectory(prefix="flyweight-vision-") as holder:
        path = Path(holder) / "host.gguf"
        # Two blocks (one attention, one DeltaNet) at the projection width.
        build_dense_qwen35_gguf(
            path,
            DenseQwenSpec(hidden=config.projection, layers=2, attention_every=2,
                          heads=4, kv_heads=2, head_dim=64, vocabulary=64,
                          intermediate=256),
        )
        with V2Model(path, mmproj=args.mmproj) as model:
            info = model.vision
            print("tower:", info)
            resize = model.vision_resize(width, height)
            print("resize:", resize)
            runtime = model.native_runtime(context_limit=64, mtp_drafts=0)
            runtime.prepare()
            try:
                started = time.perf_counter()
                native = np.asarray(runtime.encode_image(pixels.tobytes(), width, height), dtype=np.float32)
                elapsed = time.perf_counter() - started
                started = time.perf_counter()
                native = np.asarray(runtime.encode_image(pixels.tobytes(), width, height), dtype=np.float32)
                warm = time.perf_counter() - started
            finally:
                runtime.close()
    native = native.reshape(-1, int(info["row_width"]))
    print(f"native: {native.shape}, first call {elapsed * 1000:.1f} ms, warm {warm * 1000:.1f} ms")
    if args.skip_reference:
        return 0
    started = time.perf_counter()
    expected = reference.encode(args.mmproj, pixels)
    print(f"reference: {expected.shape} in {time.perf_counter() - started:.1f} s")
    if expected.shape != native.shape:
        print(f"FAIL: shape {native.shape} != {expected.shape}")
        return 1
    difference = np.abs(native - expected)
    scale = np.abs(expected).max()
    worst = float(difference.max())
    print(f"max |diff| {worst:.3e} (reference max {scale:.3e}), "
          f"mean |diff| {float(difference.mean()):.3e}, "
          f"relative-to-max {worst / scale:.3e}")
    if worst / scale > args.tolerance:
        print("FAIL")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
