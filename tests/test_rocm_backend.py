"""The ROCm backend: AMD RDNA GPUs through HIP and hipRTC.

Selection and probing run everywhere. The device tests need a wave32 AMD GPU
with the HIP runtime installed and skip otherwise; they run in a subprocess,
because the GPU platform is process-global -- once a HIP context exists, CUDA
cannot load in the same process (and the reverse), so running them in the
test process would move every later CUDA test onto the AMD card.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from flyweight.v2 import V2Model

from tests.dense_gguf_fixture import build_dense_qwen35_gguf

ROOT = Path(__file__).resolve().parents[1]


def _in_subprocess(script: str, *args: str, timeout: float = 900) -> dict:
    """Run `script` in a fresh interpreter; it prints one JSON object last."""
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), *args],
        capture_output=True, text=True, timeout=timeout, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(
            filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))},
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"subprocess failed ({completed.returncode}):\n{completed.stderr[-4000:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _rocm_device_available() -> bool:
    try:
        return bool(_in_subprocess("""
            import json
            from flyweight.v2 import V2Model
            V2Model.select_backend("rocm")
            print(json.dumps({"available": V2Model.gpu_info(0)["available"]}))
        """, timeout=120)["available"])
    except Exception:  # noqa: BLE001 - no ROCm is a skip, not a failure
        return False


ROCM = _rocm_device_available()


class RocmSelectionTests(unittest.TestCase):
    def tearDown(self) -> None:
        V2Model.select_backend("auto")

    def test_rocm_round_trips(self) -> None:
        self.assertEqual(V2Model.select_backend("rocm"), "rocm")
        self.assertEqual(V2Model.active_backend(), "rocm")

    def test_the_probe_names_the_backend_it_answered_for(self) -> None:
        V2Model.select_backend("rocm")
        info = V2Model.gpu_info(0)
        self.assertIn(info["platform"], {"rocm", "none"})
        if info["platform"] == "none":
            self.assertEqual(info["available"], 0)
        else:
            self.assertRegex(info["arch"], r"^gfx[0-9a-f]+")
            # The NVIDIA tensor-core paths key off the compute capability; a
            # ROCm device must never report one.
            self.assertEqual((info["compute_major"], info["compute_minor"]), (0, 0))
            self.assertEqual(bool(info["available"]), info["warp_size"] == 32)

    def test_cuda_reports_its_own_platform(self) -> None:
        V2Model.select_backend("cuda")
        info = V2Model.gpu_info(0)
        if not info["available"]:
            self.skipTest("no CUDA device")
        self.assertEqual(info["platform"], "cuda")
        self.assertEqual(info["arch"], f"sm_{info['compute_major']}{info['compute_minor']}")


@unittest.skipUnless(ROCM, "test requires a wave32 AMD GPU with the HIP runtime")
class RocmDeviceTests(unittest.TestCase):
    def test_every_corpus_compiles_under_hiprtc(self) -> None:
        # Qwen/Bailing/diffusion share one corpus; DeepSeek-V4 adds its own.
        # The runtime's own assembly is what matters, so this goes through
        # flyweight_gpu_compile: prelude, include guards and the named-kernel
        # table that must all resolve in the loaded code object.
        result = _in_subprocess("""
            import json, re, sys
            from pathlib import Path
            from flyweight.v2 import V2Model
            chunk = re.compile(r'R"FLYWEIGHT_CUDA\\((.*?)\\)FLYWEIGHT_CUDA"', re.DOTALL)
            def corpus(*names):
                return "".join(
                    part for name in names
                    for part in chunk.findall(Path(
                        f"native/include/flyweight_v2_{name}_kernels.hpp").read_text()))
            # DeepSeek-V4 splices a generated IQ1_S grid between the shared
            # corpus and its own kernels; its values do not matter here.
            grid = ('extern "C" __device__ __constant__ unsigned long long '
                    "ds4_iq1s_grid[2048]={" + "0ULL," * 2048 + "};\\n")
            V2Model.select_backend("rocm")
            V2Model.gpu_prepare(corpus("qwen", "native", "diffusion"), 0, [])
            V2Model.gpu_prepare(corpus("qwen", "native") + grid + corpus("deepseek4"), 0, [])
            print(json.dumps({"ok": True}))
        """)
        self.assertTrue(result["ok"])

    def test_a_dense_model_decodes_like_the_cpu_backend(self) -> None:
        # The CPU backend runs the same corpus with the same kernel choices
        # (neither has NVIDIA tensor cores), so greedy tokens agree exactly
        # unless the shim mistranslates something.
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dense.gguf"
            build_dense_qwen35_gguf(path, quantize="q8_0")
            script = """
                import json, sys
                from flyweight.v2 import V2Model
                V2Model.select_backend(sys.argv[1])
                with V2Model(sys.argv[2]) as model:
                    with model.native_qwen_runtime(context_limit=256) as runtime:
                        runtime.prepare()
                        prompt = [(index * 7 + 3) % 64 for index in range(40)]
                        produced = []
                        runtime.generate(prompt, 12, produced.append)
                print(json.dumps({"tokens": produced}))
            """
            rocm = _in_subprocess(script, "rocm", str(path))["tokens"]
            cpu = _in_subprocess(script, "cpu", str(path))["tokens"]
        self.assertEqual(len(rocm), 12)
        self.assertEqual(rocm, cpu)


if __name__ == "__main__":
    unittest.main()
