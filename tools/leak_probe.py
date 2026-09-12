"""Dynamic leak probe: RSS and CUDA device usage per iteration."""
import ctypes
import os
import sys
import gc
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.dense_gguf_fixture import build_dense_qwen35_gguf  # noqa: E402
from flyweight.v2 import V2Model, TASK_EVENT_DONE, TASK_EVENT_ERROR  # noqa: E402

def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
def dev_used_mb():
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
        return (total.value - free.value) / 2**20
    except Exception:
        return float("nan")

def run_task(rt, prompt, n, cancel_after=None):
    tid = rt.task_submit(prompt, n, temperature=0.7, seed=1)
    got = 0
    while True:
        for t, tok, kind in rt.engine_step():
            if t != tid:
                continue
            if kind == TASK_EVENT_ERROR:
                raise RuntimeError(rt.task_error(tid))
            if kind == TASK_EVENT_DONE:
                return got
            if kind == 0:
                got += 1
                if cancel_after and got >= cancel_after:
                    rt.task_cancel(tid)
                    cancel_after = None
        # drain until done

def report(tag, i):
    print(f"{tag:14s} it={i:3d} rss={rss_mb():8.1f}MB dev={dev_used_mb():8.1f}MB", flush=True)

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    backend = os.environ.get("PROBE_BACKEND", "auto")
    V2Model.select_backend(backend)
    with TemporaryDirectory() as d:
        path = Path(d) / "m.gguf"
        build_dense_qwen35_gguf(path, quantize="q8_0")
        opts = dict(context_limit=512, expert_mode="cpu", mtp_drafts=0)
        if mode in ("all", "reload"):
            for i in range(iters):
                with V2Model(path) as model:
                    with model.native_runtime(**opts) as rt:
                        rt.prepare()
                        run_task(rt, [(k*7+3)%64 for k in range(40)], 8)
                gc.collect()
                report("reload", i)
        if mode in ("all", "tasks", "cancel", "lookup"):
            os.environ.setdefault("FLYWEIGHT_LOOKUP_DRAFTS", "4" if mode == "lookup" else "0")
            with V2Model(path) as model, model.native_runtime(**opts) as rt:
                rt.prepare()
                for i in range(iters):
                    plen = 20 + (i * 37) % 300
                    prompt = [(k*7+i+3)%64 for k in range(plen)]
                    run_task(rt, prompt, 32, cancel_after=5 if mode == "cancel" else None)
                    gc.collect()
                    report(mode, i)
                    print("   plen", plen)
main()
