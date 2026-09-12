"""Where does the 9.3 ms/token of GPU expert time go: uploads, or the kernel?

The device-side `expert` phase brackets both the H2D uploads of newly admitted
expert bundles and the grouped expert GEMM that consumes them. The existing
[moe-profile] instrument measures HOST-serial dispatch work (that is how the
"95-97% staging memcpy" reading was taken) and the CUDA profile measures the
device timeline; neither alone splits the device phase.

Together they do: staged MiB/token at the measured H2D rate gives the upload
share of the device phase, and the remainder is the kernel. Which of the two
dominates decides what to attack -- paging format and residency, or the
grouped-kernel shape.

Warmup is charged into the [moe-profile] averages (it accumulates from process
start), so the measured run is made long enough to dilute the cold-cache
staging burst rather than pretend it is not there.
"""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import sys

MODEL = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/yair/Downloads/gguf/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf")
WARM = 48
MEASURE = 256
# Measured host-to-device rate for registered mappings on this box.
H2D_GIB_PER_S = 26.0

DRIVER = f'''
import os
from flyweight.v2 import V2Model
model = V2Model({MODEL!r})
rt = model.native_qwen_runtime(context_limit=8192, gpu_cache_bytes=8500*1024**2,
                               moe_device="hybrid")
rt.prepare()
prompt = list(model.tokenize(
    "Explain how mixture-of-experts routing works and why expert weights "
    "dominate decode latency.\\n\\n", capacity=8192))
out = []
rt.generate(prompt, {WARM}, out.append)
FIELDS = ("decode_calls", "decode_nanoseconds", "expert_page_nanoseconds",
          "expert_cache_hits", "expert_cache_misses", "route_expert_sum")
before = {{f: int(rt.info[f]) for f in FIELDS}}
out.clear()
rt.generate(prompt, {MEASURE}, out.append)
after = {{f: int(rt.info[f]) for f in FIELDS}}
d = {{f: after[f] - before[f] for f in FIELDS}}
n = max(1, d["decode_calls"])
info = rt.info
slots = max(1, int(info["expert_cache_slots"]))
print("HOST decode_ms=%.3f page_ms=%.3f over %d tokens" % (
    d["decode_nanoseconds"]/n/1e6, d["expert_page_nanoseconds"]/n/1e6, n))
# What the grouped kernel actually reads: the experts that were resident, times
# the bytes one expert bundle occupies in the cache.
print("WORK gpu_experts_per_token=%.1f routed_per_token=%.1f "
      "expert_bytes=%d" % (
      d["expert_cache_hits"]/n, d["route_expert_sum"]/n,
      int(info["expert_cache_bytes"])//slots))
'''

env = dict(os.environ, FLYWEIGHT_TIMING="1", FLYWEIGHT_CUDA_PROFILE="1",
           FLYWEIGHT_MOE_PROFILE="1")
proc = subprocess.run([sys.executable, "-c", DRIVER], env=env,
                      capture_output=True, text=True)
if proc.returncode:
    print(proc.stdout[-2000:], proc.stderr[-4000:])
    raise SystemExit(proc.returncode)

gpu = re.compile(r"shared=([\d.]+)ms expert=([\d.]+)ms tail=([\d.]+)ms "
                 r"\(lm=[\d.]+ms\) total=([\d.]+)ms")
rows = [tuple(float(x) for x in m.groups())
        for m in gpu.finditer(proc.stderr)][-MEASURE:]
staged = re.findall(r"([\d.]+) MiB staged per token", proc.stderr)
host = re.findall(r"bundle staging memcpy\s+([\d.]+) ms/token", proc.stderr)
launch = re.findall(r"kernel launch \+ uploads\s+([\d.]+) ms/token", proc.stderr)

print(proc.stdout.strip())
if not rows:
    raise SystemExit("no [cuda-profile] lines")
expert_ms = statistics.median(r[1] for r in rows)
total_ms = statistics.median(r[3] for r in rows)
print(f"GPU  expert {expert_ms:6.3f} ms/token of {total_ms:6.3f} ms total "
      f"({100 * expert_ms / total_ms:.0f}%)")
if staged:
    mib = float(staged[-1])
    upload_ms = mib / 1024.0 / H2D_GIB_PER_S * 1000.0
    print(f"     staged {mib:6.2f} MiB/token "
          f"=> {upload_ms:6.3f} ms at {H2D_GIB_PER_S:.0f} GiB/s H2D "
          f"({100 * upload_ms / expert_ms:.0f}% of the expert phase)")
    print(f"     residual (grouped kernel etc) {expert_ms - upload_ms:6.3f} ms "
          f"({100 * (expert_ms - upload_ms) / expert_ms:.0f}%)")
if host:
    print(f"HOST staging memcpy {host[-1]} ms/token, "
          f"launch+uploads {launch[-1] if launch else '?'} ms/token")

# The reason this matters: a routed matvec is a weight-streaming problem, so the
# ceiling is device bandwidth. How close is the kernel to it?
work = re.search(r"gpu_experts_per_token=([\d.]+) routed_per_token=([\d.]+) "
                 r"expert_bytes=(\d+)", proc.stdout)
if work:
    gpu_experts, routed, expert_bytes = (float(work.group(1)),
                                         float(work.group(2)),
                                         int(work.group(3)))
    read_gib = gpu_experts * expert_bytes / 1024 ** 3
    achieved = read_gib / (expert_ms / 1000.0)
    print()
    print(f"     {gpu_experts:.0f} resident experts/token x "
          f"{expert_bytes / 1024 ** 2:.2f} MiB = {read_gib * 1024:.0f} MiB read")
    print(f"     {achieved:6.1f} GiB/s effective vs ~391 GiB/s device wall "
          f"({100 * achieved / 391:.0f}% of peak)")
    print(f"     at the wall that read would take "
          f"{read_gib / 391 * 1000:.2f} ms, not {expert_ms:.2f} ms")
