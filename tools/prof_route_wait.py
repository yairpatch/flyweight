"""Is route_wait GPU compute, or a bubble?

route_wait is a host-side `cuMemcpy`-free event sync: the host blocks until the
device reaches the point where this layer's routes have been downloaded. Which
means it waits for EVERYTHING enqueued before it -- this layer's attention or
DeltaNet block, the router projection, the top-k, and the small downloads.

So the counter alone cannot say whether the host is idling on a slow
round-trip (fixable by overlap) or simply watching the GPU do the layer's work
(fixable only by faster kernels). FLYWEIGHT_CUDA_PROFILE=1 measures the device
side of the same region with CUDA events; comparing them settles it.
"""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import sys

MODEL = sys.argv[1]
TOKENS = 64

DRIVER = r'''
import os, time
from flyweight.v2 import V2Model
model = V2Model(os.environ["FLYWEIGHT_MODEL"])
rt = model.native_qwen_runtime(context_limit=8192, gpu_cache_bytes=8500*1024**2,
                               moe_device="hybrid")
rt.prepare()
prompt = list(model.tokenize(
    "Explain how mixture-of-experts routing works and why expert weights "
    "dominate decode latency.\n\n", capacity=8192))
out = []
rt.generate(prompt, 48, out.append)      # warm: clocks, experts, graphs
before = {f: int(rt.info[f]) for f in
          ("decode_calls", "decode_nanoseconds", "route_wait_nanoseconds",
           "expert_compute_nanoseconds", "expert_page_nanoseconds",
           "tail_wait_nanoseconds")}
out.clear()
rt.generate(prompt, %d, out.append)
after = {f: int(rt.info[f]) for f in before}
d = {f: after[f] - before[f] for f in before}
n = max(1, d["decode_calls"])
print("HOST decode_ms=%%.3f route_wait_ms=%%.3f cpu_experts_ms=%%.3f "
      "page_ms=%%.3f tail_ms=%%.3f" %% (
      d["decode_nanoseconds"]/n/1e6, d["route_wait_nanoseconds"]/n/1e6,
      d["expert_compute_nanoseconds"]/n/1e6, d["expert_page_nanoseconds"]/n/1e6,
      d["tail_wait_nanoseconds"]/n/1e6))
''' % TOKENS

env = dict(os.environ, FLYWEIGHT_MODEL=MODEL, FLYWEIGHT_TIMING="1",
           FLYWEIGHT_CUDA_PROFILE="1")
proc = subprocess.run([sys.executable, "-c", DRIVER], env=env,
                      capture_output=True, text=True)
if proc.returncode:
    print(proc.stdout[-3000:])
    print(proc.stderr[-3000:])
    raise SystemExit(proc.returncode)

# Device-side per-token phase times. Keep only the profiled region's tail, so
# the warm generation's lines cannot skew it.
pattern = re.compile(
    r"delta=([\d.]+)ms/\d+ \(recurrent=([\d.]+)ms\) "
    r"attention=([\d.]+)ms/\d+ \(core=([\d.]+)ms\) "
    r"shared=([\d.]+)ms expert=([\d.]+)ms tail=([\d.]+)ms \(lm=([\d.]+)ms\) "
    r"total=([\d.]+)ms")
rows = [tuple(float(x) for x in m.groups())
        for m in pattern.finditer(proc.stderr)][-TOKENS:]
if not rows:
    raise SystemExit("no [cuda-profile] lines: is FLYWEIGHT_CUDA_PROFILE wired?")

names = ("delta", "recurrent", "attention", "attn_core", "shared", "expert",
         "tail", "lm", "gpu_total")
median = {n: statistics.median(r[i] for r in rows)
          for i, n in enumerate(names)}

print(proc.stdout.strip())
print(f"GPU  (median of {len(rows)} tokens, ms)")
for name in names:
    print(f"       {name:>10} {median[name]:7.3f}")
print()
print(f"route_wait waits on the pre-route region = delta + attention = "
      f"{median['delta'] + median['attention']:.3f} ms of GPU work")
