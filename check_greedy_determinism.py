#!/usr/bin/env python3
"""Run-to-run greedy determinism for a single configuration.

check_q8_decode_parity.py compares two *configurations* against each other.
This compares one configuration against itself across process restarts, which
is what catches nondeterminism sourced in run-varying state -- expert residency
built during prefill, expert cache occupancy, uninitialised scratch -- rather
than in a kernel swap. At temperature 0 the only way two runs diverge is that
the logits differed, so a mismatch localises to arithmetic rather than sampling.

Each sample is a fresh subprocess, but a fresh process does NOT mean isolated:
placement reads state that survives the process. Two such inputs are known --
`<model>.expert-history` next to the checkpoint (disable with
COLIBRI_EXPERT_HISTORY=off) and free VRAM sampled by the gpu_cache_bytes=0
auto-fit (pin with --option gpu_cache_bytes=...). Runs that leave either
uncontrolled measure the drift of that state, not the runtime.

    ./check_greedy_determinism.py MODEL.gguf --runs 3
    ./check_greedy_determinism.py MODEL.gguf --env COLIBRI_IQ2_Q8_DECODE=0
    ./check_greedy_determinism.py MODEL.gguf --option expert_residency=immutable
"""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter

PROMPT = "Explain how a turbocharger works, step by step."

# Counters worth correlating against a divergence. The memory note on this bug
# records that expert_cache_prompt_bypasses varies run to run even on
# checkpoints that stay deterministic, so a varying counter is evidence only
# when it moves *together* with the tokens.
COUNTERS = (
    "expert_cache_slots",
    "expert_cache_prompt_bypasses",
    "expert_cache_hits",
    "expert_cache_misses",
    "expert_cache_rejections",
    "expert_cache_deferred_admissions",
    "prefill_cache_seed_auto_skips",
)

# COLIBRI_DETERMINISM_TRAJECTORY=<path> makes the child dump per-token counter
# rows read from inside the generate() callback, which runs synchronously on
# the native loop. Diffing two trajectories finds the first token at which
# *placement* diverged, to compare against the first token at which *routing*
# diverged (COLIBRI_EXPERT_TRACE) -- whichever moves first is upstream.

CHILD = """
import json, os, sys
from colibri_next.v2 import V2Model, V2QwenRuntime

options = json.loads(os.environ["COLIBRI_DETERMINISM_OPTIONS"])
prompt = os.environ["COLIBRI_DETERMINISM_PROMPT"]
count = int(os.environ["COLIBRI_DETERMINISM_TOKENS"])

model = V2Model(os.environ["COLIBRI_DETERMINISM_MODEL"])
runtime = V2QwenRuntime(model, **options)
runtime.prepare()
tokens = []
trajectory = []
watched = json.loads(os.environ["COLIBRI_DETERMINISM_COUNTERS"])

def receive(token):
    # The callback runs synchronously on the native generate loop, so info
    # read here is the exact counter state after this token's placement
    # decisions -- a per-token trajectory with no rebuild.
    tokens.append(token)
    if os.environ.get("COLIBRI_DETERMINISM_TRAJECTORY"):
        info = runtime.info
        trajectory.append([int(info.get(k, 0)) for k in watched])
    return True

runtime.generate(model.tokenize(prompt), count, receive)
if os.environ.get("COLIBRI_DETERMINISM_TRAJECTORY"):
    with open(os.environ["COLIBRI_DETERMINISM_TRAJECTORY"], "w") as sink:
        json.dump({"counters": watched, "rows": trajectory}, sink)
info = {}
try:
    info = {k: v for k, v in runtime.info.items() if k in json.loads(
        os.environ["COLIBRI_DETERMINISM_COUNTERS"])}
except Exception:
    pass
print(json.dumps({"tokens": tokens, "info": info}))
"""


def sample(model, count, options, overrides):
    environment = dict(
        os.environ,
        PYTHONPATH="src",
        COLIBRI_DETERMINISM_MODEL=model,
        COLIBRI_DETERMINISM_PROMPT=PROMPT,
        COLIBRI_DETERMINISM_TOKENS=str(count),
        COLIBRI_DETERMINISM_OPTIONS=json.dumps(options),
        COLIBRI_DETERMINISM_COUNTERS=json.dumps(list(COUNTERS)),
        **overrides,
    )
    result = subprocess.run(
        [sys.executable, "-c", CHILD], capture_output=True, text=True,
        env=environment,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr[-2000:])
        raise SystemExit(f"sample failed with status {result.returncode}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def parse_pairs(values, cast=str):
    pairs = {}
    for value in values or ():
        if "=" not in value:
            raise SystemExit(f"expected KEY=VALUE, got {value!r}")
        key, _, raw = value.partition("=")
        pairs[key] = cast(raw)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--env", action="append", metavar="KEY=VALUE",
                        help="environment override, repeatable")
    parser.add_argument("--option", action="append", metavar="KEY=VALUE",
                        help="V2QwenRuntime keyword, repeatable")
    parser.add_argument("--dump", metavar="PATH",
                        help="append each run's full token list as a JSON line, "
                             "so runs from separate invocations can be diffed")
    arguments = parser.parse_args()

    overrides = parse_pairs(arguments.env)
    options = {k: coerce(v) for k, v in parse_pairs(arguments.option).items()}
    options.setdefault("context_limit", 2048)

    if overrides:
        print(f"env    : {overrides}")
    print(f"options: {options}")

    runs = []
    for index in range(arguments.runs):
        result = sample(arguments.model, arguments.tokens, options, overrides)
        runs.append(result)
        head = ",".join(map(str, result["tokens"][:12]))
        print(f"run {index}: {head}...  {result['info']}")
        if arguments.dump:
            with open(arguments.dump, "a") as sink:
                sink.write(json.dumps(
                    {"env": overrides, "options": options,
                     "tokens": result["tokens"], "info": result["info"]}
                ) + "\n")

    groups = Counter(tuple(r["tokens"]) for r in runs)
    if len(groups) == 1:
        print(f"OK: {arguments.runs} runs identical ({arguments.tokens} tokens)")
        return 0

    distinct = sorted(groups.items(), key=lambda kv: -kv[1])
    first = min(
        next((i for i, (a, b) in enumerate(zip(x, y)) if a != b), min(len(x), len(y)))
        for x in groups
        for y in groups
        if x != y
    )
    shape = "/".join(str(count) for _, count in distinct)
    print(f"NONDETERMINISTIC: {len(groups)} distinct outputs ({shape}), "
          f"first divergence at token {first}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
