"""Prompt-lookup drafting (FLYWEIGHT_LOOKUP_DRAFTS) off vs on, interleaved.

Two prompts: one whose answer echoes its own input (a code rewrite, the
regime lookup drafting exists for) and one that does not (free prose, the
regime where it must not lose). Greedy decode is bit-identical by
construction, so the token streams are compared exactly. Lookup mode is
fixed at runtime creation, so each arm builds its own runtime; the arms
alternate so page-cache and clock drift land on both.

    python bench_lookup.py <model.gguf> [--drafts 4] [--rounds 2] [--tokens 320]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from flyweight.v2 import V2Model

CODE = '''\
def load_records(path, delimiter=","):
    records = []
    with open(path) as handle:
        header = handle.readline().rstrip("\\n").split(delimiter)
        for line in handle:
            fields = line.rstrip("\\n").split(delimiter)
            if len(fields) != len(header):
                continue
            records.append(dict(zip(header, fields)))
    return records


def summarize(records, key):
    totals = {}
    counts = {}
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        totals[value] = totals.get(value, 0.0) + number
        counts[value] = counts.get(value, 0) + 1
    return {name: totals[name] / counts[name] for name in totals}


def write_report(summary, path):
    with open(path, "w") as handle:
        for name in sorted(summary):
            handle.write(f"{name}: {summary[name]:.3f}\\n")
'''

PROMPTS = {
    "code": (
        "Add type hints and a one-line docstring to every function in this "
        "file. Output the complete file, unchanged otherwise, in one code "
        "block.\n\n```python\n" + CODE + "```"
    ),
    "prose": (
        "Explain how speculative decoding works in large language models, "
        "covering draft models, verification, acceptance rates, and why it "
        "speeds up autoregressive generation."
    ),
}

FIELDS = (
    "decode_calls", "decode_nanoseconds",
    "mtp_draft_tokens", "mtp_accepted_tokens", "mtp_rejected_tokens",
    "mtp_verify_nanoseconds", "mtp_rollback_nanoseconds",
)


def chat(text: str) -> str:
    return (
        "<|im_start|>user\n" + text + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def run_once(runtime, prompt: list[int], tokens: int):
    runtime.reset()
    before = {f: runtime.info[f] for f in FIELDS}
    out: list[int] = []
    started = time.perf_counter()
    runtime.generate(prompt, tokens, out.append)
    elapsed = time.perf_counter() - started
    delta = {f: runtime.info[f] - before[f] for f in FIELDS}
    return out, elapsed, delta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--drafts", type=int, default=4)
    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=320)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--moe-device", default=None,
                        help="cpu makes greedy output deterministic run to run (auto/hybrid "
                             "round GPU-resident and CPU experts differently)")
    args = parser.parse_args()

    results: dict[tuple[str, int, int], list] = {}
    streams: dict[tuple[str, int], list[int]] = {}
    with V2Model(args.model) as model:
        prompts = {name: model.tokenize(chat(text)) for name, text in PROMPTS.items()}
        for name, ids in prompts.items():
            print(f"prompt {name}: {len(ids)} tokens", flush=True)
        for round_index in range(args.rounds):
            for drafts in (0, args.drafts):
                os.environ["FLYWEIGHT_LOOKUP_NGRAM_MIN"] = str(args.ngram_min)
                if drafts:
                    os.environ["FLYWEIGHT_LOOKUP_DRAFTS"] = str(drafts)
                else:
                    os.environ.pop("FLYWEIGHT_LOOKUP_DRAFTS", None)
                with model.native_qwen_runtime(context_limit=args.context,
                                               moe_device=args.moe_device) as runtime:
                    runtime.prepare()
                    # Warm-up: cold expert residency after a fresh runtime.
                    run_once(runtime, prompts["prose"], 32)
                    for name, ids in prompts.items():
                        out, elapsed, delta = run_once(runtime, ids, args.tokens)
                        tok_s = len(out) / elapsed
                        results.setdefault((name, drafts, round_index), [out, tok_s, delta])
                        key = (name, drafts)
                        streams.setdefault(key, out)
                        line = f"[round {round_index}] {name:5s} drafts={drafts}: {tok_s:6.2f} tok/s  ({len(out)} tokens, {elapsed:.1f}s)"
                        if drafts:
                            d = delta
                            drafted = d["mtp_draft_tokens"]
                            acc = d["mtp_accepted_tokens"]
                            line += (f"  drafted={drafted} accepted={acc} "
                                     f"accept={acc / max(1, drafted):.0%} "
                                     f"verify={d['mtp_verify_nanoseconds'] / 1e6:.0f}ms "
                                     f"rollback={d['mtp_rollback_nanoseconds'] / 1e6:.0f}ms")
                        print(line, flush=True)
                        if (name, 0) in streams and out != streams[(name, 0)]:
                            base = streams[(name, 0)]
                            first = next(i for i in range(min(len(out), len(base))) if out[i] != base[i])
                            print(f"   !!! OUTPUT DIFFERS FROM BASELINE (first at token {first})", flush=True)
    print("\n=== summary ===")
    for name in PROMPTS:
        base = [v[1] for k, v in results.items() if k[0] == name and k[1] == 0]
        look = [v[1] for k, v in results.items() if k[0] == name and k[1] != 0]
        same = streams.get((name, 0)) == streams.get((name, args.drafts))
        print(f"  {name:5s}: baseline {sum(base) / len(base):.2f} tok/s  lookup {sum(look) / len(look):.2f} tok/s  "
              f"({(sum(look) / len(look)) / (sum(base) / len(base)) - 1:+.1%})  output {'identical' if same else 'DIFFERENT'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
