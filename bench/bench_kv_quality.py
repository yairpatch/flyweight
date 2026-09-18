#!/usr/bin/env python3
"""What a quantized KV cache costs in *retrieval*, not in fluency.

`bench_kv_types.py` answers "what does a narrower cache buy" -- bytes, decode
rate, and whether the greedy continuation moved. What it cannot answer is
whether the continuation moved for the *worse*, and that is the whole reason
`auto` is not the default (v2.py:474): quantization damage at long context
concentrates in retrieval and instruction adherence, and both degrade well
before anything about the prose looks wrong. A bench that reports "DIFFERENT"
cannot distinguish a synonym from a forgotten fact.

So this one is a task eval. A needle sits at a known depth in a haystack of
numbered notes; the model is asked for it under a formatting instruction given
at position zero. Three numbers come out per arm:

  * retrieval -- did the exact needle value come back;
  * format    -- did the instruction from the top of the context survive to
                 the answer (this is what goes first, before fluency does);
  * divergence -- where the free-greedy continuation first parts from the
                 reference arm's, which is continuous and moves before either
                 accuracy figure does.

Two controls, because without them the numbers are not evidence:

  * A CONTROL ARM. The reference type is run a second time, last, as far from
    its own reference run as the session allows. Expert placement is an
    ambient input to the arithmetic (bench_topk_quality.py, and the
    greedy-nondeterminism note in plans/paged-kv-cache.md), so two identical
    configurations do not have to agree. Whatever the control loses is this
    box's noise floor, and a candidate arm inside it has shown nothing.

  * EQUAL EXPERT CACHE. `gpu_cache_bytes` is the *total* CUDA budget, so a
    narrower KV silently hands the difference to the expert cache -- which
    changes which experts are resident, which changes whether a routed token
    is computed on the GPU or the host, which changes the token. Pinning the
    budget across arms does not pin the comparison; it guarantees the arms
    differ by two things at once. The calibration pass below prepares each arm
    once, reads back what the expert cache actually got, and lowers the budget
    of the arms that got more until they match the stingiest one. Phase 1b in
    plans/paged-kv-cache.md is what happens without this.

Expert history is off for the same reason it is off in bench_kv_types.py, and
the prompt cache and extra slots stay at their defaults (off, one slot) so no
donation or restore lands in the middle of a measurement.

Usage:

    python bench/bench_kv_quality.py MODEL.gguf --contexts 8192,32768 \\
        --types q8_0,turbo4 --gpu-cache-mib 9000

    python bench/bench_kv_quality.py MODEL.gguf --preview   # grid + one prompt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass, field

from flyweight.v2 import V2Model

MIB = 1024 * 1024

# Digit-free operations prose. Every emitted note is numbered, so no two lines
# of the haystack are textually identical: a haystack that repeats one sentence
# is retrieved by induction rather than by attention, and reads far too easy.
NOTE_POOL = (
    "The overnight replication job finished without operator intervention.",
    "Cold aisle temperature held steady through the maintenance window.",
    "The west rack was reseated after a fan controller warning cleared itself.",
    "Backup verification reported no checksum mismatches for the period.",
    "A firmware advisory was filed for review at the next change board.",
    "The courier collected the sealed tape set from the loading dock.",
    "Ingress latency stayed within the agreed envelope across all regions.",
    "The standby generator completed its scheduled load test as planned.",
    "An access badge was reissued after the previous one failed to scan.",
    "Housekeeping cleared the staging area ahead of the delivery.",
    "The humidity sensor near the door was recalibrated by the contractor.",
    "A spare drive was pulled from inventory and logged against the cabinet.",
    "The night shift reported no alarms on the environmental panel.",
    "Cable management in the south row was tidied during the quiet period.",
    "The fire suppression inspection passed with no findings recorded.",
    "A visitor was escorted to the meeting room and signed out on departure.",
    "The parts cage was audited and reconciled against the manifest.",
    "Dust filters were replaced on the units flagged in the prior review.",
    "The loading bay door mechanism was lubricated and tested.",
    "A supplier confirmed the replacement part shipped from the depot.",
    "The security camera facing the corridor was realigned slightly.",
    "Shift handover notes were filed by the outgoing supervisor.",
    "The uninterruptible supply reported all cells within tolerance.",
    "A label printer was restocked ahead of the inventory count.",
    "The perimeter gate sensor was tested and responded normally.",
    "Floor tiles near the east wall were checked for movement.",
    "The chilled water loop pressure remained nominal all evening.",
    "An old monitor was removed from the operator desk and recycled.",
)

# Distinct carriers, one value shape. The shape matters: a needle that looks
# like the note numbering can be answered by copying the nearest index, and a
# plain integer is too easy to hit by accident over a few hundred tokens.
NEEDLES = (
    ("the Zurich vault access code", "K-73412"),
    ("the staging cluster passphrase", "Q-58109"),
    ("the archive retrieval token", "R-24875"),
    ("the night-shift escalation code", "M-96023"),
    ("the courier manifest reference", "T-31760"),
    ("the cold-storage unlock key", "B-64298"),
)

SYSTEM = (
    "You are reading an operations log. Answer only from the log. "
    "Reply with exactly one line in the form 'ANSWER: <value>' and nothing "
    "else -- no explanation, no preamble, no restatement of the question."
)

ANSWER_LINE = re.compile(r"^\s*ANSWER:\s*\S+", re.IGNORECASE)
DEPTH_NAMES = {0.05: "start", 0.25: "early", 0.5: "middle", 0.75: "late",
               0.95: "end"}


@dataclass
class Item:
    """One (needle, depth) question against one haystack."""
    needle: int
    depth: float
    subject: str
    value: str
    prompt: list[int] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"n{self.needle}@{DEPTH_NAMES.get(self.depth, f'{self.depth:.2f}')}"


@dataclass
class Outcome:
    retrieved: bool
    formatted: bool
    tokens: list[int]
    text: str
    divergence: int | None = None  # index of first disagreement with reference


def normalize(text: str) -> str:
    """Whitespace-insensitive, case-insensitive: a model that answers
    'K - 73412' has retrieved the needle and merely spaced it out."""
    return re.sub(r"\s+", "", text).upper()


def encode_prompt(tokenizer, body: str, subject: str) -> list[int]:
    """The templated prompt. The question names the subject, so the needle
    cannot be answered by returning whatever code-shaped string is nearest."""
    return tokenizer.encode_messages(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": body + f"\n\nWhat is {subject}? "
                                            f"Reply with the exact value from the log."}],
        enable_thinking=False,
    )


def build_haystack(tokenizer, model: V2Model, budget: int,
                   seed: int) -> tuple[list[str], list[int]]:
    """`budget`-ish tokens of numbered notes, and what each one cost.

    One haystack per context, shared by every item: a per-needle haystack would
    make each cell a different reading task, and the grid is meant to vary the
    needle and its depth, nothing else. The template/system/question overhead
    is measured rather than guessed -- it differs per checkpoint, and guessing
    it is how a 32k arm quietly becomes a 29k one.
    """
    overhead = len(encode_prompt(tokenizer, "", NEEDLES[0][0]))
    rng = random.Random(seed)
    notes: list[str] = []
    counts: list[int] = []
    total = 0
    # Each note is tokenized on its own, which is off by a token or two against
    # the joined string; the assembled prompt is measured again by the caller
    # and it is that figure the run reports.
    while total + overhead < budget:
        line = f"Note {len(notes) + 1}: {rng.choice(NOTE_POOL)}"
        cost = len(model.tokenize(line + "\n", capacity=256))
        notes.append(line)
        counts.append(cost)
        total += cost
    if not notes:
        raise SystemExit(
            f"context budget {budget} leaves no room for a haystack "
            f"(template overhead is {overhead} tokens)")
    return notes, counts


def place_needle(tokenizer, notes: list[str], counts: list[int],
                 item: Item) -> list[int]:
    """The haystack with the needle at `item.depth`, encoded.

    The insertion point is chosen by cumulative token count rather than by note
    count: the notes are not all the same length, and "halfway through the
    list" is not "halfway through the context".
    """
    target = item.depth * sum(counts)
    running = 0
    where = len(notes)
    for position, cost in enumerate(counts):
        if running >= target:
            where = position
            break
        running += cost
    # Suffixed rather than renumbered: a second "Note 400" would be the only
    # repeated number in the haystack, which is a landmark the needle should
    # not get for free.
    needle_line = (
        f"Note {where}a (for the record): {item.subject} is {item.value}."
    )
    return encode_prompt(
        tokenizer, "\n".join(notes[:where] + [needle_line] + notes[where:]),
        item.subject)


def generate(runtime, prompt: list[int], max_tokens: int,
             stop: tuple[int, ...]) -> list[int]:
    """Free greedy from a clean slot, stopping at end-of-turn.

    `reset()` between items on purpose: letting the runtime reuse a shared
    haystack prefix across items would be faster and would put the router,
    donation and checkpoint machinery inside a quality measurement.
    """
    out: list[int] = []
    runtime.reset()

    def receive(token: int):
        if token in stop:
            return False
        out.append(token)
        return len(out) < max_tokens

    runtime.generate(list(prompt), max_tokens, receive)
    return out


def teacher_forced(runtime, prompt: list[int], reference: list[int],
                   stop: tuple[int, ...]) -> float:
    """Top-1 agreement with the reference continuation, drift-free.

    The prompt is prefilled in one batch and only the continuation is stepped,
    so this costs one extra prefill per item rather than one decode per prompt
    token -- which at 32k is the difference between seconds and minutes.
    """
    if not reference:
        return 0.0
    first: list[int] = []
    runtime.reset()
    runtime.generate(list(prompt), 1, lambda token: first.append(token))
    agree = int(bool(first) and first[0] == reference[0])
    context = reference[0]
    for expected in reference[1:]:
        agree += int(runtime.decode(context) == expected)
        context = expected
    return agree / len(reference)


def score(item: Item, tokens: list[int], model: V2Model) -> Outcome:
    text = model.decode_tokens(tokens) if tokens else ""
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    return Outcome(
        retrieved=normalize(item.value) in normalize(text),
        formatted=bool(ANSWER_LINE.match(first_line)),
        tokens=tokens,
        text=text,
    )


def divergence(candidate: list[int], reference: list[int]) -> int | None:
    """First index where the two continuations part, or None if neither does
    within the shorter one. Free with the run, and it moves while retrieval is
    still pinned at 1.00."""
    for index, (a, b) in enumerate(zip(candidate, reference)):
        if a != b:
            return index
    return None if len(candidate) == len(reference) else min(
        len(candidate), len(reference))


def open_runtime(model: V2Model, arm: str, context: int, budget: int,
                 args: argparse.Namespace):
    return model.native_runtime(
        context_limit=context + args.slack,
        gpu_cache_bytes=budget,
        cache_type_k=arm, cache_type_v=arm,
        # The caller picked the context; a silently shrunk one would compare
        # two different experiments.
        context_explicit=True,
    )


def calibrate(model: V2Model, arms: list[str], context: int,
              args: argparse.Namespace) -> dict[str, int]:
    """Per-arm total budget that lands every arm on the same expert cache.

    Prepares each distinct cache type once and reads what the cache actually
    got. A narrower KV leaves more room, so the narrow arms give the surplus
    back rather than running with an advantage the comparison cannot separate
    from precision.
    """
    base = args.gpu_cache_mib * MIB
    observed: dict[str, tuple[int, int]] = {}
    for arm in dict.fromkeys(arms):
        with open_runtime(model, arm, context, base, args) as runtime:
            runtime.prepare()
            info = runtime.info
            observed[arm] = (int(info["expert_cache_bytes"]),
                             int(info["kv_reserved_bytes"]))
        print(f"  calibrate {arm:7s}: KV {observed[arm][1] // MIB:6d} MiB, "
              f"expert cache {observed[arm][0] // MIB:6d} MiB", flush=True)
    caches = [cache for cache, _ in observed.values()]
    if not any(caches):
        return {arm: base for arm in observed}  # dense model: nothing to equalize
    floor = min(caches)
    return {arm: base - (cache - floor) for arm, (cache, _) in observed.items()}


def run_arm(model: V2Model, tokenizer, arm: str, label: str, items: list[Item],
            context: int, budget: int, reference: dict[str, list[int]],
            args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    outcomes: dict[str, Outcome] = {}
    agreements: list[float] = []
    print(f"  {label:9s} ", end="", flush=True)
    with open_runtime(model, arm, context, budget, args) as runtime:
        runtime.prepare()
        info = dict(runtime.info)
        for item in items:
            tokens = generate(runtime, item.prompt, args.max_tokens,
                              tokenizer.eos_token_ids)
            outcome = score(item, tokens, model)
            if reference:
                outcome.divergence = divergence(tokens, reference[item.label])
                if args.teacher_forced:
                    agreements.append(teacher_forced(
                        runtime, item.prompt, reference[item.label],
                        tokenizer.eos_token_ids))
            outcomes[item.label] = outcome
            mark = "." if outcome.retrieved else "X"
            if not outcome.formatted:
                mark = mark.lower() if outcome.retrieved else "#"
            print(mark, end="", flush=True)
    elapsed = time.perf_counter() - started
    retrieval = sum(o.retrieved for o in outcomes.values()) / len(outcomes)
    formatted = sum(o.formatted for o in outcomes.values()) / len(outcomes)
    diverged = [o.divergence for o in outcomes.values() if o.divergence is not None]
    print(f"  retrieval {retrieval:5.0%}  format {formatted:5.0%}  "
          f"({elapsed:5.1f}s)", flush=True)
    return {
        "arm": arm,
        "label": label,
        "retrieval": retrieval,
        "format": formatted,
        "identical": sum(o.divergence is None for o in outcomes.values()),
        "divergence": statistics.median(diverged) if diverged else None,
        "agreement": statistics.mean(agreements) if agreements else None,
        "kv_mib": int(info["kv_reserved_bytes"]) // MIB,
        "expert_cache_mib": int(info["expert_cache_bytes"]) // MIB,
        "expert_slots": int(info["expert_cache_slots"]),
        "seconds": elapsed,
        "outcomes": outcomes,
    }


def by_depth(rows: list[dict[str, object]], items: list[Item]) -> None:
    """Retrieval per depth. The aggregate hides the shape of the failure, and
    the shape is the finding: a cache that has lost the middle of the context
    while both ends still answer is a different problem from one that has lost
    the instruction."""
    depths = sorted({item.depth for item in items})
    print("\n  retrieval by depth")
    print(f"  {'arm':>10}  " + "  ".join(
        f"{DEPTH_NAMES.get(d, f'{d:.2f}'):>7}" for d in depths))
    for row in rows:
        outcomes = row["outcomes"]
        cells = []
        for depth in depths:
            hits = [outcomes[item.label].retrieved
                    for item in items if item.depth == depth]
            cells.append(f"{sum(hits) / len(hits):7.0%}")
        print(f"  {row['label']:>10}  " + "  ".join(cells))


def report(context: int, rows: list[dict[str, object]], reference: str,
           control: str | None) -> None:
    print(f"\n  context {context}")
    print(f"  {'arm':>10}  {'KV':>9}  {'experts':>14}  {'retrieval':>9}  "
          f"{'format':>7}  {'identical':>9}  {'diverge@':>8}  {'agree':>6}")
    for row in rows:
        agree = f"{row['agreement']:6.1%}" if row["agreement"] is not None else "     -"
        diverge = f"{row['divergence']:8.0f}" if row["divergence"] is not None else "       -"
        print(f"  {row['label']:>10}  {row['kv_mib']:5d} MiB  "
              f"{row['expert_slots']:5d} / {row['expert_cache_mib']:5d} MiB  "
              f"{row['retrieval']:9.0%}  {row['format']:7.0%}  "
              f"{row['identical']:9d}  {diverge}  {agree}")

    slots = {row["expert_slots"] for row in rows}
    if len(slots) > 1:
        print(f"  !! expert cache differs across arms ({sorted(slots)}): the "
              f"arms differ by placement as well as by precision")

    # A ceiling is not a pass. Every arm at 100% means this context could not
    # tell them apart, which is the expected result well below the regime the
    # question is about -- and exactly the reading that would wrongly promote a
    # quantized default off an 8k run.
    if all(row["retrieval"] == 1.0 and row["format"] == 1.0 for row in rows):
        identical = all(row["identical"] == len(row["outcomes"]) for row in rows)
        print(f"  every arm at ceiling: {context} tokens does not discriminate"
              + (" (and every continuation was identical, so the caches agree "
                 "bit for bit here)" if identical else
                 " on the task, though the continuations already differ"))

    if control is None:
        print("  no control arm: run with --control to get this box's noise floor")
        return
    base = next(r for r in rows if r["label"] == reference)
    check = next(r for r in rows if r["label"] == control)
    floor_retrieval = abs(base["retrieval"] - check["retrieval"])
    floor_format = abs(base["format"] - check["format"])
    print(f"\n  noise floor, same config twice: retrieval {floor_retrieval:.0%}, "
          f"format {floor_format:.0%}, {check['identical']}/{len(base['outcomes'])} "
          f"continuations identical to the reference")
    for row in rows:
        if row["label"] in (reference, control):
            continue
        lost_retrieval = base["retrieval"] - row["retrieval"]
        lost_format = base["format"] - row["format"]
        verdict = (
            "INSIDE the noise floor -- shows nothing either way"
            if lost_retrieval <= floor_retrieval and lost_format <= floor_format
            else f"retrieval {-lost_retrieval:+.0%}, format {-lost_format:+.0%} "
                 f"vs {reference}, outside the floor")
        print(f"  {row['label']:>10}: {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--types", default="q8_0,turbo4",
                        help="candidate cache types, against --reference")
    parser.add_argument("--reference", default="f16",
                        help="the arm every candidate is scored against")
    parser.add_argument("--contexts", default="8192,32768",
                        help="prompt budgets to sweep; quantization damage is a "
                             "long-context effect, so one point proves nothing")
    parser.add_argument("--depths", default="0.05,0.25,0.5,0.75,0.95",
                        help="needle positions as a fraction of the haystack")
    parser.add_argument("--needles", type=int, default=3,
                        help=f"distinct needles per depth (max {len(NEEDLES)}); "
                             "items per cell is what sets the resolution")
    parser.add_argument("--gpu-cache-mib", type=int, default=9000,
                        help="total CUDA budget, pinned across arms before "
                             "calibration; auto-fit drifts run to run and would "
                             "make the arms differ by expert placement")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--slack", type=int, default=512,
                        help="context headroom over the prompt budget")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--control", action="store_true", default=True,
                        help="re-run the reference type last as a noise floor")
    parser.add_argument("--no-control", dest="control", action="store_false")
    parser.add_argument("--equalize", action="store_true", default=True,
                        help="calibrate per-arm budgets to an equal expert cache")
    parser.add_argument("--no-equalize", dest="equalize", action="store_false")
    parser.add_argument("--teacher-forced", action="store_true",
                        help="also measure drift-free top-1 agreement with the "
                             "reference continuation; one extra prefill per item")
    parser.add_argument("--preview", action="store_true",
                        help="print the grid and one assembled prompt, run nothing")
    parser.add_argument("--jsonl", help="append per-arm results here")
    args = parser.parse_args()

    if args.needles > len(NEEDLES):
        raise SystemExit(f"--needles at most {len(NEEDLES)}")
    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]
    depths = [float(d) for d in args.depths.split(",") if d.strip()]
    candidates = [t.strip() for t in args.types.split(",") if t.strip()]

    os.environ["FLYWEIGHT_EXPERT_HISTORY"] = "0"
    model = V2Model(args.model)
    try:
        from flyweight.v2_server import NativeV2Tokenizer
    except ImportError as error:  # jinja2 is a server dependency
        raise SystemExit(f"chat templating unavailable ({error}); "
                         "install the server extras") from None
    tokenizer = NativeV2Tokenizer(model)

    for context in contexts:
        items = [
            Item(needle=n, depth=d, subject=NEEDLES[n][0], value=NEEDLES[n][1])
            for d in depths for n in range(args.needles)
        ]
        notes, counts = build_haystack(tokenizer, model, context, args.seed)
        for item in items:
            item.prompt = place_needle(tokenizer, notes, counts, item)
        lengths = [len(item.prompt) for item in items]
        print(f"\ncontext {context}: {len(items)} items, prompts "
              f"{min(lengths)}-{max(lengths)} tokens, {len(depths)} depths x "
              f"{args.needles} needles", flush=True)

        if args.preview:
            text = model.decode_tokens(items[len(items) // 2].prompt)
            print(f"--- head ---\n{text[:600]}\n--- tail ---\n{text[-600:]}")
            continue

        arms = [(args.reference, args.reference)]
        arms += [(arm, arm) for arm in candidates]
        control = None
        if args.control:
            # Last, and as far from its own reference run as the session goes:
            # a noise floor measured back to back understates the drift that a
            # real arm sitting in the middle of the session is exposed to.
            control = f"{args.reference}'"
            arms.append((args.reference, control))

        budgets = {arm: args.gpu_cache_mib * MIB for arm, _ in arms}
        if args.equalize:
            budgets = calibrate(model, [arm for arm, _ in arms], context, args)

        reference: dict[str, list[int]] = {}
        rows: list[dict[str, object]] = []
        for arm, label in arms:
            row = run_arm(model, tokenizer, arm, label, items, context,
                          budgets[arm], reference, args)
            if label == args.reference:
                reference = {name: outcome.tokens
                             for name, outcome in row["outcomes"].items()}
            rows.append(row)
            if args.jsonl:
                with open(args.jsonl, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps({
                        k: v for k, v in row.items() if k != "outcomes"
                    } | {"context": context, "model": args.model}) + "\n")

        report(context, rows, args.reference, control)
        by_depth(rows, items)

    model.close()


if __name__ == "__main__":
    sys.exit(main())
