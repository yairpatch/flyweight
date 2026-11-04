"""`flyweight tune`: find the fastest serve settings for one model on this machine.

Every measurement is a real `flyweight serve`, started with a candidate's
flags and timed over HTTP the way a client sees it: a warm-up request, then
streamed requests whose time to first token gives the prefill rate and whose
remaining tokens give the decode rate. Each prompt opens with a fresh nonce,
so the prompt cache can never stand in for prefill.

The search is coordinate descent from the settings it was given: one setting
at a time, each value tried against the best so far, a win kept only when it
holds up when measured again beside the incumbent (a fresh load places
experts differently, so one run of noise must not move the answer). Only
settings that apply to this model and this machine are tried, most
consequential first, until the time limit. Approximate settings
(--route-prior, --route-tail-skip) are never tried.

The winner is then loaded once more and given a prompt that fills the
context, so the answer is a setup that holds the context that was asked for,
not only the shorter benchmark prompt.
"""
from __future__ import annotations

import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

# Weight of prefill in the score; decode gets the rest.
GOALS = {"balanced": 0.5, "prefill": 0.8, "decode": 0.2}

# Options a candidate never carries over from the settings it was given:
# where the server listens is the tuner's business.
_NOT_PASSED = {"host", "port", "context_window"}


@dataclass
class Measurement:
    ok: bool
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    ttft: float = 0.0
    load_seconds: float = 0.0
    prompt_tokens: int = 0
    decode_tokens: int = 0
    error: str = ""
    health: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class Knob:
    """One setting the search may move: its serve dest and the values to try."""
    name: str
    dest: str
    values: list[Any]
    flag: str
    # Other dests this one supersedes, dropped when it is set (kv_dtype sets
    # both cache types).
    replaces: tuple[str, ...] = ()
    # serve's own default: what an absent value means.
    default: Any = None


@dataclass
class Run:
    label: str
    changes: dict[str, Any]
    measurement: Measurement
    score: float | None = None


@dataclass
class Features:
    moe: bool = False
    gpu: bool = True
    mtp: bool = False
    dense_bf16: bool = False
    cpus: int = 1


def score(m: Measurement, reference: Measurement, goal: str) -> float:
    """Log-ratio against the reference, weighted by the goal; 0 is the reference."""
    weight = GOALS[goal]
    prefill = math.log(max(m.prefill_tps, 1e-9) / max(reference.prefill_tps, 1e-9))
    decode = math.log(max(m.decode_tps, 1e-9) / max(reference.decode_tps, 1e-9))
    return weight * prefill + (1 - weight) * decode


def gain_percent(value: float) -> float:
    return (math.exp(value) - 1.0) * 100.0


def knobs_for(features: Features, skip: set[str] = frozenset()) -> list[Knob]:  # type: ignore[assignment]
    """The settings worth trying here, the ones that move the most first."""
    found: list[Knob] = []
    gpu, moe = features.gpu, features.moe
    if moe and gpu:
        found.append(Knob("expert-mode", "expert_mode",
                          ["auto", "hybrid", "cpu", "gpu", "resident"], "--expert-mode",
                          default="auto"))
    if gpu:
        found.append(Knob("kv", "kv_dtype", [None, "q8_0", "turbo4"], "--kv-dtype",
                          replaces=("cache_type_k", "cache_type_v")))
    if features.mtp:
        found.append(Knob("mtp", "mtp_drafts", [0, 2, 3, 4], "--mtp-drafts", default=0))
    if gpu and features.dense_bf16:
        found.append(Knob("requant", "dense_requant", ["off", "q8"], "--dense-requant",
                          default="off"))
    if moe and gpu:
        found.append(Knob("routed-moe", "routed_moe", [None, True], "--routed-moe"))
        found.append(Knob("hybrid-prefill", "hybrid_prefill", [None, "split", "cpu"],
                          "--hybrid-prefill"))
    if moe or not gpu:
        threads = sorted({max(1, features.cpus // 2), max(1, features.cpus * 3 // 4)})
        found.append(Knob("threads", "cpu_threads", [0, *threads], "--cpu-threads",
                          default=0))
    if moe and gpu:
        found.append(Knob("prefill-cache-seed", "prefill_cache_seed", [None, "off"],
                          "--prefill-cache-seed"))
        found.append(Knob("expert-paging", "expert_paging", ["auto", "direct", "staged"],
                          "--expert-paging", default="auto"))
    return [k for k in found if k.name not in skip]


KNOB_NAMES = ("expert-mode", "kv", "mtp", "requant", "routed-moe", "hybrid-prefill",
              "threads", "prefill-cache-seed", "expert-paging")


def describe(changes: dict[str, Any], knobs: list[Knob]) -> str:
    flags = {k.dest: k.flag for k in knobs}
    defaults = {k.dest: k.default for k in knobs}
    parts = []
    for dest, value in changes.items():
        flag = flags.get(dest, "--" + dest.replace("_", "-"))
        if value is None:
            value = defaults.get(dest)
        if value is True:
            parts.append(flag)
        elif value is None:
            parts.append(f"{flag} default")
        else:
            parts.append(f"{flag} {value}")
    return " ".join(parts) or "the given settings"


def apply(values: dict[str, Any], knob: Knob, value: Any) -> dict[str, Any]:
    out = dict(values)
    if value == knob.default:
        out.pop(knob.dest, None)
    else:
        out[knob.dest] = value
        for dest in knob.replaces:
            out.pop(dest, None)
    return out


def search(base: dict[str, Any], knobs: list[Knob],
           measure: Callable[[dict[str, Any], str], Measurement], *,
           goal: str = "balanced", time_limit: float = 1200.0, min_gain: float = 0.03,
           clock: Callable[[], float] = time.monotonic,
           report: Callable[[str], None] = lambda line: None,
           baseline: tuple[Measurement, float] | None = None) -> dict[str, Any]:
    """Coordinate descent over `knobs` from `base`; see the module docstring.

    `measure(values, label)` runs one server. `baseline`, when given, is
    `base` already measured and how long that took. Returns the runs, the
    best values, and why the search stopped.
    """
    started = clock()
    runs: list[Run] = []
    durations: list[float] = [baseline[1]] if baseline else []
    threshold = math.log1p(min_gain)

    def timed(values: dict[str, Any], label: str) -> Measurement:
        t0 = clock()
        result = measure(values, label)
        durations.append(clock() - t0)
        return result

    def out_of_time(runs_needed: int = 1) -> bool:
        typical = statistics.median(durations) if durations else 0.0
        return clock() - started + typical * runs_needed > time_limit

    def changes_of(values: dict[str, Any]) -> dict[str, Any]:
        keys = set(values) | set(base)
        return {k: values.get(k) for k in sorted(keys) if values.get(k) != base.get(k)}

    reference: Measurement | None = None
    incumbent = dict(base)
    incumbent_m = baseline[0] if baseline else timed(incumbent, "the given settings")
    runs.append(Run("the given settings", {}, incumbent_m))
    if incumbent_m.ok:
        reference = incumbent_m
        runs[-1].score = 0.0
        report(_line(runs[-1], reference))
    else:
        report(f"the given settings did not run: {incumbent_m.error}")

    stopped = ""
    for knob in knobs:
        current = incumbent.get(knob.dest, knob.default)
        for value in knob.values:
            if value == current:
                continue
            if out_of_time():
                stopped = "time limit"
                break
            candidate = apply(incumbent, knob, value)
            label = describe(changes_of(candidate), knobs)
            m = timed(candidate, label)
            run = Run(label, changes_of(candidate), m)
            runs.append(run)
            if not m.ok:
                report(f"{label}: did not run ({m.error})")
                continue
            if reference is None:
                # Nothing ran before this: it is the reference and the incumbent.
                reference, incumbent, incumbent_m = m, candidate, m
                run.score = 0.0
                current = incumbent.get(knob.dest, knob.default)
                report(_line(run, reference))
                continue
            run.score = score(m, reference, goal)
            report(_line(run, reference))
            ahead = run.score - score(incumbent_m, reference, goal)
            if ahead <= threshold:
                continue
            if out_of_time(2):
                stopped = "time limit"
                break
            # Measure both again, interleaved, before believing it.
            again_inc = timed(incumbent, describe(changes_of(incumbent), knobs) + " (again)")
            again_new = timed(candidate, label + " (again)")
            if not (again_inc.ok and again_new.ok):
                if again_new.ok and not again_inc.ok:
                    pass  # the incumbent stopped loading; take the one that does
                else:
                    continue
            inc_avg = _average(incumbent_m, again_inc)
            new_avg = _average(m, again_new)
            if score(new_avg, reference, goal) - score(inc_avg, reference, goal) > threshold:
                incumbent, incumbent_m = candidate, new_avg
                current = incumbent.get(knob.dest, knob.default)
                run.score = score(new_avg, reference, goal)
                report(f"  kept: {label} ({gain_percent(run.score):+.1f}% over the given settings)")
            else:
                report(f"  not kept: {label} did not hold up when measured again")
        if stopped:
            break
    return {"runs": runs, "best": incumbent, "best_measurement": incumbent_m,
            "reference": reference, "stopped": stopped,
            "elapsed": clock() - started, "changes": changes_of(incumbent)}


def _average(a: Measurement, b: Measurement) -> Measurement:
    if not b.ok:
        return a
    if not a.ok:
        return b
    return Measurement(True, (a.prefill_tps + b.prefill_tps) / 2,
                       (a.decode_tps + b.decode_tps) / 2, (a.ttft + b.ttft) / 2,
                       (a.load_seconds + b.load_seconds) / 2, a.prompt_tokens,
                       a.decode_tokens)


def _line(run: Run, reference: Measurement) -> str:
    m = run.measurement
    gain = "" if run.score is None else f"  {gain_percent(run.score):+.1f}%"
    return (f"{run.label}: prefill {m.prefill_tps:,.0f} tok/s, decode {m.decode_tps:.1f} "
            f"tok/s, first token {m.ttft:.2f}s, load {m.load_seconds:.0f}s{gain}")


# -- running real servers ------------------------------------------------------

_SENTENCES = (
    "Memory hierarchies trade latency against capacity at every level.",
    "A cache line holds sixty-four bytes on most current processors.",
    "Branch predictors guess the direction of conditional jumps before they resolve.",
    "Virtual memory maps each process's addresses onto physical pages.",
    "Out-of-order cores retire instructions in program order after executing them early.",
    "Prefetchers watch access patterns and fetch the next lines ahead of demand.",
    "Translation lookaside buffers cache recent page-table walks.",
    "Store buffers let a core continue while writes drain to the cache.",
    "Simultaneous multithreading shares one core's units between two threads.",
    "Non-uniform memory access makes some memory closer to some sockets than others.",
    "Vector units apply one instruction to many data elements at once.",
    "Coherence protocols keep every core's view of a line consistent.",
)


def filler(tokens: int, chars_per_token: float) -> str:
    """Roughly `tokens` tokens of varied prose, opened by a nonce."""
    want = int(tokens * chars_per_token)
    parts = [f"[{uuid.uuid4()}]"]
    size, i = 0, 0
    while size < want:
        sentence = f"{i + 1}. {_SENTENCES[i % len(_SENTENCES)]}"
        parts.append(sentence)
        size += len(sentence) + 1
        i += 1
    return " ".join(parts)


def engine_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--engine"]
    return [sys.executable, "-m", "flyweight"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Server:
    """One `flyweight serve` on a free loopback port, stopped on exit."""

    def __init__(self, argv: list[str], api_key: str | None, load_timeout: float) -> None:
        self.port = _free_port()
        self.api_key = api_key
        self.load_timeout = load_timeout
        self.log = tempfile.TemporaryFile()
        self.process = subprocess.Popen(
            engine_command() + argv + ["--host", "127.0.0.1", "--port", str(self.port)],
            stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log,
        )

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def _request(self, path: str, body: dict | None = None, timeout: float = 10.0):
        request = urllib.request.Request(
            self.url(path), data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        return urllib.request.urlopen(request, timeout=timeout)

    def health(self) -> dict[str, Any]:
        with self._request("/health") as response:
            return json.load(response)

    def wait_ready(self) -> float:
        started = time.monotonic()
        while time.monotonic() - started < self.load_timeout:
            if self.process.poll() is not None:
                raise RuntimeError(self.failure())
            try:
                self.health()
                return time.monotonic() - started
            except (urllib.error.URLError, OSError, ValueError):
                time.sleep(0.5)
        raise RuntimeError(f"did not finish loading in {self.load_timeout:.0f}s")

    def failure(self) -> str:
        self.log.seek(0)
        text = self.log.read().decode("utf-8", "replace")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            lowered = line.lower()
            if any(w in lowered for w in ("error", "out of memory", "does not fit",
                                          "failed", "cannot", "refus")):
                return line[-300:]
        return lines[-1][-300:] if lines else f"exited with code {self.process.returncode}"

    def stream(self, prompt: str, max_tokens: int, timeout: float) -> tuple[float, float, int, int]:
        """(time to first token, decode seconds, prompt tokens, completion tokens)."""
        body = {"model": "local", "stream": True, "max_tokens": max_tokens,
                "temperature": 0.0, "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": prompt}]}
        started = time.perf_counter()
        first: float | None = None
        last = started
        usage: dict[str, Any] = {}
        with self._request("/v1/chat/completions", body, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except ValueError:
                    continue
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                if any(delta.get(k) for k in ("content", "reasoning_content", "tool_calls")):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
        if first is None:
            raise RuntimeError("the server streamed no tokens")
        return (first - started, last - first, int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0))

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()


class Bench:
    """Measures candidate settings with real servers."""

    def __init__(self, model: str, context: int, prompt_tokens: int, decode_tokens: int,
                 repeats: int, load_timeout: float, build_argv: Callable[[dict], list[str]],
                 api_key: str | None, report: Callable[[str], None]) -> None:
        self.model, self.context = model, context
        self.prompt_tokens, self.decode_tokens = prompt_tokens, decode_tokens
        self.repeats, self.load_timeout = repeats, load_timeout
        self.build_argv, self.api_key, self.report = build_argv, api_key, report
        self.chars_per_token = 4.0
        self.last_health: dict[str, Any] = {}

    def _calibrate(self, sent: str, prompt_tokens: int) -> None:
        if prompt_tokens > 100:
            self.chars_per_token = max(1.0, len(sent) / prompt_tokens)

    def measure(self, values: dict[str, Any], label: str) -> Measurement:
        self.report(f"loading: {label}")
        server = Server(self.build_argv(values), self.api_key, self.load_timeout)
        try:
            try:
                load = server.wait_ready()
                warm = filler(min(1024, self.prompt_tokens), self.chars_per_token)
                _, _, sent_tokens, _ = server.stream(warm, 32, timeout=1800)
                self._calibrate(warm, sent_tokens)
                rows = []
                for _ in range(self.repeats):
                    prompt = filler(self.prompt_tokens, self.chars_per_token) + \
                        "\n\nWrite a long, detailed essay on how these ideas fit together."
                    rows.append(server.stream(prompt, self.decode_tokens, timeout=3600))
                health = server.health()
            except (RuntimeError, OSError, urllib.error.URLError, ValueError) as error:
                detail = str(error)
                if server.process.poll() is not None:
                    detail = server.failure()
                return Measurement(False, error=detail)
        finally:
            server.stop()
        prefill = statistics.median(p / t if t > 0 else 0.0 for t, _, p, _ in rows)
        decode = statistics.median((c - 1) / d if d > 0 and c > 1 else 0.0
                                   for _, d, _, c in rows)
        self.last_health = health
        return Measurement(True, prefill, decode, statistics.median(r[0] for r in rows),
                           load, int(statistics.median(r[2] for r in rows)),
                           int(statistics.median(r[3] for r in rows)), health=health)

    def holds_context(self, values: dict[str, Any]) -> tuple[bool, str]:
        """Load `values` and run a prompt that fills the context."""
        server = Server(self.build_argv(values), self.api_key, self.load_timeout)
        try:
            server.wait_ready()
            tokens = max(256, int((self.context - 64) * 0.92) - 32)
            server.stream(filler(tokens, self.chars_per_token), 16, timeout=7200)
            return True, ""
        except (RuntimeError, OSError, urllib.error.URLError, ValueError) as error:
            return False, server.failure() if server.process.poll() is not None else str(error)
        finally:
            server.stop()


def _interrupt(*_: object) -> None:
    raise KeyboardInterrupt


def features_from(health: dict[str, Any], model: str, cpus: int,
                  backend: str = "auto") -> Features:
    execution = health.get("execution", {}) if health else {}
    gpu_backend = str(execution.get("backend", ""))
    gpu = (bool(execution.get("cuda_ready")) or "cuda" in gpu_backend or "rocm" in gpu_backend
           or "hip" in gpu_backend) if execution else backend != "cpu"
    return Features(moe=int(execution.get("expert_count") or 0) > 0,
                    gpu=gpu, mtp=bool(execution.get("mtp_available")),
                    dense_bf16=_has_bf16_dense(model), cpus=cpus)


def _has_bf16_dense(model: str) -> bool:
    """Whether the checkpoint has BF16/F16 dense layer weights --dense-requant acts on."""
    if not model.lower().endswith(".gguf"):
        return False
    try:
        from .v2 import V2Model

        with V2Model(model) as m:
            for tensor in m.tensors():
                name = str(tensor.get("name", ""))
                if (name.startswith("blk.") and "_exps" not in name
                        and int(tensor.get("ggml_type", -1)) in (1, 30)
                        and len(tensor.get("shape") or ()) >= 2):  # type: ignore[arg-type]
                    return True
    except Exception:  # noqa: BLE001 - a metadata read failing just skips the knob
        return False
    return False


def run(model: str, base: dict[str, Any], *, context: int, goal: str, time_limit: float,
        prompt_tokens: int, decode_tokens: int, repeats: int, min_gain: float,
        load_timeout: float, skip: set[str], verify: bool,
        build_argv: Callable[[dict], list[str]], api_key: str | None,
        report: Callable[[str], None]) -> dict[str, Any]:
    """The whole tune: features, search, a full-context check of the winner."""
    bench = Bench(model, context, prompt_tokens, decode_tokens, repeats, load_timeout,
                  build_argv, api_key, report)
    started = time.monotonic()
    # The first load reads the model from disk into the page cache and is
    # slower for it; timing it would make every later setup look better.
    report("warming up: the first load reads the model from disk and is not counted")
    baseline = bench.measure(base, "warm-up")
    if baseline.ok:
        # This run also says what the model and the machine are, which
        # decides the settings worth trying.
        baseline = bench.measure(base, "the given settings")
    features = features_from(baseline.health, model, os.cpu_count() or 1,
                             str(base.get("backend", "auto")))
    knobs = knobs_for(features, skip)
    report("trying: " + (", ".join(k.name for k in knobs) or "nothing applies here"))
    spent = time.monotonic() - started
    result = search(base, knobs, bench.measure, goal=goal, time_limit=time_limit - spent,
                    min_gain=min_gain, report=report, baseline=(baseline, spent))
    best, changes = result["best"], result["changes"]
    verified: bool | None = None
    verify_error = ""
    if verify and result["best_measurement"].ok:
        report(f"checking that the best setup holds a {context:,}-token prompt")
        verified, verify_error = bench.holds_context(best)
        if not verified:
            report(f"  it did not ({verify_error}); falling back to the next best that does")
            ranked = sorted((r for r in result["runs"] if r.measurement.ok and r.score is not None
                             and r.changes != changes), key=lambda r: -(r.score or 0.0))
            for candidate in ranked[:3]:
                values = dict(base)
                values.update({k: v for k, v in candidate.changes.items() if v is not None})
                for k, v in candidate.changes.items():
                    if v is None:
                        values.pop(k, None)
                ok, _ = bench.holds_context(values)
                if ok:
                    best, changes, verified = values, candidate.changes, True
                    result["best_measurement"] = candidate.measurement
                    break
    reference = result["reference"]
    best_m = result["best_measurement"]
    return {
        "model": model, "context": context, "goal": goal,
        "prompt_tokens": prompt_tokens, "decode_tokens": decode_tokens,
        "features": asdict(features),
        "knobs": [k.name for k in knobs],
        "runs": [{"label": r.label, "changes": r.changes, "score": r.score,
                  "gain_percent": None if r.score is None else round(gain_percent(r.score), 1),
                  **{k: v for k, v in asdict(r.measurement).items() if k != "health"}}
                 for r in result["runs"]],
        "reference": None if reference is None else {
            "prefill_tps": reference.prefill_tps, "decode_tps": reference.decode_tps},
        "best": {"changes": changes, "values": best,
                 "argv": build_argv(best),
                 "prefill_tps": best_m.prefill_tps, "decode_tps": best_m.decode_tps,
                 "gain_percent": (round(gain_percent(score(best_m, reference, goal)), 1)
                                  if reference is not None and best_m.ok else None),
                 "ok": best_m.ok},
        "verified": verified, "verify_error": verify_error,
        "stopped": result["stopped"] or "done",
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [f"{'setup':<64} {'prefill':>12} {'decode':>12} {'vs given':>9}"]
    for r in report["runs"]:
        if r["ok"]:
            gain = "" if r["gain_percent"] is None else f"{r['gain_percent']:+.1f}%"
            lines.append(f"{r['label'][:64]:<64} {r['prefill_tps']:>8,.0f} t/s "
                         f"{r['decode_tps']:>8.1f} t/s {gain:>9}")
        else:
            lines.append(f"{r['label'][:64]:<64} {'did not run: ' + r['error'][:80]}")
    best = report["best"]
    lines.append("")
    if not best["ok"]:
        lines.append("No setup ran at this context. Try a smaller --context.")
        return "\n".join(lines)
    gain = best["gain_percent"]
    lines.append(f"Best for '{report['goal']}': prefill {best['prefill_tps']:,.0f} tok/s, "
                 f"decode {best['decode_tps']:.1f} tok/s"
                 + (f" ({gain:+.1f}% over the given settings)" if gain else ""))
    if report["verified"] is False:
        lines.append(f"Warning: it did not hold a full {report['context']:,}-token prompt "
                     f"({report['verify_error']}).")
    lines.append("  flyweight " + " ".join(_quote(a) for a in best["argv"]))
    if report["stopped"] == "time limit":
        lines.append("Stopped at the time limit; --time-limit gives it longer.")
    return "\n".join(lines)


def _quote(text: str) -> str:
    import shlex

    return shlex.quote(text) if os.name != "nt" else (f'"{text}"' if " " in text else text)


def base_values(namespace: dict[str, Any], serve_options: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The serve settings `tune` was given: every serve option set off its default."""
    from .engine_options import _jsonable, _same

    values = {}
    for dest, option in serve_options.items():
        if option["positional"] or dest in _NOT_PASSED or dest not in namespace:
            continue
        value = _jsonable(namespace[dest])
        if value is None or _same(value, option["default"]):
            continue
        values[dest] = value
    return values


def main(args: Any) -> int:
    from .engine_options import build_argv, options_of, schema

    spec = schema("serve")
    base = base_values(vars(args), options_of(spec))
    model = str(args.model)
    context = int(args.context_window)

    def argv_for(values: dict[str, Any]) -> list[str]:
        return build_argv("serve", {**values, "context_window": context}, spec,
                          positional={"model": model})

    prompt_tokens = args.prompt_tokens or max(256, min(4096, context // 2))
    prompt_tokens = min(prompt_tokens, context - args.decode_tokens - 64)
    if prompt_tokens < 64:
        print("flyweight: --context is too small to benchmark", file=sys.stderr)
        return 2

    def say(line: str) -> None:
        print(f"[tune] {line}", file=sys.stderr, flush=True)

    say(f"{Path(model).name}, context {context:,}, goal {args.goal}, "
        f"{args.time_limit:g} min, {prompt_tokens:,}-token prompts, "
        f"{args.decode_tokens} generated tokens, {args.repeats} per setup")
    signal.signal(signal.SIGTERM, _interrupt)  # the app's Cancel stops the server too
    try:
        report = run(model, base, context=context, goal=args.goal,
                     time_limit=args.time_limit * 60.0, prompt_tokens=prompt_tokens,
                     decode_tokens=args.decode_tokens, repeats=args.repeats,
                     min_gain=args.min_gain / 100.0, load_timeout=args.load_timeout,
                     skip=set(args.skip or ()), verify=not args.no_verify,
                     build_argv=argv_for, api_key=base.get("api_key"), report=say)
    except KeyboardInterrupt:
        say("stopped")
        return 130
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))
    return 0 if report["best"]["ok"] else 1
