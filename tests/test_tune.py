"""`flyweight tune`: which settings it tries, and how the search decides."""
from __future__ import annotations

import unittest

from flyweight import tune
from flyweight.cli import _parser
from flyweight.engine_options import options_of, schema
from flyweight.tune import Features, Knob, Measurement


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def fake(rates: dict[tuple, tuple[float, float]], clock: Clock | None = None,
         cost: float = 10.0, calls: list | None = None):
    """measure() from a table keyed by sorted (dest, value) items."""
    def measure(values, label):
        if clock is not None:
            clock.now += cost
        if calls is not None:
            calls.append(dict(values))
        key = tuple(sorted(values.items()))
        if key not in rates:
            return Measurement(False, error="out of memory")
        prefill, decode = rates[key]
        return Measurement(True, prefill, decode)
    return measure


EXPERT = Knob("expert-mode", "expert_mode", ["auto", "hybrid", "cpu"], "--expert-mode",
              default="auto")
KV = Knob("kv", "kv_dtype", [None, "q8_0"], "--kv-dtype",
          replaces=("cache_type_k", "cache_type_v"))


class KnobTests(unittest.TestCase):
    def test_a_dense_gpu_model_only_gets_what_applies(self) -> None:
        names = [k.name for k in tune.knobs_for(Features(moe=False, gpu=True, cpus=16))]
        self.assertEqual(names, ["kv"])

    def test_a_moe_with_mtp_and_bf16_gets_everything_most_consequential_first(self) -> None:
        names = [k.name for k in tune.knobs_for(
            Features(moe=True, gpu=True, mtp=True, dense_bf16=True, cpus=32))]
        self.assertEqual(names[:3], ["expert-mode", "kv", "mtp"])
        self.assertEqual(set(names), set(tune.KNOB_NAMES))

    def test_cpu_only_tunes_threads_and_skip_is_honoured(self) -> None:
        knobs = tune.knobs_for(Features(moe=True, gpu=False, cpus=32))
        self.assertEqual([k.name for k in knobs], ["threads"])
        self.assertEqual(knobs[0].values, [0, 16, 24])
        self.assertEqual(tune.knobs_for(Features(moe=True, gpu=False, cpus=32),
                                        {"threads"}), [])

    def test_approximate_routing_is_never_a_knob(self) -> None:
        dests = {k.dest for k in tune.knobs_for(
            Features(moe=True, gpu=True, mtp=True, dense_bf16=True, cpus=8))}
        self.assertFalse(dests & {"route_prior", "route_tail_skip"})


class SearchTests(unittest.TestCase):
    def test_a_win_is_kept_only_after_it_holds_up(self) -> None:
        rates = {(): (100, 10), (("expert_mode", "hybrid"),): (100, 10),
                 (("expert_mode", "cpu"),): (150, 15)}
        result = tune.search({}, [EXPERT], fake(rates))
        self.assertEqual(result["best"], {"expert_mode": "cpu"})
        self.assertEqual(result["changes"], {"expert_mode": "cpu"})
        labels = [r.label for r in result["runs"]]
        self.assertIn("--expert-mode cpu", labels)

    def test_noise_that_does_not_repeat_is_not_kept(self) -> None:
        seen: dict[str, int] = {}

        def measure(values, label):
            key = values.get("expert_mode", "auto")
            seen[key] = seen.get(key, 0) + 1
            # cpu looks 50% faster the first time, 60% slower the second.
            if key == "cpu":
                return Measurement(True, 150 if seen[key] == 1 else 40, 10)
            return Measurement(True, 100, 10)

        result = tune.search({}, [Knob("expert-mode", "expert_mode", ["auto", "cpu"],
                                       "--expert-mode", default="auto")], measure, goal="prefill")
        self.assertEqual(result["best"], {})

    def test_the_goal_decides_between_prefill_and_decode(self) -> None:
        rates = {(): (100, 10), (("expert_mode", "cpu"),): (130, 8)}
        knob = Knob("expert-mode", "expert_mode", ["auto", "cpu"], "--expert-mode",
                    default="auto")
        self.assertEqual(tune.search({}, [knob], fake(rates), goal="prefill")["best"],
                         {"expert_mode": "cpu"})
        self.assertEqual(tune.search({}, [knob], fake(rates), goal="decode")["best"], {})

    def test_a_setting_that_does_not_load_is_recorded_and_skipped(self) -> None:
        rates = {(): (100, 10)}
        result = tune.search({}, [KV], fake(rates))
        failed = [r for r in result["runs"] if not r.measurement.ok]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].measurement.error, "out of memory")
        self.assertEqual(result["best"], {})

    def test_when_the_given_settings_do_not_load_the_first_that_does_leads(self) -> None:
        # The context does not fit with an f16 cache; q8_0 makes it fit.
        rates = {(("kv_dtype", "q8_0"),): (90, 9)}
        result = tune.search({}, [KV], fake(rates))
        self.assertEqual(result["best"], {"kv_dtype": "q8_0"})
        self.assertTrue(result["best_measurement"].ok)

    def test_kv_dtype_replaces_the_separate_cache_types(self) -> None:
        base = {"cache_type_k": "q8_0", "cache_type_v": "f16"}
        calls: list = []
        tune.search(base, [KV], fake({}, calls=calls))
        self.assertIn({"kv_dtype": "q8_0"}, calls)

    def test_it_stops_at_the_time_limit(self) -> None:
        clock = Clock()
        rates = {(): (100, 10)}
        result = tune.search({}, [EXPERT, KV], fake(rates, clock), time_limit=25, clock=clock)
        self.assertEqual(result["stopped"], "time limit")
        self.assertLessEqual(clock.now, 25)

    def test_a_given_setting_is_the_starting_point(self) -> None:
        calls: list = []
        tune.search({"expert_mode": "cpu"}, [EXPERT], fake({}, calls=calls))
        tried = [c.get("expert_mode", "auto") for c in calls]
        self.assertEqual(tried, ["cpu", "auto", "hybrid"])


class CommandTests(unittest.TestCase):
    def test_given_serve_options_pass_through_and_the_listener_does_not(self) -> None:
        args = _parser().parse_args(["tune", "m.gguf", "--context", "16384",
                                     "--kv-dtype", "q8_0", "--cpu-threads", "8"])
        base = tune.base_values(vars(args), options_of(schema("serve")))
        self.assertEqual(base, {"kv_dtype": "q8_0", "cpu_threads": 8})

    def test_the_report_names_the_command_to_run(self) -> None:
        report = {
            "goal": "balanced", "context": 8192, "verified": True, "verify_error": "",
            "stopped": "done",
            "runs": [{"label": "the given settings", "ok": True, "prefill_tps": 400.0,
                      "decode_tps": 50.0, "gain_percent": 0.0, "error": ""},
                     {"label": "--expert-mode gpu", "ok": False, "prefill_tps": 0.0,
                      "decode_tps": 0.0, "gain_percent": None, "error": "out of memory"}],
            "best": {"ok": True, "prefill_tps": 400.0, "decode_tps": 50.0,
                     "gain_percent": 0.0, "argv": ["serve", "m.gguf", "--context", "8192"]},
        }
        text = tune.format_report(report)
        self.assertIn("flyweight serve m.gguf --context 8192", text)
        self.assertIn("did not run: out of memory", text)


if __name__ == "__main__":
    unittest.main()


FAKE_SERVE = r'''
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

argv = sys.argv[1:]
if "gpu" in argv:
    print("flyweight: resident expert mode requires 20050 MiB", file=sys.stderr)
    sys.exit(1)
port = int(argv[argv.index("--port") + 1])
fast = "q8_0" in argv

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"status": "ok", "execution": {
            "backend": "native-v2-cpp-cuda", "cuda_ready": True, "expert_count": 8,
            "mtp_available": 0}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = len(request["messages"][0]["content"]) // 4
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(0.05 if fast else 0.1)
        for _ in range(request["max_tokens"]):
            chunk = {"choices": [{"delta": {"content": "x"}}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(0.002)
        usage = {"choices": [], "usage": {"prompt_tokens": prompt,
                                          "completion_tokens": request["max_tokens"]}}
        self.wfile.write(b"data: " + json.dumps(usage).encode() + b"\n\ndata: [DONE]\n\n")

ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''


class RealServerTests(unittest.TestCase):
    """The measuring side, against a stand-in for `flyweight serve`."""

    def test_a_search_over_real_processes(self) -> None:
        import sys
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from flyweight.engine_options import build_argv

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake_serve.py"
            script.write_text(FAKE_SERVE)
            spec = schema("serve")

            def argv_for(values):
                return build_argv("serve", {**values, "context_window": 4096}, spec,
                                  positional={"model": "m.gguf"})

            lines: list[str] = []
            with patch.object(tune, "engine_command", return_value=[sys.executable, str(script)]), \
                    patch.object(tune, "_has_bf16_dense", return_value=False):
                report = tune.run(
                    "m.gguf", {}, context=4096, goal="prefill", time_limit=120,
                    prompt_tokens=200, decode_tokens=8, repeats=1, min_gain=0.03,
                    load_timeout=30, skip={"threads", "routed-moe", "hybrid-prefill",
                                           "prefill-cache-seed", "expert-paging"},
                    verify=True, build_argv=argv_for, api_key=None, report=lines.append)
        self.assertEqual(report["knobs"], ["expert-mode", "kv"])
        self.assertEqual(report["best"]["changes"], {"kv_dtype": "q8_0"})
        self.assertIn("--kv-dtype", report["best"]["argv"])
        self.assertTrue(report["verified"])
        failed = [r for r in report["runs"] if not r["ok"]]
        self.assertEqual([r["label"] for r in failed], ["--expert-mode gpu"])
        self.assertIn("20050 MiB", failed[0]["error"])
        ok = [r for r in report["runs"] if r["ok"]]
        self.assertTrue(all(r["prefill_tps"] > 0 and r["decode_tps"] > 0 for r in ok))
