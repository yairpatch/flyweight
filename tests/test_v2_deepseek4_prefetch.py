"""Asking for the routed experts before reading them.

A token reads 2.16 GiB of expert weights, and the six experts of a layer are
known the moment the router runs -- but the loop multiplies them one matrix at
a time, so every one that is not resident stalls in turn. Hinting all eighteen
ranges up front lets the kernel fetch the later ones while the first is being
multiplied.

Measured on the real checkpoint with the page cache dropped first: 0.87 -> 1.46
tok/s, expert time 872 -> 353 ms a token, and disk traffic *halved* (1372 -> 766
MiB a token), because an explicit range read beats fault-driven readahead at
guessing what is wanted. Fully resident it costs two to three percent, which is
the price of the hints themselves; that is the trade, and FLYWEIGHT_DS4_PREFETCH=off
exists so it can be re-measured rather than taken on faith.

A hint cannot change an answer, which is what makes this safe -- the test that
matters is that it is still only a hint.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PROGRAM = """
import json, sys
from flyweight.v2 import V2Model
from flyweight.deepseek4 import Deepseek4Runtime

model = V2Model(sys.argv[1])
try:
    tokens = list(model.tokenize("the quick brown fox jumps over the lazy dog"))
    with Deepseek4Runtime(model, 256) as runtime:
        for token in tokens[:-1]:
            runtime.forward(token, logits=False)
        logits = runtime.forward(tokens[-1])
        info = runtime.info
    print(json.dumps({
        "logits": logits.tobytes().hex(),
        "hinted": int(info["expert_prefetch_bytes"]),
        "expert_bytes": int(info["routed_expert_bytes"]),
    }))
finally:
    model.close()
"""


class PrefetchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.deepseek4_gguf_fixture import DeepSeek4Spec, build_deepseek4_gguf

        cls.directory = tempfile.TemporaryDirectory(prefix="flyweight-ds4pf-")
        cls.path = Path(cls.directory.name) / "ds4.gguf"
        build_deepseek4_gguf(cls.path, DeepSeek4Spec(layers=6, hash_layers=3))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_with(self, setting: str | None) -> dict:
        import json

        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(REPO / "src")
        if setting is None:
            environment.pop("FLYWEIGHT_DS4_PREFETCH", None)
        else:
            environment["FLYWEIGHT_DS4_PREFETCH"] = setting
        completed = subprocess.run(
            [sys.executable, "-c", PROGRAM, str(self.path)],
            capture_output=True, text=True, env=environment, timeout=600,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def test_it_is_only_a_hint(self):
        # The whole safety argument: turning it off must change nothing but
        # the timing.
        self.assertEqual(self.run_with("off")["logits"], self.run_with(None)["logits"])

    def test_it_hints_exactly_the_weights_it_is_about_to_read(self):
        run = self.run_with(None)
        self.assertEqual(run["hinted"], run["expert_bytes"])

    def test_the_switch_turns_it_off(self):
        self.assertEqual(self.run_with("off")["hinted"], 0)

    def test_any_other_value_leaves_it_on(self):
        # Only the documented word disables it; a typo must not silently opt out.
        self.assertGreater(self.run_with("no")["hinted"], 0)
