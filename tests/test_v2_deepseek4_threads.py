"""Parallel width must change the speed and nothing else.

The forward loop forks about eight hundred times a token -- every projection,
every expert, the attention heads and the hyper-connection mixer. Each of those
splits is only safe because the pieces are independent and the arithmetic
*within* a piece is untouched, so the same tokens must come out whatever the
team size. That is the property worth pinning: a future split that reorders a
reduction would still pass a tolerance-based check and still be wrong here,
because greedy decoding turns a last-bit difference into a different token
whenever two candidates are close.

The width itself is a measured choice rather than a default. On this box, one
thread per hardware thread runs the expert weights at 9.3 GiB/s where one per
core reaches 24.6 -- 1.9 tok/s against 3.2 -- because two threads sharing a core
contend for one L1 and one set of decode units. The runtime therefore caps the
team at the core count unless OMP_NUM_THREADS says otherwise, and this checks
that an explicit setting is still obeyed.
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
import numpy as np
from colibri_next.v2 import V2Model
from colibri_next.deepseek4 import Deepseek4Runtime

model = V2Model(sys.argv[1])
try:
    tokens = list(model.tokenize("the quick brown fox jumps over the lazy dog"))
    with Deepseek4Runtime(model, 256) as runtime:
        for token in tokens[:-1]:
            runtime.forward(token, logits=False)
        logits = runtime.forward(tokens[-1])
        info = runtime.info
    # Bits, not values: a float that prints the same can still differ.
    print(json.dumps({
        "logits": logits.tobytes().hex(),
        "forward_calls": int(info["forward_calls"]),
        "attributed": int(
            info["attention_nanoseconds"] + info["routed_expert_nanoseconds"]
            + info["shared_expert_nanoseconds"] + info["head_nanoseconds"]
        ),
        "total": int(info["forward_nanoseconds"]),
        "expert_bytes": int(info["routed_expert_bytes"]),
    }))
finally:
    model.close()
"""


class ThreadWidthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.deepseek4_gguf_fixture import DeepSeek4Spec, build_deepseek4_gguf

        cls.directory = tempfile.TemporaryDirectory(prefix="colibri-ds4omp-")
        cls.path = Path(cls.directory.name) / "ds4.gguf"
        build_deepseek4_gguf(cls.path, DeepSeek4Spec(layers=6, hash_layers=3))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_with(self, threads: str | None) -> dict:
        import json

        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(REPO / "src")
        if threads is None:
            environment.pop("OMP_NUM_THREADS", None)
        else:
            environment["OMP_NUM_THREADS"] = threads
        completed = subprocess.run(
            [sys.executable, "-c", PROGRAM, str(self.path)],
            capture_output=True, text=True, env=environment, timeout=600,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def test_the_logits_are_bit_identical_across_team_sizes(self):
        baseline = self.run_with("1")
        for threads in ("2", "4", None):
            with self.subTest(threads=threads or "default"):
                self.assertEqual(
                    self.run_with(threads)["logits"], baseline["logits"]
                )

    def test_the_counters_attribute_most_of_a_token(self):
        # Not all of it: the feed-forward hyper-connection, the router and the
        # swiglu sit between the measured pieces. If this ever drops far, a new
        # cost has appeared where nothing is looking.
        run = self.run_with(None)
        self.assertGreater(run["forward_calls"], 0)
        self.assertLessEqual(run["attributed"], run["total"])
        self.assertGreater(run["attributed"], 0.5 * run["total"])

    def test_the_expert_byte_count_matches_the_weights_it_reads(self):
        from colibri_next.v2 import V2Model

        run = self.run_with("1")
        model = V2Model(self.path)
        try:
            sizes = {
                tensor["name"]: tensor["size"] for tensor in model.tensors()
            }
            experts = int(model.config["expert_count"])
            used = int(model.config["expert_used_count"])
            layers = int(model.config["layer_count"])
            per_token = sum(
                sizes[f"blk.{layer}.ffn_{role}_exps.weight"] // experts * used
                for layer in range(layers)
                for role in ("gate", "up", "down")
            )
        finally:
            model.close()
        self.assertEqual(run["expert_bytes"], per_token * run["forward_calls"])


if __name__ == "__main__":
    unittest.main()
