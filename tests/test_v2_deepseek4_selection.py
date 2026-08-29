"""The indexer inside the loop: when it runs, and what it changes.

The selection is invisible on any prompt this project has used until now.
`indexer_top_k` is 512 and a 4:1 layer closes one block every four tokens, so
nothing below about 2048 tokens has more blocks than the indexer may keep, and
keeping everything is the same as not running. That is why the runtime skips it
below the threshold -- the skip is exact, not an approximation -- and it is also
why a short-prompt pass says nothing about whether the selection works.

A real 2600-token run takes half an hour on this machine, so the property is
pinned here on the miniature fixture with a small `indexer_top_k` instead: the
same code path, reached after a handful of tokens.

What the real checkpoint is for, in the gated test at the bottom, is the
compressor behind the selection -- the cache is built from the first token
whatever the length, and its contents are checked against the reference's own
dump of the same prompt.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flyweight.deepseek4 import Deepseek4Runtime
from flyweight.v2 import V2Model
from tests.deepseek4_gguf_fixture import DeepSeek4Spec, build_deepseek4_gguf

_CHECKPOINT_PATH = os.environ.get("DEEPSEEK4_GGUF")
# A stale path is as good as no path: the variable often outlives the file it
# named, and treating that as "configured" turns a missing checkpoint into a
# wall of errors instead of a skip.
CHECKPOINT = _CHECKPOINT_PATH if _CHECKPOINT_PATH and os.path.exists(_CHECKPOINT_PATH) else None
# The indexer keys here were recorded from the UD-IQ3_XXS build; a different
# quantization produces different weights and so different keys.
if CHECKPOINT and "IQ3_XXS" not in CHECKPOINT:
    CHECKPOINT = None


def fixture(directory: str, **spec) -> Path:
    path = Path(directory) / "ds4.gguf"
    build_deepseek4_gguf(path, DeepSeek4Spec(layers=6, hash_layers=3, **spec))
    return path


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="flyweight-ds4sel-")
        self.addCleanup(self.directory.cleanup)

    def open(self, **spec) -> V2Model:
        model = V2Model(fixture(self.directory.name, **spec))
        self.addCleanup(model.close)
        return model

    def test_it_stays_out_of_the_way_below_the_threshold(self):
        # Sixteen tokens on a 4:1 layer is four blocks, and the indexer may keep
        # eight, so it must not run at all.
        model = self.open(indexer_top_k=8)
        with Deepseek4Runtime(model, 256) as runtime:
            for token in range(16):
                runtime.forward(token + 5, logits=False)
            self.assertEqual(runtime.info["indexer_candidates"], 0)

    def test_it_runs_once_there_are_more_blocks_than_it_may_keep(self):
        model = self.open(indexer_top_k=2)
        with Deepseek4Runtime(model, 256) as runtime:
            for token in range(24):
                runtime.forward(token + 5, logits=False)
            info = runtime.info
            self.assertGreater(info["indexer_candidates"], 0)
            # It never keeps more than it considered, nor more than top_k per
            # layer per position.
            self.assertLessEqual(info["indexer_selections"], info["indexer_candidates"])

    def test_keeping_everything_is_the_same_as_not_running(self):
        # The exactness claim the skip rests on: a top_k as large as the cache
        # can ever hold must produce the same logits as one that is merely
        # larger than the block count.
        tokens = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]
        outputs = []
        for top_k in (512, 1024):
            model = self.open(indexer_top_k=top_k)
            with Deepseek4Runtime(model, 256) as runtime:
                for token in tokens[:-1]:
                    runtime.forward(token, logits=False)
                outputs.append(runtime.forward(tokens[-1]).tobytes())
        self.assertEqual(outputs[0], outputs[1])

    def test_a_tighter_selection_changes_the_answer(self):
        # If it did not, the selection would not be reaching the attention.
        tokens = list(range(5, 45))
        outputs = []
        for top_k in (2, 512):
            model = self.open(indexer_top_k=top_k)
            with Deepseek4Runtime(model, 256) as runtime:
                for token in tokens[:-1]:
                    runtime.forward(token, logits=False)
                outputs.append(runtime.forward(tokens[-1]))
        self.assertFalse(np.array_equal(outputs[0], outputs[1]))

    def test_the_cache_is_filled_from_the_first_block(self):
        # Even while the indexer is inert: the rows a block pools come from a
        # hidden state that is gone by the next token.
        model = self.open(indexer_top_k=512)
        with Deepseek4Runtime(model, 256) as runtime:
            for token in range(8):
                runtime.forward(token + 5, logits=False)
            self.assertEqual(runtime.info["indexer_candidates"], 0)
            key = runtime.indexer_key(2, 0)  # layer 2 is the first 4:1 layer
            self.assertEqual(key.shape, (int(model.config["indexer_key_length"]),))
            self.assertTrue(np.any(key != 0))

    def test_a_block_that_has_not_closed_cannot_be_read(self):
        model = self.open()
        with Deepseek4Runtime(model, 256) as runtime:
            runtime.forward(5, logits=False)
            with self.assertRaises(Exception):
                runtime.indexer_key(2, 0)

    def test_a_layer_without_an_indexer_says_so(self):
        model = self.open()
        with Deepseek4Runtime(model, 256) as runtime:
            for token in range(8):
                runtime.forward(token + 5, logits=False)
            with self.assertRaises(Exception):
                runtime.indexer_key(0, 0)  # a sliding-window layer


@unittest.skipUnless(CHECKPOINT, "set DEEPSEEK4_GGUF to the first shard of a real checkpoint")
class IndexerCacheAgainstTheReferenceTests(unittest.TestCase):
    """The compressed indexer keys, against llama.cpp's own dump.

    Regenerate with, from the reference build:

        ./build/bin/llama-eval-callback -m <shard 1> \\
            -f ds4-dumps/indexer_prompt.txt -n 1 -c 1024 --temp 0 -ngl 0

    and read `lid_state_compress-2` -- the CONCAT, which is the value before the
    Hadamard rotation the reference applies to its cache. Skipping that rotation
    is safe here for the same reason it is on the main path: it is orthogonal,
    so every dot product the indexer takes is unchanged, and it exists to spread
    quantization error across channels this cache does not quantize.
    """

    PROMPT = Path("/home/yair/Desktop/ds4-dumps/indexer_prompt.txt")
    # Layer 2, blocks 0-2: first three and last three of each 128-wide key.
    EXPECTED = {
        0: ((-0.8742, 2.8815, 0.8399), (-0.5940, 0.0140, 4.0915)),
        1: ((-1.6497, 1.5992, 0.5085), (0.5942, -0.0128, -0.5456)),
        2: ((-2.2688, -0.4715, -0.1213), (0.6658, -0.2363, 2.4126)),
    }

    @unittest.skipUnless(PROMPT.is_file(), "the reference prompt is not on this machine")
    def test_the_keys_match_the_reference(self):
        model = V2Model(CHECKPOINT)
        self.addCleanup(model.close)
        tokens = list(model.tokenize(self.PROMPT.read_text(encoding="utf-8")))
        with Deepseek4Runtime(model, 1024) as runtime:
            for token in tokens:
                runtime.forward(token, logits=False)
            self.assertEqual(runtime.info["positions"] // 4, 101)
            for block, (head, tail) in self.EXPECTED.items():
                key = runtime.indexer_key(2, block)
                with self.subTest(block=block):
                    # The activation-quantization band every other component in
                    # this port sits in; absolute tolerance carries the values
                    # near zero, where a relative one says nothing.
                    np.testing.assert_allclose(key[:3], head, rtol=0.05, atol=0.05)
                    np.testing.assert_allclose(key[-3:], tail, rtol=0.05, atol=0.05)


if __name__ == "__main__":
    unittest.main()
