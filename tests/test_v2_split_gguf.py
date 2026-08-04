"""Split-GGUF loading: a sharded checkpoint must behave like the single file.

Large checkpoints (DeepSeek-V4-Flash's IQ3_XXS build is 104 GB across four
files) only ship as `gguf-split` output, so the loader has to stitch the shards
back into one tensor table whose descriptors point into whichever mapping holds
their bytes.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from colibri_next.v2 import V2Error, V2Model

from tests.dense_gguf_fixture import DenseQwenSpec, build_dense_qwen35_gguf
from tests.split_gguf_fixture import split_gguf


def _fixture(**kwargs) -> tuple[Path, Path, DenseQwenSpec]:
    """Write the dense fixture once and return it beside a split of itself."""
    directory = Path(tempfile.mkdtemp(prefix="colibri-split-"))
    single = directory / "dense.gguf"
    spec = build_dense_qwen35_gguf(single, DenseQwenSpec(layers=4))
    first = split_gguf(single, directory, **kwargs)
    return single, first, spec


class SplitGgufTests(unittest.TestCase):
    def test_split_model_matches_the_file_it_was_split_from(self):
        single, first, spec = _fixture(shards=3)
        whole, sharded = V2Model(single), V2Model(first)
        try:
            self.assertEqual(
                whole.info["tensor_count"], sharded.info["tensor_count"]
            )
            self.assertEqual(sharded.config["architecture"], "qwen35")
            self.assertEqual(sharded.config["layer_count"], spec.layers)
            self.assertEqual(sharded.config["hidden_size"], spec.hidden)
            expected = {tensor["name"]: tensor["shape"] for tensor in whole.tensors()}
            actual = {tensor["name"]: tensor["shape"] for tensor in sharded.tensors()}
            self.assertEqual(expected, actual)
        finally:
            whole.close()
            sharded.close()

    def test_first_shard_may_carry_metadata_and_no_tensors(self):
        # DeepSeek-V4-Flash's layout: shard 1 is 5 MB of metadata, and every
        # tensor lives in shards 2 through 4.
        single, first, _ = _fixture(shards=4, metadata_only_first=True)
        whole, sharded = V2Model(single), V2Model(first)
        try:
            self.assertEqual(
                whole.info["tensor_count"], sharded.info["tensor_count"]
            )
            self.assertGreater(int(sharded.info["tensor_count"]), 0)
            self.assertEqual(
                sorted(tensor["name"] for tensor in whole.tensors()),
                sorted(tensor["name"] for tensor in sharded.tensors()),
            )
        finally:
            whole.close()
            sharded.close()

    def test_tensor_payloads_survive_the_split(self):
        single, first, _ = _fixture(shards=3)
        whole, sharded = V2Model(single), V2Model(first)
        try:
            for name in ("token_embd.weight", "output_norm.weight", "blk.2.ffn_up.weight"):
                size = int(whole.tensor(name)["size"])
                self.assertEqual(
                    bytes(whole.read_tensor_slice(name, 0, size)),
                    bytes(sharded.read_tensor_slice(name, 0, size)),
                    msg=f"{name} differs across the split",
                )
        finally:
            whole.close()
            sharded.close()

    def test_a_missing_shard_is_reported(self):
        _, first, _ = _fixture(shards=3)
        first.with_name(first.name.replace("00001", "00003")).unlink()
        with self.assertRaises(V2Error) as raised:
            V2Model(first)
        self.assertIn("GGUF", str(raised.exception))

    def test_a_shard_that_disagrees_about_the_split_is_rejected(self):
        _, first, _ = _fixture(shards=3)
        stray = first.with_name(first.name.replace("00001", "00002"))
        # Overwrite shard 2 with an unrelated single-file model, which declares
        # no split at all.
        build_dense_qwen35_gguf(stray, DenseQwenSpec(layers=2))
        with self.assertRaises(V2Error) as raised:
            V2Model(first)
        self.assertIn("split", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
