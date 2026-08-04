"""DeepSeek-V4 (`deepseek4`) metadata loading.

Stage A of DeepSeek-V4-Flash support: the checkpoint has to open, report a
faithful configuration and refuse execution with a message that says why. These
run against a miniature fixture; `DEEPSEEK4_GGUF` points them at a real
checkpoint when one is on disk.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from colibri_next.v2 import V2Error, V2Model

from tests.deepseek4_gguf_fixture import DeepSeek4Spec, build_deepseek4_gguf
from tests.split_gguf_fixture import split_gguf


def _model(**kwargs) -> tuple[V2Model, DeepSeek4Spec]:
    directory = Path(tempfile.mkdtemp(prefix="colibri-deepseek4-"))
    path = directory / "deepseek4.gguf"
    spec = build_deepseek4_gguf(path, **kwargs)
    return V2Model(path), spec


class DeepSeek4ConfigTests(unittest.TestCase):
    def test_architecture_and_core_geometry(self):
        model, spec = _model()
        try:
            config = model.config
            self.assertEqual(config["architecture"], "deepseek4")
            self.assertEqual(config["layer_count"], spec.layers)
            self.assertEqual(config["hidden_size"], spec.hidden)
            self.assertEqual(config["attention_heads"], spec.heads)
            self.assertEqual(config["attention_kv_heads"], spec.kv_heads)
            self.assertEqual(config["context_length"], 1_048_576)
            self.assertEqual(config["rotary_dimension"], spec.rope_dimension)
        finally:
            model.close()

    def test_latent_attention_ranks(self):
        model, spec = _model()
        try:
            config = model.config
            self.assertEqual(config["q_lora_rank"], spec.q_lora_rank)
            # No explicit kv_lora_rank key ships; it is the KV latent width.
            self.assertEqual(config["kv_lora_rank"], spec.kv_lora_rank)
            self.assertEqual(config["output_lora_rank"], spec.output_lora_rank)
            self.assertEqual(config["output_group_count"], spec.output_groups)
        finally:
            model.close()

    def test_indexer_and_hyper_connection_metadata(self):
        model, spec = _model()
        try:
            config = model.config
            self.assertEqual(config["indexer_head_count"], spec.indexer_heads)
            self.assertEqual(config["indexer_key_length"], spec.indexer_key_length)
            self.assertEqual(config["indexer_top_k"], spec.indexer_top_k)
            self.assertEqual(config["hyper_connection_count"], spec.hyper_connections)
            self.assertEqual(config["sinkhorn_iterations"], spec.sinkhorn_iterations)
            self.assertAlmostEqual(config["sinkhorn_epsilon"], 1e-6, places=9)
            self.assertEqual(config["hash_layer_count"], 3)
        finally:
            model.close()

    def test_moe_metadata_matches_the_checkpoint_conventions(self):
        model, spec = _model()
        try:
            config = model.config
            self.assertEqual(config["expert_count"], spec.experts)
            self.assertEqual(config["expert_used_count"], spec.experts_used)
            self.assertEqual(config["expert_shared_count"], 1)
            self.assertEqual(config["intermediate_size"], spec.expert_intermediate)
        finally:
            model.close()

    def test_compress_ratios_are_reported_per_layer(self):
        model, spec = _model()
        try:
            ratios = model.compress_ratios
            # The header carries more entries than layers; all of them survive
            # so the trailing draft-block entries stay inspectable.
            self.assertEqual(ratios, spec.compress_ratios)
            self.assertEqual(len(ratios), spec.layers + spec.extra_compress_ratios)
            self.assertEqual(set(ratios[: spec.layers]) - {0, 4, 128}, set())
            self.assertEqual(model.config["compress_ratios_length"], len(ratios))
            self.assertAlmostEqual(
                model.config["compress_rope_freq_base"], 160_000.0, places=1
            )
        finally:
            model.close()

    def test_a_short_compress_ratio_array_is_rejected(self):
        directory = Path(tempfile.mkdtemp(prefix="colibri-deepseek4-"))
        path = directory / "short.gguf"
        # One entry fewer than the block count.
        build_deepseek4_gguf(path, DeepSeek4Spec(layers=6, extra_compress_ratios=-1))
        with self.assertRaises(V2Error) as raised:
            V2Model(path)
        self.assertIn("compress-ratio", str(raised.exception))

    def test_execution_is_refused_with_a_reason(self):
        model, _ = _model()
        try:
            with self.assertRaises(V2Error) as raised:
                model.native_runtime(context_limit=64, mtp_drafts=0)
            message = str(raised.exception)
            self.assertIn("deepseek4", message)
            self.assertIn("cannot execute", message)
        finally:
            model.close()

    def test_a_split_deepseek4_checkpoint_loads_like_the_real_layout(self):
        directory = Path(tempfile.mkdtemp(prefix="colibri-deepseek4-"))
        single = directory / "deepseek4.gguf"
        spec = build_deepseek4_gguf(single)
        first = split_gguf(single, directory, shards=4, metadata_only_first=True)
        whole, sharded = V2Model(single), V2Model(first)
        try:
            self.assertEqual(whole.info["tensor_count"], sharded.info["tensor_count"])
            self.assertEqual(sharded.config["architecture"], "deepseek4")
            self.assertEqual(sharded.config["layer_count"], spec.layers)
            self.assertEqual(sharded.compress_ratios, spec.compress_ratios)
            self.assertEqual(
                sorted(tensor["name"] for tensor in whole.tensors()),
                sorted(tensor["name"] for tensor in sharded.tensors()),
            )
        finally:
            whole.close()
            sharded.close()


class DeepSeek4TensorPlanTests(unittest.TestCase):
    """The tensor roles a deepseek4 block carries, by the reference's names."""

    def setUp(self):
        self.model, self.spec = _model()
        self.addCleanup(self.model.close)
        self.names = {str(tensor["name"]) for tensor in self.model.tensors()}

    def test_every_block_carries_latent_attention_and_hyper_connections(self):
        for layer in range(self.spec.layers):
            prefix = f"blk.{layer}."
            for role in (
                "attn_q_a.weight",
                "attn_q_a_norm.weight",
                "attn_q_b.weight",
                "attn_kv.weight",
                "attn_kv_a_norm.weight",
                "attn_output_a.weight",
                "attn_output_b.weight",
                "attn_sinks.weight",
                "hc_attn_base.weight",
                "hc_attn_fn.weight",
                "hc_attn_scale.weight",
                "hc_ffn_base.weight",
                "hc_ffn_fn.weight",
                "hc_ffn_scale.weight",
            ):
                self.assertIn(prefix + role, self.names)

    def test_compressors_follow_the_compress_ratio(self):
        for layer in range(self.spec.layers):
            prefix = f"blk.{layer}."
            ratio = self.spec.compress_ratios[layer]
            compressor = prefix + "attn_compressor_kv.weight"
            indexer = prefix + "indexer.proj.weight"
            # Sliding-window layers compress nothing; only 4:1 layers index.
            self.assertEqual(compressor in self.names, ratio != 0, msg=prefix)
            self.assertEqual(indexer in self.names, ratio == 4, msg=prefix)

    def test_the_output_head_has_its_own_hyper_connection_mixer(self):
        for role in ("output_hc_base.weight", "output_hc_fn.weight", "output_hc_scale.weight"):
            self.assertIn(role, self.names)

    def test_only_the_hash_tables_are_undecodable(self):
        # Same shape of gap as the real checkpoint: the int32 routing tables
        # are index data rather than weights, so they have no decode kernel.
        self.assertEqual(
            {kind: len(names) for kind, names in self.model.unsupported_quant_types().items()},
            {26: self.spec.hash_layers},
        )

    def test_hash_layers_route_by_table_and_the_rest_by_router_bias(self):
        for layer in range(self.spec.layers):
            prefix = f"blk.{layer}."
            hashed = layer < self.spec.hash_layers
            self.assertEqual(prefix + "ffn_gate_tid2eid.weight" in self.names, hashed, msg=prefix)
            self.assertEqual(prefix + "exp_probs_b.bias" in self.names, not hashed, msg=prefix)


@unittest.skipUnless(
    os.environ.get("DEEPSEEK4_GGUF"),
    "set DEEPSEEK4_GGUF to the first shard of a real checkpoint",
)
class DeepSeek4CheckpointTests(unittest.TestCase):
    """Runs against the published UD-IQ3_XXS build when it is on disk."""

    def setUp(self):
        self.model = V2Model(Path(os.environ["DEEPSEEK4_GGUF"]))
        self.addCleanup(self.model.close)

    def test_real_checkpoint_geometry(self):
        config = self.model.config
        self.assertEqual(config["architecture"], "deepseek4")
        self.assertEqual(config["layer_count"], 43)
        self.assertEqual(config["hidden_size"], 4096)
        self.assertEqual(config["attention_heads"], 64)
        self.assertEqual(config["attention_kv_heads"], 1)
        self.assertEqual(config["expert_count"], 256)
        self.assertEqual(config["expert_used_count"], 6)
        self.assertEqual(config["expert_shared_count"], 1)
        self.assertEqual(config["q_lora_rank"], 1024)
        self.assertEqual(config["kv_lora_rank"], 512)
        self.assertEqual(config["output_lora_rank"], 1024)
        self.assertEqual(config["output_group_count"], 8)
        self.assertEqual(config["indexer_top_k"], 512)
        self.assertEqual(config["hyper_connection_count"], 4)
        self.assertEqual(config["sinkhorn_iterations"], 20)
        self.assertEqual(config["context_length"], 1_048_576)

    def test_all_four_shards_are_mapped(self):
        # split.tensors.count for the UD-IQ3_XXS build.
        self.assertEqual(int(self.model.info["tensor_count"]), 1328)

    def test_compress_ratios_cover_every_layer(self):
        ratios = self.model.compress_ratios
        self.assertGreaterEqual(len(ratios), 43)
        self.assertEqual(set(ratios) - {0, 4, 128}, set())

    def test_the_only_undecodable_types_are_the_two_known_gaps(self):
        # Unsloth's dynamic quants mix types per tensor, so this is where a
        # format the runtime has no kernel for surfaces. UD-IQ3_XXS needs two
        # things the runtime does not have yet, and nothing else:
        #
        #   26 = I32, the `ffn_gate_tid2eid` routing tables in the three hash
        #        layers. An integer lookup table rather than a quantized
        #        weight, so it needs routing support, not a decode kernel.
        #   39 = MXFP4 (17 bytes per 32 values, 4.25 bits). Adjacent to the
        #        NVFP4 (type 40) already implemented, which is 4.5 bits over
        #        blocks of 16, so the two are not interchangeable.
        #
        # Pinning the exact set means a third gap appearing is a test failure
        # rather than a surprise during Stage B.
        offenders = self.model.unsupported_quant_types()
        self.assertEqual(
            {kind: len(names) for kind, names in sorted(offenders.items())},
            {26: 3, 39: 2},
            msg="undecodable GGML types: "
            + ", ".join(
                f"type {kind} in {len(names)} tensors e.g. {names[0]}"
                for kind, names in sorted(offenders.items())
            ),
        )

    def test_hash_layers_carry_a_routing_table_each(self):
        tables = [
            str(tensor["name"])
            for tensor in self.model.tensors()
            if str(tensor["name"]).endswith("ffn_gate_tid2eid.weight")
        ]
        # One per hash layer, each a [experts_used, vocabulary] int32 table.
        self.assertEqual(len(tables), self.model.config["hash_layer_count"])
        shape = self.model.tensor(tables[0])["shape"]
        self.assertEqual(
            shape,
            (self.model.config["expert_used_count"], self.model.config["vocabulary_size"]),
        )

    def test_the_tokenizer_is_the_one_this_port_implements(self):
        self.assertEqual(
            self.model.pretokenize("mixed 日本語 and 123 text"),
            ("mixed", " ", "日本語", " and", " ", "123", " text"),
        )


if __name__ == "__main__":
    unittest.main()
