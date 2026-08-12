"""HF safetensors loading, quantization at load, and name translation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from colibri_next.v2 import V2Model
from tests import hf_safetensors_fixture as fixture


class HfLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = fixture.build(Path(cls._directory.name) / "ling-tiny-fixture")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def open_model(self) -> V2Model:
        return V2Model(self.path)

    def test_it_opens_a_directory_as_safetensors(self) -> None:
        with self.open_model() as model:
            info = model.info
            self.assertEqual(info["format"], "safetensors")
            self.assertEqual(info["architecture"], "bailingmoe3")
            self.assertEqual(info["name"], "ling-tiny-fixture")

    def test_it_translates_every_tensor_name(self) -> None:
        with self.open_model() as model:
            names = {str(tensor["name"]) for tensor in model.tensors()}
        # Nothing may keep an HF-style name.
        self.assertFalse([name for name in names if name.startswith("model.layers")])
        self.assertIn("token_embd.weight", names)
        self.assertIn("output_norm.weight", names)
        self.assertIn("output.weight", names)
        # Layer 3 closes the group of 4, so it is the MLA layer; 0-2 are KDA.
        self.assertIn("blk.3.attn_kv_a_mqa.weight", names)
        self.assertIn("blk.3.attn_q_a.weight", names)
        self.assertIn("blk.0.ssm_q.weight", names)
        self.assertIn("blk.0.ssm_dt.bias", names)
        self.assertNotIn("blk.0.attn_q_a.weight", names)
        self.assertNotIn("blk.3.ssm_q.weight", names)

    def test_it_keeps_the_two_gate_projections_apart(self) -> None:
        # attention.g_proj means the head-wise attention gate on an MLA layer
        # and the KDA output gate on a linear one. Collapsing them would be
        # silent, so both spellings must appear on the right layers.
        with self.open_model() as model:
            names = {str(tensor["name"]) for tensor in model.tensors()}
        self.assertIn("blk.3.attn_gate.weight", names)
        self.assertIn("blk.0.ssm_g.weight", names)
        self.assertNotIn("blk.0.attn_gate.weight", names)
        self.assertNotIn("blk.3.ssm_g.weight", names)

    def test_it_stacks_routed_experts_across_shards(self) -> None:
        with self.open_model() as model:
            catalog = {str(t["name"]): t for t in model.tensors()}
        # The fixture puts experts 0-1 in one shard and 2-3 in another; the
        # stacked descriptor has to span both.
        for projection in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            tensor = catalog[f"blk.1.{projection}.weight"]
            self.assertEqual(
                tensor["shape"][-1],
                fixture.EXPERTS,
                f"{projection} did not stack every expert",
            )
        # Layer 0 is the dense leading block and has no routed experts.
        self.assertNotIn("blk.0.ffn_gate_exps.weight", catalog)
        self.assertIn("blk.0.ffn_gate.weight", catalog)

    def test_it_quantizes_weights_and_leaves_small_tensors_alone(self) -> None:
        with self.open_model() as model:
            catalog = {str(t["name"]): t for t in model.tensors()}
        # 2-D weights become Q4_K (12) by default.
        self.assertEqual(catalog["blk.1.ffn_gate_exps.weight"]["ggml_type"], 12)
        self.assertEqual(catalog["blk.3.attn_q_a.weight"]["ggml_type"], 12)
        # The embedding and head take a higher target.
        self.assertEqual(catalog["token_embd.weight"]["ggml_type"], 14)
        self.assertEqual(catalog["output.weight"]["ggml_type"], 14)
        # Norms and 1-D tensors stay f32, where quantization error is worst per
        # byte saved.
        self.assertEqual(catalog["output_norm.weight"]["ggml_type"], 0)
        self.assertEqual(catalog["blk.0.attn_norm.weight"]["ggml_type"], 0)
        self.assertEqual(catalog["blk.0.ssm_dt.bias"]["ggml_type"], 0)
        # The 4-tap conv kernel is too short a row to quantize.
        self.assertEqual(catalog["blk.0.ssm_q_conv1d.weight"]["ggml_type"], 0)

    def test_it_selects_the_pretokenizer_from_the_regex(self) -> None:
        # The fixture carries the llama3/BailingMoE3 split pattern. It differs
        # from GPT-4o's in two places, and both are load-bearing: digits stand
        # alone, and a letter run is NOT split on an upper/lower transition.
        with self.open_model() as model:
            self.assertEqual(list(model.pretokenize("1234")), ["1", "2", "3", "4"])
            self.assertEqual(list(model.pretokenize("mV")), ["mV"])
            self.assertEqual(list(model.pretokenize("aB cD")), ["aB", " cD"])

    def test_it_keeps_a_whitespace_run_through_its_last_line_break(self) -> None:
        # `\\s*[\\r\\n]` is greedy and backtracks, so it reaches the last line
        # break in the run. Stopping at the first splits "\\r\\r" and blocks the
        # merge across it.
        with self.open_model() as model:
            self.assertEqual(list(model.pretokenize("\r\r")), ["\r\r"])
            self.assertEqual(list(model.pretokenize("\n\t\r\r \n")), ["\n\t\r\r \n"])

    def test_it_loads_the_vocabulary_and_control_tokens(self) -> None:
        with self.open_model() as model:
            self.assertEqual(model.token_id("<|startoftext|>"), 258)
            self.assertEqual(model.token_id("abc"), 257)
            # Control tokens are split off ahead of BPE; `special: false` added
            # tokens are ordinary vocabulary and are not.
            self.assertEqual(model.token_id("<|not_special|>"), 260)

    def test_it_reads_the_group_limited_routing_config(self) -> None:
        # n_group / topk_group are what make the router select over expert
        # groups rather than a flat list. Dropping them on the floor would
        # still route, just wrongly, so they are checked explicitly.
        with self.open_model() as model:
            config = model.config
        self.assertEqual(config["expert_group_count"], fixture.CONFIG["n_group"])
        self.assertEqual(config["expert_group_used"], fixture.CONFIG["topk_group"])

    def test_it_rejects_a_checkpoint_missing_a_layer_tensor(self) -> None:
        # The loader gate proves descriptors point at the right bytes; it cannot
        # prove they were given the right meaning. Resolving the layer plan at
        # open is what catches a misnamed tensor, so it has to actually fail.
        import json
        import shutil

        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "broken"
            shutil.copytree(self.path, broken)
            index_path = broken / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            # Rename one KDA tensor so its bytes are still present and readable
            # but its meaning is gone -- exactly the failure the byte-level gate
            # cannot see.
            victim = "model.layers.0.attention.dt_bias"
            self.assertIn(victim, index["weight_map"])
            shard = index["weight_map"].pop(victim)
            index_path.write_text(json.dumps(index))
            self._drop_from_shard(broken / shard, victim)

            with self.assertRaises(Exception) as caught:
                V2Model(broken).close()
            # The message names the layer and the GGUF-side tensor, which is
            # what a person debugging a conversion needs to see.
            message = str(caught.exception)
            self.assertIn("layer 0", message)
            self.assertIn("ssm_dt.bias", message)

    def test_it_rejects_a_tensor_it_cannot_map(self) -> None:
        # A name the translation table does not know is refused outright rather
        # than skipped. Skipping would leave the tensor's bytes unreachable and
        # the model quietly incomplete.
        import json
        import shutil

        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "broken"
            shutil.copytree(self.path, broken)
            index_path = broken / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            victim = "model.layers.0.attention.dt_bias"
            renamed = "model.layers.0.attention.mystery"
            shard = index["weight_map"].pop(victim)
            index["weight_map"][renamed] = shard
            index_path.write_text(json.dumps(index))
            self._rename_in_shard(broken / shard, victim, renamed)

            with self.assertRaises(Exception) as caught:
                V2Model(broken).close()
            self.assertIn("mystery", str(caught.exception))

    @staticmethod
    def _rewrite_header(path: Path, mutate) -> None:
        import json
        import struct

        blob = path.read_bytes()
        length = struct.unpack("<Q", blob[:8])[0]
        header = json.loads(blob[8 : 8 + length])
        mutate(header)
        encoded = json.dumps(header).encode()
        encoded += b" " * ((8 - len(encoded) % 8) % 8)
        # Payloads are addressed by data_offsets relative to the end of the
        # header, so they survive a header rewrite untouched.
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob[8 + length :])

    @classmethod
    def _rename_in_shard(cls, path: Path, old: str, new: str) -> None:
        cls._rewrite_header(path, lambda header: header.__setitem__(new, header.pop(old)))

    @classmethod
    def _drop_from_shard(cls, path: Path, name: str) -> None:
        cls._rewrite_header(path, lambda header: header.pop(name))

    def cache_files(self, directory: Path | None = None) -> list[Path]:
        return sorted((directory or self.path).glob("colibri-*.cache"))

    def catalog(self, model: V2Model) -> dict[str, tuple]:
        return {
            str(t["name"]): (t["ggml_type"], tuple(t["shape"]), t["size"])
            for t in model.tensors()
        }

    def test_it_caches_the_quantized_arena_beside_the_checkpoint(self) -> None:
        # The packers cost hundreds of core-seconds on a real checkpoint and
        # their inputs never change, so the second open must not redo them.
        with self.open_model() as model:
            first = self.catalog(model)
            sample = model.read_tensor("blk.1.ffn_gate_exps.weight").tobytes()
        cached = self.cache_files()
        self.assertEqual(len(cached), 1, "expected exactly one cache sidecar")

        with self.open_model() as model:
            self.assertEqual(self.catalog(model), first)
            # Descriptors matching is not enough -- they could point anywhere.
            # The bytes behind a multi-part stacked expert are the strongest
            # thing to compare, since that is where a wrong offset would land.
            self.assertEqual(
                model.read_tensor("blk.1.ffn_gate_exps.weight").tobytes(), sample
            )

    def test_it_ignores_a_cache_whose_checkpoint_changed(self) -> None:
        # The fingerprint is (config, shard sizes, shard mtimes, policy). A
        # stale cache must be a miss, not a wrong answer.
        import shutil

        with self.open_model():
            pass
        cached = self.cache_files()[0]
        with tempfile.TemporaryDirectory() as directory:
            moved = Path(directory) / "ling-tiny-fixture"
            shutil.copytree(self.path, moved)
            # Touching a shard changes its mtime and nothing else. The cache
            # file copied alongside it now describes a checkpoint that no
            # longer exists.
            shard = next(moved.glob("*.safetensors"))
            shard.touch()
            self.assertTrue((moved / cached.name).exists())
            with V2Model(moved) as model:
                names = {str(t["name"]) for t in model.tensors()}
            self.assertIn("blk.1.ffn_gate_exps.weight", names)
            # A fresh cache is written under the new fingerprint; the stale one
            # is left alone rather than guessed at.
            self.assertEqual(len(self.cache_files(moved)), 2)

    def test_it_survives_a_truncated_cache(self) -> None:
        # A cache killed mid-write, or on a full disk, must read as a miss.
        # Every offset in the table is bounds-checked for exactly this.
        import shutil

        with self.open_model():
            pass
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "ling-tiny-fixture"
            shutil.copytree(self.path, copy)
            cached = self.cache_files(copy)[0]
            blob = cached.read_bytes()
            cached.write_bytes(blob[: len(blob) // 2])
            with V2Model(copy) as model:
                self.assertIn(
                    "blk.1.ffn_gate_exps.weight",
                    {str(t["name"]) for t in model.tensors()},
                )

    def test_it_keeps_a_cache_per_quantization(self) -> None:
        # The policy is part of the fingerprint, so switching COLIBRI_HF_QUANT
        # must not hand back the arena packed for the other target.
        import os
        import shutil

        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "ling-tiny-fixture"
            shutil.copytree(self.path, copy)
            for stale in self.cache_files(copy):
                stale.unlink()

            previous = os.environ.get("COLIBRI_HF_QUANT")
            try:
                with V2Model(copy) as model:
                    self.assertEqual(
                        model.tensor("blk.3.attn_q_a.weight")["ggml_type"], 12
                    )
                os.environ["COLIBRI_HF_QUANT"] = "Q8_0"
                with V2Model(copy) as model:
                    self.assertEqual(
                        model.tensor("blk.3.attn_q_a.weight")["ggml_type"], 8
                    )
            finally:
                if previous is None:
                    os.environ.pop("COLIBRI_HF_QUANT", None)
                else:
                    os.environ["COLIBRI_HF_QUANT"] = previous
            self.assertEqual(len(self.cache_files(copy)), 2)

    def test_it_can_be_told_not_to_cache(self) -> None:
        import os
        import shutil

        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "ling-tiny-fixture"
            shutil.copytree(self.path, copy)
            for stale in self.cache_files(copy):
                stale.unlink()
            previous = os.environ.get("COLIBRI_HF_CACHE")
            os.environ["COLIBRI_HF_CACHE"] = "0"
            try:
                with V2Model(copy) as model:
                    self.assertIn(
                        "token_embd.weight",
                        {str(t["name"]) for t in model.tensors()},
                    )
            finally:
                if previous is None:
                    os.environ.pop("COLIBRI_HF_CACHE", None)
                else:
                    os.environ["COLIBRI_HF_CACHE"] = previous
            self.assertEqual(self.cache_files(copy), [])

    def test_a_prompt_means_the_same_whether_it_arrives_whole_or_in_pieces(
        self,
    ) -> None:
        # Prefill and decode run on different devices: the host has a batched
        # prefill, the device has the faster decode, and the per-layer caches
        # are transferred between them at the transition. Forget that transfer
        # and decode reads a cache the prompt never touched -- the model answers
        # with no context, first token plausible and everything after it
        # invented.
        #
        # Feeding a prompt one token at a time keeps both phases on one path, so
        # it is the control. Only sending the prompt whole splits them, which is
        # why the original bug survived a test suite that only ever did the
        # former. This does both and requires they agree.
        from colibri_next.v2 import BailingRuntime

        prompt = [3, 9, 27, 81, 5, 15, 45, 7]
        with self.open_model() as model:
            runtime = BailingRuntime(model, capacity=128)
            try:
                runtime.reset()
                runtime.eval(prompt)
                whole = [runtime.sample()]
                for _ in range(5):
                    runtime.eval([whole[-1]])
                    whole.append(runtime.sample())

                runtime.reset()
                for token in prompt:
                    runtime.eval([token])
                pieces = [runtime.sample()]
                for _ in range(5):
                    runtime.eval([pieces[-1]])
                    pieces.append(runtime.sample())
            finally:
                runtime.close()

        self.assertEqual(whole, pieces)

    def test_it_generates_from_a_thread_that_did_not_build_the_runtime(self) -> None:
        # CUDA driver contexts are per thread, and a server builds the runtime
        # on one thread and generates on an engine worker. Every device call
        # from that worker fails until the primary context is bound there -- so
        # a device entry point that forgets to bind works perfectly in tests and
        # dies in the server.
        #
        # That is not hypothetical: the cache transfer between the host prefill
        # and the device decode forgot exactly this, and every single-threaded
        # test here passed while the server failed on its first token with
        # "bailing cache transfer failed". Driving from a second thread is what
        # closes that gap, and it costs nothing on a host-only machine.
        import threading

        from colibri_next.v2 import BailingRuntime

        prompt = [3, 9, 27, 81, 5, 15, 45, 7]
        with self.open_model() as model:
            runtime = BailingRuntime(model, capacity=128)
            result: dict[str, object] = {}

            def generate() -> None:
                try:
                    runtime.reset()
                    # A whole prompt then single tokens: the shape that crosses
                    # from the batched host prefill to the device decode.
                    runtime.eval(prompt)
                    tokens = [runtime.sample()]
                    for _ in range(3):
                        runtime.eval([tokens[-1]])
                        tokens.append(runtime.sample())
                    result["tokens"] = tokens
                except Exception as error:  # surfaced below, not swallowed
                    result["error"] = error

            worker = threading.Thread(target=generate)
            worker.start()
            worker.join(timeout=120)
            runtime.close()

        self.assertFalse(worker.is_alive(), "generation thread did not finish")
        self.assertNotIn("error", result, f"generating off-thread failed: {result.get('error')}")
        self.assertEqual(len(result["tokens"]), 4)  # type: ignore[arg-type]

    def test_it_quantizes_narrow_rows_rather_than_leaving_them_f32(self) -> None:
        # A row shorter than a 256-element super-block cannot hold one, but it
        # can hold Q8_0's 32-element blocks. The MLA up-projections have
        # q_lora_rank-wide rows and are matvecs on every full-attention layer;
        # left f32 they had no GPU matvec kernel at all, so the whole model fell
        # back to the host.
        with self.open_model() as model:
            catalog = {str(t["name"]): t for t in model.tensors()}
        for name in ("blk.3.attn_q_b.weight", "blk.3.attn_kv_b.weight"):
            tensor = catalog[name]
            self.assertLess(int(tensor["shape"][0]), 256, f"{name} is not a narrow row")
            self.assertEqual(tensor["ggml_type"], 8, f"{name} should be Q8_0")
        # The 4-tap conv kernel is still too short for even a 32-element block.
        self.assertEqual(catalog["blk.0.ssm_q_conv1d.weight"]["ggml_type"], 0)

    def test_it_shrinks_the_checkpoint(self) -> None:
        source = sum(
            path.stat().st_size for path in self.path.glob("*.safetensors")
        )
        with self.open_model() as model:
            quantized = sum(int(t["size"]) for t in model.tensors())
        self.assertLess(quantized, source)


if __name__ == "__main__":
    unittest.main()
