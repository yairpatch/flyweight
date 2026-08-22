"""Running BailingMoE3 from a GGUF conversion rather than an HF checkpoint."""

from __future__ import annotations

import contextlib
import os
import tempfile
import unittest
from pathlib import Path

from colibri_next.v2 import BailingRuntime, V2Error, V2Model
from tests import bailing_gguf_fixture as gguf
from tests import hf_safetensors_fixture as safetensors


class BailingGgufTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        cls.hf_path = safetensors.build(root / "ling-tiny-fixture")
        cls.gguf_path = gguf.build(root / "converted", cls.hf_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_it_reads_the_architecture_and_geometry_from_gguf_metadata(self) -> None:
        with V2Model(self.gguf_path) as model:
            info = model.info
            self.assertEqual(info["format"], "gguf")
            self.assertEqual(info["architecture"], "bailingmoe3")
            config = model.config
            # Derived from the per-layer kv head counts: the GGUF carries no
            # full_attention_interval of its own.
            self.assertEqual(config["full_attention_interval"], 4)
            self.assertEqual(config["expert_group_count"], 2)
            self.assertEqual(config["expert_group_used"], 1)

    @staticmethod
    @contextlib.contextmanager
    def unquantized():
        """Open the safetensors side f32, as the conversion is written.

        The loader otherwise quantizes on the way in, and a difference of that
        size would swamp the mapping mistakes these tests exist to catch.
        """
        previous = os.environ.get("COLIBRI_HF_QUANT")
        os.environ["COLIBRI_HF_QUANT"] = "F32"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("COLIBRI_HF_QUANT", None)
            else:
                os.environ["COLIBRI_HF_QUANT"] = previous

    def test_it_tokenizes_the_same_text_the_same_way(self) -> None:
        text = "the quick brown fox abc 123"
        with V2Model(self.gguf_path) as model:
            from_gguf = model.tokenize(text)
        with V2Model(self.hf_path) as model:
            from_hf = model.tokenize(text)
        self.assertEqual(from_gguf, from_hf)

    def logits(self, path, prompt: list[int]) -> list[float]:
        with V2Model(path) as model:
            runtime = BailingRuntime(model, capacity=64)
            try:
                runtime.reset()
                return list(runtime.eval(prompt))
            finally:
                runtime.close()

    def test_a_conversion_answers_exactly_as_the_checkpoint_it_came_from(self) -> None:
        # The same weights through both loaders. Everything the conversion
        # renames, splits, transposes or exponentiates is covered by this one
        # assertion, and each of those is a mistake that produces fluent but
        # wrong text rather than an error.
        #
        # More than one token matters: a rope or recurrence mistake is exactly
        # zero at position 0 and only appears once there is history, which is
        # how a broken kernel launch passed a single-token check.
        prompt = [5, 11, 23, 4, 9, 17, 3, 8]
        for length in (1, 2, len(prompt)):
            with self.subTest(prompt_tokens=length):
                converted = self.logits(self.gguf_path, prompt[:length])
                with self.unquantized():
                    original = self.logits(self.hf_path, prompt[:length])
                self.assertEqual(len(converted), len(original))
                worst = max(abs(a - b) for a, b in zip(converted, original))
                scale = max(abs(value) for value in original)
                self.assertLess(worst, 1e-3 * max(scale, 1.0))
                self.assertEqual(
                    converted.index(max(converted)),
                    original.index(max(original)),
                )

    def test_the_resume_separator_is_the_markup_the_template_emits(self) -> None:
        # Resuming a cached conversation splices this between the reused turn
        # and the next one, and an agentic loop resumes on EVERY turn: a turn
        # that ends in a tool call is cancelled rather than finished on EOS, so
        # the separator is what closes it.
        #
        # The ChatML default was not just wrong markup, it was not markup at
        # all -- <|im_end|> is absent from this vocabulary and tokenizes as
        # five ordinary text tokens, which is a literal string of junk in the
        # middle of the conversation. So assert the separator is a single token
        # and that the template really emits it between turns.
        from colibri_next.v2_server import NativeV2Tokenizer

        with V2Model(self.gguf_path) as model:
            tokenizer = NativeV2Tokenizer(model)
            separator = tokenizer.turn_separator
            self.assertEqual(len(model.tokenize(separator)), 1, separator)
            rendered = tokenizer.format_messages([
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ])
            # What the template puts between a closed assistant turn and the
            # next user turn is exactly separator + finished_turn_separator.
            self.assertIn(
                "two" + separator + tokenizer.finished_turn_separator,
                rendered,
            )

    def test_it_decodes_the_same_continuation(self) -> None:
        prompt = [5, 11, 23, 4, 9, 17, 3, 8]

        def generate(path) -> list[int]:
            with V2Model(path) as model:
                runtime = BailingRuntime(model, capacity=64)
                try:
                    runtime.reset()
                    runtime.eval_into(prompt)
                    out = []
                    for _ in range(6):
                        out.append(runtime.sample())
                        runtime.eval_into([out[-1]])
                    return out
                finally:
                    runtime.close()

        converted = generate(self.gguf_path)
        with self.unquantized():
            original = generate(self.hf_path)
        self.assertEqual(converted, original)


class BailingRuntimeStateTests(unittest.TestCase):
    """Snapshotting a sequence, and watching a prompt while it runs.

    Both are runtime bookkeeping rather than arithmetic, so they run on the host
    path on purpose: the device keeps its own copy of every cache, and letting
    the machine decide which one these assertions are about would make them
    report on the hardware instead of the code.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        hf_path = safetensors.build(root / "ling-tiny-fixture")
        cls.gguf_path = gguf.build(root / "converted", hf_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    @contextlib.contextmanager
    def host_runtime(self, capacity: int = 64):
        previous = os.environ.get("COLIBRI_BAILING_GPU")
        os.environ["COLIBRI_BAILING_GPU"] = "0"
        try:
            with V2Model(self.gguf_path) as model:
                runtime = BailingRuntime(model, capacity=capacity)
                try:
                    runtime.reset()
                    yield runtime
                finally:
                    runtime.close()
        finally:
            if previous is None:
                os.environ.pop("COLIBRI_BAILING_GPU", None)
            else:
                os.environ["COLIBRI_BAILING_GPU"] = previous

    def test_it_reports_the_host_path_it_was_forced_onto(self) -> None:
        with self.host_runtime() as runtime:
            self.assertIs(runtime.uses_gpu, False)

    def test_a_restored_snapshot_continues_exactly_as_the_original_would(self) -> None:
        # The property the prefix cache is built on, and the only one that
        # proves the serialization is complete: anything left out of the
        # snapshot -- a convolution window, the KDA recurrent state, the rope
        # keys -- still produces a plausible continuation, just not this one.
        #
        # Decoding several tokens past the restore matters. The recurrent state
        # is a running summary, so a partial restore is nearly right at the
        # first token and drifts; a single-token comparison passed against a
        # snapshot that carried the latents alone.
        prompt = [5, 11, 23, 4, 9, 17, 3, 8]
        tail = [12, 6, 19, 2]

        def continue_from(runtime) -> tuple[list[float], list[int]]:
            logits = list(runtime.eval(tail))
            decoded = []
            for _ in range(5):
                decoded.append(runtime.sample())
                runtime.eval_into([decoded[-1]])
            return logits, decoded

        with self.host_runtime() as runtime:
            runtime.eval_into(prompt)
            snapshot = runtime.save_state()
            straight = continue_from(runtime)
            # Run an unrelated sequence in between. A "restore" that quietly
            # kept whatever was live would otherwise be flattered by the fact
            # that the live caches already held the right prompt.
            runtime.reset()
            runtime.eval_into([31, 29, 27, 25, 23, 21])
            runtime.load_state(snapshot)
            restored = continue_from(runtime)
        self.assertEqual(restored, straight)

    def test_a_snapshot_is_sized_by_the_sequence_not_the_capacity(self) -> None:
        short = [(index * 7) % 400 + 1 for index in range(8)]
        rest = [(index * 13) % 400 + 1 for index in range(1000)]
        with self.host_runtime(capacity=1024) as runtime:
            runtime.eval_into(short)
            brief = runtime.save_state()
            runtime.eval_into(rest)
            lengthy = runtime.save_state()
        # The per-token part of the cache is the MLA latents and rope keys; the
        # KDA layers are a fixed recurrent state whatever the length. So a
        # thousand tokens is a multiple of eight tokens, not a thousand times
        # it -- but the growth has to be there.
        self.assertGreater(len(lengthy), 2 * len(brief))
        # And the sixteen-times-smaller runtime snapshots the same eight tokens
        # to the same bytes. That is the whole claim: capacity is not in the
        # size, so a 32-token side-call does not pay for a 128k context.
        with self.host_runtime(capacity=64) as runtime:
            runtime.eval_into(short)
            self.assertEqual(runtime.save_state(), brief)

    def test_a_damaged_snapshot_is_refused_rather_than_misread(self) -> None:
        prompt = [5, 11, 23, 4, 9, 17, 3, 8]
        with self.host_runtime() as runtime:
            runtime.eval_into(prompt)
            snapshot = runtime.save_state()
            damaged = {
                "empty": b"",
                "header only, half of it": snapshot[:20],
                "wrong magic": b"NOTACACH" + snapshot[8:],
                "truncated payload": snapshot[:-4],
                "a byte flipped in the geometry": (
                    snapshot[:12] + bytes([snapshot[12] ^ 0x7F]) + snapshot[13:]
                ),
            }
            for name, blob in damaged.items():
                with self.subTest(damage=name):
                    with self.assertRaises(V2Error):
                        runtime.load_state(blob)
            # A refused load must not have consumed the runtime on the way out.
            runtime.load_state(snapshot)
            self.assertEqual(len(runtime.save_state()), len(snapshot))

    def test_a_watched_prompt_reports_its_way_through(self) -> None:
        prompt = [5, 11, 23, 4, 9, 17, 3, 8]
        seen: list[tuple[int, int]] = []
        with self.host_runtime() as runtime:
            runtime.set_progress(lambda processed, total: seen.append(
                (processed, total)) is None)
            runtime.eval_into(prompt)
            # A decoded token is not a prompt: reporting "0 of 1" then "1 of 1"
            # for every token of every generation is pure overhead.
            before = len(seen)
            runtime.eval_into([7])
            self.assertEqual(len(seen), before)
            runtime.set_progress(None)
            runtime.eval_into([7])
        self.assertEqual(len(seen), before)
        self.assertEqual([total for _, total in seen], [len(prompt)] * len(seen))
        processed = [value for value, _ in seen]
        self.assertGreater(len(processed), 2)
        self.assertEqual(processed[0], 0)
        self.assertEqual(processed[-1], len(prompt))
        # Strictly increasing: a watcher rendering a progress bar has to be able
        # to treat these as a position, not as a sequence of hints.
        self.assertEqual(processed, sorted(set(processed)))

    def test_a_watcher_that_says_stop_stops_the_prompt(self) -> None:
        prompt = [5, 11, 23, 4, 9, 17, 3, 8]
        with self.host_runtime() as runtime:
            reference = list(runtime.eval(prompt))
            runtime.reset()

            reports = []

            def watcher(processed: int, total: int) -> bool:
                reports.append(processed)
                # Not on the first call: entering the eval is easy to abandon,
                # and abandoning it part way through the layers is the case
                # that has to leave the runtime usable.
                return len(reports) < 2

            runtime.set_progress(watcher)
            with self.assertRaises(V2Error):
                runtime.eval_into(prompt)
            runtime.set_progress(None)
            self.assertEqual(len(reports), 2)
            # A cancelled prompt leaves a partly advanced sequence behind, which
            # reset is what clears -- and after it the runtime answers exactly
            # as it did before anything was cancelled.
            runtime.reset()
            self.assertEqual(list(runtime.eval(prompt)), reference)


if __name__ == "__main__":
    unittest.main()
