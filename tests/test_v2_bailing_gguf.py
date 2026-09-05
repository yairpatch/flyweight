"""Running BailingMoE3 from a GGUF conversion rather than an HF checkpoint."""

from __future__ import annotations

import contextlib
import os
import struct
import tempfile
import unittest
from pathlib import Path

from flyweight.v2 import BailingRuntime, V2Error, V2Model
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
        previous = os.environ.get("FLYWEIGHT_HF_QUANT")
        os.environ["FLYWEIGHT_HF_QUANT"] = "F32"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("FLYWEIGHT_HF_QUANT", None)
            else:
                os.environ["FLYWEIGHT_HF_QUANT"] = previous

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
        from flyweight.v2_server import NativeV2Tokenizer

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


class BailingFlashGgufTests(unittest.TestCase):
    """The Ling 3.0 Flash shape: no query LoRA, per-layer SwiGLU clamps, MTP.

    Same weights through both loaders again -- the assertion that catches a
    mapping mistake -- but over the three things this checkpoint family does
    differently from the larger Ling models. Each is silent when dropped: an
    un-factored query reads the wrong tensor, a dropped clamp changes only the
    tails of the last layers, and a draft block treated as executable adds a
    layer that was never trained to run.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        cls.hf_path = safetensors.build(root / "flash-fixture", flash=True)
        cls.gguf_path = gguf.build(root / "converted", cls.hf_path, flash=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_the_draft_block_is_not_part_of_the_decoder_stack(self) -> None:
        with V2Model(self.gguf_path) as model:
            # block_count counts the draft block; layer_count must not.
            self.assertEqual(model.config["layer_count"], safetensors.LAYERS)
            self.assertEqual(model.config["q_lora_rank"], 0)
            names = {tensor["name"] for tensor in model.tensors()}
            self.assertIn(f"blk.{safetensors.LAYERS}.nextn.enorm.weight", names)

    def test_the_un_factored_query_answers_as_the_checkpoint_it_came_from(self) -> None:
        prompt = [5, 11, 23, 4, 9, 17, 3, 8]
        for length in (1, 2, len(prompt)):
            with self.subTest(prompt_tokens=length):
                converted = BailingGgufTests.logits(self, self.gguf_path, prompt[:length])
                with BailingGgufTests.unquantized():
                    original = BailingGgufTests.logits(self, self.hf_path, prompt[:length])
                self.assertEqual(len(converted), len(original))
                worst = max(abs(a - b) for a, b in zip(converted, original))
                scale = max(abs(value) for value in original)
                self.assertLess(worst, 1e-3 * max(scale, 1.0))
                self.assertEqual(
                    converted.index(max(converted)),
                    original.index(max(original)),
                )

    def test_the_swiglu_clamp_changes_the_answer(self) -> None:
        """A clamp that is read must be a clamp that binds.

        The parity test above cannot see this: both files carry the same
        clamps, so a runtime that ignores them on both sides still agrees with
        itself. The same weights written without the clamp metadata must give
        different logits, or the arrays are being parsed and dropped.
        """
        prompt = [5, 11, 23, 4, 9, 17, 3, 8]
        with tempfile.TemporaryDirectory() as directory:
            unclamped_path = gguf.build(
                Path(directory) / "unclamped", self.hf_path, flash=True,
                clamps=False,
            )
            unclamped = BailingGgufTests.logits(self, unclamped_path, prompt)
        clamped = BailingGgufTests.logits(self, self.gguf_path, prompt)
        worst = max(abs(a - b) for a, b in zip(clamped, unclamped))
        self.assertGreater(worst, 1e-4)


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
        previous = os.environ.get("FLYWEIGHT_BAILING_GPU")
        os.environ["FLYWEIGHT_BAILING_GPU"] = "0"
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
                os.environ.pop("FLYWEIGHT_BAILING_GPU", None)
            else:
                os.environ["FLYWEIGHT_BAILING_GPU"] = previous

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


def gpu_present() -> bool:
    try:
        return bool(V2Model.gpu_info()["available"])
    except Exception:
        return False


class BailingSlotTests(unittest.TestCase):
    """Several sequences in one runtime, running INTERLEAVED.

    Coexistence is not the property. Two sequences that each run to completion
    before the other starts will agree with their solo runs even if the slots
    share every buffer they own, because nothing is ever live at the same time.
    The property is that a token of A between two tokens of B changes neither.

    This codebase has already paid for the difference: a DeltaNet CUDA graph
    captured against one slot corrupted another (f00b4fe), and only an
    interleaved test saw it. So every case below alternates, and the two
    sequences are given DIFFERENT prompt lengths so their positions diverge --
    a shared position would otherwise be right by coincidence.

    Both paths run: the host is the oracle, and the device is where the bug
    class lives.

    One device path is out of reach here. The fixture is f32 on purpose -- it
    exists to check mapping, not quantization -- and the tiled GPU prefill only
    takes Q6_K experts, so these runs go through the host batched prefill and
    the device decode instead. The tiled path's slot threading was checked
    against a real Ling-3.0-tiny: a tiled prefill and a decode on slot 1
    between two of slot 0's tokens leaves slot 0 bit-identical. If this fixture
    ever gains quantized experts, that case starts running here for free.
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

    # Deliberately different lengths.
    PROMPT_A = [5, 11, 23, 4, 9, 17, 3, 8]
    PROMPT_B = [31, 29, 27, 25, 23, 21, 19, 15, 13, 2, 1, 6]

    @contextlib.contextmanager
    def runtime(self, on_gpu: bool, slots: int = 1, capacity: int = 64):
        previous = os.environ.get("FLYWEIGHT_BAILING_GPU")
        os.environ["FLYWEIGHT_BAILING_GPU"] = "1" if on_gpu else "0"
        try:
            with V2Model(self.gguf_path) as model:
                runtime = BailingRuntime(model, capacity=capacity, slots=slots)
                self.assertIs(runtime.uses_gpu, on_gpu)
                try:
                    yield runtime
                finally:
                    runtime.close()
        finally:
            if previous is None:
                os.environ.pop("FLYWEIGHT_BAILING_GPU", None)
            else:
                os.environ["FLYWEIGHT_BAILING_GPU"] = previous

    def paths(self):
        """The device path only when there is a device to run it on."""
        yield False
        if gpu_present():
            yield True

    def solo(self, runtime, prompt, slot: int = 0, steps: int = 6):
        """Run one sequence to completion, recording everything it produced.

        Greedy, and the tokens are recorded as well as the logits: a drifting
        cache shows up in the logits first, but a differing token is the thing
        a caller would actually see.
        """
        runtime.reset(slot)
        logits = [list(runtime.eval(prompt, slot=slot))]
        tokens = []
        for _ in range(steps):
            tokens.append(runtime.sample())
            logits.append(list(runtime.eval([tokens[-1]], slot=slot)))
        return tokens, logits

    @staticmethod
    def snapshot_position(snapshot: bytes) -> int:
        # magic[8], version, layers, kv_lora, qk_rope, heads, head_dim,
        # conv_width, position -- the last of the nine header words.
        return struct.unpack_from("<I", snapshot, 8 + 7 * 4)[0]

    def test_interleaved_sequences_answer_exactly_as_they_do_alone(self) -> None:
        for on_gpu in self.paths():
            with self.subTest(gpu=on_gpu):
                with self.runtime(on_gpu) as reference:
                    tokens_a, solo_a = self.solo(reference, self.PROMPT_A)
                    tokens_b, solo_b = self.solo(reference, self.PROMPT_B)
                # Distinct sequences, or the comparison proves nothing.
                self.assertNotEqual(tokens_a, tokens_b)

                with self.runtime(on_gpu, slots=2) as runtime:
                    runtime.reset(0)
                    runtime.reset(1)
                    # Prompts first, alternating, then a token each in turn.
                    both_a = [list(runtime.eval(self.PROMPT_A, slot=0))]
                    both_b = [list(runtime.eval(self.PROMPT_B, slot=1))]
                    for step in range(len(tokens_a)):
                        both_a.append(
                            list(runtime.eval([tokens_a[step]], slot=0)))
                        both_b.append(
                            list(runtime.eval([tokens_b[step]], slot=1)))
                        # The token each slot would decode next, checked at
                        # every step rather than only at the end: a slot that
                        # reads another's cache diverges once and then carries
                        # on consistently from the wrong place.
                        self.assertEqual(both_a[-1], solo_a[step + 1])
                        self.assertEqual(both_b[-1], solo_b[step + 1])
                    self.assertEqual(both_a, solo_a)
                    self.assertEqual(both_b, solo_b)

    def test_a_slot_nobody_touched_stays_where_it_started(self) -> None:
        for on_gpu in self.paths():
            with self.subTest(gpu=on_gpu):
                with self.runtime(on_gpu, slots=3) as runtime:
                    for slot in range(3):
                        runtime.reset(slot)
                    untouched = runtime.save_state(slot=2)
                    self.assertEqual(self.snapshot_position(untouched), 0)
                    runtime.eval_into(self.PROMPT_A, slot=0)
                    runtime.eval_into(self.PROMPT_B, slot=1)
                    for step in range(4):
                        runtime.eval_into([step + 1], slot=0)
                        runtime.eval_into([step + 2], slot=1)
                    after = runtime.save_state(slot=2)
                self.assertEqual(self.snapshot_position(after), 0)
                # Byte-identical, which covers the KDA recurrent state and the
                # convolution windows -- the parts that are the same size at
                # position zero as at any other, and so the parts a leak
                # between slots would quietly modify without moving anything.
                self.assertEqual(after, untouched)

    def test_slot_isolation_survives_a_snapshot_and_restore(self) -> None:
        for on_gpu in self.paths():
            with self.subTest(gpu=on_gpu):
                with self.runtime(on_gpu) as reference:
                    tokens_a, solo_a = self.solo(reference, self.PROMPT_A)
                    tokens_b, _ = self.solo(reference, self.PROMPT_B)

                with self.runtime(on_gpu, slots=2) as runtime:
                    runtime.reset(0)
                    runtime.reset(1)
                    runtime.eval_into(self.PROMPT_A, slot=0)
                    snapshot = runtime.save_state(slot=0)
                    self.assertEqual(
                        self.snapshot_position(snapshot), len(self.PROMPT_A))
                    # Run slot 1 far enough to overwrite anything shared, and
                    # run slot 0 somewhere else entirely, so the restore has
                    # something to undo rather than something to confirm.
                    runtime.eval_into(self.PROMPT_B, slot=1)
                    for token in tokens_b:
                        runtime.eval_into([token], slot=1)
                    for token in [40, 41, 42]:
                        runtime.eval_into([token], slot=0)
                    runtime.load_state(snapshot, slot=0)
                    # And keep slot 1 moving while slot 0 continues, so the
                    # restored slot is interleaved rather than alone.
                    restored = []
                    for step, token in enumerate(tokens_a):
                        restored.append(list(runtime.eval([token], slot=0)))
                        runtime.eval_into([tokens_b[step]], slot=1)
                self.assertEqual(restored, solo_a[1:])

    def test_a_slot_index_past_the_end_is_refused(self) -> None:
        with self.runtime(False, slots=2) as runtime:
            self.assertEqual(runtime.slot_count, 2)
            # Named rather than bare lambdas: a subtest parameter is
            # serialized back to the controller under `pytest -n`, and a
            # function is not serializable.
            for name, call in (
                ("eval", lambda: runtime.eval([5], slot=2)),
                ("eval_into", lambda: runtime.eval_into([5], slot=7)),
                ("reset", lambda: runtime.reset(2)),
                ("save_state", lambda: runtime.save_state(slot=2)),
                ("load_state", lambda: runtime.load_state(b"", slot=2)),
            ):
                with self.subTest(call=name):
                    with self.assertRaises(V2Error) as failure:
                        call()
                    # The message has to name the runtime's actual count: "out
                    # of range" alone leaves the caller guessing whether they
                    # asked for too many or built too few.
                    self.assertIn("2 slots", str(failure.exception))

    def test_one_slot_is_the_runtime_that_was_always_there(self) -> None:
        for on_gpu in self.paths():
            with self.subTest(gpu=on_gpu):
                with self.runtime(on_gpu, slots=1) as single:
                    self.assertEqual(single.slot_count, 1)
                    explicit = self.solo(single, self.PROMPT_A)
                # The pre-slots constructor call, unchanged, and the pre-slots
                # methods with no slot argument.
                previous = os.environ.get("FLYWEIGHT_BAILING_GPU")
                os.environ["FLYWEIGHT_BAILING_GPU"] = "1" if on_gpu else "0"
                try:
                    with V2Model(self.gguf_path) as model:
                        old = BailingRuntime(model, capacity=64)
                        try:
                            self.assertEqual(old.slot_count, 1)
                            old.reset()
                            logits = [list(old.eval(self.PROMPT_A))]
                            tokens = []
                            for _ in range(6):
                                tokens.append(old.sample())
                                logits.append(list(old.eval([tokens[-1]])))
                        finally:
                            old.close()
                finally:
                    if previous is None:
                        os.environ.pop("FLYWEIGHT_BAILING_GPU", None)
                    else:
                        os.environ["FLYWEIGHT_BAILING_GPU"] = previous
                self.assertEqual((tokens, logits), explicit)


if __name__ == "__main__":
    unittest.main()
