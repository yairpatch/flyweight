from __future__ import annotations

import argparse
import json
import tempfile
import threading
import unittest
from pathlib import Path
from queue import Empty, Queue

from flyweight.cli import _parser
from flyweight.sampling import SamplingConfig
from flyweight.server import InferenceService, _chat_messages
from flyweight.v2_server import (
    BailingEngine,
    NativeV2Generator,
    NativeV2InferenceService,
    NativeV2Tokenizer,
    THINKING_BUDGET_CLOSE,
    _generation_config_for_model,
    _merge_generation_defaults,
)
from flyweight.v2 import TASK_EVENT_PREFILL


class StubV2Model:
    info = {"architecture": "qwen3moe"}
    pieces = {20: "Hello", 30: " world", 99: "<|im_end|>"}
    # Raw UTF-8 bytes per token; 40/41 split one character ("⚽" = e2 9a bd)
    # across two tokens, as byte-level BPE routinely does.
    piece_bytes = {
        20: b"Hello",
        30: b" world",
        40: b"\xe2\x9a",
        41: b"\xbd",
        99: b"<|im_end|>",
    }

    def tokenize(self, text: str) -> list[int]:
        return [1, 2]

    def decode_tokens(self, tokens: list[int]) -> str:
        return "".join(self.pieces.get(token, "") for token in tokens)

    def decode_token_bytes(self, tokens: list[int]) -> bytes:
        return b"".join(self.piece_bytes.get(token, b"") for token in tokens)

    def token_id(self, text: str) -> int:
        if text in ("<|im_end|>", "<|endoftext|>"):
            return 99
        raise KeyError(text)


class StubV2Runtime:
    def __init__(self, outputs: list[int]):
        self.outputs = iter(outputs)
        self.inputs: list[int] = []
        self.resets = 0
        self.cancels = 0

    @property
    def info(self) -> dict[str, int]:
        return {
            "position": len(self.inputs),
            "prefix_cache_hits": 3,
            "prefix_cache_misses": 1,
            "prefix_cache_reused_tokens": 42,
        }

    def reset(self) -> None:
        self.resets += 1

    def decode(self, token: int) -> int:
        self.inputs.append(token)
        return next(self.outputs)

    def generate(self, prompt: list[int], max_tokens: int, callback) -> None:
        self.reset()
        next_token = 0
        for token in prompt:
            next_token = self.decode(token)
        for index in range(max_tokens):
            if callback(next_token) is False:
                break
            if index + 1 < max_tokens:
                next_token = self.decode(next_token)

    def cancel(self) -> None:
        self.cancels += 1


    # Cooperative-engine API mirroring the native semantics (prefill the whole
    # prompt, then per step: emit token; stop on stop-token or max; else decode
    # the emitted token as the next input).
    def task_submit(
        self,
        prompt,
        max_tokens,
        stop_tokens=(),
        *,
        temperature=0.0,
        top_k=20,
        top_p=0.95,
        seed=None,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        penalty_window=64,
        tools=None,
        response_format=None,
        forbid_tool_calls=False,
    ):
        self._tasks = getattr(self, "_tasks", {})
        self._next_task = getattr(self, "_next_task", 0) + 1
        self._last_sampling = (temperature, top_k, top_p, seed)
        self._last_penalties = (
            repetition_penalty, presence_penalty, frequency_penalty, penalty_window
        )
        self._last_tools = tools
        self._last_response_format = response_format
        self._last_forbid_tool_calls = forbid_tool_calls
        self._tasks[self._next_task] = (list(prompt), max_tokens, tuple(stop_tokens))
        return self._next_task

    def engine_step(self, capacity: int = 256):
        events = []
        for task_id, (prompt, max_tokens, stops) in list(
            getattr(self, "_tasks", {}).items()
        ):
            self.reset()
            next_token = 0
            for token in prompt:
                next_token = self.decode(token)
            emitted = 0
            while True:
                events.append((task_id, next_token, 0))
                emitted += 1
                if next_token in stops or emitted >= max_tokens:
                    break
                next_token = self.decode(next_token)
            events.append((task_id, 0, 1))
            del self._tasks[task_id]
        return events

    def task_cancel(self, task_id: int) -> None:
        self.cancels += 1
        getattr(self, "_tasks", {}).pop(task_id, None)


class StubBailingRuntime:
    """The runtime the engine talks to, with as many slots as asked for.

    Positions are per slot, and `position` reports slot 0's, so the
    single-slot tests read exactly as they did before slots existed.
    """

    def __init__(self, slots: int = 1) -> None:
        self.resets = 0
        self.eval_calls: list[list[int]] = []
        self.next_token = 20
        self.uses_gpu = False
        self.slot_count = slots
        self.positions = [0] * slots
        self.saves = 0
        self.loads: list[bytes] = []
        self.progress = None
        # Which slot each evaluation landed on, for the interleaving tests.
        self.eval_slots: list[int] = []

    @property
    def position(self) -> int:
        return self.positions[0]

    @position.setter
    def position(self, value: int) -> None:
        self.positions[0] = value

    def reset(self, slot: int = 0) -> None:
        self.resets += 1
        self.positions[slot] = 0

    def set_progress(self, callback) -> None:
        self.progress = callback

    def eval_into(self, tokens, slot: int = 0) -> None:
        tokens = list(tokens)
        # The runtime reports per tile while a prompt runs and stops when the
        # watcher says so; a tile here is two tokens, so tests stay small.
        if self.progress is not None and len(tokens) > 1:
            for offset in range(0, len(tokens), 2):
                if self.progress(offset, len(tokens)) is False:
                    self.positions[slot] += offset
                    raise RuntimeError("bailing prompt evaluation was cancelled")
        self.eval_calls.append(tokens)
        self.eval_slots.append(slot)
        self.positions[slot] += len(tokens)

    def save_state(self, slot: int = 0) -> bytes:
        self.saves += 1
        return f"state@{self.positions[slot]}".encode()

    def load_state(self, snapshot: bytes, slot: int = 0) -> None:
        self.loads.append(snapshot)
        # Trailing padding lets a subclass give a snapshot a realistic size.
        self.positions[slot] = int(snapshot.decode().split("@")[1].rstrip("."))

    def sample(self, _sampling) -> int:
        self.next_token += 1
        return self.next_token


class BlockingV2Runtime(StubV2Runtime):
    def __init__(self):
        super().__init__([10])
        self.entered = threading.Event()
        self.cancelled = threading.Event()

    def engine_step(self, capacity: int = 256):
        self.entered.set()
        self.cancelled.wait(2)
        return []

    def task_cancel(self, task_id: int) -> None:
        super().task_cancel(task_id)
        self.cancelled.set()


class ProgressV2Runtime(StubV2Runtime):
    def engine_step(self, capacity: int = 256):
        events = []
        for task_id, (prompt, _max_tokens, _stops) in list(self._tasks.items()):
            events.extend(
                (task_id, processed, TASK_EVENT_PREFILL)
                for processed in (0, 1, len(prompt))
            )
            events.extend(((task_id, 20, 0), (task_id, 0, 1)))
            del self._tasks[task_id]
        return events


class NativeV2ServerTests(unittest.TestCase):
    def make_generator(self, outputs: list[int]):
        model = StubV2Model()
        runtime = StubV2Runtime(outputs)
        tokenizer = NativeV2Tokenizer(model)  # type: ignore[arg-type]
        generator = NativeV2Generator(  # type: ignore[arg-type]
            model, runtime, tokenizer
        )
        return generator, runtime

    def test_bailing_engine_reuses_exact_live_prefix_and_reports_progress(self) -> None:
        runtime = StubBailingRuntime()
        engine = BailingEngine(runtime)  # type: ignore[arg-type]

        def run(prompt: list[int]) -> list[tuple[str, object]]:
            task_id, events = engine.submit(prompt, 1, ())
            received: list[tuple[str, object]] = []
            while True:
                event = events.get(timeout=2)
                received.append(event)
                if event[0] == "done":
                    break
            engine.forget(task_id)
            return received

        try:
            first = run([1, 2])
            second = run([1, 2, 21, 3])
        finally:
            engine.close()

        self.assertEqual(runtime.resets, 1)
        # A prompt is evaluated as (all but its last token, then that token) so
        # the snapshot point sits one token short of the end; see _run_task.
        self.assertEqual(runtime.eval_calls, [[1], [2], [21], [3], [22]])
        self.assertEqual([value for kind, value in first if kind == "prefill"], [0, 2])
        self.assertEqual([value for kind, value in second if kind == "prefill"], [3, 4])

    def test_bailing_engine_survives_a_side_call_between_two_turns(self) -> None:
        # What a coding harness actually sends: a long conversation, a short
        # unrelated call (title/summary/second agent), then the next turn of the
        # long one. With one set of caches and no snapshots the side-call left
        # nothing behind and that next turn re-prefilled the whole prompt.
        runtime = StubBailingRuntime()
        engine = BailingEngine(runtime)  # type: ignore[arg-type]

        def run(prompt: list[int]) -> list[int]:
            task_id, events = engine.submit(prompt, 1, ())
            prefills: list[int] = []
            while True:
                kind, value = events.get(timeout=2)
                if kind == "prefill":
                    prefills.append(int(value))  # type: ignore[arg-type]
                if kind == "done":
                    break
            engine.forget(task_id)
            return prefills

        conversation = [1, 2, 3, 4, 5]
        try:
            run(conversation)
            run([90, 91])                       # the side-call
            # The next turn: the conversation, our reply (21), and a new message.
            resumed = run(conversation + [21, 6, 7])
        finally:
            engine.close()

        # Only the two new tokens are evaluated; the conversation and the reply
        # it ended on come back from the snapshot the side-call displaced.
        self.assertEqual(runtime.eval_calls[-3:], [[6], [7], [23]])
        self.assertEqual(resumed, [6, 8])
        self.assertEqual(engine.reused_tokens, 6)
        self.assertTrue(runtime.loads, "the displaced sequence was never restored")

    def test_bailing_engine_reports_a_long_prompt_as_it_runs(self) -> None:
        # A prompt is one call that can run for minutes. Without per-tile
        # reporting the log went quiet between "starting" and "done", which is
        # the whole time anyone wants to know what is happening.
        runtime = StubBailingRuntime()
        engine = BailingEngine(runtime)  # type: ignore[arg-type]
        try:
            task_id, events = engine.submit([1, 2, 3, 4, 5, 6, 7], 1, ())
            prefills: list[int] = []
            while True:
                kind, value = events.get(timeout=2)
                if kind == "prefill":
                    prefills.append(int(value))  # type: ignore[arg-type]
                if kind == "done":
                    break
            engine.forget(task_id)
        finally:
            engine.close()

        # Movement through the prompt, not just its two ends.
        self.assertEqual(prefills, [0, 0, 2, 4, 7])

    def test_bailing_engine_abandons_the_prompt_of_a_cancelled_task(self) -> None:
        # Generation stopped promptly on cancel; the prompt did not, so an
        # abandoned request still cost its full prefill -- and on this runtime
        # that also blocks every request behind it.
        runtime = StubBailingRuntime()
        engine = BailingEngine(runtime)  # type: ignore[arg-type]
        try:
            task_id, events = engine.submit(list(range(40)), 1, ())
            engine.cancel(task_id)
            with self.assertRaises(Empty):
                while True:
                    if events.get(timeout=0.5)[0] == "done":
                        break
        finally:
            engine.close()

        # It stopped inside the prompt rather than evaluating all of it.
        self.assertLess(runtime.position, 40)
        self.assertEqual(runtime.eval_calls, [])
        # And what the runtime holds is no longer known, so it is forgotten.
        self.assertFalse(engine._slot_initialized[0])

    def test_bailing_engine_interleaves_requests_across_slots(self) -> None:
        # The point of slots. With one, a request submitted behind a long
        # generation waited out all of it: a 50-token answer sat behind a
        # 4000-token one. Two slots must make progress on both at once.
        class GatedRuntime(StubBailingRuntime):
            """Holds its very first evaluation until the test says go.

            Without it the stub finishes the long generation microseconds
            after submit, before the second request is even queued, and the
            test proves nothing about interleaving.
            """

            def __init__(self) -> None:
                super().__init__(slots=2)
                self.reached = threading.Event()
                self.gate = threading.Event()

            def eval_into(self, tokens, slot: int = 0) -> None:
                if not self.reached.is_set():
                    self.reached.set()
                    self.gate.wait(2)
                super().eval_into(tokens, slot)

        runtime = GatedRuntime()
        engine = BailingEngine(runtime)
        self.assertEqual(engine._slot_count, 2)
        try:
            long_id, long_events = engine.submit([1, 2], 40, ())
            # The long request is now stopped inside its prompt, so the second
            # one genuinely arrives behind it rather than racing it.
            self.assertTrue(runtime.reached.wait(2))
            short_id, short_events = engine.submit([5, 6], 2, ())
            runtime.gate.set()

            short_tokens = []
            while True:
                kind, value = short_events.get(timeout=2)
                if kind == "token":
                    short_tokens.append(value)
                if kind == "done":
                    break
            self.assertEqual(len(short_tokens), 2)
            while long_events.get(timeout=2)[0] != "done":
                pass
        finally:
            engine.forget(long_id)
            engine.forget(short_id)
            engine.close()

        # Each request stayed on its own slot rather than trampling the other.
        order = runtime.eval_slots
        self.assertEqual(set(order), {0, 1})
        # And they alternated: the long generation kept going after the short
        # one started. Serial execution would put every slot-0 evaluation
        # before every slot-1 one, whichever ran first.
        self.assertIn(0, order[order.index(1):],
                      "the long request stalled until the short one finished")

    def test_bailing_engine_gives_a_conversation_back_its_own_slot(self) -> None:
        # Affinity, not round-robin. The second turn of a conversation extends
        # the sequence its slot is still holding; handing it the other slot
        # would throw that away and re-prefill from a snapshot at best.
        runtime = StubBailingRuntime(slots=2)
        engine = BailingEngine(runtime)

        def run(prompt: list[int]) -> None:
            task_id, events = engine.submit(prompt, 1, ())
            while events.get(timeout=2)[0] != "done":
                pass
            engine.forget(task_id)

        try:
            run([1, 2])          # lands on a slot, leaves [1, 2, 21] live
            run([5, 6])          # a different conversation, the other slot
            runtime.eval_slots.clear()
            runtime.eval_calls.clear()
            run([1, 2, 21, 3])   # continues the first
        finally:
            engine.close()

        # Only the last token of the continuation had to be evaluated, on the
        # slot that already held the rest.
        self.assertEqual(runtime.eval_calls, [[3], [23]])
        self.assertEqual(set(runtime.eval_slots), {0})

    def test_bailing_engine_keeps_the_snapshot_it_is_about_to_restore(self) -> None:
        # Observed against a real coding-harness session: a 37,810-token turn,
        # then a 37,830-token one that shared all but its tail, and the log said
        # "0 reused" -- 267 seconds of prefill that the cache was holding.
        #
        # Putting the live sequence aside is what evicted it. The two are nearly
        # the same size, so saving the second pushed the store over budget and
        # the LRU end -- the older snapshot, the one this very request was about
        # to be restored from -- was what went.
        class SizedRuntime(StubBailingRuntime):
            """Snapshots that cost in proportion to the tokens they hold."""

            def save_state(self, slot: int = 0) -> bytes:
                return super().save_state(slot).ljust(self.positions[slot] * 10, b".")

        runtime = SizedRuntime()
        # Room for one sequence of this length, not two.
        engine = BailingEngine(runtime, 120)
        engine._SNAPSHOT_PREFILL_THRESHOLD = 1

        def run(prompt: list[int]) -> None:
            task_id, events = engine.submit(prompt, 1, ())
            while events.get(timeout=2)[0] != "done":
                pass
            engine.forget(task_id)

        conversation = list(range(1, 11))
        try:
            run(conversation)
            # The next turn shares the conversation but not the reply the
            # engine generated, which is what a harness sending its own
            # rendering of that turn back produces.
            runtime.eval_calls.clear()
            run(conversation + [77, 78])
        finally:
            engine.close()

        self.assertEqual(engine.reused_tokens, 9)
        self.assertEqual(runtime.eval_calls, [[10, 77], [78], [22]])

    def test_bailing_engine_forgets_the_live_sequence_after_a_failed_eval(
        self,
    ) -> None:
        # Evaluation advances the runtime token by token and the token list is
        # only extended once the call returns, so a throw partway through leaves
        # the two disagreeing. Snapshotting that state would hand it back later
        # as a prefix it is not -- the wrong context, with no error to show for
        # it -- so the live sequence has to be forgotten instead.
        class FailingRuntime(StubBailingRuntime):
            def eval_into(self, tokens, slot: int = 0) -> None:
                if list(tokens) == [7]:
                    self.positions[slot] += 1  # it advanced before it failed
                    raise RuntimeError("device fell over")
                super().eval_into(tokens, slot)

        runtime = FailingRuntime()
        engine = BailingEngine(runtime)  # type: ignore[arg-type]

        def run(prompt: list[int]) -> str:
            task_id, events = engine.submit(prompt, 1, ())
            while True:
                kind, _ = events.get(timeout=2)
                if kind in ("done", "error"):
                    engine.forget(task_id)
                    return kind

        try:
            self.assertEqual(run([1, 2, 7]), "error")
            self.assertEqual(engine._slot_tokens[0], [])
            self.assertFalse(engine._snapshots)
            runtime.eval_calls.clear()
            run([1, 2, 3])
        finally:
            engine.close()

        # The next prompt starts from a reset, not from a cache whose contents
        # nobody can name.
        self.assertEqual(runtime.resets, 2)
        self.assertEqual(runtime.eval_calls[:2], [[1, 2], [3]])

    def test_bailing_engine_takes_its_snapshot_budget_from_the_caller(self) -> None:
        runtime = StubBailingRuntime()
        engine = BailingEngine(runtime, 0)  # --cache off
        engine._SNAPSHOT_PREFILL_THRESHOLD = 1

        def run(prompt: list[int]) -> None:
            task_id, events = engine.submit(prompt, 1, ())
            while events.get(timeout=2)[0] != "done":
                pass
            engine.forget(task_id)

        try:
            run([1, 2, 3])
            run([9, 9])
        finally:
            engine.close()

        self.assertEqual(runtime.saves, 0)
        self.assertEqual(engine.cache_stats()["used"], 0)

    def test_bailing_engine_leaves_a_token_to_evaluate_when_a_prompt_repeats(
        self,
    ) -> None:
        # A prompt that is exactly a known sequence must still evaluate its last
        # token: the sampler reads the logits of the last token evaluated, so a
        # fully restored prompt would sample from whatever ran before it.
        runtime = StubBailingRuntime()
        engine = BailingEngine(runtime)  # type: ignore[arg-type]
        engine._SNAPSHOT_PREFILL_THRESHOLD = 2

        def run(prompt: list[int]) -> None:
            task_id, events = engine.submit(prompt, 1, ())
            while events.get(timeout=2)[0] != "done":
                pass
            engine.forget(task_id)

        try:
            run([1, 2, 3])
            runtime.eval_calls.clear()
            run([1, 2, 3])
        finally:
            engine.close()

        self.assertEqual(runtime.eval_calls[0], [3])

    def test_generation_config_adjacent_to_model_supplies_sampling_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            model.touch()
            config = root / "generation_config.json"
            config.write_text(
                json.dumps(
                    {
                        "do_sample": True,
                        "temperature": 0.6,
                        "top_k": 20,
                        "top_p": 0.9,
                        "max_new_tokens": 512,
                    }
                ),
                encoding="utf-8",
            )

            defaults, source = _generation_config_for_model(model)

        self.assertEqual(
            defaults,
            {"temperature": 0.6, "top_k": 20, "top_p": 0.9, "max_new_tokens": 512},
        )
        self.assertEqual(source, str(config))

    def test_generation_config_disables_sampling_when_do_sample_is_false(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            model.touch()
            config = root / "model.generation_config.json"
            config.write_text(
                json.dumps({"do_sample": False, "temperature": 0.7}),
                encoding="utf-8",
            )

            defaults, _ = _generation_config_for_model(model)

        self.assertEqual(defaults["temperature"], 0.0)

    def test_hf_directory_uses_its_own_generation_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "checkpoint"
            model.mkdir()
            config = model / "generation_config.json"
            config.write_text(json.dumps({"temperature": 0.4, "top_p": 0.8}))

            defaults, source = _generation_config_for_model(model)

        self.assertEqual(defaults, {"temperature": 0.4, "top_p": 0.8})
        self.assertEqual(source, str(config))

    def test_every_sampling_setting_is_reachable_from_every_surface(self) -> None:
        # The point of sampling.SETTINGS: a setting the sampler honours must be
        # settable by a server flag, by the checkpoint's generation config, and
        # by a request. These lists were written out by hand once each and
        # drifted -- `seed` reached only OpenAI requests, the penalties only a
        # flag or an OpenAI request -- so each surface is checked against the
        # table rather than against a second hand-written list.
        from flyweight.sampling import SERVER_SETTINGS, SETTINGS

        parser = _parser()
        serve = next(
            action.choices["serve"]
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        flags = {option for action in serve._actions for option in action.option_strings}
        for setting in SERVER_SETTINGS:
            self.assertIn(f"--{setting.name.replace('_', '-')}", flags)

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.write_bytes(b"")
            values = {
                "temperature": 0.3, "top_k": 7, "top_p": 0.5,
                "repetition_penalty": 1.2, "presence_penalty": 0.25,
                "frequency_penalty": -0.5, "penalty_window": 16,
            }
            (model.parent / "generation_config.json").write_text(json.dumps(values))
            defaults, _ = _generation_config_for_model(model)
        for setting in SERVER_SETTINGS:
            self.assertEqual(defaults[setting.name], values[setting.name],
                             f"{setting.name} was dropped by the config loader")

        # And a request may set every one of them, including seed.
        from flyweight.server import _sampling_from_payload
        from flyweight.sampling import defaults as builtin_defaults
        payload = dict(values, seed=99)
        sampling = _sampling_from_payload(payload, builtin_defaults())
        for setting in SETTINGS:
            self.assertEqual(getattr(sampling, setting.name), payload[setting.name],
                             f"{setting.name} was dropped by the request parser")

    def test_server_flags_win_over_the_checkpoints_generation_config(self) -> None:
        # The file says what the model shipped with; the flag says what this
        # server was told to do. Losing that order would mean an operator who
        # sets --repetition-penalty on the command line silently gets the
        # checkpoint's value instead.
        defaults, source = _merge_generation_defaults(
            {"temperature": 0.4, "top_p": 0.8}, "/models/gen.json",
            {"temperature": 0.0, "repetition_penalty": 1.15},
        )
        self.assertEqual(
            defaults,
            {"temperature": 0.0, "top_p": 0.8, "repetition_penalty": 1.15},
        )
        self.assertEqual(source, "/models/gen.json+flags(repetition_penalty,temperature)")

    def test_an_unset_flag_leaves_the_checkpoints_value_alone(self) -> None:
        # The CLI passes only what it was given: an absent flag must not arrive
        # here as an argparse default and overwrite the file.
        defaults, source = _merge_generation_defaults(
            {"temperature": 0.4}, "/models/gen.json", {}
        )
        self.assertEqual(defaults, {"temperature": 0.4})
        self.assertEqual(source, "/models/gen.json")
        # None is treated the same way, so a caller may pass a full dict of
        # optional values without filtering it first.
        defaults, source = _merge_generation_defaults(
            {"temperature": 0.4}, "engine", {"top_p": None, "penalty_window": 0}
        )
        self.assertEqual(defaults, {"temperature": 0.4, "penalty_window": 0})
        self.assertEqual(source, "engine+flags(penalty_window)")

    def test_health_exposes_resolved_expert_policy(self) -> None:
        class Runtime:
            info = {
                "expert_mode": "cpu",
                "requested_expert_mode": "auto",
                "expert_fallback_reason": "working set did not fit",
                "moe_device": 1,
                "position": 0,
                "prefix_cache_hits": 0,
                "prefix_cache_misses": 0,
                "prefix_cache_reused_tokens": 0,
            }

        service = object.__new__(NativeV2InferenceService)
        InferenceService.__init__(
            service,
            "native-test",
            self.make_generator([20])[0],
            context_window=128,
        )
        service.v2_runtime = Runtime()
        service.expert_mode = "cpu"
        service.requested_expert_mode = "auto"
        service.expert_fallback_reason = "working set did not fit"
        service.moe_device = "cpu"
        service.mtp_drafts = 0
        service.gpu_cache_mib = 0
        execution = service.health()["execution"]
        self.assertEqual(execution["expert_mode"], "cpu")
        self.assertEqual(execution["requested_expert_mode"], "auto")
        self.assertEqual(
            execution["expert_fallback_reason"], "working set did not fit"
        )

    def test_health_reports_the_device_the_bailing_runtime_actually_got(self) -> None:
        # The runtime picks the GPU whenever there is one and falls back to the
        # host silently, so the environment cannot be used to answer this: a
        # plain GPU run reported "cpu" while every token ran on the device.
        for uses_gpu, expected in ((True, "gpu"), (False, "cpu")):
            with self.subTest(uses_gpu=uses_gpu):
                runtime = StubBailingRuntime()
                runtime.uses_gpu = uses_gpu
                service = object.__new__(NativeV2InferenceService)
                InferenceService.__init__(
                    service,
                    "native-test",
                    self.make_generator([20])[0],
                    context_window=128,
                )
                service.architecture = "bailingmoe3"
                service.bailing_runtime = runtime  # type: ignore[assignment]
                execution = service.health()["execution"]
                self.assertEqual(execution["device"], expected)
                self.assertEqual(execution["backend"], "native-v2-bailingmoe3")

    def test_native_generator_prefills_and_greedily_decodes(self) -> None:
        generator, runtime = self.make_generator([10, 20, 30])
        result = generator.generate_messages(
            [{"role": "user", "content": "Hi"}],
            max_new_tokens=2,
            sampling=SamplingConfig(temperature=0),
        )
        self.assertEqual(result.generated_ids, (20, 30))
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(runtime.inputs, [1, 2, 20])
        self.assertEqual(runtime.resets, 1)

    def test_native_generator_reports_live_prefill_progress(self) -> None:
        model = StubV2Model()
        runtime = ProgressV2Runtime([20])
        generator = NativeV2Generator(  # type: ignore[arg-type]
            model, runtime, NativeV2Tokenizer(model)  # type: ignore[arg-type]
        )
        progress: list[tuple[int, int]] = []

        result = generator.generate_text(
            "Hi", max_new_tokens=1, progress=lambda done, total: progress.append(
                (done, total)
            )
        )

        self.assertEqual(result.generated_ids, (20,))
        self.assertEqual(progress, [(0, 2), (1, 2), (2, 2)])

    def test_stream_steps_use_stable_constant_time_token_snapshots(self) -> None:
        generator, _ = self.make_generator([10, 20, 30])
        steps = list(generator.stream_text("Hi", max_new_tokens=2))
        live = [step for step in steps if not step.finished]
        self.assertEqual(tuple(live[0].generated_ids), (20,))
        self.assertEqual(tuple(live[1].generated_ids), (20, 30))
        self.assertEqual(live[0].text, "")
        self.assertEqual(steps[-1].generated_ids, (20, 30))
        self.assertEqual(steps[-1].text, "Hello world")

    def test_a_dropped_task_fails_the_stream_instead_of_hanging(self) -> None:
        # An engine that loses a task's terminal event used to leave the
        # consumer in an untimed queue.get() forever, and the SSE layer above it
        # kept writing keepalives -- so the client sat in a "working" state with
        # no output, no error and no end. The wait is now checked against the
        # engine's own view of the task.
        generator, _ = self.make_generator([20])
        generator._STALL_POLL_SECONDS = 0.05

        class DroppingEngine:
            """Accepts a task, emits nothing, and forgets it."""

            def submit(self, *args, **kwargs):
                return 7, Queue()

            def task_is_live(self, task_id: int) -> bool:
                return False

            def active_task_count(self) -> int:
                return 0

            def cancel(self, task_id: int) -> None:
                pass

            def forget(self, task_id: int) -> None:
                pass

        generator.engine = DroppingEngine()
        with self.assertRaises(RuntimeError) as caught:
            list(generator.stream_text("Hi", max_new_tokens=8))
        self.assertIn("stopped scheduling", str(caught.exception))

    def test_a_task_still_queued_for_a_slot_keeps_waiting(self) -> None:
        # The counterpart: silence from a task the engine still holds is a
        # request queued behind another, which must not be failed.
        generator, _ = self.make_generator([20])
        generator._STALL_POLL_SECONDS = 0.02
        events: Queue = Queue()

        class SlowEngine:
            """Live throughout, but says nothing until released."""

            def __init__(self) -> None:
                self.polls = 0

            def submit(self, *args, **kwargs):
                return 7, events

            def task_is_live(self, task_id: int) -> bool:
                self.polls += 1
                if self.polls == 3:  # a slot frees on the third check
                    events.put(("token", 20))
                    events.put(("done", None))
                return True

            def active_task_count(self) -> int:
                return 2

            def cancel(self, task_id: int) -> None:
                pass

            def forget(self, task_id: int) -> None:
                pass

        engine = SlowEngine()
        generator.engine = engine
        steps = list(generator.stream_text("Hi", max_new_tokens=8))
        self.assertGreaterEqual(engine.polls, 3)
        self.assertEqual(steps[-1].generated_ids, (20,))
        self.assertEqual(steps[-1].text, "Hello")

    def test_bailing_progress_survives_a_library_without_the_entry_point(
        self,
    ) -> None:
        # flyweight_v2_bailing_set_progress shipped in the bindings before any
        # native implementation existed, and the unguarded lookup failed
        # every BailingMoE3 generation with an undefined-symbol stream error.
        # The guard restores the pre-progress behaviour instead: no reports,
        # an uninterruptible prompt, and a working answer.
        from flyweight.v2 import BailingRuntime

        class SymbolFreeLibrary:
            def __getattr__(self, name):
                raise AttributeError(name)

        runtime = object.__new__(BailingRuntime)
        runtime._lib = SymbolFreeLibrary()
        runtime.set_progress(lambda processed, total: True)
        self.assertIsNone(runtime._progress)
        runtime.set_progress(None)

    def _budget_generator(self):
        """A generator over a scripted engine, for thinking-budget tests.

        The stub model gains think markers and encodes the forced-close text
        to a single `</think>` token so the splice is visible in token IDs.
        """

        class ThinkModel(StubV2Model):
            pieces = {
                **StubV2Model.pieces,
                50: "<think>", 51: "mull", 52: "</think>", 60: "done",
            }
            piece_bytes = {
                **StubV2Model.piece_bytes,
                50: b"<think>", 51: b"mull", 52: b"</think>", 60: b"done",
            }

            def tokenize(self, text: str) -> list[int]:
                return [52] if text == THINKING_BUDGET_CLOSE else [1, 2]

        class ScriptedEngine:
            """Serves one queue of events per submit, recording each."""

            def __init__(self, scripts: list[list[tuple[str, object]]]):
                self.scripts = scripts
                self.submits: list[tuple[list[int], int, object]] = []
                self.cancelled: list[int] = []

            def submit(
                self, prompt, max_new_tokens, stop_tokens, sampling,
                tools=None, response_format=None, forbid_tool_calls=False,
            ):
                self.submits.append(
                    (list(prompt), max_new_tokens, response_format)
                )
                events: Queue = Queue()
                for event in self.scripts[len(self.submits) - 1]:
                    events.put(event)
                return len(self.submits), events

            def cancel(self, task_id: int) -> None:
                self.cancelled.append(task_id)

            def forget(self, task_id: int) -> None:
                pass

        model = ThinkModel()
        tokenizer = NativeV2Tokenizer(model)  # type: ignore[arg-type]
        generator = NativeV2Generator(  # type: ignore[arg-type]
            model, StubV2Runtime([]), tokenizer
        )
        return generator, ScriptedEngine

    def test_reasoning_budget_forces_the_thinking_block_closed(self) -> None:
        # The model opens a think block and would mull forever; the budget
        # counts tokens from the opening marker on, cancels the task at the
        # limit, splices the forced close, and resumes the answer on a fresh
        # task whose prompt carries everything decoded so far.
        generator, ScriptedEngine = self._budget_generator()
        engine = ScriptedEngine([
            [("token", 50)] + [("token", 51)] * 10,
            [("token", 60), ("token", 99), ("done", None)],
        ])
        generator.engine = engine

        steps = list(generator.stream_text(
            "Hi", max_new_tokens=8, reasoning_budget_tokens=3,
            thinking_open=False,
        ))

        final = steps[-1]
        # The opening marker's own token is the first spent; two "mull"
        # tokens exhaust the budget of 3, and 52 is the forced close.
        self.assertEqual(tuple(final.generated_ids), (50, 51, 51, 52, 60, 99))
        self.assertEqual(final.text, "<think>mullmull</think>done")
        self.assertTrue(final.stopped_on_eos)
        self.assertEqual(engine.cancelled, [1])
        resumed_prompt, resumed_max, _ = engine.submits[1]
        self.assertEqual(resumed_prompt, [1, 2, 50, 51, 51, 52])
        self.assertEqual(resumed_max, 4)

    def test_reasoning_budget_meters_a_prompt_opened_block(self) -> None:
        # thinking_open says the prompt already ended with <think>, so every
        # generated token counts from the first one on.
        generator, ScriptedEngine = self._budget_generator()
        engine = ScriptedEngine([
            [("token", 51)] * 10,
            [("token", 60), ("token", 99), ("done", None)],
        ])
        generator.engine = engine

        steps = list(generator.stream_text(
            "Hi", max_new_tokens=8, reasoning_budget_tokens=2,
            thinking_open=True,
        ))

        self.assertEqual(
            tuple(steps[-1].generated_ids), (51, 51, 52, 60, 99)
        )
        self.assertEqual(steps[-1].text, "mullmull</think>done")

    def test_a_closed_block_spends_no_further_budget(self) -> None:
        # A model that finishes thinking within the budget is untouched:
        # answer tokens after </think> are free, no matter how many.
        generator, ScriptedEngine = self._budget_generator()
        engine = ScriptedEngine([
            [
                ("token", 50), ("token", 51), ("token", 52),
                ("token", 60), ("token", 60), ("token", 60),
                ("token", 99), ("done", None),
            ],
        ])
        generator.engine = engine

        steps = list(generator.stream_text(
            "Hi", max_new_tokens=16, reasoning_budget_tokens=3,
            thinking_open=False,
        ))

        self.assertEqual(engine.cancelled, [])
        self.assertEqual(len(engine.submits), 1)
        self.assertEqual(
            steps[-1].text, "<think>mull</think>donedonedone"
        )

    def test_gemma4_chat_format_uses_turn_and_channel_tokens(self) -> None:
        tokenizer = object.__new__(NativeV2Tokenizer)
        tokenizer.architecture = "gemma4"
        prompt = tokenizer.format_messages(
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
            ]
        )
        self.assertEqual(
            prompt,
            "<bos><|turn>system\nBe concise.<turn|>\n"
            "<|turn>user\nHello<turn|>\n"
            "<|turn>model\n<|channel>thought\n<channel|>",
        )

    def test_qwen_chat_format_controls_thinking_mode(self) -> None:
        tokenizer = object.__new__(NativeV2Tokenizer)
        tokenizer.architecture = "qwen3moe"
        messages = [{"role": "user", "content": "Explain MoE simply."}]

        thinking = tokenizer.format_messages(messages, enable_thinking=True)
        direct = tokenizer.format_messages(messages, enable_thinking=False)

        self.assertTrue(thinking.endswith("<|im_start|>assistant\n<think>\n"))
        self.assertTrue(
            direct.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        )

    def test_gguf_chat_template_takes_precedence_over_architecture_fallback(self) -> None:
        class TemplateModel(StubV2Model):
            config = {}
            chat_template = (
                "{% for message in messages %}[{{ message.role }}]"
                "{{ message.content }}{% endfor %}"
                "{% if add_generation_prompt %}{% generation %}[assistant]"
                "{% if enable_thinking %}<think>{% endif %}"
                "{% endgeneration %}{% endif %}"
            )

        tokenizer = NativeV2Tokenizer(TemplateModel())  # type: ignore[arg-type]
        prompt = tokenizer.format_messages(
            [{"role": "user", "content": "Hello"}], enable_thinking=True
        )

        self.assertEqual(prompt, "[user]Hello[assistant]<think>")
        self.assertEqual(tokenizer.chat_template_source, "gguf")

    def test_gguf_chat_template_can_access_missing_tool_calls(self) -> None:
        class TemplateModel(StubV2Model):
            config = {}
            chat_template = (
                "{% for message in messages %}[{{ message.role }}]"
                "{{ message.content }}"
                "{% if message.tool_calls %}[tools]{% endif %}{% endfor %}"
                "{% if add_generation_prompt %}[assistant]{% endif %}"
            )

        tokenizer = NativeV2Tokenizer(TemplateModel())  # type: ignore[arg-type]

        prompt = tokenizer.format_messages(
            [{"role": "user", "content": "Hello"}]
        )

        self.assertEqual(prompt, "[user]Hello[assistant]")

    def test_gguf_chat_template_receives_tools_and_calls(self) -> None:
        # A template that renders its own tool section reads the schemas from
        # the top-level `tools`, and a previous call from `message.tool_calls`.
        # Both were dropped on the way in: the compatibility layer attaches them
        # to a message, and the generator reduced every message to role and
        # content before rendering -- so a tool-capable model was prompted as
        # though no tools existed and simply refused the request.
        class TemplateModel(StubV2Model):
            config = {}
            chat_template = (
                "{% if tools %}[schemas]{% for t in tools %}"
                "{{ t.function.name }}{% endfor %}{% endif %}"
                "{% for message in messages %}[{{ message.role }}]"
                "{{ message.content }}"
                "{% for call in message.tool_calls %}"
                "[call]{{ call.function.name }}"
                "{% for k, v in call.function.arguments.items() %}"
                "({{ k }}={{ v }}){% endfor %}"
                "{% endfor %}{% endfor %}"
                "{% if add_generation_prompt %}[assistant]{% endif %}"
            )

        tokenizer = NativeV2Tokenizer(TemplateModel())  # type: ignore[arg-type]
        prompt = tokenizer.format_messages([
            {
                "role": "user",
                "content": "Read it",
                "tools": [{"type": "function", "function": {"name": "read_file"}}],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "/etc"}},
                }],
            },
            {"role": "tool", "content": "ok"},
        ])

        self.assertEqual(
            prompt,
            "[schemas]read_file[user]Read it[assistant]"
            "[call]read_file(path=/etc)[tool]ok[assistant]",
        )

    def test_gguf_chat_template_tojson_takes_jinja_keywords(self) -> None:
        # BailingMoE3 renders a non-string tool argument with
        # tojson(ensure_ascii=False). Jinja's filter accepts that keyword and
        # ours did not, so the first tool call carrying a number or an object
        # -- rather than a string -- failed to render at all.
        class TemplateModel(StubV2Model):
            config = {}
            chat_template = (
                "{% for message in messages %}"
                "{% for call in message.tool_calls %}"
                "{{ call.function.arguments | tojson(ensure_ascii=False) }}"
                "{% endfor %}{% endfor %}"
            )

        tokenizer = NativeV2Tokenizer(TemplateModel())  # type: ignore[arg-type]
        prompt = tokenizer.format_messages([
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "type": "function",
                    "function": {"name": "read", "arguments": {"limit": 10}},
                }],
            },
        ])

        # Spacing matters: this is what the model saw in training and what
        # every other runtime emits. The compact form was ours alone.
        self.assertEqual(prompt, '{"limit": 10}')

    def test_an_empty_assistant_turn_renders_instead_of_failing(self) -> None:
        # A cancelled generation, or one the token ceiling cut mid-tool-call,
        # comes back to the client as an empty assistant turn. The protocol
        # layer keeps it on purpose so the conversation can replay -- and every
        # renderer here refused it, so the request 400'd as "unable to tokenize
        # the formatted prompt" and kept doing so for the rest of that
        # conversation, because the turn is history the client cannot edit.
        class TemplateModel(StubV2Model):
            config = {}
            chat_template = (
                "{% for message in messages %}[{{ message.role }}]"
                "{{ message.content }}{% endfor %}"
                "{% if add_generation_prompt %}[assistant]{% endif %}"
            )

        conversation = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "go on"},
        ]

        templated = NativeV2Tokenizer(TemplateModel())  # type: ignore[arg-type]
        self.assertEqual(
            templated.format_messages(conversation),
            "[user]hi[assistant][user]go on[assistant]",
        )

        # ...and every architecture fallback, which renders without a template.
        for architecture in ("qwen3moe", "gemma4", "laguna", "deepseek4"):
            with self.subTest(architecture=architecture):
                tokenizer = object.__new__(NativeV2Tokenizer)
                tokenizer.architecture = architecture
                rendered = tokenizer.format_messages(conversation)
                self.assertIn("go on", rendered)

    def test_what_the_protocol_layer_accepts_this_renderer_renders(self) -> None:
        # The two layers disagreeing is what made the bug survive: the request
        # parser deliberately preserves empty assistant and tool turns so a
        # conversation replays, and the renderer one call later refused the
        # very same turns. Drive the parser's own output through the renderer
        # so neither side can tighten alone.
        payload = {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "read it"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "read",
                                     "arguments": '{"path": "/etc"}'},
                    }],
                },
                {"role": "tool", "tool_call_id": "call_0", "content": ""},
                {"role": "assistant", "content": None},
                {"role": "user", "content": "go on"},
            ],
            "tools": [{
                "type": "function",
                "function": {"name": "read", "parameters": {"type": "object"}},
            }],
        }
        for architecture in ("qwen3moe", "gemma4", "laguna", "deepseek4",
                             "bailingmoe3"):
            with self.subTest(architecture=architecture):
                messages, _ = _chat_messages(payload, architecture=architecture)
                tokenizer = object.__new__(NativeV2Tokenizer)
                tokenizer.architecture = architecture
                if architecture in ("gemma4", "laguna"):
                    # Neither fallback renders tool turns at all; the parser
                    # rewrote them into user text for these, so the turn that
                    # reaches them is an ordinary one.
                    messages = [
                        message for message in messages
                        if message["role"] != "tool"
                    ]
                self.assertIn("go on", tokenizer.format_messages(messages))

    def test_an_empty_user_turn_names_the_message_it_rejects(self) -> None:
        # The other half of the rule: an empty user turn is a client bug, and
        # the caller sees this through a wrapper that reports only that
        # tokenization failed -- so the message has to say which turn.
        tokenizer = object.__new__(NativeV2Tokenizer)
        tokenizer.architecture = "qwen3moe"
        with self.assertRaisesRegex(ValueError, r"messages\[1\] has role 'user'"):
            tokenizer.format_messages([
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "   "},
            ])

    def test_an_empty_tool_result_renders_on_a_native_tool_template(self) -> None:
        # A tool that printed nothing is a real result. The compatibility layer
        # keeps the empty tool turn for architectures whose template renders
        # tool results itself; refusing it here failed the request instead.
        class TemplateModel(StubV2Model):
            config = {}
            chat_template = (
                "{% for message in messages %}[{{ message.role }}]"
                "{{ message.content }}{% endfor %}"
                "{% if add_generation_prompt %}[assistant]{% endif %}"
            )

        tokenizer = NativeV2Tokenizer(TemplateModel())  # type: ignore[arg-type]
        prompt = tokenizer.format_messages([
            {"role": "user", "content": "read it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "type": "function",
                    "function": {"name": "read", "arguments": {"path": "/etc"}},
                }],
            },
            {"role": "tool", "content": ""},
        ])

        self.assertEqual(
            prompt, "[user]read it[assistant][tool][assistant]"
        )

    def test_multibyte_character_split_across_tokens_decodes_intact(self) -> None:
        # Byte-level BPE splits "⚽" (e2 9a bd) across tokens 40+41. Per-token
        # string decoding turned each half into U+FFFD, and that corruption was
        # written into files by tool calls. The incremental UTF-8 decoder must
        # reassemble the character.
        generator, _ = self.make_generator([0, 40, 41, 99])
        result = generator.generate_text("Hi", max_new_tokens=8)
        self.assertEqual(result.text, "⚽")
        self.assertNotIn("�", result.text)

    def test_native_generator_stops_on_eos(self) -> None:
        generator, runtime = self.make_generator([10, 99])
        result = generator.generate_text("Hi", max_new_tokens=4)
        self.assertEqual(result.generated_ids, (99,))
        self.assertEqual(result.text, "")
        self.assertTrue(result.stopped_on_eos)
        self.assertEqual(runtime.inputs, [1, 2])

    def test_native_generator_forwards_sampling_to_engine(self) -> None:
        generator, runtime = self.make_generator([10, 20])
        result = generator.generate_text(
            "Hi",
            max_new_tokens=1,
            sampling=SamplingConfig(
                temperature=0.5, top_k=7, top_p=0.8, seed=42
            ),
        )
        self.assertIsNotNone(result)
        self.assertEqual(runtime._last_sampling, (0.5, 7, 0.8, 42))

    def test_native_generator_forwards_penalties_to_engine(self) -> None:
        # The penalties travel by their own arguments rather than riding on
        # temperature, so a request that sets them must reach the engine
        # unchanged -- silently dropping them would restore exactly the
        # unpenalized sampling that lets the model loop.
        generator, runtime = self.make_generator([10, 20])
        generator.generate_text(
            "Hi",
            max_new_tokens=1,
            sampling=SamplingConfig(
                temperature=0.5,
                repetition_penalty=1.3,
                presence_penalty=0.4,
                frequency_penalty=0.2,
                penalty_window=128,
            ),
        )
        self.assertEqual(runtime._last_penalties, (1.3, 0.4, 0.2, 128))

    def test_native_generator_penalizes_repetition_by_default(self) -> None:
        # An unset penalty must not mean "off": no penalty is the setting that
        # produced the observed loop, so the default carries a real one.
        generator, runtime = self.make_generator([10, 20])
        generator.generate_text("Hi", max_new_tokens=1,
                                sampling=SamplingConfig(temperature=0.5))
        repetition, _, _, window = runtime._last_penalties
        self.assertGreater(repetition, 1.0)
        self.assertGreater(window, 0)

    def test_native_generator_forwards_tool_schemas_to_the_sampler(self) -> None:
        # The sampler constrains a tool call to the caller's schemas, so the
        # names and which parameters are required have to reach the engine. A
        # request whose tools are dropped here samples freely and produces
        # exactly the malformed call the constraint exists to prevent.
        generator, runtime = self.make_generator([10, 20])
        generator.generate_messages(
            [
                {
                    "role": "user",
                    "content": "run ls",
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "command": {"type": "string"},
                                        "description": {"type": "string"},
                                        "timeout": {"type": "number"},
                                        # The sampler holds this one to JSON:
                                        # a string here reaches the client as
                                        # "expected array, received string".
                                        "questions": {"type": "array"},
                                    },
                                    "required": ["command", "description"],
                                },
                            },
                        }
                    ],
                }
            ],
            max_new_tokens=1,
        )
        self.assertEqual(
            runtime._last_tools,
            [
                {
                    "name": "bash",
                    "parameters": [
                        {"name": "command", "required": True, "type": "string"},
                        {"name": "description", "required": True, "type": "string"},
                        # A number is text as far as the constraint is
                        # concerned; the server coerces "5" on the way out.
                        {"name": "timeout", "required": False, "type": "string"},
                        {"name": "questions", "required": False, "type": "array"},
                    ],
                }
            ],
        )

    def test_native_generator_sends_no_schemas_when_there_are_no_tools(self) -> None:
        # Prose must not be constrained: an empty specification is what leaves
        # the sampler alone, and sending one anyway would arm a grammar with
        # nothing to accept.
        generator, runtime = self.make_generator([10, 20])
        generator.generate_messages(
            [{"role": "user", "content": "hello"}], max_new_tokens=1
        )
        self.assertIn(runtime._last_tools, (None, []))
        self.assertIsNone(runtime._last_response_format)
        # And with no tools declared, the markup ban arms: no parser exists
        # for a <tool_call> block, so the sampler must not write one -- the
        # opencode compaction bug stored exactly that as its session summary.
        self.assertTrue(runtime._last_forbid_tool_calls)

    def test_declared_tools_leave_the_markup_ban_off(self) -> None:
        generator, runtime = self.make_generator([10, 20])
        generator.generate_messages(
            [{"role": "user", "content": "hello"}],
            max_new_tokens=1,
            tools=[{
                "type": "function",
                "function": {"name": "bash", "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                }},
            }],
        )
        self.assertTrue(runtime._last_tools)
        self.assertFalse(runtime._last_forbid_tool_calls)

    def test_native_generator_forwards_the_response_constraint(self) -> None:
        # The server's response_format shape must reach the native sampler;
        # dropping it here is the prompt-only JSON mode all over again.
        generator, runtime = self.make_generator([10, 20])
        generator.generate_messages(
            [{"role": "user", "content": "give me json"}],
            max_new_tokens=1,
            response_format={"shape": "object", "thinking_open": False},
        )
        self.assertEqual(
            runtime._last_response_format,
            {"shape": "object", "thinking_open": False},
        )

    def test_native_generator_reports_runtime_prefix_cache(self) -> None:
        generator, _ = self.make_generator([10])
        self.assertEqual(
            generator.prefix_cache_stats(),
            {
                "entries": 0,
                "capacity": 1,
                "ram_entries": 0,
                "ram_bytes": 0,
                "hits": 3,
                "misses": 1,
                "evictions": 0,
                "reused_tokens": 42,
                "last_prompt_tokens": 0,
                "last_reused_tokens": 0,
                "last_lcp_live": 0,
                "last_lcp_snapshot": 0,
                "kv_reserved_bytes": 0,
                "kv_peak_live_bytes": 0,
                "kv_peak_tokens": 0,
                "kv_peak_tokens_max": 0,
                "kv_occupancy_samples": 0,
                "donations": 0,
                "donated_tokens": 0,
            },
        )

    def test_close_cancels_tasks_and_joins_engine_before_runtime_teardown(self) -> None:
        model = StubV2Model()
        runtime = BlockingV2Runtime()
        generator = NativeV2Generator(  # type: ignore[arg-type]
            model, runtime, NativeV2Tokenizer(model)  # type: ignore[arg-type]
        )
        _, queue = generator.engine.submit([1, 2], 8, ())
        self.assertTrue(runtime.entered.wait(1))

        generator.close()

        self.assertEqual(runtime.cancels, 1)
        self.assertEqual(queue.get(timeout=1)[0], "error")
        self.assertIsNotNone(generator.engine._thread)
        self.assertFalse(generator.engine._thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            generator.engine.submit([1, 2], 8, ())
        generator.close()  # idempotent

    def test_native_engine_rejects_work_past_active_task_limit(self) -> None:
        runtime = BlockingV2Runtime()
        generator = NativeV2Generator(  # type: ignore[arg-type]
            StubV2Model(), runtime, NativeV2Tokenizer(StubV2Model())  # type: ignore[arg-type]
        )
        generator.engine._MAX_ACTIVE_TASKS = 1
        generator.engine.submit([1], 2, ())
        self.assertTrue(runtime.entered.wait(1))

        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            generator.engine.submit([2], 2, ())

        generator.close()

    def test_native_engine_cancels_stalled_output_consumer(self) -> None:
        runtime = StubV2Runtime([10, 11, 12, 13])
        generator = NativeV2Generator(  # type: ignore[arg-type]
            StubV2Model(), runtime, NativeV2Tokenizer(StubV2Model())  # type: ignore[arg-type]
        )
        generator.engine._MAX_BUFFERED_EVENTS = 2
        _, task_queue = generator.engine.submit([1], 4, ())

        kind, message = task_queue.get(timeout=1)
        self.assertEqual(kind, "error")
        self.assertIn("output queue overflow", str(message))
        self.assertEqual(runtime.cancels, 1)
        generator.close()

    def test_unstated_thinking_is_not_answered_on_the_checkpoint_s_behalf(self) -> None:
        """None means "the request did not say", and only the template decides.

        Coercing it to False rendered `<think></think>` on every request that
        never mentioned thinking, which tells a model trained to reason first
        not to -- and takes any reasoning-effort instruction with it, since a
        template only grades reasoning it is doing.
        """
        generator, _ = self.make_generator([10])
        seen: list[object] = []
        original = generator.tokenizer.encode_messages

        def record(messages, **options):
            seen.append(options.get("enable_thinking"))
            return original(messages, **options)

        generator.tokenizer.encode_messages = record
        generator.prepare_messages([{"role": "user", "content": "Hi"}])
        self.assertEqual(seen, [None])
        seen.clear()
        generator.prepare_messages(
            [{"role": "user", "content": "Hi"}], enable_thinking=False
        )
        self.assertEqual(seen, [False])

    def test_native_chat_continuation_preserves_generated_token_ids(self) -> None:
        generator, _ = self.make_generator([10, 20, 30])
        first_messages = (("user", "Hi"),)
        generator.generate_messages(
            [{"role": "user", "content": "Hi"}], max_new_tokens=2
        )
        # None, not False: the generation above never mentioned thinking, and
        # the two render differently -- a prefix rendered with thinking
        # explicitly off is not a prefix of a conversation that left it to the
        # checkpoint.
        continued = generator._continued_chat_prompt(
            (
                *first_messages,
                ("assistant", "Hello world"),
                ("user", "Again"),
            ),
            None,
        )
        self.assertEqual(continued, [1, 2, 20, 30, 1, 2, 1, 2])

    def test_native_tool_call_round_trip_preserves_generated_token_ids(self) -> None:
        generator, _ = self.make_generator([10])
        generator._chat_messages = (("user", "Write it"),)
        generator._chat_prompt_ids = (7, 8)
        generator._chat_generated_ids = (20, 30)
        generator._chat_text = (
            "Internal reasoning that the API may omit.\n"
            "<tool_call>\n<function=write_file>\n"
            "<parameter=path>\n/tmp/example\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        generator._chat_thinking = False

        continued = generator._continued_chat_prompt(
            (
                ("user", "Write it"),
                (
                    "assistant",
                    "<tool_call>\n<function=write_file>\n"
                    "<parameter=path>\n/tmp/example\n</parameter>\n"
                    "</function>\n</tool_call>",
                ),
                ("user", "<tool_response>\ndone\n</tool_response>"),
            ),
            False,
        )

        self.assertEqual(continued, [7, 8, 20, 30, 1, 2, 1, 2])

    def test_concurrent_side_chat_does_not_evict_main_continuation(self) -> None:
        generator, _ = self.make_generator([10])
        main = (("user", "Long main conversation"),)
        generator._chat_continuations[main] = (
            (7, 8),
            (20, 30),
            "Hello world",
            False,
            None,   # the reasoning effort the prefix was rendered at
        )
        # A later short request owns the legacy last-writer fields.
        generator._chat_messages = (("user", "Make a title"),)
        generator._chat_prompt_ids = (90,)
        generator._chat_generated_ids = (91,)
        generator._chat_text = "Title"
        generator._chat_thinking = False

        continued = generator._continued_chat_prompt(
            (
                *main,
                ("assistant", "Hello world"),
                ("user", "Continue"),
            ),
            False,
        )

        self.assertEqual(continued, [7, 8, 20, 30, 1, 2, 1, 2])

    def test_native_generator_works_through_server_contract(self) -> None:
        generator, _ = self.make_generator([10, 20, 30])
        service = InferenceService(
            "native-test", generator, max_new_tokens=8, context_window=32
        )
        response = service.chat_completion(
            {
                "model": "native-test",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 2,
                "temperature": 0,
            }
        )
        self.assertEqual(response["choices"][0]["message"]["content"], "Hello world")
        self.assertEqual(response["usage"]["completion_tokens"], 2)

    def test_serve_v2_cli_defaults_to_auto_expert_mode(self) -> None:
        args = _parser().parse_args(["serve-v2", "model.gguf"])
        self.assertEqual(args.expert_mode, "auto")
        # 0 = auto-fit the GPU expert cache to free VRAM (manual MiB still settable).
        self.assertEqual(args.gpu_cache_mib, 0)
        self.assertEqual(args.context_window, 32768)
        self.assertEqual(args.mtp_drafts, 0)
        self.assertIsNone(args.prefill_cache_seed)
        self.assertEqual(args.expert_paging, "auto")
        self.assertEqual(args.cpu_prefetch_mib, 0)
        self.assertFalse(args.cpu_prefetch_auto)
        self.assertEqual(args.next_layer_prefetch, 0)
        self.assertEqual(args.cpu_threads, 0)
        # No parser default: "not asked" has to be distinguishable from an
        # explicit choice, because the runtime defaults prompt processing to the
        # host under `auto` expert placement. The parser used to say "split"
        # here and the runtime replaced it with "cpu" regardless, so the flag
        # silently did nothing.
        self.assertIsNone(args.hybrid_prefill)
        self.assertIsNone(args.expert_residency)
        self.assertEqual(args.dense_requant, "auto")
        self.assertEqual(args.prompt_cache_mib, (1 << 32) - 1)

    def test_serve_cache_has_simple_public_modes(self) -> None:
        automatic = _parser().parse_args(["serve", "model.gguf", "--cache", "auto"])
        disabled = _parser().parse_args(["serve", "model.gguf", "--cache", "off"])
        explicit = _parser().parse_args(["serve", "model.gguf", "--cache", "2048"])
        legacy = _parser().parse_args(
            ["serve", "model.gguf", "--prompt-cache-mib", "512"]
        )
        self.assertEqual(automatic.prompt_cache_mib, (1 << 32) - 1)
        self.assertEqual(disabled.prompt_cache_mib, 0)
        self.assertEqual(explicit.prompt_cache_mib, 2048)
        self.assertEqual(legacy.prompt_cache_mib, 512)

    def test_benchmark_v2_exposes_native_runtime_tuning_options(self) -> None:
        defaults = _parser().parse_args(["benchmark-v2", "model.gguf"])
        self.assertEqual(defaults.gpu_cache_mib, 0)
        self.assertEqual(defaults.iterations, 128)

        args = _parser().parse_args([
            "benchmark-v2", "model.gguf",
            "--cache-type-k", "f32",
            "--cache-type-v", "q8_0",
            "--expert-top-k", "6",
            "--expert-top-p", "0.9",
            "--parallel", "2",
            "--prompt-cache-mib", "512",
            "--prefill-cache-seed", "8",
            "--expert-paging", "direct",
            "--cpu-prefetch-mib", "768",
            "--next-layer-prefetch", "6",
            "--hybrid-prefill", "cpu",
            "--expert-residency", "immutable",
            "--dense-requant", "off",
            '--cpu-threads', '12',
            "--cold-cache",
        ])
        self.assertEqual(args.cache_type_k, "f32")
        self.assertEqual(args.cache_type_v, "q8_0")
        self.assertEqual(args.expert_top_k, 6)
        self.assertAlmostEqual(args.expert_top_p, 0.9)
        self.assertEqual(args.parallel_sequences, 2)
        self.assertEqual(args.prompt_cache_mib, 512)
        self.assertEqual(args.prefill_cache_seed, 8)
        self.assertEqual(args.expert_paging, "direct")
        self.assertEqual(args.cpu_prefetch_mib, 768)
        self.assertEqual(args.next_layer_prefetch, 6)
        self.assertEqual(args.cpu_threads, 12)
        self.assertEqual(args.hybrid_prefill, "cpu")
        self.assertEqual(args.expert_residency, "immutable")
        self.assertEqual(args.dense_requant, "off")
        self.assertTrue(args.cold_cache)
        self.assertFalse(args.cpu_prefetch_auto)

    def test_benchmark_v2_exposes_auto_cpu_prefetch(self) -> None:
        args = _parser().parse_args([
            "benchmark-v2", "model.gguf", "--cpu-prefetch-auto",
        ])
        self.assertEqual(args.cpu_prefetch_mib, 0)
        self.assertTrue(args.cpu_prefetch_auto)

    def test_serve_v2_cli_accepts_legacy_gpu_alias(self) -> None:
        args = _parser().parse_args([
            "serve-v2", "model.gguf",
            "--moe-device", "gpu",
            "--expert-paging", "direct",
            "--next-layer-prefetch", "4",
        ])
        self.assertEqual(args.expert_mode, "gpu")
        self.assertEqual(args.expert_paging, "direct")
        self.assertEqual(args.next_layer_prefetch, 4)

    def test_serve_v2_cli_accepts_canonical_expert_modes(self) -> None:
        for mode in ("cpu", "auto", "resident"):
            with self.subTest(mode=mode):
                args = _parser().parse_args(
                    ["serve-v2", "model.gguf", "--expert-mode", mode]
                )
                self.assertEqual(args.expert_mode, mode)


if __name__ == "__main__":
    unittest.main()
