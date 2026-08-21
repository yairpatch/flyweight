from __future__ import annotations

import argparse
import json
import tempfile
import threading
import unittest
from pathlib import Path
from queue import Empty, Queue

from colibri_next.cli import _parser
from colibri_next.sampling import SamplingConfig
from colibri_next.server import InferenceService
from colibri_next.v2_server import (
    BailingEngine,
    NativeV2Generator,
    NativeV2InferenceService,
    NativeV2Tokenizer,
    _generation_config_for_model,
    _merge_generation_defaults,
)
from colibri_next.v2 import TASK_EVENT_PREFILL


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
    ):
        self._tasks = getattr(self, "_tasks", {})
        self._next_task = getattr(self, "_next_task", 0) + 1
        self._last_sampling = (temperature, top_k, top_p, seed)
        self._last_penalties = (
            repetition_penalty, presence_penalty, frequency_penalty, penalty_window
        )
        self._last_tools = tools
        self._last_response_format = response_format
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
    def __init__(self) -> None:
        self.resets = 0
        self.eval_calls: list[list[int]] = []
        self.next_token = 20
        self.uses_gpu = False
        self.position = 0
        self.saves = 0
        self.loads: list[bytes] = []
        self.progress = None

    def reset(self) -> None:
        self.resets += 1
        self.position = 0

    def set_progress(self, callback) -> None:
        self.progress = callback

    def eval_into(self, tokens) -> None:
        tokens = list(tokens)
        # The runtime reports per tile while a prompt runs and stops when the
        # watcher says so; a tile here is two tokens, so tests stay small.
        if self.progress is not None and len(tokens) > 1:
            for offset in range(0, len(tokens), 2):
                if self.progress(offset, len(tokens)) is False:
                    self.position += offset
                    raise RuntimeError("bailing prompt evaluation was cancelled")
        self.eval_calls.append(tokens)
        self.position += len(tokens)

    def save_state(self) -> bytes:
        self.saves += 1
        return f"state@{self.position}".encode()

    def load_state(self, snapshot: bytes) -> None:
        self.loads.append(snapshot)
        # Trailing padding lets a subclass give a snapshot a realistic size.
        self.position = int(snapshot.decode().split("@")[1].rstrip("."))

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
        self.assertFalse(engine._cache_initialized)

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

            def save_state(self) -> bytes:
                return super().save_state().ljust(self.position * 10, b".")

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
            def eval_into(self, tokens) -> None:
                if list(tokens) == [7]:
                    self.position += 1  # the runtime advanced before it failed
                    raise RuntimeError("device fell over")
                super().eval_into(tokens)

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
            self.assertEqual(engine._cached_tokens, [])
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
        from colibri_next.sampling import SERVER_SETTINGS, SETTINGS

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
        from colibri_next.server import _sampling_from_payload
        from colibri_next.sampling import defaults as builtin_defaults
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
        self.assertEqual(args.hybrid_prefill, "split")
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
