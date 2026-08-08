"""The DeepSeek-V4 scheduler, and the server surface it carries.

The engine the other architectures share schedules slots of one KV pair per
layer. This model has no such thing -- a sliding-window latent ring, a
compressed-block cache, the compressor's partial block and, on a 4:1 layer, the
indexer's own cache -- so it brings its own scheduler and keeps everything above
it unchanged.

What is worth testing is exactly what the scheduler had to decide, since the
transport above and the forward below are already covered elsewhere:

- a slot can be reused only by a prompt that *extends* it, because the runtime
  advances a sequence and cannot take a position back;
- tasks interleave at chunk granularity on one thread, because the slots share
  the weights and running two forwards at once buys nothing;
- the context limit is refused up front rather than truncated silently.

The miniature fixture stands in for the 104 GB checkpoint: its weights are
arbitrary, so nothing here asserts on the *text*, only on the bookkeeping.
"""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from colibri_next.deepseek4_server import (
    Deepseek4Engine,
    NativeDeepseek4InferenceService,
    sample_token,
)
from colibri_next.sampling import SamplingConfig
from colibri_next.v2 import V2Model
from tests.deepseek4_gguf_fixture import DeepSeek4Spec, build_deepseek4_gguf

_CHECKPOINT_PATH = os.environ.get("DEEPSEEK4_GGUF")
# A stale path is as good as no path.
CHECKPOINT = _CHECKPOINT_PATH if _CHECKPOINT_PATH and os.path.exists(_CHECKPOINT_PATH) else None


class _Fixture:
    """A tiny real deepseek4 model, shared by every test in the module."""

    directory: tempfile.TemporaryDirectory | None = None
    path: Path

    @classmethod
    def open(cls) -> V2Model:
        if cls.directory is None:
            cls.directory = tempfile.TemporaryDirectory(prefix="colibri-ds4svc-")
            cls.path = Path(cls.directory.name) / "ds4.gguf"
            build_deepseek4_gguf(cls.path, DeepSeek4Spec(layers=6, hash_layers=3))
        return V2Model(cls.path)


def tearDownModule():
    if _Fixture.directory is not None:
        _Fixture.directory.cleanup()
        _Fixture.directory = None


def drain(queue) -> list[tuple[str, object]]:
    events = []
    while True:
        event = queue.get(timeout=120)
        events.append(event)
        if event[0] in ("done", "error"):
            return events


class SamplingTests(unittest.TestCase):
    def test_zero_temperature_is_greedy(self):
        logits = np.array([0.1, 5.0, 0.2], dtype=np.float32)
        chosen = sample_token(
            logits, SamplingConfig(temperature=0.0), np.random.default_rng(0)
        )
        self.assertEqual(chosen, 1)

    def test_top_k_of_one_is_greedy_whatever_the_temperature(self):
        logits = np.array([0.1, 5.0, 0.2], dtype=np.float32)
        for seed in range(5):
            chosen = sample_token(
                logits,
                SamplingConfig(temperature=2.0, top_k=1),
                np.random.default_rng(seed),
            )
            self.assertEqual(chosen, 1)

    def test_a_seed_makes_sampling_reproducible(self):
        logits = np.array([1.0, 1.1, 0.9, 1.2], dtype=np.float32)
        runs = [
            [
                sample_token(
                    logits,
                    SamplingConfig(temperature=1.0),
                    np.random.default_rng(11),
                )
                for _ in range(8)
            ]
            for _ in range(2)
        ]
        self.assertEqual(runs[0], runs[1])

    def test_the_nucleus_is_never_empty(self):
        # One token holds more mass than top_p, which a naive cumulative cut
        # would leave with nothing to sample from.
        logits = np.array([20.0, 0.0, 0.0], dtype=np.float32)
        chosen = sample_token(
            logits,
            SamplingConfig(temperature=1.0, top_k=0, top_p=0.5),
            np.random.default_rng(3),
        )
        self.assertEqual(chosen, 0)


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _Fixture.open()

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def engine(self, **options) -> Deepseek4Engine:
        engine = Deepseek4Engine(self.model, options.pop("context", 256), **options)
        self.addCleanup(engine.close)
        return engine

    def test_it_prefills_then_generates(self):
        engine = self.engine()
        _, queue = engine.submit([5, 6, 7, 8], 3, ())
        events = drain(queue)
        kinds = [kind for kind, _ in events]
        self.assertEqual(kinds[-1], "done")
        self.assertEqual(kinds.count("token"), 3)
        # Prefill reports progress against the whole prompt and completes.
        prefill = [value for kind, value in events if kind == "prefill"]
        self.assertEqual(prefill[-1], 4)
        # The first report is read as the count served from cache, so on a
        # cold slot it has to be zero rather than the first chunk.
        self.assertEqual(prefill[0], 0)

    def test_the_first_prefill_report_is_the_reused_prefix(self):
        engine = self.engine()
        _, queue = engine.submit([5, 6, 7], 2, ())
        generated = [value for kind, value in drain(queue) if kind == "token"]
        _, queue = engine.submit([5, 6, 7] + generated + [9, 10], 1, ())
        prefill = [value for kind, value in drain(queue) if kind == "prefill"]
        self.assertEqual(prefill[0], 4)
        self.assertEqual(prefill[-1], 7)

    def test_a_stop_token_ends_generation_early(self):
        engine = self.engine()
        _, queue = engine.submit([5, 6], 8, ())
        first = next(value for kind, value in drain(queue) if kind == "token")
        # Feed the same prompt again, this time stopping on what it produced.
        _, queue = engine.submit([5, 6], 8, (first,))
        tokens = [value for kind, value in drain(queue) if kind == "token"]
        self.assertEqual(tokens, [first])

    def test_a_continuation_reuses_the_slot_it_extends(self):
        engine = self.engine()
        _, queue = engine.submit([5, 6, 7], 2, ())
        generated = [value for kind, value in drain(queue) if kind == "token"]
        self.assertEqual(engine.misses, 1)
        continuation = [5, 6, 7] + generated + [9, 10]
        _, queue = engine.submit(continuation, 1, ())
        drain(queue)
        self.assertEqual(engine.hits, 1)
        # Four, not five: the last token a task generates is never forwarded --
        # its logits would be the *next* token's, which nothing asked for -- so
        # the slot holds the prompt plus all but the final reply token.
        self.assertEqual(engine.last_reused_tokens, 4)

    def test_an_unrelated_prompt_starts_from_scratch(self):
        engine = self.engine()
        _, queue = engine.submit([5, 6, 7], 1, ())
        drain(queue)
        _, queue = engine.submit([11, 12, 13], 1, ())
        drain(queue)
        self.assertEqual(engine.hits, 0)
        self.assertEqual(engine.misses, 2)

    def test_an_identical_prompt_is_not_reused(self):
        # The last token has to be forwarded again for its logits, and the
        # runtime cannot take that position back, so the slot must be reset.
        engine = self.engine()
        prompt = [5, 6, 7]
        _, queue = engine.submit(prompt, 1, ())
        drain(queue)
        _, queue = engine.submit(prompt, 1, ())
        drain(queue)
        self.assertEqual(engine.hits, 0)
        self.assertEqual(engine.misses, 2)

    def test_reuse_produces_the_same_tokens_as_a_cold_run(self):
        # The whole point of the reuse: it must be an optimization only, so a
        # continuation served from a warm slot has to match the same token
        # sequence served from an empty one.
        warm = self.engine()
        _, queue = warm.submit([5, 6, 7], 1, ())
        first = [value for kind, value in drain(queue) if kind == "token"]
        continuation = [5, 6, 7] + first + [8, 9]
        _, queue = warm.submit(continuation, 3, ())
        reused = [value for kind, value in drain(queue) if kind == "token"]
        self.assertEqual(warm.hits, 1)

        cold = self.engine()
        _, queue = cold.submit(continuation, 3, ())
        fresh = [value for kind, value in drain(queue) if kind == "token"]
        self.assertEqual(cold.misses, 1)
        self.assertEqual(reused, fresh)

    def test_a_prompt_past_the_context_limit_is_refused(self):
        engine = self.engine(context=64)
        with self.assertRaises(ValueError) as raised:
            engine.submit(list(range(60)), 16, ())
        self.assertIn("context limit", str(raised.exception))

    def test_an_empty_prompt_is_refused(self):
        with self.assertRaises(ValueError):
            self.engine().submit([], 4, ())

    def test_two_requests_interleave_on_two_slots(self):
        engine = self.engine(slots=2)
        _, first = engine.submit([5, 6, 7, 8], 4, ())
        _, second = engine.submit([11, 12, 13, 14], 4, ())
        for queue in (first, second):
            events = drain(queue)
            self.assertEqual(events[-1][0], "done")
            self.assertEqual(sum(1 for kind, _ in events if kind == "token"), 4)

    def test_one_slot_serves_both_requests_in_turn(self):
        engine = self.engine(slots=1)
        _, first = engine.submit([5, 6, 7], 2, ())
        _, second = engine.submit([11, 12, 13], 2, ())
        for queue in (first, second):
            self.assertEqual(drain(queue)[-1][0], "done")

    def test_a_cancelled_task_releases_its_slot(self):
        engine = self.engine(slots=1)
        task_id, queue = engine.submit([5, 6, 7], 32, ())
        queue.get(timeout=120)
        engine.cancel(task_id)
        engine.forget(task_id)
        _, other = engine.submit([11, 12, 13], 2, ())
        self.assertEqual(drain(other)[-1][0], "done")

    def test_closing_reports_shutdown_to_waiting_requests(self):
        engine = Deepseek4Engine(self.model, 256)
        _, queue = engine.submit([5, 6, 7], 64, ())
        queue.get(timeout=120)
        engine.close()
        self.assertEqual(
            [kind for kind, _ in drain(queue)][-1], "error"
        )

    def test_submitting_after_close_is_refused(self):
        engine = Deepseek4Engine(self.model, 256)
        engine.close()
        with self.assertRaises(RuntimeError):
            engine.submit([5, 6], 1, ())

    def test_slots_must_be_positive(self):
        with self.assertRaises(ValueError):
            Deepseek4Engine(self.model, 256, slots=0)


class ServiceTests(unittest.TestCase):
    def service(self, **options) -> NativeDeepseek4InferenceService:
        _Fixture.open().close()
        service = NativeDeepseek4InferenceService(
            _Fixture.path,
            context_window=options.pop("context_window", 512),
            max_new_tokens=options.pop("max_new_tokens", 8),
            **options,
        )
        self.addCleanup(service.close)
        return service

    def test_a_chat_completion_round_trip(self):
        service = self.service()
        response = service.chat_completion({
            "model": service.model_name,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        })
        self.assertEqual(response["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(response["usage"]["completion_tokens"], 4)
        self.assertGreater(response["usage"]["prompt_tokens"], 0)

    def test_streaming_yields_a_terminal_step(self):
        service = self.service()
        steps = list(service.generator.stream_messages(
            [{"role": "user", "content": "hello"}], max_new_tokens=3
        ))
        self.assertTrue(steps[-1].finished)
        self.assertEqual(sum(1 for step in steps if step.token_id is not None), 3)

    def test_a_second_turn_reuses_the_first(self):
        service = self.service()
        first = service.chat_completion({
            "model": service.model_name,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        })
        reply = first["choices"][0]["message"]["content"]
        service.chat_completion({
            "model": service.model_name,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "again"},
            ],
            "max_tokens": 2,
        })
        stats = service.health()["prefix_cache"]
        self.assertEqual(stats["hits"], 1)
        self.assertGreater(stats["reused_tokens"], 0)

    def test_health_reports_the_deepseek4_backend(self):
        service = self.service(parallel_sequences=2)
        execution = service.health()["execution"]
        self.assertEqual(execution["backend"], "native-v2-deepseek4-cpu")
        self.assertEqual(execution["slots"], 2)
        self.assertEqual(execution["layers"], 6)

    def test_concurrent_requests_both_complete(self):
        service = self.service(parallel_sequences=2)
        results: dict[int, object] = {}

        def run(index: int, content: str) -> None:
            results[index] = service.chat_completion({
                "model": service.model_name,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 3,
            })

        threads = [
            threading.Thread(target=run, args=(index, content))
            for index, content in enumerate(("alpha", "beta"))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=300)
        self.assertEqual(sorted(results), [0, 1])

    def test_a_runtime_knob_this_path_cannot_honour_is_refused(self):
        with self.assertRaises(ValueError) as raised:
            self.service(cache_type_k="q8_0")
        self.assertIn("cache_type_k", str(raised.exception))

    def test_an_untouched_knob_is_accepted(self):
        # Its default matches the Qwen service's, so it asked for nothing.
        self.service(cache_type_k="f16", gpu_cache_mib=0)

    def test_an_unknown_keyword_is_refused(self):
        with self.assertRaises(TypeError):
            self.service(nonsense=1)

    def test_a_non_deepseek4_checkpoint_is_refused(self):
        from tests.dense_gguf_fixture import DenseQwenSpec, build_dense_qwen35_gguf

        with tempfile.TemporaryDirectory(prefix="colibri-ds4svc-") as directory:
            path = Path(directory) / "dense.gguf"
            build_dense_qwen35_gguf(path, DenseQwenSpec(layers=2))
            with self.assertRaises(ValueError):
                NativeDeepseek4InferenceService(path)


@unittest.skipUnless(
    CHECKPOINT,
    "set DEEPSEEK4_GGUF to the first shard of a real checkpoint",
)
class ContinuationOnTheRealCheckpointTests(unittest.TestCase):
    """A reused prefix has to be the prompt the template would have rendered.

    Splicing generated ids onto a cached prefix saves re-running the prompt,
    but it also bypasses the template -- so the two must agree token for token,
    or the model sees markup its training never contained and the saving is
    paid for in output quality.
    """

    def test_a_spliced_second_turn_matches_a_fresh_render(self):
        service = NativeDeepseek4InferenceService(
            CHECKPOINT, context_window=2048, max_new_tokens=16
        )
        self.addCleanup(service.close)
        messages = [{"role": "user", "content": "Name the capital of France in one word."}]
        reply = service.chat_completion({
            "model": service.model_name,
            "messages": messages,
            "max_tokens": 8,
        })["choices"][0]["message"]["content"]
        messages = messages + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": "And of Italy? One word."},
        ]
        spliced = service.generator.prepare_messages(messages)
        fresh = service.generator.tokenizer.encode_messages(messages)
        self.assertEqual(spliced, fresh)


if __name__ == "__main__":
    unittest.main()
