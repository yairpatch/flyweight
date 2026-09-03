"""The client's "answer now": POST .../{id}/stop_thinking on a live stream.

The server keeps an Event per streaming request, keyed by the id the stream
reports in its first event; the generator honors a set Event by forcing an
open thinking block closed through the same path the token budget uses.
"""

from __future__ import annotations

import http.client
import json
import threading
import unittest
from queue import Queue

from flyweight.server import (
    APIError,
    FlyweightHTTPServer,
    InferenceService,
    _stop_thinking_target,
    create_handler,
)
from flyweight.v2_server import THINKING_BUDGET_CLOSE, ChatGenerator

from tests.test_server import StubGenerator


class InterruptibleStub(StubGenerator):
    """A generator that reports the option and lets the test trip it."""

    supports_stop_thinking = True

    def __init__(self) -> None:
        super().__init__()
        self.events: list[threading.Event] = []

    def stream_messages(self, messages, **options):
        event = options.get("stop_thinking")
        self.events.append(event)
        yield from super().stream_messages(messages, **options)


class ServiceTests(unittest.TestCase):
    def test_route_parser(self) -> None:
        self.assertEqual(_stop_thinking_target("/v1/chat/completions/chatcmpl-ab12/stop_thinking"), "chatcmpl-ab12")
        self.assertEqual(_stop_thinking_target("/v1/messages/msg_ab12/stop_thinking"), "msg_ab12")
        self.assertIsNone(_stop_thinking_target("/v1/chat/completions"))
        self.assertIsNone(_stop_thinking_target("/v1/responses/resp_1/stop_thinking"))
        self.assertIsNone(_stop_thinking_target("/v1/messages/../x/stop_thinking"))

    def test_capability_follows_the_generator(self) -> None:
        plain = InferenceService("m", StubGenerator())
        self.assertNotIn("stop_thinking", plain.properties()["capabilities"])
        with self.assertRaises(APIError) as caught:
            plain.stop_thinking("chatcmpl-x")
        self.assertEqual(caught.exception.status, 501)
        able = InferenceService("m", InterruptibleStub())
        self.assertIn("stop_thinking", able.properties()["capabilities"])
        with self.assertRaises(APIError) as caught:
            able.stop_thinking("chatcmpl-unknown")
        self.assertEqual(caught.exception.status, 404)

    def test_chat_stream_id_reaches_the_generator_event(self) -> None:
        generator = InterruptibleStub()
        service = InferenceService("m", generator)
        stream = service.stream_chat_completion(
            {"messages": [{"role": "user", "content": "Hi"}], "stream": True}
        )
        first = next(stream)
        self.assertIsInstance(first, dict)
        completion_id = first["id"]
        self.assertTrue(completion_id.startswith("chatcmpl-"))
        # The generator has not been entered yet; pull one token.
        next(stream)
        self.assertEqual(len(generator.events), 1)
        self.assertFalse(generator.events[0].is_set())
        result = service.stop_thinking(completion_id)
        self.assertEqual(result["status"], "requested")
        self.assertTrue(generator.events[0].is_set())
        for _ in stream:
            pass
        # The registry is cleared once the stream ends.
        with self.assertRaises(APIError):
            service.stop_thinking(completion_id)

    def test_anthropic_stream_id_reaches_the_generator_event(self) -> None:
        generator = InterruptibleStub()
        service = InferenceService("m", generator)
        stream = service.stream_anthropic_message(
            {"model": "m", "max_tokens": 8, "stream": True,
             "messages": [{"role": "user", "content": "Hi"}]}
        )
        start = next(stream)
        self.assertEqual(start["type"], "message_start")
        message_id = start["message"]["id"]
        # Drain up to the first text delta so the generator is live.
        for event in stream:
            if event.get("type") == "content_block_delta":
                break
        service.stop_thinking(message_id)
        self.assertTrue(generator.events[0].is_set())
        for _ in stream:
            pass


class RouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = InferenceService("m", InterruptibleStub())
        self.server = FlyweightHTTPServer(("127.0.0.1", 0), create_handler(self.service))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)

    def tearDown(self) -> None:
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_unknown_stream_is_a_json_404(self) -> None:
        self.connection.request(
            "POST", "/v1/chat/completions/chatcmpl-nope/stop_thinking",
            body="{}", headers={"Content-Type": "application/json"},
        )
        response = self.connection.getresponse()
        payload = json.loads(response.read())
        self.assertEqual(response.status, 404)
        self.assertEqual(payload["error"]["type"], "not_found_error")


# ---- The generator side: a fake engine streams a think block that never
# closes on its own, and the interrupt forces it shut.

class _Tokenizer:
    """One token per character, so the forced close is encode("</think>")."""

    eos_token_ids = (0,)
    architecture = "qwen3.5"

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, tokens, *, skip_special_tokens=True):
        return "".join(chr(t) for t in tokens)

    def token_bytes(self, token):
        return chr(token).encode("utf-8")


class _Engine:
    """Streams a fixed script per submit and records every submit/cancel."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.submits = []
        self.cancelled = []
        self.next_id = 1

    def submit(self, prompt_ids, max_new_tokens, stop_tokens, sampling=None, **kw):
        task_id = self.next_id
        self.next_id += 1
        self.submits.append((task_id, list(prompt_ids), max_new_tokens))
        queue: Queue = Queue()
        script = self.scripts.pop(0) if self.scripts else ""
        for char in script[:max_new_tokens]:
            queue.put(("token", ord(char)))
        queue.put(("token", 0))  # eos
        queue.put(("done", None))
        return task_id, queue

    def cancel(self, task_id):
        self.cancelled.append(task_id)

    def forget(self, task_id):
        pass

    def task_is_live(self, task_id):
        return True


class GeneratorInterruptTests(unittest.TestCase):
    def test_interrupt_closes_the_block_and_resumes_the_answer(self) -> None:
        # First task thinks forever; after the forced close the second task
        # (prompt = everything so far + close) yields the answer.
        engine = _Engine(["<think>" + "x" * 500, "The answer."])
        generator = ChatGenerator(None, engine, _Tokenizer())
        interrupt = threading.Event()
        pieces = []
        tripped = False
        for step in generator.stream_text("Q:", max_new_tokens=600, stop_thinking=interrupt):
            pieces.append(step.text_delta)
            # Trip the interrupt once, a few tokens into the block.
            if not tripped and "".join(pieces).count("x") == 5:
                tripped = True
                interrupt.set()
        text = "".join(pieces)
        self.assertIn(THINKING_BUDGET_CLOSE, text)
        self.assertLess(text.index(THINKING_BUDGET_CLOSE), 40)
        self.assertTrue(text.endswith("The answer."))
        self.assertEqual(engine.cancelled, [1])
        self.assertEqual(len(engine.submits), 2)
        # The continuation prompt carries the decoded prefix and the close.
        continuation = "".join(chr(t) for t in engine.submits[1][1])
        self.assertTrue(continuation.startswith("Q:<think>xxxxx"))
        self.assertIn(THINKING_BUDGET_CLOSE, continuation)
        # One interrupt closes one block; the generator consumed it.
        self.assertFalse(interrupt.is_set())

    def test_interrupt_outside_a_block_is_ignored(self) -> None:
        engine = _Engine(["Plain answer"])
        generator = ChatGenerator(None, engine, _Tokenizer())
        interrupt = threading.Event()
        interrupt.set()
        text = "".join(
            step.text_delta for step in generator.stream_text("Q:", max_new_tokens=20, stop_thinking=interrupt)
        )
        self.assertEqual(text, "Plain answer")
        self.assertEqual(engine.cancelled, [])
        self.assertEqual(len(engine.submits), 1)


if __name__ == "__main__":
    unittest.main()
