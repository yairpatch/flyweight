import http.client
import json
import threading
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import Mock, patch

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from colibri_next.cli import main
from colibri_next.generation import GenerationResult, GenerationStep
from colibri_next.server import (
    APIError,
    ColibriHTTPServer,
    InferenceService,
    create_handler,
)


class StubTokenizer:
    eos_token_ids = (99,)

    def encode(self, text):
        return [ord(character) for character in text]

    def encode_messages(self, messages, *, enable_thinking=False):
        return self.encode("".join(message["content"] for message in messages))

    def decode(self, tokens, *, skip_special_tokens=True):
        return "".join(chr(token) for token in tokens)

class StubGenerator:
    def __init__(self) -> None:
        self.calls = []
        self.tokenizer = StubTokenizer()

    def generate_messages(self, messages, **options) -> GenerationResult:
        self.calls.append((messages, options))
        return GenerationResult(
            prompt_ids=(1, 2, 3),
            generated_ids=(4, 5),
            text="Hello!",
            stopped_on_eos=True,
            state_tokens=4,
        )

    def generate_text(self, prompt, **options) -> GenerationResult:
        return self.generate_messages(
            [{"role": "user", "content": prompt}], **options
        )

    def stream_text(self, prompt, **options):
        return self.stream_messages(
            [{"role": "user", "content": prompt}], **options
        )
    def stream_messages(self, messages, **options):
        self.calls.append((messages, options))
        yield GenerationStep(
            token_id=4,
            text_delta="Hello",
            prompt_ids=(1, 2, 3),
            generated_ids=(4,),
            text="Hello",
            stopped_on_eos=False,
            finished=False,
            state_tokens=3,
        )
        yield GenerationStep(
            token_id=5,
            text_delta="!",
            prompt_ids=(1, 2, 3),
            generated_ids=(4, 5),
            text="Hello!",
            stopped_on_eos=True,
            finished=False,
            state_tokens=4,
        )
        yield GenerationStep(
            token_id=None,
            text_delta="",
            prompt_ids=(1, 2, 3),
            generated_ids=(4, 5),
            text="Hello!",
            stopped_on_eos=True,
            finished=True,
            state_tokens=4,
        )


class ToolStubGenerator(StubGenerator):
    def generate_messages(self, messages, **options) -> GenerationResult:
        self.calls.append((messages, options))
        return GenerationResult(
            prompt_ids=(1, 2, 3),
            generated_ids=(4, 5),
            text=(
                "<tool_call>\n<function=get_weather>\n"
                "<parameter=city>\nParis\n</parameter>\n"
                "</function>\n</tool_call>"
            ),
            stopped_on_eos=True,
            state_tokens=4,
        )

class InferenceServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = StubGenerator()
        self.service = InferenceService(
            "qwen-local", self.generator, max_new_tokens=32
        )

    def test_chat_completion_supports_developer_and_text_parts(self) -> None:
        response = self.service.chat_completion(
            {
                "model": "qwen-local",
                "messages": [
                    {"role": "developer", "content": "Be concise"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Con"},
                            {"type": "text", "text": "tinue"},
                        ],
                    },
                ],
                "max_tokens": 7,
                "temperature": 0,
            }
        )
        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["choices"][0]["message"]["content"], "Hello!")
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(response["usage"]["total_tokens"], 5)
        self.assertEqual(
            self.generator.calls[0][0],
            [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Continue"},
            ],
        )
        self.assertEqual(self.generator.calls[0][1]["max_new_tokens"], 7)

    def test_health_reports_cpu_moe_placement(self) -> None:
        service = InferenceService(
            "qwen-local", self.generator, cpu_moe_layers=12
        )
        self.assertEqual(service.health()["cpu_moe_layers"], 12)

    def test_context_window_limits_combined_prompt_and_output(self) -> None:
        service = InferenceService(
            "qwen-local",
            self.generator,
            max_new_tokens=10,
            context_window=10,
        )
        accepted = service.chat_completion(
            {
                "messages": [{"role": "user", "content": "123456"}],
                "max_tokens": 4,
            }
        )
        self.assertEqual(accepted["object"], "chat.completion")
        with self.assertRaisesRegex(APIError, "at most 4 output tokens"):
            service.chat_completion(
                {
                    "messages": [{"role": "user", "content": "123456"}],
                    "max_tokens": 5,
                }
            )
        self.assertEqual(service.properties()["context_window"], 10)

    def test_context_window_limits_legacy_text_completion(self) -> None:
        service = InferenceService(
            "qwen-local",
            self.generator,
            max_new_tokens=8,
            context_window=8,
        )
        with self.assertRaisesRegex(APIError, "at most 3 output tokens"):
            service.completion({"prompt": "12345", "max_tokens": 4})

    def test_chat_stream_emits_deltas_usage_and_done(self) -> None:
        events = list(
            self.service.stream_chat_completion(
                {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            )
        )
        self.assertEqual(events[0]["object"], "chat.completion.chunk")
        self.assertEqual(events[0]["choices"][0]["delta"]["role"], "assistant")
        deltas = [
            event["choices"][0]["delta"].get("content", "")
            for event in events
            if isinstance(event, dict) and event.get("choices")
        ]
        self.assertEqual("".join(deltas), "Hello!")
        self.assertEqual(events[-2]["usage"]["total_tokens"], 5)
        self.assertEqual(events[-1], "[DONE]")

    def test_responses_api_returns_output_text_shape(self) -> None:
        response = self.service.response(
            {
                "model": "qwen-local",
                "instructions": "Be concise",
                "input": "Say hi",
                "max_output_tokens": 4,
            }
        )
        self.assertEqual(response["object"], "response")
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["output"][0]["content"][0]["text"], "Hello!")
        self.assertEqual(response["usage"]["total_tokens"], 5)

    def test_function_calling_uses_native_qwen_format(self) -> None:
        service = InferenceService("qwen-local", ToolStubGenerator())
        payload = {
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        }
        response = service.chat_completion(payload)
        choice = response["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        call = choice["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "get_weather")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Paris"})

        stream_events = list(service.stream_chat_completion({**payload, "stream": True}))
        tool_chunks = [
            event
            for event in stream_events
            if isinstance(event, dict)
            and event.get("choices")
            and event["choices"][0]["delta"].get("tool_calls")
        ]
        self.assertEqual(
            tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "get_weather",
        )
    def test_responses_function_tools_stream_and_continue(self) -> None:
        service = InferenceService("qwen-local", ToolStubGenerator())
        tool = {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
        payload = {
            "input": "Weather in Paris?",
            "tools": [tool],
            "tool_choice": "required",
            "max_output_tokens": 8,
        }

        response = service.response(payload)
        call = response["output"][0]
        self.assertEqual(call["type"], "function_call")
        self.assertEqual(call["name"], "get_weather")
        self.assertEqual(json.loads(call["arguments"]), {"city": "Paris"})

        events = list(service.stream_response({**payload, "stream": True}))
        event_types = [event["type"] for event in events]
        self.assertIn("response.function_call_arguments.delta", event_types)
        self.assertIn("response.function_call_arguments.done", event_types)
        self.assertEqual(
            events[-1]["response"]["output"][0]["type"], "function_call"
        )

        service.response(
            {
                "previous_response_id": response["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": '{"temperature": 24}',
                    }
                ],
                "max_output_tokens": 8,
            }
        )
        continued_messages = service.generator.calls[-1][0]
        self.assertIn("<tool_call>", continued_messages[-2]["content"])
        self.assertIn("<tool_response>", continued_messages[-1]["content"])

    def test_previous_response_state_retrieve_and_delete(self) -> None:
        first = self.service.response({"input": "First", "max_output_tokens": 2})
        second = self.service.response(
            {
                "input": "Second",
                "previous_response_id": first["id"],
                "max_output_tokens": 2,
            }
        )
        second_messages = self.generator.calls[1][0]
        self.assertEqual(second["previous_response_id"], first["id"])
        self.assertEqual(second_messages[-2], {"role": "assistant", "content": "Hello!"})
        self.assertEqual(self.service.retrieve_response(first["id"])["id"], first["id"])
        deleted = self.service.delete_response(first["id"])
        self.assertTrue(deleted["deleted"])
        with self.assertRaises(APIError):
            self.service.retrieve_response(first["id"])
    def test_rejects_invalid_history_and_tools(self) -> None:
        with self.assertRaises(APIError):
            self.service.chat_completion(
                {"messages": [{"role": "assistant", "content": "Hi"}]}
            )
        with self.assertRaises(APIError) as tools_error:
            self.service.chat_completion(
                {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "tools": [{"type": "function"}],
                }
            )
        self.assertEqual(tools_error.exception.parameter, "tools")


class HTTPServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = StubGenerator()
        self.service = InferenceService("qwen-local", self.generator)
        self.server = ColibriHTTPServer(
            ("127.0.0.1", 0), create_handler(self.service)
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=5
        )

    def tearDown(self) -> None:
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def request_json(self, method: str, path: str, payload=None):
        body = None if payload is None else json.dumps(payload)
        headers = {} if payload is None else {"Content-Type": "application/json"}
        self.connection.request(method, path, body=body, headers=headers)
        response = self.connection.getresponse()
        return response, json.loads(response.read())

    def test_chat_ui_static_assets(self) -> None:
        self.connection.request("GET", "/")
        response = self.connection.getresponse()
        html = response.read().decode("utf-8")
        self.assertEqual(response.status, 200)
        self.assertTrue(response.getheader("Content-Type").startswith("text/html"))
        self.assertIn("default-src 'self'", response.getheader("Content-Security-Policy"))
        self.assertIn("Colibri Chat", html)

        self.connection.request("GET", "/app.js")
        response = self.connection.getresponse()
        javascript = response.read().decode("utf-8")
        self.assertEqual(response.status, 200)
        self.assertTrue(
            response.getheader("Content-Type").startswith("text/javascript")
        )
        self.assertIn('/v1/chat/completions', javascript)

        self.connection.request("GET", "/preview.html")
        response = self.connection.getresponse()
        preview = response.read().decode("utf-8")
        self.assertEqual(response.status, 200)
        self.assertTrue(response.getheader("Content-Type").startswith("text/html"))
        policy = response.getheader("Content-Security-Policy")
        self.assertIn("sandbox allow-scripts", policy)
        self.assertIn("frame-ancestors 'self'", policy)
        self.assertIn("location.hash", preview)

    def test_health_models_chat_and_responses_endpoints(self) -> None:
        response, health = self.request_json("GET", "/health")
        self.assertEqual(response.status, 200)
        self.assertEqual(health["status"], "ok")

        _, models = self.request_json("GET", "/v1/models")
        self.assertEqual(models["data"][0]["id"], "qwen-local")
        _, model = self.request_json("GET", "/v1/models/qwen-local")
        self.assertEqual(model["object"], "model")

        _, completion = self.request_json(
            "POST",
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 2},
        )
        self.assertEqual(completion["choices"][0]["message"]["role"], "assistant")

        _, response_object = self.request_json(
            "POST",
            "/v1/responses",
            {"model": "qwen-local", "input": "Hi", "max_output_tokens": 2},
        )
        self.assertEqual(response_object["object"], "response")
        _, tokenized = self.request_json("POST", "/tokenize", {"content": "Hi"})
        self.assertEqual(tokenized["count"], 2)
        _, token_count = self.request_json(
            "POST", "/v1/responses/input_tokens", {"input": "Hi"}
        )
        self.assertEqual(token_count["input_tokens"], 2)
        _, properties = self.request_json("GET", "/props")
        self.assertEqual(properties["max_output_tokens"], 64)

    def test_chat_stream_uses_sse_framing(self) -> None:
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        self.connection.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = self.connection.getresponse()
        stream_body = response.read().decode("utf-8")
        self.assertEqual(response.status, 200)
        self.assertTrue(response.getheader("Content-Type").startswith("text/event-stream"))
        self.assertIn('"object": "chat.completion.chunk"', stream_body)
        self.assertIn("data: [DONE]", stream_body)
class AuthenticationTests(unittest.TestCase):
    def test_bearer_auth_and_cors_preflight(self) -> None:
        server = ColibriHTTPServer(
            ("127.0.0.1", 0),
            create_handler(
                InferenceService(
                    "qwen-local",
                    StubGenerator(),
                    api_key="secret",
                    cors_origin="http://localhost:3000",
                )
            ),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=5
        )
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 200)

            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 401)
            self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")

            connection.request(
                "GET", "/v1/models", headers={"Authorization": "Bearer secret"}
            )
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 200)

            connection.request("OPTIONS", "/v1/chat/completions")
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 204)
            self.assertEqual(
                response.getheader("Access-Control-Allow-Origin"),
                "http://localhost:3000",
            )
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

@unittest.skipIf(OpenAI is None, "OpenAI SDK is not installed")
class OpenAISDKTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ColibriHTTPServer(
            ("127.0.0.1", 0),
            create_handler(InferenceService("qwen-local", StubGenerator())),
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = OpenAI(
            base_url=f"http://127.0.0.1:{self.server.server_port}/v1",
            api_key="local",
        )

    def tearDown(self) -> None:
        self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_chat_stream_and_responses_clients(self) -> None:
        chat = self.client.chat.completions.create(
            model="qwen-local",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=2,
        )
        self.assertEqual(chat.choices[0].message.content, "Hello!")

        stream = self.client.chat.completions.create(
            model="qwen-local",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=2,
            stream=True,
        )
        streamed_text = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in stream
            if chunk.choices
        )
        self.assertEqual(streamed_text, "Hello!")

        response = self.client.responses.create(
            model="qwen-local", input="Hi", max_output_tokens=2
        )
        self.assertEqual(response.output_text, "Hello!")
        response_stream = self.client.responses.create(
            model="qwen-local", input="Hi", max_output_tokens=2, stream=True
        )
        response_text = "".join(
            event.delta
            for event in response_stream
            if event.type == "response.output_text.delta"
        )
        self.assertEqual(response_text, "Hello!")

        completion = self.client.completions.create(
            model="qwen-local", prompt="Hi", max_tokens=2
        )
        self.assertEqual(completion.choices[0].text, "Hello!")

class ServerCLITests(unittest.TestCase):
    @patch("colibri_next.cli.serve_http")
    @patch("colibri_next.cli.InferenceService.from_model_directory")
    def test_serve_command_loads_once_and_starts_http(self, load_model, serve_http) -> None:
        load_model.return_value = Mock(model_name="demo-model")
        with redirect_stderr(StringIO()):
            result = main(
                [
                    "serve",
                    "model-root",
                    "--port",
                    "9012",
                    "--max-new-tokens",
                    "24",
                    "--cpu-moe-layers",
                    "10",
                ]
            )
        self.assertEqual(result, 0)
        load_model.assert_called_once()
        self.assertEqual(load_model.call_args.kwargs["cpu_moe_layers"], 10)
        serve_http.assert_called_once_with(
            load_model.return_value, host="127.0.0.1", port=9012
        )


if __name__ == "__main__":
    unittest.main()
