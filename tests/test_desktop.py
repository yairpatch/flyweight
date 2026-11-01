"""The desktop app's decisions, tested without a display.

The window is a thin skin over these: what a picked path means, which `serve`
command the settings become, which port is used, and what a failed log is
explained as. Getting any of them wrong is exactly the confusion the app
exists to remove, so they are pinned here.
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flyweight import desktop
from flyweight.cli import _parser


def _gguf(path: Path, size: int = 64) -> Path:
    path.write_bytes(b"GGUF" + b"\0" * (size - 4))
    return path


class ModelCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_empty_and_missing_paths_say_what_to_do(self) -> None:
        self.assertFalse(desktop.check_model("").ok)
        missing = desktop.check_model(str(self.root / "gone.gguf"))
        self.assertFalse(missing.ok)
        self.assertIn("moved", missing.message)

    def test_a_gguf_is_accepted_and_quotes_are_tolerated(self) -> None:
        model = _gguf(self.root / "m.gguf")
        check = desktop.check_model(f'"{model}"')
        self.assertTrue(check.ok)
        self.assertEqual((check.kind, check.path), ("gguf", str(model)))

    def test_a_truncated_download_is_refused(self) -> None:
        (self.root / "m.gguf").write_bytes(b"<html>")
        check = desktop.check_model(str(self.root / "m.gguf"))
        self.assertFalse(check.ok)
        self.assertIn("not a valid GGUF", check.message)

    def test_the_vision_projector_is_not_the_model_but_is_found_beside_it(self) -> None:
        projector = _gguf(self.root / "mmproj-F16.gguf")
        refused = desktop.check_model(str(projector))
        self.assertFalse(refused.ok)
        self.assertIn("projector", refused.message)
        model = _gguf(self.root / "model-Q4_K_M.gguf")
        accepted = desktop.check_model(str(model))
        self.assertTrue(accepted.ok)
        self.assertEqual(accepted.mmproj, str(projector))

    def test_a_later_split_part_resolves_to_the_first(self) -> None:
        first = _gguf(self.root / "big-00001-of-00003.gguf")
        second = _gguf(self.root / "big-00002-of-00003.gguf")
        check = desktop.check_model(str(second))
        self.assertTrue(check.ok)
        self.assertEqual(check.path, str(first))
        first.unlink()
        self.assertFalse(desktop.check_model(str(second)).ok)

    def test_a_folder_with_one_gguf_uses_it(self) -> None:
        model = _gguf(self.root / "only.gguf")
        _gguf(self.root / "mmproj.gguf")
        check = desktop.check_model(str(self.root))
        self.assertTrue(check.ok)
        self.assertEqual(check.path, str(model))

    def test_a_folder_with_several_ggufs_asks_for_one(self) -> None:
        _gguf(self.root / "a.gguf")
        _gguf(self.root / "b.gguf")
        self.assertFalse(desktop.check_model(str(self.root)).ok)

    def test_a_safetensors_folder_or_a_file_inside_it(self) -> None:
        (self.root / "config.json").write_text("{}")
        shard = self.root / "model.safetensors"
        shard.write_bytes(b"\0")
        for picked in (self.root, shard, self.root / "config.json"):
            check = desktop.check_model(str(picked))
            self.assertTrue(check.ok, picked)
            self.assertEqual((check.kind, check.path), ("safetensors", str(self.root)))

    def test_other_files_are_refused(self) -> None:
        other = self.root / "notes.txt"
        other.write_text("hi")
        self.assertFalse(desktop.check_model(str(other)).ok)


class ServeArgumentTests(unittest.TestCase):
    def test_defaults_are_a_plain_local_serve(self) -> None:
        model = desktop.ModelCheck(True, "gguf", "/m.gguf")
        args = desktop.serve_arguments(desktop.Settings(), model, 8001)
        self.assertEqual(args, ["serve", "/m.gguf", "--host", "127.0.0.1",
                                "--port", "8001", "--backend", "auto"])

    def test_every_setting_reaches_the_command_and_the_parser_accepts_it(self) -> None:
        settings = desktop.Settings(
            backend="cpu", context=16384, share_on_network=True,
            expert_mode="cpu", quant="Q4_K", cpu_threads=6, api_key=" k ",
            extra_args="--parallel 2 --temperature 0.7",
        )
        model = desktop.ModelCheck(True, "safetensors", "/ckpt", mmproj="/ckpt/mmproj.gguf")
        args = desktop.serve_arguments(settings, model, 9000)
        parsed = _parser().parse_args(args)
        self.assertEqual(parsed.host, "0.0.0.0")
        self.assertEqual(parsed.port, 9000)
        self.assertEqual(parsed.backend, "cpu")
        self.assertEqual(parsed.context_window, 16384)
        self.assertEqual(parsed.quant, "Q4_K")
        self.assertEqual(parsed.expert_mode, "cpu")
        self.assertEqual(parsed.cpu_threads, 6)
        self.assertEqual(parsed.api_key, "k")
        self.assertEqual(str(parsed.mmproj), "/ckpt/mmproj.gguf")
        self.assertEqual(parsed.parallel_sequences, 2)

    def test_a_gguf_never_gets_a_quant_and_the_projector_can_be_declined(self) -> None:
        model = desktop.ModelCheck(True, "gguf", "/m.gguf", mmproj="/mmproj.gguf")
        args = desktop.serve_arguments(desktop.Settings(use_mmproj=False), model, 8000)
        self.assertNotIn("--quant", args)
        self.assertNotIn("--mmproj", args)

    def test_the_serve_command_is_run_under_this_interpreter(self) -> None:
        command = desktop.engine_command()
        self.assertEqual(command[1:], ["-m", "flyweight"])
        with patch.object(desktop.sys, "frozen", True, create=True):
            self.assertEqual(desktop.engine_command()[1:], ["--engine"])


class PortTests(unittest.TestCase):
    def test_a_busy_port_is_skipped(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", 0))
            held.listen()
            busy = held.getsockname()[1]
            self.assertFalse(desktop.port_is_free(busy))
            chosen = desktop.pick_port(busy)
            self.assertNotEqual(chosen, busy)
            self.assertTrue(desktop.port_is_free(chosen))


class SettingsTests(unittest.TestCase):
    def test_round_trip_and_a_damaged_file_falls_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sub" / "desktop.json"
            settings = desktop.Settings(model="/m.gguf", port=8123)
            desktop.remember_model(settings, "/m.gguf")
            desktop.save_settings(settings, path)
            self.assertEqual(desktop.load_settings(path), settings)
            path.write_text(json.dumps({"port": "eighty", "context": True,
                                        "model": "/x.gguf", "unknown": 1}))
            loaded = desktop.load_settings(path)
            self.assertEqual((loaded.port, loaded.context, loaded.model),
                             (desktop.DEFAULT_PORT, 0, "/x.gguf"))
            path.write_text("{not json")
            self.assertEqual(desktop.load_settings(path), desktop.Settings())

    def test_recent_models_are_unique_newest_first_and_bounded(self) -> None:
        settings = desktop.Settings()
        for name in ["a", "b", "a", *map(str, range(10))]:
            desktop.remember_model(settings, name, keep=4)
        self.assertEqual(settings.recent_models, ["9", "8", "7", "6"])


class ExplanationTests(unittest.TestCase):
    def test_known_failures_get_a_specific_fix(self) -> None:
        cases = {
            "V2Error: CUDA_ERROR_OUT_OF_MEMORY while allocating": "does not fit",
            "OSError: [Errno 98] Address already in use": "port",
            "cannot find cuda_fp16.h under CUDA_PATH": "CUDA toolkit",
        }
        for log, expected in cases.items():
            self.assertIn(expected, desktop.explain_failure(log))
        self.assertIn("damaged", desktop.explain_failure(
            "[flyweight] running on the CPU backend\nflyweight: invalid GGUF string"))
        generic = desktop.explain_failure("loading\nflyweight: the frobnicator refused\n")
        self.assertIn("the frobnicator refused", generic)

    def test_the_doctor_report_is_read_line_by_line(self) -> None:
        report = desktop.parse_doctor(
            "[ok  ] python: 3.12.3\n"
            "[FAIL] native runtime: not built\n"
            "       -> python -m flyweight.native_build\n"
            "[warn] nvidia gpu: not probed\n\n1 blocking problem above\n"
        )
        self.assertEqual([line.status for line in report], ["ok", "fail", "warn"])
        self.assertEqual(report[1].fix, "python -m flyweight.native_build")


class ShortcutTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "the .desktop entry is the Linux shortcut")
    def test_install_and_uninstall_a_menu_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"XDG_DATA_HOME": directory}):
                written = desktop.install_shortcut()
                self.assertEqual(len(written), 1)
                entry = written[0].read_text()
                self.assertIn("Exec=", entry)
                self.assertIn("-m flyweight app", entry)
                self.assertIn("Terminal=false", entry)
                self.assertEqual(desktop.uninstall_shortcut(), written)
                self.assertFalse(written[0].exists())

    @unittest.skipIf(os.name == "nt", "AppImages are Linux")
    def test_an_appimage_keeps_a_menu_entry_pointing_at_itself(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "Downloads", "Flyweight.AppImage")
            with patch.dict(os.environ, {"XDG_DATA_HOME": directory, "APPIMAGE": first}):
                self.assertTrue(desktop.refresh_appimage_entry())
                entry = Path(directory, "applications", "flyweight.desktop").read_text()
                self.assertIn(f"Exec={first}", entry)
                self.assertTrue(Path(directory, "icons/hicolor/scalable/apps/flyweight.svg").is_file())
                self.assertFalse(desktop.refresh_appimage_entry())  # unchanged: left alone
            moved = os.path.join(directory, "Apps", "Flyweight.AppImage")
            with patch.dict(os.environ, {"XDG_DATA_HOME": directory, "APPIMAGE": moved}):
                self.assertTrue(desktop.refresh_appimage_entry())
                self.assertIn(f"Exec={moved}", Path(directory, "applications", "flyweight.desktop").read_text())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("APPIMAGE", None)
            self.assertFalse(desktop.refresh_appimage_entry())

    def test_the_cli_exposes_the_app(self) -> None:
        args = _parser().parse_args(["app", "--install-shortcut"])
        self.assertTrue(args.install_shortcut)
        with patch.object(desktop, "install_shortcut", return_value=[]) as install:
            self.assertEqual(desktop.main(["--install-shortcut"]), 0)
            install.assert_called_once()


class LocalApiTests(unittest.TestCase):
    """The page's backend: who may call it, and what the verbs do."""

    def setUp(self) -> None:
        import threading

        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        self.model = _gguf(self.root / "m.gguf")
        self.controller = desktop.Controller(settings_path=self.root / "desktop.json")
        self.picked: list[tuple[str, str]] = []

        def picker(kind: str, start: str) -> str | None:
            self.picked.append((kind, start))
            return str(self.model)

        self.server = desktop.make_server(self.controller, "secret-token", picker)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._dir.cleanup()

    def call(self, path: str, body: dict | None = None, token: str = "secret-token",
             host: str | None = None) -> tuple[int, object]:
        import http.client

        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"X-Flyweight-Token": token, "Content-Type": "application/json"}
        if host:
            headers["Host"] = host
        payload = None if body is None else json.dumps(body)
        connection.request("GET" if body is None else "POST", path, payload, headers)
        response = connection.getresponse()
        raw = response.read()
        connection.close()
        try:
            return response.status, json.loads(raw)
        except ValueError:
            return response.status, raw

    def test_the_page_is_served_with_a_strict_policy(self) -> None:
        import http.client

        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.request("GET", "/")
        response = connection.getresponse()
        html = response.read().decode()
        self.assertEqual(response.status, 200)
        self.assertIn("frame-ancestors 'none'", response.getheader("Content-Security-Policy"))
        self.assertIn('src="/app.js"', html)
        self.assertNotRegex(html, r"<script(?![^>]*\bsrc=)")
        for asset in ("/app.js", "/app.css", "/icon.svg"):
            self.assertEqual(self.call(asset)[0], 200, asset)

    def test_the_api_needs_the_token_and_a_loopback_host(self) -> None:
        self.assertEqual(self.call("/api/state", token="wrong")[0], 403)
        self.assertEqual(self.call("/api/start", {}, token="")[0], 403)
        self.assertEqual(self.call("/api/state", host="evil.example:80")[0], 421)
        status, state = self.call("/api/state")
        self.assertEqual(status, 200)
        assert isinstance(state, dict)
        self.assertEqual(state["status"], "idle")

    def test_settings_are_validated_saved_and_reflected(self) -> None:
        status, state = self.call("/api/settings", {
            "model": str(self.model), "backend": "cpu", "port": "not a port",
            "recent_models": ["/injected"]})
        self.assertEqual(status, 200)
        assert isinstance(state, dict)
        self.assertEqual(state["settings"]["backend"], "cpu")
        self.assertEqual(state["settings"]["port"], desktop.DEFAULT_PORT)
        self.assertEqual(state["settings"]["recent_models"], [])
        self.assertTrue(state["model"]["ok"])
        saved = json.loads((self.root / "desktop.json").read_text())
        self.assertEqual(saved["model"], str(self.model))

    def test_browse_sets_the_model_and_a_missing_dialog_is_explained(self) -> None:
        status, result = self.call("/api/browse", {"kind": "folder"})
        self.assertEqual((status, result["path"]), (200, str(self.model)))  # type: ignore[index]
        self.assertEqual(self.picked[0][0], "folder")
        self.assertEqual(self.controller.settings.model, str(self.model))

        def no_dialog(kind: str, start: str) -> str | None:
            raise LookupError

        self.server.shutdown()
        self.server.server_close()
        self.server = desktop.make_server(self.controller, "secret-token", no_dialog)
        self.port = self.server.server_address[1]
        import threading
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        _, result = self.call("/api/browse", {"kind": "file"})
        self.assertFalse(result["ok"])  # type: ignore[index]
        self.assertIn("Paste", result["error"])  # type: ignore[index]

    def test_start_refuses_a_bad_model_without_spawning(self) -> None:
        self.controller.update_settings({"model": str(self.root / "missing.gguf")})
        with patch.object(desktop, "Engine") as engine:
            _, result = self.call("/api/start", {})
        engine.assert_not_called()
        self.assertFalse(result["ok"])  # type: ignore[index]
        self.assertIn("moved", result["error"])  # type: ignore[index]

    def test_start_runs_serve_and_an_exit_is_explained(self) -> None:
        self.controller.update_settings({"model": str(self.model), "backend": "cpu"})
        self.controller.frame_origin = "http://127.0.0.1:1234"
        with patch.object(desktop, "Engine") as engine, \
                patch.object(desktop.Controller, "_wait_ready"):
            _, result = self.call("/api/start", {})
        self.assertTrue(result["ok"])  # type: ignore[index]
        arguments = engine.call_args.args[0]
        self.assertEqual(arguments[:2], ["serve", str(self.model)])
        self.assertEqual(engine.call_args.kwargs["env"],
                         {"FLYWEIGHT_UI_FRAME_ANCESTOR": "http://127.0.0.1:1234"})
        self.assertEqual(self.controller.status, "loading")
        self.assertEqual(self.controller.settings.recent_models, [str(self.model)])
        engine.return_value.stopping = False
        self.controller.put(("line", "flyweight: CUDA_ERROR_OUT_OF_MEMORY"))
        self.controller.put(("exit", 1))
        self.assertEqual(self.controller.status, "failed")
        self.assertIn("does not fit", self.controller.hint)
        _, state = self.call("/api/state?since=0")
        self.assertTrue(any("OUT_OF_MEMORY" in line["text"] for line in state["logs"]))  # type: ignore[index]

    def test_gpu_setup_is_offered_only_when_needed(self) -> None:
        self.controller.gpu = {"state": "ready"}
        _, result = self.call("/api/gpu/setup", {})
        self.assertFalse(result["ok"])  # type: ignore[index]
        self.controller.gpu = {"state": "needs-setup", "major": 12}
        with patch.object(self.controller.gpu_installer, "start") as start:
            _, result = self.call("/api/gpu/setup", {})
        self.assertTrue(result["ok"])  # type: ignore[index]
        self.assertEqual(start.call_args.args[0], 12)
        _, state = self.call("/api/state")
        self.assertIn("gpuSetup", state)  # type: ignore[operator]

    def test_the_doctor_report_reaches_the_page(self) -> None:
        self.controller.put(("doctor-line", "[FAIL] native runtime: not built"))
        self.controller.put(("doctor-line", "       -> python -m flyweight.native_build"))
        self.controller.put(("doctor-exit", 1))
        _, state = self.call("/api/state")
        doctor = state["doctor"]  # type: ignore[index]
        self.assertTrue(doctor["ran"])
        self.assertEqual(doctor["lines"][0]["fix"], "python -m flyweight.native_build")


class FramingTests(unittest.TestCase):
    def test_only_an_exact_loopback_origin_may_frame_the_chat(self) -> None:
        from flyweight import server

        with patch.dict(os.environ, {"FLYWEIGHT_UI_FRAME_ANCESTOR": "http://127.0.0.1:4321"}):
            self.assertIn("frame-ancestors http://127.0.0.1:4321", server._ui_csp())
        for hostile in ("*", "https://evil.example", "http://127.0.0.1:1; script-src *", ""):
            with patch.dict(os.environ, {"FLYWEIGHT_UI_FRAME_ANCESTOR": hostile}):
                self.assertEqual(server._ui_csp(), server.UI_CSP)


if __name__ == "__main__":
    unittest.main()
