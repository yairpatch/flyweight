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
import time
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
    def test_an_empty_profile_is_a_plain_serve_on_the_chosen_port(self) -> None:
        model = desktop.ModelCheck(True, "gguf", "/m.gguf")
        args = desktop.serve_arguments(desktop.Settings(), model, 8001)
        self.assertEqual(args, ["serve", "/m.gguf", "--port", "8001"])

    def test_every_profile_value_reaches_the_command_and_the_parser_accepts_it(self) -> None:
        settings = desktop.Settings(model="/ckpt", profiles={"/ckpt": {
            "backend": "cpu", "context_window": 16384, "host": "0.0.0.0",
            "expert_mode": "cpu", "quant": "Q4_K", "cpu_threads": 6, "api_key": "k",
            "temperature": 0.7, "cache_type_k": "q8_0", "routed_moe": True,
            "parallel_sequences": 2, desktop.EXTRA_KEY: "--verbose"}})
        model = desktop.ModelCheck(True, "safetensors", "/ckpt", mmproj="/ckpt/mmproj.gguf")
        args = desktop.serve_arguments(settings, model, 9000)
        parsed = _parser().parse_args(args)
        expected = {"host": "0.0.0.0", "port": 9000, "backend": "cpu", "context_window": 16384,
                    "quant": "Q4_K", "expert_mode": "cpu", "cpu_threads": 6, "api_key": "k",
                    "temperature": 0.7, "cache_type_k": "q8_0", "routed_moe": True,
                    "parallel_sequences": 2, "verbose": True}
        for dest, value in expected.items():
            self.assertEqual(getattr(parsed, dest), value, dest)
        self.assertEqual(str(parsed.mmproj), "/ckpt/mmproj.gguf")

    def test_safetensors_always_gets_a_quant_and_the_projector_can_be_declined(self) -> None:
        folder = desktop.ModelCheck(True, "safetensors", "/ckpt")
        self.assertIn("Q6_K", desktop.serve_arguments(desktop.Settings(), folder, 8000))
        gguf = desktop.ModelCheck(True, "gguf", "/m.gguf", mmproj="/mmproj.gguf")
        declined = desktop.Settings(profiles={"": {desktop.AUTO_MMPROJ_KEY: False}})
        args = desktop.serve_arguments(declined, gguf, 8000)
        self.assertNotIn("--quant", args)
        self.assertNotIn("--mmproj", args)

    def test_one_profile_drives_other_commands_skipping_what_they_lack(self) -> None:
        model = desktop.ModelCheck(True, "gguf", "/m.gguf")
        args = desktop.command_arguments("plan", {"context_window": 8192, "temperature": 0.5,
                                                  "cache_type_k": "q8_0"}, model)
        self.assertEqual(args, ["plan", "/m.gguf", "--context", "8192", "--cache-type-k", "q8_0"])
        with self.assertRaises(ValueError):
            desktop.command_arguments("serve", {"cpu_threads": "many"}, model)

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
            settings = desktop.Settings(model="/m.gguf", profiles={"/m.gguf": {"port": 8123}})
            desktop.remember_model(settings, "/m.gguf")
            desktop.save_settings(settings, path)
            self.assertEqual(desktop.load_settings(path), settings)
            path.write_text(json.dumps({"profiles": {"/m": {"bad": [1]}}, "open_chat": "yes",
                                        "model": "/x.gguf", "unknown": 1}))
            loaded = desktop.load_settings(path)
            self.assertEqual((loaded.profiles, loaded.open_chat, loaded.model),
                             ({}, True, "/x.gguf"))
            path.write_text("{not json")
            self.assertEqual(desktop.load_settings(path), desktop.Settings())

    def test_a_settings_file_from_before_profiles_becomes_the_default_profile(self) -> None:
        old = {"model": "/m.gguf", "backend": "cpu", "context": 65536, "share_on_network": True,
               "use_mmproj": False, "cpu_threads": 0, "extra_args": "--parallel 2", "port": 8100}
        settings = desktop.Settings.from_dict(old)
        self.assertEqual(settings.profile(), {
            "backend": "cpu", "context_window": 65536, "port": 8100, "host": "0.0.0.0",
            desktop.AUTO_MMPROJ_KEY: False, desktop.EXTRA_KEY: "--parallel 2"})
        self.assertEqual(settings.profile("/other.gguf"), settings.profile())

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


@unittest.skipIf(os.name == "nt", "the fake helpers are shell scripts")
class FilePickerTests(unittest.TestCase):
    """A helper that cannot run must hand over to the next one, never pass for a cancel."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def helper(self, name: str, body: str) -> list[str]:
        path = self.root / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)
        return [str(path)]

    def pick(self, *helpers: list[str]) -> str | None:
        with patch.object(desktop, "_picker_commands",
                          return_value=[(h, {}) for h in helpers]):
            return desktop.pick_path("file")

    def test_a_broken_helper_falls_through_to_the_next(self) -> None:
        broken = self.helper("zenity", "echo 'cannot open display: :0' >&2; exit 1")
        works = self.helper("kdialog", "echo /models/m.gguf")
        self.assertEqual(self.pick(broken, works), "/models/m.gguf")

    def test_each_helpers_own_cancel_is_a_cancel(self) -> None:
        later = self.helper("python3", "echo /should/not/run")
        self.assertIsNone(self.pick(self.helper("zenity", "exit 1"), later))
        self.assertIsNone(self.pick(self.helper("powershell", "exit 0"), later))

    def test_a_script_error_is_not_a_cancel(self) -> None:
        # A PowerShell failure used to read as "cancelled": nothing happened.
        failing = self.helper("powershell", "echo 'Exception calling ShowDialog' >&2; exit 1")
        tk_missing = self.helper("python3", "exit 2")
        with self.assertRaises(LookupError) as raised:
            self.pick(failing, tk_missing)
        self.assertIn("ShowDialog", str(raised.exception))
        self.assertIn("exit code 2", str(raised.exception))


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
            "model": str(self.model), "open_chat": "no", "recent_models": ["/injected"]})
        self.assertEqual(status, 200)
        assert isinstance(state, dict)
        self.assertTrue(state["settings"]["open_chat"])
        self.assertEqual(state["settings"]["recent_models"], [])
        self.assertTrue(state["model"]["ok"])
        saved = json.loads((self.root / "desktop.json").read_text())
        self.assertEqual(saved["model"], str(self.model))

    def test_profiles_keep_only_real_non_default_options(self) -> None:
        self.controller.update_settings({"model": str(self.model)})
        _, result = self.call("/api/profile", {"values": {
            "temperature": 0.6, "port": 8000, "bogus": 1, "model": "/x", "mmproj": "",
            desktop.EXTRA_KEY: "--verbose"}})
        self.assertEqual(result["profile"], {"temperature": 0.6, desktop.EXTRA_KEY: "--verbose"})  # type: ignore[index]
        _, state = self.call("/api/state")
        self.assertEqual(state["profile"]["temperature"], 0.6)  # type: ignore[index]
        self.assertTrue(state["ownProfile"])  # type: ignore[index]
        self.call("/api/profile", {"reset": True})
        _, state = self.call("/api/state")
        self.assertFalse(state["ownProfile"])  # type: ignore[index]

    def test_a_pasted_command_line_is_imported(self) -> None:
        _, result = self.call("/api/import", {"text":
            "flyweight serve moe.gguf --expert-mode cpu --cache-type-k q8_0 --context 65536"})
        self.assertEqual(result["model"], "moe.gguf")  # type: ignore[index]
        self.assertEqual(result["values"], {"expert_mode": "cpu", "cache_type_k": "q8_0",  # type: ignore[index]
                                            "context_window": 65536})
        _, bad = self.call("/api/import", {"text": "flyweight serve m.gguf --no-such-flag"})
        self.assertFalse(bad["ok"])  # type: ignore[index]
        self.assertIn("unrecognized", bad["error"])  # type: ignore[index]

    def test_the_schema_lists_every_serve_option(self) -> None:
        _, schemas = self.call("/api/schema")
        serve = schemas["serve"]  # type: ignore[index]
        dests = {o["dest"] for g in serve["groups"] for o in g["options"]}
        parser_dests = {a.dest for a in desktop.engine_options._subparser("serve")._actions}
        self.assertEqual(parser_dests - dests, {"help", "quiet", "verbose"})
        self.assertIn("benchmark", schemas)  # type: ignore[operator]

    def test_the_monitor_reads_the_request_table(self) -> None:
        self.controller.put(("line", "08:34:28  chat           71      0%   2.4s       8    43.1  length         2.5s"))
        _, monitor = self.call("/api/monitor")
        row = monitor["requests"][0]  # type: ignore[index]
        self.assertEqual((row["endpoint"], row["rate"], row["ttft"]), ("chat", 43.1, 2.4))
        self.assertIn("memory", monitor["system"])  # type: ignore[index]

    def test_browse_sets_the_model_and_a_missing_dialog_is_explained(self) -> None:
        status, result = self.call("/api/browse", {"kind": "folder"})
        self.assertEqual((status, result["path"]), (200, str(self.model)))  # type: ignore[index]
        self.assertEqual(self.picked[0][0], "folder")
        self.assertEqual(self.controller.settings.model, str(self.model))

        def no_dialog(kind: str, start: str) -> str | None:
            raise LookupError("zenity: cannot open display")

        self.server.shutdown()
        self.server.server_close()
        self.server = desktop.make_server(self.controller, "secret-token", no_dialog)
        self.port = self.server.server_address[1]
        import threading
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        _, result = self.call("/api/browse", {"kind": "file"})
        self.assertFalse(result["ok"])  # type: ignore[index]
        self.assertIn("Paste", result["error"])  # type: ignore[index]
        self.assertTrue(any("could not open" in text for _, text, _ in self.controller.logs))

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

    def test_a_catalog_download_ends_selected_on_home(self) -> None:
        from tests.hf_stub import HubStub, StubFile
        from flyweight import models

        hub = HubStub()
        hub.add("org/Tiny-GGUF", [StubFile("Tiny-Q8_0.gguf", b"GGUF" + b"\0" * 4096)])
        hub.start()
        try:
            with patch.dict(os.environ, {"HF_ENDPOINT": hub.url}):
                self.controller.update_settings({"models_dir": str(self.root / "models")})
                machine = models.Machine(ram=16 * 2**30, vram=0, gpu="", disk_free=100 * 2**30)
                self.controller.machine = machine
                self.controller.catalog = [models.resolve_entry(
                    {"id": "tiny", "name": "Tiny", "repo": "org/Tiny-GGUF", "quants": ["Q8_0"]},
                    machine)]
                _, result = self.call("/api/models/download", {"id": "tiny"})
                self.assertTrue(result["ok"])  # type: ignore[index]
                deadline = time.monotonic() + 20
                while not self.controller.settings.model and time.monotonic() < deadline:
                    time.sleep(0.05)
            expected = self.root / "models" / "org__Tiny-GGUF" / "Tiny-Q8_0.gguf"
            self.assertEqual(self.controller.settings.model, str(expected))
            self.assertTrue(desktop.check_model(str(expected)).ok)
            _, state = self.call("/api/state")
            self.assertEqual(state["models"]["library"][0]["name"], "Tiny")  # type: ignore[index]
            _, refused = self.call("/api/models/download", {"repo": "../x", "files": []})
            self.assertFalse(refused["ok"])  # type: ignore[index]
        finally:
            hub.stop()

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
