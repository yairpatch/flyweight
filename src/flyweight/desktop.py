"""Flyweight Desktop: a window that runs the engine for people who do not
want a terminal.

Everything the command line asks of a new user -- find the model file, know
which flags matter, notice that the port is taken, read a stack trace to learn
the CUDA toolkit is missing -- is done here with a file picker, a few plain
choices and a log that explains its own failures. The engine itself is the
same `flyweight serve`, run as a child process, so the window never holds the
GPU and a crash in the runtime cannot take the window down with it.

Linux and Windows are supported. The interface is a web page (desktop_ui/)
served on loopback and shown in a native window: pywebview when it is
installed (WebView2 on Windows), otherwise a Chromium-family browser in app
mode (Edge ships with every Windows; Chrome, Chromium, Brave or Edge on
Linux), otherwise an ordinary browser tab. Nothing beyond the standard
library is required.

    flyweight app                      open the window
    flyweight app --install-shortcut   add Flyweight to the applications menu
"""
from __future__ import annotations

import json
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Protocol

from . import gpu_setup, models

APP_NAME = "Flyweight"
DEFAULT_PORT = 8000
RELEASES_PAGE = "https://github.com/yairpatch/flyweight/releases/latest"
RELEASES_API = "https://api.github.com/repos/yairpatch/flyweight/releases/latest"
READY_TIMEOUT_SECONDS = 60 * 60  # a 200 GB MoE off a spinning disk is slow
_SPLIT_PART = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


# --------------------------------------------------------------------------
# Settings


@dataclass
class Settings:
    """What the window remembers between runs."""

    model: str = ""
    backend: str = "auto"
    context: int = 0
    port: int = DEFAULT_PORT
    share_on_network: bool = False
    open_chat: bool = True
    use_mmproj: bool = True
    expert_mode: str = "auto"
    quant: str = "Q6_K"
    cpu_threads: int = 0
    api_key: str = ""
    extra_args: str = ""
    recent_models: list[str] = field(default_factory=list)
    models_dir: str = ""        # where "Get a model" downloads to; "" = ~/Flyweight Models

    @classmethod
    def from_dict(cls, data: object) -> "Settings":
        """Read a saved file, keeping defaults for anything missing or wrong.

        A settings file that a newer or older version wrote, or one a person
        edited by hand, must never stop the window from opening.
        """
        settings = cls()
        if not isinstance(data, dict):
            return settings
        for spec in fields(cls):
            if spec.name not in data:
                continue
            value = data[spec.name]
            default = getattr(settings, spec.name)
            if isinstance(default, bool):
                ok = isinstance(value, bool)
            elif isinstance(default, int):
                ok = isinstance(value, int) and not isinstance(value, bool)
            elif isinstance(default, list):
                ok = isinstance(value, list) and all(isinstance(v, str) for v in value)
            else:
                ok = isinstance(value, str)
            if ok:
                setattr(settings, spec.name, value)
        if not 1 <= settings.port <= 65535:
            settings.port = DEFAULT_PORT
        return settings


def config_dir() -> Path:
    """Per-user settings directory: %APPDATA%\\flyweight or $XDG_CONFIG_HOME/flyweight."""
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "flyweight"


def load_settings(path: Path | None = None) -> Settings:
    path = path or config_dir() / "desktop.json"
    try:
        return Settings.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return Settings()


def save_settings(settings: Settings, path: Path | None = None) -> None:
    path = path or config_dir() / "desktop.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        pass  # an unwritable home costs the memory, not the session


def remember_model(settings: Settings, model: str, keep: int = 8) -> None:
    recent = [m for m in settings.recent_models if m != model]
    settings.recent_models = [model, *recent][:keep]


# --------------------------------------------------------------------------
# Model checks


@dataclass
class ModelCheck:
    """What a chosen path is, and whether `serve` can take it."""

    ok: bool
    kind: str = ""          # "gguf" or "safetensors"
    path: str = ""          # what to pass to serve (may differ from the pick)
    message: str = ""
    mmproj: str = ""        # a vision projector found beside the model


def _find_mmproj(directory: Path) -> str:
    try:
        candidates = sorted(p for p in directory.iterdir()
                            if p.is_file() and p.suffix.lower() == ".gguf"
                            and p.name.lower().startswith("mmproj"))
    except OSError:
        return ""
    return str(candidates[0]) if candidates else ""


def check_model(raw: str) -> ModelCheck:
    """Turn whatever the user picked into a path `serve` will accept.

    The mistakes this catches are the ones people actually make: picking the
    vision projector instead of the model, picking part 2 of a split GGUF,
    picking the folder a single GGUF sits in, or a path that moved.
    """
    text = raw.strip().strip('"').strip("'")
    if not text:
        return ModelCheck(False, message="Choose a model file (.gguf) or a "
                                         "safetensors model folder.")
    path = Path(text).expanduser()
    if not path.exists():
        return ModelCheck(False, message=f"Nothing exists at {path}. "
                                         "Was it moved or renamed?")
    if path.is_dir():
        try:
            entries = list(path.iterdir())
        except OSError as error:
            return ModelCheck(False, message=f"Cannot read {path}: {error}")
        names = {p.name.lower() for p in entries}
        if "config.json" in names and any(n.endswith(".safetensors") for n in names):
            return ModelCheck(True, "safetensors", str(path),
                              "Safetensors model. It is converted on the first "
                              "start (this can take a while), then cached.",
                              _find_mmproj(path))
        ggufs = sorted(p for p in entries if p.is_file()
                       and p.suffix.lower() == ".gguf"
                       and not p.name.lower().startswith("mmproj"))
        firsts = [p for p in ggufs if not _SPLIT_PART.search(p.name)
                  or _SPLIT_PART.search(p.name).group(1) == "00001"]  # type: ignore[union-attr]
        if len(firsts) == 1:
            return check_model(str(firsts[0]))
        if firsts:
            return ModelCheck(False, message="This folder holds several models; "
                                             "choose the .gguf file you want.")
        return ModelCheck(False, message="This folder has no .gguf file and is "
                                         "not a safetensors model (config.json "
                                         "plus *.safetensors).")
    if path.suffix.lower() != ".gguf":
        if path.suffix.lower() == ".safetensors" or path.name.lower() == "config.json":
            return check_model(str(path.parent))
        return ModelCheck(False, message=f"{path.name} is not a model. Choose a "
                                         ".gguf file or a safetensors folder.")
    if path.name.lower().startswith("mmproj"):
        return ModelCheck(False, message=f"{path.name} is the image projector "
                                         "(mmproj), not the language model. "
                                         "Choose the main .gguf beside it; the "
                                         "projector is picked up automatically.")
    split = _SPLIT_PART.search(path.name)
    if split and split.group(1) != "00001":
        first = path.with_name(_SPLIT_PART.sub(f"-00001-of-{split.group(2)}.gguf", path.name))
        if first.exists():
            return ModelCheck(True, "gguf", str(first),
                              f"Split model: using the first part, {first.name}.",
                              _find_mmproj(path.parent))
        return ModelCheck(False, message=f"{path.name} is part {int(split.group(1))} "
                                         "of a split model and part 1 is missing.")
    try:
        with path.open("rb") as handle:
            magic = handle.read(4)
    except OSError as error:
        return ModelCheck(False, message=f"Cannot read {path.name}: {error}")
    if magic != b"GGUF":
        return ModelCheck(False, message=f"{path.name} is not a valid GGUF file "
                                         "(an unfinished download?).")
    size = path.stat().st_size
    shown = f"{size / 2**30:.1f} GiB" if size >= 2**30 else f"{size / 2**20:.0f} MiB"
    return ModelCheck(True, "gguf", str(path), f"GGUF model, {shown}.",
                      _find_mmproj(path.parent))


# --------------------------------------------------------------------------
# Running the engine


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if os.name != "nt":
            # As the server binds: a port a stopped server left in TIME_WAIT
            # is free to it, and must not be reported busy. (On Windows the
            # option means "share a live port", so it is left off there.)
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def pick_port(preferred: int, host: str = "127.0.0.1", attempts: int = 50) -> int:
    """The preferred port if free, else the next free one above it."""
    for port in range(preferred, min(preferred + attempts, 65536)):
        if port_is_free(port, host):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


def engine_command() -> list[str]:
    """How to run `flyweight <command>` from here.

    A frozen (PyInstaller) build is its own interpreter, so it re-enters
    itself with --engine. A pip install runs `-m flyweight` under the console
    interpreter: pythonw.exe has no console to inherit and is swapped for the
    python.exe beside it.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, "--engine"]
    interpreter = Path(sys.executable)
    if os.name == "nt" and interpreter.name.lower() == "pythonw.exe":
        console = interpreter.with_name("python.exe")
        if console.exists():
            interpreter = console
    return [str(interpreter), "-m", "flyweight"]


def serve_arguments(settings: Settings, model: ModelCheck, port: int) -> list[str]:
    """The `serve` arguments for these settings, without the interpreter."""
    args = ["serve", model.path,
            "--host", "0.0.0.0" if settings.share_on_network else "127.0.0.1",
            "--port", str(port),
            "--backend", settings.backend]
    if settings.context:
        args += ["--context", str(settings.context)]
    if model.kind == "safetensors":
        # `ask` would wait on a terminal the window does not have.
        args += ["--quant", settings.quant]
    if settings.expert_mode != "auto":
        args += ["--expert-mode", settings.expert_mode]
    if settings.cpu_threads > 0:
        args += ["--cpu-threads", str(settings.cpu_threads)]
    if settings.api_key.strip():
        args += ["--api-key", settings.api_key.strip()]
    if settings.use_mmproj and model.mmproj:
        args += ["--mmproj", model.mmproj]
    if settings.extra_args.strip():
        args += shlex.split(settings.extra_args, posix=os.name != "nt")
    return args


def chat_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/"


def lan_address() -> str | None:
    """This machine's address on the local network, for the share hint."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("192.0.2.1", 9))  # TEST-NET; nothing is sent
            return probe.getsockname()[0]
        except OSError:
            return None


def server_answers(port: int, timeout: float = 1.0) -> bool:
    """True once anything answers HTTP on the port -- 401 included."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True
    except (OSError, ValueError):
        return False


# Log text -> what to do about it. First match wins; the patterns are the
# runtime's and the OS's own wording, lowercased.
_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("out of memory", "cuda_error_out_of_memory", "does not fit", "cannot hold",
      "hiperroroutofmemory", "not enough memory"),
     "The model does not fit in memory. Try a smaller context length, "
     "'Keep experts on the CPU' under Advanced, or a smaller quantization."),
    (("address already in use", "only one usage of each socket address"),
     "The port is taken by another program. Choose a different port."),
    (("cuda_fp16.h", "nvrtc", "cuda headers", "cuda toolkit"),
     "The NVIDIA CUDA toolkit is missing or incomplete. Install it (see "
     "'Check this computer'), or choose 'CPU only' as the device."),
    (("native library", "not built", "flyweight_native"),
     "The engine's native library is missing. Reinstall with "
     "'pip install --force-reinstall flyweight-llm'."),
    (("invalid gguf", "truncated", "unexpected end of file", "unexpected eof",
      "bad magic"),
     "The model file is damaged or incomplete. Download it again."),
    (("unsupported architecture", "unknown architecture", "not supported"),
     "This model's architecture is not supported by Flyweight. See the "
     "README for the supported families."),
    (("permission denied", "access is denied"),
     "Flyweight was not allowed to read a file. Check the model's permissions."),
    (("no module named",),
     "A Python package is missing. Reinstall with 'pip install flyweight-llm'."),
)


def last_error(log: str) -> str:
    """The engine's own last word: its `flyweight:` refusal, else the last
    line that names an error, else the last line."""
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("flyweight:"):
            return line[len("flyweight:"):].strip()
    for line in reversed(lines):
        if "error" in line.lower():
            return line
    return lines[-1] if lines else ""


def explain_failure(log: str) -> str:
    """One sentence saying what to do, for a log that ended in a failure."""
    lowered = log.lower()
    for needles, hint in _HINTS:
        if any(needle in lowered for needle in needles):
            return hint
    said = last_error(log)
    reason = f"It said: {said}. " if said else ""
    return (reason + "'Check this computer' may name a missing piece; the "
            "log has the details.")


@dataclass
class DoctorLine:
    status: str   # "ok", "warn", "fail"
    name: str
    detail: str
    fix: str = ""


def parse_doctor(output: str) -> list[DoctorLine]:
    """Read `flyweight doctor`'s report: `[ok  ] name: detail` / `-> fix`."""
    lines: list[DoctorLine] = []
    for raw in output.splitlines():
        match = re.match(r"\[(ok|warn|FAIL)\s*\]\s*([^:]+):\s*(.*)", raw.strip())
        if match:
            lines.append(DoctorLine(match.group(1).lower(), match.group(2).strip(),
                                    match.group(3).strip()))
        elif raw.strip().startswith("->") and lines:
            lines[-1].fix = raw.strip()[2:].strip()
    return lines


def _child_environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _popen_flags() -> dict[str, object]:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {"start_new_session": True}


class _Sink(Protocol):
    def put(self, item: tuple[str, object]) -> None: ...


class Engine:
    """One `flyweight` child process whose output arrives on a sink."""

    def __init__(self, arguments: list[str], output: _Sink, prefix: str = "",
                 env: dict[str, str] | None = None) -> None:
        self.command = engine_command() + arguments
        self.output = output
        self.prefix = prefix
        self.stopping = False
        self.was_ready = False
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env={**_child_environment(), **(env or {})},
            **_popen_flags(),  # type: ignore[arg-type]
        )
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.process.stdout is not None
        for raw in iter(self.process.stdout.readline, b""):
            self.output.put((self.prefix + "line",
                             raw.decode("utf-8", "replace").rstrip("\r\n")))
        self.output.put((self.prefix + "exit", self.process.wait()))

    def running(self) -> bool:
        return self.process.poll() is None

    def stop(self, grace: float = 10.0) -> None:
        """Ask nicely (SIGTERM shuts the server down cleanly), then insist."""
        self.stopping = True
        if not self.running():
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        except OSError:
            pass


# --------------------------------------------------------------------------
# Shortcuts


def _launch_target() -> tuple[str, list[str]]:
    """Executable and arguments a menu entry should run (no console window)."""
    if os.environ.get("APPIMAGE"):
        # Inside an AppImage sys.executable is in a temporary mount; the
        # AppImage file itself is what stays.
        return os.environ["APPIMAGE"], []
    if getattr(sys, "frozen", False):
        return sys.executable, []
    interpreter = Path(sys.executable)
    if os.name == "nt":
        windowless = interpreter.with_name("pythonw.exe")
        if windowless.exists():
            interpreter = windowless
    return str(interpreter), ["-m", "flyweight", "app"]


def _icon_path() -> Path:
    return Path(__file__).resolve().parent / "ui" / "favicon.svg"


def linux_desktop_entry() -> str:
    target, arguments = _launch_target()
    exec_line = " ".join(shlex.quote(part) for part in [target, *arguments])
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={APP_NAME}\n"
        "GenericName=Local AI chat\n"
        "Comment=Run AI models on this computer\n"
        f"Exec={exec_line}\n"
        "Icon=flyweight\n"
        "Terminal=false\n"
        "Categories=Utility;Development;\n"
        "StartupWMClass=Flyweight\n"
    )


def _shortcut_paths() -> list[Path]:
    if os.name == "nt":
        appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        start_menu = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        return [start_menu / f"{APP_NAME}.lnk", Path.home() / "Desktop" / f"{APP_NAME}.lnk"]
    data = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return [Path(data) / "applications" / "flyweight.desktop"]


def install_shortcut() -> list[Path]:
    """Add Flyweight to the applications menu (and the Windows desktop)."""
    written: list[Path] = []
    if os.name == "nt":
        target, arguments = _launch_target()
        for path in _shortcut_paths():
            if not path.parent.is_dir():
                continue
            script = (
                "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:FW_LINK);"
                "$s.TargetPath=$env:FW_TARGET;$s.Arguments=$env:FW_ARGS;"
                "$s.WorkingDirectory=$env:USERPROFILE;"
                "$s.Description='Run a local language model';$s.Save()"
            )
            env = dict(os.environ, FW_LINK=str(path), FW_TARGET=target,
                       FW_ARGS=subprocess.list2cmdline(arguments))
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                            "-Command", script], env=env, check=True,
                           capture_output=True, **_popen_flags())  # type: ignore[arg-type]
            written.append(path)
        return written
    path = _shortcut_paths()[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    # The icon is copied out of the package: an AppImage's own copy lives in
    # a mount that disappears when the app quits.
    icon = path.parent.parent / "icons" / "hicolor" / "scalable" / "apps" / "flyweight.svg"
    icon.parent.mkdir(parents=True, exist_ok=True)
    icon.write_bytes(_icon_path().read_bytes())
    path.write_text(linux_desktop_entry(), encoding="utf-8")
    path.chmod(0o755)
    written.append(path)
    return written


def refresh_appimage_entry() -> bool:
    """Running as an AppImage: keep a menu entry that opens this file.

    Downloaded AppImages are double-clicked from the Downloads folder and
    then never found again; the entry fixes that, and is rewritten if the
    file moved. Returns True when it wrote one.
    """
    if os.name == "nt" or not os.environ.get("APPIMAGE"):
        return False
    path = _shortcut_paths()[0]
    wanted = linux_desktop_entry()
    try:
        if path.read_text(encoding="utf-8") == wanted:
            return False
    except OSError:
        pass
    try:
        install_shortcut()
    except OSError:
        return False
    return True


def uninstall_shortcut() -> list[Path]:
    removed = []
    for path in _shortcut_paths():
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


# --------------------------------------------------------------------------
# The app: state, a local API, and a window that shows it


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    for distribution in ("flyweight-llm", "flyweight"):
        try:
            return version(distribution)
        except PackageNotFoundError:
            continue
    return "dev"


def _check_dict(check: ModelCheck) -> dict[str, object]:
    data: dict[str, object] = asdict(check)
    path = Path(check.path) if check.path else None
    data["name"] = path.name if path else ""
    data["folder"] = str(path.parent) if path else ""
    data["mmprojName"] = Path(check.mmproj).name if check.mmproj else ""
    return data


class Controller:
    """Everything the window shows, owned in one place and safe across threads.

    The page polls `snapshot()`; buttons call the verbs. The engine and the
    system check are child processes whose lines land here through `put`.
    """

    LOG_LIMIT = 4000

    def __init__(self, settings: Settings | None = None,
                 settings_path: Path | None = None) -> None:
        self.lock = threading.RLock()
        self.settings_path = settings_path
        self.settings = settings or load_settings(settings_path)
        self.status = "idle"            # idle, loading, running, stopping, failed
        self.engine: Engine | None = None
        self.port: int | None = None
        self.hint = ""
        self.started_at = 0.0
        self.ready_at = 0.0
        self.logs: list[tuple[int, str, str]] = []
        self.seq = 0
        self.engine_log: list[str] = []
        self.doctor_running = False
        self.doctor_raw: list[str] = []
        self.doctor: list[DoctorLine] = []
        self.doctor_ran = False
        self.frame_origin = ""          # the page's origin, allowed to frame the chat
        self.last_poll = 0.0
        self.quit_requested = threading.Event()
        self.first_run = not (settings_path or config_dir() / "desktop.json").exists()
        self.gpu: dict[str, object] | None = None    # gpu_setup.status(), cached
        self.gpu_installer = gpu_setup.Installer()
        self.update: dict[str, str] | None = None
        self.machine: models.Machine | None = None
        self.catalog: list[dict[str, object]] = []
        self.catalog_loading = False
        self.catalog_cache = (settings_path.parent if settings_path else config_dir()) \
            / "model-catalog.json"
        self.fetch_catalog = True
        self.notice = ""                 # a one-off message for the page to toast
        self._library: tuple[float, list[dict[str, object]]] = (0.0, [])
        self.downloads = models.Downloads(self.models_dir, self._download_done)

    # -- models ------------------------------------------------------------------

    def models_dir(self) -> Path:
        chosen = self.settings.models_dir.strip()
        return Path(chosen).expanduser() if chosen else models.default_models_dir()

    def refresh_catalog(self) -> None:
        """Measure this computer and look the catalog's files up (slow: network)."""
        with self.lock:
            if self.catalog_loading:
                return
            self.catalog_loading = True
        try:
            machine = models.machine(self.models_dir())
            entries = models.load_catalog(self.catalog_cache, fetch=self.fetch_catalog)
            resolved = [models.resolve_entry(entry, machine) for entry in entries]
            with self.lock:
                self.machine, self.catalog = machine, resolved
        finally:
            with self.lock:
                self.catalog_loading = False

    def library(self) -> list[dict[str, object]]:
        stamp, cached = self._library
        if time.monotonic() - stamp < 3:
            return cached
        found = models.library(self.models_dir())
        self._library = (time.monotonic(), found)
        return found

    def _download_done(self, job: "models.Job") -> None:
        self._library = (0.0, [])
        with self.lock:
            self.log(f"Downloaded {job.name}.", "ok")
            if self.engine is None:
                self.settings.model = job.result
                save_settings(self.settings, self.settings_path)
                self.notice = f"{job.name} is ready. Press Start on Home."
            else:
                self.notice = f"{job.name} is downloaded. Stop the current model to switch."

    def download(self, body: dict[str, object]) -> dict[str, object]:
        entry_id = body.get("id")
        if entry_id:
            with self.lock:
                entry = next((e for e in self.catalog if e.get("id") == entry_id), None)
            if not entry or not entry.get("available"):
                return {"ok": False, "error": "That model is not available right now."}
            repo, name, files = str(entry["repo"]), str(entry["name"]), entry["files"]
        else:
            repo, name, files = str(body.get("repo", "")), str(body.get("name", "")), body.get("files")
            if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or not isinstance(files, list) or not files:
                return {"ok": False, "error": "Nothing to download."}
        if not isinstance(files, list) or not all(isinstance(f, dict) for f in files):
            return {"ok": False, "error": "Nothing to download."}
        machine = self.machine or models.machine(self.models_dir())
        total = sum(int(f.get("size") or 0) for f in files)
        if machine.disk_free and total > machine.disk_free - 2**30:
            return {"ok": False, "error": f"Not enough disk space: this needs "
                                          f"{total / 2**30:.1f} GB in {self.models_dir()}."}
        try:
            job = self.downloads.add(repo, name or repo.split("/")[-1], files)  # type: ignore[arg-type]
        except ValueError as error:
            return {"ok": False, "error": str(error)}
        self.log(f"Downloading {job.name} from Hugging Face ({total / 2**30:.1f} GB).", "note")
        return {"ok": True, "job": job.id}

    def use_model(self, path: str) -> dict[str, object]:
        with self.lock:
            if self.engine is not None:
                return {"ok": False, "error": "Stop the running model first."}
        self.update_settings({"model": path})
        return {"ok": True}

    def delete_model(self, folder: str) -> dict[str, object]:
        with self.lock:
            if self.engine is not None and self.settings.model.startswith(folder):
                return {"ok": False, "error": "Stop the model before deleting it."}
        try:
            models.delete_model(self.models_dir(), folder)
        except (OSError, ValueError) as error:
            return {"ok": False, "error": str(error)}
        self._library = (0.0, [])
        with self.lock:
            if self.settings.model.startswith(folder):
                self.settings.model = ""
            self.settings.recent_models = [m for m in self.settings.recent_models
                                           if not m.startswith(folder)]
            save_settings(self.settings, self.settings_path)
        return {"ok": True}

    # -- background facts ----------------------------------------------------

    def refresh_gpu(self) -> None:
        """Detect NVIDIA support (slow on Windows, so off the request path)."""
        try:
            found = gpu_setup.status_dict()
        except Exception as error:  # noqa: BLE001 - detection must never break the app
            found = {"state": "none", "title": "GPU check failed", "detail": str(error)}
        with self.lock:
            self.gpu = found

    def setup_gpu(self) -> dict[str, object]:
        with self.lock:
            gpu = self.gpu or {}
        if gpu.get("state") != "needs-setup":
            return {"ok": False, "error": "GPU setup is not needed here."}

        def done() -> None:
            self.refresh_gpu()
            snapshot = self.gpu_installer.snapshot()
            if snapshot["error"]:
                self.log(str(snapshot["error"]), "error")
            else:
                self.log("GPU support installed. New starts use the GPU.", "ok")
                self.doctor_ran = False  # the old report is stale now

        self.log(f"Setting up GPU support (CUDA {gpu.get('major')})…", "note")
        self.gpu_installer.start(int(gpu.get("major") or 12), done)  # type: ignore[call-overload]
        return {"ok": True}

    def check_for_update(self) -> None:
        """Note a newer release, so the page can offer it. Silent on failure."""
        current = _version()
        if not re.fullmatch(r"\d+(\.\d+)*", current):
            return
        try:
            request = urllib.request.Request(RELEASES_API, headers={
                "Accept": "application/vnd.github+json", "User-Agent": "flyweight-desktop"})
            with urllib.request.urlopen(request, timeout=10) as response:
                release = json.load(response)
        except (OSError, ValueError):
            return
        tag = str(release.get("tag_name", "")).lstrip("v")
        if not re.fullmatch(r"\d+(\.\d+)*", tag):
            return
        if tuple(map(int, tag.split("."))) > tuple(map(int, current.split("."))):
            with self.lock:
                self.update = {"version": tag, "url": str(release.get("html_url") or RELEASES_PAGE)}

    # -- log ---------------------------------------------------------------

    def log(self, text: str, kind: str = "") -> None:
        with self.lock:
            self.seq += 1
            self.logs.append((self.seq, text, kind))
            del self.logs[:-self.LOG_LIMIT]

    def put(self, item: tuple[str, object]) -> None:
        """Engine and doctor output (see Engine)."""
        kind, payload = item
        with self.lock:
            if kind == "line":
                text = str(payload)
                self.engine_log.append(text)
                del self.engine_log[:-400]
                lowered = text.lower()
                tag = "error" if ("error" in lowered or "traceback" in lowered
                                  or text.startswith("flyweight:")) else ""
                self.log(text, tag)
            elif kind == "exit":
                self._engine_exited(int(payload))  # type: ignore[call-overload]
            elif kind == "doctor-line":
                text = str(payload)
                self.doctor_raw.append(text)
                stripped = text.strip()
                tag = ("error" if stripped.startswith("[FAIL") else
                       "warn" if stripped.startswith("[warn") else
                       "ok" if stripped.startswith("[ok") else "")
                self.log(text, tag)
            elif kind == "doctor-exit":
                self.doctor = parse_doctor("\n".join(self.doctor_raw))
                if not self.doctor and self.doctor_raw:
                    # doctor itself crashed: show its last words as a failure
                    self.doctor = [DoctorLine("fail", "check", last_error(
                        "\n".join(self.doctor_raw)) or "the check could not run")]
                self.doctor_running = False
                self.doctor_ran = True

    def _engine_exited(self, code: int) -> None:
        engine = self.engine
        self.engine = None
        if code == 0 or (engine is not None and engine.stopping):
            self.status = "idle"
            self.hint = ""
            self.log("Stopped.", "note")
            return
        self.status = "failed"
        self.hint = explain_failure("\n".join(self.engine_log))
        self.log(f"Engine exited with code {code}. {self.hint}", "error")

    # -- settings ----------------------------------------------------------

    def update_settings(self, changes: dict[str, object]) -> None:
        with self.lock:
            merged = {**asdict(self.settings), **changes}
            merged["recent_models"] = self.settings.recent_models  # not the page's to set
            self.settings = Settings.from_dict(merged)
            save_settings(self.settings, self.settings_path)

    def forget_model(self, model: str) -> None:
        with self.lock:
            self.settings.recent_models = [m for m in self.settings.recent_models if m != model]
            save_settings(self.settings, self.settings_path)

    # -- verbs -------------------------------------------------------------

    def start(self) -> dict[str, object]:
        with self.lock:
            if self.engine is not None:
                return {"ok": False, "error": "Already running."}
            check = check_model(self.settings.model)
            if not check.ok:
                return {"ok": False, "error": check.message}
            try:
                serve_arguments(self.settings, check, self.settings.port)
            except ValueError as error:
                return {"ok": False, "error": f"The extra flags could not be read: {error}"}
            bind = "0.0.0.0" if self.settings.share_on_network else "127.0.0.1"
            port = pick_port(self.settings.port, bind)
            if port != self.settings.port:
                self.log(f"Port {self.settings.port} is busy; using {port} instead.", "note")
            remember_model(self.settings, check.path)
            save_settings(self.settings, self.settings_path)
            arguments = serve_arguments(self.settings, check, port)
            secret = self.settings.api_key.strip()
            shown = " ".join("****" if secret and a == secret else shlex.quote(a)
                             for a in arguments)
            self.log("", "")
            self.log(f"$ flyweight {shown}", "note")
            env = gpu_setup.engine_environment()
            if self.frame_origin:
                env["FLYWEIGHT_UI_FRAME_ANCESTOR"] = self.frame_origin
            self.engine_log = []
            try:
                self.engine = Engine(arguments, self, env=env)
            except OSError as error:
                self.status = "failed"
                self.hint = f"Could not start the engine: {error}"
                return {"ok": False, "error": self.hint}
            self.port = port
            self.status = "loading"
            self.hint = ""
            self.started_at = time.time()
            self.ready_at = 0.0
            engine = self.engine
        threading.Thread(target=self._wait_ready, args=(engine, port), daemon=True).start()
        return {"ok": True}

    def _wait_ready(self, engine: Engine, port: int) -> None:
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        while engine.running() and time.monotonic() < deadline:
            if server_answers(port):
                with self.lock:
                    if self.engine is engine:
                        engine.was_ready = True
                        self.status = "running"
                        self.ready_at = time.time()
                        self.log(f"Ready at {chat_url(port)} "
                                 f"(loaded in {self.ready_at - self.started_at:.0f} s)", "ok")
                return
            time.sleep(0.4)

    def stop(self, wait: bool = False) -> None:
        with self.lock:
            engine = self.engine
            if engine is None:
                return
            self.status = "stopping"
        if wait:
            engine.stop()
        else:
            threading.Thread(target=engine.stop, daemon=True).start()

    def run_doctor(self) -> None:
        with self.lock:
            if self.doctor_running:
                return
            self.doctor_running = True
            self.doctor_raw = []
            self.log("", "")
            self.log("Checking this computer…", "note")
            try:
                Engine(["doctor"], self, prefix="doctor-", env=gpu_setup.engine_environment())
            except OSError as error:
                self.doctor_running = False
                self.doctor = [DoctorLine("fail", "check", f"could not run: {error}")]
                self.doctor_ran = True

    # -- what the page sees --------------------------------------------------

    def snapshot(self, since: int = 0) -> dict[str, object]:
        with self.lock:
            self.last_poll = time.monotonic()
            check = check_model(self.settings.model) if self.settings.model else \
                ModelCheck(False, message="")
            lan = None
            if self.status == "running" and self.settings.share_on_network and self.port:
                address = lan_address()
                lan = f"http://{address}:{self.port}/" if address else None
            now = time.time()
            return {
                "status": self.status,
                "hint": self.hint,
                "port": self.port,
                "chatUrl": chat_url(self.port) if self.port and self.status == "running" else None,
                "lanUrl": lan,
                "elapsed": (now - self.started_at) if self.status == "loading" else
                           (self.ready_at - self.started_at if self.ready_at else 0),
                "settings": asdict(self.settings),
                "model": _check_dict(check),
                "recent": [{"path": m, "name": Path(m).name, "folder": str(Path(m).parent),
                            "exists": Path(m).exists()} for m in self.settings.recent_models],
                "logs": [{"seq": n, "text": t, "kind": k} for n, t, k in self.logs if n > since],
                "lastSeq": self.seq,
                "lastLine": next((t for _, t, _k in reversed(self.logs) if t.strip()), ""),
                "doctor": {"running": self.doctor_running, "ran": self.doctor_ran,
                           "lines": [asdict(line) for line in self.doctor]},
                "version": _version(),
                "platform": sys.platform,
                "framed": bool(self.frame_origin),
                "gpu": self.gpu,
                "gpuSetup": self.gpu_installer.snapshot(),
                "update": self.update,
                "driverUrl": gpu_setup.DRIVER_DOWNLOAD_URL,
                "notice": self._take_notice(),
                "models": {
                    "machine": asdict(self.machine) if self.machine else None,
                    "summary": self.machine.summary() if self.machine else "",
                    "catalog": self.catalog,
                    "loading": self.catalog_loading,
                    "downloads": self.downloads.snapshot(),
                    "library": self.library(),
                    "dir": str(self.models_dir()),
                },
            }

    def _take_notice(self) -> str:
        notice, self.notice = self.notice, ""
        return notice

    def shutdown(self) -> None:
        self.stop(wait=True)
        with self.lock:
            save_settings(self.settings, self.settings_path)


# -- native file dialogs ------------------------------------------------------

_WINDOWS_PICKER = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
# The dialog belongs to a real, shown, topmost window. Owned by a window that
# was never shown, it opened BEHIND the app (a background process may not
# take the foreground on Windows), which looked like nothing happening.
$owner = New-Object System.Windows.Forms.Form -Property @{
  TopMost = $true; ShowInTaskbar = $false; Opacity = 0
  FormBorderStyle = 'None'; StartPosition = 'CenterScreen'
}
$owner.Show()
$owner.Activate()
try {
  if ($env:FW_KIND -eq 'folder') {
    $d = New-Object System.Windows.Forms.FolderBrowserDialog
    $d.Description = 'Choose a safetensors model folder'
    $d.ShowNewFolderButton = $false
    if ($env:FW_START) { $d.SelectedPath = $env:FW_START }
    $answer = $d.ShowDialog($owner)
    if ($answer -eq 'OK') { [Console]::Out.Write($d.SelectedPath) }
  } else {
    $d = New-Object System.Windows.Forms.OpenFileDialog
    $d.Title = 'Choose a model'
    $d.Filter = 'GGUF models (*.gguf)|*.gguf|All files (*.*)|*.*'
    if ($env:FW_START) { $d.InitialDirectory = $env:FW_START }
    $answer = $d.ShowDialog($owner)
    if ($answer -eq 'OK') { [Console]::Out.Write($d.FileName) }
  }
} finally {
  $owner.Close()
}
"""


def _picker_commands(kind: str, start: str) -> list[tuple[list[str], dict[str, str]]]:
    """Dialog programs to try, best looking first."""
    import shutil

    folder = kind == "folder"
    commands: list[tuple[list[str], dict[str, str]]] = []
    if os.name == "nt":
        commands.append((["powershell", "-NoProfile", "-NonInteractive", "-STA",
                          "-Command", _WINDOWS_PICKER], {"FW_KIND": kind, "FW_START": start}))
    else:
        desktop_env = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
        zenity = shutil.which("zenity")
        kdialog = shutil.which("kdialog")
        zenity_cmd = [zenity, "--file-selection", "--title",
                      "Choose a model folder" if folder else "Choose a model"] if zenity else []
        if zenity_cmd:
            zenity_cmd += ["--directory"] if folder else \
                ["--file-filter=GGUF models | *.gguf", "--file-filter=All files | *"]
            if start:
                zenity_cmd += ["--filename", start.rstrip("/") + "/"]
        kdialog_cmd = ([kdialog, "--getexistingdirectory", start or str(Path.home())] if folder
                       else [kdialog, "--getopenfilename", start or str(Path.home()),
                             "GGUF models (*.gguf)"]) if kdialog else []
        ordered = [kdialog_cmd, zenity_cmd] if "kde" in desktop_env else [zenity_cmd, kdialog_cmd]
        commands += [(c, {}) for c in ordered if c]
    own = ([sys.executable, "--pick", kind] if getattr(sys, "frozen", False)
           else [sys.executable, "-m", "flyweight.desktop", "--pick", kind])
    commands.append((own, {"FW_START": start}))
    return commands


def _cancelled(command: list[str], result: subprocess.CompletedProcess) -> bool:
    """Did this helper run its dialog and the person close it?

    Each helper says "cancelled" its own way: PowerShell and Flyweight's own
    Tk dialog exit 0 with no answer, zenity and kdialog exit 1. Anything else,
    including those codes with an error on stderr (no display, a missing
    library, a script error), is a failure, and the next helper gets a turn.
    Reading every exit-1 as a cancel is what made "Choose file" silently do
    nothing when a helper could not run.
    """
    name = Path(command[0]).name.lower()
    errors = result.stderr.lower()
    if name.startswith(("zenity", "kdialog")):
        return result.returncode == 1 and not any(
            word in errors for word in ("cannot open display", "could not connect",
                                        "error", "failed"))
    return result.returncode == 0 and not result.stdout.strip() and \
        "error" not in errors and "exception" not in errors


def pick_path(kind: str, start: str = "") -> str | None:
    """Show a native open dialog; the chosen path, None if cancelled.

    Raises LookupError, naming what went wrong, when no dialog program could
    run, so the page can say so instead of silently doing nothing.
    """
    failures: list[str] = []
    for command, extra in _picker_commands(kind, start):
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                env={**os.environ, **extra}, timeout=3600,
                **_popen_flags(),  # type: ignore[arg-type]
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            failures.append(f"{Path(command[0]).name}: {error}")
            continue
        chosen = result.stdout.strip().splitlines()
        if result.returncode == 0 and chosen:
            return chosen[-1].strip()
        if _cancelled(command, result):
            return None
        detail = (result.stderr.strip().splitlines() or [f"exit code {result.returncode}"])[-1]
        failures.append(f"{Path(command[0]).name}: {detail}")
    raise LookupError("; ".join(failures) or "no file dialog is available")


def _tk_pick(kind: str) -> int:  # pragma: no cover - needs a display
    """`--pick file|folder`: the last-resort dialog, in its own process."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return 2
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    start = os.environ.get("FW_START") or str(Path.home())
    if kind == "folder":
        path = filedialog.askdirectory(title="Choose a model folder", initialdir=start)
    else:
        path = filedialog.askopenfilename(
            title="Choose a model", initialdir=start,
            filetypes=[("GGUF models", "*.gguf"), ("All files", "*")])
    root.destroy()
    if path:
        print(path)
    return 0


# -- the local API --------------------------------------------------------------

UI_DIRECTORY = Path(__file__).resolve().parent / "desktop_ui"
_UI_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
_UI_CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
           "img-src 'self' data:; connect-src 'self'; "
           "frame-src http://127.0.0.1:* http://localhost:*; "
           "object-src 'none'; base-uri 'none'; frame-ancestors 'none'")


def make_server(controller: Controller, token: str, picker=None):
    """The page's backend, on a free loopback port.

    Every /api call carries the per-run token in a header. A custom header
    cannot be sent cross-origin without a CORS preflight this server never
    answers, so another web page open in the browser cannot drive it; the
    Host check closes DNS rebinding.
    """
    import hmac
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    picker = picker or pick_path

    class Handler(BaseHTTPRequestHandler):
        server_version = "FlyweightDesktop"

        def log_message(self, *_: object) -> None:
            pass

        def _host_ok(self) -> bool:
            host = self.headers.get("Host", "")
            port = self.server.server_address[1]  # type: ignore[index]
            return host in (f"127.0.0.1:{port}", f"localhost:{port}")

        def _send(self, status: int, body: bytes, content_type: str,
                  extra: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: object) -> None:
            self._send(status, json.dumps(data).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self) -> None:
            if not self._host_ok():
                self._send(421, b"", "text/plain")
                return
            path, _, query = self.path.partition("?")
            if path in _UI_FILES:
                name, content_type = _UI_FILES[path]
                self._send(200, (UI_DIRECTORY / name).read_bytes(), content_type,
                           {"Content-Security-Policy": _UI_CSP} if name.endswith(".html") else None)
                return
            if path == "/icon.svg":
                self._send(200, _icon_path().read_bytes(), "image/svg+xml")
                return
            if path == "/api/state" and self._authorized():
                since = 0
                for part in query.split("&"):
                    if part.startswith("since="):
                        try:
                            since = int(part[6:])
                        except ValueError:
                            pass
                self._json(200, controller.snapshot(since))
                return
            if path.startswith("/api/"):
                return
            self._send(404, b"not found", "text/plain")

        def _authorized(self) -> bool:
            sent = self.headers.get("X-Flyweight-Token", "")
            if hmac.compare_digest(sent, token):
                return True
            self._json(403, {"error": "forbidden"})
            return False

        def _models(self, verb: str, body: dict) -> object:
            """The Get-a-model page's verbs."""
            if verb == "refresh":
                threading.Thread(target=controller.refresh_catalog, daemon=True).start()
                return {"ok": True}
            if verb == "download":
                return controller.download(body)
            if verb in ("pause", "resume", "cancel"):
                job = str(body.get("job", ""))
                {"pause": controller.downloads.pause, "resume": controller.downloads.resume,
                 "cancel": controller.downloads.remove}[verb](job)
                return {"ok": True}
            if verb == "search":
                query = str(body.get("query", "")).strip()
                if not query:
                    return {"ok": True, "results": []}
                try:
                    return {"ok": True, "results": models.search(query)}
                except models.HubError as error:
                    return {"ok": False, "error": str(error)}
            if verb == "repo":
                repo = str(body.get("repo", ""))
                if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
                    return {"ok": False, "error": "That is not a Hugging Face repository name."}
                machine = controller.machine or models.machine(controller.models_dir())
                try:
                    return {"ok": True, "repo": models.repo_details(repo, machine)}
                except models.HubError as error:
                    return {"ok": False, "error": f"{repo}: {error}"}
            if verb == "use":
                return controller.use_model(str(body.get("path", "")))
            if verb == "delete":
                return controller.delete_model(str(body.get("folder", "")))
            if verb == "reveal":
                target = str(body.get("path") or "")
                if not target:  # the downloads folder itself, created on first use
                    controller.models_dir().mkdir(parents=True, exist_ok=True)
                    target = str(controller.models_dir())
                try:
                    models.reveal(target)
                except OSError as error:
                    return {"ok": False, "error": str(error)}
                return {"ok": True}
            if verb == "folder":
                try:
                    chosen = picker("folder", str(controller.models_dir()))
                except LookupError:
                    return {"ok": False, "error": "No folder dialog is available here."}
                if chosen:
                    controller.update_settings({"models_dir": chosen})
                    controller._library = (0.0, [])
                return {"ok": True, "path": chosen}
            return {"ok": False, "error": "unknown"}

        def do_POST(self) -> None:
            if not self._host_ok():
                self._send(421, b"", "text/plain")
                return
            if not self._authorized():
                return
            length = min(int(self.headers.get("Content-Length") or 0), 1 << 20)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                body = {}
            if not isinstance(body, dict):
                body = {}
            path = self.path.partition("?")[0]
            if path == "/api/settings":
                controller.update_settings(body)
                self._json(200, controller.snapshot(controller.seq))
            elif path == "/api/forget":
                controller.forget_model(str(body.get("path", "")))
                self._json(200, {"ok": True})
            elif path == "/api/start":
                self._json(200, controller.start())
            elif path == "/api/stop":
                controller.stop()
                self._json(200, {"ok": True})
            elif path == "/api/doctor":
                controller.run_doctor()
                self._json(200, {"ok": True})
            elif path == "/api/browse":
                kind = "folder" if body.get("kind") == "folder" else "file"
                current = controller.settings.model
                start = str(Path(current).parent) if current else ""
                try:
                    chosen = picker(kind, start)
                except LookupError as error:
                    controller.log(f"The file dialog could not open: {error}", "error")
                    self._json(200, {"ok": False, "error":
                                     "The file dialog could not open (details in "
                                     "Activity). Paste the model's path into the "
                                     "box under the buttons instead."})
                    return
                if chosen:
                    controller.update_settings({"model": chosen})
                self._json(200, {"ok": True, "path": chosen})
            elif path == "/api/open":
                url = str(body.get("url", ""))
                allowed = {gpu_setup.DRIVER_DOWNLOAD_URL, RELEASES_PAGE}
                if controller.update:
                    allowed.add(controller.update["url"])
                if url in allowed or re.fullmatch(r"http://(127\.0\.0\.1|[0-9.]+):\d+/", url):
                    webbrowser.open(url)
                self._json(200, {"ok": True})
            elif path == "/api/shortcut":
                try:
                    written = install_shortcut()
                    self._json(200, {"ok": True, "paths": [str(p) for p in written]})
                except (OSError, subprocess.CalledProcessError) as error:
                    self._json(200, {"ok": False, "error": str(error)})
            elif path.startswith("/api/models/"):
                self._json(200, self._models(path[len("/api/models/"):], body))
            elif path == "/api/gpu/setup":
                self._json(200, controller.setup_gpu())
            elif path == "/api/gpu/refresh":
                controller.refresh_gpu()
                self._json(200, {"ok": True})
            elif path == "/api/quit":
                self._json(200, {"ok": True})
                controller.quit_requested.set()
            else:
                self._json(404, {"error": "unknown"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    return server


# -- the window ---------------------------------------------------------------


def _app_browsers() -> list[str]:
    """Chromium-family browsers that can open a page as a bare app window."""
    import shutil

    found: list[str] = []
    if os.name == "nt":
        roots = [os.environ.get(name, "") for name in
                 ("ProgramFiles(x86)", "ProgramFiles", "LocalAppData")]
        relative = (r"Microsoft\Edge\Application\msedge.exe",
                    r"Google\Chrome\Application\chrome.exe",
                    r"BraveSoftware\Brave-Browser\Application\brave.exe")
        for rel in relative:
            for root in roots:
                candidate = Path(root) / rel if root else None
                if candidate and candidate.is_file():
                    found.append(str(candidate))
                    break
    else:
        for name in ("google-chrome", "google-chrome-stable", "chromium",
                     "chromium-browser", "microsoft-edge", "microsoft-edge-stable",
                     "brave-browser", "brave", "vivaldi"):
            path = shutil.which(name)
            if path:
                found.append(path)
    return found


def _open_app_window(url: str) -> bool:  # pragma: no cover - needs a display
    """Show the page in a browser app window and wait for it to close."""
    for browser in _app_browsers():
        profile = config_dir() / "window"
        profile.mkdir(parents=True, exist_ok=True)
        command = [browser, f"--app={url}", f"--user-data-dir={profile}",
                   "--no-first-run", "--no-default-browser-check",
                   "--window-size=1200,820", "--class=Flyweight",
                   "--disable-features=Translate"]
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            command.append("--no-sandbox")  # Chromium refuses to run as root otherwise
        try:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
        except OSError:
            continue
        started = time.monotonic()
        code = process.wait()
        if code == 0 or time.monotonic() - started > 5:
            return True
    return False


def run_window(mode: str | None = None) -> int:  # pragma: no cover - needs a display
    """Serve the page and show it: pywebview, else a browser app window,
    else an ordinary browser tab kept alive by the page's polling."""
    import secrets

    mode = mode or os.environ.get("FLYWEIGHT_DESKTOP_WINDOW", "")
    controller = Controller()
    token = secrets.token_urlsafe(24)
    webview_window: list[object] = []

    def webview_picker(kind: str, start: str) -> str | None:
        import webview  # type: ignore[import-not-found]

        window = webview_window[0]
        dialog = webview.FOLDER_DIALOG if kind == "folder" else webview.OPEN_DIALOG
        kwargs = {} if kind == "folder" else {
            "file_types": ("GGUF models (*.gguf)", "All files (*.*)")}
        result = window.create_file_dialog(dialog, directory=start, **kwargs)  # type: ignore[attr-defined]
        return result[0] if result else None

    use_webview = False
    if mode in ("", "webview"):
        try:
            import webview  # type: ignore[import-not-found]  # noqa: F401
            use_webview = True
        except ImportError:
            use_webview = False

    server = make_server(controller, token, webview_picker if use_webview else None)
    port = server.server_address[1]
    controller.frame_origin = f"http://127.0.0.1:{port}"
    url = f"http://127.0.0.1:{port}/#t={token}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if refresh_appimage_entry():
        controller.log("Added Flyweight to your applications menu.", "note")
    threading.Thread(target=controller.refresh_gpu, daemon=True).start()
    threading.Thread(target=controller.refresh_catalog, daemon=True).start()
    threading.Thread(target=controller.check_for_update, daemon=True).start()
    if controller.first_run:
        controller.run_doctor()

    try:
        if use_webview:
            import webview  # type: ignore[import-not-found]

            window = webview.create_window(f"{APP_NAME}", url, width=1200, height=820,
                                           min_size=(860, 600), background_color="#0f1115")
            webview_window.append(window)
            webview.start()
        elif mode != "browser" and _open_app_window(url):
            pass
        else:
            webbrowser.open(url)
            print(f"{APP_NAME} is open in your browser: {url}\n"
                  "Close the tab or press Ctrl-C here to quit.", file=sys.stderr)
            # The page polls every second (once a minute when hidden). No
            # poll for three minutes means the tab is gone.
            opened = time.monotonic()
            while not controller.quit_requested.wait(2):
                # Never connected (no browser came up), or stopped polling.
                since = controller.last_poll or opened
                if time.monotonic() - since > (180 if controller.last_poll else 120):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown()
        server.shutdown()
    return 0


# --------------------------------------------------------------------------
# Entry points


def _reattach_windows_streams() -> None:  # pragma: no cover - Windows, frozen
    """A windowed (no-console) build starts with sys.stdout None even when its
    parent handed it pipes. Rebuild the streams from the inherited handles so
    the engine's output reaches the window's log."""
    if os.name != "nt" or (sys.stdout is not None and sys.stderr is not None):
        return
    try:
        import ctypes
        import msvcrt
        for number, name in ((-11, "stdout"), (-12, "stderr")):
            if getattr(sys, name) is not None:
                continue
            handle = ctypes.windll.kernel32.GetStdHandle(number)  # type: ignore[attr-defined]
            if handle in (0, -1, None):
                continue
            descriptor = msvcrt.open_osfhandle(handle, os.O_WRONLY)  # type: ignore[attr-defined]
            setattr(sys, name, open(descriptor, "w", buffering=1,
                                    encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 - without streams the engine still serves
        pass


def main(argv: list[str] | None = None) -> int:
    """`flyweight-desktop` and the frozen app. `--engine ARGS` runs the CLI."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--engine"]:
        _reattach_windows_streams()
        from .cli import main as cli_main
        return cli_main(argv[1:])
    if "--install-shortcut" in argv:
        for path in install_shortcut():
            print(f"added {path}")
        return 0
    if argv[:1] == ["--pick"]:
        return _tk_pick(argv[1] if len(argv) > 1 else "file")
    if "--uninstall-shortcut" in argv:
        for path in uninstall_shortcut():
            print(f"removed {path}")
        return 0
    mode = None
    for flag in ("--webview", "--app-window", "--browser"):
        if flag in argv:
            mode = {"--webview": "webview", "--app-window": "app", "--browser": "browser"}[flag]
    return run_window(mode)


if __name__ == "__main__":
    raise SystemExit(main())
