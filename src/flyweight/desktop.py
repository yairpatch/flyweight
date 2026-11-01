"""Flyweight Desktop: a window that runs the engine for people who do not
want a terminal.

Everything the command line asks of a new user -- find the model file, know
which flags matter, notice that the port is taken, read a stack trace to learn
the CUDA toolkit is missing -- is done here with a file picker, a few plain
choices and a log that explains its own failures. The engine itself is the
same `flyweight serve`, run as a child process, so the window never holds the
GPU and a crash in the runtime cannot take the window down with it.

Linux and Windows are supported. The window is Tk, which ships with the
python.org Windows installer and is one package away on Linux
(`python3-tk`, `tk`); nothing else is required. The module is importable
without Tk so the planning logic below can be tested headless.

    flyweight app                      open the window
    flyweight app --install-shortcut   add Flyweight to the applications menu
"""
from __future__ import annotations

import json
import os
import queue
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

APP_NAME = "Flyweight"
# What the user sees, mapped to what `serve` takes. Kept as pairs so the
# combobox shows words while the command line gets the flag value.
BACKENDS = (
    ("Automatic (recommended)", "auto"),
    ("NVIDIA GPU (CUDA)", "cuda"),
    ("AMD GPU (ROCm, Linux)", "rocm"),
    ("CPU only (no GPU)", "cpu"),
)
CONTEXTS = (
    ("Automatic (recommended)", 0),
    ("8K tokens", 8192),
    ("16K tokens", 16384),
    ("32K tokens", 32768),
    ("64K tokens", 65536),
    ("128K tokens", 131072),
)
EXPERT_MODES = (
    ("Automatic (recommended)", "auto"),
    ("Keep experts on the CPU (big MoE models)", "cpu"),
    ("All experts on the GPU", "resident"),
)
QUANTS = (
    ("Q6_K (recommended)", "Q6_K"),
    ("Q4_K (smaller)", "Q4_K"),
    ("Q8_0 (larger, closest to original)", "Q8_0"),
    ("IQ4_XS (smallest usable)", "IQ4_XS"),
)
DEFAULT_PORT = 8000
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


class Engine:
    """One `flyweight` child process whose output arrives on a queue."""

    def __init__(self, arguments: list[str], output: "queue.Queue[tuple[str, object]]",
                 prefix: str = "") -> None:
        self.command = engine_command() + arguments
        self.output = output
        self.prefix = prefix
        self.stopping = False
        self.was_ready = False
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env=_child_environment(), **_popen_flags(),
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
        "Comment=Run a local language model and chat with it\n"
        f"Exec={exec_line}\n"
        f"Icon={_icon_path()}\n"
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
    path.write_text(linux_desktop_entry(), encoding="utf-8")
    path.chmod(0o755)
    written.append(path)
    return written


def uninstall_shortcut() -> list[Path]:
    removed = []
    for path in _shortcut_paths():
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


# --------------------------------------------------------------------------
# The window


_TK_MISSING = """\
Flyweight Desktop needs Tk, the GUI toolkit that ships with Python.
Install it and run `flyweight app` again:

  Debian / Ubuntu:   sudo apt install python3-tk
  Fedora:            sudo dnf install python3-tkinter
  Arch:              sudo pacman -S tk
  Windows:           re-run the python.org installer and tick "tcl/tk and IDLE"

The engine itself works without it: `flyweight serve MODEL` in a terminal."""


def _label(pairs: tuple[tuple[str, object], ...], value: object) -> str:
    for label, candidate in pairs:
        if candidate == value:
            return label
    return pairs[0][0]


def _value(pairs: tuple[tuple[str, object], ...], label: str) -> object:
    for candidate_label, value in pairs:
        if candidate_label == label:
            return value
    return pairs[0][1]


def run_window() -> int:  # pragma: no cover - exercised by hand and under Xvfb
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError:
        print(_TK_MISSING, file=sys.stderr)
        return 1
    try:
        root = tk.Tk(className="Flyweight")
    except tk.TclError as error:
        print(f"flyweight: cannot open a window ({error}). Is a display "
              "available? Use `flyweight serve MODEL` in a terminal instead.",
              file=sys.stderr)
        return 1

    settings = load_settings()
    events: "queue.Queue[tuple[str, object]]" = queue.Queue()
    state: dict[str, object] = {"engine": None, "port": None, "ready": False,
                                "log": [], "doctor": None}

    root.title(f"{APP_NAME} Desktop")
    root.minsize(720, 560)
    try:
        root.tk.call("tk", "scaling")  # touch Tk so a broken install fails here
    except tk.TclError:
        pass
    style = ttk.Style(root)
    if "clam" in style.theme_names() and os.name != "nt":
        style.theme_use("clam")
    style.configure("Status.TLabel", font=("TkDefaultFont", 11, "bold"))
    style.configure("Big.TButton", padding=(18, 8))

    outer = ttk.Frame(root, padding=14)
    outer.pack(fill="both", expand=True)
    outer.columnconfigure(1, weight=1)

    # --- Step 1: model
    ttk.Label(outer, text="1. Model", font=("TkDefaultFont", 11, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w")
    model_var = tk.StringVar(value=settings.model)
    model_box = ttk.Combobox(outer, textvariable=model_var,
                             values=settings.recent_models)
    model_box.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
    pick = ttk.Frame(outer)
    pick.grid(row=1, column=2, sticky="e", padx=(6, 0), pady=(4, 0))
    model_note = ttk.Label(outer, text="", wraplength=660)
    model_note.grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 8))

    def choose_file() -> None:
        path = filedialog.askopenfilename(
            title="Choose a GGUF model",
            filetypes=[("GGUF models", "*.gguf"), ("All files", "*")])
        if path:
            model_var.set(path)

    def choose_folder() -> None:
        path = filedialog.askdirectory(title="Choose a safetensors model folder")
        if path:
            model_var.set(path)

    ttk.Button(pick, text="Choose file…", command=choose_file).pack(side="left")
    ttk.Button(pick, text="Folder…", command=choose_folder).pack(side="left", padx=(4, 0))

    # --- Step 2: options
    ttk.Label(outer, text="2. Options", font=("TkDefaultFont", 11, "bold")).grid(
        row=3, column=0, columnspan=3, sticky="w")
    options = ttk.Frame(outer)
    options.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(4, 0))
    options.columnconfigure(1, weight=1)
    options.columnconfigure(3, weight=1)

    backend_var = tk.StringVar(value=_label(BACKENDS, settings.backend))
    context_var = tk.StringVar(value=_label(CONTEXTS, settings.context))
    port_var = tk.StringVar(value=str(settings.port))
    share_var = tk.BooleanVar(value=settings.share_on_network)
    open_var = tk.BooleanVar(value=settings.open_chat)
    mmproj_var = tk.BooleanVar(value=settings.use_mmproj)

    ttk.Label(options, text="Run on").grid(row=0, column=0, sticky="w")
    ttk.Combobox(options, textvariable=backend_var, state="readonly",
                 values=[label for label, _ in BACKENDS]).grid(
        row=0, column=1, sticky="ew", padx=(6, 16))
    ttk.Label(options, text="Context length").grid(row=0, column=2, sticky="w")
    ttk.Combobox(options, textvariable=context_var, state="readonly",
                 values=[label for label, _ in CONTEXTS]).grid(
        row=0, column=3, sticky="ew", padx=(6, 0))
    ttk.Label(options, text="Port").grid(row=1, column=0, sticky="w", pady=(6, 0))
    ttk.Spinbox(options, from_=1024, to=65535, textvariable=port_var, width=8).grid(
        row=1, column=1, sticky="w", padx=(6, 16), pady=(6, 0))
    ttk.Checkbutton(options, text="Open the chat when ready", variable=open_var).grid(
        row=1, column=2, columnspan=2, sticky="w", pady=(6, 0))
    ttk.Checkbutton(options, text="Let other devices on my network use it",
                    variable=share_var).grid(row=2, column=0, columnspan=2,
                                             sticky="w", pady=(6, 0))
    mmproj_check = ttk.Checkbutton(options, text="Image input (projector found)",
                                   variable=mmproj_var)

    # Advanced, folded away.
    advanced_shown = tk.BooleanVar(value=False)
    advanced = ttk.Frame(outer)
    advanced.columnconfigure(1, weight=1)
    advanced.columnconfigure(3, weight=1)
    expert_var = tk.StringVar(value=_label(EXPERT_MODES, settings.expert_mode))
    quant_var = tk.StringVar(value=_label(QUANTS, settings.quant))
    threads_var = tk.StringVar(value=str(settings.cpu_threads))
    key_var = tk.StringVar(value=settings.api_key)
    extra_var = tk.StringVar(value=settings.extra_args)
    ttk.Label(advanced, text="MoE experts").grid(row=0, column=0, sticky="w")
    ttk.Combobox(advanced, textvariable=expert_var, state="readonly",
                 values=[label for label, _ in EXPERT_MODES]).grid(
        row=0, column=1, sticky="ew", padx=(6, 16))
    ttk.Label(advanced, text="Safetensors quant").grid(row=0, column=2, sticky="w")
    ttk.Combobox(advanced, textvariable=quant_var, state="readonly",
                 values=[label for label, _ in QUANTS]).grid(
        row=0, column=3, sticky="ew", padx=(6, 0))
    ttk.Label(advanced, text="CPU threads (0 = auto)").grid(row=1, column=0, sticky="w", pady=(6, 0))
    ttk.Spinbox(advanced, from_=0, to=512, textvariable=threads_var, width=8).grid(
        row=1, column=1, sticky="w", padx=(6, 16), pady=(6, 0))
    ttk.Label(advanced, text="API key (optional)").grid(row=1, column=2, sticky="w", pady=(6, 0))
    ttk.Entry(advanced, textvariable=key_var, show="•").grid(
        row=1, column=3, sticky="ew", padx=(6, 0), pady=(6, 0))
    ttk.Label(advanced, text="Extra serve flags").grid(row=2, column=0, sticky="w", pady=(6, 0))
    ttk.Entry(advanced, textvariable=extra_var).grid(
        row=2, column=1, columnspan=3, sticky="ew", padx=(6, 0), pady=(6, 0))

    def toggle_advanced() -> None:
        if advanced_shown.get():
            advanced.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        else:
            advanced.grid_remove()

    ttk.Checkbutton(outer, text="Show advanced options", variable=advanced_shown,
                    command=toggle_advanced).grid(row=5, column=0, columnspan=3,
                                                  sticky="w", pady=(8, 0))

    # --- Step 3: start
    ttk.Label(outer, text="3. Start", font=("TkDefaultFont", 11, "bold")).grid(
        row=7, column=0, columnspan=3, sticky="w", pady=(12, 0))
    controls = ttk.Frame(outer)
    controls.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(4, 0))
    start_button = ttk.Button(controls, text="Start", style="Big.TButton")
    start_button.pack(side="left")
    chat_button = ttk.Button(controls, text="Open chat", state="disabled")
    chat_button.pack(side="left", padx=(6, 0))
    doctor_button = ttk.Button(controls, text="Check this computer")
    doctor_button.pack(side="right")
    status_var = tk.StringVar(value="Stopped")
    ttk.Label(outer, textvariable=status_var, style="Status.TLabel",
              wraplength=680).grid(row=9, column=0, columnspan=3, sticky="w", pady=(8, 4))

    log_frame = ttk.Frame(outer)
    log_frame.grid(row=10, column=0, columnspan=3, sticky="nsew")
    outer.rowconfigure(10, weight=1)
    log = tk.Text(log_frame, height=12, wrap="word", state="disabled",
                  font=("TkFixedFont", 9))
    scroll = ttk.Scrollbar(log_frame, command=log.yview)
    log.configure(yscrollcommand=scroll.set)
    log.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")
    log.tag_configure("fail", foreground="#c62828")
    log.tag_configure("warn", foreground="#b26a00")
    log.tag_configure("ok", foreground="#2e7d32")
    log.tag_configure("note", foreground="#1565c0")

    def write(text: str, tag: str = "") -> None:
        log.configure(state="normal")
        log.insert("end", text + "\n", tag)
        # Bound the widget: a long session's per-request lines add up.
        if int(log.index("end-1c").split(".")[0]) > 5000:
            log.delete("1.0", "1000.0")
        log.see("end")
        log.configure(state="disabled")

    current: dict[str, ModelCheck] = {}
    doctor_text: list[str] = []

    def refresh_model(*_: object) -> None:
        check = check_model(model_var.get())
        current["model"] = check
        if check.ok:
            model_note.configure(text="✓ " + check.message, foreground="#2e7d32")
        else:
            model_note.configure(text=check.message if model_var.get().strip() else
                                 "Choose a model file (.gguf) or a safetensors folder.",
                                 foreground="#c62828" if model_var.get().strip() else "")
        if check.ok and check.mmproj:
            mmproj_check.configure(text=f"Image input ({Path(check.mmproj).name})")
            mmproj_check.grid(row=2, column=2, columnspan=2, sticky="w", pady=(6, 0))
        else:
            mmproj_check.grid_remove()

    model_var.trace_add("write", refresh_model)
    refresh_model()

    def collect() -> Settings | None:
        try:
            port = int(port_var.get())
            threads = int(threads_var.get() or 0)
        except ValueError:
            messagebox.showerror(APP_NAME, "Port and CPU threads must be whole numbers.")
            return None
        if not 1024 <= port <= 65535:
            messagebox.showerror(APP_NAME, "Choose a port between 1024 and 65535.")
            return None
        settings.model = model_var.get().strip()
        settings.backend = str(_value(BACKENDS, backend_var.get()))
        settings.context = int(_value(CONTEXTS, context_var.get()))  # type: ignore[call-overload]
        settings.port = port
        settings.share_on_network = share_var.get()
        settings.open_chat = open_var.get()
        settings.use_mmproj = mmproj_var.get()
        settings.expert_mode = str(_value(EXPERT_MODES, expert_var.get()))
        settings.quant = str(_value(QUANTS, quant_var.get()))
        settings.cpu_threads = max(0, threads)
        settings.api_key = key_var.get()
        settings.extra_args = extra_var.get()
        return settings

    def set_running(running: bool) -> None:
        start_button.configure(text="Stop" if running else "Start")
        # ttk's state flags, not configure(state=...): the latter would turn a
        # readonly combobox into a free-text one on the way back.
        flag = ["disabled"] if running else ["!disabled"]
        for widget in (model_box, *pick.winfo_children(), *options.winfo_children(),
                       *advanced.winfo_children()):
            set_state = getattr(widget, "state", None)
            if callable(set_state):
                try:
                    set_state(flag)
                except tk.TclError:
                    pass
        if not running:
            chat_button.configure(state="disabled")

    def start() -> None:
        if state["engine"] is not None:
            stop()
            return
        chosen = collect()
        if chosen is None:
            return
        check = check_model(chosen.model)
        if not check.ok:
            messagebox.showwarning(APP_NAME, check.message)
            return
        try:
            serve_arguments(chosen, check, chosen.port)
        except ValueError as error:
            messagebox.showerror(APP_NAME, f"The extra flags could not be read: {error}")
            return
        bind = "0.0.0.0" if chosen.share_on_network else "127.0.0.1"
        port = pick_port(chosen.port, bind)
        if port != chosen.port:
            write(f"Port {chosen.port} is busy; using {port} instead.", "note")
        remember_model(chosen, check.path)
        model_box.configure(values=chosen.recent_models)
        save_settings(chosen)
        arguments = serve_arguments(chosen, check, port)
        write("")
        write("$ flyweight " + " ".join(shlex.quote(a) if a != chosen.api_key.strip()
                                        else "****" for a in arguments), "note")
        try:
            state["engine"] = Engine(arguments, events)
        except OSError as error:
            messagebox.showerror(APP_NAME, f"Could not start the engine: {error}")
            return
        state["port"] = port
        state["ready"] = False
        state["log"] = []
        state["started"] = time.monotonic()
        status_var.set("Loading the model… (large models can take several minutes)")
        set_running(True)
        threading.Thread(target=wait_ready, args=(state["engine"], port), daemon=True).start()

    def wait_ready(engine: Engine, port: int) -> None:
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        while engine.running() and time.monotonic() < deadline:
            if server_answers(port):
                events.put(("ready", port))
                return
            time.sleep(0.5)

    def stop() -> None:
        engine = state["engine"]
        if isinstance(engine, Engine) and engine.running():
            status_var.set("Stopping…")
            start_button.state(["disabled"])
            # Off the Tk thread: a clean shutdown can take seconds, and the
            # exit event re-enables the button.
            threading.Thread(target=engine.stop, daemon=True).start()

    def open_chat() -> None:
        port = state.get("port")
        if isinstance(port, int):
            webbrowser.open(chat_url(port))

    def run_doctor() -> None:
        doctor_button.configure(state="disabled")
        write("")
        write("Checking this computer…", "note")
        doctor_text.clear()
        try:
            Engine(["doctor"], events, prefix="doctor-")
        except OSError as error:
            write(f"Could not run the check: {error}", "fail")
            doctor_button.configure(state="normal")

    def handle(kind: str, payload: object) -> None:
        if kind == "line":
            text = str(payload)
            log_lines = state["log"]
            assert isinstance(log_lines, list)
            log_lines.append(text)
            del log_lines[:-400]
            lowered = text.lower()
            tag = "fail" if ("error" in lowered or "traceback" in lowered
                             or text.startswith("flyweight:")) else ""
            write(text, tag)
        elif kind == "ready":
            state["ready"] = True
            if isinstance(state["engine"], Engine):
                state["engine"].was_ready = True
            port = int(payload)  # type: ignore[call-overload]
            seconds = time.monotonic() - float(state.get("started", time.monotonic()))  # type: ignore[arg-type]
            message = f"Running — chat at {chat_url(port)}  (loaded in {seconds:.0f} s)"
            if settings.share_on_network:
                address = lan_address()
                if address:
                    message += f"\nOther devices: http://{address}:{port}/"
            status_var.set(message)
            write(message, "ok")
            chat_button.configure(state="normal")
            if settings.open_chat:
                open_chat()
        elif kind == "exit":
            code = payload
            engine = state["engine"]
            state["engine"] = None
            state["ready"] = False
            set_running(False)
            start_button.state(["!disabled"])
            if code == 0 or (isinstance(engine, Engine) and engine.stopping):
                status_var.set("Stopped")
                write("Stopped.", "note")
                return
            log_lines = state["log"]
            hint = explain_failure("\n".join(log_lines))  # type: ignore[arg-type]
            served = isinstance(engine, Engine) and engine.was_ready
            status_var.set(("The engine stopped: " if served else "Could not start: ") + hint)
            write(f"Engine exited with code {code}. {hint}", "fail")
        elif kind == "doctor-line":
            text = str(payload)
            stripped = text.strip()
            tag = ("fail" if stripped.startswith("[FAIL") else
                   "warn" if stripped.startswith("[warn") else
                   "ok" if stripped.startswith("[ok") else
                   "note" if stripped.startswith("->") else "")
            doctor_text.append(text)
            # A missing package or library makes doctor itself crash.
            if "traceback" in text.lower() or "error" in text.lower():
                tag = tag or "fail"
            write(text, tag)
        elif kind == "doctor-exit":
            doctor_button.configure(state="normal")
            report = parse_doctor("\n".join(doctor_text))
            failures = [line for line in report if line.status == "fail"]
            if failures:
                status_var.set(f"{len(failures)} problem(s) found — the red lines "
                               "in the log say how to fix each.")
            elif state["engine"] is None:
                status_var.set("This computer can run Flyweight. Choose a model and press Start.")

    def pump() -> None:
        try:
            while True:
                kind, payload = events.get_nowait()
                handle(kind, payload)
        except queue.Empty:
            pass
        root.after(100, pump)

    def on_close() -> None:
        engine = state["engine"]
        if isinstance(engine, Engine) and engine.running():
            if not messagebox.askokcancel(APP_NAME, "Stop the model and quit?"):
                return
            engine.stop()  # wait for it: quitting must not orphan the server
        collect()
        save_settings(settings)
        root.destroy()

    start_button.configure(command=start)
    chat_button.configure(command=open_chat)
    doctor_button.configure(command=run_doctor)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, pump)

    write(f"{APP_NAME} Desktop. Choose a model, press Start, and the chat opens "
          "in your browser when the model is loaded.", "note")
    if not (config_dir() / "desktop.json").exists():
        run_doctor()  # first run: say up front whether this machine can serve
    root.mainloop()
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
    if "--uninstall-shortcut" in argv:
        for path in uninstall_shortcut():
            print(f"removed {path}")
        return 0
    return run_window()


if __name__ == "__main__":
    raise SystemExit(main())
