"""Make an NVIDIA GPU usable without installing the CUDA toolkit.

The runtime needs three things from CUDA besides the driver: NVRTC (the
runtime compiler, a shared library), the CUDA headers (`cuda_fp16.h` and
friends) and CCCL's headers. NVIDIA publishes all three as redistributable
wheels on PyPI. This module downloads the ones that match the installed
driver, unpacks them into a private folder laid out like a toolkit
(`bin`/`lib64`, `include`, `include/cccl`), and hands the engine an
environment that points `CUDA_PATH` at it -- the variable the runtime already
reads. Nothing outside that folder is touched, and deleting it undoes it.

Only the driver itself cannot come from here; when it is missing the status
says so and links to NVIDIA's download page.
"""
from __future__ import annotations

import ctypes
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

DRIVER_DOWNLOAD_URL = "https://www.nvidia.com/en-us/drivers/"
PYPI_JSON = "https://pypi.org/pypi/{name}/json"

# The wheels for each CUDA major. A driver runs any NVRTC of its own major
# (the runtime falls back to a cubin when NVRTC is newer than the driver).
PACKAGES = {
    12: ("nvidia-cuda-nvrtc-cu12", "nvidia-cuda-runtime-cu12", "nvidia-cuda-cccl-cu12"),
    13: ("nvidia-cuda-nvrtc", "nvidia-cuda-runtime", "nvidia-cuda-cccl"),
}
MARKER = "flyweight-cuda.json"


def data_dir() -> Path:
    """Per-user data (large, machine-local): %LOCALAPPDATA% or XDG_DATA_HOME."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "Flyweight"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "flyweight"


def cuda_dir(major: int) -> Path:
    return data_dir() / f"cuda-{major}"


# -- what this machine has ------------------------------------------------------


def driver_cuda_version() -> int | None:
    """The CUDA version the NVIDIA driver supports (e.g. 12040), or None."""
    names = ["nvcuda.dll"] if os.name == "nt" else ["libcuda.so.1", "libcuda.so"]
    for name in names:
        try:
            library = ctypes.CDLL(name)
        except OSError:
            continue
        version = ctypes.c_int(0)
        try:
            if library.cuDriverGetVersion(ctypes.byref(version)) == 0 and version.value > 0:
                return version.value
        except AttributeError:
            continue
    return None


def nvidia_gpu_names() -> list[str]:
    """NVIDIA display adapters the OS sees, driver or not."""
    names: list[str] = []
    if os.name == "nt":
        import subprocess

        try:
            output = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return names
        return [line.strip() for line in output.splitlines() if "nvidia" in line.lower()]
    devices = Path("/sys/bus/pci/devices")
    try:
        for device in devices.iterdir():
            try:
                vendor = (device / "vendor").read_text().strip()
                klass = (device / "class").read_text().strip()
            except OSError:
                continue
            if vendor == "0x10de" and klass.startswith("0x03"):
                names.append("NVIDIA GPU")
    except OSError:
        pass
    return names


def _system_toolkit() -> bool:
    """A CUDA toolkit the runtime would already find on its own."""
    roots = [os.environ.get("CUDA_PATH"), os.environ.get("CUDA_HOME")]
    if os.name != "nt":
        roots += ["/opt/cuda", "/usr/local/cuda"]
    for root in filter(None, roots):
        if (Path(root) / "include" / "cuda_fp16.h").is_file():
            return True
    return False


def installed_major() -> int | None:
    """The CUDA major of a complete private install, newest first."""
    for major in sorted(PACKAGES, reverse=True):
        marker = cuda_dir(major) / MARKER
        if marker.is_file():
            return major
    return None


def engine_environment() -> dict[str, str]:
    """Variables that point the engine at the private install, if there is one."""
    driver = driver_cuda_version()
    major = installed_major()
    if major is None or driver is None or driver // 1000 < major:
        return {}
    root = cuda_dir(major)
    env = {"CUDA_PATH": str(root), "CUDA_HOME": str(root)}
    if os.name != "nt":
        lib = str(root / "lib64")
        previous = os.environ.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = lib + (os.pathsep + previous if previous else "")
    return env


@dataclass
class GpuStatus:
    """What the page shows about NVIDIA support, and what it can offer."""

    state: str            # ready, needs-setup, needs-driver, old-driver, none
    title: str
    detail: str
    driver: str = ""      # "12.4"
    major: int = 0        # the CUDA major setup would install
    download_mb: int = 0
    gpus: list[str] = field(default_factory=list)


def status() -> GpuStatus:
    driver = driver_cuda_version()
    if driver is None:
        gpus = nvidia_gpu_names()
        if gpus:
            return GpuStatus("needs-driver", "Install the NVIDIA driver",
                             "This computer has an NVIDIA graphics card, but its "
                             "driver is not installed, so models would run on the "
                             "much slower CPU.", gpus=gpus)
        return GpuStatus("none", "No NVIDIA GPU", "Models run on the CPU.")
    shown = f"{driver // 1000}.{driver % 1000 // 10}"
    major = driver // 1000
    if major < min(PACKAGES):
        return GpuStatus("old-driver", "Update the NVIDIA driver",
                         f"The installed driver supports CUDA {shown}; Flyweight "
                         f"needs CUDA {min(PACKAGES)} or newer.", driver=shown)
    target = max(m for m in PACKAGES if m <= major)
    current = installed_major()
    if current is not None and current <= major:
        return GpuStatus("ready", "NVIDIA GPU ready",
                         f"Driver CUDA {shown}; GPU support installed by Flyweight.",
                         driver=shown, major=current)
    if _system_toolkit():
        return GpuStatus("ready", "NVIDIA GPU ready",
                         f"Driver CUDA {shown}; using the installed CUDA toolkit.",
                         driver=shown, major=target)
    return GpuStatus("needs-setup", "Finish GPU setup",
                     "Your NVIDIA driver is installed. One more download (from "
                     "NVIDIA, via PyPI) lets Flyweight compile its GPU code.",
                     driver=shown, major=target,
                     download_mb=95 if target == 12 else 60)


# -- installing -------------------------------------------------------------------


def _platform_tag() -> str:
    return "win_amd64" if os.name == "nt" else "manylinux"


def _wheel_for(name: str) -> tuple[str, str, int]:
    """(url, sha256, size) of the newest wheel of `name` for this platform."""
    with urllib.request.urlopen(PYPI_JSON.format(name=name), timeout=60) as response:
        data = json.load(response)
    tag = _platform_tag()
    for entry in data["urls"]:
        filename = entry["filename"]
        if tag in filename and ("x86_64" in filename or "amd64" in filename):
            return entry["url"], entry["digests"]["sha256"], int(entry["size"])
    raise RuntimeError(f"{name} has no wheel for this platform")


def target_path(member: str, root: Path) -> Path | None:
    """Where a wheel member goes in the toolkit-shaped folder, or None.

    `nvidia/<anything>/include/...` keeps its path under include/ (so CUDA
    13's include/cccl/ stays put and CUDA 12's cub/ lands beside cuda_fp16.h,
    both on the runtime's -I list). Libraries flatten into lib64/ (Linux) and
    DLLs into bin/ (Windows), the two places the runtime looks.
    """
    parts = member.split("/")
    if len(parts) < 4 or parts[0] != "nvidia" or member.endswith("/"):
        return None
    rest = parts[2:]
    if ".." in rest:
        return None
    if rest[0] == "include":
        return root.joinpath(*rest)
    name = rest[-1]
    # Only NVRTC itself; the .alt variants are an unused alternate build.
    if ".alt." in name:
        return None
    if rest[0] == "lib" and os.name != "nt" and name.startswith("libnvrtc"):
        return root / "lib64" / name
    if rest[0] == "bin" and os.name == "nt" and name.lower().startswith("nvrtc") \
            and name.lower().endswith(".dll"):
        return root / "bin" / name
    return None


def install(major: int, progress: Callable[[str, float], None] | None = None) -> Path:
    """Download and unpack NVRTC + headers for CUDA `major`; returns the folder.

    `progress(message, fraction)` is called as it goes. The folder is built
    beside its final name and renamed into place, so an interrupted install
    never looks complete.
    """
    report = progress or (lambda message, fraction: None)
    root = cuda_dir(major)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"cuda-{major}-", dir=root.parent))
    try:
        wheels = []
        report("Looking up NVIDIA's packages…", 0.0)
        for name in PACKAGES[major]:
            wheels.append((name, *_wheel_for(name)))
        total = sum(size for *_, size in wheels) or 1
        done = 0
        versions = {}
        for name, url, digest, size in wheels:
            buffer = io.BytesIO()
            hasher = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=120) as response:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    buffer.write(chunk)
                    hasher.update(chunk)
                    done += len(chunk)
                    report(f"Downloading {name} ({done / 2**20:.0f} of "
                           f"{total / 2**20:.0f} MB)", 0.95 * done / total)
            if hasher.hexdigest() != digest:
                raise RuntimeError(f"{name} failed its checksum; try again")
            versions[name] = url.rsplit("/", 1)[-1]
            with zipfile.ZipFile(buffer) as archive:
                for member in archive.infolist():
                    destination = target_path(member.filename, staging)
                    if destination is None:
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, destination.open("wb") as sink:
                        shutil.copyfileobj(source, sink)
        report("Finishing…", 0.97)
        if not (staging / "include" / "cuda_fp16.h").is_file():
            raise RuntimeError("the downloaded packages had no cuda_fp16.h")
        (staging / MARKER).write_text(json.dumps({"major": major, "wheels": versions}),
                                      encoding="utf-8")
        if root.exists():
            shutil.rmtree(root)
        os.replace(staging, root)
        report("GPU support installed.", 1.0)
        return root
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def remove() -> None:
    for major in PACKAGES:
        shutil.rmtree(cuda_dir(major), ignore_errors=True)


class Installer:
    """`install` on a thread, with progress the page can poll."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.message = ""
        self.fraction = 0.0
        self.error = ""

    def start(self, major: int, done: Callable[[], None] | None = None) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running, self.message, self.fraction, self.error = True, "Starting…", 0.0, ""

        def progress(message: str, fraction: float) -> None:
            with self.lock:
                self.message, self.fraction = message, fraction

        def run() -> None:
            try:
                install(major, progress)
            except Exception as error:  # noqa: BLE001 - shown to the user
                with self.lock:
                    self.error = f"GPU setup failed: {error}"
            finally:
                with self.lock:
                    self.running = False
                if done:
                    done()

        threading.Thread(target=run, daemon=True).start()
        return True

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {"running": self.running, "message": self.message,
                    "fraction": self.fraction, "error": self.error}


def status_dict() -> dict[str, object]:
    return asdict(status())


if __name__ == "__main__":  # python -m flyweight.gpu_setup [install|remove]
    if sys.argv[1:] == ["remove"]:
        remove()
    elif sys.argv[1:] == ["install"]:
        current = status()
        if current.state != "needs-setup":
            print(current.title + ": " + current.detail)
        else:
            print(install(current.major, lambda m, f: print(f"{f:4.0%} {m}")))
    else:
        print(json.dumps(status_dict(), indent=2))
