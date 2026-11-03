"""Find, fit and download models for Flyweight Desktop.

Three parts, none of which needs the engine:

- `machine()` measures what this computer has: RAM, GPU memory, free disk.
- A catalog of recommended models (model_catalog.json, refreshed from the
  repository so it can be corrected without a release) names Hugging Face
  repositories and a preferred quantization. The actual file, its size and
  its checksum are looked up live, so the catalog never goes stale on a
  renamed file.
- `Downloads` fetches files one job at a time, resumably, verifying each
  against the SHA-256 Hugging Face publishes, into a visible folder the user
  can find and clean up.

`HF_ENDPOINT` (Hugging Face's own variable) points everything at a mirror.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

CATALOG_FILE = Path(__file__).with_name("model_catalog.json")
CATALOG_URL = ("https://raw.githubusercontent.com/yairpatch/flyweight/main/"
               "src/flyweight/model_catalog.json")
SIDECAR = "flyweight-model.json"
USER_AGENT = "flyweight-desktop"

# What the runtime will load (v2_runtime.cpp's own check, plus the two
# families with their own runtimes). Image-generation weights also start with
# "qwen" and are not chat models.
SUPPORTED_ARCHITECTURES = {
    "gemma4", "laguna", "muse-glimmer", "k2-horizon", "lfm2", "lfm2moe",
    "nemotron_h_moe", "bailingmoe3", "deepseek4",
}
_NOT_CHAT = re.compile(r"(image|dit|vae|clip|mmproj)", re.IGNORECASE)
_SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_QUANT = re.compile(r"(?:^|[-_.])((?:UD-)?(?:IQ\d_[A-Z0-9]+|Q\d_K(?:_[SMLX]{1,2})?|Q\d_\d|"
                    r"Q\d_K_XL|BF16|F16|F32|MXFP4(?:_MOE)?|NVFP4))(?=[-_.]|$)", re.IGNORECASE)


def endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


def architecture_support(architecture: str | None) -> str:
    """'yes', 'no' or 'unknown' for a GGUF general.architecture value."""
    if not architecture:
        return "unknown"
    name = architecture.lower()
    if name in SUPPORTED_ARCHITECTURES:
        return "yes"
    if name.startswith("qwen") and not _NOT_CHAT.search(name):
        return "no" if name.startswith("qwen2") else "yes"
    return "no"


def quant_of(filename: str) -> str:
    match = _QUANT.search(Path(filename).name.removesuffix(".gguf").removesuffix(".GGUF") + ".")
    return match.group(1).upper() if match else ""


# -- this computer -----------------------------------------------------------------


def _ram_bytes() -> int:
    if os.name == "nt":
        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                        ("page_total", ctypes.c_ulonglong), ("page_available", ctypes.c_ulonglong),
                        ("virtual_total", ctypes.c_ulonglong),
                        ("virtual_available", ctypes.c_ulonglong),
                        ("extended", ctypes.c_ulonglong)]
        status = Status()
        status.length = ctypes.sizeof(Status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return int(status.total)
        except (AttributeError, OSError):
            return 0
        return 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    sysconf = getattr(os, "sysconf", None)
    try:
        return int(sysconf("SC_PAGE_SIZE") * sysconf("SC_PHYS_PAGES")) if sysconf else 0
    except (ValueError, OSError):
        return 0


def _gpus() -> list[dict[str, object]]:
    """[{name, vram}] from nvidia-smi, else AMD's sysfs counters (Linux)."""
    found: list[dict[str, object]] = []
    smi = shutil.which("nvidia-smi")
    if smi is None and os.name == "nt":
        candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        smi = str(candidate) if candidate.is_file() else None
    if smi:
        try:
            output = subprocess.run(
                [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
            for line in output.splitlines():
                name, _, mib = line.rpartition(",")
                if name.strip() and mib.strip().isdigit():
                    found.append({"name": name.strip(), "vram": int(mib) * 2**20})
        except (OSError, subprocess.SubprocessError):
            pass
    if not found and os.name != "nt":
        for counter in sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_total")):
            try:
                vram = int(counter.read_text().strip())
            except (OSError, ValueError):
                continue
            if vram >= 2**30:  # an APU's sliver of carve-out is not a GPU to plan for
                found.append({"name": "AMD GPU", "vram": vram})
    return found


def default_models_dir() -> Path:
    return Path.home() / "Flyweight Models"


@dataclass
class Machine:
    ram: int
    vram: int
    gpu: str
    disk_free: int

    def summary(self) -> str:
        parts = [f"{self.ram / 2**30:.0f} GB memory"]
        if self.gpu:
            parts.append(f"{self.gpu} ({self.vram / 2**30:.0f} GB)")
        else:
            parts.append("no GPU")
        parts.append(f"{self.disk_free / 2**30:.0f} GB free disk")
        return " · ".join(parts)


def machine(models_dir: Path | None = None) -> Machine:
    gpus = _gpus()
    best = max(gpus, key=lambda g: int(g["vram"]), default=None)  # type: ignore[call-overload]
    folder = models_dir or default_models_dir()
    probe = folder
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        free = 0
    return Machine(_ram_bytes(), int(best["vram"]) if best else 0,  # type: ignore[call-overload]
                   str(best["name"]) if best else "", free)


def fit(size: int, moe: bool, m: Machine) -> dict[str, str]:
    """How a model of `size` bytes will run here: a level and a phrase.

    The runtime needs the weights plus a few GB for the context and its
    workspace. A dense model is fast only when it fits the GPU; a mixture of
    experts can keep its experts in RAM and still run well.
    """
    need = size * 1.1 + 1.5 * 2**30
    usable_ram = m.ram * 0.75
    if m.disk_free and size > m.disk_free - 2 * 2**30:
        return {"level": "no", "label": "Not enough disk space"}
    if m.vram and need <= m.vram * 0.92:
        return {"level": "great", "label": "Runs fast on your GPU"}
    if not moe and m.vram and need <= m.vram * 1.2 and need <= m.vram + usable_ram:
        # A few layers spill to the CPU: still most of the GPU's speed.
        return {"level": "good", "label": "Runs well (mostly on your GPU)"}
    if moe and m.vram and need <= m.vram + usable_ram:
        return {"level": "good", "label": "Runs well (GPU + memory)"}
    if not m.vram and need <= usable_ram:
        # Dense decode on a CPU is memory-bandwidth bound: comfortable up to
        # a ~4B model at Q4. A mixture of experts touches only a few billion
        # parameters per token, so even a large one stays quick.
        return ({"level": "good", "label": "Runs well on the CPU"} if size < 3.5 * 2**30 or moe
                else {"level": "slow", "label": "Slow on the CPU"})
    if need <= m.vram + usable_ram:
        return {"level": "slow", "label": "Runs, but slowly"}
    return {"level": "no", "label": "Too big for this computer"}


# -- Hugging Face ---------------------------------------------------------------------


class HubError(RuntimeError):
    pass


def _get_json(path: str, timeout: float = 20) -> object:
    request = urllib.request.Request(endpoint() + path, headers={"User-Agent": USER_AGENT})
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise HubError("this model needs a Hugging Face login (it is gated)") from None
        if error.code == 404:
            raise HubError("not found on Hugging Face") from None
        raise HubError(f"Hugging Face answered {error.code}") from None
    except (OSError, ValueError) as error:
        raise HubError(f"cannot reach Hugging Face ({error})") from None


@dataclass
class HubFile:
    path: str
    size: int
    sha256: str
    quant: str = ""


@dataclass
class Variant:
    """One downloadable model: its shard(s) of a quantization."""

    quant: str
    files: list[HubFile]
    size: int


def repo_files(repo: str) -> list[HubFile]:
    quoted = urllib.parse.quote(repo, safe="/")
    data = _get_json(f"/api/models/{quoted}/tree/main?recursive=true")
    files: list[HubFile] = []
    if not isinstance(data, list):
        return files
    for entry in data:
        if not isinstance(entry, dict) or entry.get("type") != "file":
            continue
        path = str(entry.get("path", ""))
        if not path.lower().endswith(".gguf"):
            continue
        lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else {}
        size = int((lfs or {}).get("size") or entry.get("size") or 0)
        sha = str((lfs or {}).get("oid") or "")
        files.append(HubFile(path, size, sha if re.fullmatch(r"[0-9a-f]{64}", sha) else "",
                             quant_of(path)))
    return files


def variants(files: list[HubFile]) -> tuple[list[Variant], list[HubFile]]:
    """Group a repository's GGUFs into whole models; split out projectors."""
    projectors = [f for f in files if "mmproj" in Path(f.path).name.lower()]
    groups: dict[str, list[HubFile]] = {}
    for f in files:
        if f in projectors:
            continue
        name = Path(f.path).name
        split = _SPLIT.search(name)
        key = str(Path(f.path).parent / _SPLIT.sub("", name)) if split else f.path
        groups.setdefault(key, []).append(f)
    found = []
    for members in groups.values():
        members.sort(key=lambda f: f.path)
        first = _SPLIT.search(Path(members[0].path).name)
        if first and (first.group(1) != "00001" or len(members) != int(first.group(2))):
            continue  # an incomplete split set is not a model
        found.append(Variant(members[0].quant or quant_of(members[0].path), members,
                             sum(f.size for f in members)))
    found.sort(key=lambda v: v.size)
    return found, projectors


def pick_variant(found: list[Variant], preferences: list[str]) -> Variant | None:
    for wanted in preferences:
        for variant in found:
            if variant.quant.upper() == wanted.upper():
                return variant
    middle = [v for v in found if v.quant.startswith(("Q4", "IQ4"))]
    choices = middle or found
    return choices[0] if choices else None


def pick_projector(projectors: list[HubFile]) -> HubFile | None:
    for wanted in ("F16", "BF16", "Q8_0", "F32"):
        for f in projectors:
            if wanted.lower() in Path(f.path).name.lower():
                return f
    return projectors[0] if projectors else None


def search(query: str, limit: int = 20) -> list[dict[str, object]]:
    params = urllib.parse.urlencode([("search", query), ("filter", "gguf"),
                                     ("sort", "downloads"), ("direction", "-1"),
                                     ("limit", str(limit)), ("full", "false")])
    expanded = params + "".join(f"&expand%5B%5D={key}" for key in
                                ("downloads", "likes", "gguf", "gated", "lastModified"))
    try:
        data = _get_json(f"/api/models?{expanded}")
    except HubError:
        data = _get_json(f"/api/models?{params}")  # an endpoint without expand[]
    results = []
    for entry in data if isinstance(data, list) else []:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        gguf = entry.get("gguf") if isinstance(entry.get("gguf"), dict) else {}
        architecture = str((gguf or {}).get("architecture") or "")
        results.append({
            "repo": entry["id"],
            "downloads": int(entry.get("downloads") or 0),
            "likes": int(entry.get("likes") or 0),
            "architecture": architecture,
            "supported": architecture_support(architecture),
            "gated": bool(entry.get("gated")),
            "params": int((gguf or {}).get("total") or 0),
        })
    return results


def repo_details(repo: str, m: Machine) -> dict[str, object]:
    """Everything the page needs to offer a repository's files."""
    quoted = urllib.parse.quote(repo, safe="/")
    info = _get_json(f"/api/models/{quoted}?expand%5B%5D=gguf&expand%5B%5D=gated")
    gguf = info.get("gguf") if isinstance(info, dict) and isinstance(info.get("gguf"), dict) else {}
    architecture = str((gguf or {}).get("architecture") or "")
    found, projectors = variants(repo_files(repo))
    moe = "moe" in architecture or bool(re.search(r"-A\d+(\.\d+)?B", repo, re.IGNORECASE))
    return {
        "repo": repo,
        "architecture": architecture,
        "supported": architecture_support(architecture),
        "gated": bool(isinstance(info, dict) and info.get("gated")),
        "moe": moe,
        "variants": [{"quant": v.quant or Path(v.files[0].path).name, "size": v.size,
                      "files": [asdict(f) for f in v.files], "fit": fit(v.size, moe, m)}
                     for v in found],
        "projector": asdict(p) if (p := pick_projector(projectors)) else None,
    }


# -- the catalog -------------------------------------------------------------------------


def _valid_catalog(data: object) -> bool:
    return (isinstance(data, dict) and isinstance(data.get("models"), list)
            and all(isinstance(e, dict) and isinstance(e.get("repo"), str)
                    and isinstance(e.get("name"), str) for e in data["models"]))


def load_catalog(cache: Path | None = None, fetch: bool = True) -> list[dict[str, object]]:
    """The repository's current list, else the last one fetched, else the bundled one."""
    if fetch:
        try:
            request = urllib.request.Request(CATALOG_URL, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.load(response)
            if _valid_catalog(data):
                if cache:
                    try:
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        cache.write_text(json.dumps(data), encoding="utf-8")
                    except OSError:
                        pass
                return list(data["models"])
        except (OSError, ValueError):
            pass
    for source in (cache, CATALOG_FILE):
        if source is None:
            continue
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if _valid_catalog(data):
            return list(data["models"])
    return []


def resolve_entry(entry: dict[str, object], m: Machine) -> dict[str, object]:
    """A catalog entry with its real file, size and fit filled in from the Hub."""
    resolved = dict(entry)
    try:
        found, projectors = variants(repo_files(str(entry["repo"])))
    except HubError as error:
        resolved.update(available=False, error=str(error))
        return resolved
    quants = entry.get("quants")
    preferences = [str(q) for q in quants] if isinstance(quants, list) else []
    variant = pick_variant(found, preferences or ["Q4_K_M"])
    if variant is None:
        resolved.update(available=False, error="no GGUF files in this repository")
        return resolved
    projector = pick_projector(projectors) if entry.get("vision") else None
    size = variant.size + (projector.size if projector else 0)
    resolved.update(
        available=True, quant=variant.quant, size=size,
        files=[asdict(f) for f in variant.files] + ([asdict(projector)] if projector else []),
        fit=fit(variant.size, bool(entry.get("moe")), m),
    )
    return resolved


# -- the library on disk ---------------------------------------------------------------------


def repo_folder(models_dir: Path, repo: str) -> Path:
    return models_dir / repo.replace("/", "__")


def library(models_dir: Path) -> list[dict[str, object]]:
    """Downloaded models: each folder's first shard, newest first."""
    found: list[dict[str, object]] = []
    try:
        folders = [p for p in models_dir.iterdir() if p.is_dir()]
    except OSError:
        return found
    for folder in folders:
        meta: dict[str, object] = {}
        try:
            meta = json.loads((folder / SIDECAR).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        for gguf in sorted(folder.rglob("*.gguf")):
            name = gguf.name.lower()
            split = _SPLIT.search(gguf.name)
            if "mmproj" in name or (split and split.group(1) != "00001"):
                continue
            shards = sorted(gguf.parent.glob(_SPLIT.sub("", gguf.name) + "-*-of-*.gguf")) \
                if split else [gguf]
            try:
                size = sum(s.stat().st_size for s in shards)
                modified = gguf.stat().st_mtime
            except OSError:
                continue
            found.append({"path": str(gguf), "name": str(meta.get("name") or gguf.stem),
                          "file": gguf.name, "repo": str(meta.get("repo") or ""),
                          "size": size, "folder": str(folder), "modified": modified})
    found.sort(key=lambda m: float(m["modified"]), reverse=True)  # type: ignore[arg-type]
    return found


def delete_model(models_dir: Path, folder: str) -> None:
    """Remove a downloaded model's folder -- only ever inside models_dir."""
    target = Path(folder).resolve()
    if target.parent != models_dir.resolve():
        raise ValueError("only models Flyweight downloaded can be deleted here")
    shutil.rmtree(target)


def reveal(path: str) -> None:
    folder = Path(path) if Path(path).is_dir() else Path(path).parent
    if os.name == "nt":
        os.startfile(folder)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)


# -- downloading ------------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    repo: str
    name: str
    files: list[dict[str, object]]
    total: int
    done: int = 0
    state: str = "queued"     # queued, downloading, verifying, paused, done, failed
    error: str = ""
    speed: float = 0.0
    result: str = ""          # the path to load, once done
    started: float = field(default_factory=time.time)


class Downloads:
    """A one-at-a-time download queue with resume and verification."""

    CHUNK = 1 << 20

    def __init__(self, models_dir: Callable[[], Path],
                 on_done: Callable[[Job], None] | None = None) -> None:
        self.models_dir = models_dir
        self.on_done = on_done
        self.lock = threading.Lock()
        self.jobs: list[Job] = []
        self.wake = threading.Event()
        self.cancelled: set[str] = set()
        threading.Thread(target=self._worker, daemon=True).start()

    def add(self, repo: str, name: str, files: list[dict[str, object]]) -> Job:
        clean = []
        for f in files:
            path = str(f.get("path", ""))
            if not path or ".." in Path(path).parts or path.startswith(("/", "\\")):
                raise ValueError(f"refusing the file name {path!r}")
            clean.append({"path": path, "size": int(f.get("size") or 0),  # type: ignore[call-overload]
                          "sha256": str(f.get("sha256") or "")})
        with self.lock:
            for job in self.jobs:
                if job.repo == repo and [f["path"] for f in job.files] == [f["path"] for f in clean] \
                        and job.state in ("queued", "downloading", "verifying", "paused"):
                    job.state = "queued" if job.state == "paused" else job.state
                    self.wake.set()
                    return job
            job = Job(f"{int(time.time() * 1000)}-{len(self.jobs)}", repo, name, clean,
                      sum(int(f["size"]) for f in clean))  # type: ignore[call-overload]
            self.jobs.append(job)
        self.wake.set()
        return job

    def pause(self, job_id: str) -> None:
        with self.lock:
            for job in self.jobs:
                if job.id == job_id and job.state in ("queued", "downloading", "verifying"):
                    self.cancelled.add(job_id)
                    if job.state == "queued":
                        job.state = "paused"

    def resume(self, job_id: str) -> None:
        with self.lock:
            for job in self.jobs:
                if job.id == job_id and job.state in ("paused", "failed"):
                    job.state, job.error = "queued", ""
                    self.cancelled.discard(job_id)
        self.wake.set()

    def remove(self, job_id: str) -> None:
        """Cancel and forget a job, deleting its partial files."""
        self.pause(job_id)
        with self.lock:
            job = next((j for j in self.jobs if j.id == job_id), None)
        if job is None:
            return
        for _ in range(50):  # let the worker notice and close its file
            if job.state != "downloading" and job.state != "verifying":
                break
            time.sleep(0.1)
        folder = repo_folder(self.models_dir(), job.repo)
        for f in job.files:
            part = folder / (str(f["path"]) + ".part")
            part.unlink(missing_ok=True)
        with self.lock:
            self.jobs = [j for j in self.jobs if j.id != job_id]

    def snapshot(self) -> list[dict[str, object]]:
        with self.lock:
            return [asdict(job) for job in self.jobs if job.state != "done"
                    or time.time() - job.started < 24 * 3600]

    # -- the worker --------------------------------------------------------------------

    def _next(self) -> Job | None:
        with self.lock:
            return next((j for j in self.jobs if j.state == "queued"), None)

    def _worker(self) -> None:
        while True:
            job = self._next()
            if job is None:
                self.wake.wait()
                self.wake.clear()
                continue
            try:
                self._run(job)
            except _Paused:
                with self.lock:
                    job.state, job.speed = "paused", 0.0
                    self.cancelled.discard(job.id)
            except Exception as error:  # noqa: BLE001 - shown to the user
                with self.lock:
                    job.state, job.error, job.speed = "failed", _plain(error), 0.0
            else:
                if self.on_done:
                    self.on_done(job)

    def _run(self, job: Job) -> None:
        folder = repo_folder(self.models_dir(), job.repo)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / SIDECAR).write_text(json.dumps({"repo": job.repo, "name": job.name,
                                                  "files": job.files}), encoding="utf-8")
        with self.lock:
            job.state = "downloading"
            job.done = sum(_existing(folder, f) for f in job.files)
        for f in job.files:
            final = folder / str(f["path"])
            if final.is_file() and (not f["size"] or final.stat().st_size == f["size"]):
                continue
            self._fetch(job, f, folder)
        with self.lock:
            job.state, job.speed = "done", 0.0
            main = folder / str(job.files[0]["path"])
            job.result = str(main)

    def _fetch(self, job: Job, f: dict[str, object], folder: Path) -> None:
        final = folder / str(f["path"])
        part = final.with_name(final.name + ".part")
        part.parent.mkdir(parents=True, exist_ok=True)
        size = int(f["size"])  # type: ignore[call-overload]
        hasher = hashlib.sha256()
        have = part.stat().st_size if part.exists() else 0
        if size and have > size:
            part.unlink()
            have = 0
        if have:  # resume: the hash has to cover what is already there
            with part.open("rb") as existing:
                for chunk in iter(lambda: existing.read(self.CHUNK), b""):
                    hasher.update(chunk)
        if not size or have < size:
            url = f"{endpoint()}/{urllib.parse.quote(job.repo, safe='/')}/resolve/main/" \
                  f"{urllib.parse.quote(str(f['path']))}?download=true"
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            if have:
                request.add_header("Range", f"bytes={have}-")
            token = os.environ.get("HF_TOKEN")
            if token:
                request.add_header("Authorization", f"Bearer {token}")
            try:
                response = urllib.request.urlopen(request, timeout=60)
            except urllib.error.HTTPError as error:
                if error.code == 416 and size and have == size:
                    response = None
                elif error.code in (401, 403):
                    raise RuntimeError("this model needs a Hugging Face login (it is gated)") from None
                else:
                    raise RuntimeError(f"download failed: HTTP {error.code}") from None
            if response is not None:
                with response:
                    if have and response.status != 206:  # the server ignored Range
                        hasher = hashlib.sha256()
                        with self.lock:
                            job.done -= have
                        have = 0
                    mode = "ab" if have else "wb"
                    window_start, window_bytes = time.monotonic(), 0
                    with part.open(mode) as sink:
                        while True:
                            if job.id in self.cancelled:
                                raise _Paused()
                            chunk = response.read(self.CHUNK)
                            if not chunk:
                                break
                            sink.write(chunk)
                            hasher.update(chunk)
                            window_bytes += len(chunk)
                            now = time.monotonic()
                            with self.lock:
                                job.done += len(chunk)
                                if now - window_start >= 1.0:
                                    rate = window_bytes / (now - window_start)
                                    job.speed = rate if not job.speed else job.speed * 0.6 + rate * 0.4
                                    window_start, window_bytes = now, 0
        with self.lock:
            job.state = "verifying"
        if size and part.stat().st_size != size:
            raise RuntimeError(f"{final.name} arrived incomplete; press Resume to continue")
        expected = str(f.get("sha256") or "")
        if expected and hasher.hexdigest() != expected:
            part.unlink(missing_ok=True)
            with self.lock:
                job.done = max(0, job.done - size)
            raise RuntimeError(f"{final.name} failed its checksum and was deleted; try again")
        os.replace(part, final)
        with self.lock:
            job.state = "downloading"


class _Paused(Exception):
    pass


def _existing(folder: Path, f: dict[str, object]) -> int:
    final = folder / str(f["path"])
    if final.is_file():
        return final.stat().st_size
    part = final.with_name(final.name + ".part")
    return part.stat().st_size if part.is_file() else 0


def _plain(error: BaseException) -> str:
    if isinstance(error, urllib.error.URLError):
        return f"cannot reach Hugging Face ({error.reason}); check the connection and press Resume"
    if isinstance(error, OSError) and getattr(error, "errno", None) == 28:
        return "the disk is full; free some space and press Resume"
    return str(error)
