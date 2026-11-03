"""Live machine readings for Flyweight Desktop's monitor: memory, CPU, GPU,
and the engine process's own footprint. Linux and Windows; every reading is
optional and comes back as None when the platform will not say."""
from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

_lock = threading.Lock()
_cpu_last: tuple[float, float] | None = None
_gpu_cache: tuple[float, list[dict[str, object]]] = (0.0, [])


def memory() -> dict[str, int | None]:
    """{'total', 'available'} in bytes."""
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
                return {"total": int(status.total), "available": int(status.available)}
        except (AttributeError, OSError):
            pass
        return {"total": None, "available": None}
    fields: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, _, rest = line.partition(":")
            if name in ("MemTotal", "MemAvailable"):
                fields[name] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return {"total": fields.get("MemTotal"), "available": fields.get("MemAvailable")}


def _cpu_times() -> tuple[float, float] | None:
    """(busy, total) seconds since boot, summed over cores."""
    if os.name == "nt":
        idle, kernel, user = (ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong())
        try:
            if not ctypes.windll.kernel32.GetSystemTimes(  # type: ignore[attr-defined]
                    ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                return None
        except (AttributeError, OSError):
            return None
        seconds = (kernel.value + user.value) / 1e7  # kernel time includes idle
        return seconds - idle.value / 1e7, seconds
    try:
        parts = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        numbers = [float(x) for x in parts]
    except (OSError, ValueError, IndexError):
        return None
    idle_time = numbers[3] + (numbers[4] if len(numbers) > 4 else 0.0)
    all_time = sum(numbers[:8])
    return all_time - idle_time, all_time


def cpu_percent() -> float | None:
    """Whole-machine CPU use since the previous call."""
    global _cpu_last
    now = _cpu_times()
    if now is None:
        return None
    with _lock:
        previous, _cpu_last = _cpu_last, now
    if previous is None or now[1] <= previous[1]:
        return None
    return max(0.0, min(100.0, 100.0 * (now[0] - previous[0]) / (now[1] - previous[1])))


def process_rss(pid: int | None) -> int | None:
    """Resident memory of a process, in bytes."""
    if not pid:
        return None
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong),
                        ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
                        ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                        ("d", ctypes.c_size_t), ("pagefile", ctypes.c_size_t),
                        ("peak_pagefile", ctypes.c_size_t)]
        try:
            kernel = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel.OpenProcess(0x1000 | 0x0010, False, pid)
            if not handle:
                return None
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
                handle, ctypes.byref(counters), counters.cb)
            kernel.CloseHandle(handle)
            return int(counters.working_set) if ok else None
        except (AttributeError, OSError):
            return None
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found is None and os.name == "nt":
        candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        found = str(candidate) if candidate.is_file() else None
    return found


def gpus(max_age: float = 1.5) -> list[dict[str, object]]:
    """[{name, used, total, util, temp, power}] -- cached briefly: nvidia-smi is slow."""
    global _gpu_cache
    with _lock:
        stamp, cached = _gpu_cache
    if time.monotonic() - stamp < max_age:
        return cached
    found: list[dict[str, object]] = []
    smi = _nvidia_smi()
    if smi:
        try:
            output = subprocess.run(
                [smi, "--query-gpu=name,memory.used,memory.total,utilization.gpu,"
                      "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
        except (OSError, subprocess.SubprocessError):
            output = ""
        for line in output.splitlines():
            cells = [c.strip() for c in line.split(",")]
            if len(cells) < 6:
                continue

            def number(text: str) -> float | None:
                return float(text) if re.fullmatch(r"[\d.]+", text) else None

            used, total = number(cells[1]), number(cells[2])
            found.append({"name": cells[0],
                          "used": int(used * 2**20) if used is not None else None,
                          "total": int(total * 2**20) if total is not None else None,
                          "util": number(cells[3]), "temp": number(cells[4]),
                          "power": number(cells[5])})
    if not found and os.name != "nt":
        for device in sorted(Path("/sys/class/drm").glob("card*/device")):
            try:
                total = int((device / "mem_info_vram_total").read_text())
                used = int((device / "mem_info_vram_used").read_text())
            except (OSError, ValueError):
                continue
            if total < 2**30:
                continue
            util = None
            try:
                util = float((device / "gpu_busy_percent").read_text())
            except (OSError, ValueError):
                pass
            found.append({"name": "AMD GPU", "used": used, "total": total, "util": util,
                          "temp": None, "power": None})
    with _lock:
        _gpu_cache = (time.monotonic(), found)
    return found


_ROW = re.compile(r"^(\d\d:\d\d:\d\d)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)(?:\s+(.*))?$")


def _compact_number(text: str) -> float | None:
    """'1.2k' -> 1200, '35%' -> 35, '2.4s' -> 2.4, '--' -> None."""
    match = re.fullmatch(r"([\d.]+)([kKmM%s]?)", text)
    if not match:
        return None
    value = float(match.group(1))
    return value * {"k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6}.get(match.group(2), 1)


def parse_request_row(line: str) -> dict[str, object] | None:
    """One row of the server's request table, or None for any other line."""
    match = _ROW.match(line.strip())
    if not match:
        return None
    time_, endpoint, prompt, cached, ttft, out, rate, finish, total, trailing = match.groups()
    if endpoint == "endpoint":
        return None
    return {"time": time_, "endpoint": endpoint, "prompt": _compact_number(prompt),
            "cached": _compact_number(cached), "ttft": _compact_number(ttft),
            "out": _compact_number(out), "rate": _compact_number(rate), "finish": finish,
            "total": _compact_number(total), "error": (trailing or "").strip()}


def snapshot(engine_pid: int | None) -> dict[str, object]:
    return {"memory": memory(), "cpu": cpu_percent(), "gpus": gpus(),
            "engine_rss": process_rss(engine_pid), "at": time.time()}
