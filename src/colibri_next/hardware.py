from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class TierKind(StrEnum):
    STORAGE = "storage"
    RAM = "ram"
    PINNED_RAM = "pinned_ram"
    VRAM = "vram"
    UNIFIED = "unified"


class DeviceKind(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"
    VULKAN = "vulkan"
    HIP = "hip"


@dataclass(frozen=True, slots=True)
class MemoryTier:
    id: str
    kind: TierKind
    capacity_bytes: int
    available_bytes: int
    bandwidth_gbps: float | None = None
    latency_us: float | None = None
    device_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ComputeDevice:
    id: str
    kind: DeviceKind
    name: str
    memory_tier_id: str
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HardwareTopology:
    system: str
    architecture: str
    cpu_name: str
    cpu_threads: int
    tiers: tuple[MemoryTier, ...]
    devices: tuple[ComputeDevice, ...]

    def tier(self, tier_id: str) -> MemoryTier:
        for tier in self.tiers:
            if tier.id == tier_id:
                return tier
        raise KeyError(f"unknown memory tier: {tier_id}")

    def tiers_of_kind(self, kind: TierKind) -> tuple[MemoryTier, ...]:
        return tuple(tier for tier in self.tiers if tier.kind == kind)

    def accelerator_devices(self) -> tuple[ComputeDevice, ...]:
        return tuple(device for device in self.devices if device.kind != DeviceKind.CPU)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "architecture": self.architecture,
            "cpu_name": self.cpu_name,
            "cpu_threads": self.cpu_threads,
            "tiers": [asdict(tier) for tier in self.tiers],
            "devices": [asdict(device) for device in self.devices],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HardwareTopology":
        tiers = tuple(
            MemoryTier(
                id=tier["id"],
                kind=TierKind(tier["kind"]),
                capacity_bytes=int(tier["capacity_bytes"]),
                available_bytes=int(tier["available_bytes"]),
                bandwidth_gbps=tier.get("bandwidth_gbps"),
                latency_us=tier.get("latency_us"),
                device_id=tier.get("device_id"),
                metadata=dict(tier.get("metadata", {})),
            )
            for tier in value["tiers"]
        )
        devices = tuple(
            ComputeDevice(
                id=device["id"],
                kind=DeviceKind(device["kind"]),
                name=device["name"],
                memory_tier_id=device["memory_tier_id"],
                capabilities=tuple(device.get("capabilities", ())),
            )
            for device in value["devices"]
        )
        return cls(
            system=value["system"],
            architecture=value["architecture"],
            cpu_name=value["cpu_name"],
            cpu_threads=int(value["cpu_threads"]),
            tiers=tiers,
            devices=devices,
        )

    @classmethod
    def from_json_file(cls, path: Path | str) -> "HardwareTopology":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def probe_hardware(storage_path: Path | str = ".") -> HardwareTopology:
    storage_root = Path(storage_path).resolve()
    ram_capacity, ram_available = _probe_ram()
    disk = shutil.disk_usage(storage_root)
    tiers: list[MemoryTier] = [
        MemoryTier(
            id="storage:0",
            kind=TierKind.STORAGE,
            capacity_bytes=disk.total,
            available_bytes=disk.free,
            metadata={"path": str(storage_root), "medium": "unknown"},
        ),
        MemoryTier(
            id="ram:0",
            kind=TierKind.RAM,
            capacity_bytes=ram_capacity,
            available_bytes=ram_available,
        ),
    ]
    architecture = platform.machine().lower() or "unknown"
    devices: list[ComputeDevice] = [
        ComputeDevice(
            id="cpu:0",
            kind=DeviceKind.CPU,
            name=platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "CPU"),
            memory_tier_id="ram:0",
            capabilities=_cpu_capabilities(architecture),
        )
    ]
    gpu_tiers, gpu_devices = _probe_nvidia()
    tiers.extend(gpu_tiers)
    devices.extend(gpu_devices)

    if platform.system() == "Darwin" and architecture in {"arm64", "aarch64"}:
        devices.append(
            ComputeDevice(
                id="metal:0",
                kind=DeviceKind.METAL,
                name="Apple Silicon GPU",
                memory_tier_id="ram:0",
                capabilities=("unified_memory", "fp16"),
            )
        )

    return HardwareTopology(
        system=platform.system() or "unknown",
        architecture=architecture,
        cpu_name=devices[0].name,
        cpu_threads=os.cpu_count() or 1,
        tiers=tuple(tiers),
        devices=tuple(devices),
    )


def available_ram_bytes() -> int:
    return _probe_ram()[1]


def _probe_ram() -> tuple[int, int]:
    if platform.system() == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical), int(status.available_physical)

    page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    pages = os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else 0
    available_pages = os.sysconf("SC_AVPHYS_PAGES") if hasattr(os, "sysconf") else pages
    return int(page_size * pages), int(page_size * available_pages)


def _cpu_capabilities(architecture: str) -> tuple[str, ...]:
    capabilities = [architecture]
    if architecture in {"arm64", "aarch64"}:
        capabilities.append("neon")
    return tuple(capabilities)


def _probe_nvidia() -> tuple[list[MemoryTier], list[ComputeDevice]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return [], []
    command = [
        executable,
        "--query-gpu=index,name,memory.total,memory.free,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    creation_flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.SubprocessError):
        return [], []

    tiers: list[MemoryTier] = []
    devices: list[ComputeDevice] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        index, name, total_mib, free_mib, compute_capability = fields
        tier_id = f"cuda:{index}:vram"
        device_id = f"cuda:{index}"
        tiers.append(
            MemoryTier(
                id=tier_id,
                kind=TierKind.VRAM,
                capacity_bytes=int(float(total_mib) * 1024**2),
                available_bytes=int(float(free_mib) * 1024**2),
                device_id=device_id,
            )
        )
        devices.append(
            ComputeDevice(
                id=device_id,
                kind=DeviceKind.CUDA,
                name=name,
                memory_tier_id=tier_id,
                capabilities=("fp16", "int8", "int4", f"sm_{compute_capability}"),
            )
        )
    return tiers, devices
