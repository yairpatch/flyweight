from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .hardware import ComputeDevice, DeviceKind, HardwareTopology


class BackendKind(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"
    VULKAN = "vulkan"
    HIP = "hip"


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    id: str
    kind: BackendKind
    device_id: str
    memory_tier_id: str
    capabilities: tuple[str, ...]


class TensorBackend(ABC):
    """Execution boundary for future CPU, CUDA, Metal, Vulkan, and HIP kernels."""

    descriptor: BackendDescriptor

    @abstractmethod
    def allocate(self, byte_size: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def upload(self, destination: Any, source: memoryview) -> None:
        raise NotImplementedError

    @abstractmethod
    def execute_expert(self, expert: Any, hidden: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def synchronize(self) -> None:
        raise NotImplementedError


def discover_backends(topology: HardwareTopology) -> tuple[BackendDescriptor, ...]:
    return tuple(_descriptor_for(device) for device in topology.devices)


def _descriptor_for(device: ComputeDevice) -> BackendDescriptor:
    kind_by_device = {
        DeviceKind.CPU: BackendKind.CPU,
        DeviceKind.CUDA: BackendKind.CUDA,
        DeviceKind.METAL: BackendKind.METAL,
        DeviceKind.VULKAN: BackendKind.VULKAN,
        DeviceKind.HIP: BackendKind.HIP,
    }
    return BackendDescriptor(
        id=f"{device.id}:backend",
        kind=kind_by_device[device.kind],
        device_id=device.id,
        memory_tier_id=device.memory_tier_id,
        capabilities=device.capabilities,
    )
