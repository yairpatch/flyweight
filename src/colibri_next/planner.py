from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .backends import BackendDescriptor, BackendKind, discover_backends
from .hardware import HardwareTopology, MemoryTier, TierKind
from .models import MoEModelSpec


GIB = 1024**3
MIB = 1024**2


@dataclass(frozen=True, slots=True)
class PlacementDecision:
    group: str
    parameter_count: int
    byte_size: int
    quantization: str
    home_tier_id: str
    compute_backend_id: str
    cache_tier_id: str | None = None
    cache_bytes: int = 0


@dataclass(frozen=True, slots=True)
class PlacementPlan:
    model_id: str
    context_length: int
    supported: bool
    primary_backend_id: str
    estimated_model_bytes: int
    estimated_state_bytes: int
    decisions: tuple[PlacementDecision, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "context_length": self.context_length,
            "supported": self.supported,
            "primary_backend_id": self.primary_backend_id,
            "estimated_model_bytes": self.estimated_model_bytes,
            "estimated_state_bytes": self.estimated_state_bytes,
            "decisions": [asdict(decision) for decision in self.decisions],
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def write(self, path: Path | str) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


class PlacementPlanner:
    def __init__(self, topology: HardwareTopology):
        self.topology = topology
        self.backends = discover_backends(topology)

    def plan(
        self,
        model: MoEModelSpec,
        *,
        context_length: int = 32_768,
        expert_bits: int = 4,
    ) -> PlacementPlan:
        if context_length <= 0:
            raise ValueError("context length must be positive")
        if expert_bits not in {4, 8, 16}:
            raise ValueError("expert bits must be 4, 8, or 16")

        ram = self._largest_tier(TierKind.RAM)
        storage = self._largest_tier(TierKind.STORAGE)
        accelerator = self._best_accelerator()
        accelerator_tier = (
            self.topology.tier(accelerator.memory_tier_id) if accelerator else None
        )
        unified_accelerator = (
            accelerator is not None
            and accelerator_tier is not None
            and accelerator_tier.kind in {TierKind.RAM, TierKind.UNIFIED}
        )
        cpu = self._cpu_backend()
        primary = accelerator or cpu
        warnings: list[str] = []
        decisions: list[PlacementDecision] = []

        state_bytes = model.state_bytes_per_token * context_length
        workspace_bytes = 512 * MIB
        ram_reserve = max(2 * GIB, int(ram.capacity_bytes * 0.20))
        ram_budget = max(0, ram.available_bytes - ram_reserve)
        expert_bytes = model.bytes_for(model.routed_expert_parameters, expert_bits)

        accelerator_budget = 0
        if accelerator_tier is not None and accelerator_tier.kind == TierKind.VRAM:
            accelerator_reserve = max(GIB, int(accelerator_tier.capacity_bytes * 0.10))
            accelerator_budget = max(
                0,
                accelerator_tier.available_bytes
                - accelerator_reserve
                - state_bytes
                - workspace_bytes,
            )

        static_bits = 16
        static_bytes = model.bytes_for(model.static_parameters, static_bits)
        static_tier = ram
        static_backend = cpu
        if unified_accelerator:
            required_ram = static_bytes + state_bytes + workspace_bytes
            if required_ram > ram_budget:
                static_bits = 8
                static_bytes = model.bytes_for(model.static_parameters, static_bits)
                warnings.append("Static tensors reduced to INT8 to fit unified memory.")
            if static_bytes + state_bytes + workspace_bytes > ram_budget:
                return self._unsupported(
                    model,
                    context_length,
                    expert_bytes + static_bytes,
                    state_bytes,
                    primary,
                    "Insufficient unified memory for static tensors and runtime state.",
                    warnings,
                )
            static_backend = accelerator
            ram_budget -= static_bytes + state_bytes + workspace_bytes
        elif accelerator is not None and static_bytes <= accelerator_budget:
            static_tier = accelerator_tier
            static_backend = accelerator
            accelerator_budget -= static_bytes
        else:
            required_ram = static_bytes + state_bytes + workspace_bytes
            if required_ram > ram_budget:
                static_bits = 8
                static_bytes = model.bytes_for(model.static_parameters, static_bits)
                warnings.append("Static tensors reduced to INT8 to fit the available memory budget.")
            if static_bytes + state_bytes + workspace_bytes > ram_budget:
                return self._unsupported(
                    model,
                    context_length,
                    expert_bytes + static_bytes,
                    state_bytes,
                    primary,
                    "Insufficient RAM for static tensors and runtime state.",
                    warnings,
                )
            ram_budget -= static_bytes + state_bytes + workspace_bytes

        decisions.append(
            PlacementDecision(
                group="static_and_shared",
                parameter_count=model.static_parameters,
                byte_size=static_bytes,
                quantization=_quantization_name(static_bits),
                home_tier_id=static_tier.id,
                compute_backend_id=static_backend.id,
            )
        )

        active_layer_bytes = model.bytes_for(
            model.parameters_per_expert * model.routed_experts, expert_bits
        )
        expert_home = ram
        expert_backend = cpu
        cache_tier: MemoryTier | None = None
        cache_bytes = 0

        if unified_accelerator and expert_bytes <= ram_budget:
            expert_backend = accelerator
            ram_budget -= expert_bytes
        elif accelerator is not None and expert_bytes <= accelerator_budget:
            expert_home = accelerator_tier
            expert_backend = accelerator
            accelerator_budget -= expert_bytes
        else:
            if expert_bytes <= ram_budget:
                ram_budget -= expert_bytes
            else:
                expert_home = storage
                cache_tier = ram
                cache_bytes = max(0, ram_budget)
                ram_budget = 0
                warnings.append("Routed experts will stream from storage through the RAM cache.")

            if unified_accelerator and cache_bytes >= active_layer_bytes:
                expert_backend = accelerator
            elif accelerator is not None and accelerator_budget >= active_layer_bytes:
                expert_backend = accelerator
                cache_tier = accelerator_tier
                cache_bytes = accelerator_budget
            elif accelerator is not None:
                warnings.append(
                    "Accelerator memory cannot stage one active expert set; experts will execute on CPU."
                )

        if expert_home == storage and storage.bandwidth_gbps is None:
            warnings.append(
                "Storage bandwidth is unknown; benchmark it before expert streaming."
            )

        packaged_bytes = int((expert_bytes + static_bytes) * 1.05)
        if packaged_bytes > storage.available_bytes:
            return self._unsupported(
                model,
                context_length,
                expert_bytes + static_bytes,
                state_bytes,
                primary,
                "Insufficient storage for converted model weights.",
                warnings,
            )

        decisions.append(
            PlacementDecision(
                group="routed_experts",
                parameter_count=model.routed_expert_parameters,
                byte_size=expert_bytes,
                quantization=_quantization_name(expert_bits),
                home_tier_id=expert_home.id,
                compute_backend_id=expert_backend.id,
                cache_tier_id=cache_tier.id if cache_tier else None,
                cache_bytes=cache_bytes,
            )
        )
        state_tier = accelerator_tier if static_backend != cpu else ram
        decisions.append(
            PlacementDecision(
                group="runtime_state",
                parameter_count=0,
                byte_size=state_bytes + workspace_bytes,
                quantization="mixed",
                home_tier_id=state_tier.id,
                compute_backend_id=static_backend.id,
            )
        )

        if context_length > model.native_context_length:
            warnings.append("Requested context exceeds the model's native context length.")
        return PlacementPlan(
            model_id=model.id,
            context_length=context_length,
            supported=True,
            primary_backend_id=primary.id,
            estimated_model_bytes=expert_bytes + static_bytes,
            estimated_state_bytes=state_bytes + workspace_bytes,
            decisions=tuple(decisions),
            warnings=tuple(warnings),
        )

    def _largest_tier(self, kind: TierKind) -> MemoryTier:
        tiers = self.topology.tiers_of_kind(kind)
        if not tiers:
            raise ValueError(f"hardware profile has no {kind.value} tier")
        return max(tiers, key=lambda tier: tier.available_bytes)

    def _best_accelerator(self) -> BackendDescriptor | None:
        candidates = [backend for backend in self.backends if backend.kind != BackendKind.CPU]
        dedicated = [
            backend
            for backend in candidates
            if self.topology.tier(backend.memory_tier_id).kind == TierKind.VRAM
        ]
        pool = dedicated or candidates
        if not pool:
            return None
        return max(
            pool,
            key=lambda backend: self.topology.tier(backend.memory_tier_id).available_bytes,
        )

    def _cpu_backend(self) -> BackendDescriptor:
        return next(backend for backend in self.backends if backend.kind == BackendKind.CPU)

    def _unsupported(
        self,
        model: MoEModelSpec,
        context_length: int,
        model_bytes: int,
        state_bytes: int,
        primary: BackendDescriptor,
        reason: str,
        warnings: list[str],
    ) -> PlacementPlan:
        warnings.append(reason)
        return PlacementPlan(
            model_id=model.id,
            context_length=context_length,
            supported=False,
            primary_backend_id=primary.id,
            estimated_model_bytes=model_bytes,
            estimated_state_bytes=state_bytes + 512 * MIB,
            decisions=(),
            warnings=tuple(warnings),
        )


def _quantization_name(bits: int) -> str:
    return {4: "q4", 8: "int8", 16: "bf16"}[bits]
