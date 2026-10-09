"""The placement report: where a prepared runtime put its weights and arenas.

Read back from the native runtime after prepare, so it describes what was
allocated rather than what a planner predicted. The device rows are the
runtime's own arenas; "other" is what the card lost beyond them while prepare
ran (cuBLAS workspaces and the like), measured from free VRAM either side --
so another process allocating on the same card during the load lands in it too.
"""

from __future__ import annotations

from collections.abc import Mapping

from .v2 import CACHE_TYPE_NAMES

_MIB = 1024 * 1024
_EXPERT_MODES = {0: "gpu (streamed)", 1: "cpu", 2: "auto (hybrid)"}
_CACHE_TYPES = CACHE_TYPE_NAMES
_NO_WHOLE_LAYERS = 0xFFFFFFFF

_DEVICE_ROWS = (
    ("static_weights_bytes", "weights"),
    ("kv_state_bytes", "KV + recurrent state"),
    ("snapshot_bytes", "prefix checkpoints"),
    ("expert_cache_bytes", "expert cache"),
    ("value_expert_bytes", "value experts"),
    ("workspace_bytes", "workspace"),
    ("vision_workspace_bytes", "vision workspace"),
    ("prefill_stream_bytes", "prefill expert stream"),
    ("expert_staging_bytes", "expert staging"),
    ("host_ffn_stage_bytes", "spilled FFN staging"),
    ("turbo_kv_bytes", "turbo KV staging"),
    ("embedding_stage_bytes", "embedding staging"),
)

_HOST_ROWS = (
    ("host_ffn_bytes", "spilled dense FFN (mapped)"),
    ("host_ffn_reencoded_bytes", "spilled FFN re-encode"),
    ("host_pinned_bytes", "pinned staging"),
    ("prompt_cache_limit_bytes", "prompt cache budget"),
)


def _mib(value: int) -> str:
    return f"{value / _MIB:>9,.0f} MiB"


def device_total(plan: Mapping[str, int]) -> int:
    return sum(plan[key] for key, _ in _DEVICE_ROWS)


def format_placement(plan: Mapping[str, int]) -> str:
    lines: list[str] = []
    budget = plan["gpu_budget_bytes"]
    total = plan["gpu_total_bytes"]
    if total:
        source = "auto-fit" if plan["gpu_budget_auto"] else "--gpu-cache-mib"
        lines.append(
            f"GPU  {total / _MIB:,.0f} MiB card, budget {budget / _MIB:,.0f} MiB ({source})"
        )
    else:
        lines.append("GPU  none (CPU backend)")
    itemised = device_total(plan)
    for key, label in _DEVICE_ROWS:
        if plan[key]:
            lines.append(f"  {label:<26}{_mib(plan[key])}")
    lines.append(f"  {'runtime arenas':<26}{_mib(itemised)}")
    measured = plan.get("gpu_used_by_prepare_bytes")
    if measured is not None:
        lines.append(f"  {'other (measured)':<26}{_mib(max(0, measured - itemised))}")
    if total:
        lines.append(f"  {'free after load':<26}{_mib(plan['gpu_free_bytes'])}")

    lines.append("RAM")
    if plan["expert_weight_bytes"]:
        lines.append(f"  {'expert weights (mapped)':<26}{_mib(plan['expert_weight_bytes'])}")
    for key, label in _HOST_ROWS:
        if plan[key]:
            lines.append(f"  {label:<26}{_mib(plan[key])}")
    if plan.get("preload_expert_bytes"):
        lines.append(f"  {'expert preload (bg)':<26}{_mib(plan['preload_expert_bytes'])}")

    lines.append("Layers")
    layers = plan["layers"]
    if plan["dense_ffn_layers"]:
        on_gpu = plan["dense_ffn_layers"] - plan["host_ffn_layers"]
        lines.append(
            f"  dense FFN: {on_gpu} of {plan['dense_ffn_layers']} blocks on GPU, "
            f"{plan['host_ffn_layers']} on CPU"
        )
    if plan["moe_layers"]:
        mode = _EXPERT_MODES.get(plan["expert_mode"], str(plan["expert_mode"]))
        if plan["gpu_expert_layers"] != _NO_WHOLE_LAYERS:
            placed = (
                f"{plan['gpu_expert_layers']} of {plan['moe_layers']} layers' experts "
                f"pinned on GPU, the rest on CPU"
            )
        elif plan["expert_mode"] == 1 or not plan["expert_cache_slots"]:
            placed = "all on CPU"
        else:
            share = plan["expert_cache_slots"] / max(1, plan["expert_bundles"])
            placed = (
                f"cache holds {plan['expert_cache_slots']:,} of "
                f"{plan['expert_bundles']:,} ({share:.0%}), misses on CPU"
            )
        lines.append(f"  experts [{mode}]: {placed}")
    lines.append(
        f"  attention: all {layers} layers on GPU, context {plan['context_limit']:,}, KV "
        f"{_CACHE_TYPES.get(plan['cache_type_k'], '?')}/"
        f"{_CACHE_TYPES.get(plan['cache_type_v'], '?')}"
        + (f", {plan['parallel_sequences']} slots" if plan["parallel_sequences"] > 1 else "")
    )
    return "\n".join(lines)
