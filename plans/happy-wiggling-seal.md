# Configurable KV cache precision (llama.cpp `-ctk`/`-ctv` parity)

## Context

The v2 native runtime stores its attention KV cache in **f32**, which is 2× heavier
than llama.cpp's `f16` default. Only 11 of 40 layers are full-attention (the other
30 are DeltaNet with a constant recurrent state), but each KV entry is chunky
(kv_heads=2, head_dim=256), so f32 KV costs **~44 KB/token → 5.58 GB @ 132k ctx**.
On the 11 GB laptop GPU that does not fit alongside static weights + the 6 GB expert
cache + workspace, which is why long context is a problem for v2 today while
llama.cpp runs 132k comfortably.

Goal: offer the same configurable KV cache precision llama.cpp does — independent
`--cache-type-k` / `--cache-type-v` (aliases of `-ctk`/`-ctv`), default `f16`.
Full target set: `f32, f16, bf16, q8_0, q5_0, q5_1, q4_0, q4_1, iq4_nl`. Built in
verified phases (each type diffed against the f32 reference before the next), since
a prior unverified Q8-KV attempt was dropped as likely-incorrect.

`f16` is the phase that matters most: it halves KV VRAM (2.79 GB @ 132k) and unlocks
long context, at negligible quality cost.

## Status: Phase 1 (f32 + f16) is CODE-COMPLETE but UNVERIFIED

All edits below are already applied to the working tree (uncommitted, on top of
`cd0b01e`). The C++ **compiles** (`python -m flyweight.native_build` reached
`Built target flyweight_v2`). It has NOT been run — nvrtc compiles the CUDA kernels
at runtime, and the f16-vs-f32 correctness A/B could not be executed (environment
Bash outage). **Do not trust it until the verification below passes.**

### What Phase 1 changed

- **KV codec + templated kernels** — `native/include/flyweight_v2_qwen_kernels.hpp`:
  added `kv_ld`/`kv_st` overloads (float/`__half`); made `kv_attention_scores`
  (K-type), `kv_attention_values` (V-type), `kv_append` (K×V) templated, emitting
  `*_f16*` variants. `native/include/flyweight_v2_native_kernels.hpp`: same for the
  fused `kv_attention_prefill` (K×V combos).
- **Kernel registration** — `native/src/gpu_driver.cpp`: added the new `_f16`
  variant names to the soft (non-fatal) `cuModuleGetFunction` lookup list.
- **State layout** — `native/src/v2_runtime.cpp` (runtime prepare, ~line 945):
  refactored the state arena from an f32-count accumulator to a **byte cursor**
  (`reserve()` lambda, 16-byte aligned regions) so attention KV sizes per type
  (`kv_k`/`kv_v` = 2 or 4 bytes/elem) while DeltaNet state stays f32.
- **Options + guards** — `native/include/flyweight_v2.h`: `cache_type_k`,
  `cache_type_v` on `FlyweightV2QwenRuntimeOptions`. `v2_runtime.cpp` create:
  range-check (0/1), and MTP is guarded to f32-only for now.
- **Kernel selection helpers** — `v2_runtime.cpp`: `kv_append_kernel` /
  `kv_scores_kernel` / `kv_values_kernel` / `kv_prefill_kernel` chosen by the
  configured type; wired at the main-decode launches and the prefill-rows launches
  in `native/src/v2_mtp_verifier.inc`. (MTP-path launches left on f32 kernels,
  matching the guard.)
- **Python + CLI** — `src/flyweight/v2.py` (`_QwenRuntimeOptions` ctypes fields +
  `native_qwen_runtime` / `V2QwenRuntime` params, string→enum `{f32:0, f16:1}`);
  `src/flyweight/v2_server.py` (`NativeV2InferenceService`); `src/flyweight/cli.py`
  (`serve-v2 --cache-type-k/--cache-type-v`, default `f16`).

## Verification (run first, before committing Phase 1)

1. `python -m flyweight.native_build` — reconfirm the build.
2. **nvrtc + correctness A/B** (`/tmp/kvverify.py`, self-contained): build a runtime
   with `cache_type_k/v="f32"` and another with `"f16"` on the same prompt; assert
   (a) both prepare without an nvrtc compile error, (b) f16 `state_bytes` <  f32
   `state_bytes` (attention KV halved), (c) greedy tokens match for a short answer
   ("capital of France" → Paris); small divergence only deep into long outputs is
   acceptable f16 rounding.
3. Also run `moe_device="cpu"`/`"gpu"` once and a >1 chunk prefill (long prompt) to
   exercise `kv_attention_prefill_f16_f16` and the per-token fallback.
4. Run `tests/test_v2_server.py tests/test_v2.py` (stub-level, fast) to confirm no
   regression, plus the existing reuse-path check (same prompt twice).
5. If green, commit Phase 1.

## Follow-on phases (each: implement → diff vs f32 → commit)

2. **bf16** — trivial: add `__nv_bfloat16` overloads to `kv_ld`/`kv_st` (needs
   `cuda_bf16.h`; prepare already adds cuda include paths), emit `_bf16` kernel
   variants, extend the enum/validation/registration. Same size as f16.
3. **q8_0** — first blocked-quant type (32-elem blocks + f16 scale). `kv_st` writes
   a block scale + int8; scores/values/prefill read blocks. K is straightforward
   (score dot); V needs the read kernels to dequant blocks. This is where the
   correctness diff matters most.
4. **q4_0 / q4_1 / q5_0 / q5_1** — more blocked variants, incremental once q8_0's
   block plumbing exists.
5. **iq4_nl** — nonlinear codebook; last, rarely used.

Enum values (keep stable, mirror a fixed table): 0=f32, 1=f16, 2=bf16, 3=q8_0,
4=q5_0, 5=q5_1, 6=q4_0, 7=q4_1, 8=iq4_nl. Update the Python `cache_types` map and
the C `create` range-check per phase.

## Critical files

- `native/include/flyweight_v2_qwen_kernels.hpp` — codec + scores/values/append kernels
- `native/include/flyweight_v2_native_kernels.hpp` — fused prefill kernel
- `native/src/v2_runtime.cpp` — state layout, options/validation, decode launches, helpers
- `native/src/v2_mtp_verifier.inc` — prefill-rows launches
- `native/src/gpu_driver.cpp` — kernel name registration
- `native/include/flyweight_v2.h` — options struct
- `src/flyweight/v2.py`, `v2_server.py`, `cli.py` — Python/CLI plumbing
