# Concurrent decode (close the last llama.cpp gap)

## Context

colibri now matches llama.cpp on prefix reuse (LCP slot routing, spread checkpoints,
host-backed prompt cache), but requests still execute strictly one at a time:

- Python: `InferenceService._generation_lock` (server.py) serializes every endpoint;
  the v2 generation pool is `max_workers=1` (v2_server.py).
- Native: `colibri_v2_qwen_runtime_generate` is one blocking call — full prefill then
  the whole token loop — on a single CUDA stream, one shared workspace arena, the
  shared GPU expert cache, and OpenMP CPU MoE.

Observed cost in live logs: a short side-request queues behind a 150 s prefill and
reports 0.5 tok/s "decode" that is actually queue wait; llama.cpp meanwhile decodes
multiple slots simultaneously (user log: two tasks at ~30 t/s each).

## Constraints that shape the design

- All CUDA work must stay on one thread/stream (workspace, expert cache, stats are
  single-owner; nothing is thread-safe today, and keeping it that way avoids a huge
  audit). Concurrency therefore means **interleaving steps**, not parallel threads.
- Decode is ~90% route_wait + expert_page (memory: hybrid cache economics), so true
  multi-sequence batching (Phase B) has real upside: several sequences amortize each
  expert page-in.
- MTP stays excluded (single-slot, as with parallel slots).
- Sequences already have isolated arenas (`QwenSequence`), so interleaving decode
  steps across slots is a pointer swap (`qwen_switch_sequence`), not a state copy.

## Phase A — cooperative round-robin scheduler ✅ SHIPPED 2026-07-22

Implemented as designed below. Notes from the build:
- The blocking generate path was refactored onto the same helpers the engine
  uses (`qwen_prompt_begin` / `qwen_prefill_unit` / `qwen_prompt_finish` +
  `QwenPromptPlan`), so the two paths cannot drift; blocking generate now
  errors if the engine has tasks in flight.
- Task start routing is ownership-aware (`qwen_engine_try_start`): a slot owned
  by a running task is untouchable, and if the busiest prefix match IS an owned
  slot (same conversation already generating) the task waits rather than
  forking the history onto another slot.
- EOS stops natively via a stop-token list passed at submit, so no token is
  decoded past EOS (this is what makes single-task engine output bit-identical
  to blocking generate, which relied on the Python callback for the same).
- Per-task try/catch inside engine_step isolates one task's failure.
- Python: `_NativeEngine` (one daemon thread pumping engine_step, fanning
  events to per-request SimpleQueues), `_stream` rewritten onto it,
  `InferenceService._serialize_generation` (v2 sets False -> the HTTP layer no
  longer serializes requests), chat-continuation state guarded by a lock.
- Gates (live 35B Q5, --parallel 2): GATE1 blocking==engine bit-identical ✓;
  GATE2 short request submitted 2 s into a 4.2k-token prefill got its first
  token at +4.4 s while the long task finished at +14.2 s ✓; GATE3 two
  concurrently-decoded conversations bit-identical to sequential runs ✓.

### Original design (implemented)

Replace the blocking generate with a task engine, all CUDA still on one thread:

- New ABI:
  - `colibri_v2_qwen_task_submit(runtime, prompt, count, max_tokens, *task_id)` —
    routes to a slot (existing `qwen_route_sequence` logic), computes reuse, queues.
  - `colibri_v2_qwen_engine_step(runtime, events*, capacity, *count)` — runs ONE
    scheduling cycle and reports per-task events (token emitted / finished / error).
    A cycle: for each runnable task in round-robin order, either one prefill chunk
    (bounded, e.g. `min(prefill_rows, remaining)`) or one decode token, switching
    slots via `qwen_switch_sequence` between tasks.
  - `colibri_v2_qwen_task_cancel(runtime, task_id)`.
- Python: one dedicated engine thread per service loops `engine_step` while tasks are
  live and fans events out to per-request queues (the queue plumbing already exists in
  `NativeV2Generator._stream`). The v2 path drops the global generation lock; requests
  submit + consume their own queue. Legacy blocking `generate` stays for CLI/bench.
- Preemption granularity: a 1024-row prefill chunk ≈ 2.5 s at ~400 t/s prefill. Good
  enough for v1; chunk can be reduced for latency-sensitive setups via the existing
  COLIBRI_PREFILL_ROWS. Checkpoint saves stay inside the owning task's prefill steps.
- Semantics preserved per task: same reuse, same checkpoints, same greedy output
  (single-task workloads must remain bit-identical — gate A).
- Outcome: a short request interleaves with a long prefill/decode instead of waiting
  for it. Total throughput unchanged (steps are time-sliced).

Gates:
1. Single task: bit-identical output vs the blocking path.
2. Long prefill + short request submitted mid-way: short request's first token arrives
   within one chunk time (~seconds), not after the full prefill; both outputs correct.
3. Two interleaved decodes: outputs bit-identical to sequential runs of the same
   prompts (interleave must not corrupt either sequence).

## Phase B — multi-sequence decode via CPU/GPU overlap (throughput)

REVISED after a full read of `colibri_v2_qwen_runtime_decode` (2026-07-22): the
original "kernels gain a row dimension" plan is NOT the right first move. Everything
runs on ONE CUDA stream, and the per-layer cost is dominated by the CPU-side MoE
phase (route event sync -> expert paging memcpy -> `qwen_cpu_moe` GEMMs) while the
GPU is idle. So the big win needs NO new kernels: drive one layer across N sequences
with the EXISTING single-row kernels, queueing every sequence's GPU pre-MoE work
first, so the CPU expert phase of sequence A overlaps the GPU phase of sequence B.

Design (function `qwen_decode_tokens_multi(runtime, n, slots[], inputs[], outputs[])`,
used by engine_step when >=2 tasks are in decode phase; N=1 and the blocking path
keep the existing decode untouched):

- Per layer: for each seq -> queue GPU pre-MoE (attention/DeltaNet or shared-expert
  part) + router + async DtoH + record that seq's event; then for each seq -> sync
  its event (GPU is already ahead), run route policy + paging + CPU MoE + queue the
  layer tail. During seq A's CPU phase the GPU executes seq B's queued pre-MoE.
- Concrete hazards found in the code (must handle):
  1. Single `route_event` today -> needs one event per slot (create/destroy alongside
     the slot arenas in prepare/release).
  2. `host_staging` is shared and consumed ASYNCHRONOUSLY (pending expert-bundle
     uploads, pointer-table upload, cpu_output upload are queued on the stream).
     Before seq B's CPU phase writes staging, it must `event_sync` a staging event
     recorded after seq A's last staging-dependent upload -- otherwise B's memcpy
     races A's in-flight uploads. (CPU phases themselves are serial on the engine
     thread, so ONE staging event suffices.)
  3. Workspace must be partitioned per sequence (hidden/residual/normalized/
     first..fourth/activated/router buffers/logits/argmax/attention_scores); compute
     a decode-slice size at prepare and cap the multi-decode batch at
     `workspace_bytes / slice` (the rows-forward workspace is far larger than a few
     decode slices, so N<=4 fits easily).
  4. The active slot's `position/processed_tokens/last_output` live MIRRORED on the
     runtime; the multi driver must park the mirror back into `sequences[active]`
     first and address all state via `sequences[slot]` uniformly, then restore.
  5. Union expert paging across sequences per layer is a small extra win (dedup
     page-ins when sequences route to the same expert) but NOT the main one; skip
     in v1.
- Expected: aggregate ~2x at N=2 when both tasks decode (CPU MoE is ~the whole step;
  overlapping it with the other sequence's GPU phase hides most of it). True row
  batching in kernels (the original plan) remains a later Phase B2 if profiling shows
  the GPU phase becoming the new bottleneck.
- Gates: (1) N=2 concurrent outputs bit-identical to sequential runs; (2) aggregate
  tok/s at N=2 measurably above 1x serial (target >=1.5x); (3) N=1 path untouched
  (bit-identical, no timing regression).

## Phase C — polish

Continuous batching (admit tasks mid-flight — mostly free after A), fairness weights,
optional priority for the "main" (largest-reuse) conversation, surfacing per-task
queue/decode timing in /health.

## Non-goals

- Multi-threaded CUDA or per-request streams.
- Sampling (engine stays greedy argmax; sampling is an orthogonal feature).
- MTP interaction.

## Instrumentation

Extend /health.execution with: active_tasks, queued_tasks, per-cycle step time, and
cumulative interleave switches, so queue-wait vs decode time is finally separable in
the metrics rather than inferred from client-side tok/s.
