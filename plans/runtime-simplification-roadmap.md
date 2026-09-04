# Native CUDA runtime simplification roadmap

## Goal

Reduce native Qwen CPU/GPU orchestration complexity without giving up the
specialized CUDA kernels, bounded memory use, prefix reuse, or concurrent
sequence support that already work.

The target execution policy is:

1. Prepare one explicit memory and placement plan.
2. Run attention, DeltaNet, routing, shared experts, and logits on CUDA.
3. During prompt prefill, run routed experts on CPU and record route frequency.
4. At the prefill/decode boundary, bulk-load a bounded hot-expert set.
5. During decode, run resident experts on GPU and misses on CPU.
6. Keep expert residency immutable for the lifetime of the request.
7. Reconsider placement only between requests.

This combines llama.cpp-style prepare-time placement with vLLM-style bounded
prefill scheduling while retaining Flyweight's native Qwen kernels.

## Rules for every change

- One numbered change at a time. Do not combine independent cleanup with a
  semantic policy change.
- Record the baseline before editing and compare the same model, prompt,
  context, cache precision, CPU thread count, and GPU budget afterward.
- Behavior-preserving refactors must pass bit-identical greedy-output tests.
- Policy changes must preserve model quality within the existing CPU/GPU
  numerical-parity tolerance and must not corrupt KV, recurrent, prefix-cache,
  or multi-sequence state.
- Every semantic change starts behind an explicit runtime option. Environment
  flags may be used during development, but the final supported control belongs
  in the runtime options/CLI.
- Keep the previous implementation available until the new path passes both
  correctness and live-performance gates.
- Stop after each item, report results, and decide whether to continue, revise,
  or roll back.

## Baseline test and benchmark matrix

Establish this once in Change 0 and reuse it for every later item.

### Automated correctness

- Native provider and ABI contract tests.
- Python unit tests for runtime option plumbing and health/info fields.
- Dense and MoE GGUF fixture tests.
- Greedy output equality for blocking generation versus cooperative engine.
- Prompt boundaries around the active row capacity:
  `1`, `2`, `rows-1`, `rows`, `rows+1`, and multiple chunks plus the final
  one-token logits step.
- Prefix reuse from the live slot, a device snapshot, and the host prompt cache.
- Two interleaved sequence slots and two-sequence decode.
- Cancellation during prefill and decode.
- CPU, hybrid, and GPU expert-policy coverage.
- F16, Q8 K/V cache coverage where supported by the fixture.
- MTP verification coverage, because it shares the batched rows forward path.

### Live correctness

Use the Qwen3.6 35B Q5/Q6 models already used by local development:

- Fixed token IDs and temperature zero.
- Short prompt, 1K prompt, 4K prompt, and a prompt crossing the 4K attention
  dispatch threshold.
- Cold runtime and warmed runtime.
- Prefix continuation after an interleaved side request with `--parallel 2`.
- Compare emitted token IDs, final position, reused-token counters, expert
  route counts, and absence of CUDA errors/NaNs.

### Live performance

For each configuration, run at least one warm-up and five measured samples;
report median and worst measured value rather than a single best run.

Record:

- Prompt tokens/s and time to first token.
- Steady decode tokens/s and per-token p50/p95 latency.
- `prefill_{nanoseconds,route_wait_nanoseconds,expert_nanoseconds}`.
- `prefill_gpu_{core,router,transfer}_nanoseconds` when profiling is enabled.
- Expert hits, misses, admissions, evictions, rejections, and bytes uploaded.
- GPU allocation breakdown, free VRAM, peak SM/memory utilization, power, and
  temperature.
- Host RSS, pinned staging size, and CPU utilization.
- Aggregate throughput and per-request latency with two concurrent sequences.

Primary live configurations:

1. 12 GiB GPU, Q6, 58K context, F16 K/V, CPU MoE.
2. 12 GiB GPU, Q6, 58K context, hybrid MoE, auto-fit.
3. Q5, 8K context, hybrid MoE, auto-fit.
4. GPU MoE only where the selected test placement is expected to fit.

## Priority order

### Change 0 — Freeze the baseline and add a repeatable comparison harness ✅ COMPLETE 2026-07-29

**Purpose:** make every following decision evidence-based.

Work:

- Extend or replace `bench_prefill.py` with a checked-in comparison harness
  that emits JSON Lines.
- Capture runtime configuration, model identity, environment overrides, native
  counters, wall timing, and GPU allocation fields in every result.
- Add a comparison command that reports percentage differences and highlights
  failed correctness/performance gates.
- Add synthetic fixture tests for prompt chunk boundaries and expert-policy
  counters before changing native execution.
- Document the exact live benchmark commands.

Gate:

- Two unchanged runs of the same warmed configuration produce identical greedy
  tokens and reasonably stable medians.
- The harness clearly separates prefill from final-token/decode time.
- No native behavior changes in this item.

Rollback: none; this item only adds diagnostics/tests.

Reference commands:

```bash
# Q6, long-context hybrid baseline
PYTHONPATH=src python bench_runtime.py run \
  /home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q6_K.gguf \
  --output /tmp/flyweight-q6-58k-hybrid.jsonl \
  --label q6-58k-hybrid \
  --prompt "Explain how expert routing behaves during a long prompt." \
  --prompt-lengths 256,1024,4096,10000 \
  --context 58000 --cache-type-k f16 --cache-type-v f16 \
  --moe-device hybrid --gpu-cache-mib 0 \
  --samples 5 --sample-warmup 1 \
  --warmup-decode 3 --decode-iterations 10

# Same placement with routed experts fixed to CPU
PYTHONPATH=src python bench_runtime.py run \
  /home/yair/Downloads/Qwen3.6-35B-A3B-UD-Q6_K.gguf \
  --output /tmp/flyweight-q6-58k-cpu.jsonl \
  --label q6-58k-cpu \
  --prompt "Explain how expert routing behaves during a long prompt." \
  --prompt-lengths 256,1024,4096,10000 \
  --context 58000 --cache-type-k f16 --cache-type-v f16 \
  --moe-device cpu --gpu-cache-mib 0 \
  --samples 5 --sample-warmup 1 \
  --warmup-decode 3 --decode-iterations 10

# Behavior-preserving candidate gate (defaults to 3% latency/throughput limits)
PYTHONPATH=src python bench_runtime.py compare \
  /tmp/flyweight-before.jsonl /tmp/flyweight-after.jsonl
```

Run the GPU configurations in isolation. A resident server changes the free
VRAM observed by auto-fit and makes the result incomparable.

Baseline result: [runtime-baseline-2026-07-29.md](runtime-baseline-2026-07-29.md).

### Change 1 — Make workspace layout a single source of truth ✅ COMPLETE 2026-07-29

**Purpose:** remove mirrored allocation formulas and `take()` sequences before
changing execution policy.

Work:

- Introduce typed `QwenDecodeWorkspaceLayout` and
  `QwenRowsWorkspaceLayout` structures.
- Build layouts once from model dimensions and row capacity during prepare.
- Derive `workspace_bytes` from the layout rather than a separate formula.
- Address regions by stored offsets in single decode, rows forward, MTP, and
  multi-sequence decode.
- Keep kernel order and expert behavior unchanged.

Gate:

- Bit-identical outputs for all automated and live correctness cases.
- Exact equality of relevant runtime counters except timing.
- No more than 3% median regression in prefill or decode.
- Tests prove every region is aligned, non-overlapping, and within the arena for
  capacities `1`, `9`, `64`, `1024`, and `4096`.

Rollback: revert the layout consumer while retaining its unit tests.

Implementation result:

- Added typed decode/rows device layouts and matching host-staging layouts in
  `native/include/flyweight_v2_workspace.hpp`.
- The runtime builds all four layouts once during prepare. Allocation sizing,
  single decode, rows prefill/MTP verification, and multi-sequence decode now
  consume their stored offsets.
- The native workspace contract proves legacy size equality, 256-byte
  alignment, non-overlap, and arena bounds for row capacities `1`, `9`, `64`,
  `1024`, and `4096`, plus active decode top-k sizes `1`, `4`, and `8`.
- Full suite against the rebuilt native library: 199 passed, 1 skipped.
- CUDA dense-prefill boundary cases `1`, `2`, `63`, `64`, `65`, `66`, `67`,
  `128`, and `129` remained bit-identical to the single-token path.
- Fixed 8192 MiB hybrid runs with expert history disabled produced identical
  generated tokens and zero non-timing counter differences at prompt lengths
  `256`, `1024`, and `4096`.
- The saved Q6/58K hybrid comparison passed every 3% gate and matched generated
  tokens at `256`, `1024`, `4096`, and `10000`. Candidate JSONL:
  `/tmp/flyweight-q6-58k-hybrid-workspace-candidate.jsonl` (SHA-256
  `6100d285a9f632fdb794f8a70416b9db2ddae3b8e515c3017b5393c70c7adf29`).

### Change 2 — Introduce one explicit expert execution policy ✅ COMPLETE 2026-07-30

**Purpose:** centralize mode checks without changing behavior.

Work:

- Replace scattered `moe_device == 0/1/2` conditionals with an enum and a small
  policy interface.
- Policy answers:
  - whether routed CPU execution is allowed;
  - whether routed GPU execution is allowed;
  - whether misses may be admitted;
  - whether residency may change during the current phase;
  - whether prefill records frequency.
- Use the same policy object in blocking decode, rows forward, MTP verification,
  cooperative tasks, and multi-sequence decode.
- Preserve existing `cpu`, `hybrid`, and `gpu` CLI meanings.

Gate:

- Bit-identical outputs and equal cache counters against Change 1.
- Unit tests cover every policy decision for prefill, decode, and verification.
- No more than 1% median performance regression; this should compile down to
  the same branches.

Rollback: map the policy calls back to the original conditions.

Implementation result:

- Added `ExpertExecutionMode`, `ExpertExecutionPhase`, and
  `ExpertExecutionPolicy` in
  `native/include/flyweight_v2_expert_policy.hpp`.
- The policy explicitly answers routed CPU/GPU permission, miss admission,
  residency mutation, prefill-frequency recording, and route-pruning
  decisions. Streamed-GPU rows retain their existing CPU fallback for
  non-resident prefill/verification experts.
- Blocking decode, rows prefill/MTP verification, Gemma decode, cache sizing,
  expert seeding/prefetch, cooperative tasks, and multi-sequence decode now use
  the policy. Direct numeric `moe_device` access remains only at the public ABI
  validation/reporting boundary and when recording the hybrid-to-CPU fallback.
- The native policy contract covers every mode across prepare, prefill, decode,
  and verification, including enabled/disabled admission.
- Full suite against the rebuilt native library: 199 passed, 1 skipped.
- Fixed-budget live checks for `cpu`, `hybrid`, and streamed `gpu` modes
  produced identical tokens, reported modes, and non-timing counters against
  the pre-policy library. Ten paired hybrid samples also had zero non-timing
  counter differences.
- Pooled reverse-order performance results (ten measured samples per side,
  1024-token prompt, 50 decode tokens/sample): first-token latency `+0.135%`,
  native prefill `+0.004%`, decode median `+0.021%`, and decode throughput
  `+0.381%`. Generated tokens were identical. Decode p95 was `+1.293%`, noted
  as tail noise; the stated median-performance gate passed.
- Candidate/reference benchmark SHA-256 values:
  `1150a19764853c3c23bd45b6c9d1f1fc8d7dd6fd0ac50023f6c3106c3624e852`
  and
  `9a89586b8d47f1b0ac9cf2736cb3beffcc5cd4e6c792f22b379d6a9fb4c38249`
  for the first order, and
  `3a950896826bb1595108bbb3978332cc4efd9917c384131c616f8d3cd6bcf641`
  and
  `36005d038323cab421c6986605e510735b266332afb17c52dff98963e261a4ce`
  for the reverse order.

### Change 3 — Add CPU-routed-expert prefill as an opt-in hybrid policy

**Purpose:** simplify the busiest prompt path and eliminate split CPU/GPU
routed-expert execution during prefill.

Work:

- Add a runtime/CLI option such as `--hybrid-prefill cpu|split`, initially
  defaulting to `split`.
- In `cpu` prefill:
  - keep CUDA attention/DeltaNet, router, and shared expert unchanged;
  - execute all routed experts on CPU;
  - record selected-expert frequency;
  - skip GPU resident-expert pointer tables, grouped expert kernels, and the
    CPU/GPU routed-result merge.
- Do not alter single-token decode or MTP verification in this item.

Gate:

- Greedy outputs match the existing split path for the same quantized model
  within the established numerical policy.
- Prefix snapshots, cancellation, and concurrent prefill remain correct.
- No increase in time to first token greater than 5% on the primary Q6/58K
  configuration.
- Prefer the new path only if median prefill improves, p95 improves materially,
  or the implementation removes complexity with no meaningful regression.
- GPU transfer time and PCIe traffic must not increase.

Rollback: select `--hybrid-prefill split`.

Status: complete as an opt-in policy; `split` remains the default.

- Added `hybrid_prefill="split"|"cpu"` across the Python runtime API,
  `benchmark-v2`, `probe-native-v2`, `serve-v2`, the native server, and the
  checked-in JSONL benchmark harness. The C ABI field is appended, defaults to
  split, and is validated on both sides.
- CPU-prefill is scoped only to batched prompt rows. It records route
  frequency, bypasses resident-expert lookup/table upload/grouped kernels, and
  directly combines the CPU routed result with the shared expert. The final
  scalar prompt boundary, ordinary decode, and MTP verification retain the
  split policy.
- Native policy/ABI targets compiled and the policy contract passed. The full
  suite against the rebuilt isolated shared library passed: `201 passed`,
  `9 skipped`, and `19 subtests passed`.
- Live Q5 correctness covered warmed resident experts, three repeated 1K
  prompts, a 768-token device-prefix reuse plus 256-token divergent suffix,
  callback cancellation/recovery, and two cooperative concurrent prefills.
  Split and CPU-prefill produced identical tokens in every case.
- Active cancellation during a 4K prefill stopped both policies with the
  expected cancellation error; reset/recovery tokens matched. Live MTP depth
  2 also matched all eight generated tokens and accounting (`3` accepted,
  `0` rejected), proving verification remains unchanged.
- On fresh Q6/58K hybrid runs with expert history disabled, CPU-prefill reduced
  prefill-window cache misses from `327680` to `320`; the remaining `320` are
  the intentionally unchanged 80-layer, top-4 scalar boundary. GPU transfer
  time remained zero for both policies.
- Six measured samples per policy, pooled across forward and reverse order,
  gave CPU-prefill first-token latency `+2.091%`, native prefill `+2.064%`,
  decode median `+0.300%`, decode p95 `+4.487%`, and decode throughput
  `-3.997%`. Outputs were identical. The 5% gates passed, but
  CPU-prefill did not justify promotion over `split`; it remains an explicit
  experiment/rollback-safe option.
- Q6/58K JSONL SHA-256 values: split forward
  `918d1c294b4f09ea8e22e25c3b380f9ea8a3148584d5ba9d4f43a5cbb939a2f5`,
  CPU forward
  `b282be1390bec05a49f973d4fa324c0582a0d6e0a1f78e8dc3135ef2f5ac3edb`,
  CPU reverse
  `5405118ec36d014382b606417f3217ce3f7973ce85b5230d2ae8100ab8ca1aa4`,
  and split reverse
  `1a83b8e31a0f339ad71c4797c69a9bd235ce7a94663ec6681fa31042fca8c6cb`.

### Change 4 — Make expert residency immutable during each request

**Purpose:** remove cache admission, eviction, and expert uploads from the token
critical path.

Work:

- Freeze the expert-to-slot map when a request enters decode.
- Cache hits execute on GPU; misses execute on CPU.
- Record misses/frequency for future placement, but never admit or evict while
  the request is active.
- With concurrent tasks, treat residency as an engine-cycle epoch: do not
  mutate it while any task from the epoch is running.
- Add explicit counters for deferred admissions and residency epochs.

Gate:

- No expert upload occurs between the first emitted token and request finish.
- Bit-identical request output versus the mutable hybrid path within the
  existing cross-device tolerance.
- Decode p95 latency improves or remains within 3%; median throughput remains
  within 5%.
- Two concurrent sequences cannot change each other's resident mapping or
  corrupt pointer tables.

Rollback: runtime option `mutable` restores Change 2 behavior.

Status: complete as an opt-in prepared-map policy; `mutable` remains the
default until Change 5 supplies normal between-request placement.

- Added `expert_residency="mutable"|"immutable"` across the runtime API,
  `benchmark-v2`, `probe-native-v2`, `serve-v2`, the native server, and the
  JSONL harness. The appended C ABI field is validated on both sides.
- Immutable hybrid mode freezes placement immediately before the first token
  is emitted. Resident hits remain on GPU; misses execute on CPU, still update
  route history, and increment `expert_cache_deferred_admissions`. Blocking
  requests unfreeze on every exit path, direct decode stays frozen until
  reset, and cooperative tasks share one `expert_residency_epochs` epoch until
  the final task finishes.
- Immutable cooperative epochs retain round-robin request interleaving but use
  the per-sequence decode driver, avoiding the mutable multi-sequence paging
  driver's cache-update assumptions.
- The native policy contract now explicitly proves frozen decode permits
  CPU/GPU execution but forbids admission/residency mutation. The complete
  suite against the rebuilt production library passed: `201 passed`,
  `9 skipped`, and `19 subtests passed`.
- Live Q5 testing compared five mutable and five immutable 1K/32-token
  requests. Greedy outputs matched. At every immutable callback, admissions
  and evictions were constant, frozen state was visible, deferred admissions
  advanced by exactly 320 per decoded token, and the runtime unfroze after
  request completion. Median end-to-end time improved `6.35%`.
- Two live concurrent immutable tasks shared one epoch, generated 12 tokens
  each, recorded `7040` deferred admissions, performed zero admissions and
  evictions, and unfroze only after both tasks finished.
- A cold Q6/58K map improved decode p95 `44.31%` but reduced throughput
  `9.35%`, failing the throughput gate. With the existing bounded
  `--prefill-cache-seed 4` establishing placement before the epoch, p95
  improved `47.62%`, throughput improved `4.98%`, first-token latency was
  `+1.17%`, and tokens matched. This prepared-map configuration passes the
  stated p95/throughput gate; the cold result is why immutable is not yet the
  default.
- Seeded Q6/58K JSONL SHA-256 values: mutable
  `6a90c6deb40cab4a0509bc7c5ba54bd11c9ddceb20f5a24d7b5d8cc90c6bcb92`
  and immutable
  `2134ba618f66faced462c18d942cead4939505bfb08407719d0aa7a96e15f194`.

### Change 5 — Promote post-prefill hot-expert seeding to the normal auto policy

**Purpose:** replace reactive paging with one bounded placement decision.

Work:

- Reuse the existing prefill route history and bulk seed machinery.
- At the prefill/decode boundary, select hot experts within the prepared cache
  budget and upload them in one bounded phase.
- Pin the resulting set for the request/epoch.
- Combine current-prompt frequency with decayed persistent history so a short
  prompt does not discard a useful learned set.
- Bound seed latency and skip seeding when predicted value is too small.

Gate:

- Report seed time, bytes, selected experts, later hits, and avoided misses.
- Time to first token including seeding must not regress by more than 5% unless
  the first 32 decode tokens recover the cost.
- Median decode throughput should improve over immutable residency without
  seeding on at least the hybrid test configurations.
- A cold or adversarial prompt cannot trigger unbounded uploads.

Rollback: seed count/budget zero keeps immutable existing residency.

Status: complete; immutable residency plus bounded auto placement is now the
normal hybrid policy.

- `prefill_cache_seed` accepts `auto`, `off`, or a manual count in `[0, 256]`.
  Python, the CLI, server, and JSONL harness default to `auto`; immutable
  residency is now their default prepared-map policy. `off`/`0` retains the
  existing map, while `mutable` restores reactive paging.
- Prompt frequency is isolated per sequence and scored at 8x weight alongside
  the already-decayed persistent history. Auto placement uses at most four
  experts per layer, reuses already-resident selections, retains the old pinned
  set when a cold prompt has fewer than 32 tokens of route evidence, and stops
  at 1 GiB or 100 ms.
- Runtime info now reports uploaded bytes, selected and uploaded experts, seed
  time, later pinned-set hits, avoided immutable misses, auto skips, and budget
  stops. The standalone native seed contract covers sizing, threshold, and
  overflow-safe scoring.
- A live cold Q5 396-token prompt selected/uploaded `160` experts (`4 x 40`),
  transferred `371,359,744` bytes in `40.64 ms`, recorded `114` later seed
  hits, and matched `off` output exactly. A cold one-token prompt recorded one
  auto skip and uploaded zero bytes. After learning a 160-expert set, a
  one-token follow-up retained it with zero replacements and zero uploads.
- On the live Q6 hybrid test, auto and `off` generated identical tokens.
  Median decode throughput improved from `40.14` to `45.30 tok/s` (`+12.86%`).
  TTFT increased `13.88%` (`47.65 ms`), but prompt plus the next 30 measured
  tokens improved `8.77%`, recovering the placement cost within the required
  first-32 window.
- Two concurrent auto-seeded sequences emitted four tokens each, shared one
  immutable epoch, held admissions/evictions constant after the first emit,
  and unfroze after completion. The rebuilt production library passed all six
  native contracts and the full suite: `203 passed`, `9 skipped`, and
  `23 subtests passed`.

### Change 6 — Simplify and clarify public modes

**Purpose:** make user-facing names match actual behavior.

**Status (2026-07-30): implemented.**

- Public modes are `cpu`, `auto` (default), and strict `resident`.
- Canonical modes are explicit. For backward compatibility and performance,
  `hybrid` preserves the former policy and resolves to `legacy-hybrid`; `gpu`
  resolves to `legacy-paging`.
- Startup, `/health`, and the web runtime badge expose the resolved policy and
  fallback reason.
- `resident` preloads the complete expert set and reports a componentized
  memory-fit error instead of silently paging.
- Validation: `211 passed, 9 skipped, 26 subtests passed`; all 6 native
  contracts passed. A live undersized `resident` preparation reported the
  static, KV/state, workspace, staging, snapshot, and expert requirements.

Proposed modes:

- `cpu`: routed experts always execute on CPU.
- `auto`: CPU routed-expert prefill, one request-stable hot GPU set for decode,
  CPU misses.
- `resident`: routed experts execute on GPU only and preparation fails if the
  required resident placement does not fit.

Work:

- Retain aliases: `hybrid -> auto`, `gpu -> resident` only after documenting
  the behavior change.
- Keep explicit legacy paging behind an experimental option if benchmarks
  justify retaining it.
- Print the resolved placement and fallback reason at startup.
- Expose the resolved policy, not just the requested alias, through `/health`.

Gate:

- CLI/API compatibility tests cover old aliases.
- Memory-fit errors report static, KV, workspace, staging, and expert
  requirements.
- No silent fallback from `resident` to paging.

Rollback: retain the old names and meanings for one compatibility period.

### Change 7 — Consolidate forward orchestration

**Purpose:** share state/workspace/routing/MoE orchestration without forcing
decode and prefill to use the same kernels.

Work:

- Introduce a `QwenForwardBatch` descriptor containing rows, sequences,
  purpose, state views, workspace layout, and expert policy.
- Share preparation, routing result representation, expert partitioning, state
  advancement, and result merging.
- Keep specialized single-row, multi-row, attention, DeltaNet, and quantized
  kernels.
- Migrate one caller at a time: rows/MTP, blocking decode, then multi-decode.

Gate:

- Each caller migration is a separate sub-step with bit-identical output.
- N=1 and N=2 concurrent performance remains within 3%.
- Cancellation and error isolation remain task-local.
- Delete duplicated code only after its replacement passes the full matrix.

Rollback: migrate callers independently so each can return to its old driver.

### Change 8 — Replace fixed prefill rows with a scheduler token budget

**Purpose:** improve latency under concurrency with one understandable batching
control.

Work:

- Add `--batch-token-budget`.
- Schedule ready decode tokens first.
- Spend the remaining budget on one or more prefill chunks.
- Preserve checkpoint boundaries and per-task fairness.
- Initially map the existing prefill-row setting to the new budget for
  compatibility.

Gate:

- Single-request throughput remains within 3%.
- A short decode request arriving during a long prefill reaches its next token
  within one configured budget unit.
- Aggregate two-request throughput does not regress by more than 5%.
- Outputs remain identical to sequential execution.

Rollback: retain the existing fixed-row cooperative scheduler.

### Change 9 — Remove superseded dynamic-paging paths and development flags

**Purpose:** realize the code-size and test-matrix reduction after the new
policy is proven.

Work:

- Remove legacy per-token admission/eviction only after Changes 4-6 have been
  the tested default.
- Replace overlapping `FLYWEIGHT_*` tuning variables with documented runtime
  options or one `experimental` configuration group.
- Remove counters that no longer describe reachable behavior; version the ABI
  change if fields must move.
- Delete stale benchmark scripts after their cases exist in the comparison
  harness.

Gate:

- Search confirms no supported CLI/API depends on removed flags.
- README, `--help`, Python bindings, C ABI, tests, and health schema agree.
- Full test and live benchmark matrices pass after deletion.

Rollback: perform removal in a dedicated commit so it can be reverted without
reverting the new policy.

## Deferred ideas

These may improve performance but do not simplify the runtime, so they should
not interrupt the ordered work above:

- CUDA graph capture/replay.
- Full continuous batching.
- Paged KV allocation.
- Multi-GPU expert parallelism.
- Asynchronous expert rebalancing.
- Disaggregated prefill and decode.
- New speculative decoding strategies.

## First implementation target

Start with **Change 0 only**. Do not modify native execution until the baseline
harness and missing boundary tests are committed and produce a saved comparison
record for the current Q6/58K hybrid server configuration.
