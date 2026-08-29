# Native runtime baseline — 2026-07-29

## Configuration

- Source revision: `ddc036d92e9fa2d5896f2af7989bfc03ac7a0d10`
  with a dirty working tree.
- Model: Qwen3.6-35B-A3B Q6_K GGUF.
- Device: RTX 5070 Ti Laptop, 12 GiB, compute capability 12.0.
- Runtime: native v2, hybrid MoE, auto-fit GPU allocation.
- Context: 58,000.
- K/V cache: F16/F16.
- CPU expert threads: 16.
- Prefill cache seed: disabled.
- Per case: one excluded warm-up and five fresh-runtime measured samples.
- Per sample: three decode warm-ups and ten measured decode tokens.
- Raw JSONL during this development session:
  `/tmp/flyweight-q6-58k-hybrid-baseline.jsonl`.
- Raw JSONL SHA-256:
  `2c7c16646b0364cad5b71aa77428c4c32628bbabc72bbd91b8e2511d4c74ade5`.

Fresh runtimes were used for every sample so an exact-prefix snapshot from a
previous sample could not turn prompt measurement into a prefix-cache hit.

## Results

| Prompt | Native prefill median | First token median | Wall prompt rate | Decode median | Worst measured decode rate |
|---:|---:|---:|---:|---:|---:|
| 256 | 1.416 s | 1.454 s | 176.07 tok/s | 24.74 tok/s | 24.30 tok/s |
| 1,024 | 4.739 s | 4.777 s | 214.36 tok/s | 23.77 tok/s | 23.59 tok/s |
| 4,096 | 12.363 s | 12.405 s | 330.20 tok/s | 24.58 tok/s | 24.26 tok/s |
| 10,000 | 37.890 s | 37.931 s | 263.64 tok/s | 23.43 tok/s | 21.53 tok/s |

All five samples in every case emitted identical greedy token sequences.

## Important observations

- Fresh hybrid prefill recorded zero resident GPU expert hits at every prompt
  length.
- Routed expert misses were exactly `layers * top_k * prefill_tokens`:
  81,920 at 256, 327,680 at 1K, 1,310,720 at 4K, and 3,200,000 at 10K.
- This means the measured fresh-prompt hybrid path executed routed prefill
  experts entirely on CPU while still paying the hybrid residency lookup and
  pointer-management path. Change 3 can test removing that overhead directly.
- Auto-fit GPU allocation varied with free VRAM:
  approximately 8.1-8.6 GiB total, including a 3.9-4.35 GiB expert cache.
- The 4K tensor-core attention transition is represented in the baseline.
- The final-token boundary added roughly 35-50 ms beyond native batched
  prefill in the median cases.

## Validation

- New measurement/comparison unit tests passed.
- Native prompt boundaries `1, 2, 63, 64, 65, 66, 67, 128, 129` matched the
  single-token reference path on a synthetic dense CUDA model.
- Complete automated suite: 199 tests passed, 1 skipped.
- The JSONL baseline passed a self-comparison with deterministic-token and 3%
  performance gates.

