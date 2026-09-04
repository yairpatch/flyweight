# Routed expert kernels: the decode token's biggest GPU term

## How this got here

Phase 1b of [paged-kv-cache.md] measured a decode token on the device and found
the GPU ~93% busy, with the routed expert phase the largest single term. Two
recorded beliefs did not survive contact with that measurement:

- *"Route wait is the bottleneck."* It is a host event sync placed after the
  layer's shared expert is enqueued, so it blocks until the device drains a
  backlog several layers deep. It measures the host waiting on a saturated GPU.
- *"The MoE decode cost is 95-97% expert staging memcpy."* True when the cache
  thrashes. At an 8500 MiB budget the working set fits and **staging is
  0.00 MiB/token** — every byte of the expert phase is the kernel.

## The measurement (Qwen3.6-35B-A3B-UD-Q5_K_M, hybrid, ctx 8192)

```
242 resident experts/token x 2.44 MiB = 590 MiB read
expert phase 10.78 ms/token, 45% of a 24.0 ms GPU token
53.3 GiB/s effective vs ~391 GiB/s device wall  (14% of peak)
```

A routed matvec is a weight-streaming problem, so device bandwidth is the
ceiling. Reading 590 MiB at the wall is 1.47 ms; the kernel took 10.78.

## Why it was at 14%

`q5k_grouped_swiglu` called `q5k_value(packed, absolute)` **once per weight**,
and every call rebuilt the whole superblock preamble:

```cuda
const int block = absolute / 256;                              // locate superblock
const float d    = __half2float(*(const __half*)(base));       // reloaded per weight
const float dmin = __half2float(*((const __half*)(base + 2))); // reloaded per weight
q5k_scale_min(scales, group * 2 + sub, &scale, &minimum);      // 6-bit unpack, per weight
const unsigned char low  = base[48 + qindex];                  // 1-byte load
const unsigned char high = base[16 + (offset & 31)];           // 1-byte load
```

A 32-weight sub-block paid for its metadata 32 times, and every quantized byte
arrived on its own load: roughly six loads and twenty ALU ops to produce one
multiply-add. That is an ALU/issue-bound kernel wearing a bandwidth-bound
problem's clothes, which is exactly what 14% of the wall looks like.

`q5k_grouped_accumulate` (the down projection) is worse: its expert loop sits
*inside* the element loop, so it rebuilds the preamble once per expert per
weight.

The efficient shape already existed forty lines below, unused by the K-quant
paths: `FLYWEIGHT_Q8_MATVEC` walks 32-value *groups* with a group function.

## Shipped

**`q5k_grouped_swiglu` rewritten to four weights per step.** ✅ 2026-08-29. The
preamble is hoisted to once per 32-bit nibble word and the nibble and high-bit
planes are read as words. `ds = d * scale` is the same product the scalar path
formed, so **each weight decodes to the same float**; only the accumulation
order changes.

Measured both orderings (the wall-clock drift on this box is ±10%, so a single
ordering proves nothing):

| | expert ms/token | GPU total | host decode |
|---|---|---|---|
| old, run A | 10.78 | 24.02 | 24.93 |
| new, run A | 6.91 | 17.59 | 19.20 |
| new, run B | 7.40 | 17.58 | 19.75 |
| old, run B | 9.46 | 20.27 | 21.61 |

**Expert phase −29%, GPU total −21%, decode −16% (≈43 → 51 tok/s), and greedy
output token-identical over 96 tokens.** Effective bandwidth 14-16% → 20-21% of
the wall.

Worth noting which arm is *stable*: the new kernel measured 17.59 and 17.58 ms
GPU total across the two orderings, the old one 24.02 and 20.27. An ALU-bound
kernel tracks the SM clock and so tracks the box's drift; a more
memory-bound one does not. The stability is itself evidence the rewrite moved
the kernel toward its intended regime.

**The three sibling kernels followed.** ✅ 2026-08-29. `q6k_grouped_swiglu` plus
both `*_grouped_accumulate` down projections, same treatment. Two details worth
keeping:

- **Q6_K takes no word loads.** Its superblock is 210 bytes, not a multiple of
  four, so every odd block lands 2-byte aligned and a 32-bit read of the nibble
  plane would be misaligned. `q6k_quad` hoists the preamble and keeps byte
  loads; the hoist was the large half of the win anyway. (Q5_K's 176-byte
  superblock is 4-aligned, so `q5k_quad` does take words.)
- Four consecutive Q6_K weights share a scale because the scale index moves with
  `l / 16` and `l` is a multiple of four, so a quad never straddles the boundary.

Against the original scalar build, still greedy token-identical 96/96:

| build | expert ms/token | GPU total | host decode |
|---|---|---|---|
| original (3 runs) | 10.78 / 9.46 / 9.33 | 24.02 / 20.27 / 20.20 | 24.93 / 21.61 / 21.83 |
| q5k swiglu only (2 runs) | 6.91 / 7.40 | 17.59 / 17.58 | 19.20 / 19.75 |
| all four | **6.38** | **16.76** | **18.12** |

**Expert phase ~9.9 → 6.4 ms (−35%), GPU total ~21.5 → 16.8 (−22%), decode
~22.8 → 18.1 ms/token (−21%, roughly 44 → 55 tok/s), 16% → 23% of the
bandwidth wall.** The three siblings added about a further 11% on the expert
phase over the q5k rewrite alone.

## What a server request actually sees

The numbers above are decode measured in isolation, with the prompt already
warm. That is the right instrument for a decode kernel and the wrong one for
answering "is the server faster". Reported as a speedup it overstates what a
user gets, because a request also pays prefill — and prefill runs the `_rows`
kernels, which this work did not touch.

End to end through `serve-v2` (`--context 5000 --gpu-cache-mib 8500
--moe-device=hybrid`), streaming, both orderings, median of 3
(`bench_server_client.py`):

| build | decode 2k / short | ttft | total 2k / short |
|---|---|---|---|
| original | 44.90 / 45.38 tok/s | 4.96 s | 7.76 / 3.03 s |
| rewritten (first) | 50.67 / 52.71 | 4.57 s | 7.09 / 2.62 s |
| rewritten (second) | 49.43 / 51.70 | 4.40 s | 6.97 / 2.68 s |

**Decode +11-14% at the server boundary, against +21% measured in isolation**
(engine, sampling and SSE overhead sit between them). Time to first token is
unchanged within noise, as expected — prefill was not touched.

**And that is why the change is hard to feel.** On the 2k-prompt request, ttft
is ~4.5 s of a ~7 s response: **63% of the wall time is prefill**. A 21% decode
win moves the whole response by ~10%. At this checkpoint's 5000-token context a
prompt-heavy workload is mostly prefill, so the perceptible share shrinks
further. Quoting the isolated decode figure as though it were the user-visible
speedup was misleading, and the `_rows` variants are now the item that matters
for felt latency, not more decode work.

Measurement note: this checkpoint streams its thinking as `reasoning_content`,
not `content`. A client that reads only `content` sees a stream with no tokens
in it and reports nothing — which is how the first attempt at this table came
back empty.

## Still on the table

1. **Push past 4-weight amortization.** Four weights per thread-step amortizes
   the preamble 4x; a full 32-weight sub-block would amortize it 32x. The
   blocker is the launcher's fixed 256-thread block: at input_size 2048 there
   are only 64 sub-blocks per row, so one-sub-block-per-thread would idle 75% of
   the block. Warp-cooperative decoding, or a per-kernel block size, is the way
   in. At 23% of the wall there is still 4x of theoretical headroom.
2. **The `_rows` variants** carry the same pattern into prefill, unmeasured here.
3. **The other quantizations.** Only q5_K and q6_K were touched because that is
   what this checkpoint uses; q4_K and the IQ family have the same shape.

## Discipline this cost to learn

- **Bit identity is the wrong gate.** Reordering accumulation moves results at
  ulp scale. Greedy token identity over a real continuation is the gate; pin the
  GPU budget and disable expert history first, or placement drift will be
  mistaken for a kernel bug ([ornith-greedy-nondeterminism]).
- **Both orderings, always.** See the table above, and the four A/B pairs in
  [paged-kv-cache.md] where three consistent orderings still gave the wrong
  answer.
- The kernel corpus has a ~16KB raw-string segment limit; this change opens a
  new `R"FLYWEIGHT_CUDA(` segment rather than growing the existing one.
