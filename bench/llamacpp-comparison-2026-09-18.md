# Flyweight 0.2.0 vs llama.cpp, 2026-09-18

Same files, same machine, same client, same settings wherever both engines
have the setting. Speed only; this says nothing about output quality.

## Setup

| | |
| --- | --- |
| Machine | RTX 5070 Ti Laptop 12 GB (driver 610.57.04), Ryzen 9 9955HX 16 cores, 60 GB RAM, Linux |
| flyweight | 0.2.0, main `aad4fb7` |
| llama.cpp | master `f3a33dff2` (2026-09-12), CUDA build |
| Client | `bench/bench_stream_ab.py`: raw `/v1/completions`, 1929-token prompt, 128 tokens out, temperature 0, streamed |
| Common settings | context 32768, one sequence slot, flash attention on, KV f16 unless stated, 16 threads |
| Method | each engine started fresh per configuration, one untimed warm-up request, then five timed requests whose prompts share no prefix; the table shows the median. Configurations were interleaved between engines in two passes so both saw the same page-cache state; the two passes agreed within 5% and are averaged |

Prefill is prompt tokens over time to first token as the client saw it, so it
includes HTTP and tokenization. Decode is tokens after the first over the time
after the first. llama.cpp's own `timings` agree with the client figures to
within 5% and are not shown separately.

## Qwen3.8-Flash-Next UD-IQ1_S (MoE, 512 experts, 69 GB on disk)

The case flyweight is built for: the experts do not fit the card.

| Engine and configuration | TTFT | prefill tok/s | decode tok/s |
| --- | --- | --- | --- |
| llama.cpp `-fit on` (default automatic placement) | 6.1 s | 315 | 34 |
| llama.cpp `-ngl 99 --cpu-moe` (all experts on the CPU) | 6.6 s | 294 | 33 |
| **flyweight `--expert-mode auto`** (default; 3.7 GB hot-expert cache on the card) | **5.1 s** | **383** | **35** |
| flyweight `--expert-mode cpu` | 9.7 s | 199 | 33 |
| flyweight auto + `--mtp-model` (the shared MTP file), 2 drafts | 5.4 s | 357 | 37 |

- Flyweight's default is about 20% faster on prefill and level on decode.
- Flyweight's pure-CPU expert mode is a third slower on prefill than
  llama.cpp's `--cpu-moe`. llama.cpp moves batched expert matmuls to the GPU
  for prompt processing even when the weights live on the CPU; flyweight's
  `cpu` mode means what it says. Use `auto`.
- MTP adds 7% decode for a small prefill cost. llama.cpp has no draft path for
  this model.
- llama.cpp warns that CPU tensor overrides with mmap cost performance and
  suggests `--load-mode none`; that was not tried, so its numbers may have a
  few percent of headroom.

## Qwen3.8-27B UD-IQ2_XXS (dense, 6.9 GB)

The model fits the card; the question is what 32K of KV cache does to it.

| Engine and configuration | TTFT | prefill tok/s | decode tok/s |
| --- | --- | --- | --- |
| **llama.cpp, f16 KV** | **2.2 s** | **894** | **41** |
| llama.cpp, q8_0 KV | 2.2 s | 878 | 40 |
| flyweight, f16 KV (3 of 64 dense blocks spilled to the CPU) | 4.0 s | 482 | 32 |
| flyweight, turbo4 KV (nothing spilled) | 4.1 s | 466 | 39 |

- llama.cpp is about 1.9x faster on dense prefill. This is the largest gap
  in either direction and the clearest thing to work on.
- With f16 KV flyweight reserves more VRAM than llama.cpp at 32K (prompt
  cache, expert workspace, checkpoints) and spills three dense blocks, which
  costs a quarter of decode. Its 4-bit `turbo4` KV recovers decode to within
  5% of llama.cpp; llama.cpp's `q8_0` KV changes nothing here because it was
  not spilling in the first place.
- Flyweight's time to first token carries about 1.5 s more than its prefill
  rate explains, which points at fixed per-request overhead rather than the
  kernels.

## Summary

Where flyweight is ahead: MoE checkpoints larger than VRAM, which is the
design target, by about 20% on prefill with decode level; 4-bit KV that
keeps a dense model on the card at long context; a draft path for
Flash-Next. Where llama.cpp is ahead: dense prefill by nearly 2x, dense
decode when it fits and flyweight spills, time to first token overhead, and
pure-CPU expert prefill.

Not compared: models only one engine runs (NVFP4, Q2_0, DeepSeek-V4 here;
the long tail of architectures on llama.cpp's side), multi-slot throughput,
and quality.

## Reproducing

~~~bash
# llama.cpp
llama-server -m MODEL.gguf -c 32768 -fa on -t 16 -ctk f16 -ctv f16 -np 1 --no-webui --port 8201 [-fit on | -ngl 99 --cpu-moe]
python bench/bench_stream_ab.py --base http://127.0.0.1:8201 --label llamacpp --repeats 5 --out results.jsonl

# flyweight
flyweight serve MODEL.gguf --context 32768 --cache-type-k f16 --cache-type-v f16 --cpu-threads 16 --port 8202 [--expert-mode auto|cpu]
python bench/bench_stream_ab.py --base http://127.0.0.1:8202 --label flyweight --repeats 5 --out results.jsonl
~~~

Run a warm-up request first (`--repeats 1 --tokens 256 --note-offset 100`),
never run both engines at once, and alternate them if you want the page cache
to treat them equally.

## Follow-up, same day: dense prefill

The prefill gap above was profiled with the runtime's launch timer. There was
no fixed per-request cost: the client's time to first token matched the
server's own prefill time to within 60 ms, so the whole gap is kernel time,
of which the MMQ GEMMs were 64%, attention 15%, and the per-token rope and
KV-store launches (16,000 per 256-token chunk) about 12%.

Three changes, each gated on identical greedy output against the previous
build (27B, and Flash-Next with experts on the CPU so the comparison is
deterministic):

| change | 27B prefill, 1929 tokens | tok/s |
| --- | --- | --- |
| baseline (table above) | 3.78 s | 510 |
| one MMQ launch per projection, token tiles on grid.y | 3.61 s | 534 |
| single-scale MMQ epilogue for IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_XS, Q8_0 | 3.51 s | 550 |
| rope and KV store batched over the chunk (4 launches per layer, not 4 per token) | 3.09 s | 624 |

Flash-Next IQ1_S in `auto` mode gains about 4% from the same changes; its
prefill is expert-bound.

Ruled out by measurement: larger prefill chunks (no change), every other MMQ
tile shape (all slower than 128x128 over 16 warps), and, by building
variants that skip them, both the weight decode and the tensor-core phase of
the MMQ kernel, each of which is under a fifth of its time. What remains is
inside the k-loop's staging and barriers and needs Nsight Compute to see,
which this machine does not have.

llama.cpp on the same file: 894 tok/s. The gap is now 1.43x, from 1.9x.

### What Nsight Compute says about the remaining MMQ gap

Profiled `iq2xxs_q8_mmq` on a 5120-wide projection at 256 rows (one launch,
full metric set, `FLYWEIGHT_NVRTC_LINEINFO=1`):

- The int8 tensor pipe is at 40% of its sustained peak over the kernel and
  saturated whenever it runs: 40% of stall samples sit on the `IMMA`
  instructions themselves (math-pipe throttle). The other stalls are
  long-scoreboard on the decode's byte-wide global loads (20%), `wait`
  (11%), scheduler selection (17%) and barriers (7%).
- Occupancy is one 512-thread block per SM (128 registers, 41 KB shared),
  so the decode and staging phases between the two barriers never overlap
  another block's MMAs. On this 80-block launch the second wave is half
  empty.
- Register spills are small (96 bytes).

Tried against that picture, all measured on the 27B and all no better than
the 128x128 tile over 16 warps at one block per SM: every shape that fits
two or three blocks per SM (worse, 3.4 to 5.8 s, because the smaller
per-warp tile loses fragment reuse), and a register-prefetch pipeline that
issues the next k-step's weight and activation loads before the MMA phase
(3.10 s, no change: the expand and staging work between the barriers still
serializes against the MMAs). The design that would close the gap is warp
specialization over a double-buffered tile, which needs more than 48 KB of
shared memory per block and a driver-side opt-in for dynamic shared memory
that the launcher does not have yet. Ceiling if the pipe ran flat out:
about 2.5x on the GEMMs, which would put the 27B near 1000 tok/s.

### Second pass, same day: the tensor pipe and the attention path

Nsight had shown the int8 tensor pipe saturated at about 46% of its peak
rate while it ran. That number was the answer, misread the first time: on
Ampere and later the `m16n8k16` int8 MMA shape runs at half the rate of the
native `m16n8k32` shape, so a kernel issuing k16 pairs tops out near 50%.
Three more overlap designs were tried first and all measured identical to
the baseline (a double-buffered tile with one barrier per k-step, the same
with odd and even warps in opposite stage/compute order, and the register
prefetch), which is what pointed at the pipe rate rather than the phases.

| change | 27B prefill, 1929 tokens, turbo4 KV | tok/s |
| --- | --- | --- |
| after the first pass (above) | 3.09 s | 624 |
| single-scale kernel on `m16n8k32` (one MMA per 32-group) | 2.88 s | 670 |
| IQ1_S moved to the single-scale kernel (its decoder is single-scale) | 2.73 s | 707 |
| cuBLAS attention prefill from 1024 visible tokens instead of 4096 | 2.47 s | 781 |

The cuBLAS attention path already existed and was the default only once the
visible prefix reached 4K, which for a 4K prompt is its last chunk. Measured
here it is level with the warp kernel at 512 tokens and ahead from there
(+4% at 1024, +11% at 2048, +21% at 4096 prompt tokens). Under turbo4 KV
the greedy output is unchanged; under f16 KV, where cuBLAS reads the cache
in place, one word of the 48-token continuation changes, the same numerics
the path has always had above 4K. Flash-Next output is byte-identical with
experts on the CPU, short and long prompts.

The double-buffered tile stayed (it costs nothing and the kernel now needs
the dynamic-shared opt-in the launcher gained for it); the warp-order and
prefetch variants did not.

llama.cpp on the same file: 894 tok/s. The gap is now 1.14x, from 1.9x at
the start of the day.

### Third pass, same day: the decode phase, the quantizer, and the spill

With the tensor math halved, skipping each phase of the single-scale kernel
in turn put the codebook decode first: it read its packed bytes one at a
time (a 66-byte block aligns to two bytes, so the compiler could not widen
the loads) and looked the codebook up in global memory four times per
group. IQ2_XXS, IQ3_XXS and IQ1_S now decode from 16-bit loads with the
codebook copied once per block into shared memory. The activation
quantizer, one 32-thread block per 32-element group (40,000 blocks per
chunk), runs eight warps per block. And the planner's headroom no longer
doubles to 2 GiB when the context is set explicitly: the KV state it was
guarding against is already subtracted, and the double count is what
spilled three dense blocks with 1.6 GB of VRAM unused.

| change | 27B prefill, turbo4 KV | tok/s |
| --- | --- | --- |
| after the second pass | 2.47 s | 781 |
| quantizer in 256-thread blocks | 2.42 s | 797 |
| 16-bit-load decoders with shared codebooks | 2.34 s | 824 |

| 27B, f16 KV (the matched configuration) | prefill | decode |
| --- | --- | --- |
| llama.cpp | 894 tok/s | 41 tok/s |
| flyweight, before (3 of 64 blocks spilled) | 740 tok/s | 32 tok/s |
| flyweight, after (nothing spilled) | 824 tok/s | 41.7 tok/s |

Greedy output identical throughout, and under f16 KV it now matches the
exact GPU path rather than the host-re-encoded one the spill produced.

Tried and dropped on this pass: the same decoder treatment for the
two-scale families (no change), the odd/even warp order again on top of
k32 (no change), three more tile shapes under k32 (all slower), 512- and
1024-row chunks (within noise, and 1024 loses), mid-prefill checkpoints
off and the prompt cache off (no host time between chunks to recover).

llama.cpp on the same file: 894 tok/s. The gap is 1.08x on prefill and
closed on decode.

### Fourth pass: what the attention path and the k32 kernel are made of

Timing each piece of the cuBLAS attention routine with events (the
`FLYWEIGHT_ATTN_PROFILE=1` switch added for it) put the whole routine at
0.42 ms per layer-chunk, 2.4% of the prefill; the rest of that phase was
the attention layers' output projection, an MMQ. So attention was never the
lever it looked like. Two things came out of it anyway: the query tile
now takes the largest of 64, 32 or 16 rows whose buffers fit (6% on the
routine), and the workspace region those buffers live in is floored for
the 64-row tile instead of scaling with the context alone -- at
`--context 4096` it could not hold even the 16-row tile, every prefill fell
back to the warp kernel, and the 27B ran 12% slower than at 32K. Both
contexts now prefill at the same 2.33 s.

The k32 kernel's SASS (offline `nvcc -cubin` of the dumped corpus) issues
about 1,300 instructions per k-step for 32 MMAs: 12 per MMA of epilogue by
design, the rest integer address math, bounds checks and the decode. A
version hoisting the per-thread staging pointers and validity out of the
k-loop measured 3% slower, so the compiler was already doing better than
the static count suggests. Also measured and dropped on this pass: single
buffering, 2- and 8-group k-steps, a four-tile `ldmatrix`, the chunked
DeltaNet path at 256 rows, and the odd/even warp order under k32.

Final on the 27B, 1929-token prompt: 2.33 s, 828 tok/s, against llama.cpp's
894. Decode 41.7 vs 41 at f16 KV.

### Fifth pass: llama.cpp's kernel, measured, and split-K

llama.cpp's own GEMM benchmark (`test-backend-ops perf`) against flyweight's
per-kernel time on the same prefill, in TFLOPS: IQ2_XXS 72 vs 65, IQ1_S 65
vs 61, IQ3_XXS 68 vs 57, IQ2_S 52 vs 48, IQ2_XS 52 vs 37, IQ1_M 55 vs 54.
Weighted by the 27B's MACs the kernels are 8% behind, which is the whole
remaining prefill gap. llama.cpp's configuration for these types (8 warps,
one block per SM, 128x128 tile, 256 of K per iteration, stream-K) was
tried in this kernel and measured 3.5 s: at 128 registers our per-warp
state spills. Stream-K's target, wave quantization, is real here: 14% of
MMQ time by the launch list. A deterministic split-K (grid.y carries the
splits, fixed-order reduction) recovered 1.5% of it, changed Flash-Next's
greedy output through the partials' summation order, and its partials
region tipped the f16 configuration past its VRAM fit. Reverted.

Final: 27B 2.33 s, 828 tok/s, 41.7 tok/s decode at f16 KV; llama.cpp 894
and 41. Closing the last 7% means adopting llama.cpp's MMQ register and
tile layout, which is a port, not a pass.

### Port checkpoint: llama.cpp's warp layout in this kernel

A kernel in llama.cpp's MMQ layout for IQ2_XXS, IQ1_S and IQ3_XXS -- eight
warps at one block per SM, each warp owning 16 rows and all 128 tokens so
one weight fragment feeds sixteen k32 MMAs, a 256-wide k-step, single
buffered in 78 KB of dynamic shared -- with our decoders and epilogue,
behind `FLYWEIGHT_MMQ_LC=1`. Greedy output identical; 2.72 s against 2.33,
17% slower end to end. ptxas: 255 registers. With one row fragment per
warp every MMA needs its own activation loads (four LDS per MMA against
two in the current tile), and eight warps hide less of that latency than
sixteen. llama.cpp's layout works with its swizzled tile loader and its
Q8_1 activation format, not on its own; the port is all of those together
or nothing. Reverted.

### Sixth pass: the standalone kernel, the GPU timeline, and the tail

Two measurements that were missing all day. First, our kernels on
llama.cpp's own benchmark shape (4096 x 14336, 512 tokens, repeated
launches, nothing else on the stream), against `test-backend-ops perf`:

| kernel | llama.cpp | flyweight |
| --- | --- | --- |
| IQ2_XXS | 72 TFLOPS | 78 |
| IQ1_S | 65 | 73 |
| IQ3_XXS | 68 | 74 |

The kernels were never behind; the in-situ figures earlier in this report
carried the serializing launch timer's overhead. The port would have gained
nothing. Second, a GPU timeline (`FLYWEIGHT_PREFILL_TIMELINE=1`: events
around every launch, read back at chunk end) showed the GPU busy 97% of the
prefill's wall time and the full 256-row chunks running at 889 tok/s, level
with llama.cpp's overall rate. The whole remaining gap was the 137-row tail
chunk: its 9 rows past the last 128-token tile cost a second full tile,
244 ms for a chunk that should take 150.

Every kernel decodes the matrix once regardless of row count (the per-row
and tiled dp4a kernels cost the same as the empty tile they were tried
against), so the only reducible part is the MMA phase: a 32-token variant
of each MMQ kernel (`*_q8_mmq_n`, one token fragment per warp, the thing
llama.cpp's J-templates do) now takes remainders of up to 32 rows. Tail
chunk 244 -> 210 ms; prefill 2.33 -> 2.30 s, 839 tok/s at both KV types;
greedy output identical on the 27B and Flash-Next. The remainder's decode
floor, about 45 ms per prompt on the 27B, is what stands between this and
llama.cpp's 894, plus some 40 ms of fixed per-request work before the
first chunk.

The timeline also reports the host time between chunks (about 1 ms), the
mid-prefill checkpoints and the prompt cache together cost about 1%, and
tokenizing the prompt takes 3 ms; the 50 ms or so between the chunks' GPU
wall and the client's first token is request admission and the first
sampling step, not the prefill. Small batches of 2 to 32 rows now take the
narrow tile in the rows driver too (a 26-row chunk was measured on it).
Prompts of 8 tokens or fewer still go through the runtime's per-row path,
80 ms for a 7-token prompt on the 27B, which is a short-prompt latency
item of its own.

### Seventh pass: the three itemized remainders

- **The 48-wide DeltaNet heads.** A one-thread-per-(token, output) kernel
  with the same group order and epilogue expression as the tile measured
  slower: it re-decodes each weight row once per token where the tile
  decodes it once per 128. Dropped. The exact fix is to fuse the two
  48-wide projections into one 96-wide GEMM at load, which touches the
  recurrence kernel's operand strides on both backends; not done here.
- **Small-batch routing.** Routing 2..32-row batches to the narrow tile had
  silently moved the K-quant MIN families, which have no narrow twin, from
  their MMQ tile to the tiled dp4a kernel, and Flash-Next's short-prompt
  greedy hash changed. Families without a narrow variant now keep their
  full tile; both Flash-Next hashes are back to baseline.
- The remainder's decode floor and the ~50 ms of request admission stand
  as described above.

Final on the 27B: 2.30 s, 839 tok/s; llama.cpp 894. Greedy output
identical to main on the 27B and on Flash-Next, short and long.

## Pass 8: the request's host time, and where the kernels go

The per-kernel timeline says the GPU is busy essentially all of the prefill,
so anything left is either host time around the request or kernel time
itself. Both were measured rather than guessed.

**Host time.** `FLYWEIGHT_ADMIT_TRACE=1` timestamps the stages between the
POST arriving and the first chunk launching. On a cold 1929-token request:

| stage | ms |
|---|---|
| body read, tokenize, submit | 4.2 |
| task admission | 77.3 |
| eight prefill chunks | 2354 |
| last chunk to the first token out | 24.1 |

All 77 ms of the admission was one item: spilling the conversation the
arriving prompt displaces into the host prompt cache. On this model that is
878 MiB of device-to-host copying, 274 MiB of packed live KV plus four
151 MiB checkpoint arenas, and none of it is work the arriving request
needs -- the entry it builds can only be read by a later turn. The engine
now does it in its idle branch instead (`engine_idle_maintenance`), which
took a cold request from 2.35 to 2.27 s. Recall is unaffected and still
exact: a conversation displaced and then asked for again restores in 0.63 s
against 2.80 s cold, with identical greedy text.

Under sustained load the engine never idles, so the spill still happens at
admission exactly as before. The win is real for a chat client and absent
for a benchmark that hammers requests back to back.

**Kernel time.** `FLYWEIGHT_PREFILL_TIMELINE=2` lists every kernel in a
chunk. For one 256-row chunk, 287 ms of kernel time:

| group | ms | share |
|---|---|---|
| quantized GEMMs | 216.9 | 76% |
| DeltaNet recurrence and conv | 31.5 | 11% |
| attention | 23.0 | 8% |
| elementwise (quantize, SiLU, norm, add) | 15.6 | 5% |

Two things this settled:

- **The DeltaNet's parallel form is not a win here.** Forcing the chunked WY
  path at 256 rows (`FLYWEIGHT_DELTA_CHUNKED_MIN_ROWS`) replaces a 25.4 ms
  recurrence with 27.7 ms across four kernels and 144 more launches, and
  time to first token goes 2.28 -> 2.31 s. The 512-row floor is right.
- **The half-rate MMQ families cannot be converted.** IQ2_XS, IQ2_S and
  IQ1_M carry a separate 4-bit scale for each 16-element half of a 32-group,
  so the k32 int8 MMA cannot apply one scale to the pair; the k16 pair is
  arithmetic, not an omission. Those families are 21% of a chunk. llama.cpp
  has the same constraint for the same formats.

**Chunk size.** A dense model has no routed-expert cache to churn, so the
256-row prefill chunk (tuned on a MoE) is not obviously right for it. Three
runs of 512 in a row read 1.3% faster, which did not survive interleaving:
over three interleaved passes 512 is +0.4% on prefill and -1.2% on decode.
Left alone. 1024 needs 718 MiB of row workspace and does not fit beside a
32K f16 KV cache on a 12 GB card.

**Where the comparison stands.** Three interleaved passes, nine repeats
each, same session, 1929-token prompt, 32K f16 KV:

| engine | TTFT | prefill tok/s | decode tok/s |
|---|---|---|---|
| llama.cpp f3a33dff2 | 2.044 s | 944 | 40.8 |
| flyweight | 2.390 s | 808 | 40.4 |

Decode is level. Prefill is 86% of llama.cpp back to back, and 90% when
requests arrive with the gap a chat client leaves, where the deferred spill
applies (2.27 s, 850 tok/s). Note llama.cpp measures 944 here against 894 in
the first report; the machine, not the build, moved, which is why only
same-session interleaved pairs are quoted.

**What is left is kernel time.** The timeline accounts for essentially all
of the prefill as GPU-busy. The GEMMs are 76% of it and already beat
llama.cpp's standalone on this model's shapes, so the remaining ~300 ms is
in the other 24%: the DeltaNet recurrence at 11% of a chunk is the largest
single item, and it runs as a serial walk over the chunk on a
(value_heads x 128) grid. Its parallel form as written is not faster.
Making that recurrence genuinely parallel is the next real lever and is a
kernel project, not a tuning pass.

### Aside: Turing (issue #70)

Assembling the dumped corpus with ptxas for sm_75 showed the `mma.sync.m16n8k16`
int8 and f16 instructions need sm_80; the corpus guarded them at 7.5, so a
Tesla T4 compiled the PTX and had it rejected at load. Guards and the host's
tensor-core test now sit at sm_80. Unrelated to the comparison above, but
found with the same offline-ptxas method.

The final timings in this report were taken on a laptop GPU that reaches its
power cap after an hour of continuous test runs; late-evening re-measurements
of the same build read 5-10% slower on prefill and decode alike, with the
throttle reason set. The 2.30 s / 839 tok/s figure is from the cooler runs.
