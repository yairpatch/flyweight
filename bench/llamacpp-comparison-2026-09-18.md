
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
