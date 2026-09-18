
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
