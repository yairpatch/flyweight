# HF safetensors loading + the BailingMoE3 hybrid architecture

## Context

Two goals that are usually conflated but are almost entirely independent:

1. **Load HF-format checkpoints** (`config.json` + sharded `*.safetensors` +
   `tokenizer.json`) directly, alongside GGUF.
2. **Run `bailing_hybrid` / `BailingMoeV3`** (Ling 3.0), which is a KDA-linear +
   gated-MLA hybrid MoE that no runtime path here supports.

Neither implies the other. The reference checkpoint on this machine is
`~/Downloads/Ling-3.0-tiny`: **15.79 GB bf16**, 9,283 tensors across 32 shards,
~7.9B total parameters (13.9 GB of that is the routed experts, 0.97 GB the tied
pair of 157k embeddings). Note the checkpoint's own
`index.json -> metadata.total_size` reads 3,869,925,683,584, which matches
neither the file sizes nor anything else; it is wrong. Trust the shard headers.

Decision taken up front: the HF path gets **on-the-fly K-quantization at load**,
not just the existing bf16 -> Q8_0 pass. This is load-bearing, not a nicety:
15.79 GB of bf16 does not fit the 11 GB laptop GPU this repo targets, so even
the *small* member of this family needs quantization to run. Ling-3.0-flash is
127B, where bf16 is ~250 GB.

## What the checkpoint actually is

Verified by reading `model.safetensors.index.json` and
`modeling_bailing_moe_v3.py`, not from the model card:

- 24 layers, hidden 1536, 16 heads, head_dim 128, vocab 157,184, rope_theta 6e6.
- **Layer cadence 3:1.** `attention_layer_type` is full attention when
  `(layer_idx + 1) % layer_group_size == 0` **or** the layer is past the last
  whole group (`modeling:1004-1009`). With `layer_group_size=4`, `num_layers=24`
  that is layers 3, 7, 11, 15, 19, 23 -- 6 MLA layers, 18 KDA layers. The second
  clause is dead here but must be carried; it fires on non-multiple layer counts.
- **MoE from layer 1** (`first_k_dense_replace=1`); layer 0 is a dense MLP at
  `intermediate_size=4608`. 128 routed experts (`moe_intermediate_size=512`),
  top-8, plus 1 shared expert at 512.
- **Router is `noaux_tc`**, not plain top-k: sigmoid over fp32 logits, add
  `expert_bias`, group-limited top-k over `n_group=8` groups keeping
  `topk_group=4` (group score = sum of that group's **top 2**), then gather the
  *unbiased* scores at the chosen indices, renormalize, scale by
  `routed_scaling_factor=2.5` (`modeling:368-406`). The bias steers selection
  only; it never reaches the weights.
- **MLA layers** are DeepSeek-style with a twist: q_lora 256, kv_lora 512,
  qk_nope 128 + qk_rope 64 (qk_head_dim 192), v_head_dim 128, `rope_interleave`,
  and a **head-wise sigmoid output gate** `g_proj` (hidden -> 16) applied to the
  attention output before `dense` (`modeling:709-718`). Scale is
  `qk_head_dim**-0.5` = 192^-0.5, *not* over head_dim.
- **KDA layers**: separate `q/k/v_conv1d` (kernel 4, silu), `f_proj` for the
  per-channel decay, `b_proj` -> sigmoid beta per head, `A_log` per head,
  `dt_bias` per channel, gated output RMSNorm `o_norm` with sigmoid activation.
  `no_kda_lora=true` so `f_proj`/`g_proj` are single matrices, not the a/b pair.
- `expert_swiglu_limit_list` is null for tiny -- no per-layer SwiGLU clamp. The
  flash checkpoints DO have it (llama.cpp shipped a metadata repair script for
  exactly this); carry the field, do not hardcode its absence.
- `num_nextn_predict_layers=0` -- no MTP block in this checkpoint. Out of scope.

## External references (use them; do not re-derive)

Each of these semantics is quiet when wrong, which is the same trap the
muse-glimmer port documented.

- llama.cpp `LLM_ARCH_KIMI_LINEAR` -- KDA and MLA already implemented upstream.
  This is the primary reference for the KDA recurrence.
- llama.cpp PR ggml-org/llama.cpp#26608 (`bailingmoe3`) -- this exact
  architecture, adapting Kimi-Linear KDA + DeepSeek2 MLA + Qwen3-Next MTP.
  Follow its GGUF key naming (`bailingmoe3.*`) so quants interop.
- `fla` (flash-linear-attention) `chunk_kda` / `fused_recurrent_kda` -- the
  authoritative recurrence, including `use_qk_l2norm_in_kernel=True`,
  `safe_gate`, and `lower_bound=-5`.
- The bundled `modeling_bailing_moe_v3.py` is ground truth for everything
  outside the fused kernels.

Note the naming split in the wild: `bailing-hybrid` (prometheusAIR) vs
`bailingmoe3` (upstream PR). Accept both strings; emit `bailingmoe3`.

## Stage 1 -- safetensors as a WeightProvider

The seam already exists. `ColibriV2Model` implements
`colibri::v2::WeightProvider` (`v2_runtime.cpp:94`) whose entire surface is
`format()`, `tensor(index)`, `read_tensor()` (`:122-125`), over descriptors of
`{name, shape, type, offset, size}` on an mmap. Safetensors is that exact shape:
JSON header of dtype/shape/byte-range plus a flat blob.

1. `map_and_parse_safetensors` beside `map_and_parse` (`:3772`), sharing the
   mmap/madvise/mlock body. Detect by directory-with-`config.json` vs `.gguf`.
2. Multi-shard via `model.safetensors.index.json`, reusing the split-GGUF shard
   attach machinery (`:3850-3893`). `Tensor::source` already exists for
   per-shard base pointers.
3. dtype map: BF16 -> 30, F16 -> 1, F32 -> 0. bf16 is a real weight type here
   (CPU matvec `:2472`, GPU `:4576`, embeddings `:2989`), so an unquantized load
   runs on day one.
4. `config.json` -> `ModelConfig`, replacing the GGUF metadata walk
   (`:1527-1730`). Keep it a separate function feeding the same struct.
5. Name mapping HF -> the `blk.N.*` convention every downstream lookup assumes
   (`has_tensor`/`tensor_index`, `:1766-1810`). Table-driven per architecture.

**Gate (revised):** the original gate was "a bf16 HF Qwen3 matches its own GGUF
bit for bit", which isolates loader bugs from architecture bugs. No such
checkpoint pair is on this machine, so by decision the gate is instead run
against Ling directly: every descriptor must resolve **byte-exact against the
`safetensors` reference library**, with full name coverage and correct expert
ordering. That validates the loader without needing the architecture to run --
but it does *not* validate that the descriptors mean what the runtime thinks
they mean. That risk moves into stage 3.

### The expert-layout problem -- SETTLED

Measured, not assumed: a single layer's 128 experts are spread across **8
different shard mappings**. So a stacked descriptor cannot be one base pointer
plus an offset, which is all `Tensor::source` can express, and the
`(expert_count, expert_stride)` idea in the first draft of this plan is simply
not representable.

Resolution: `hf::HfTensor` carries `std::vector<Part>`, an ordered list of
disjoint `(source, offset, size)` ranges whose concatenation is the tensor.
Single-part descriptors are the common case and cost nothing extra. `offset` is
left zero on multi-part tensors so that any consumer reading it without checking
`contiguous()` fails loudly rather than silently reading the wrong bytes.

Still open: the runtime consumers that alias straight into the mapping
(`qwen_alias_static_tensor`, the expert cache page-in) must either learn `Part`
or take a materialized copy. Since stage 2 rewrites every expert into a
quantized arena anyway, materializing during quantization is probably the answer
-- but that means the HF path cannot run *unquantized* on the GPU without more
work, which is fine given the size finding above.

## Stage 2 -- on-the-fly K-quantization at load

The repo currently only *consumes* K-quants; the sole producer is the bf16 ->
Q8_0 requant pass (`:7720-7830`). That pass is the integration template -- it
already walks tensors, respects a policy, rewrites `device_tensor_types`, and
reports savings -- but its quantizer needs siblings.

Q6_K, Q5_K and Q4_K packers now exist in `src/qwen_kquant_pack.h`, ported from
llama.cpp's `make_qx_quants` / `make_qkx2_quants` rather than invented -- a naive
absmax fit costs perplexity without ever failing visibly. IQ* stays out: those
need an importance matrix.

**Gate: PASSED.** `llama-quantize --pure` on the 3.5 GB bf16 MTP checkpoint gives
a byte-level reference with no imatrix in play. Against it, over 970,752
super-blocks of real weights:

| type | RMSE ratio (mine/llama.cpp) | blocks better | worse | identical |
|------|------|------|------|------|
| Q4_K | 1.000000 | 24 | 18 | 970,710 |
| Q5_K | 1.000000 | 1 | 0 | 970,751 |
| Q6_K | 1.000000 | 0 | 0 | 970,752 |

Q6_K is bit-identical everywhere. The 42 divergent Q4_K/Q5_K blocks are ties in
the scale search resolved differently, split near-evenly between better and
worse, with no aggregate accuracy cost. Ruled out as causes, by experiment
rather than assertion: rounding mode (`lrintf` vs llama.cpp's magic-number
`nearest_int` give byte-identical output) and FP contraction (`-ffp-contract=off`
changes nothing). Bit-identity across compilers was never the goal; equal
accuracy is, and that is what the ratio measures.

Two real bugs were found getting there, both silent:

1. **`qwen_half_bits` flushed subnormals to zero.** Its comment asserted block
   scales are "never subnormal in f16" -- true for Q8_0 (absmax/127), false for
   every K-quant, whose super-block scale is a scale *of scales* (`/63`,
   `/-128`) and routinely lands below the f16 normal minimum of 6.104e-5. The
   effect was zeroing whole super-blocks. Now encoded properly. This also fixes
   a latent defect on the existing Q8_0 path for any tensor with absmax below
   ~0.0078.
2. **Q4_K and Q5_K use different search parameters** in the reference (`rmin`
   -1.0/20 steps vs -0.5/15). Sharing one pair silently costs accuracy.

### Load-time driver -- DONE

`include/colibri_v2_hf_quantize.hpp` sizes an arena from the descriptors, then
fills it in parallel (per tensor, `schedule(dynamic)`). Two passes rather than
one so every destination is known before any work starts and the threads never
coordinate.

Measured on the real checkpoint: **15.787 GB bf16 -> 4.566 GB Q4_K in ~24 s**
across 520 tensors. That is the result that makes this family runnable on an
11 GB GPU at all.

Accuracy, decoded back out of the arena through the runtime's own decoders:
7.14% relative RMSE on the stacked expert blocks, against llama.cpp's 7.2857%
for Q4_K on comparable weights -- i.e. the expected figure for the format, not a
defect. The Q4_K/Q5_K/Q6_K ladder is 7.29% / 3.69% / 1.83%.

This also **settles the `Part` question**: quantization has to copy the bytes
anyway, so the arena is where a 128-range, 8-mapping stacked expert becomes
contiguous again. Every descriptor leaving the quantizer is single-part, so no
downstream consumer ever has to learn about `Part`. Nothing in the runtime needs
widening.

Type policy lives in one auditable function (`target_for`): 1-D tensors and
anything with a row under 256 stay f32 (norms, `dt_bias`, `A_log`, the 4-tap
conv kernels), embeddings and the output head take a higher target than the
bulk, everything else takes the target type. On this checkpoint that is 192 f32
/ 326 Q4_K / 2 Q6_K.

### Wiring -- DONE

`colibri_v2_model_open` now accepts a directory. Detection is the whole rule: a
directory containing `config.json` is HF, anything else is GGUF. `map_and_parse`
was split into `map_file` (mmap, no interpretation) plus the GGUF parse, so both
formats share the mapping path including the mlock/madvise behaviour.

The arena reuses `Tensor::source`, the same indirection split-GGUF shards
already use, so **no downstream consumer can tell an HF model from a GGUF one**.
The source mappings are dropped once quantization finishes -- keeping 15.8 GB of
bf16 alive for the process lifetime to serve nothing would be wasteful.

Verified end to end through the Python `V2Model` on the real checkpoint:
`format=safetensors`, `arch=bailingmoe3`, 520 tensors, 4.566 GB, ~25 s.

Target type is selectable through `COLIBRI_HF_QUANT` for now.

**Trap worth recording:** the Python package loads
`src/colibri_next/_native/colibri_v2.so`, *not* the `build/cmake-native` output.
Building with `cmake --build` alone leaves the tests running against a stale
binary that silently passes. Use `python -m colibri_next.native_build`.

Tests: `tests/hf_safetensors_fixture.py` synthesizes a structurally faithful
BailingMoE3 checkpoint (3:1 cadence, dense leading block, experts deliberately
split across two shards) and `tests/test_v2_hf_loader.py` covers translation,
the two-meanings-of-`g_proj` hazard, cross-shard expert stacking, and the type
policy. Suite is 329 passed / 0 failed.

### Quantized-arena cache -- DONE. 2.79s -> 0.00s.

Quantizing is the whole cost of an HF open and its inputs never change. Measured
per-core packer throughput on the 9955HX (`native` sources, `-O3 -march=native`):

| target | Melem/s/core |
|---|---|
| Q8_0 | 388 |
| Q6_K | 55 |
| Q5_K | 39 |
| Q4_K | 30 |

Ling-3.0-tiny is ~7.9G elements, so Q4_K is ~260 core-seconds of packing *per
open* -- the ~25 s above, spent again every launch.

So the arena is written to a sidecar (`colibri_v2_hf_cache.hpp`) and mapped
thereafter. On a 1.5 GiB synthetic checkpoint: **2.79 s cold, 0.00 s warm**, and
the arena is byte-identical across all 172 tensors (sha256 over every tensor
with `COLIBRI_HF_CACHE=0` vs. warm). A hit never opens the shards at all.

Three decisions worth keeping:

* **Fingerprint from metadata, never content.** (config.json's bytes, each
  shard's *base name* + size + mtime, the policy, a format version, and
  `kPackerVersion`). Hashing 15.8 GB to decide whether to avoid packing it would
  cost more than the packing. Base name rather than path so moving a checkpoint
  keeps its cache; config.json's bytes are what keep two models apart.
  `kPackerVersion` MUST be bumped when a packer changes its output -- the arena
  is bytes, and nothing downstream can see that it was packed by older code.
* **Every failure is a miss.** Absent, stale, truncated, corrupt, wrong
  endianness, unwritable directory -- all of them fall through to quantizing.
  The reader bounds-checks every table offset against the mapping for exactly
  this reason, and `write` goes through a temporary + rename so a file is never
  half-there.
* **Beside the checkpoint first, `$XDG_CACHE_HOME/colibri-next` second.** Hub
  caches and shared mounts are read-only often enough to need the fallback. The
  file name carries the fingerprint, so one directory can hold caches for
  several models or several quantizations. `COLIBRI_HF_CACHE=0` disables it (the
  packers cannot be benchmarked otherwise); any other value is a directory.

### Cold open -- DONE. 22.4s -> 3.1s on the packers.

With the cache in, the only slow path left is the first open. Measured on a
12.33 GiB synthetic checkpoint (32 threads, Ryzen 9 9955HX), `quantize` phase:

| | seconds |
|---|---|
| per-tensor work items, scalar packers | 22.4 |
| + 64 Ki-element tiles | 15.5 |
| + AVX2 scale search | **3.1** |

Full cold open is now 4.36 s (3.1 quantize + 1.0 cache write + 0.14 tokenizer);
warm open is 0.16 s, of which 0.14 s is parsing tokenizer.json. Phase timings
are available under `COLIBRI_HF_PROFILE=1`, which is what found all of this.

**Threading was already done.** 1/4/8/16/32 threads gave 286/74/40/21.5/17.0 s
-- 13.3x at 16 threads, and SMT adds another 27%. There was nothing left there,
which is why the work went into the packer instead. Worth knowing before
anyone tries again.

**Tiles, not tensors.** The fill used to allocate a whole-tensor f32 buffer per
work item -- 400 MB per thread on the 201 MB expert stacks -- written once and
read back once, so every weight made two round trips through RAM. Tiles are
256 KB and stay in L2 between the widen and the pack. It also killed the tail:
69 tensors are ~100M elements and the rest are small.

**The AVX2 search is vectorized across sub-blocks, not within one.** A Q4_K
super-block is 8 independent 32-element sub-blocks, so lane j runs sub-block j
through the identical scalar operations in the identical order. No sum is
reassociated, no comparison can land differently, and the result is **bit-exact
against the scalar path** rather than merely close -- which matters because
`make_qkx2_quants` is a search, and a last-ulp difference in the objective flips
which trial step wins. Vectorizing within a sub-block would have been easier and
would have silently changed what every checkpoint quantizes to.

Three things that cost real time to find, all of them invisible in a diff:

* **The span is not loop-invariant.** `make_qkx2_quants` recomputes
  `maximum - minimum` inside the step loop against a `minimum` that a winning
  step moves. Hoisting it looks obviously safe and is not: it made 28839 of
  40000 super-blocks disagree.
* **GCC contracts multiply-adds by default, and that was already changing the
  weights.** `-ffp-contract=fast` fuses the accumulations in both search
  routines, which flips the winning step on ~1 sub-block in 300 -- so the *old*
  scalar packer produced different bytes on a machine with FMA than without.
  The packers now live in `qwen_kquant_pack.cpp` / `qwen_kquant_pack_avx2.cpp`,
  compiled with `-ffp-contract=off` (and `-mavx2` without `-mfma`, so there is
  no FMA to fuse into even if the flag is lost). `kPackerVersion` went to 2.
* **`#pragma GCC optimize("fp-contract=off")` does not work.** It appeared to --
  the AVX2 and scalar paths agreed under it -- because it fused *both* of them
  the same way. Only the command-line flag actually disables contraction. This
  is why `hf_quantize_tiling_contract` pins the packed bytes by hash: it is the
  only thing that catches the flag going missing.

Dispatch follows the existing CPU-backend arrangement: baseline library, one TU
per ISA, hook installed after a CPUID check, and `COLIBRI_CPU_BACKEND=scalar`
forces the reference path. The contract test passes identically both ways.

Still on the table: `make_qx_quants` (Q6_K, the embedding and head) is still
scalar -- ~8% of elements, so a smaller prize with the same shape of solution
(16 sub-blocks of 16, so two AVX2 passes). And `Target` tops out at Q4_K, so an
HF checkpoint cannot reach the IQ2_S / IQ4_XS kernels a GGUF can -- that is a
decode ceiling, not a load one, and the bigger remaining win.

### Tokenizer -- DONE

`tokenizer.json` is read at open: vocabulary, merges, control tokens and the
pre-tokenizer selector all land in the same fields the GGUF metadata path fills,
and `build_tokenizer_tables` is now shared by both so they cannot drift.

**Gate: PASSED.** 3000 randomized adversarial strings (mixed scripts, emoji,
CJK, Cyrillic, Greek, accents, punctuation runs, whitespace runs, contractions,
digits) tokenized identically to `AutoTokenizer` -- 0 mismatches.

Getting there cost three bugs, and the sequence is worth keeping because each
one hid the next:

1. **The pre-tokenizer is not GPT-4o with a smaller digit group.** That was the
   first guess from diffing the two regexes and it was wrong. They also differ
   in the letter rule: GPT-4o splits an upper-to-lower transition, the llama3
   pattern's `\p{L}+` does not. Sharing GPT-4o's rule split "mV" into two
   pre-tokens, so BPE could never apply the (m, V) merge however correct the
   merge table was. Symptom: ~2% mismatch on mixed-case input, nothing else.
   Hence a separate `llama_bpe_pretokenize` transcription.
2. **`\s*[\r\n]` is greedy and backtracks**, so it runs to the *last* line break
   in a whitespace run. Taking the first splits "\r\r" and blocks that merge.
3. **A pre-existing bug in `V2Model.tokenize`**, unrelated to HF and affecting
   every model including Qwen GGUF. `_SPECIAL_TOKEN_PATTERN` is `<[^<>]+>`,
   which happily spans ordinary prose containing a `<` and a later `>`. The old
   code split there unconditionally and, when the candidate turned out not to be
   in the vocabulary, tokenized it *alone* -- inventing a piece boundary the
   pre-tokenizer never produces. Now an unrecognised candidate is left inside
   the surrounding run. `test_tokenize_recognizes_non_pipe_qwen_control_tokens`
   asserted the old behaviour and was updated; the justification is the 27 -> 0
   mismatch change against the reference.

Note what found #3: only the randomized sweep. The ten hand-written cases all
passed both before and after, because none of them happened to contain a `<`
and a later `>` in the same string.

Still to do for this stage:

1. Policy: promote `COLIBRI_HF_QUANT` into a real runtime option, folded into
   `dense_requant` rather than left as a second parallel knob.

## Stage 3 -- the `bailingmoe3` architecture

Only now, with the loader and quantizer proven, add the architecture. Follow the
muse-glimmer commit (0d93e96) as the shape of the change: config plumbing,
kernels, runtime wiring, a synthetic GGUF fixture, and a parity test.

1. **Layer plan -- DONE. Registration in the execution path still open.**

   `bailing::layer_requirements` names every tensor a layer must carry, by
   kind, and `bailing_validate_plan` resolves the whole plan at open. This is
   what closes the hole left by the weakened stage-1 gate: byte-level checking
   proved descriptors point at the right BYTES, never that they were given the
   right MEANING. A misnamed tensor survived that gate and would have surfaced
   much later as a wrong kernel result, indistinguishable from a numerics bug.

   Verified on the real checkpoint -- all 24 layers resolve:
   `24 layers (6 MLA, 18 KDA), 128 experts top-8 in 8 groups keeping 4, 1 shared`.

   The cadence is now derived twice from the same config, once by the parser
   into `sliding_window_pattern` and once by `layer_is_full_attention`, and a
   disagreement throws. Two independent transcriptions of
   modeling:1004-1009 that must agree is cheap insurance on the rule that
   decides which kernel each layer runs.

   Both guards are tested by making them fail: a dropped tensor produces
   `bailingmoe3 layer 0 is missing blk.0.ssm_dt.bias`, and an unknown tensor
   name is refused outright rather than skipped (skipping would leave its bytes
   unreachable and the model quietly incomplete).

   **Gap found and closed while doing this:** `n_group` and `topk_group` were
   never stored. `config_from_json` read every other routing field, so the
   router would have been handed zero groups and silently degraded to flat
   top-k over all 128 experts -- correct-looking output, wrong routing. Now in
   `ModelConfig`, in the public ABI (appended, layout-compatible), and asserted
   through Python.
2. **MoE routing -- DONE.** `include/colibri_v2_bailing.hpp`. The DeepSeek-V4
   router here shares the bias-steers-selection structure but scores with
   `sqrt(softplus)` over a flat expert list, where this scores with sigmoid over
   expert groups, so it needed its own implementation rather than a flag.

   **Gate: PASSED.** Cross-checked against `BailingMoeV3Gate`'s own selection
   path under torch: 2000 randomized cases at the real geometry (128 experts,
   top-8, 8 groups keeping 4) with zero mismatches in either selected ids or
   weights, plus 1500 deliberately tie-heavy cases (three distinct logit values,
   all-identical logits, saturated sigmoids) also with zero mismatches. Ties are
   compared as a weight multiset, since which of several equal-scoring experts
   wins is not determined by the maths -- only the resulting weights are.

   `tests/bailing_router_contract.cpp` pins it for CI on a case built so each
   property is falsifiable alone: the highest-scoring expert is excluded by its
   bias, the lowest-scoring one is included by its bias, and the weight of that
   included expert comes from its raw 0.119 rather than its biased 0.919. All
   expected values come from torch, not from running this code.

   **MoE block assembly -- DONE.** `bailing::moe_block` composes the router with
   the expert application and the shared expert; `swiglu` carries the per-layer
   clamp. Checked against `BailingMoeV3SparseMoeBlock.forward` under torch at
   four scale/clamp combinations, worst relative error 1.2e-06.

   Two assembly properties are pinned because both are quiet when wrong:
   * The shared expert is added to the routed sum and is **not** scaled by
     `routed_scaling_factor` -- that factor belongs to the router's weights,
     which the shared expert does not have (modeling:448-449). Folding it in
     would scale the shared path by 2.5x here and still produce fluent text.
     Tested by zeroing every routed expert so the output IS the shared path,
     then asserting the scale changes nothing, with a guard against the whole
     thing being trivially zero.
   * The SwiGLU clamp bounds **both** halves before combining, not just the
     gate.
3. **MLA layers -- decision SETTLED by inspection, kernels partly done.**

   The open question was whether to lift MLA out of the `deepseek4` runtime or
   write a second one. Reading the code answers it: **there is nothing to
   lift.** DeepSeek-V4's attention here keeps ONE shared KV latent per position
   (`sc.latent`, head_dim wide, cached as f16) that every head reads directly,
   with sink logits and the compressed-block indexer. It has no `kv_b_proj` and
   never materializes per-head keys and values at all.

   BailingMoE3's MLA is the classic DeepSeek-V2/V3 form: `kv_a_proj_with_mqa`
   -> [512 latent | 64 rope], `kv_a_layernorm` on the latent, then `kv_b_proj`
   DECOMPRESSES it into per-head nope keys (128) and values (128). These are
   different operators that share a name. A second implementation is not
   duplication; the only genuinely common parts are rms_norm, rope and matvec,
   which are already shared helpers.

   Done and pinned in `include/colibri_v2_bailing.hpp`:

   * `partial_rope_norm` -- **the trap in this architecture.**
     `apply_rotary_pos_emb_interleave` (modeling:541-577) de-interleaves the
     rope span (`view(d/2,2).transpose(4,3)`) and then applies the ordinary
     half-split rotation. Composing the two gives back exactly the
     ADJACENT-PAIR rotation, leaving a permutation on the output -- and that
     permutation is unobservable, since it is orthogonal, applied to both q and
     k, and the only consumer is their dot product. Verified numerically: the
     reference and this function disagree element for element and agree on
     every attention score, over 300 random rows at positions up to 100k.

     Both nearby readings are wrong. Reproducing the reshape faithfully copies
     a no-op; taking the half-split rotation from the reference's second half
     -- the obvious misreading -- produces fluent, wrong text. Neither is
     visible without doing the composition.
   * `apply_head_gate` -- the head-wise sigmoid gate on the attention output
     that DeepSeek's MLA does not have (modeling:709-718).

   `tests/bailing_attention_contract.cpp` asserts the SCORE rather than the
   vector, for the reason above. Writing it caught a bad assertion of my own:
   an earlier case claimed different positions give different scores, which is
   false when q and k move together -- the rotation is orthogonal, so an equal
   rotation preserves the score exactly. The correct property, now tested, is
   that the score depends on the RELATIVE offset: equal distances score equal,
   different distances score differently.

   **KV cache layout -- DECIDED: cache the latent, staged behind a decompressed
   reference.** The numbers on this checkpoint, f16, 6 MLA layers of 24:

   | context | latent (576/pos) | decompressed (5120/pos) |
   |---|---|---|
   | 8k | 0.06 GB | 0.50 GB |
   | 32k | 0.23 GB | 2.01 GB |
   | 128k | **0.91 GB** | **8.05 GB** |

   Weights are 4.57 GB at Q4_K on an 11 GB card, so decompressed does not reach
   128k and barely reaches 64k once the expert cache wants its share. The cost
   of absorbing `kv_b` into the query and output projections is 3.4x more MACs
   per cached position (17408 vs 5120), which is the right side of the trade on
   a decode this repo has already measured as bandwidth-bound.

   Both forms are implemented in `include/colibri_v2_bailing.hpp`:
   `mla_attention_decompressed` (reference), `mla_attention_absorbed` (latent),
   `mla_decompress` (bridge). Checked against a torch transcription of
   `BailingMoeV3MultiLatentAttention.forward` at 1, 2, 7, 64 and 257 positions:
   worst relative error 4.5e-06 for either form, and 4.8e-06 between them.

   The decompressed path is kept, not deleted. The saving rests on an identity
   -- `q.(W_k @ latent) == (W_k^T @ q).latent` -- and identities that plainly
   hold are precisely where this project's silent bugs have been. The contract
   test asserts the two forms agree, with a guard against both being trivially
   zero.

   **Open risk, unmeasured:** turbo4 KV quantization does not obviously carry
   over. The existing `kv_store_turbo4_k`/`_v` kernels quantize per-head K and V,
   where an error stays inside one head. A 512-wide latent feeds every head
   through `kv_b`, so quantization error there fans out across all 16, amplified
   by whatever that matrix's spectral norm turns out to be. If it proves
   intolerable, decompressed + turbo4 is the fallback: ~2 GB at 32-64k, which
   does not reach 128k but is livable.

   ### Decoder layer -- DONE (host side)

`bailing::decoder_layer` composes the verified pieces into a whole layer:
pre-norm, MLA or KDA, residual, pre-norm, MoE or dense FFN, residual. With it:
`rms_norm`, `short_conv_step` (causal depthwise conv + SiLU, window carried in
the cache so prefill and decode are one call), `mla_step`, `kda_step`, and the
`Geometry` / `LayerWeights` / `LayerCache` structures.

`tests/bailing_layer_contract.cpp` pins five structural properties:
* a layer with every projection zeroed is the **identity** -- this alone catches
  a dropped, doubled, or misplaced residual join;
* **history changes the result**: the same token at position 1 must differ after
  two different position-0 tokens, or the layer is stateless;
* **only the MLA cache grows**: the KDA state stays the same size over six
  tokens while the MLA cache advances, which is the whole point of the 3:1 mix;
* the two layer kinds **take different paths** given identical weights;
* 128 tokens stay finite and bounded, which is where a sign-flipped decay would
  surface.

These are structural, not numeric, and deliberately so -- see below.

**Why no numeric parity yet.** The oracle is unusable: repeated runs of the
reference return different activations, and the corruption is not confined to
the runs that produce NaN. Two clean-looking dumps of the same layer-3 forward
disagreed with each other (token 0 `max|diff|` 0.098 against one, 0.384 against
the next; `ref std` 0.299 then 0.258). So the finite-check guard is necessary
but NOT sufficient, and any parity number measured against this reference right
now would be measuring its corruption. The numeric grounding for every component
inside the layer already exists from the contracts above, all taken against
torch on inputs the harness was not involved in.

### Execution path -- DONE (host, f32). THE MODEL RUNS.

Four ABI entries: `colibri_v2_bailing_create` / `_reset` / `_eval` / `_destroy`.
`_eval` consumes tokens from the runtime's current position and returns logits
for the last one, so repeated calls decode. Weights must be f32 (open with
`COLIBRI_HF_QUANT=F32`); the quantized path waits on the kernels.

**Gate: PASSED.** Full 24-layer forward against the causal reference, real
checkpoint, prompt "The capital of France is":

| metric | value |
|---|---|
| argmax | agrees (13997 = " Paris") |
| top-50 overlap | 50/50 |
| relative RMS over 157,184 logits | 1.9e-04 |
| correlation | 0.999999899 |
| max probability difference | 1.5e-04 |
| KL(ref \|\| ours) | 3.3e-07 |

1.9e-04 relative after 24 layers is consistent with the single-layer measurement
above, where the *reference's* own f32 rounding was 8.1e-05.

**Two bugs this found that no component test could.**

1. **`head_dim` is not `hidden_size / heads`.** This checkpoint is 1536/16 = 96
   by that formula and declares `head_dim: 128`; the KDA projections are
   `heads * head_dim` = 2048 wide. `ModelConfig` had no field for it, so the
   runtime computed 96 and every KDA layer got the wrong shape. Output stayed
   finite and fluent -- it just predicted "\n" instead of " Paris". Every
   component test passed throughout, because each was handed the correct
   geometry by hand; only assembling the real model from the real config
   exposed it. `ModelConfig::attention_head_dim` now carries it.
2. **`COLIBRI_HF_QUANT=F32` left the embedding at Q6_K**, because the embedding
   carries its own target in the policy. "Unquantized" has to mean unquantized.

**And one bug in the reference, now corrected in the harness.** The MLA layers
were running BIDIRECTIONAL: transformers passes no mask and the checkpoint's
`eager_attention_forward` only masks when given one. For a causal LM that is
wrong -- every position below the last sees the future. It hides well, because
the final position attends to the same set either way and the model still
answers " Paris"; but the polluted hidden states propagate through 24 layers and
do reach the final logits. Before forcing causality the same comparison sat at
~0.2 absolute logit error, which reads as "close but something is off" and would
have been very easy to blame on our kernels. The harness now installs a causal
mask when transformers supplies none.

### Generation -- WORKING

`colibri_next.v2.BailingRuntime` wraps the ABI: `eval` advances the caches and
returns the last token's logits, `generate` greedy-decodes, `reset` starts a new
sequence. The prompt goes through in one call and each generated token in
another, which is what the per-layer caches are for.

    COLIBRI_HF_QUANT=F32 python -c "..."   # see /tmp/chat.py pattern
    prompt 'The capital of France is' -> ' Paris.\n\nI want to know the total
    number of ways to arrange the letters of the word "PARIS".'

24 tokens in 15.5 s, 0.64 s/token, f32 on CPU.

### Quantized execution -- DONE

`bailing::Matrix` carries a pointer plus a GGML type code and is implicitly
constructible from `const float*`, so f32 callers (the contract tests, anything
already dequantized) needed no change. `matvec` dispatches through `row_dot`.

Prerequisites, both of which were repo cleanups worth doing anyway:
* The Q8_0/Q4_K/Q5_K/Q6_K element decoders lived in `v2_runtime.cpp` while their
  Q2_K/Q3_K/IQ siblings lived in `src/qwen_kquant.h`. The four formats the HF
  quantizer actually emits were the only ones an out-of-runtime consumer could
  not decode. Moved, with their block-size constants.
* Added `qwen_q8_dot_row` / `q4k` / `q5k` / `q6k`. The `_value` decoders re-unpack
  a block's scales for every element, which is 32-256x redundant inside a
  matvec; these unpack once and stream the codes. Checked against the element
  decoders to 4e-7.

`kv_b` stays f32 even when quantized: the absorbed attention accumulates *along*
its rows rather than dotting them, so it needs them materialized. 8 MB a layer,
48 MB total, against the 4.6 GB the rest saves.

Measured, real checkpoint, "What is the capital city of Japan?":

| format | resident | s/token | output |
|---|---|---|---|
| F32 | 30.1 GB | 0.64 | correct |
| Q8_0 | 7.9 GB | 0.49 | identical text |
| Q6_K | 6.2 GB | 0.70 | identical text |
| Q4_K | 4.4 GB | 0.66 | identical text |

**A false alarm worth recording.** On "The capital of France is", Q4_K/Q5_K/Q8_0
answered "\n" while F32 and Q6_K answered " Paris", which looked exactly like a
bug in the quantized path -- and Q8_0 being wrong while the *less* accurate Q6_K
was right seemed to rule out quantization damage. It was not a bug. The f32
logits for " Paris" and "\n" are 11.2894 and 11.2809: a gap of **0.0086**. Any
perturbation flips them, and Q6_K keeping " Paris" was luck. Confirmed by
measuring properly: relative logit error ranks correctly (Q8_0 0.023 < Q6_K
0.032 < Q4_K 0.082) with top-5 overlap 5/5 for all three, and an unambiguous
prompt gives identical text across every format. Sampled output is a terrible
correctness signal near a tie; the logits were the thing to look at.

### Threading -- DONE. 7.3x.

Q4_K decode went 0.66 -> 0.090 s/token (11 tok/s) from two changes:

* **Parallelize matvec over output rows.** Rows are independent. The scalar
  single-threaded version ran at ~4 GFLOP/s on a 32-thread machine, one to two
  orders of magnitude off the hardware. A size threshold keeps the small
  projections (the 16-wide head gate, the router) out of the thread pool, where
  the fork costs more than the work.
* **Default to physical cores, not logical.** This is the non-obvious half.
  OpenMP defaults to the logical cpu count and that is measurably the wrong
  answer here -- on a Ryzen 9 9955HX (16 cores / 32 threads):

  | threads | s/token |
  |---|---|
  | 4 | 0.303 |
  | 16 | 0.095-0.101 |
  | 32 | 0.180 |

  32 threads is nearly **2x slower** than 16, reproducibly. Weight-streaming
  matvecs are bandwidth- and cache-bound rather than issue-bound, so a second
  thread on the same physical core buys nothing and costs contention for its
  L1/L2. The thread count is now read from Linux's `thread_siblings_list`
  rather than assumed; an explicit `OMP_NUM_THREADS` always wins.

  Worth remembering as a benchmarking trap: the scaling curve is not monotonic,
  so "measure at 1 thread and at max" would have concluded that threading gave
  3.7x and stopped, missing half the win.

### SIMD -- DONE. 12x total.

No new kernels were needed. `qwen_quant_dot_avx2` / `_avx512` already existed in
the CPU backend and already covered exactly the types the HF quantizer emits
(Q8_0, Q4_K, Q5_K, Q6_K); `matvec` now selects the widest available path once,
outside the row loop, and falls back to the scalar `row_dot`. The standalone
contract tests build with `COLIBRI_BAILING_NO_SIMD` since they do not link the
AVX objects.

Q4_K decode, cumulative:

| stage | s/token | tok/s |
|---|---|---|
| scalar, single-threaded | 0.66 | 1.5 |
| + threaded over rows | 0.165 | 6.1 |
| + physical-core default | 0.090 | 11.1 |
| + existing SIMD kernels | **0.055** | **18.2** |

Verified equivalent rather than assumed: SIMD against scalar on the same
weights is 4.6e-07 relative RMS with the same argmax, and both sit at exactly
0.0818 relative to F32 with 5/5 top-5 overlap.

The generated *text* did change, which looked alarming for a moment. It is the
near-tie sensitivity again: greedy decoding compounds a 1e-5 logit difference
over 16 steps. Logits are the signal; sampled tokens are not.

### Batched prefill -- DONE, but only 1.6x

`decoder_layer_batch` runs a whole prompt through each layer at once. What
batches and what does not:

* **batches**: norms, every projection, the dense FFN, and the MoE experts --
  the last by inverting the routing and running each expert once over the tokens
  that chose it, so a popular expert's matrices are read once rather than once
  per token;
* **does not**: the KDA convolution and recurrence, and MLA attention. All three
  are sequential in position by construction, so they still run token by token,
  reusing the batched projections.

Verified equivalent to token-by-token: max abs logit diff 7.9e-06 with the same
argmax over a 24-token prompt.

Prefill went 40 -> 61-65 tok/s. **Less than hoped, and worth recording why the
obvious next step did not help.** The naive batch (weight row re-read per token)
gave 1.5x; switching to the register-blocked `qwen_quant_dot_quad/oct` kernels,
which decode a weight row once and apply it to 4 or 8 activation vectors, added
only a further ~5%. That says weight traffic is no longer the constraint. Likely
candidates, unmeasured: the sequential KDA/MLA portions now dominate, and MoE
grouping thins out (256 tokens over 128 experts at top-8 is ~16 tokens per
expert, so each expert's batch is small). Profiling before optimizing further is
the right move rather than adding more kernels on a guess.

Current state, Q4_K on a Ryzen 9 9955HX:

| phase | throughput |
|---|---|
| prefill | ~62 tok/s |
| decode | ~18 tok/s |

### Profiling -- and the 3x it found

`COLIBRI_BAILING_PROFILE=1` turns on phase timers (`bailing::Profile`), split
along the line that matters: what batches versus what is sequential in position.
Prefill of 256 tokens, Q4_K:

| phase | before | after |
|---|---|---|
| KDA conv+recurrence | 1.710 s (43.2%) | 0.361 s (24.1%) |
| MLA rope+attention | 1.368 s (34.5%) | 0.145 s (9.7%) |
| MoE router+experts | 0.530 s (13.4%) | 0.615 s (41.0%) |
| projections (batched) | 0.319 s (8.1%) | 0.349 s (23.3%) |
| dense FFN + norms | 0.035 s (0.8%) | 0.028 s (1.9%) |
| **total** | **3.96 s** | **1.50 s** |

The first column is why batched prefill only bought 1.6x and why the
register-blocked kernels added 5%: **77.7% of the time was in the two phases
that do not batch**, and everything being optimized lived in the other 21.5%.
That was a guess in the previous round; measuring it took twenty minutes and
turned a stalled 5% into 3x.

The fix follows from reading the loop rather than the profile: sequential *in
position* is not the same as serial. Each KDA head's state evolves over rows
without touching any other head's, so the nest was inverted to head-outer /
row-inner -- still strictly ordered in position, now parallel across heads, and
forking once per call instead of once per row. MLA attention is independent per
head outright. Both were scalar and single-threaded before.

Result: KDA 4.7x faster, MLA 9.4x faster.

| phase | before profiling | after |
|---|---|---|
| prefill | 62 tok/s | **186 tok/s** |
| decode | 18 tok/s | **29 tok/s** |

Correctness unchanged throughout: batched still equals stepwise to 7.9e-06 with
the same argmax, and Q4_K sits at exactly the same 0.0818 relative to F32 with
5/5 top-5 overlap it had before any of this.

**MoE is now the dominant phase at 41%** -- so that, not more kernel work on the
attention paths, is where the next CPU effort belongs.

### MoE: two hypotheses, both killed by measurement

Sub-profiling `moe_block_batch` (256-token prefill, Q4_K) put essentially all of
MoE in the expert matmuls: routing 0.9%, gather/scatter 1.6%, shared expert
3.9%, **expert matmuls 28.6%** of total prefill. Routing and the gather/scatter
design are fine and need no work.

The expert matmuls run at ~25 GFLOP/s against ~420 for the batched projections,
doing identical arithmetic. Two explanations were tried:

1. **"Fork overhead dominates."** With ~16 tokens per expert, three matmuls per
   expert and 23 MoE layers, the inner-parallel arrangement forks ~8800 times
   per prefill. Rewrote it to parallelize over experts instead, with per-thread
   accumulation buffers. **Slower: 165 tok/s against 183.** The reduction over
   16 private `tokens x hidden` buffers costs more than the forks it removes --
   and, decisively, a version with *no* inner forks at all was still slower, so
   fork overhead was not the constraint. Reverted.
2. **"Weight reuse scales with batch size."** Each expert sees ~16 tokens at a
   256-token prefill, so its 2.4M parameters are read almost per-token; longer
   prefills should give more tokens per expert and better amortization.
   Measured across prefill lengths:

   | tokens | per expert | tok/s |
   |---|---|---|
   | 32 | ~2 | 169.9 |
   | 128 | ~8 | **202.0** |
   | 512 | ~32 | 170.2 |
   | 1024 | ~64 | 153.8 |

   Throughput *falls* with more tokens per expert. MLA attention is O(n^2) in
   context and swamps any reuse gain, so the reuse effect is not observable this
   way and cannot be exploited by simply batching harder.

Both are recorded because they are the two obvious ideas, and the next person
would otherwise spend the same afternoon. What remains true is the gap itself:
the expert matmuls are ~17x less efficient than the projections, and the cause
is MoE sparsity limiting weight reuse. That is a structural property, not a
tuning problem, and the answer to it is the GPU rather than more CPU
arrangement.

Still to do, in value order:

1. ~~**MoE, now 41% of prefill.**~~ See above -- two attempts, both reverted. Expert grouping thins out (256 tokens over 128
   experts at top-8 is ~16 tokens each), so each expert's batch is small and the
   weight read is amortized poorly.
### KDA CUDA kernel -- DONE (kernel only, not yet wired to the GPU runtime)

`bailing_kda_recurrent_chunk` is in `colibri_v2_native_kernels.hpp`. As
predicted from reading the DeltaNet kernel, it is that kernel with one change:

    DeltaNet   local_state[key] *= decay_scale        (one scalar per head)
    KDA        local_state[key] *= shared_decay[key]  (one per key channel)

The decay becomes a shared vector, filled by thread `lane` for key=lane, because
a thread owns one VALUE column and touches every KEY row. Everything else --
the register-held state, the warp reductions for the q/k L2 norms, the beta
gate, the guarded softplus, the 1/sqrt(head_dim) query scale -- is unchanged.

**Verified without a GPU.** The build compiles the CUDA corpus as host C++ for
the CPU backend, so `tests/bailing_kda_kernel_contract.cpp` launches the real
kernel text and compares it to `bailing::kda_recurrence`:

| shape | output | state |
|---|---|---|
| 1 head, 1 row | 1.4e-06 | 4.9e-07 |
| 4 heads, 5 rows | 2.8e-06 | 2.2e-06 |
| 16 heads, 17 rows | 3.7e-06 | 4.8e-06 |
| 2 heads, 64 rows | 9.9e-06 | 2.2e-06 |

That chains back to the reference: the host implementation is pinned to
flash-linear-attention by `kda_reference_check.py` and to the real checkpoint's
layer 0 at 9.8e-07. This matters on this machine specifically, where PyTorch
cannot see the GPU at all -- colibri's own CUDA path initializes fine
(`colibri_v2_gpu_init` returns 0 on the RTX 5070 Ti), but nothing else could
have tested this kernel here.

Two mechanical notes for whoever adds the next kernel: the corpus is split into
raw-string segments to stay under MSVC's 16 KB literal limit, and the generator
tells you exactly where to break when you overflow one. Adding a kernel needed
two new breaks.

### MLA CUDA kernel -- DONE (kernel only)

`bailing_mla_attention`, one block per head, five stages: pull the query through
kv_b's key half, score against the raw latents plus the shared rope half,
max-shifted softmax over a strided range with two block reductions, mix in
latent space, then decompress once through kv_b's value half. It runs the
absorbed form, so attention never decompresses the cache -- which is what makes
the KV cache 8.9x smaller.

`scores` is caller-supplied scratch of `heads * positions`: positions is
unbounded, so the softmax needs somewhere that is not shared memory. Requires
`kv_lora <= 512`, which holds here (512).

Checked against `mla_attention_absorbed` the same way:

| shape | worst relative |
|---|---|
| 1 head, 1 position | 0 |
| 4 heads, 7 positions | 6.9e-07 |
| 16 heads, 128 positions | 2.7e-06 |
| 8 heads, 257 positions | 2.5e-06 |

The 128 and 257 cases are deliberate: the softmax reductions walk a strided
range in a 128-thread block, so an off-by-one only shows up once positions
crosses that stride, and again once it crosses it twice.

Both kernels now live in `tests/bailing_kernels_contract.cpp`.

### Both kernels verified ON THE DEVICE

The CPU-emulated contract proves the kernel *text* is right. It cannot prove the
kernel compiles for the target architecture or runs there. Both now do, on the
RTX 5070 Ti (sm_120):

| kernel | vs host implementation |
|---|---|
| `bailing_kda_recurrent_chunk` | output 3.9e-06, state 5.1e-06 |
| `bailing_mla_attention` | output 3.5e-06 |

Worth noting for the record: PyTorch on this machine cannot use this GPU at all
(its build stops at sm_90), while colibri's own NVRTC path compiles the whole
corpus for it without complaint. The card was never the problem.

**One integration step that is easy to miss.** `colibri_gpu_launch_named`
resolves kernels from a hardcoded name list in `gpu_driver.cpp`, not by
enumerating the module. A kernel that compiles fine and is present in the
cubin still returns -2 ("not found") until its name is added there. Both are
now listed, alongside the DeepSeek-V4 entries, in the same
resolved-if-present style so a build without them still loads.

Still to do: the runtime wiring proper -- device arena sizing, weight uploads,
the expert cache, stream management. The kernels are correct and reachable; what
is missing is a runtime that calls them instead of the host path.

### GPU on by default -- DONE. Host prefill + device decode.

The GPU path was opt-in behind `COLIBRI_BAILING_GPU=1` because the device had to
take prefill as well as decode, and device prefill is a token-at-a-time loop
where the host's is batched. Measured, 12.33 GiB synthetic checkpoint:

| | prefill | decode Q4_K | decode Q6_K |
|---|---|---|---|
| host | 158-165 | 58.7 | 44.0 |
| device, before | 84 | 82.8 | -- |
| **host prefill + device decode** | **155-159** | **61.7** | **172.5** |

Break-even before this was `generated ≈ 1.35 × prompt` at Q4_K -- so on a long
prompt with a short answer, which is most of them, the flag-off default really
was the faster choice. Now the device is not worse at anything, so the flag has
nothing left to trade and the path is on wherever a device exists.
`COLIBRI_BAILING_GPU=0` still forces the host, which is what keeps the two
comparable; the host remains the correctness oracle.

**What made this cheap: not writing a batched device prefill.** The obvious
route was to give the device its own batched prefill -- several hundred lines of
new CUDA, and the `*_matmul_rows` kernels are already there for it. But the host
prefill is *already* batched and correct, and the only reason it could not feed
device decode is that the two caches are separate. So the fix is to move the
cache instead of duplicating the code: `bailing_cache_transfer` uploads or
downloads at the transition. The layouts are identical -- same strides, same
[position][channel] order, same q/k/v window order -- because the device path
was written against the host one, so it is a transfer and not a conversion.
~13 MB per switch, once per prefill/decode transition rather than per token.

`cache_on_device` tracks which copy is authoritative. It is tracked rather than
inferred from the call shape, because this is precisely the split that produced
the earlier "answers with no context" bug -- decode reading a device cache the
prompt never touched, first token plausible and everything after it invented.

**That bug now has a test.** `test_a_prompt_means_the_same_whether_it_arrives_
whole_or_in_pieces` sends a prompt whole and one token at a time and requires
the generated tokens to match. Feeding one token at a time keeps both phases on
one path, which is why the original bug survived a suite that only ever did
that; sending it whole is what splits them.

**And it shipped with a second one anyway: the transfer did not bind the CUDA
context.** Driver contexts are per thread; the server builds the runtime on one
thread and generates on an engine worker. `bailing_gpu_step` had bound the
primary context there for exactly this reason, with a comment saying so, and
`bailing_cache_transfer` did not -- so every upload failed and the server died
on its first token with "bailing cache transfer failed" while every test passed.
The binding is now one shared `bailing_gpu_bind_thread`, and the rule is: every
entry point that touches the device binds, not just the ones that run kernels.

`test_it_generates_from_a_thread_that_did_not_build_the_runtime` drives
generation from a second thread, which is the gap the whole suite had. Note the
shape of both failures -- tests exercised one path, the server exercised two.
When this path is changed, test the *transition*, not the steady state.

**Then the server still showed 82 tok/s where the benchmark showed 128.** Not
the context window (128192 and 8192 measure the same) and not the prompt length
(a 100-token prompt costs 4%). It was the sampler: every benchmark here used
`temperature: 0` and every real client sends a temperature.

| request | decode |
|---|---|
| short prompt, `temperature: 0` | 146.3 tok/s |
| 100-token prompt, `temperature: 0` | 140.1 tok/s |
| short prompt, `temperature: 0.7, top_p: 0.95` | 87.9 tok/s |

`BailingRuntime.sample` did the whole thing over the vocabulary: widened all
157k logits to float64, masked all but `top_k` (default **20**) to -inf, then
argsorted, cumsummed and `np.random.choice`d over 157k entries of which at most
20 could be drawn -- and built a fresh `default_rng()` each token, seeding from
the OS entropy pool 150 times a second. ~5 ms against a ~7 ms token.

Now it selects candidates first and does everything on those. Exact, not an
approximation: discarded entries have probability zero, so they can neither
enter the top_p prefix nor be drawn. With no top_k it takes a 256-wide prefix
and doubles until the prefix holds `top_p` of the mass, which keeps it exact
while touching a few hundred entries instead of the vocabulary. **87.9 -> 141.6
tok/s, and greedy is unchanged.** Checked by sampling 4000 draws per
configuration against the old implementation's distribution.

The lesson is the same one as the threading bug, in a third costume: the
benchmark and the server were not doing the same thing. `temperature: 0` is not
a representative decode benchmark, because nothing in production runs that way.

On the real Ling-3.0-tiny at Q6_K, through the server and directly:

| | prefill | decode |
|---|---|---|
| host | 181 | 43.9 |
| host prefill + device decode | 186 | **128.2** |

Two defects found on the way, both HF-specific:

* **HF checkpoints could not use the GPU at all.** `target_for` left any 2-D
  tensor with rows < 256 as f32 -- the MLA up-projections, whose rows are
  `q_lora_rank` wide. There is no f32 GPU matvec, so it failed outright with
  `bailing GPU matvec failed for type 0`. A 128-wide row cannot hold a 256
  -element K-quant block but holds exactly four Q8_0 blocks, so it takes those.
* **`bailing_q4k_matvec` was dead code.** Written because the generic Q4_K
  matvec calls `q4k_value` per element (its own comment measures that at 58-131
  GB/s on a ~670 GB/s card), compiled, resolved into the function table -- and
  never launched. Wiring it up gained 10%, not the 2x hoped for: Q4_K is still
  ~1.9x slower than Q6_K, because the fast path is block-per-row with a block
  reduction while `q6k_matvec_transposed_warp` is warp-per-row over 8 rows. A
  warp-per-row Q4_K is the remaining work, and it is worth real money: at Q6_K
  the device decodes 3.9x the host, at Q4_K only 1.05x.

The quant type now dominates device decode, and not in the direction size would
suggest -- Q4_K 86.7, Q5_K 147.9, Q6_K 168.0, Q8_0 153.3 tok/s. The *smallest*
type is the slowest, which is a kernel property, not a bandwidth one. Anyone
choosing a default for the HF policy should read that table first.

### GPU decode -- WORKING, bit-identical to the host

`COLIBRI_BAILING_GPU=1` puts decode on the device. Opt-in, because it covers
decode only: prefill would need either a device-side top-k or a sync per token
for MoE routing, and neither is worth building before the wiring is proven.

Everything except residency and sequencing already existed --
`qwen_gpu_matvec_by_type` dispatches the quantized matvecs, `colibri_gpu_rms_norm`
the norms, and the two kernels above cover KDA and MLA. Seven small helpers were
added for the rest (rope, query split, head gate, short conv, gated head norm,
SwiGLU, copy), each the device half of a host function in
`colibri_v2_bailing.hpp`.

Result on Ling-3.0-tiny at Q4_K: 579 tensors resident, **decode 34 -> 50 tok/s**,
and the logits are **bit-identical to the host path** (relative RMS 0.0000,
same top-5, same argmax). Generation matches too.

**The bug worth recording.** The first working version ran at 96.7 tok/s and
produced finite but completely uncorrelated logits -- range [-1.28, 0.97]
against the host's [-12.95, 11.75]. Cause: `colibri_gpu_rms_norm` and
`colibri_gpu_scaled_add` take no stream argument and run on the DEFAULT stream,
while the matvecs and custom launches were issued on `gpu.stream`. The two
halves of every layer were unordered with respect to each other. Moving
everything to the default stream fixed it exactly -- and the "speedup" from 50
to 96.7 tok/s was simply the work not happening in order.

Finite-but-uncorrelated output is the signature to remember here: NaN would have
been easier, since it points at arithmetic. This pointed at scheduling.

### Chasing the slowness: what it actually was

50 tok/s on this card was far too slow, so everything got measured rather than
guessed. Three suspects were eliminated by measurement before the real one
turned up:

| suspect | measured | verdict |
|---|---|---|
| per-layer routing sync + download | 15.8 us each, 0.36 ms/token | 1.8%, not it |
| kernel launch overhead | 1.59 us each, 1168 launches, 1.9 ms/token | 9%, not it |
| MoE launch count (874 of 1168) | grouped kernels: 50 -> 54 tok/s | small, not it |

Device-side phase timing then showed 11.4 ms/token of GPU work against 18.7 ms
measured wall clock. **The missing 7 ms was not on the GPU at all.**

`BailingRuntime.eval` did `list(self._logits)` -- converting 157,184 ctypes
floats into a Python list on every call -- and `generate` then ran a
Python-level `max()` over it:

| step | cost |
|---|---|
| `list()` of 157k ctypes floats | 6.93 ms |
| Python `max()` argmax | 1.96 ms |
| numpy frombuffer + argmax | **0.01 ms** |

8.9 ms of interpreter overhead per token, against a model that takes 11.4 ms.
`eval` now returns a numpy array when numpy is present and `generate` argmaxes
the ctypes buffer in place without copying.

**Decode: 54 -> 96 tok/s on GPU, and 34 -> 54 tok/s on CPU** -- the same fix
helps both, since the overhead was per call rather than per backend.

The lesson is the same one the MoE round taught, in a different place: profile
the whole path, not the part you are proud of. Three rounds of kernel and
launch-count work moved 50 to 54 tok/s; deleting a list comprehension moved
54 to 96.

Two smaller wins after that, both found by measurement:

* **92 blocking uploads per token.** The grouped MoE path pushed three expert
  pointer arrays and the weights separately, four `colibri_gpu_upload_sync`
  calls per MoE layer at ~15 us each -- more than the expert matmuls they fed.
  Packed into one buffer and one transfer: 96 -> 100 tok/s.
* **A kernel rewrite that did NOT work, kept as a negative result.**
  `bailing_q4k_matvec` unpacks each 32-element sub-block's scales once per warp
  instead of per element, exactly the change that gave the CPU its speedup. On
  the GPU it is 0.71-1.14x -- no better, and slower on the lm head. The scale
  loads are shared across a warp and hit L1, so they were never the cost. The
  kernel is in the tree, bit-identical to `q4k_matvec_transposed` and unused,
  because the measurement is worth more than the code.

**The ceiling, measured rather than assumed.** A plain streaming copy on this
card reaches **361 GB/s** -- it is the laptop 5070 Ti, not the 670 GB/s desktop
part. Active weights are ~0.65 GB per token at Q4_K, so the floor is ~1.8 ms and
the ceiling ~550 tok/s. At 100 tok/s there is ~5x left, and it is all in the
matvec kernels: 60-77 GB/s on expert shapes, 128 GB/s on the lm head, against
361 achievable. Low occupancy on small matrices is the likely cause -- an expert
matvec launches 512 blocks of 128 threads and each block reduces only 1536
values -- but that is a hypothesis, and this section is a record of how often
those have been wrong.

Where the remaining GPU time goes, per token (device timing, 11.4 ms):

| phase | share |
|---|---|
| MoE | 56.3% |
| KDA | 29.3% |
| MLA | 9.3% |
| lm head | 4.3% |

And the matvec kernels themselves are the floor: measured 58-78 GB/s on expert
shapes and 131 GB/s on the lm head, against a card that does ~670. `q4k_value`
decodes each element independently -- recomputing block offsets and reloading
the f16 scales per value -- so these kernels are instruction-bound, not
bandwidth-bound. That is the same inefficiency the CPU side fixed with
`qwen_q4k_dot_row`, and it is where the next real speedup is.

### Server integration -- WORKING

`NativeV2InferenceService` now branches on architecture: `bailingmoe3` gets
`BailingGenerator` + `BailingEngine` instead of the Qwen runtime. The engine
presents the same submit/cancel/forget/close surface `ChatGenerator` already
expects, so nothing above it changed.

    SERVICE UP: {'backend': 'native-v2-bailingmoe3', 'device': 'gpu',
                 'parallel_sequences': 1}
    REPLY: 'Tokyo'

Chat completions, streaming and sampling (temperature / top-k / top-p) all work.
Sampling runs on the ctypes logits buffer through numpy, never a Python list --
the same lesson as the decode path.

**The server layer costs ~10%, not half.** The first reading -- 106 raw against
54 streaming -- was measured on a 32-token generation and was mostly prefill,
not overhead. Profiling put 1.613 s of 1.626 s inside `_eval_into_buffer`;
sampling, detokenization, the queue and every lock together were under 5 ms.
The GIL switch interval was ruled out too (0.005 / 0.001 / 0.0001 s all give
61 tok/s).

Varying the output length separates the fixed cost from the per-token one:

| output tokens | time | apparent rate |
|---|---|---|
| 17 | 0.39 s | 44 tok/s |
| 65 | 0.86 s | 75 tok/s |
| 193 | 2.22 s | 87 tok/s |

A straight-line fit gives **10.63 ms/token (94 tok/s) plus a fixed 169 ms**, and
predicts all three points to 0.04 s. That fixed cost is the 33-token prompt
prefilled at ~195 tok/s -- which is exactly the measured CPU prefill rate,
because the GPU path handles single tokens only.

So: decode through the server is 94 tok/s against 106 raw, and the thing worth
fixing is prefill, not the server layer. It matters more for serving than these
numbers suggest -- a 1000-token prompt costs ~5 s before the first token.

Recorded because the shape is easy to misread: a fixed startup cost divided by
a short run looks exactly like per-token overhead, and the fix for one is
nothing like the fix for the other.

**The bug that mattered: CUDA contexts are per thread.** The weights upload on
whichever thread builds the runtime, but the server generates on an engine
worker, which starts with no current context -- so every launch there failed
with a bare "launch failed". `bailing_gpu_step` now binds the primary context
once per thread via a `thread_local` guard, which costs nothing after the first
call and makes the runtime safe to drive from any thread.

### Two GPU bugs the server found that every test had missed

Both produced fluent-then-degenerate output -- correct first token, then
invention, eventually collapsing into repetition like `"ItttIttItt..."`.

**1. `reset` never cleared the device caches.** It cleared
`runtime->caches` (host) and left `gpu.layers[].state`, `conv_windows`,
`latents` and `rope_keys` untouched. A KDA layer's recurrent state is a running
summary of everything it has seen, so the contamination compounded across
requests rather than washing out. Caught by the sharpest available test: five
*identical greedy* requests returning five *different* answers. Greedy is
deterministic, so that is not a quality problem, it is state leaking.

**2. The host and device caches are separate, and prefill used the wrong one.**
`eval` routed multi-token calls to the host batched path and single-token calls
to the device. So a server sending a whole prompt at once filled the HOST cache
and then decoded on the DEVICE, whose cache the prompt had never touched -- the
model answered with no context at all.

The second is the one worth dwelling on, because **the test methodology hid it**.
Every GPU test in this file fed prompts one token at a time
(`for t in ids: rt.eval([t])`), which keeps prefill and decode on the same path
and makes the bug invisible. Only a real server, sending `eval(prompt_ids)` in
one call, splits them. Bit-identical logits against the host path, verified
repeatedly, were bit-identical *for the one calling pattern the tests used*.

Fix: once the GPU path is live, every token goes through it, prompt included.
Prefill drops from the host path's ~195 tok/s to ~94 until the device gets a
batched prefill of its own, which is the right trade -- and the device caches
are now cleared on reset, including the MLA ones that do not strictly need it,
because a "reset" that leaves some state behind is a distinction nobody
remembers a year later.

After both fixes: identical greedy requests give identical answers on both
paths, and GPU is ~2x faster end to end (48 tokens in 0.71 s against 1.44 s).

Deliberate limitations, all consequences of one sequence of caches:

* **one request at a time** -- concurrent requests queue rather than interleave,
  and `_serialize_generation` is set accordingly;
* **no prefix cache** -- `prefix_cache_stats` reports zeros rather than the Qwen
  runtime's numbers, which would describe a different runtime;
* **prefill stays on the CPU**, since the GPU path handles single tokens only;
* none of the expert-paging, KV-precision or parallel-slot options apply.

Remaining on the GPU path:

1. **Prefill.** Needs device-side routing to avoid a sync per token.
2. **The per-layer routing sync.** Decode currently downloads 128 router logits
   and routes on the host, 23 times per token. A device-side top-k would remove
   it and is probably most of the gap between 50 tok/s and what the card can do.

2. ~~**Wire the GPU path into `ColibriV2BailingRuntime`**~~ -- done, decode only.
   (`colibri_v2_native_kernels.hpp`), which is where the real speed is. KDA is a
   targeted edit of `qwen_delta_recurrent_chunk`: scalar decay becomes a
   per-channel vector.
2. **Batched prefill.** Every token currently walks the layer stack alone.
3. **SIMD row dots.** The new `*_dot_row` functions are scalar; the AVX2/AVX-512
   paths in `qwen_cpu_*.cpp` are the model to follow.
### KDA oracle -- DONE (implementation not started)

`native/tools/kda_reference.py` is a NumPy oracle for the recurrence, in the
style of `deltanet_reference.py`, and `kda_reference_check.py` pins it to
flash-linear-attention's own definitions. Both of fla's forms agree with it at
float32 noise level:

| compared against | output rel-err | state rel-err |
|---|---|---|
| `naive_recurrent_kda` (decode path) | 5.7e-06 | 8.0e-06 |
| `naive_chunk_kda` (prefill path) | 1.5e-05 | 3.4e-05 |

Checking both matters: the model picks between them by sequence length
(`fused_recurrent` at <= 64 tokens, `chunk` above), so the port has to match
both, and a single oracle agreeing with each is what makes that possible.

The structural finding, which sets the kernel shape: **KDA is DeltaNet with a
vector decay instead of a scalar one.** DeltaNet decays the whole state by one
number per head per step; KDA decays it per key channel, so the update is
`S <- diag(exp(g)) @ S`, and `g` is a full [heads][head_dim] tensor from its own
projection. The delta rule, the beta gate and the query readout are otherwise
the same shapes the DeltaNet kernel here already handles.

**Not verified:** the Triton kernels were not compared. This machine's PyTorch
build does not support the GPU's compute capability (sm_120), so that comparison
skips rather than silently passing. fla's naive forms are what its Triton
kernels are tested against upstream, so this is a reasonable oracle -- but it is
one step removed from what the checkpoint actually executes.

**Open question, recorded not resolved.** `modeling_bailing_moe_v3.py` passes
`safe_gate=True` and `lower_bound=-5` into `chunk_kda`. In fla 0.4.1 --
the version the checkpoint runs against -- `chunk_kda` takes `**kwargs` and
**both are silently discarded**; `lower_bound` is implemented for HGRN and never
for KDA. So the reference model's observable behaviour has no clamp, and the
oracle reproduces that. But llama.cpp's bailingmoe3 PR carries a
`bailingmoe3.kda.safe_gate` hyperparameter key, so some implementation does
honour it. If it turns out to matter it is one line on the gate, and it changes
long-context decay only.

### KDA recurrence -- DONE (host side)

`bailing::kda_recurrence` in `include/colibri_v2_bailing.hpp`, verified against
the oracle at float32 noise level across every shape that matters:

| shape | output rel-err | state rel-err |
|---|---|---|
| rows=1, 16x128 (decode) | 4.3e-06 | 4.5e-06 |
| rows=8, 16x128 | 4.3e-06 | 4.2e-06 |
| rows=64, 16x128 (chunk boundary) | 5.7e-06 | 5.3e-06 |
| rows=128, 16x128 (prefill) | 6.7e-06 | 5.7e-06 |
| rows=5, 4x32 (non-square) | 1.6e-06 | 1.9e-06 |

Reading the DeltaNet CUDA kernel confirmed the relationship is even closer than
the oracle suggested: `qwen_delta_recurrent_chunk` computes the decay as
`expf(coefficient * softplus(x + bias))` with a negative coefficient, which is
*exactly* KDA's `-exp(A_log) * softplus(...)` folded into one exp. The only
differences are that DeltaNet's softplus argument and bias are per head where
KDA's are per channel, and that DeltaNet writes `state[key] *= decay_scale`
where KDA writes `state[key] *= decay[key]`. So the eventual CUDA kernel is a
targeted edit of an existing one, not a new kernel.

`tests/bailing_kda_contract.cpp` pins three things:
- the numeric case, from the oracle;
- **decode equals prefill**: feeding rows one at a time with the state carried
  must equal feeding them at once. Measured bit-identical, not merely close.
  This is the property that keeps a conversation from diverging from its own
  prefill, and it is cheap to break when the kernel is later split into chunked
  and single-token paths;
- **the decay contracts**: the gate is negative for every finite input, so
  `exp(gate) < 1`. A sign slip makes the state grow without bound over long
  context instead of failing outright, so it is driven 256 steps with a large
  positive gate where a wrong sign is unmissable.

Not done: the CUDA/CPU-corpus kernel in `colibri_v2_native_kernels.hpp`. This is
the host reference it will be built against.

4. **KDA layers.** Closest existing code is the Qwen3-Next gated-delta path
   (`ssm_conv1d`, `ssm_a`, `ssm_dt.bias`, `ssm_alpha`/`ssm_beta` -- `:1828`,
   `:8990`) and the validated chunked-WY DeltaNet prototype. KDA differs by: a
   per-channel diagonal decay from `f_proj` instead of a scalar gate, L2-normed
   q/k inside the kernel, three separate convs rather than one fused, and the
   gated output RMSNorm. The recurrent state is per-layer and constant-size,
   which the hybrid KV accounting must know about.
5. **Tokenizer.** 157k vocab from `tokenizer.json`. The runtime selects a
   pre-tokenizer by GGUF string (`:7031`); HF needs the regex read from the
   tokenizer JSON instead. `chat_template.jinja` renders through the existing
   path.

**Gate:** per-layer activation parity against the torch reference on CPU -- one
layer at a time, KDA and MLA separately, before any end-to-end text.

### Parity harness -- DONE

`native/tools/bailing_reference.py` runs the real checkpoint on CPU and dumps
per-layer activations. It works, and it predicts " Paris" for "The capital of
France is", which is the end-to-end check that the substitutions below did not
change what the model does.

Getting it running needed four fixes, all environment drift rather than model
problems, and all worth recording because the next person hits them too:

1. `is_torch_fx_available` was removed from transformers v5; the checkpoint's
   modeling code was written against 4.45 and imports it. Shimmed rather than
   downgrading the library -- downgrading would silently change the reference
   this whole exercise is measured against.
2. `config.json` says `"rope_scaling": null`, but v5 fills the field with a
   default dict. The modeling code then takes its `rope_scaling is not None`
   branch and fails looking for a "factor". Forced back to None, which matters
   beyond loading: that branch also multiplies the attention scale by an mscale
   the model was never trained with.
3. v5 dropped the `"default"` entry from `ROPE_INIT_FUNCTIONS`. Re-registered
   with the plain unscaled formula -- deliberately the same one
   `partial_rope_norm` was verified against, so the harness and the code under
   test cannot be fed different rope definitions.
4. The CPU `ShortConvolution` substitute inherited `torch.nn.Conv1d`'s
   `bias=True`, but the checkpoint carries only `*_conv1d.weight`. Three biases
   per KDA layer were silently random-initialized, and the only visible sign was
   a MISSING line in transformers' load report. Every KDA layer produced NaN.

The Triton kernels cannot run here (PyTorch does not support this GPU's
sm_120), so `ShortConvolution`, `FusedRMSNormGated` and `chunk_kda` are replaced
with pure-PyTorch equivalents. The first two are transcriptions of small
unambiguous operations; the third is the one carrying real risk and is exactly
the one already pinned -- `kda_reference_check.py` shows fla's naive forms agree
with our oracle to ~1e-5.

### RESOLVED: the reference needs transformers 4.57.x

The corruption below was a **version mismatch**, confirmed by fixing it. In a
clean CPU-only venv with `transformers==4.57.1`, `torch 2.13.0+cpu`, `fla 0.5.2`
and `triton` (which fla imports even on CPU), five consecutive runs produce
byte-identical logits and no NaN.

Do not guess the version from `config.json`: it says `transformers_version
4.45.0`, which is stale. The modeling code imports `dynamic_rope_update` (added
~4.51), `DynamicLayer` and `TransformersKwargs` (added ~4.54), while also
importing `is_torch_fx_available` and `is_torch_greater_or_equal_than_1_13`
(both removed in v5). 4.57.1 satisfies all six, and makes the three
compatibility shims in the harness no-ops -- they are kept only so it still
starts on a newer stack.

**MLA parity, real layer-3 weights, clean reference:**

| variant | per-token max abs diff |
|---|---|
| causal | 1.4e-01, 2.6e-02, 4.0e-02, 1.1e-02, **5.9e-07** |
| non-causal | 1.0e-06, 5.2e-07, 1.6e-07, 5.1e-07, 5.9e-07 |

The reference runs **non-causal** -- transformers hands the checkpoint's
`eager_attention_forward` no mask, and that function only masks when given one.
This persists in 4.57.1, so it is the checkpoint's own behaviour under a modern
transformers, not a version artifact. Two consequences:

* The last token is exact parity either way (it attends to everything under
  both), which is why it matches at 5.9e-07 in the causal column.
* Running our implementation non-causally reproduces the reference on **every**
  token at float32 noise. That turns "the difference is the mask" from an
  assumption into a measurement, and confirms `mla_step` is correct end to end
  on real weights: both RMS norms, the low-rank query path, the compressed KV
  path, interleaved partial RoPE, the absorbed attention, the head gate and the
  output projection.

**KDA parity, real layer-0 weights, clean reference.** KDA is causal by
construction, so every token is a valid comparison; all five match, worst
`max|diff|` 1.9e-05 against outputs with std ~0.013.

That is coarser than the MLA result, so it was checked rather than accepted:
the error is unbiased (mean signed diff 7.5e-09 against a spread of 1.0e-06,
correlation 0.9999999975), which points at rounding rather than a defect. To
settle it, the layer was recomputed independently in numpy float64 and both
implementations measured against that:

| | relative error vs float64 truth |
|---|---|
| this implementation (f32) | **9.8e-07** |
| torch reference (f32) | 8.1e-05 |

So the gap is the *reference's* rounding, not ours -- we land ~83x closer to the
exact answer, because `rms_norm` here accumulates in double and fla's naive
recurrence does not. Worth remembering the next time a parity number looks
mediocre: "differs from the reference" and "wrong" are not the same claim, and
which side is wrong is a question with an answer.

Both attention paths are now verified end to end on real weights:
`mla_step` (both RMS norms, low-rank query, compressed KV, interleaved partial
RoPE, absorbed attention, head gate, output projection) and `kda_step` (three
convolved projections with SiLU, the per-channel gate, the beta gate, the
recurrence, the gated output norm, output projection).

The original diagnosis is kept below because the reasoning that isolated it --
deterministic within a process, varying across processes -- is what pointed at
the environment rather than at the model.

### Original diagnosis: the reference stack is not reproducible on this machine

The harness cannot yet serve as an oracle. Evidence, in the order it was
gathered:

* Repeated runs of the same prompt give **different logits**, and roughly one
  run in three produces NaN. The top token is " Paris" whenever the run is
  finite, so the corruption is partial rather than total.
* **Within** a single process, five consecutive forward passes are
  bit-identical. The variation is strictly across processes.
* The full state dict -- all 9283 parameters -- hashes **identically** across
  processes (`705682a975fa3515161d`). The weights are not the problem, and
  nothing is reported MISSING once the conv-bias fix is in.
* torch's own matmuls are stable in isolation: 0/200 divergences on a
  4096x512 @ 512x1536, and 0/20000 non-finite on the exact 32x1 @ 1x5 shape the
  rotary embedding uses.
* Not memory pressure: bf16 halves the footprint and behaves the same way
  (finite / NaN / finite, with the two finite runs disagreeing).
* Not threading: single-threaded (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`)
  behaves the same way.

Deterministic within a process, varying across processes, with identical inputs
and weights, is the signature of a **read from uninitialized memory** in an
activation buffer: the allocator hands back the same stale block within a
process, and different garbage in a new one. It is somewhere in the modeling
code or fla, not in the weights or in torch's kernels.

Worth noting what this does NOT cast doubt on. Before the corruption was
understood, the reference's internals were captured mid-forward and compared
against our implementation on real layer-3 weights:

| tensor | max abs diff |
|---|---|
| `v` | 7.7e-07 |
| `q_nope` | 3.0e-06 |
| `k_nope` | 1.8e-06 |

Those are the projections, both RMS norms, and the `kv_b_proj` decompression,
all matching at f32 noise on real weights. The rope halves could not be compared
because the reference's own `q_rope`/`k_rope` were the NaN carriers in that run.

Also found while chasing this: the reference runs with **`attention_mask=None`**,
because transformers v5 returns no mask here and the checkpoint's
`eager_attention_forward` skips masking when the mask is None. The reference is
therefore computing NON-CAUSAL attention. Any parity comparison must either
force a causal mask or compare against non-causal, or it will disagree for
reasons that have nothing to do with the runtime.

Next step is the user's call, since it is their environment: pin transformers
and fla to versions contemporary with the checkpoint (4.45-era) in a separate
venv, which also removes the three shims above; or run the reference elsewhere.
The shims exist because the installed stack is years newer than the modeling
code, and that mismatch is the most likely source of the bad buffer.

### Ling 3.0 Flash -- DONE. Loads, generates, serves.

`Ling-3.0-flash-IQ2_XXS.gguf` (43 blocks, 512 experts top-8 in 8 groups
keeping 4, 262k context) is the same architecture with four differences, all of
which were silent-or-fatal rather than merely new:

* **No query LoRA.** `q_lora_rank: null`, so an MLA layer carries an
  un-factored `attn_q` (hidden -> heads*qk) instead of q_a/q_a_norm/q_b. The
  layer plan, both host paths and both device paths now branch on
  `Geometry::q_lora == 0`. On the HF side `attention.q_proj.weight` had to come
  out of the flat rename table: it means the KDA query on a linear layer and
  the MLA query on a full-attention one, the same ambiguity `g_proj` already
  had.
* **IQ4_NL (GGML type 20).** The quantizer falls back to it for rows that are
  not a multiple of 256, which here is `attn_k_b` at 128 elements a row. Only
  the load-time decode needs it -- k_b/v_b are merged into an f32 `attn_kv_b`
  and dropped -- so it is a decoder, not a kernel. Verified bit-identical
  against the `gguf` package's own `IQ4_NL.dequantize_blocks` on the real
  tensor, through the per-head un-transpose.
* **Per-layer SwiGLU clamps.** `swiglu_clamp_exp` is 4.0 on blocks 35-41 and
  `swiglu_clamp_shexp` is 5.0/7.0 on 34-41. Both arrays were parsed and then
  dropped at the Geometry boundary (`g.swiglu_limit = 0.0f`, hardcoded). They
  are per LAYER and differ between the routed and the shared half, so they now
  live on `LayerWeights`; `Geometry::swiglu_limit` is gone rather than left as
  a field nothing reads. The three fused device expert kernels took a `limit`
  argument for the same reason.
* **An MTP draft block that is present.** `block_count` is 43 and blk.42 is the
  nextn block, written entirely at Q4_0. `detect_mtp_layer` already trimmed
  `layer_count` to 42, but the device-support scan walked every tensor in the
  file, so one unexecuted Q4_0 tensor vetoed the GPU for the whole model.

`kda.safe_gate`/`kda.gate_lower_bound` are carried in the file and still
deliberately unimplemented -- see the KDA oracle note above: fla 0.4.1 discards
both, so the reference model's observable behaviour has no clamp.

Not addressed: at 35 GB the model does not fit a 12 GB card, and the bailing
device path is all-or-nothing about residency, so it runs on the host
(9.1 tok/s decode measured, which is the memory wall for a file this size).
Partial residency for bailing is its own piece of work and has never existed
for any checkpoint of this family.

### Host decode -- 13.5 -> 15.3 tok/s, and the profiler that was lying

The first thing found was not a slow kernel, it was a blind instrument.
`decoder_layer_batch` delegates to `decoder_layer` at `tokens == 1`, and the
single-token path carried NO ProfileScopes at all -- nor did `moe_block`. So
`COLIBRI_BAILING_PROFILE=1` reported prefill and silently omitted every decode
token, while presenting its buckets as percentages of the whole. The lm head
and the embedding gather sat outside every bucket on both paths as well, so
even the prefill numbers were shares of about half the work.

Instrumenting decode closed the accounting to 66.7 ms accounted against
66.1 ms wall. What it shows, per token, against a measured 52 GB/s DRAM read
wall (and note that ONE core reaches 52 GB/s on this part -- threads buy 12%,
not 16x):

| phase | ms | bytes | GB/s | vs wall |
|---|---|---|---|---|
| KDA proj + conv + recurrence | 26.2 | 1.33 GB | 51 | 98% |
| routed expert matmuls | 20.6 | 0.49 GB | 24 | 46% |
| lm head | 4.5 | 0.28 GB | 62 | at wall |
| MLA | 4.3 | -- | -- | -- |
| shared expert | 4.2 | 0.19 GB | 46 | 92% |
| routing | 4.0 | 0.21 GB | 51 | 98% |

Everything is at the memory wall except the routed experts. The experts are
the one phase that is COMPUTE-bound rather than bandwidth-bound: a single core
sustains only 3.1 GB/s on IQ2_XXS, because the cost is the grid lookup. So
they need every core -- and calling `matvec` once per expert per projection
fired 24 OpenMP regions a layer, 960 a token, each over ~0.5 MB.

Fixed by giving the CALLER the parallel region: `RowKernel`/`row_kernel()`
resolve the ISA/type dispatch once, and `moe_block` now runs all the chosen
experts in TWO regions -- one over (gate|up) x slot x row, one over output rows
with the expert loop inside, so each thread owns its rows and no reduction is
needed (the shape the device's `bailing_q6_expert_accumulate_rows` already
used). Measured 29.8 -> 40.5 GB/s in isolation; the expert bucket fell 26% and
decode went 13.5 -> 15.3 tok/s. Logits are unchanged -- the GGUF/HF parity test
is what says so.

Measured and rejected, so they are not tried again:

* **An AVX-512 IQ2_XXS kernel.** Type 16 is absent from the AVX-512 allowlist
  and this part has full-width AVX-512, which looks like an obvious gap. It is
  not: IQ2_XS *on AVX-512* manages 3.4 GB/s/core against IQ2_XXS *on AVX2* at
  3.1. SIMD width does not fix a grid lookup. (For scale, Q4_K reaches 14.4 and
  Q6_K 15.3 GB/s/core -- the IQ2 family is ~4x more expensive per core.)
* **More threads.** 16 (one per physical core) is right; 24 is slightly worse
  and 32 -- the SMT siblings -- collapses decode to 6.2 tok/s.
* **Cold page faults / huge pages.** The file reads back at 15 GB/s, already
  resident, and the loader already calls `madvise(MADV_HUGEPAGE)`.
* **The scattered access pattern.** 8 experts drawn at random from 512 costs
  ~8% against 8 contiguous, not the 1.7x that would explain the gap.

The experts are still at ~46% of the wall in situ against ~80% in isolation,
and that residue is NOT explained. Repeated attempts to isolate it gave
inconsistent answers (28-48 GB/s depending on how the benchmark was warmed),
so no further claim is made about it and no fix is proposed on a guess.

## Sequencing

Stages 1 and 2 are useful on their own and de-risk stage 3 (a bug in stage 3 is
otherwise indistinguishable from a loader bug). Stage 3 is the long pole and is
where the schedule will actually go.

Explicitly out of scope: MTP/nextn *execution* (the flash checkpoints ship a
draft block; it is skipped, not run), vision, DFlash drafting, IQ-type
quantization at load.
