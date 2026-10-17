# Nemotron H MoE (Nemotron 3.5 Bolt, `nemotron_h_moe`)

> **STATUS 2026-10-17**: runs natively on branch `feat/nemotron-h-moe`. Target
> checkpoint `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf` (22 GB). Decode
> and batched prefill match an independent float64 forward on the fixture;
> tokenization matches llama.cpp token for token; the real checkpoint answers
> coherently, calls tools through its own template, and retrieves a fact
> planted 6.9K tokens into a 13.4K prompt. Routed experts run on the CPU only.
> Semantics reference: llama.cpp `src/models/nemotron-h.cpp` and
> `src/models/mamba-base.cpp` (`build_mamba2_layer`).

## What the checkpoint is

A Mamba-2 / attention / MoE hybrid. 88 blocks, each one RMS pre-norm
(`attn_norm`) and exactly one sublayer, classified by tensor presence:

| Block | count | marker |
| --- | --- | --- |
| Mamba-2 | 40 | `ssm_in.weight` |
| GQA attention | 8 | `attn_q.weight` (`head_count_kv` = 2 there, 0 elsewhere) |
| MoE | 40 | `ffn_gate_inp.weight` |

Geometry: hidden 4096, d_inner 8192, d_state 128, 128 scan heads
(`ssm.time_step_rank`, 64 channels each), 8 groups, 4 conv taps. Attention: 32
heads x 128, 2 KV heads, **no RoPE**, 1/sqrt(128) scale. MoE: latent width
`moe_latent_size` 1024; 240..496 routed experts per block, FF width
1536..2560 per block, top-9 everywhere; sigmoid routing with `exp_probs_b`
affecting selection only, `expert_weights_norm`, `expert_weights_scale` 5;
relu² experts with **no gate tensor**; one always-on shared expert (5376 wide,
relu², full width, unweighted).

Metadata quirks the loader has to absorb:

- `expert_count` is a scalar (the maximum, 496). A block's count is its router
  width; `expert_feed_forward_length` and `expert_used_count` are per-layer
  arrays. Uniform conversions store scalars, which the plan accepts too.
- `ssm_in` / `ssm_out` are NVFP4 (type 40) with an out-of-line f32 `.scale`
  each (`weight_scale_2`), which the existing NVFP4 loader folds in.
  Attention, latent projections, embeddings and head are Q8_0; experts and
  shared expert IQ2_XXS; norms, conv and the SSM vectors f32.
- `tokenizer.ggml.pre` is `pixtral`, which llama.cpp gives the llama3 regex
  with `ignore_merges` (our `lfm2` arm). `<think>`, `<tool_call>`,
  `<tool_response>` and their closers are user-defined (type 4) tokens that
  must split out whole. Without the pre-tokenizer, indented code reached the
  model as `"\n" "    " "a"` instead of `"\n" "   " " a"`.
- No MTP tensors in this file.

## Math

**Mamba-2**: `ssm_in` gives `z | xBC | dt` (`2*d_inner + 2*n_group*d_state +
n_head`). Depthwise causal conv over xBC only (taps `{d_conv, channels}`,
ring of the previous `d_conv-1` raw values, oldest first), `+ conv1d.bias`,
silu. `dt_f = softplus(dt + ssm_dt.bias)` (passthrough above 20),
`dA = exp(dt_f * ssm_a)` (`ssm_a` is stored already negated);
`state[h,i1,:] = state*dA + B[g] * x[h,i1] * dt_f`, `g = h / (n_head/n_group)`;
`y = state·C[g] + x*D`, then `y *= silu(z)` (z times sigmoid z), grouped RMS
over `n_group` chunks of `d_inner/n_group` with `ssm_norm`, `ssm_out`.
State per slot and layer: conv `3 x 10240` f32, scan `128 x 64 x 128` f32.

**Attention**: plain GQA into the KV cache, no RoPE, no head norms, no gate.

**MoE**: router on the normed stream; `sigmoid`, select on `p + exp_probs_b`,
weight with the unbiased `p`, normalize, times 5. Routed experts run in the
latent space: `latent = ffn_latent_down · h`, per expert
`down_e(relu²(up_e(latent)))`, weighted sum, `ffn_latent_up` back to hidden.
Plus `down_shexp(relu²(up_shexp(h)))`.

## Runtime shape

- `build_nemotron_plan` beside `build_lfm2_plan`; `QwenLayerPlan` carries
  `mamba`, named SSM roles (the conv bias is optional), and per-block expert
  count / FF width / top-k / latent pair / shared pair.
- Kernels in `flyweight_v2_native_kernels.hpp`: `nemotron_ssm_conv_step/_rows`,
  `nemotron_ssm_step` / `nemotron_ssm_scan_rows`, `nemotron_ssm_norm/_rows`,
  `nemotron_relu2_mul/_rows`. The rows twins walk the chunk in order, so a
  chunk leaves exactly the state token-by-token decode would.
- Recurrent state rides `qwen_recurrent_state`; MoE-only blocks reserve and
  snapshot nothing. Prefix reuse restores Mamba state from checkpoints like
  DeltaNet's.
- Decode: the shared expert and latent projections run on the GPU while
  `nemotron_cpu_moe` computes the routed experts from the mapping.
- Prefill (`v2_mtp_verifier.inc`): Mamba rows, attention through the one-launch
  warp prefill kernel (per-row decode dispatch for <= 8 rows), and
  `nemotron_cpu_moe_rows`, which groups routes by expert so each expert's
  weights are decoded once per chunk. `queue_combine` projects the latent sum
  back up. Works under the two-half pipelined driver.
- Expert placement: `auto` keeps experts on the CPU with a note; `resident`
  refuses. MTP, lookup drafts and multi-slot batched decode are off.

## Numbers (5070 Ti Laptop 12 GB, 9955HX, 2026-10-17)

GPU holds 4.7 GB of weights; 16.3 GB of experts stay mapped in RAM.

| | |
| --- | --- |
| decode | 30-32 tok/s |
| prefill, 2755-token prompt | 50 -> 207 tok/s (expert grouping) |

Prefill is now all CPU expert compute (13.0 of 13.3 s). The up projection takes
the int16 VNNI fold (`FLYWEIGHT_ROWS_I16=0` holds it off); the down projection
stays f32. int16 down measured ~12% faster but moved the greedy output where
int16 up alone reproduced the f32 tokens exactly: relu² outputs have squared
tails a 14-bit block scale flattens.

Decode spends ~20 ms/token in the CPU experts, ~420 MB of IQ2_XXS read at
~20 GB/s: the AVX2 float IQ2_XXS dot is compute-bound, not DRAM-bound.

## Next levers

1. Prefill: stream experts to the GPU with relu² grouped kernels (a gateless
   twin of the routed MMQ path). A 2K chunk touches nearly every expert, so
   ~16 GB over 7 GB/s PCIe is still several times faster than the CPU GEMMs.
2. Decode: an IQ2_XXS Q8_K VNNI dot (as IQ1_S and IQ3_S have), or a GPU
   expert cache once the relu² kernels exist.
3. The Bolt MTP head, if NVIDIA ships one as GGUF.

## Tests

`tests/nemotron_gguf_fixture.py` writes a toy checkpoint (Mamba, attention and
three heterogeneous MoE blocks, one without the latent bridge) and carries an
independent float64 forward. `tests/test_v2_nemotron.py` pins decode,
4-row and pipelined prefill, prefix reuse, reset and the CPU backend against
that forward, plus the tokenizer and the tool-call markers.
