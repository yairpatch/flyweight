# Nemotron H MoE (Nemotron 3.5 Bolt, `nemotron_h_moe`)

> **STATUS 2026-10-18**: merged (#140, #142, #143). Target checkpoint
> `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf` (22 GB). Decode and batched
> prefill match an independent float64 forward on the f32 fixture and on a
> quantized one in the released formats; tokenization matches llama.cpp token
> for token. Teacher-forced against upstream llama.cpp (cd0a5b6, the first
> build that loads Bolt's per-block router widths) the real checkpoint agrees
> on 98.6-99.2% of greedy steps in every placement mode, every miss a top-2
> swap within 0.41 nats. Routed experts run on the CPU, the GPU expert cache,
> whole GPU layers (`-ngl`) or fully resident. Semantics reference: llama.cpp
> `src/models/nemotron-h.cpp` and `src/models/mamba-base.cpp`
> (`build_mamba2_layer`).

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
  the routed experts run through the expert cache (`qwen_run_hybrid_experts`
  with the gateless `<type>_grouped_relu2` / accumulate kernels) or, for
  misses and in `cpu` mode, `nemotron_cpu_moe` from the mapping. On
  AVX512-VNNI the CPU up projection takes the latent stream as Q8_K (the
  IQ2_XXS VNNI dot); the down projection stays float -- see below.
- Prefill (`v2_mtp_verifier.inc`): Mamba rows, attention through the one-launch
  warp prefill kernel (per-row decode dispatch for <= 8 rows), and
  `nemotron_cpu_moe_rows`, which groups routes by expert so each expert's
  weights are decoded once per chunk. `queue_combine` projects the latent sum
  back up. Works under the two-half pipelined driver.
- Expert placement: every mode. The cache is sliced over the MoE blocks only
  (the Mamba-2 and attention blocks route nothing), `-ngl N` pins the last N
  MoE blocks at their own expert counts, `resident` packs every block's
  experts. NVFP4 experts stay on the CPU (no relu² kernel). Grouped routing,
  per-expert `ffn_*_exps.scale` and attention biases are refused at load.
  MTP, lookup drafts and multi-slot batched decode are off.

## Numbers (5070 Ti Laptop 12 GB, 9955HX, 2026-10-18)

GPU holds 4.7 GB of weights; 16.3 GB of experts stay mapped in RAM.

| | flyweight | llama.cpp cd0a5b6 (experts on CPU) |
| --- | --- | --- |
| decode, `cpu` | ~41 tok/s (39 all-float, 45 with int8 down) | 41-46 tok/s |
| decode, `auto` | 46-48 tok/s | |
| prefill, 2117-token prompt | ~350 tok/s | 422 tok/s |

The 2026-10-17 numbers (30-32 decode, 207 prefill) predate the GPU expert
cache and the VNNI dot. Prefill is still dominated by the CPU expert GEMMs.

**Activation precision.** The prefill up projection takes the int16 VNNI fold
(`FLYWEIGHT_ROWS_I16=0` holds it off), decode's the Q8_K VNNI dot
(`FLYWEIGHT_IQ1S_Q8=0` holds it off). The down projection stays f32 in both
(`FLYWEIGHT_NEMOTRON_Q8_DOWN=1` opts decode into Q8_K). bench_kernel_quality on
Bolt, 7 prompts x 256 tokens against all-float decode: Q8_K up agrees 99.6%
teacher-forced, inside the 99.3% of the float GPU kernels (another exact-
precision path), median first greedy divergence ~token 98; adding Q8_K down
pulls that divergence to token 12 for ~8% decode. In prefill, int16 down
measured ~12% faster but moved the greedy output where int16 up alone
reproduced the f32 tokens exactly: relu² outputs have squared tails a narrow
block scale flattens.

The dense NVFP4 projections (`ssm_in`, `ssm_out`) take Q8 activations at decode
and in batches of up to 8 rows (`nvfp4_q8_matvec_transposed_{warp,rows}`), the
way llama.cpp runs NVFP4 against Q8_1. `FLYWEIGHT_NVFP4_Q8=0` restores the
float-activation warp kernels. Same 7x256 A/B against float: 99.4% teacher-forced,
level with the float GPU kernels' 99.4%, 3/7 continuations moved against their
4/7. Against llama.cpp, teacher-forced: 98.99% against float's 99.08% (one token
in 1086, a top-2 swap at margin 0.41). A 2-row verify drops from 60 to 44 ms in
`cpu` mode and from 38 to 34 ms in `auto`.

## Next levers

1. Prefill: stream experts to the GPU through the relu² grouped kernels (the
   prefill stream arena is Qwen's 3-role SwiGLU only). A 2K chunk touches
   nearly every expert, so ~16 GB over 7 GB/s PCIe is still several times
   faster than the CPU GEMMs; llama.cpp is 1.2x ahead here.
2. Vision: upstream runs Bolt as a VL model (`--mmproj`, RADIO encoder,
   images and video); flyweight has no Nemotron vision path.
3. The MTP draft layer: llama.cpp loads it (`nextn`), this GGUF omits it.

## Tests

`tests/nemotron_gguf_fixture.py` writes a toy checkpoint (Mamba, attention and
three heterogeneous MoE blocks, one without the latent bridge) and carries an
independent float64 forward; `NemotronSpec(quantized=True)` stores it as Bolt
does (IQ2_XXS experts, NVFP4 + `.scale` ssm projections, Q8_0 elsewhere) from
random well-formed codes the reference decodes. `tests/test_v2_nemotron.py`
pins decode, chunked and pipelined prefill, every placement mode, `-ngl`,
`resident`, prefix reuse, reset and the CPU backend against that forward, plus
the tokenizer, tool-call markers, refused variants and the epsilon fallback.
The quantized runs switch the int8 activation paths and the FP4 tensor-core
prefill off (in a subprocess: both are read once per process) -- on a random
model their rounding flips steps whose logit lead or 1e-3-wide routing gap is
narrow, which is a quality question for the bench, not a parity one.
