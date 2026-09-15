# Z-Image-Turbo image generation

Started 2026-09-14. The first diffusion model on the engine, chosen because
every component is already the engine's shape: the text encoder is a plain
Qwen3-4B, the DiT is one block type (single-stream, RMSNorm, SwiGLU, qk-norm,
rope), and the VAE is the FLUX 16-channel `AutoencoderKL` that FLUX and
Qwen-Image also use. Eight distilled steps, no CFG.

## Reference

diffusers 0.36 `pipeline_z_image.py` / `transformer_z_image.py`. Facts the
port depends on, all verified against a dump (`tools/zimage_reference.py`):

- Conditioning is `hidden_states[-2]` of Qwen3-4B on the chat-templated
  prompt (`<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n`), so the
  encoder runs 35 of 36 layers and no final norm.
- Caption rows pad to a multiple of 32 by repeating the last row, then the
  padding rows are replaced by `cap_pad_token` after `cap_embedder`; image
  rows likewise with `x_pad_token`. Padding rows are attended, not masked.
- Positions: caption `(1 + i, 0, 0)`, image `(cap_padded + 1, h, w)`, pads
  at the origin. Rope is complex-pair (adjacent elements) per axis, ramps
  built in float64.
- Modulation: `adaLN(t_emb)` -> `[scale_msa, gate_msa, scale_mlp, gate_mlp]`,
  scales used as `1 + s`, gates `tanh`. Blocks: `x += gate * norm2(f(norm1(x) * scale))`.
- The model predicts the negative velocity; `x_next = x - (sigma_next - sigma) * out`.
  Sigmas: `linspace(1, 0, steps)` shifted by `3 s / (1 + 2 s)`, trailing 0,
  the zero-width last step skipped. Model time is `1 - sigma`, scaled by 1000
  into the 256-wide cos/sin embedding.
- VAE: `z / 0.3611 + 0.1159`, decoder in f32, output `(x / 2 + 0.5)` rounded.

## Loader

`hf.hpp` recognises three more `config.json` shapes: `Qwen3ForCausalLM`
(architecture `qwen3`, name tables shared with Qwen3.5's full-attention
layers, tokenizer read from the sibling `tokenizer/`), `_class_name ==
ZImageTransformer2DModel` (`zimage-dit`) and `AutoencoderKL`
(`autoencoder-kl`). Diffusion components skip the tokenizer. The DiT's
`to_q/to_k/to_v` are stacked into one `attention.qkv.weight` through the
expert-stack path (`expert_count = 3`), so a block's projection is one GEMM
into the fused layout the attention kernel reads. `[1][dim]` tensors stay
f32. Default policies: DiT Q8_0, VAE F32, encoder the loader default.

## Runtime

`native/src/v2_diffusion.inc`, kernels in
`flyweight_v2_diffusion_kernels.hpp`, appended to the one corpus every
Qwen-family runtime compiles (`qwen_cuda_corpus()`), so a chat model and the
image model share a module. Quantized GEMMs go through the format table:
MMQ with Q8 activations where the width allows, the per-format rows kernel
otherwise; f32 through the vision tiled GEMM. Attention is
`diff_attention_rows`, the vision kernel generalised to separate strided
q/k/v, grouped heads and a causal flag, so the same kernel serves the
encoder and the DiT. The VAE is direct 3x3 convolution (32x32 output tile,
2x2 pixels x 4 channels per thread), two-pass group norm in double, and the
mid-block attention as chunked score GEMM + row softmax + GEMM.

One workspace arena, sized at create for the largest phase at
`--image-max-size`. With `weights=host` (the auto choice on a card the
weights would half fill) the encoder's and DiT's layer matrices are pinned
in host memory and `DiffStreamer` stages layer i+1 into the other of two
device slots on a copy stream while layer i runs, with staged/consumed
events pairing the slots; norms, embedders and the VAE stay resident. The
whole DiT (refiners then the stack) is one run list so the stream never
bubbles. Measured: identical output, +0.4 s per 512x512 image, 1.4 GB of
VRAM instead of 10 GB, and Qwen3.6-35B fits beside it on 12 GB. `FLYWEIGHT_DIFF_SYNC=1` syncs after every launch and
names a faulting kernel; `FLYWEIGHT_DIFF_TRACE=1` prints launches (with
times when syncing).

## After the first round (2026-09-15)

- Text encoder default Q8_0: caption 0.87% rms vs the bf16 reference
  (Q6_K: 2.1%); the DiT step with the native caption went from 15% to 7%
  at 1024.
- Tiled VAE decode (64-latent tiles, stride 48, 128-pixel blends), so 1024
  fits; peak activation stays the 512 figure.
- `diff_flash_attention_bf16`: bf16 mma.sync flash attention, 64 queries
  per block, f32 accumulate. 8.7 ms vs 234 ms per layer at 4160 rows,
  0.3% rms from the f32 kernel. `FLYWEIGHT_DIFF_FLASH=0` restores f32.
- `diff_conv2d_bf16`: implicit-GEMM conv on mma.sync, channels-last bf16
  input written by the group norm / upsample in fragment slot order,
  weights packed [tap][out][in/16][16]. 2.6-9 TFLOPS depending on shape
  (load-bound; a shared-memory staged version is the next step). Decode at
  1024: 8.7 s -> 2.8 s.
- MMQ token tiles batched into grid.y (the LLM prefill path launches with
  grid.y = 1 and is unchanged): 2656 -> 86 launches per step.
- Against an f32 step-0 reference at 1024: native 6.95% rms, diffusers
  bf16 5.14%, error maps correlated 0.58 -- the residual is bf16-class
  noise, not a defect.
- Per 1024x1024 image with the 35B chat model resident: 14.8 s. Per step
  ~1.65 s: MMQ ~1 s (47 TFLOP at ~45 TOPS), attention 0.15 s, the rest
  elementwise. 512x512: 3.5 s.

## Decoder fixes (2026-09-15)

The 2.8 s tiled decode at 1024 was not the convolutions: the group-norm
apply kernel re-summed 64 double partials per element and wrote its
channels-last output uncoalesced, 3.6 s of a 4.7 s synced trace. A finalize
kernel folds the partials once per group and the apply and the layout
conversion go through a 32x32 shared tile. Decode at 1024: 2.8 s -> 0.75 s;
at 512: 0.44 s -> 0.10 s. The bf16 conv measures 44-48 TFLOPS once the
harness stops timing its own reference einsums (the earlier 2.6-9 TFLOPS
figures were that artifact); the shared-memory staging with register
prefetch stays, it is no slower.

## Balanced precision (2026-09-15)

`diff_q8_bf16_gemm`: a 128x64-tile bf16 mma GEMM that stages Q8_0 blocks as
bf16 (28 TFLOPS on 3840->11520 at 4160 rows, 16 on 10240->3840) with the
activations packed to bf16 by `diff_pack_rows_bf16`. Step error vs f32 at
1024: fast 6.6%, balanced 3.7%, exact 3.3%, diffusers bf16 5.1%; balanced
costs the same wall time as fast, so it is the default. Remaining error in
balanced is the Q8_0 weights plus bf16 rounding of activations; exact
removes the latter for 8x the time.

## Precision switch (2026-09-15)

`--image-precision exact` (`FLYWEIGHT_V2_DIFFUSION_EXACT` in the create
flags) keeps every fast path off: rows kernels with f32 activations instead
of MMQ, `diff_attention_rows` instead of the bf16 flash kernel, the f32
conv instead of the bf16 implicit GEMM. At 1024 against the f32 step-0
reference: fast 6.6%, exact 3.3%, diffusers bf16 5.1%. Exact is ~16 s per
step. What remains in exact mode is the Q8_0 weights; a bf16 pack target
in the loader would remove that at 2x the host memory.

## Parity (2026-09-14, 512x512, RTX 5070 Ti laptop)

Against the bf16 diffusers run (`tools/check_zimage_parity.py`):

| stage | rms vs reference | note |
|---|---|---|
| encoder hidden_states[-2] | 2.1% (cos 0.99987) | bf16-vs-f32 torch alone is 1.3% |
| DiT step 0 | 2.7% (cos 0.99964) | Q8 weights and activations |
| VAE decode of reference latents | 0.55% (cos 0.99999) | |
| full pipeline from reference noise | same picture, 22% pixel rms | texture drift over 8 steps |

Per step: MMQ 188 ms, attention 163 ms, quantize 12 ms; VAE 1 s; 6.75 s in all.

## Next

- The Q8 GEMMs: cuBLASLt int8 needs per-channel scales (a requant of the
  DiT away from Q8_0 blocks), or an MMQ variant with cp.async pipelining.
- Fusions: pack the GEMM input to bf16 inside the preceding norm, and fold
  SiLU x up into the w1/w3 GEMM; ~10% of a step.
- Encoder at Q4_K by default would save 0.8 GB of host memory for a
  quality cost nobody has measured yet (moot for VRAM in host mode).
- Plain Qwen3 dense GGUFs (Qwen3-1.7B etc.) have never run on the chat
  runtime: the qwen plan wants `post_attention_norm` and a gated query.
  Unrelated to images, but it is the first small chat model people reach
  for beside them.
