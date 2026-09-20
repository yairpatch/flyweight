# Qwen-Image-2.1 text-to-image

> **STATUS 2026-09-20**: text-to-image WORKING, parity below. Editing with
> condition images is deliberately out of scope for this pass (see "Not in
> this pass").

Started 2026-09-20. The second diffusion model on the engine. Z-Image-Turbo
(`plans/zimage-diffusion.md`) established the tower — component models opened
from a diffusers snapshot, one workspace arena, host-streamed layer weights,
kernels launched by name — and most of that scaffolding carries over
unchanged. What is new is one model's worth of differences, listed below.

## Reference

diffusers `main` (`0.37.0.dev0` in the checkpoint's `model_index.json`):
`pipelines/qwenimage21/pipeline_qwenimage21.py`,
`models/transformers/transformer_qwenimage21.py`,
`models/autoencoders/autoencoder_kl_qwenimage21.py`. Copies under
`build/qwen21_refs/` (gitignored). `tools/qwenimage21_reference.py` dumps,
`tools/check_qwenimage21_parity.py` compares.

Four components instead of three: `text_encoder` (Qwen3-VL-8B),
`transformer` (7B, 32 single-stream layers), `vae`
(`AutoencoderKLQwenImage21`), and `processor/` where Z-Image had
`tokenizer/`.

## What is the same as Z-Image

The DiT block is the same shape: tanh-gated residuals over `1 + scale`
modulation, fused qkv, per-head q/k RMSNorm, complex-pair rope on adjacent
elements over three axes, SwiGLU. `diff_modulation_prepare`'s
`[scale, gate, scale, gate]` layout is exactly `modulation.chunk(2)` then
`chunk(2)` again, so it is reused as is. The timestep embedding is the same
256-wide `[cos, sin]` over `exp(-log(10000) i / 128)` scaled by 1000. The
rope ramp is the same `theta^(-2j/axis_dim)` built in float64. The VAE's
convolutions are plain 3x3 `padding=1` and its mid-block attention is one
head over `H*W`, so `diff_conv2d`, `diff_conv2d_bf16`, the chunked score
GEMM and the nearest-2x upsample all carry over.

## What is different

**Text encoder.** `Qwen3VLForConditionalGeneration`, 36 layers, hidden 4096,
GQA 32/8, head_dim 128, rope theta 5e6. For text-to-image the prompt carries
no image tokens, so mrope's three axes all hold the same position and it
degenerates to the plain rope the encoder path already runs — the vision
tower and `mrope_section` are not needed here. The tensors are the plain
Qwen3 block names under `model.language_model.layers.N.`, which is exactly
what `qwen3_5_layer_names()` already maps, and `model.visual.` is skipped the
way Qwen3.5's tower is.

Two behavioural differences from Z-Image's encoder:

- the conditioning is `hidden_states[-1]` with the final RMSNorm neutralized
  (the pipeline hooks `norm` to return its input), so **all 36 layers run**
  and no final norm is applied. Z-Image took `hidden_states[-2]` and skipped
  the last layer's weights entirely.
- the prompt is a raw template, not `apply_chat_template`:
  `<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n`,
  and the leading system-role tokens are then **dropped** from the hidden
  states (`_drop_idx`, derived by tokenizing the system message alone).

**DiT.** `mlp_ratio` 3 (12288), `in/out_channels` 64, `patch_size` 1 (so
"patchify" is a plain `[C][H][W] -> [H*W][C]` transpose), `axes_dims_rope`
`(16, 56, 56)`, theta 10000, `eps` 1e-6.

- `img_norm1`/`img_norm2` are **LayerNorm without affine**, not RMSNorm, and
  there is no second norm inside the residual: the block is
  `x += tanh(gate) * f(LN(x) * (1 + scale))` twice.
- modulation is **shared**: one `modulation.1` projection on `SiLU(temb)`
  feeds all 32 blocks, so blocks carry no adaLN weights of their own.
- `causal_condition`: text tokens modulate from `t = 0`, image tokens from
  the sampled `t`. Two modulation rows, selected per row range.
- attention is **block-causal**: `(q >= kv) or same_image_block`. For
  text-to-image the joint sequence is `[text][target image]`, so this is
  exactly "text rows causal over text, image rows full over everything" —
  two attention calls per block, which is the reference's
  `QwenImage21AttnProcessor` segment path with `use_kv_cache=False`.
- `txt_in` is `ZeroCenterRMSNorm -> Linear -> GELU(tanh) -> Linear`. The
  zero-centred norm stores `scale - 1`, so 1.0 is folded into the weight
  vector at upload and the existing `rms_norm_rows` runs unchanged.
- rope positions: text tokens advance on all three axes; the image block
  freezes the frame axis at the text length and lays out **centred** height
  and width ramps, `range(-(n - n//2), n//2)`. Negative positions are new —
  the reference reaches them through a wrapped lookup table, `sinf/cosf` of a
  negative angle is the same thing, so `diff_rope_axes_rows` needs no change.
- the final norm is scale-only: `LN(x) * (1 + Linear(SiLU(temb)))`.

**VAE.** `AutoencoderKLQwenImage21` is Wan's causal-3D autoencoder, but
`QwenImage21CausalConv3d` subclasses `nn.Conv2d` and folds the single frame
away, and at `T = 1` every temporal path is inert: `feat_cache` entries are
all `None` on the first (and only) chunk, so `upsample3d`'s `time_conv` is
skipped entirely and the residual blocks take their plain branch. The
decoder that actually runs is 2D. `decoder.*.time_conv` is therefore
recognised and dropped, the way the encoder half is.

- 64 latent channels, **16x** spatial (not 8x), so a latent row is a 16x16
  pixel tile and sides must be multiples of 32.
- **4 output channels**: RGBA. This is where "native transparency" comes
  from; the prompt prefix "This is an RGBA image with transparency" is what
  makes the model use the alpha.
- normalization is per-pixel **RMS over the channel axis** times `gamma`
  (`F.normalize(x, dim=1) * sqrt(C) * gamma`), not group norm. New kernel;
  it replaces `diff_group_norm_*` in this decoder.
- latents are denormalized per channel: `z * latents_std + latents_mean`,
  64 values each, not a scalar scale/shift.
- five levels, `decoder_base_dim` 144, `dim_mult [1,2,4,8,8]`, three resnets
  per block, and a parameterless `DupUp3D` shortcut added after each
  upsample: `out[oc, 2h+hs, 2w+ws] = in[(oc*factor + g) / repeats, h, w]`
  with `g = t_keep*4 + hs*2 + ws`. New kernel.
- the mid-block attention has a **fused** `to_qkv` 1x1 conv and an RMS norm.

**Scheduler.** `FlowMatchEulerDiscreteScheduler` with dynamic shifting, which
Z-Image's constant-shift `diff_sigmas` does not cover:

```
sigmas = linspace(1, 1/steps, steps)                    # note: not to 0
mu     = 0.5 + (0.9 - 0.5) * (seq_len - 256) / (8192 - 256)
sigmas = exp(mu) / (exp(mu) + (1/sigmas - 1))           # exponential shift
sigmas = 1 - (1 - sigmas) / ((1 - sigmas[-1]) / (1 - 0.02))   # shift_terminal
sigmas = [*sigmas, 0]
```

The model's time input is `sigma` itself and the step is
`x += (sigma_next - sigma) * out` — **not** negated, unlike Z-Image.
40 steps by default, `true_cfg_scale` 1.0 (no guidance).

## Shape of the port

`native/src/v2_qwenimage.inc`, included by `v2_runtime.cpp` after
`v2_diffusion.inc` so it reuses that file's `DiffMatrix`, `DiffLaunch`,
`DiffStreamer`, `DiffArena` and upload helpers from the same anonymous
namespace. The C API does not grow a second tower: the existing
`flyweight_v2_diffusion_*` entry points dispatch on the transformer's
architecture, so `--image-model` pointed at either snapshot does the right
thing and the server, the Python wrapper and the UI need no new concepts.

`FlyweightV2DiffusionInfo` grows the numbers the caller cannot guess:
`latent_stride` (8 or 16), `size_multiple` (16 or 32), `output_channels`
(3 or 4), `default_steps` and `default_shift` (0 meaning "the model's own
dynamic schedule").

Kernel changes shared with Z-Image: `diff_attention_rows`,
`diff_pack_attention_bf16` and `diff_flash_attention_bf16` split their single
`rows` into `plane_rows`, `q_first`, `q_rows` and `kv_rows` so one block can
run a causal segment over the prefix and a full segment over the image rows.
Existing call sites pass the same value for all of them, and
`tools/check_zimage_parity.py` is the check that they still do.

## Not in this pass

Image editing. It needs the Qwen3-VL **vision tower** (27 layers, hidden
1152, patch 16, deepstack at 8/16/24) to put condition images into the text
stream, the **VAE encoder** to put them into the latent stream, and
block-causal attention over more than two blocks. The transformer and
scheduler work here is a prerequisite for it, and the attention segment split
already generalises to more than one image block.

## Sizing

7B DiT and 8B encoder, Q8_0 at load: ~7 GB and ~8 GB, both host-streamed on a
12 GB card. The decoder is the memory problem: at 2048x2048 the level-3 block
holds 288 channels over 4M pixels, 4.8 GB in a plane and three planes live.
Tiled decode is therefore not optional the way it was for Z-Image — 32-latent
tiles (512 pixels out) at a stride of 24, blended over 128 pixels.

## Parity (2026-09-20, 512x512, 8 steps, balanced, RTX 5070 Ti laptop)

Against the bf16 diffusers run with `use_kv_cache=False`
(`tools/check_qwenimage21_parity.py`):

| stage | rms vs reference | note |
|---|---|---|
| encoder conditioning rows | 2.9% typical (cos 0.9996) | row 0 24.7%: a massive-activation token; costs 0.06% on the step below |
| DiT step 0, reference caption | 0.96% (cos 0.99996) | Q8_0 weights, bf16 activations |
| DiT step 0, own caption | 1.02% (cos 0.99995) | |
| VAE decode of reference latents | 0.98% (cos 0.99996) | |
| full pipeline from reference noise | 4.1% pixel rms (cos 0.9992) | same picture; Z-Image's figure here was 22% |

Timings, host-streamed weights beside nothing else, balanced:

| size | tokens | step | image | VRAM |
|---|---|---|---|---|
| 512x512 | 1 024 | 1.06 s | 10.4 s at 8 steps | 3.5 GB |
| 1024x1024 | 4 096 | 2.2 s | 51 s at 20 steps | 3.5 GB |
| 2048x2048 | 16 384 | 12.9 s | 284 s at 20 steps | 5.9 GB (arena grown to 3.9 GB) |

Encoder 1.2 s, decode 0.6 s at 512. `fast` matches balanced's time at 512
(the MMQ and bf16 GEMMs are both weight-bound at these widths); `exact` is
4.0 s a step. 14 GB of pinned host memory in every case. The 2048 step is
well over the 4x of 1024 that its tokens alone would predict: attention is
quadratic (16k rows) and the bf16 GEMM tiles are no longer weight-bound
there, so that size is where the prefix KV cache and MLP chunking below
would pay. Z-Image's `check_zimage_parity.py` numbers are
bit-identical before and after the attention-kernel split.

## Next

- The prefix KV cache (`use_kv_cache=True`): text rows are step-independent
  under `causal_condition`, so a block could store their K/V once and later
  steps run only the image rows. Saves 2-10% of a step at these prompt lengths;
  the sequence layout and attention segments here already allow it.
- The DiT workspace at 2048 is ~4 GB (the 12288-wide `gate`/`up` planes are
  half of it); chunking the MLP over rows would bring it to ~2.5 GB. Since
  2026-09-20 the arena is a cap that grows on demand (`diff_ensure_workspace`,
  `--image-reserve` for the old reserve-at-startup behaviour), so this costs
  memory only when someone renders that large.

## Workspace as a cap (2026-09-20)

`--image-max-size` used to reserve the arena at startup, which for a model
whose range is 512-2752 meant deciding the largest render before serving.
Now it is a cap (default: the model's native side) and the arena starts at
1024x1024, growing when a request needs more: the stream is drained, the
larger arena is taken before the old is freed, and a card that cannot give
it fails that request with the sizes in the message. The trade is that next
to a chat model the free memory at startup may have gone to the expert cache
by then; `--image-reserve` keeps the guarantee for people who want it.
- Encoder row 0: check whether the massive-activation channel wants f32
  through the Q8 GEMM, as the Z-Image plan wondered for its encoder.
- Editing: the Qwen3-VL vision tower, the VAE encoder, and image blocks in the
  prefix (the attention segment split generalises; the modulation split does
  too, since condition images also take the t = 0 row).
