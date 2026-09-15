# MiniMax-H3 text-to-video

Started 2026-09-15, after Z-Image. The second diffusion model on the engine
and the first video one. Weights come from `unsloth/MiniMax-H3-GGUF`
(Qwen3-VL-32B conditioner at Q4_K_M, the 15B DiT at Q6_K, the ViT video VAE
as fp16 safetensors) with the configs and tokenizer from
`MiniMaxAI/MiniMax-H3`; the model directory layout is in the README.

## What runs

- `native/src/v2_h3.inc`: the DiT (50 blocks plus 2 token-refiner blocks,
  hidden 5376, 56 heads, fused fc1 `[gate; up]`, 3-axis rope with the
  checkpoint's `rope.inv_freq`, the pruned adaLN table lerped by `t` and one
  linear per block, per-modality modulation rows) and the ViT VAE decoder
  (36 layers, per-head interleaved `to_qkv` reordered at load, register and
  zero tokens, rope over `(2(i+0.5)/n - 1) * 2pi`, diffusers' `_decode`
  chunk/blend/pad-cut, and the same 256-pixel spatial tiles with 64-pixel
  linearly blended overlaps). Without the tiles the decode drifts from 0.7% to
  27% RMS as the width grows past 16 latents and every frame shows a grid.
- The sequence is `[text | audio | video]`; audio rows are channel-major, two
  channels of 32-wide latents, 40 per second. Text rows inherit the video
  timestep.
- `flyweight.videos.VideoGenerator` runs the two rectified-flow schedules
  (`linspace(1, 0, steps)` shifted by 12 for video and 3 for audio, data-ward
  velocity `x0 = x_t + sigma v`, Euler blend `r x_t + (1 - r) x0`), decodes
  the video latents and writes an H.264 MP4 with PyAV. Frames snap to
  `17n + 5` (latent `5n + 2`).
- `/v1/videos/generations` (JSON or SSE with `stream: true`), CLI flags
  `--video-model/--video-max-size/--video-max-frames/--video-weights/
  --video-precision`, and the Video studio tab in the web UI.

## Verified

`tools/h3_reference.py` (torch, streaming the GGUF blocks with gguf-py) on the
same weights: encoder 0.02% RMS, DiT step cosine 0.99994 (video rows) and
0.9977 (audio rows), VAE decode 0.65% RMS with a max of 7/255 from the Q8_0
weights. Native step at 640x384 x 124 frames: 16 s; at 256x256 x 22: 2.3 s
(the host-streaming floor for 16.6 GB of DiT weights).

## Performance round (2026-09-15 evening)

Step at 640x384 x 124 frames (about 9,300 tokens): 16.0 s -> 8.3 s.

- Profile (FLYWEIGHT_DIFF_SYNC=1 FLYWEIGHT_DIFF_TRACE=1): the Q6_K MMQ GEMM
  was 66% of the step at 30 TOPS, attention 26% at 27 TFLOPS. cuBLAS reaches
  200 TOPS on int8 and 54 TFLOPS on bf16 at these shapes; PyTorch's flash
  attention 53 TFLOPS.
- GEMM precision study (tools/h3_reference.py dit --gemm-sim): only per-32
  int8 activation blocks hold accuracy (2.4% per step vs f32). Per-token int8
  10%, FP8 16%, MXFP8 13%, SmoothQuant 11%. So cuBLASLt's int8/FP8 modes are
  out; the answer is our own block-scaled int8 tensor-core GEMM.
- `diff_int8_gemm`: m16n8k32 int8 mma, ldmatrix fragments, cp.async double
  buffering, grouped raster, magic-number int-to-float. 83 TOPS on the qkv
  shape (unscaled ceiling of the same structure 119). Weights come from a
  load-time Q6_K -> per-32 int8 requantization on the device
  (`diff_requant_q6k_int8`, 0.5% RMS from the dequantized Q6_K). Against
  the f32 reference at a realistic 280-row step: video 2.6% (MMQ 3.8%),
  audio 1.0%.
- `diff_flash_attention_bf16`: rewritten with ldmatrix (V through
  `.trans`, no scalar transpose) and 32-key double-buffered cp.async tiles:
  30 -> 50 TFLOPS.
- `diff_quantize_q8_rows`: float4 activation quantizer, bit-identical, 5x
  faster than the 32-thread-block one.

Left on the table: the GEMM's scale epilogue (~30% over the unscaled kernel),
elementwise fusions (~12% of the step in norms, rope, modulation, gating),
and the same int8 GEMM for Z-Image's Q8_0 DiT.

## Decisions

- Weights stream from pinned host memory by default: the encoder and DiT
  are 31 GB together and the card has 12 GB.
- Under `balanced` precision, quantized weights other than Q8_0 use the
  int8 tensor-core GEMM; the per-token rows kernels re-read the weights per
  row and cost 10x at a few hundred tokens.
- Sound is deferred: the audio rows take part in every step (the video
  attends to them), but the audio VAE decoder is not ported, so clips are
  silent.

## Left

- Audio VAE decoder and muxing the track into the MP4.
- FL2VA / Ref2VA references (needs the Qwen3-VL vision tower for images).
- Spatial chunking so canvases beyond the planned workspace are not refused.
- Q6_K to bf16 dequant in the balanced GEMM; row-chunked qkv/FFN for very
  long sequences.
