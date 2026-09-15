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
  chunk/blend/pad-cut).
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
