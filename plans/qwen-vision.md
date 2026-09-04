# Qwen vision (image input) for the qwen35 / qwen35moe family

Target: serve Qwen 3.5-family GGUF checkpoints with their llama.cpp-style
`mmproj-*.gguf` vision tower (projector type `qwen3vl_merger`), so an OpenAI
`image_url` part or an Anthropic `image` block reaches the model as image
tokens. Local test pair: `Ornith-1.5-35B-Q6_K.gguf` (qwen35moe, rope
sections [11, 11, 10, 0]) + `mmproj-Ornith-1.5-35B-BF16.gguf` (27-block
SigLIP ViT, 1152 wide, 16 heads, patch 16, temporal patch 2, 2x2 merge into
2048, no deepstack layers).

## What the tower computes (llama.cpp tools/mtmd/models/qwen3vl.cpp)

1. Image resized to multiples of 32 (patch 16 x merge 2), normalized with
   mean/std 0.5. Patches are ordered window-major: for each 2x2 merge window,
   its four 16x16 patches. Each patch is 3x16x16, fed twice (temporal patch 2)
   through the split conv weights `v.patch_embd.weight` + `.weight.1`, plus
   `v.patch_embd.bias`.
2. Learned 48x48 position table bilinearly resized (align corners) to the
   patch grid, permuted into the same window-major order, added.
3. 27 pre-LN blocks: LayerNorm(eps 1e-6, bias) -> fused qkv (bias) -> 2D rope
   on q, k (head_dim 72: pairs (j, j+36); j < 18 rotates by row with
   10000^(-2j/36), j >= 18 by column with 10000^(-2(j-18)/36)) -> full
   non-causal attention (scale 1/sqrt(72)) -> out proj (bias) -> residual ->
   LayerNorm -> up (bias) -> GELU (tanh) -> down (bias) -> residual.
4. `v.post_ln`, then groups of 4 consecutive patches concatenated (4608) ->
   `mm.0` (4608x4608, bias) -> GELU -> `mm.2` (4608x2048, bias). One row per
   merged token, in row-major merged-grid order.

## LLM side: interleaved M-RoPE

`qwen35*.rope.dimension_sections = [t, h, w, 0]` over the 32 rotary pairs.
Pair p takes the position component by `p % 3` (0 -> t, 1 -> h, 2 -> w) while
`p < 3 * section`; text tokens carry t = h = w so the rotation equals the
current one. Image tokens at KV index `s..s+n`: t = P, h = P + row,
w = P + col over the merged grid, where P is the rope position the preceding
text reached; the next text token continues at P + max(gh, gw). So the rope
position of a token is its KV index minus a running delta of
`n - max(gh, gw)` per earlier image.

Rope is applied per token inside `qwen_attention_query` / `qwen_attention_key`
(native_kernels.hpp ~712) from a scalar host `position`; the prefill rows
driver (v2_mtp_verifier.inc ~1072) and decode (v2_runtime.cpp ~18710) and
the MTP draft (~16241) each derive it from `runtime.position`. Nothing else
in the attention/KV path reads the position for rope.

## Phases

1. **Vision encoder, native.** Open the mmproj through `flyweight_v2_model_open`
   (already parses it), read the `clip.*` keys, upload tensors, run the tower
   with new corpus kernels (layernorm+bias rows, bf16 matmul with bias, 2D
   rope, dense non-causal attention, GELU, window-major gather, bilinear
   pos-embed resample done on the host). C API:
   `flyweight_v2_model_attach_vision(model, path)` and
   `flyweight_v2_qwen_vision_encode(runtime, pixels, width, height, out)`.
   Reference: `native/tools/qwen_vision_reference.py` (numpy over the same
   GGUF) with a parity check.
2. **M-RoPE + embedding overlay.** `flyweight_v2_qwen_task_submit_vision`
   takes images with their token offsets; the engine encodes them at prefill
   start, `qwen_forward_rows` overwrites the staged embedding rows for image
   spans, and every rope site asks `qwen_rope_position(runtime, kv_index)`
   for its (t, h, w). Image spans (offset, count, gh, gw, content hash) live
   next to `processed_tokens` in the sequence, its snapshots, host spills and
   donations, and cap prefix matches so a different image behind the same
   token ids never reuses KV.
3. **Server.** `--mmproj PATH` (explicit: a directory can hold several
   models' towers, so nothing is guessed), Pillow preprocessing (optional `[vision]` extra), OpenAI
   `image_url` (data: and http(s)) and Anthropic `image` blocks, template
   content lists so `<|vision_start|><|image_pad|><|vision_end|>` is emitted
   and the pad expanded to the merged token count, usage/token-count
   endpoints, README.

Deepstack (`clip.vision.is_deepstack_layers`) is carried in the encoder
interface but rejected at attach until a checkpoint that uses it is on hand.

## Status (2026-09-03)

All three phases landed on `v2-native-runtime`:

- Tower: `native/src/v2_vision.inc` + corpus kernels (`vision_*` in
  flyweight_v2_native_kernels.hpp); parity against
  `native/tools/qwen_vision_reference.py` via `check_vision_parity.py` is
  within 2e-5 absolute on CUDA and the CPU backend. A 1024x1024 image (1024
  tokens) encodes in ~1.9 s on the RTX 5070 Ti laptop (tiled online-softmax
  attention; the GEMMs are the plain 64x64 tile kernel, not tensor cores).
- Decoder: every Qwen rope launch goes through `qwen_attention_*_mrope`
  with `qwen_rope_args`; image spans ride beside `processed_tokens` in
  sequences, checkpoints, host spills and donations (`qwen_spans_agree`
  caps prefix matches). Images are encoded lazily on first prefill touch,
  so a picture inside a reused prefix never runs the tower again.
- Server: `--mmproj`, `--image-max-tokens`, `--image-urls`; OpenAI
  `image_url`, Responses `input_image`, Anthropic `image`; `usage` counts
  image tokens; `/health` reports `execution.vision`.

Verified end to end on Ornith-1.5-35B (qwen35moe) + its mmproj: the model
reads a handwritten date off a photographed letter through the HTTP API,
second turns reuse the cached prefix across the image, and llama-server on
the same files gives a comparable description.

Open: deepstack injection into the decoder (the rows driver's two-half
pipeline needs the add inside `queue_combine`); tensor-core GEMMs for the
tower; the MTP draft embeds image rows as the pad token (acceptance-only).
