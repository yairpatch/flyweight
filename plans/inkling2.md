# Inkling 2 (`inkling2`)

Status: **load/config/tokenizer and a CPU execution path are implemented.**
Prefill runs as a sequence of single-token steps, so it is correct but much
slower than a fused batched kernel; there is no GPU path yet.

`Inkling-2-Mini-UD-IQ2_XXS.gguf` (`general.architecture = inkling2`) is
Thinking Machines Lab's 26.4B-parameter looped MoE transformer. llama.cpp's
`src/models/inkling2.cpp` is the reference for the remaining work; the
`llama_model_inkling2_loop` variant in `src/models/models.h` handles the loop.

## What is in place

- `ModelConfig` and the public `FlyweightV2ModelConfig` carry the loop geometry
  (`num_loops`, `loop_core_start/end`, `loop_gate_rank`, `attn_res_block_size`),
  the relative-position parameters (`relative_length/extent`, `qk_clip`,
  `relative_clip`, `log_scale_floor/alpha`), the unpadded vocabulary and the SiTU
  constants. All are parsed in `v2_runtime.cpp::parse()` and visible through
  `V2Model.config`.
- The Qwen runtime gate recognizes `inkling2` and points callers at the
  dedicated runtime; the desktop/model catalog treats it as a known family.
- `tokenizer.ggml.pre = inkling2` maps to llama.cpp's GPT-4o regex with
  `clean_spaces = false`, so `gguf_pretokenize()` uses `gpt4o_pretokenize(text,3)`
  and control tokens are split out by exact match. The GGUF's embedded chat
  template is used verbatim by `v2_server.py`; turn separators were added.

## Geometry of the Mini checkpoint

| Key | Value |
| --- | --- |
| block_count (layer applications) | 48 |
| stored layers | 24 |
| looped core | layers 6..17, three passes (12 extra cache slots) |
| embedding / head dim | 2304 / 128 |
| heads / KV heads | 24 / 8 |
| experts / top-k | 128 / 8 |
| sliding window | 1024, pattern alternates 5 sliding : 1 full |
| conv taps (`shortconv.l_cache`) | 4 |
| relative length / extent | 16 / 1024 |
| logit softcap | 30 |
| expert weight scale | 8 |

## Execution path (implemented)

`native/src/v2_inkling2.inc` is a dedicated runtime that keeps weights in their
GGUF quantization and reuses `bailing::row_dot`/`matvec` for every projection;
no f32 copy of the 8.5 GB checkpoint is made. It resolves the layer tensors
(`attn_q_clip`, `attn_sink_proj`, `attn_rel*`, four conv weights per layer,
`ffn_*_res_query`, and the `loop_*_{down,gain,forget}` controllers), then for
each token:

1. Embeds and RMS-normalizes the token; runs the prelude layers.
2. Runs `num_loops` passes over the core. Pass 0 uses the file's weights; later
   passes reset to the prelude output and mix the previous pass's bank through
   `loop_res_query` into four messages, applied by the LoRA-style `control()`
   gates. Each cache slot keeps its own KV and convolution state.
3. Per layer: `mix()` over block outputs, Q/K/V/R projections, row clips,
   RMS-normalized V, log-scaled tau on full layers, causal conv (with the
   full layers' per-head dynamic taps), relative-bias attention with a zero
   sink logit, sigmoid output gate, post-attention conv and block residual.
4. FFN: sigmoid top-8 routing with `exp_probs_b`, SiTU experts (beta 4, linear
   beta 25), expert scale 8, post-FFN conv, residual.
5. Output: `mix()` with `output_res_query`, RMS norm, LM head, softcap 30,
   slice to `vocab_size_unpadded`.

`V2Inkling2Runtime` in `v2.py` exposes the same interface as `BailingRuntime`,
so `v2_server.py` drives it with the shared streaming engine. Measured on the
real Mini checkpoint: ~6 s/token single-threaded before any batching or GPU
work, with finite logits in the expected softcapped range.

## Remaining work

- Batched prefill (today it is one forward per prompt token) and a GPU path.
- Native cache snapshots: the Python adapter currently saves the token list
  and replays it, which is correct but slow for prefix reuse.
- The loop message query is global in this GGUF (`loop_res_query`,
  `loop_kv_res_query`); llama.cpp master uses a per-layer
  `LLM_TENSOR_LOOP_MSG_QUERY`. Verify against the exact conversion revision
  before claiming numerical parity.
