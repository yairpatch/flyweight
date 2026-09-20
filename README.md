# Flyweight

Flyweight serves GGUF and safetensors language models from a native C++/CUDA
runtime, behind OpenAI- and Anthropic-compatible HTTP APIs and a bundled chat
UI. It is built for one NVIDIA card and a lot of host RAM: mixture-of-experts
checkpoints larger than VRAM run with their routed experts on the CPU, or
split between the two, and the runtime measures what fits rather than asking.

| Family | Formats | Highlights |
| --- | --- | --- |
| Qwen 3 / 3.5 / 3.6 / 3.8, dense and MoE | GGUF; safetensors (3.5 family) | Full feature set: multi-token prediction, expert offload, image input via `mmproj` |
| Qwen3.8-Flash-Next (`qwen4exp`) | GGUF | Hybrid DeltaNet + gated attention; image input; MTP with a draft block or `--mtp-model` |
| DeepSeek-V4 / V4-Flash | GGUF | Its own CPU/hybrid runtime; DSpark speculative drafts via `--mtp-model`; DSML tool calls |
| BailingMoE3 (Ling 3.0) | GGUF; safetensors | Its own runtime with per-slot snapshot prefix reuse |
| K2-Horizon (dense, MoVA) | GGUF | Grouped norms, softplus attention gate; the MoVA value experts page through their own cache |
| Gemma 4 | GGUF | Routed experts must be Q4_0 (the QAT release); no MTP |
| Laguna 2.1 | GGUF | Per-head attention gate; no MTP |
| Muse Glimmer | GGUF | Channel-tagged reasoning; no speculative decoding |

Image generation with [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
or [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (RGBA out)
runs on the same engine beside a chat model (`--image-model`).

## Install

Wheels are on PyPI as `flyweight-llm` for Linux x86-64 (manylinux 2.28) and
Windows x64. The import package and the command are `flyweight`:

~~~bash
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\Activate.ps1 on Windows
pip install flyweight-llm
flyweight doctor
~~~

`doctor` reports whether this machine can serve and names the fix for
anything missing. Only `FAIL` lines block; a missing GPU is a warning, since
`--backend cpu` serves without one.

### GPU requirements

Nothing CUDA is linked into the wheel. At startup the runtime loads the
NVIDIA driver and NVRTC, compiles its kernels for your card, and reads the
CUDA headers (`cuda_fp16.h`, CCCL) from `CUDA_PATH`, `CUDA_HOME`,
`/opt/cuda` or `/usr/local/cuda`. So you need:

- the proprietary NVIDIA driver (`nvidia-smi` shows a device),
- the CUDA toolkit for NVRTC and the headers: the `cuda` package on Arch,
  `cuda-toolkit` on Debian, Ubuntu and Fedora, or NVIDIA's installer. `nvcc`
  is never used.

cuBLAS is optional and used when present. Any compute capability runs; int8
tensor-core kernels need 7.5 or newer, and the NVFP4 prefill path needs
Blackwell.

### From source

Needed on macOS, on ARM, and for developing Flyweight. Python 3.11+, CMake
3.24+, and a C++20 compiler: GCC 13+ or Clang 16+ on Linux, MSVC v143 (any
Visual Studio 2022 edition or the Build Tools) on Windows. Ninja is optional
but makes the Windows build parallel.

~~~bash
git clone https://github.com/yairpatch/flyweight
cd flyweight
pip install .            # compiles the runtime; a few minutes
flyweight doctor
~~~

If `pip install` fails on the compiler, `PYTHONPATH=src python -m flyweight
doctor` from the checkout says which of CMake, the compiler and the build
tool it cannot see. If the `flyweight` command is not on PATH, `python -m
flyweight` takes the same arguments.

## Quick start

~~~bash
flyweight serve model.gguf
~~~

Open `http://127.0.0.1:8000/` for the chat UI, or call the API:

~~~bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "local", "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 64}'
~~~

`MODEL` is a `.gguf` file (the first shard of a split archive) or a
safetensors checkpoint directory. The defaults choose the backend and memory
layout from what the machine has; the flags below are the ones worth
knowing. `flyweight serve --help` lists everything, grouped.

## Serving

### Context and memory

| Flag | Default | Meaning |
| --- | --- | --- |
| `--context N` | 32768 | prompt plus output tokens per sequence |
| `--max-tokens N` | 4096 | output cap for requests that do not set one |
| `--parallel N` | 1 | independent sequence slots, each with its own KV cache |
| `--scratch-context N` | 0 | smaller context for the slots past the first |
| `--cache auto\|off\|MIB` | auto | host-RAM cache for displaced conversations, restored by longest prefix. `auto` is one eighth of free RAM, capped at 8 GiB |
| `--cache-type-k`, `--cache-type-v` | f16 | KV precision: `f32`, `f16`, `bf16`, `q8_0`, `turbo3`, `turbo4` |
| `--gpu-cache-mib N` | 0 | VRAM for the expert cache; 0 sizes it from what is free at startup |
| `--expert-mode` | auto | where routed experts run, below |
| `--cpu-threads N` | 0 | CPU expert workers; 0 picks the physical cores |
| `--dense-requant auto\|q8\|off` | auto | repack BF16 dense weights to Q8_0 on the GPU when VRAM is tight |
| `--mtp-drafts N` | 0 | speculative decode with the checkpoint's draft block, up to 8 |
| `--mtp-model PATH` | | a standalone draft GGUF (Qwen MTP files, DSpark for DeepSeek-V4) |
| `--backend auto\|cuda\|cpu` | auto | `cpu` runs every kernel on the host |

Expert modes:

| Mode | Prompt experts | Decode experts | Use |
| --- | --- | --- | --- |
| `auto` | CPU | hot set on the GPU, misses on the CPU | default |
| `cpu` | CPU | CPU | least VRAM |
| `resident` | GPU | GPU | refuses to start unless every expert fits |

The older `hybrid`, `gpu`, `legacy-hybrid` and `legacy-paging` values are
accepted as aliases. On `--backend cpu` the mode is forced to `cpu`.

Prompt caching is automatic and needs no flag: a conversation that leaves
the GPU is packed into the host cache and restored on its next request, and
mid-prompt checkpoints (`--prefill-checkpoint-interval`, default 256 tokens)
let a conversation that diverged part-way reuse the part it shares.

### Choosing KV precision

`q8_0` halves the KV cache and `turbo4` quarters it. That matters most on a
dense model with a wide `head_dim`, where KV competes with the weights for
VRAM and every dense block that loses is re-read over PCIe on each token.
Qwen3.8-27B (UD-IQ2_XXS) on a 12 GB card:

| context | KV | dense blocks spilled | decode |
| --- | --- | --- | --- |
| 16K | `f16` | 5 of 64 | 16.6 tok/s |
| 16K | `q8_0` | none | 23.2 tok/s |
| 32K | `f16` | 16 of 64 | 10.8 tok/s |
| 32K | `q8_0` | 5 of 64 | 15.3 tok/s |
| 32K | `turbo4` | none | 22.9 tok/s |

If the startup banner reports dense blocks on the CPU, spend KV precision to
buy them back before anything else. `auto` only reaches for `turbo4` on a
MoE checkpoint above 32K context.

### Public deployment

~~~bash
flyweight serve model.gguf --host 0.0.0.0 --api-key "$KEY" \
  --concurrency 8 --max-connections 64 --cors-origin https://app.example
~~~

- `--api-key` (or `FLYWEIGHT_API_KEY`) requires `Authorization: Bearer` or
  `x-api-key` on every route except the UI's static files. No key means no
  authentication.
- `--concurrency` (default 64) bounds requests admitted to inference; the
  rest get HTTP 429 with `Retry-After`. `--max-connections` (default 128)
  bounds HTTP threads, `--request-timeout-seconds` (default 30) the time a
  client may take to send its body.
- `--strict-model` rejects requests naming a model other than `--model-name`.
- Request bodies are capped at 16 MiB.

### The server log

One row per request:

~~~
          endpoint   prompt  cached   ttft     out   tok/s  finish        total
10:49:45  chat        26.5k     98%   1.7s     112    35.8  tool call      4.9s
11:30:01  chat           --      --     --      --      --  400         0.1s  prompt is too long: 70212 tokens > 65536 maximum
~~~

`prompt` is the rendered prompt length, `cached` how much of it the prefix
cache served, `ttft` the wait before the first token, `finish` why generation
stopped. A low `cached` on a conversation that only appended is what a cache
problem looks like. `--quiet` prints failures only; `--verbose` adds the
access log and the prefix-cache diagnostics. Colour is off when redirected
or under `NO_COLOR`.

### Chat UI

The single-page app at `/` reaches everything the server exposes: chat over
any of the three protocols with a collapsible thinking panel and an **Answer
now** button, tool definitions with cards to paste results into, image
attachments when a vision tower is loaded, PDF, Word, Excel and text
attachments always (extracted in the browser, never uploaded as files), a
settings panel for every sampling and reasoning knob, a runtime dashboard
over `/health`, a tokenizer view, a raw-completions playground, an image
studio when `--image-model` is set, and an inspector that keeps each
request's body, SSE frames and an equivalent `curl` line. Conversations live
in the browser's IndexedDB; the API key lives in session storage.

## API

| Route | Protocol |
| --- | --- |
| `POST /v1/chat/completions`, `POST /v1/completions` | OpenAI |
| `POST /v1/responses`, `GET`/`DELETE /v1/responses/{id}`, `POST /v1/responses/input_tokens` | OpenAI Responses; the 128 most recent are kept, `store: false` skips one |
| `GET /v1/models`, `GET /v1/models/{id}`, `GET /v1/me` | OpenAI |
| `POST /v1/messages`, `POST /v1/messages/count_tokens` | Anthropic |
| `POST /v1/images/generations`, `/v1/images/edits` | OpenAI Images, with `--image-model` (edits: Qwen-Image-2.1) |
| `POST /v1/chat/completions/{id}/stop_thinking`, `POST /v1/messages/{id}/stop_thinking` | Flyweight |
| `GET /health`, `GET /props`, `GET /slots`, `POST /tokenize`, `POST /detokenize` | llama.cpp-style |

Every generation route streams over SSE with `"stream": true`; chat streams
honour `stream_options.include_usage`. `usage` reports `cached_tokens` (the
prompt prefix reused) and `reasoning_tokens`. `logprobs`, `logit_bias` and
`n` other than 1 are rejected with 400 rather than ignored.

Chat requests render through the checkpoint's own `chat_template`. A
`generation_config.json` beside the model supplies sampling defaults;
otherwise the defaults are llama.cpp's (temperature 0.8, `top_k` 40, `top_p`
0.95, `min_p` 0.05, penalties off). Precedence is request, then server flag,
then `generation_config.json`, then built-in. `GET /props` reports the
resolved defaults, the reasoning-effort levels the template accepts, and a
`capabilities` list.

### Thinking

- `reasoning_effort` (`low`, `medium`, `high`, `xhigh`, or `none` for off)
  is read from the flat field, `reasoningEffort`, Responses-style
  `reasoning.effort`, `chat_template_kwargs.reasoning_effort`, and Anthropic
  `output_config.effort`, so Claude Code, opencode and pi all work unchanged.
  A level the template does not name maps to its nearest neighbour.
  `--reasoning-effort` sets the server default; `none` there means the server
  does not reason unless a request asks.
- `reasoning_budget_tokens` is a hard ceiling: at the limit the sampler
  closes the thinking block and the answer resumes. Anthropic's `thinking:
  {"type": "enabled", "budget_tokens": N}` maps onto it, and a `/v1/messages`
  request that thinks without a budget gets half its `max_tokens` or 2048,
  whichever is smaller. `--thinking-budget N` applies one cap everywhere; 0
  removes it.
- The `stop_thinking` routes interrupt a live stream's thinking on the next
  token. `{id}` is the id from the stream's first event.
- `prefill_progress: true` on a streaming request adds `ping` frames with
  `flyweight.prefill` progress while a long prompt is evaluated.

Chain-of-thought arrives in `reasoning_content` on OpenAI routes and as
`thinking` blocks on `/v1/messages`, never in `content`.

### Tools and structured output

Declared tools are enforced by a sampler grammar: the name must be declared,
required parameters present, array and object values well-formed JSON.
`response_format` (`json_object`, `json_schema`; `text.format` on
`/v1/responses`) is enforced the same way. Tool-call arguments stream
incrementally. DeepSeek-V4, BailingMoE3 and K2-Horizon render tools through
their own templates and are parsed but not grammar-enforced; every other
family gets the Hermes-style tool prompt and the grammar.
`parallel_tool_calls: false` caps a turn at one call, and
`--max-tool-call-tokens` bounds a runaway one.

Inside a tool call, repetition penalties pause and temperature is capped at
0.2, because an edit tool's arguments must reproduce file text exactly.

### Sampling

`temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty` (1 = off,
looks over the last `penalty_window` = 64 generated tokens),
`presence_penalty`, `frequency_penalty`, `seed`, `stop`. Each has a server
flag of the same name. A heavily quantized checkpoint that locks onto a line
usually wants `repetition_penalty: 1.1`; otherwise leave it off, since it also
penalizes quoted file content.

### Images in

Qwen 3.5-family and Qwen3.8-Flash-Next GGUFs accept images when their vision
tower is attached. The tower is the `mmproj-*.gguf` published beside the
model; decoding needs Pillow (`pip install 'flyweight-llm[vision]'`):

~~~bash
flyweight serve Qwen3.5-35B-A3B-Q6_K.gguf --mmproj mmproj-Qwen3.5-35B-A3B-BF16.gguf
~~~

OpenAI `image_url`, Responses `input_image` and Anthropic `image` parts are
accepted, as `data:` or `http(s)` URLs (`--image-urls deny` refuses the
latter). Each image is resized to at most `--image-max-tokens` (default 1024,
one token per 32x32 block), counted as prompt tokens, and never re-encoded
when it sits inside a reused prefix. Without a tower an image part degrades
to a visible `[image omitted]` note rather than failing the request.

### Images out

`--image-model DIR` loads a diffusers snapshot of
[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) or
[Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
(`text_encoder/`, `transformer/`, `vae/`, as `hf download` leaves it) and
serves it at `/v1/images/generations` and in the UI's image studio:

~~~bash
flyweight serve Qwen3.6-35B-A3B-Q6_K.gguf --image-model /path/to/Z-Image-Turbo
curl http://127.0.0.1:8000/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt": "a red bicycle leaning on a brick wall", "size": "1024x1024", "seed": 7}'
~~~

The request takes `prompt`, `size` (on the model's grid -- multiples of 16
for Z-Image, 32 for Qwen-Image-2.1 -- up to `--image-max-size`, which
defaults to the model's own maximum: 1024 and 2048), `n` (1 to
4), `seed`, `steps` (the model's own default: 8 for Z-Image, 40 for
Qwen-Image-2.1) and `shift` (0 lets Qwen-Image-2.1's shift follow the
image's size, as its pipeline does); the response is base64 PNG. Qwen-Image-2.1
renders RGBA: the PNG carries an alpha channel, and a prompt starting "This is
an RGBA image with transparency" is what makes the model use it. It also
edits: up to ten reference images go in as `images` (data URLs or base64) on
the same request, or as `image` parts of an OpenAI-shaped multipart
`POST /v1/images/edits`; the picture takes the last reference's shape unless
`size` is set, and the studio has an attach button for them. One render at a
time; a concurrent request gets
429. The encoder and DiT are quantized to Q8_0 on first open and cached
beside the checkpoint. The workspace starts sized for 1024x1024 and grows when a request needs
more; a render the card cannot fit fails on its own rather than at startup,
and `--image-reserve` holds the whole `--image-max-size` from the start
instead. `--image-weights` keeps them on the GPU (`device`) or
streams them from pinned RAM (`host`, about 1.7 GB of VRAM at 1024x1024);
`auto` picks `host` when they would take more than half the card, which is
what lets a 12 GB card serve a 35B chat model and Z-Image together.
`--image-precision balanced` (default) runs bf16 tensor-core GEMMs; `fast`
uses int8; `exact` keeps f32 activations at about eight times the time. A
1024x1024 render takes about 14 s on an RTX 5070 Ti laptop. Render at 1024:
the model is trained there.

## Models and formats

### GGUF

Split archives (`-00001-of-0000N`) load from their first shard. Every
quantization type llama.cpp writes is readable, plus NVFP4 and the Q2_0
type. Which kernel serves a tensor depends on where it runs:

- **Dense projections on the GPU**: all of F32, F16, BF16, the K-quants,
  Q4_0, Q8_0, the IQ family, IQ1_S and IQ1_M. Q2_0 and IQ4_NL dense tensors
  are requantized to Q8_0 on upload.
- **Routed experts on the GPU** (grouped kernels): Q2_K, Q4_K, Q5_K, Q6_K,
  Q4_0, Q8_0, Q2_0, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS,
  IQ4_NL and NVFP4. Q3_K experts run on the CPU.
- **Routed experts on the CPU**: every type above except IQ1_M, with
  AVX-512 VNNI kernels where the host has them. IQ1_M routed experts are
  unsupported, and an IQ1_M LM head is requantized to Q8_0.
- **BailingMoE3** has grouped GPU expert kernels for Q4_K and Q6_K only;
  other types run one expert at a time, correctly but slowly.

### Safetensors

A safetensors checkpoint (Qwen 3.5 family and BailingMoE3) is packed to a
quantization on first open and cached beside it. On a terminal the CLI asks
which, listing sizes and marking cached ones; `--quant` or
`FLYWEIGHT_HF_QUANT` answers ahead of time, and a non-interactive run takes
`Q6_K`. Choices: `IQ2_XS`, `Q2_K`, `IQ3_XXS`, `Q3_K`, `IQ4_XS`, `Q4_K`,
`Q5_K`, `Q6_K`, `Q8_0`, `F32`.

- Pick the largest target that fits VRAM, not the largest you can pack. A
  dense block that does not fit runs on the CPU at about 3 ms per token.
  `--gpu-cache-mib` pins which blocks spill on a card shared with a desktop.
- `Q2_K` and `Q3_K` are refused on MoE checkpoints. `IQ4_XS` is the smallest
  MoE target below `Q4_K` whose experts stay on the GPU.
- `IQ2_XS` needs an importance matrix. An `imatrix.dat` beside the checkpoint
  is used automatically; `--imatrix PATH` names one elsewhere and `off`
  disables it. `flyweight imatrix MODEL --text calibration.txt` gathers one
  over any Qwen-family model, in llama.cpp's `.dat` layout.
- `IQ3_XXS` packing takes minutes rather than seconds, once.
- A spilled dense block in a codebook format is re-encoded to Q3_K for the
  host kernels, which is lossy. `FLYWEIGHT_HOST_FFN_FORMAT` picks `q2_k`,
  `q3_k`, `q8_0` or `off`.

### Speculative decoding

| Family | Mechanism |
| --- | --- |
| Qwen (all) | `--mtp-drafts N` with the checkpoint's draft block or a `--mtp-model` overlay. Drafting is trialled against plain decode and kept only when it wins |
| DeepSeek-V4-Flash | `--mtp-model` pointing at the DSpark draft GGUF |
| BailingMoE3 | draft block loads; the server does not drive it yet |
| Qwen without a draft block | `FLYWEIGHT_LOOKUP_DRAFTS=N` (up to 7): prompt-lookup self-speculation, worth 10-15% on code, neutral on prose |
| Gemma 4, Laguna, Muse Glimmer, K2-Horizon | none |

Every verified token passes through the request's own sampler, so a
drafting request answers as a non-drafting one would.

## Other commands

| Command | What it does |
| --- | --- |
| `flyweight generate MODEL --prompt TEXT` | print one response and exit (`--max-tokens`, `--temperature`, `--seed`, `--enable-thinking`) |
| `flyweight benchmark MODEL` | preparation, prefill and steady decode timings as JSON (`--chat`, `--iterations`, `--cold-cache`) |
| `flyweight inspect MODEL` | metadata, resolved config and tensor list as JSON |
| `flyweight imatrix MODEL --text FILE` | gather an importance matrix |
| `flyweight probe MODEL` | run a few tokens and dump runtime counters |
| `flyweight doctor` | check the install |
| `flyweight transcript-audit DIR` | explain a coding harness's failed edits from a request dump |

For repeatable comparisons, `python -m flyweight.runtime_benchmark run` writes
JSONL across prompt lengths and `compare` diffs two files. Run GPU benchmarks
alone on the card: another process changes free VRAM and therefore the
automatic expert-cache size.

`transcript-audit` answers one question: when a harness's edit replaced text
that was not in the file, did the model edit blind, or did the file content
get lost between the client and the model? With
`FLYWEIGHT_TRANSCRIPT_DUMP=DIR` set, `serve` writes one JSON file per request
holding both what the client sent and what the model saw.

## Environment variables

| Variable | Meaning |
| --- | --- |
| `FLYWEIGHT_API_KEY` | default for `--api-key` |
| `FLYWEIGHT_HF_QUANT`, `FLYWEIGHT_HF_IMATRIX` | defaults for `--quant` and `--imatrix` |
| `FLYWEIGHT_HF_CACHE` | where packed safetensors go; `off` disables the cache |
| `FLYWEIGHT_TRANSCRIPT_DUMP`, `FLYWEIGHT_TRANSCRIPT_PROMPT=0` | request recording for `transcript-audit`; `=0` stores a digest instead of the prompt |
| `FLYWEIGHT_LOOKUP_DRAFTS` | prompt-lookup speculation, above |
| `FLYWEIGHT_DS4_EXPERT_CACHE_MIB` | a GPU expert cache for DeepSeek-V4 |
| `FLYWEIGHT_QSA=1` | Qwen3.8-Flash-Next's learned sparse-attention indexer (experimental; dense by default) |
| `FLYWEIGHT_V2_MLOCK=1` | lock the mapped model in RAM |
| `FLYWEIGHT_TOOL_GRAMMAR=0`, `FLYWEIGHT_RESPONSE_GRAMMAR=0` | disable the tool and JSON sampler constraints |
| `CUDA_PATH`, `CUDA_HOME` | where NVRTC and the CUDA headers are looked for |

The native runtime reads many more `FLYWEIGHT_*` switches (`_PROFILE`,
`_TRACE`, kernel A/B toggles such as `FLYWEIGHT_CUDA_GRAPHS=0` and
`FLYWEIGHT_PREFILL_PIPELINE=0`). They exist for measurement; leave them unset
when serving.

## Limitations

- CUDA is the only accelerator. `--backend cpu` runs everything on the host.
- Two platforms ship wheels; everything else builds from source.
- No embeddings, audio, fine-tuning or hosted-tool APIs. Response records
  and prompt caches are process-local.
- Qwen3.8-Flash-Next runs its sparse-attention layers as dense attention by
  default, exact within the trained 2048-token selection budget and an
  approximation beyond it. MTP needs a release with a draft block (the
  Q4_K_XL builds) or the standalone MTP file.
- Vision covers still images through a GGUF `mmproj` on the Qwen 3.5 family
  and Flash-Next. The safetensors loader drops the tower. Towers with
  deepstack layers are refused.
- Gemma 4 has no MTP and its routed experts must be Q4_0 (the QAT release).
- BailingMoE3 interleaves its slots rather than batching them, so
  `--parallel` removes waiting but does not multiply throughput, and it has
  no expert paging: a model that does not fit runs on the host entirely.
- Laguna supports only the per-head attention gate; larger Laguna
  checkpoints with the per-element gate are rejected at load.
- Text is not NFC-normalized before tokenizing, so a decomposed accent can
  tokenize differently from the reference tokenizer.
- Special-token spellings inside message content (`<|im_start|>`,
  `<tool_call>`, `<think>`) are tokenized as the control tokens, as the HF
  and llama.cpp tokenizers do. A client relaying untrusted text should strip
  them.

## Development

~~~bash
pip install -e '.[test]'
python -m flyweight.native_build     # same build tree, plus contract tests and benchmarks
ctest --test-dir build/native --output-on-failure
pytest -q                            # ~3 minutes; --run-slow adds the parity tests
ruff check src tests bench tools native/tools setup.py && mypy src/flyweight
~~~

Never keep an editable and a regular install in one environment; the regular
one wins every import, and `doctor` reports the shadowing. Never copy a new
library over a loaded one; `native_build` installs it by rename.

The chat UI is a Vite + React app in `web/`, and its build is committed under
`src/flyweight/ui/` so the wheel ships it without Node. After changing
`web/`, run `pnpm build` there and commit the output; CI fails on a stale
bundle.

`FLYWEIGHT_TEST_MODEL=/path/to/model.gguf` opts the suite into the real-model
tests. `tools/check_*.py` are parity checks that need real weights.

Where things live:

- `native/src/v2_runtime.cpp`: GGUF parsing, memory planning, scheduling,
  prefix reuse, sampling and the C ABI; `v2_mtp_verifier.inc` (prefill
  driver and MTP), `v2_vision.inc` (the tower), `v2_diffusion.inc` (the image tower, Z-Image), `v2_qwenimage.inc` (Qwen-Image-2.1)
- `native/src/gpu_driver.cpp`: CUDA driver, NVRTC, cuBLAS, graphs, transfers
- `native/include/flyweight_v2_qwen_kernels.hpp` and siblings: the CUDA
  kernel source, JIT-compiled by NVRTC and also compiled as host C++ for
  `--backend cpu`; `flyweight_v2_format_dispatch.hpp` maps tensor formats to
  kernels
- `native/include/flyweight_v2_bailing.hpp`, `flyweight_v2_deepseek4*.hpp`,
  `flyweight_v2_hf*.hpp`, `flyweight_v2_tool_grammar.hpp`: the BailingMoE3
  and DeepSeek-V4 runtimes, the safetensors loader and packer, the sampler
  grammar
- `src/flyweight/cli.py`, `v2.py`, `v2_server.py`, `server.py`: the command
  line, the ctypes bindings, the engine thread, the HTTP protocols;
  `deepseek4_server.py` and `dspark.py` for DeepSeek-V4; `images.py` and
  `vision.py` for image generation and input
- `plans/`: dated design records, each with a status line, for what was
  built and what was measured and dropped

See [CONTRIBUTING.md](CONTRIBUTING.md) for how changes land, including the
version-bump rule, and [SECURITY.md](SECURITY.md) for reporting a
vulnerability.

## License

Apache-2.0.
