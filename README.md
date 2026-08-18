# Colibrì Next

Colibrì Next is a native C++/CUDA GGUF inference runtime for Qwen 3.5/3.6,
Laguna 2.1, and supported Gemma 4 text models. Python provides the CLI, tokenizer-facing server
adapter, and OpenAI/Anthropic-compatible HTTP API; model execution stays in the
native runtime.

The former converted-model Python runtime (v1), safetensors conversion pipeline,
toy residency planner, and `.coli` model format are no longer part of the project.

## Features

- Memory-mapped GGUF loading
- Native CUDA attention, DeltaNet, dense FFN, and sparse MoE execution
- CPU, automatic hybrid, and strict resident expert placement
- F32, F16, BF16, Q8_0, Turbo3, and Turbo4 KV caches
- Sliding-window attention and compact circular KV storage
- Multi-token prediction for supported Qwen checkpoints
- Independent sequence slots and host-backed prompt-cache spill/restore
- Cooperative concurrent request scheduling and multi-sequence decode overlap
- Greedy, top-k, and nucleus sampling for Qwen
- OpenAI Chat Completions, Responses, and legacy Completions APIs
- Anthropic Messages and token-count compatibility endpoints
- Streaming SSE, native tool calls, bearer authentication, and a bundled chat UI
- Repeatable JSONL runtime benchmark and regression comparison harness

## Requirements

- Python 3.11+
- CMake 3.24+
- A C++20 compiler
- An NVIDIA driver for model execution
- Optional CuPy only for low-level CUDA development checks

Windows requires Visual Studio 2022 Build Tools with the **Desktop development
with C++** workload. Linux requires a recent GCC or Clang toolchain.

## Installation

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
PYTHONPATH=src python -m colibri_next.native_build
PYTHONPATH=src python -m unittest discover -s tests -v
~~~

On Windows, activate with `.venv\Scripts\Activate.ps1` and run the same Python
commands. CUDA kernels are compiled at runtime through the NVIDIA driver API;
the project does not require a separately installed CUDA toolkit for serving.

## Serve a model

~~~bash
colibri-next serve model.gguf
~~~

Open `http://127.0.0.1:8000/` for the local chat UI. The defaults select the
backend and memory policy automatically. The common public options are kept
small:

~~~bash
colibri-next serve model.gguf \
  --context 65536 --max-tokens 16384 \
  --host 127.0.0.1 --port 8000
~~~

Prompt caching is automatic: displaced conversations are packed into a
byte-budgeted host-RAM LRU and restored by longest matching prefix, including
when only one GPU sequence slot is configured. Use `--cache off` to disable it
or `--cache 4096` to set an explicit 4 GiB budget. Automatic mode uses one
eighth of currently available RAM, capped at 8 GiB.

Older `serve-v2`, `--context-window`, and `--max-new-tokens` spellings remain
accepted for script compatibility.

The native expert modes are:

| Mode | Prompt routed experts | Decode routed experts | Behavior |
| --- | --- | --- | --- |
| `cpu` | CPU | CPU | Minimum GPU expert memory |
| `auto` | CPU | Stable hot set on GPU, misses on CPU | Default |
| `resident` | GPU | GPU | Fails preparation unless every expert fits |

Legacy `hybrid` and `gpu` spellings remain compatibility aliases for the old
paging policies, but new deployments should use the canonical modes.

For concurrent agent clients, allocate independent sequence slots and optional
host prompt-cache storage:

~~~bash
PYTHONPATH=src python -m colibri_next.cli serve-v2 model.gguf \
  --context-window 58000 \
  --expert-mode cpu --cpu-threads 12 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 2 --prompt-cache-mib 4096 --cpu-prefetch-auto
~~~

Each sequence slot has its own KV and recurrent state. More slots improve
conversation isolation but consume additional VRAM. Bound both admitted
inference work and open HTTP connections for public-facing deployments:

~~~bash
PYTHONPATH=src python -m colibri_next.cli serve-v2 model.gguf \
  --max-concurrent-requests 8 --max-connections 64 \
  --request-timeout-seconds 30
~~~

## API example

~~~bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Say hi."}],
    "max_tokens": 64,
    "temperature": 0
  }'
~~~

Set `COLIBRI_API_KEY` or pass `--api-key` to require bearer authentication.
Use `--strict-model` when request model IDs must exactly match the configured
server model name.

Chat requests use the GGUF's `tokenizer.chat_template` when it is present;
the built-in architecture formatter is only a fallback for older files. If a
`generation_config.json` is stored beside the GGUF, its `temperature`, `top_k`,
`top_p`, `max_new_tokens`, and `do_sample` defaults are also loaded. Explicit
API request values always override model defaults. `GET /props` reports the
resolved defaults and their sources, and the bundled UI adopts them until the
user saves custom settings.

Sampling also takes `repetition_penalty` (default 1.1 over the last 64
generated tokens), plus OpenAI's `presence_penalty` and `frequency_penalty`
(default 0). These are on by default because "no penalty" is not a neutral
setting: with nothing discouraging a token the model has just produced, a
heavily quantized checkpoint can lock onto a line and repeat it until the
token budget runs out. Only generated tokens are penalized -- penalizing the
prompt would push the model away from the user's own wording. Set
`repetition_penalty` to 1 to disable, or raise `penalty_window` to look
further back.

## Inspect and generate

~~~bash
colibri-next inspect model.gguf

colibri-next generate model.gguf \
  --prompt "Explain mixture-of-experts routing." \
  --max-tokens 128 --temperature 0
~~~

## Benchmarking

The direct benchmark separates preparation, prompt prefill, and steady decode:

~~~bash
colibri-next benchmark model.gguf \
  --prompt "Explain sliding-window attention." --chat \
  --context 32768 --iterations 30 --warmup 10 \
  --expert-mode auto --cache-type-k f16 --cache-type-v f16
~~~

For reproducible comparisons across prompt lengths, use the checked-in JSONL
harness:

~~~bash
PYTHONPATH=src python bench_runtime.py run model.gguf \
  --output /tmp/baseline.jsonl --label baseline \
  --prompt "Runtime regression benchmark." \
  --prompt-lengths 256,1024,4096 \
  --context 32768 --samples 5 --sample-warmup 1

PYTHONPATH=src python bench_runtime.py compare \
  /tmp/baseline.jsonl /tmp/candidate.jsonl
~~~

Run GPU benchmarks in isolation. Another process changes free VRAM and therefore
changes automatic expert-cache sizing.

## Runtime controls

Important CLI options include:

- `--quant ask|Q2_K|IQ3_XXS|Q3_K|Q4_K|Q5_K|Q6_K|Q8_0|F32`: quantization for a
  safetensors checkpoint (see below)
- `--gpu-cache-mib 0`: size allocations from currently free VRAM
- `--cache-type-k` / `--cache-type-v`: KV precision
- `--mtp-drafts N`: enable supported Qwen MTP verification
- `--dense-requant auto|q8|off`: control temporary BF16 dense-weight Q8 upload
- `--parallel N`: independent sequence slots
- `--cache auto|off|MIB`: host cache for displaced conversation state
- `--max-concurrent-requests N`: reject excess generation work with HTTP 429
- `--max-connections N`: cap simultaneous HTTP connection threads
- `--request-timeout-seconds N`: bound idle/read time on client sockets
- `--prefill-cache-seed auto|off|N`: post-prefill hot-expert placement
- `--expert-paging auto|staged|direct`: legacy paging transfer policy
- `--cpu-prefetch-auto`: warm prompt-relevant expert pages when beneficial
- `--swa-full`: trade VRAM for unrestricted sliding-layer rollback

Runtime diagnostics are exposed through `/health`. Detailed profiling and
experimental kernel switches use `COLIBRI_*` environment variables; unset
profiling variables for production serving.

A GGUF arrives quantized; a safetensors checkpoint does not, so the first open
packs it and caches the result beside the checkpoint. On a terminal the CLI asks
which quantization to pack, listing the exact size of each and marking the ones
already cached -- picking a cached one opens in about a second, an uncached one
costs a repack and the disk to store it. Anything non-interactive keeps the
default (`Q6_K`), and `--quant`, or `COLIBRI_HF_QUANT`, answers ahead of time:

~~~
Qwen3.8-27B is a safetensors checkpoint. Choose how to quantize it:
  1) Q2_K        9.5 GiB   packs on first open, writes 9.5 GiB
  2) IQ3_XXS    10.8 GiB   packs on first open, writes 10.8 GiB
  3) Q3_K       11.9 GiB   packs on first open, writes 11.9 GiB
  4) Q4_K       14.9 GiB   cached, opens immediately
  5) Q5_K       17.8 GiB   packs on first open, writes 17.8 GiB
  6) Q6_K       20.9 GiB   cached, opens immediately  [default]
  7) Q8_0       26.5 GiB   packs on first open, writes 26.5 GiB
  8) F32       101.8 GiB   packs on first open, writes 101.8 GiB
quantization [Q6_K]:
~~~

Below `Q6_K` the tradeoff is accuracy against fit, and fit is what dominates: a
dense block that does not fit in VRAM is executed on the CPU, at about 3 ms per
token in decode -- prefill batches those blocks and pays less per token, but not
little enough to ignore. On a 12 GB card the 27B above spills 51 of 64 dense
blocks at `Q6_K` and none at `Q2_K`, which is the difference between 4 and 36
tokens/s of decode. Pick the largest target that still fits, not the largest you
can pack.

`Q2_K` and `Q3_K` are dense-only: no GPU routed-expert kernel decodes either, so
a mixture-of-experts checkpoint packed to one would run every routed layer on
the CPU. Both are refused there rather than silently doing that, and the menu
marks them unavailable on such a model. `IQ3_XXS` has grouped expert kernels and
no such restriction.

`IQ3_XXS` is a codebook format -- 3.06 bits per weight, against Q3_K's 3.44 --
and quantizing to it searches 256 patterns per four weights rather than rounding
to a lattice, so packing the 27B above takes ~5 minutes against ~40 seconds for
a K-quant. It is a one-time cost, cached like any other. It also prefills
fastest of the lot on the checkpoint above (196 tok/s at 1k context, against
273 for Q2_K only because Q2_K is 1.5 GiB smaller and spills nothing). Note that it is packed
*without* an importance matrix: llama.cpp weights this search by activation
statistics gathered over calibration data, and without them IQ3_XXS lands on the
K-quant accuracy curve rather than above it. What it buys here is size. The
sub-3-bit IQ formats (IQ2_XXS, IQ1_M) are not offered for the same reason
llama.cpp refuses them without a matrix: they need one to be worth using.

`--dense-requant auto` keeps the GGUF unchanged and chooses the temporary GPU
representation from the requested or available VRAM budget. It converts BF16
dense tensors to Q8_0 when the BF16 working set plus useful routed-expert cache
would exceed that budget. Use `q8` to force the memory-saving representation or
`off` to preserve the checkpoint's dense precision exactly.

`--cache-type-k` / `--cache-type-v` default to `f16`, and `auto` only reaches
for `turbo4` on a checkpoint with routed experts, above 32K context. A *dense*
checkpoint with a wide `head_dim` is the case that default serves badly, and it
has to be set by hand. Qwen3.8-27B (`qwen35`) is the worked example: 16 full
attention layers x 4 KV heads x head_dim 256 is 64 KiB of KV per token, so KV
competes with the weights for VRAM, and every dense block that loses is re-read
over PCIe on every token. On a 12 GB card with the UD-IQ2\_XXS build:

| context | KV       | dense blocks spilled | decode      |
| ------- | -------- | -------------------- | ----------- |
| 16K     | `f16`    | 5 of 64 (408 MiB)    | 16.6 tok/s  |
| 16K     | `q8_0`   | none                 | 23.2 tok/s  |
| 16K     | `turbo4` | none                 | 24.0 tok/s  |
| 32K     | `f16`    | 16 of 64 (1306 MiB)  | 10.8 tok/s  |
| 32K     | `q8_0`   | 5 of 64 (408 MiB)    | 15.3 tok/s  |
| 32K     | `turbo4` | none                 | 22.9 tok/s  |

`q8_0` halves the cache and `turbo4` quarters it, which is why `q8_0` is enough
to clear the spill at 16K but not at 32K. Needle retrieval stays exact under
`turbo4` at 32K. The rule of thumb: if `prepare` reports dense blocks on CPU,
spend KV precision to buy them back before anything else.

Qwen sampling with `top_k <= 32` reduces candidates on the GPU by default.
`sampling_gpu_topk_*`, `sampling_full_download_bytes`, and
`sampling_nanoseconds` expose its behavior; set `COLIBRI_SAMPLING_GPU_TOPK=0`
only when comparing against the full-vocabulary host fallback.

## Testing

The default suite builds synthetic fixtures and does not require model weights:

~~~bash
ruff check src tests setup.py
mypy src/colibri_next
pytest -q
~~~

Set `COLIBRI_TEST_MODEL=/path/to/model.gguf` to opt into the real Qwen reference
tests. A configured model path that is missing or fails to load is treated as a
test failure; only an unset opt-in and an unavailable CUDA device are skipped.

## Current limitations

- CUDA is the only model-execution accelerator.
- Gemma 4 sampling, MTP, per-layer embeddings, and shared-KV tail layers are not
  implemented.
- Laguna has no MTP, and supports only the per-head attention gate, so the
  per-element gate the larger Laguna checkpoints use is rejected at load.
- Laguna prefill uses the warp-online attention kernel. The tensor-core prefill
  routines fold Qwen's per-channel sigmoid gate in themselves, so Laguna's
  per-head softplus gate cannot use them and it forgoes that long-context path.
- Laguna's pre-tokenizer classifies non-ASCII letters by Unicode block rather
  than by a full category table, so non-Latin prose can split differently from
  the reference tokenizer.
- IQ2_XS, IQ3_XXS and IQ4_XS routed experts have grouped GPU kernels; the other
  IQ formats always run on the CPU expert path. Laguna concentrates available
  expert-cache VRAM into a contiguous suffix of complete layers and pins every
  expert in those layers. This avoids the regressive partial-layer split while
  using the CPU path for earlier layers. Set `COLIBRI_LAGUNA_WHOLE_LAYERS=0` to
  restore per-expert placement for comparison, or to a positive integer to cap
  the number of complete GPU layers.
- Laguna prefill over IQ2_XS, IQ3_XXS or IQ4_XS experts uses the direct
  quantized 8-token CPU kernel by default instead of expanding expert rows to
  f32. Set `COLIBRI_PREFILL_DIRECT_QUANT=0` only for comparison; `=1` continues
  to opt other supported architectures into the same path.
- On AVX-512 hosts, IQ2_XS decode widens a complete 16-value scale group at a
  time and fuses the gate/up projections so they share each activation load.
  Set `COLIBRI_IQ_AVX512=0` to compare with the AVX2 kernel, or
  `COLIBRI_FUSED_MOE_GATE_UP=0` to disable only the automatic IQ2_XS fusion.
- IQ expert decode is sensitive to memory bandwidth, clock sharing and thread
  placement. The default uses physical cores; tune `--cpu-threads` for the
  machine rather than assuming SMT helps (14 workers beat 8, 16 and 32 on the
  reference 16-core Laguna host).
- Qwen sampled decoding currently transfers the vocabulary logits to the host.
- Dynamic MoE routing still has host synchronization points.
- Full-layer CUDA graph replay and persistent fused layer kernels are incomplete.
- Image, audio, embedding, fine-tuning, and hosted-tool APIs are out of scope.
- Response records and prompt caches are process-local.

## Architecture

- `native/src/v2_runtime.cpp`: GGUF parsing, memory planning, scheduling, model
  orchestration, prefix reuse, and native runtime ABI
- `native/src/gpu_driver.cpp`: CUDA driver, NVRTC, cuBLAS/cuBLASLt, graph, and
  transfer integration
- `native/include/colibri_v2_qwen_kernels.hpp`: generated CUDA model kernels
- `src/colibri_next/v2.py`: Python bindings for the native ABI
- `src/colibri_next/v2_server.py`: tokenizer, cooperative engine thread, and
  native inference service
- `src/colibri_next/server.py`: shared HTTP protocol implementation
- `src/colibri_next/runtime_benchmark.py`: benchmark capture and comparison

## License

Apache-2.0.
