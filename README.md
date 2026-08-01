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
PYTHONPATH=src python -m colibri_next.cli serve-v2 model.gguf \
  --context-window 32768 \
  --expert-mode auto \
  --cache-type-k f16 --cache-type-v f16 \
  --host 127.0.0.1 --port 8000
~~~

`serve` is an alias for `serve-v2`. Open `http://127.0.0.1:8000/` for the local
chat UI.

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

## Inspect and generate

~~~bash
PYTHONPATH=src python -m colibri_next.cli inspect-gguf-v2 model.gguf

PYTHONPATH=src python -m colibri_next.cli generate model.gguf \
  --prompt "Explain mixture-of-experts routing." \
  --max-new-tokens 128 --temperature 0
~~~

## Benchmarking

The direct benchmark separates preparation, prompt prefill, and steady decode:

~~~bash
PYTHONPATH=src python -m colibri_next.cli benchmark-v2 model.gguf \
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

- `--gpu-cache-mib 0`: size allocations from currently free VRAM
- `--cache-type-k` / `--cache-type-v`: KV precision
- `--mtp-drafts N`: enable supported Qwen MTP verification
- `--parallel N`: independent sequence slots
- `--prompt-cache-mib N`: host cache for spilled sequence state
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
  IQ formats do not and always run on the CPU expert path. Even where the
  kernels exist, IQ experts stay on the CPU unless `--prefill-cache-seed` pins a
  seeded set: splitting a layer between a GPU router and host experts costs a
  round trip per layer, and on a 12 GiB card that synchronization grows by more
  than the offload saves. Whole-layer residency, not per-expert paging, is what
  would make the offload pay.
- IQ expert decode is ALU-bound rather than bandwidth-bound, so it scales past
  the physical core count. The default thread heuristic uses physical cores;
  `--cpu-threads` at the SMT count is worth roughly 1.4x on such models.
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
