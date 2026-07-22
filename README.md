# Colibrì Next

Colibrì Next is a hardware-adaptive sparse Mixture-of-Experts inference prototype.

~~~text
storage -> RAM -> pinned RAM -> accelerator memory
                         |
                         v
             CPU / CUDA / Metal / Vulkan / HIP
~~~

The runtime can inspect and convert real Qwen checkpoints and generate decoded text through mixed Qwen3.5/Qwen3.6 decoder stacks.

## Implemented

- Hardware probing and portable placement profiles
- SSD, RAM, VRAM, and unified-memory placement planning
- Per-layer expert LRU, pinning, prefetch, and route prediction
- Dependency-free safetensors metadata and byte-slice reader
- Qwen checkpoint inspection and manifests
- Direct BF16-to-Q4 routed-expert conversion
- Q4 shared-expert conversion
- BF16 router, shared gate, and RMSNorm tensor support
- Normalized softmax top-k routing
- Weighted routed-expert aggregation
- Sigmoid-gated shared SwiGLU expert
- Qwen RMSNorm and residual MoE path
- BF16 full-attention projection conversion
- Partial RoPE and grouped-query KV caching
- Query output gating and attention residual path
- Depthwise causal convolution and recurrent Gated DeltaNet state
- Delta-rule updates, learned decay, gated RMSNorm, and residual path
- Mixed decoder-layer assembly and persistent sequence state
- Token embeddings, final RMSNorm, and chunked BF16 LM head
- End-to-end token-ID-to-logits execution
- Hugging Face tokenizer JSON integration and ChatML formatting
- Greedy, top-k, and nucleus sampling
- Stateful prompt-to-text generation
- Bounded cross-request conversation prefix-state caching
- Persistent OpenAI-compatible local HTTP server
- Incremental Chat Completions SSE streaming
- Streaming and stateful Responses API text generation
- Legacy text completions and tokenizer endpoints
- Native Qwen function calling with OpenAI tool-call responses
- Optional bearer authentication and browser CORS
- Packed BF16 and Q4 CUDA matrix-vector kernels
- Expert-major CUDA prompt batching with one shared-expert batch
- Bounded LRU VRAM weight cache with CPU fallback
- NumPy and portable Python execution paths
- Checksummed versioned tensor containers

## Installation

Colibrì Next supports Windows 10/11 and x86-64 Linux with Python 3.11 or
newer. The native CPU backend needs CMake 3.24+ and a C++20 compiler. CUDA is
optional and requires an NVIDIA driver plus the CuPy package matching the
installed CUDA major version.

### Windows PowerShell

~~~powershell
$env_dir = ".venv"
python -m venv $env_dir
& "$env_dir\Scripts\Activate.ps1"
python -m pip install --upgrade pip
$env:PYTHONPATH = "src"
pip install -e ".[fast,generation]"
python -m colibri_next.native_build
python -m unittest discover -s tests -v
~~~

Install Visual Studio 2022 Build Tools with the **Desktop development with
C++** workload if `native_build` cannot find a compiler.

### Linux Bash

On Ubuntu or Debian, install the system build prerequisites first:

~~~bash
sudo apt update
sudo apt install -y python3 python3-venv python3-dev build-essential cmake

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
export PYTHONPATH=src
pip install -e ".[fast,generation]"
python -m colibri_next.native_build
python -m unittest discover -s tests -v
~~~

The commands below use forward-slash paths under a local `models` directory;
Python accepts this form on both Windows and Linux. Replace the paths with the
locations of your checkpoints if needed.

For CUDA acceleration, add the CuPy package matching the installed CUDA
toolkit. This command is the same in PowerShell and Bash:

~~~console
# CUDA 13
pip install -e ".[cuda13]"

# CUDA 12 (use instead of the cuda13 extra)
pip install cupy-cuda12x
~~~

NumPy is optional but strongly recommended for conversion and CPU execution.
`native_build` produces `colibri_native.dll` on Windows and
`colibri_native.so` on Linux, compiling runtime-dispatched scalar, AVX2, and
AVX-512 Q4 CPU kernels. Portable Python execution remains available when the
native library or CuPy is unavailable.

## Hardware planning

~~~console
python -m colibri_next.cli inspect-hardware --save hardware-profile.json
python -m colibri_next.cli plan --hardware hardware-profile.json --model qwen3.6-35b-a3b --context 32768 --expert-bits 4 --save qwen36-placement.json
~~~

The detected 64 GB RAM and 12 GB VRAM profile places static tensors in VRAM, Q4 routed experts in RAM, active experts in a VRAM cache, and runtime state in VRAM.

## Qwen conversion

~~~console
python -m colibri_next.cli inspect-model models/Qwen3.6-35B-A3B

python -m colibri_next.cli convert-qwen models/Qwen3.6-35B-A3B models/colibri-qwen36 --extract-experts --quantization q4

python -m colibri_next.cli convert-moe-layers models/Qwen3.6-35B-A3B models/colibri-qwen36

python -m colibri_next.cli convert-attention-layers models/Qwen3.6-35B-A3B models/colibri-qwen36 --quantization q4

python -m colibri_next.cli convert-linear-layers models/Qwen3.6-35B-A3B models/colibri-qwen36 --quantization q4

python -m colibri_next.cli convert-model-io models/Qwen3.6-35B-A3B models/colibri-qwen36 --quantization q4

python -m colibri_next.cli convert-tokenizer models/Qwen3.6-35B-A3B models/colibri-qwen36
~~~

The first conversion creates independently loadable routed experts. The second converts every layer's router, shared expert, shared-expert gate, and post-attention RMSNorm. The remaining commands convert both token-mixer types, model input/output tensors, and tokenizer assets. Static converters accept `--quantization bf16` for compatibility or `--quantization q4` to reduce projection, embedding, and LM-head storage and VRAM-cache pressure. Existing converted models can be upgraded in place by rerunning the three static conversion commands with `--quantization q4 --overwrite`.

Q4 uses one FP16 scale and 16 packed bytes for every 32 values. Routed experts require approximately 18.1 GB instead of roughly 64.4 GB BF16. The original checkpoint and Q4 output require about 90 GB combined before safety margin.

## Execution

Execute one routed expert:

~~~console
python -m colibri_next.cli benchmark-expert models/colibri-qwen36/experts/layer-000/expert-0000.coli --iterations 10
~~~

Execute one complete Qwen MoE feed-forward block:

~~~console
python -m colibri_next.cli benchmark-moe-layer models/colibri-qwen36 --layer 0 --iterations 3

# Also apply post-attention RMSNorm and the residual connection.
python -m colibri_next.cli benchmark-moe-layer models/colibri-qwen36 --layer 0 --iterations 3 --residual
~~~

The complete block performs the reference sequence:

1. BF16 router matrix-vector multiplication
2. Float32 softmax over all experts
3. Top-k selection and selected-probability renormalization
4. Q4 SwiGLU execution for selected experts
5. Routing-weighted expert aggregation
6. Q4 shared SwiGLU execution
7. Sigmoid shared-expert gating
8. Routed plus shared output

The residual mode first applies Qwen's one-centered RMSNorm and then adds the original hidden state.

Execute incremental full attention with a growing grouped-query KV cache:

~~~console
python -m colibri_next.cli benchmark-attention-layer models/colibri-qwen36 --layer 3 --tokens 8
~~~

The attention path applies input RMSNorm, Q/K/V projection, per-head Q/K RMSNorm, partial RoPE, grouped-query causal attention, sigmoid query gating, output projection, and the residual connection.

Execute incremental Gated DeltaNet with persistent convolution and recurrent state:

~~~console
python -m colibri_next.cli benchmark-linear-layer models/colibri-qwen36 --layer 0 --tokens 8
~~~

The linear path applies input RMSNorm, projected depthwise causal convolution, Q/K L2 normalization, learned decay and delta-rule state updates, SiLU-gated RMSNorm, output projection, and the residual connection.

Execute every converted decoder layer with persistent per-layer state:

~~~console
python -m colibri_next.cli benchmark-decoder models/colibri-qwen36 --tokens 1
~~~

The decoder stack selects each layer's configured token mixer, applies both residual paths, executes its sparse MoE block, and advances full-attention or recurrent state for the next token.

Execute real token IDs through embeddings, every decoder layer, final RMSNorm, and the LM head:

~~~console
python -m colibri_next.cli benchmark-logits models/colibri-qwen36 --token-ids 1,2,3 --device cuda --gpu-cache-mib 8192
~~~

CUDA keeps packed BF16 and Q4 weights in a bounded priority-aware VRAM cache and executes them without full dequantization. Routed and shared experts use grouped gate, activation, and weighted-down kernels, attention projections are batched, and Gated DeltaNet recurrent updates remain on the GPU. During generation, decoder hidden states, RMSNorm, residuals, routing probabilities, LM-head logits, and sampling stay device-resident; only selected expert IDs and the sampled token cross to the host. Evicted expert allocations are reused through CuPy's memory pool instead of forcing allocator-wide synchronization. Omit `--device cuda` for the portable CPU path; lower `--gpu-cache-mib` on smaller GPUs.

The generator skips unnecessary vocabulary projections during prompt prefill and only evaluates the LM head for the final prompt token. CUDA full attention keeps RoPE, KV-cache updates, grouped-query attention, gating, and residual execution on the GPU. Fused RMSNorm and routing kernels reduce launch count. Layer-major CUDA prefill batches static projections and routing across prompt tokens and synchronizes selected expert IDs once per layer instead of once per token and layer. In local RTX 5070 Ti Laptop GPU benchmarks using Qwen3.5-35B-A3B with experts preloaded into RAM, the original runtime produced 1.10 output tokens per second, grouped CUDA reached 4.27, device-resident decode reached 4.80, Q4 static weights reached 6.56, and fused normalization/routing reached 6.92 output tokens per second when total response time includes prompt processing. The corrected warmed decode harness measures about 16-18 generated tokens per second with GPU sampling included. A warmed 256-token prefill improved from 13.75 to 18.05 tokens per second after batched routing. Q4 static attention, DeltaNet, embeddings, and LM-head files occupy about 1.24 GiB, protected CUDA weights use about 1.08 GiB, and short measured runs had no weight-cache evictions. Treat these single-run numbers as directional; prompt length, early EOS, storage speed, power limits, and available VRAM materially affect throughput.

Expert-major MoE dispatch groups prompt assignments by routed expert and runs the
shared expert over the complete token batch. On the same RTX 5070 Ti, warmed
256-token Q4 prefill improved from 37.93 to 39.25 tokens per second (about
3.5%), while 64-token prefill regressed from 47.89 to 43.34. The runtime
therefore uses tokenwise grouped dispatch below 128 tokens and expert-major
dispatch at or above 128 tokens. Set `COLIBRI_EXPERT_MAJOR_PREFILL` to `0` or
`1` to force either path, or tune the crossover with
`COLIBRI_EXPERT_MAJOR_MIN_TOKENS`.

DeltaNet prompt processing uses a fused causal-convolution sequence kernel and
one persistent recurrent-scan block per value head, replacing the per-token
CuPy launch loop. A focused 256-token layer-0 benchmark on the same GPU and
BF16-static validation model improved from 2,357 to 7,952 tokens per second
(about 3.37x). Set `COLIBRI_FUSED_DELTA_PREFILL=0` to use the fallback path for
parity checks or A/B benchmarks.

Pinned-memory expert upload and a request-local cross-layer transition predictor
are available experimentally. Enable them with
`$env:COLIBRI_EXPERT_PREFETCH = "1"` in PowerShell or
`export COLIBRI_EXPERT_PREFETCH=1` in Bash.
`COLIBRI_EXPERT_PREFETCH_BUDGET` controls the number of predicted experts and
defaults to `2`. The CUDA health statistics report requests, cache hits, waits,
useful prefetched tensors, and transferred MiB. This remains opt-in because a
forced 2 GiB cache benchmark showed no cold-decode improvement and a small
warm-decode regression from cache pollution, while the normal 8 GiB
short-prompt run already had all decode-used experts resident.

Generate decoded text from a chat prompt:

~~~console
python -m colibri_next.cli generate-text models/colibri-qwen36 --prompt "Say hi." --max-new-tokens 8 --temperature 0 --device cuda --gpu-cache-mib 8192 --cpu-moe-layers 0
~~~

Temperature zero performs greedy decoding. Positive temperatures support top-k and nucleus sampling through `--top-k`, `--top-p`, and `--seed`.

## Transformers validation

Install the optional reference dependencies and compare a converted model with its
original Hugging Face checkpoint using identical token IDs:

~~~console
pip install -e ".[validation]"
python -m colibri_next.cli validate-transformers models/colibri-qwen36 models/Qwen3.6-35B-A3B --token-ids 1,2,3 --generate-tokens 4 --device cuda --gpu-cache-mib 8192 --reference-device cuda --reference-dtype bfloat16
~~~

When the reference checkpoint does not fit in one GPU, use
`--reference-device auto --reference-offload-dir models/transformers-offload`
to let Accelerate place weights across GPU, RAM, and disk. The offload directory
must have enough free space for the portion of the checkpoint that does not fit
in accelerator and system memory. Use `--reference-gpu-memory-mib` and
`--reference-cpu-memory-mib` to reserve capacity for Colibrì and the operating
system.

Add `--layerwise` with exactly one token ID to compare the embedding, decoder
hidden states, and final normalization. This locates the first divergent layer
after an end-to-end parity failure. Use `--component-layer N` to split a
full-attention layer into normalization, token-mixer, MoE, and residual stages.

The validator compares the complete next-token logit vectors, top-k membership,
cosine similarity, and greedy token at every step. Generated steps are
teacher-forced with the Transformers greedy token, so a mismatch does not put the
two runtimes on different contexts. Use `--trust-remote-code` only for checkpoints
whose repository code you have reviewed.

In a local Qwen3.5-35B-A3B validation, a model with BF16 embeddings, attention,
DeltaNet, and LM head plus Q4 experts matched all five Transformers greedy
predictions for token `1` and four teacher-forced continuations. Logit cosine
similarity ranged from 0.9803 to 0.9968. Quantizing the static weights to Q4
changed the first greedy prediction and reduced cosine similarity to 0.7165.
Layerwise diagnostics traced the sharp loss to residual cancellation amplifying
accumulated quantization error. Treat Q4-static conversion as a throughput and
memory optimization that can alter generation; use BF16 static tensors when
reference fidelity is more important.

## Local server

Serve a GGUF directly with the native v2 C++/CUDA runtime and hybrid expert
execution:

~~~console
python -m colibri_next.cli serve-v2 models/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf --context-window 32768 --gpu-cache-mib 8192 --moe-device hybrid --host 127.0.0.1 --port 8000
~~~

`serve-v2` keeps model execution, prompt ingestion, and the greedy token loop
inside the native runtime. Python handles HTTP compatibility and receives
generated tokens through the C ABI streaming callback. Native v2 currently
supports greedy requests (`temperature: 0`) and serializes requests through
one persistent model session.

The converted-model v1 server remains available as a reference backend:

Start one persistent model process with an 8 GiB CUDA weight cache:

~~~console
python -m colibri_next.cli serve models/colibri-qwen35 --device cuda --gpu-cache-mib 8192 --cpu-moe-layers 0 --context-window 32768 --host 127.0.0.1 --port 8000
~~~

`--context-window` limits formatted input plus generated output. The browser
reads this value from `/props` and no longer imposes a separate 4096-token cap.
For example, a 32,768-token window with a 10,000-token conversation leaves at
most 22,768 tokens for generation. Use `--max-new-tokens` only when you want an
additional output ceiling below the total context window. Larger contexts grow
full-attention KV storage and keep more recurrent/conversation state resident.

The server defaults to `--expert-preload auto`. It preloads routed Q4 experts when available RAM can hold them while preserving an 8 GiB reserve, and otherwise continues with on-demand SSD loading. The tested Qwen3.5-35B-A3B conversion uses about 16.9 GiB for routed experts and took about 48.5 seconds to preload from SSD. Use `--expert-preload none` for faster startup or `--expert-preload all` to force preloading. The one-off `generate-text` command defaults to `none` because preload startup is usually not worthwhile for a single short response.

`--cpu-moe-layers N` runs the first N MoE blocks through the fused native Q4 CPU backend while attention, DeltaNet, and the remaining MoE blocks stay on CUDA. When the routed experts do not fit in VRAM (the tested 35B-A3B conversion needs 16.9 GiB), offloading every MoE block measured about 2x faster decode than streaming experts to the 12 GiB RTX 5070 Ti (`--cpu-moe-layers 40`: ~20 tok/s vs ~10 tok/s at `0`) — provided the experts are preloaded into RAM. Combine it with `--expert-preload all` (or `auto` with enough RAM), because lazy per-file expert loading otherwise dominates decode. The fused kernel threads across physical cores; an explicit `OMP_NUM_THREADS` overrides the default team size. The active native instruction set and CPU layer count are reported by the health endpoint.

Open `http://127.0.0.1:8000/` for the bundled browser chat UI. Conversations and generation settings are stored locally in the browser. If bearer authentication is enabled, enter the API key in the UI Settings panel.

Send an OpenAI-compatible chat completion request from another terminal:

Windows PowerShell:

~~~powershell
$body = @{
  model = "colibri-qwen35"
  messages = @(@{ role = "user"; content = "Say hi." })
  max_tokens = 8
  temperature = 0
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/chat/completions -ContentType application/json -Body $body
~~~

Linux Bash:

~~~bash
curl http://127.0.0.1:8000/v1/chat/completions \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "colibri-qwen35",
    "messages": [{"role": "user", "content": "Say hi."}],
    "max_tokens": 8,
    "temperature": 0
  }'
~~~

The server supports streaming and non-streaming `POST /v1/chat/completions`, `POST /v1/responses`, and legacy `POST /v1/completions`; stateful Responses continuation, retrieval, and deletion; Chat Completions and Responses function tools using Qwen's native tool format; `GET /v1/models`; and llama.cpp-style `/tokenize`, `/detokenize`, `/props`, and `/slots` endpoints. It keeps the CUDA cache warm and serializes generation on one GPU. The first request compiles CUDA kernels and fills VRAM, so later requests are faster.

The active attention KV cache uses float32 by default. For lower VRAM usage,
start the server with `--kv-cache-type q8`. This stores each
key/value head and token as int8 with a separate scale and dequantizes it for
attention. Float32 remains the safer default; compare a short generation
before enabling q8 for production workloads.

Completed generations retain a bounded decoder prefix state. When a later chat
request begins with the same tokenized history, the server reuses its attention
KV and DeltaNet recurrent state and prefills only newly appended tokens. The
default cache holds four conversation branches; set
`COLIBRI_PREFIX_CACHE_ENTRIES=0` to disable it or another non-negative value to
change the capacity. `/health` reports entries, hits, misses, evictions, and
reused tokens. A local two-turn Qwen3.5-35B-A3B smoke test reduced follow-up
latency from 1.474 to 0.648 seconds by reusing 19 of 40 prompt tokens. Results
depend on answer length, suffix size, expert residency, and cache pressure.

Optional bearer authentication can be enabled without putting the key in shell history:

Windows PowerShell:

~~~powershell
$env:COLIBRI_API_KEY = "replace-with-a-local-secret"
python -m colibri_next.cli serve models/colibri-qwen35 --device cuda --api-key $env:COLIBRI_API_KEY --cors-origin http://127.0.0.1:8000
~~~

Linux Bash:

~~~bash
export COLIBRI_API_KEY="replace-with-a-local-secret"
python -m colibri_next.cli serve models/colibri-qwen35 --device cuda --api-key "$COLIBRI_API_KEY" --cors-origin http://127.0.0.1:8000
~~~

The official OpenAI Python SDK can use the local server directly:

~~~console
pip install openai
~~~

~~~python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")

stream = client.chat.completions.create(
    model="colibri-qwen35",
    messages=[{"role": "user", "content": "Say hi."}],
    max_tokens=8,
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

response = client.responses.create(
    model="colibri-qwen35", input="Say hi.", max_output_tokens=8
)
print(response.output_text)
~~~
## Chat UI direction

The bundled UI is a same-origin static web application served by the Python process at `/`. It uses plain HTML, CSS, and browser-native JavaScript modules, keeping the installed server a single-command, offline-capable package with no Node.js runtime. The browser consumes Chat Completions SSE with `fetch`, `ReadableStream`, and `AbortController`; stores conversations and generation settings in `localStorage`; renders tool calls as structured cards; and polls `/health` for CUDA cache and busy-state indicators. Fenced HTML, SVG, and JavaScript code blocks include a Run button that renders the code below the block inside a sandboxed opaque-origin iframe with no access to the app origin, its storage, or the API.

## Architecture

QwenSafetensorCheckpoint validates checkpoint geometry and source shards.

QwenCheckpointConverter converts stacked routed-expert tensors into Q4 expert containers.

QwenMoELayerConverter creates one small layer container holding the BF16 router, shared gate, RMSNorm, and Q4 shared expert.

QwenMoELayer loads experts on demand and executes the official sparse MoE equations.

QwenAttentionConverter creates BF16 full-attention layer containers. QwenFullAttentionLayer executes one token at a time against AttentionKVCache.

QwenGatedDeltaConverter creates BF16 linear-layer containers. QwenGatedDeltaLayer executes one token at a time against GatedDeltaState.

QwenDecoderLayer combines one token mixer with its MoE block. QwenDecoderStack runs all configured layers against DecoderState.

QwenModelIO provides embedding lookup, final RMSNorm, and chunked vocabulary projection. QwenForCausalLM maps token IDs to logits while advancing decoder state.

HuggingFaceTokenizer loads copied tokenizer JSON assets. TextGenerator formats ChatML prompts, samples logits, advances sequence state, and decodes generated IDs.

LayeredExpertCache, ResidencyManager, TransitionPredictor, and PlacementPlanner provide the storage hierarchy. The current QwenMoELayer has a local expert cache; connecting it directly to ResidencyManager is a later integration step.

## Current limitations

- Dynamic expert selection still synchronizes selected IDs to the host once per MoE layer
- Routers, shared gates, normalization vectors, and recurrent parameters remain BF16 or F32
- Full CUDA graph replay and persistent fused layer kernels are not implemented
- CUDA is the only implemented accelerator backend
- Image, audio, embeddings, fine-tuning, and hosted OpenAI tools are not implemented
- Tool calls are buffered before streaming because the native XML call must be parsed as a complete unit
- DeltaNet recurrence is fused but inherently sequential within each value head;
  routed expert execution remains token-sequential below the expert-major threshold
- Response history and decoder prefix states are process-local; response records
  are limited to 128 and prefix states default to four entries

## Next milestones

1. Run and publish full-checkpoint Transformers parity results.
2. Move MoE routing and expert dispatch fully onto the GPU.
3. Improve expert prediction and reuse pinned staging buffers.
4. Add CUDA graph replay and persistent fused layer kernels.
5. Optimize Q8 activation and DP4A expert kernels after graph capture.
6. Refactor architecture-specific behavior behind model adapters.

## Scope

The project now executes complete mixed Qwen models from text prompts through decoded output. The current implementation supports portable CPU execution and optional bounded-memory NVIDIA CUDA acceleration.

## Native v2 runtime (opt-in)

The separate `colibri_v2` C++20 library memory-maps GGUF files and exposes
model metadata, tensor offsets, session lifetime, cancellation, deterministic
stepping, callbacks, and runtime statistics through a small C ABI. Python
bindings are available as `colibri_next.v2`.

Build it with `PYTHONPATH=src python -m colibri_next.native_build`, then use
`inspect-gguf-v2`, `probe-qwen-native-v2`, `benchmark-v2`, or `serve-v2`.
The Qwen3.6 MoE path executes direct GGUF weights through native C++/CUDA,
keeps recurrent and KV state on GPU, and supports GPU, CPU, or hybrid routed
expert execution. v1 remains available as the converted-model reference and
fallback backend.

Native v2 also reads GGUF's generic `attention.sliding_window` and
`attention.sliding_window_pattern` metadata. Sliding-attention layers use a
compact circular KV cache sized to the trained window plus one prefill batch;
global layers retain the full context cache. `serve-v2 --swa-full` keeps
full-size storage for sliding layers when unrestricted prefix-cache rollback is
more important than VRAM savings.

Gemma 4 QAT MoE text GGUFs without per-layer embeddings or shared-KV tail
layers are supported by the native CUDA runtime, including mixed
local/global head geometry, proportional global RoPE, optional K=V global
attention, QAT Q4_0 projections, dense GEGLU, and routed MoE layers. Routed
Gemma 4 experts support CPU execution (`moe_device="cpu"`) or a bounded hybrid
cache (`moe_device="hybrid"`): resident routed experts execute through grouped
Q4_0 CUDA kernels while misses execute on CPU and are admitted for later
tokens. Persistent attention, dense, router, and embedding weights remain on
GPU. Use `model.native_runtime(...)` for architecture-neutral construction.

For example, a 4 GiB total CUDA budget enables a bounded expert cache while
keeping the remaining Gemma 4 experts in host memory:

```bash
PYTHONPATH=src python -m colibri_next.cli serve-v2 model.gguf \
  --moe-device hybrid --gpu-cache-mib 4096 --context-window 32768
```

`--parallel N` allocates an independent Gemma 4 KV slot for each concurrent
conversation. Requests are interleaved safely through the cooperative engine;
Gemma slots currently decode sequentially rather than using Qwen's
architecture-specific layer-overlap driver. With `--parallel 2` or higher,
`--prompt-cache-mib` can spill inactive slots to host RAM for later reuse.
Each slot multiplies KV memory, so keep compact SWA enabled at long contexts;
`--swa-full` with multiple 58K-token slots generally exceeds consumer VRAM.

For native Qwen performance work, `benchmark-v2` measures production batched
prefill separately from steady single-token decode and reports route, expert
paging, CPU expert, and cache-hit counters. KV precision and expert policy can
be reproduced explicitly, for example:

```bash
PYTHONPATH=src python -m colibri_next.cli benchmark-v2 model.gguf \
  --prompt "Explain sliding-window attention." --chat \
  --moe-device hybrid --cache-type-k f16 --cache-type-v f16
```

`--prefill-cache-seed N` experimentally records Qwen routing frequency during
prefill without admitting pages, then bulk-loads the hottest `N` experts per
layer before generation. `COLIBRI_PREFILL_CACHE_SEED` can override the API/CLI
setting for experiments. It is opt-in while its time-to-first-token versus
early-generation tradeoff is evaluated across hardware and prompt workloads.

Hybrid MoE uses `--expert-paging auto` by default. When the CUDA driver supports
registered host memory and the machine has enough available RAM for the model
plus a safety margin, auto mode pages experts directly from the mapped GGUF and
avoids an extra CPU copy. Registration adds one-time startup latency but is
amortized by a long-running server. Use `--expert-paging staged` on constrained
machines, or `--expert-paging direct` to require the faster path and fail if the
driver cannot register it. Runtime diagnostics report `direct_paging`,
`paging_registration_nanoseconds`, and the detected `host_available_bytes`.

On x86 hosts, hybrid Qwen expert execution dispatches at runtime to AVX-512,
AVX2, or the scalar compatibility backend. AVX2-only machines therefore keep
vectorized Q5_K, Q6_K, and Q8_0 expert kernels instead of falling back to
scalar math. Developers can set `COLIBRI_CPU_BACKEND=scalar|avx2|avx512` before
process startup to reproduce backend-specific correctness and performance;
CPUID masking prevents selecting instructions unsupported by the host.

For kernel-level diagnosis, developers can set `COLIBRI_CUDA_PROFILE=1` before
starting a native Qwen process. Each single-token decode then reports CUDA
event timings for DeltaNet or attention work before MoE, the shared expert,
the routed-expert pipeline, and the final LM head. The profiler is disabled by
default and allocates no timing events in normal server or benchmark runs.
