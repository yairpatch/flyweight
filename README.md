# Flyweight

Flyweight is a native C++/CUDA GGUF inference runtime. Python provides the
CLI, tokenizer-facing server adapter, and OpenAI/Anthropic-compatible HTTP
API; model execution stays in the native runtime.

Served model families:

| Family | Formats | Notes |
| --- | --- | --- |
| Qwen 3 / 3.5 / 3.6, dense and MoE | GGUF; safetensors (3.5 family) | Full feature set: MTP, expert offload, prefill pipeline. Image input on the 3.5 family through a llama.cpp `mmproj` GGUF (`--mmproj`); the safetensors loader still drops the tower |
| Laguna 2.1 | GGUF | Per-head attention gate only; no MTP |
| K2-Horizon (dense and MoVA 36B-A4B) | GGUF | Grouped RMS norms, softplus attention gate, DeepSeek-shaped MoE; the MoVA value experts page through their own device cache with a persisted routing history; no MTP |
| Muse Glimmer | GGUF | Channel-tagged reasoning; no speculative decoding (it has no in-model MTP heads, and its separate DFlash drafter is not wired up) |
| DeepSeek-V4 / V4-Flash | GGUF (split) | Dedicated CPU/hybrid runtime with half-precision caches; DSpark speculative drafts via `--mtp-model` |
| Gemma 4 | GGUF | Sampling, penalties and the tool grammar all work; no MTP, expert placement `cpu`/`hybrid` only -- see limitations |
| BailingMoE3 | GGUF, safetensors | Independent sequence slots with snapshot prefix reuse across conversations; a GGUF conversion answers exactly as the checkpoint it came from |
| Qwen3.8-Flash-Next (qwen4exp) | GGUF (split) | Qwen4-preview hybrid: gated-residual streams, hashed n-gram embeddings (host-side table), DeltaNet + gated attention. Sparse attention runs dense by default; MTP needs a release with a draft block or the standalone MTP file via `--mtp-model`; no vision -- see limitations |

A safetensors checkpoint (Qwen 3.5 family and BailingMoE3) is packed to a
chosen quantization on first open and cached beside the checkpoint --
weighted by a llama.cpp `imatrix.dat` when one is present; everything else
loads from GGUF, including multi-file `-00001-of-0000N` splits.

## Features

- Memory-mapped GGUF loading, including split archives and metadata-only
  first shards
- Native CUDA attention, DeltaNet, dense FFN, and sparse MoE execution, with
  CUDA-graph replay for decode
- CPU, automatic hybrid, and strict resident expert placement
- Prefill pipeline: routed experts stream to the GPU behind a byte budget and
  run the dense batch kernels, with CPU experts overlapped under queued GPU
  work (default on)
- F32, F16, BF16, Q8_0, Turbo3, and Turbo4 KV caches
- Sliding-window attention and compact circular KV storage
- Multi-token prediction for Qwen checkpoints, under sampling, penalties and
  the tool grammar alike (each verified row goes through the request's own
  sampler, so a drafting request answers exactly as a non-drafting one);
  DSpark draft speculation for
  DeepSeek-V4-Flash
- Independent sequence slots and host-backed prompt-cache spill/restore
- Cooperative concurrent request scheduling; sampled and greedy requests
  decode in the same batch
- Sampler-enforced tool-call grammar (declared names, required parameters,
  well-formed JSON values) and sampler-enforced JSON response mode
- Image input for Qwen 3.5-family checkpoints: the mmproj vision tower runs
  natively, images take part in prefix reuse, and OpenAI `image_url`,
  Responses `input_image` and Anthropic `image` parts are all accepted
- Thinking controls: per-request effort for checkpoints that grade their
  reasoning, and a hard thinking-token budget the sampler cannot overrun
- OpenAI Chat Completions, Responses, and legacy Completions APIs
- Anthropic Messages with thinking blocks, plus token-count endpoints
- Streaming SSE (including incremental tool-call arguments), bearer
  authentication, CORS, and a bundled chat UI with a sandboxed preview
- Repeatable JSONL runtime benchmark and regression comparison harness

## Requirements

Wheels are published to PyPI as **`flyweight-llm`** for Linux x86-64
(manylinux 2.28) and Windows x64 -- the import package and the command are
still `flyweight`:

~~~bash
pip install flyweight-llm
flyweight doctor
~~~

Everywhere else, and for a checkout, Flyweight compiles its native runtime from
source as part of the install, so a C++ toolchain is needed once, at install
time.

| | Needed |
| --- | --- |
| Python | 3.11 or newer, 64-bit |
| CMake | 3.24 or newer |
| Compiler | MSVC v143 (Windows) or GCC 13+ / Clang 16+ (Linux) |
| GPU | A current NVIDIA driver, the NVRTC library, and the CUDA headers -- **no `nvcc`**, and nothing CUDA is linked at build time. `--backend cpu` serves without a GPU at all |
| Disk | a few tens of MB for the build tree, plus whatever the model weighs |

CUDA kernels are compiled at runtime by NVRTC through the driver API, so the
build itself needs no CUDA toolkit and `nvcc` is never invoked. At serve time
the runtime `dlopen`s `libcuda` and `libnvrtc` (`nvrtc64_*.dll` on Windows)
and hands NVRTC the toolkit headers (`cuda_fp16.h`, CUB), which it looks for
under `CUDA_PATH`, `CUDA_HOME`, `/opt/cuda` and `/usr/local/cuda`. A distro
`cuda` package or the toolkit installer provides both; when either half is
missing, the runtime says which one and falls back to `--backend cpu`. cuBLAS
is optional and only used when present. CuPy is never required: the runtime
only probes it, if installed, as one more place to find the headers.

## Installation

Pick your platform below. Both end at `flyweight doctor`, which reports whether
the machine can serve and names the fix for anything missing.

Expect the install to take a few minutes: the runtime is a few dozen large
AVX-512 and kernel translation units, and they are compiled, not downloaded.

### Windows

Run this in PowerShell. Nothing here needs a Developer Command Prompt — the
build locates the x64 MSVC toolchain itself, through the same `vswhere` lookup
`flyweight doctor` reports.

~~~powershell
# One-time prerequisites.
winget install Kitware.CMake
winget install Ninja-build.Ninja
winget install --id Microsoft.VisualStudio.2022.BuildTools --override `
  "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

# Close and reopen PowerShell so the new tools are on PATH, then:
git clone https://github.com/yairpatch/flyweight
cd flyweight
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install .
flyweight doctor
~~~

Notes specific to Windows:

- **Any Visual Studio edition works** — Community, Professional, Enterprise, or
  the standalone Build Tools, including installs on a non-system drive. Only
  the x64 C++ compiler component matters, so an existing Visual Studio with
  **Desktop development with C++** already ticked needs no `winget` line.
- **Ninja is optional but worth installing.** Without it the build falls back to
  NMake, which compiles one file at a time regardless of `--parallel`. Visual
  Studio ships its own `ninja.exe` and that copy is found automatically, so the
  `winget install Ninja-build.Ninja` line only matters if it is absent.
- **Use 64-bit Python.** The runtime library is x64; a 32-bit interpreter fails
  to load it with `WinError 193`.
- **If the build says the C++ tools were not found**, run
  `PYTHONPATH=src python -m flyweight doctor` from the checkout (the console
  script does not exist yet when the install failed). It names which of CMake,
  MSVC, and the build tool it can and cannot see, instead of stopping at the
  first one.

### Linux

~~~bash
# Debian / Ubuntu -- one-time prerequisites.
sudo apt install git python3-venv python3-pip build-essential cmake ninja-build
# Fedora / RHEL:  sudo dnf install git python3-devel gcc-c++ cmake ninja-build
# Arch:           sudo pacman -S git python gcc cmake ninja

git clone https://github.com/yairpatch/flyweight
cd flyweight
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install .
flyweight doctor
~~~

Notes specific to Linux:

- **Check the compiler version if the build fails on unknown syntax.** The
  runtime is C++20 and needs GCC 13+ or Clang 16+; `g++ --version` settles it.
  Older LTS releases ship GCC 11 or 12, where `sudo apt install g++-13` and
  `export CXX=g++-13` before `pip install .` is the smallest fix.
- **CMake older than 3.24** is the other common blocker on long-term releases.
  `pip install cmake` inside the activated venv puts a current one on PATH
  without touching the system package.
- **A GPU needs the proprietary NVIDIA driver plus NVRTC and the CUDA
  headers.** `nvidia-smi` reporting a device covers the driver; the distro
  `cuda` package (Arch) or `cuda-toolkit` (Debian/Ubuntu, Fedora) covers the
  rest, and `flyweight doctor` shows whether the runtime can see both. Without
  a GPU, serve with `--backend cpu`.

### Verifying the install

~~~
$ flyweight doctor
[ok  ] python: 3.12.10
[ok  ] jinja2: 3.1.4
[ok  ] package: .../.venv/lib/python3.12/site-packages/flyweight
[ok  ] command: .../.venv/bin/flyweight
[ok  ] native runtime: flyweight_v2.so, built 2026-08-31 14:04
[warn] sources: no checkout beside this install
       -> fine for a wheel; you cannot rebuild the runtime here
[ok  ] nvidia gpu: device 0, compute 12.0, 10.8/11.9 GiB free

this install can serve
~~~

Read the last line first. Only `FAIL` lines are blocking (and make `doctor`
exit non-zero); each one prints the command that fixes it where there is one.
`warn` lines are notes. The `sources` warning
above is the normal state of an installed copy — it means the runtime cannot be
rebuilt from there, which only matters if you intend to change it. Run
`flyweight doctor` first whenever something refuses to start.

### If `flyweight` is "not recognized" or "command not found"

The install succeeded and the console script simply is not on PATH. Run it as a
module instead — identical arguments, no PATH entry required:

~~~
python -m flyweight doctor
python -m flyweight serve model.gguf
~~~

Activating the virtual environment as shown above normally prevents this,
because activation puts the environment's script directory on PATH. It comes up
without one on a Microsoft Store Python, whose per-user
`...\LocalCache\local-packages\Python312\Scripts` is never added to PATH, and
after any `pip install --user`. Both cases make pip print a warning and install
anyway. `flyweight doctor` reports the exact directory to add if you would
rather fix PATH permanently.

### Reading the server log

Each request is one row, under a header that names the columns:

~~~
          endpoint   prompt  cached   ttft     out   tok/s  finish        total
10:49:45  chat        26.5k     98%   1.7s     112    35.8  tool call      4.9s
10:49:50  chat        26.7k     99%   1.1s     105    35.9  tool call      4.1s
11:30:01  chat           --      --     --      --      --  400         0.1s  prompt is too long: 70212 tokens > 65536 maximum
~~~

A failure fills the same columns (`--` where there is no number) and puts its
reason after them, and a notice is marked and indented to the grid, so one
unusual line never pushes the rest out of alignment:

~~~
10:56:35  •        queued behind 2 request(s): 538 prompt tokens waiting for a KV slot
~~~

`prompt` is what the request rendered to, `cached` how much of it the prefix
cache served (a low number here on a conversation that only appended is what
a cache problem looks like), `ttft` the wait before the first token, `out`
and `tok/s` the answer and its decode rate, `finish` why generation stopped
(the finish reason for a non-streaming request; a stream shows the phase it
ended in, `tool call` or `thinking`, and `--` for a plain stop). On a
terminal the row of a running request is drawn live and rewritten in place, so
the numbers a reader is watching are the ones that commit; a redirected log
gets only the committed rows.

`--quiet` prints nothing but failures. `--verbose` adds the HTTP access log,
the prefill/decode split and the prefix-cache diagnostics (where a
conversation diverged from what was cached, and the text on each side).
Colour is used only on a terminal, and never when `NO_COLOR` is set or
`TERM=dumb`, so a redirected log stays plain and greppable.

### Running the tests

~~~bash
pip install -e '.[test]'
pytest                 # ~3 minutes: everything except the slow parity tests
pytest --run-slow      # all of it
pytest -n auto --run-slow   # in parallel; ~10 minutes on 32 threads
~~~

Four parity tests are 81% of the suite's wall time. Each builds a fixture,
loads the native runtime and generates the same tokens twice to compare them
bit for bit, and the worst runs that on the CPU backend, where every CUDA
kernel is emulated. They are marked `slow` and skipped unless asked for; CI
runs them on every push.

### Developing on Flyweight itself

Use an editable install, so edits to `src/flyweight` take effect without
reinstalling, and build the contract tests and benchmarks that a plain install
skips:

~~~bash
pip install -e .
python -m flyweight.native_build     # same build tree, plus the test binaries
ctest --test-dir build/native --output-on-failure
pytest -q
~~~

Do not keep an editable and a regular install in the same environment. The
regular one wins every import, edits appear to do nothing, and `flyweight
doctor` reports the shadowing on its `package` line.

The chat UI is a Vite + React + TypeScript app in `web/`; the build is
committed under `src/flyweight/ui/` so the Python package ships it without
Node. After changing anything in `web/`, rebuild and commit the output:

~~~bash
cd web
pnpm install
pnpm test          # vitest: protocol adapters, SSE reader, thinking tags,
                   # text direction, attachments, PDF/DOCX/XLSX extraction
pnpm build         # typecheck, then write src/flyweight/ui/
pnpm dev           # live-reload dev server proxying to :8000
~~~

## Commands

| Command | What it does |
| --- | --- |
| `flyweight serve MODEL` | serve the OpenAI/Anthropic APIs and chat UI |
| `flyweight generate MODEL --prompt TEXT` | print one response and exit |
| `flyweight benchmark MODEL` | measure prompt and decode speed as JSON |
| `flyweight inspect MODEL` | print model metadata as JSON |
| `flyweight imatrix MODEL --text FILE` | gather an importance matrix |
| `flyweight probe MODEL` | run a few tokens and dump runtime counters |
| `flyweight doctor` | check the install and name the fix for anything missing |
| `flyweight transcript-audit DIR` | explain a coding harness's failed edits from a request dump (see below) |

`MODEL` is a `.gguf` file or a safetensors checkpoint directory, for every
model command. `flyweight COMMAND --help` lists every option that command
accepts, grouped by what it does: the model, the command's own inputs (server,
request, workload), the backend, hardware placement, and advanced tuning. The
older `serve-v2`, `generate-text-v2`, `benchmark-v2`, `inspect-gguf`,
`probe-native` and `probe-native-v2` spellings remain accepted, and
`flyweight --version` prints the package version.

`transcript-audit` is for one question: when a coding harness's edit replaces
text that is not in the file, did the model edit blind, or did the read result
reach the server and get lost on the way to the model? With
`FLYWEIGHT_TRANSCRIPT_DUMP=DIR` set, `serve` writes one JSON file per request
holding both the transcript the client sent and the prompt the model saw
(`FLYWEIGHT_TRANSCRIPT_PROMPT=0` keeps a digest instead of the prompt text),
and the audit checks each edit against both, in order.

## Serve a model

~~~bash
flyweight serve model.gguf
~~~

Open `http://127.0.0.1:8000/` for the local chat UI. The defaults select the
backend and memory policy automatically; the options below are the ones worth
reaching for first:

~~~bash
flyweight serve model.gguf \
  --context 65536 --max-tokens 16384 \
  --host 127.0.0.1 --port 8000
~~~

Prompt caching is automatic: displaced conversations are packed into a
byte-budgeted host-RAM LRU and restored by longest matching prefix, including
when only one GPU sequence slot is configured. Use `--cache off` to disable it
or `--cache 4096` to set an explicit 4 GiB budget. Automatic mode uses one
eighth of currently available RAM, capped at 8 GiB.

The older `--context-window` and `--max-new-tokens` spellings of the limit
flags remain accepted on `serve` and `generate` for script compatibility.
`--model-name` sets the id the model answers to in the API and `/v1/models`,
and every sampling default (`--temperature`, `--top-k`, `--top-p`, `--min-p`,
`--repetition-penalty`, `--presence-penalty`, `--frequency-penalty`,
`--penalty-window`) can be set server-wide the same way.

The native expert modes are:

| Mode | Prompt routed experts | Decode routed experts | Behavior |
| --- | --- | --- | --- |
| `cpu` | CPU | CPU | Minimum GPU expert memory |
| `auto` | CPU | Stable hot set on GPU, misses on CPU | Default |
| `resident` | GPU | GPU | Fails preparation unless every expert fits |

Legacy `hybrid`, `gpu`, `legacy-hybrid`, and `legacy-paging` spellings remain
compatibility aliases for the old paging policies, and `--moe-device` is an
accepted alias of `--expert-mode`; new deployments should use the canonical
modes. On `--backend cpu` the expert mode is forced to `cpu`.

For concurrent agent clients, allocate independent sequence slots and optional
host prompt-cache storage:

~~~bash
flyweight serve model.gguf \
  --context 58000 \
  --expert-mode cpu --cpu-threads 12 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 2 --cache 4096 --cpu-prefetch-auto
~~~

Each sequence slot has its own KV and recurrent state. More slots improve
conversation isolation but consume additional VRAM; `--scratch-context` gives
the slots past the first a smaller context than the first one. Bound both
admitted inference work and open HTTP connections for public-facing
deployments:

~~~bash
flyweight serve model.gguf \
  --concurrency 8 --max-connections 64 \
  --request-timeout-seconds 30 --cors-origin https://app.example
~~~

### Chat UI

The bundled UI at `/` is a single-page app that reaches every function the
server exposes, not only chat:

- **Chat** streams through any of the three protocols (OpenAI chat
  completions, Anthropic messages, OpenAI responses), switchable per
  conversation from the composer. Thinking shows in a collapsible panel with
  its duration, and an **Answer now** button closes it early on the chat
  completions and Anthropic protocols; tool calls render as cards where you
  paste the result and continue. Files attach by the **Attach** button, paste,
  or drop: images when a vision tower is loaded, and PDF, Word, Excel, and
  text or code files always (see below). Markdown, GFM tables, KaTeX math,
  and highlighted code with copy, download, and a sandboxed **Run** for
  HTML/SVG/JS. Right-to-left text is detected per message and laid out
  accordingly, with code blocks kept left-to-right.
- **Settings** (Ctrl+,) cover every sampling knob the server accepts,
  stop sequences, seed, reasoning effort and budget, `preserve_thinking`,
  JSON mode and JSON schema output, `chat_template_kwargs`, and named
  presets (a preset stores the tool definitions too). Defaults track
  `/props` until you change something. A knob the selected protocol has no
  field for is greyed with a *not sent on ...* hint (Anthropic messages has
  no effort or `response_format`, Responses has no budget, and
  `chat_template_kwargs` exists on chat completions only).
- **Tools** defines function tools (import OpenAI or Anthropic definitions),
  `tool_choice`, and parallel calls; the sampler grammar enforces them.
- **Runtime** polls `/health`, `/props`, and `/slots` and charts decode
  throughput, GPU memory, KV and prefix cache, expert cache hit rate, MTP
  acceptance, the time breakdown, and grammar counters, with the full
  telemetry block and raw JSON one click away.
- **Tokenizer** drives `/tokenize` and `/detokenize` and counts the current
  conversation through both `/v1/messages/count_tokens` and
  `/v1/responses/input_tokens`.
- **Playground** streams raw prompts through `/v1/completions`.
- **Inspector** keeps the exact request body, every SSE frame, and a curl
  line for each request, and retrieves or deletes stored responses.

Conversations live in IndexedDB (history from the previous UI is imported
once), with search over message bodies, pin, rename, JSON import, and JSON or
Markdown export. Ctrl+K opens a command palette; Ctrl+B toggles the sidebar;
Ctrl+Shift+O starts a conversation; Esc stops a generation. The API key is
kept in session storage, so it lives as long as the tab. A model selector
appears in the top bar when `/v1/models` lists more than one.

Document attachments never touch the server as files: the browser extracts
them to text and places it ahead of the typed message in the user turn, as a
fenced block headed `Attached file: NAME`. PDFs contribute their text layer
page by page; a PDF with no text layer (a scan) is rendered to page images
instead when a vision tower is loaded, and refused when not. Word files
become Markdown with headings, lists and tables kept; spreadsheets become one
CSV block per sheet; anything that sniffs as UTF-8 text is taken as code or
prose. Each file is cut to a quarter of the server's context window, on a
line boundary, with a visible `[truncated: ...]` marker; up to 8 files of at
most 32 MB each go in one turn.

### Images

A Qwen 3.5-family GGUF serves images when its vision tower is attached.
The tower is the `mmproj-*.gguf` llama.cpp publishes beside the model
(projector type `qwen3vl_merger`); decoding needs Pillow, installed with
`pip install 'flyweight-llm[vision]'` (or `pip install '.[vision]'` from a checkout):

~~~bash
flyweight serve Qwen3.5-35B-A3B-Q6_K.gguf \
  --mmproj mmproj-Qwen3.5-35B-A3B-BF16.gguf --image-max-tokens 1024
~~~

Each image is resized so that both sides are multiples of 32 pixels, the
aspect ratio is kept, and it covers at most `--image-max-tokens` tokens (one
per 32x32 block; the default 1024 is about a megapixel). Image tokens count
as prompt tokens in `usage`, and an image that sits inside a reused prefix is
never encoded again. `--image-urls deny` refuses `http(s)` image URLs and
keeps `data:` URLs (a remote image is capped at 32 MiB and 20 seconds);
`/health` reports the tower under `execution.vision`, and the bundled chat
UI's **Attach** button accepts images (paste or drop works too) whenever it
does. Without a tower an image part degrades to a visible
`[image omitted: ...]` note in the prompt (`[unsupported image block
omitted]` for an Anthropic `image` block) rather than failing the request,
since the part sits in the client's history and would return on every retry.

## API

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

Image parts go where the OpenAI, Responses and Anthropic APIs put them:

~~~bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "What is in this picture?"}
    ]}]
  }'
~~~

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/responses` (with
retrieval and deletion by id; the 128 most recent are kept, `store: false`
skips a record), `/v1/models` and `/v1/models/{id}`, `/v1/me`, `/v1/messages`
and `/v1/messages/count_tokens` (Anthropic), `/v1/responses/input_tokens`,
`/tokenize`, `/detokenize`, `/health`, `/props`, and `/slots`. All generation
endpoints stream over SSE, and chat streams honour
`stream_options.include_usage`. Request bodies are capped at 16 MiB. Set
`FLYWEIGHT_API_KEY` or pass `--api-key` to require bearer authentication
(`Authorization: Bearer` or `x-api-key`); `--cors-origin` sets
`Access-Control-Allow-Origin` (default `*`). Use `--strict-model` when
request model IDs must exactly match the configured server model name.

Chat requests use the GGUF's `tokenizer.chat_template` when it is present;
the built-in architecture formatter is only a fallback for older files. If a
`generation_config.json` is stored beside the GGUF, its `temperature`,
`top_k`, `top_p`, `min_p`, the penalties, `max_new_tokens`, and `do_sample`
defaults are also loaded. Without one, the built-in defaults are llama.cpp's:
temperature 0.8, `top_k` 40, `top_p` 0.95, `min_p` 0.05, penalties off.
Precedence is request, then server flag, then `generation_config.json`, then
the built-in default. `GET /props`
reports the resolved defaults and their sources, and the bundled UI adopts
them until the user saves custom settings.

### Thinking controls

Reasoning models expose two knobs, one soft and one hard:

- `reasoning_effort` (`low` / `medium` / `high` / `xhigh`) is a template
  variable for checkpoints that grade their reasoning (Qwen3.5 reads it
  natively). It is read from the flat field, the Responses-style
  `reasoning.effort`, vLLM-style `chat_template_kwargs.reasoning_effort`, or
  Anthropic `output_config.effort` -- so Claude Code's `/effort` slider and
  opencode reasoning presets work unchanged. `--reasoning-effort` sets a
  server default. OpenAI's `minimal` clamps to `low`, Anthropic's `max` to
  `xhigh`. This is trained behavior, not a limit: the checkpoint may overrun
  it.
- `reasoning_budget_tokens` is a hard ceiling the runtime enforces: at the
  limit the sampler forces the thinking block closed and the answer resumes.
  On `/v1/messages`, a request that thinks without naming a budget gets a
  default cap of half its `max_tokens` or 2048, whichever is smaller, so a
  model cannot spend the whole completion deliberating and end the turn with
  no visible text; Claude Code's `thinking: {"type": "adaptive"}` is exactly
  that request. OpenAI-endpoint requests think uncapped by default, the same
  behavior llama-server gives them. `--thinking-budget N` applies one cap to
  every endpoint, 0 removes it everywhere, and
  `thinking: {"type": "disabled"}` never arms it.
- `POST /v1/chat/completions/{id}/stop_thinking` (or
  `/v1/messages/{id}/stop_thinking`) interrupts a live stream: the runtime
  closes the open thinking block on the next token and goes straight to the
  answer, through the same path as the budget. `{id}` is the id the stream
  reported in its first event. `/props` lists `stop_thinking` under
  `capabilities` when the loaded runtime supports it, and the chat UI shows
  an **Answer now** button while the model is thinking.
  Anthropic's `thinking: {"type": "enabled", "budget_tokens": N}` maps onto
  it. Unlike hosted APIs, this budget is a guarantee, not a hint.

`enable_thinking` (top level or in `chat_template_kwargs`) switches thinking
off entirely for templates with a switch. Chain-of-thought always arrives in
`reasoning_content` (on the message and as stream deltas), never in
`content`: a model told to write a file drafts it while thinking, and
streaming that draft as the answer made harnesses render the file instead of
writing it. `separate_reasoning` is accepted for compatibility and changes
nothing. On `/v1/messages`, reasoning is returned as Anthropic thinking
blocks.

### Structured output and tools

Declared tools are enforced by a sampler grammar, not just prompted: the tool
name must be a declared one, required parameters must be present, and
array/object argument values must be complete well-formed JSON. Scalar values
are free text -- the declared schema types them after parsing.
`response_format` (`json_object` / `json_schema`; `text.format` on
`/v1/responses`) is likewise enforced at the sampler. `FLYWEIGHT_TOOL_GRAMMAR=0`
and `FLYWEIGHT_RESPONSE_GRAMMAR=0` disable each constraint independently
without a rebuild. Tool-call arguments stream incrementally as JSON fragments,
so a long file-writing call produces wire progress instead of a timeout.
DeepSeek-V4, BailingMoE3 and K2-Horizon templates render their own tool
markup; every other architecture gets the generic Hermes-style tool prompt.

### Sampling

Sampling takes `repetition_penalty` (1 = off, the default; a value above it
looks over the last 64 generated tokens), plus OpenAI's `presence_penalty`
and `frequency_penalty` (default 0). The defaults match llama.cpp, so a
client that sends nothing gets the distribution it would get there. "No
penalty" is not always a neutral setting, though: with nothing discouraging a
token the model has just produced, a heavily quantized checkpoint can lock
onto a line and repeat it until the token budget runs out, and
`repetition_penalty: 1.1` per request (or `--repetition-penalty 1.1` on
`serve`) is the usual remedy. Only generated tokens are penalized --
penalizing the prompt would push the model away from the user's own wording.
Raise `penalty_window` to look further back, or set it to 0 to switch all
three penalties off at once. The penalties pause while a tool call is open (the sampler
grammar knows when one is): a call's arguments are verbatim by contract -- an
Edit reproduces the span of the file it replaces, character for character --
and penalizing recently emitted tokens there made the quote drift and the
harness's exact-match check fail. `FLYWEIGHT_TOOL_CALL_PENALTY=1` restores the
old behaviour for comparison. Outside tool calls a penalty still applies to
quoted file content, so for edit-heavy agent work on higher-precision quants
leave it off. `seed` pins the sampler per request; `n` other than 1 is
rejected.

## Inspect and generate

~~~bash
flyweight inspect model.gguf

flyweight generate model.gguf \
  --prompt "Explain mixture-of-experts routing." \
  --max-tokens 128 --temperature 0
~~~

## Benchmarking

The direct benchmark separates preparation, prompt prefill, and steady decode:

~~~bash
flyweight benchmark model.gguf \
  --prompt "Explain sliding-window attention." --chat \
  --context 32768 --iterations 30 --warmup 10 \
  --expert-mode auto --cache-type-k f16 --cache-type-v f16
~~~

For reproducible comparisons across prompt lengths, use the checked-in JSONL
harness:

~~~bash
python -m flyweight.runtime_benchmark run model.gguf \
  --output /tmp/baseline.jsonl --label baseline \
  --prompt "Runtime regression benchmark." \
  --prompt-lengths 256,1024,4096 \
  --context 32768 --samples 5 --sample-warmup 1

python -m flyweight.runtime_benchmark compare \
  /tmp/baseline.jsonl /tmp/candidate.jsonl
~~~

(`bench_runtime.py` at the repository root is a shim for the same module.)
`bench_server_ab.py` and `bench_server_client.py` drive a running server over
HTTP for end-to-end A/B comparisons. The other `bench_*.py` and `prof_*.py`
scripts at the root are one-off investigation tools kept for reference; some
need CuPy.

Run GPU benchmarks in isolation. Another process changes free VRAM and
therefore changes automatic expert-cache sizing.

## Runtime controls

`--help` on any command lists all of these; the runtime options are shared by
every command that builds a runtime (`imatrix` leaves out `--expert-mode` and
the MTP flags), and the server options belong to `serve` alone:

- `--quant ask|IQ2_XS|Q2_K|IQ3_XXS|Q3_K|IQ4_XS|Q4_K|Q5_K|Q6_K|Q8_0|F32`:
  quantization for a safetensors checkpoint (see below)
- `--imatrix PATH|off`: importance matrix for IQ packing; defaults to an
  `imatrix.dat` beside the checkpoint when one exists
- `--backend auto|cuda|cpu`: execution backend; `auto` uses CUDA when the
  driver and NVRTC load
- `--device N`: CUDA device index (default 0)
- `--gpu-cache-mib 0`: size allocations from currently free VRAM
- `--cache-type-k` / `--cache-type-v` `auto|f32|f16|bf16|q8_0|turbo3|turbo4`:
  KV precision (default `f16`)
- `--mtp-drafts N`: multi-token prediction for Qwen checkpoints; a round
  drafts N tokens and can commit N+1 (the drafts plus the token that verifies
  the last one). The runtime times a short trial of drafting against
  ordinary decode and keeps drafting only when it wins; the verdict expires
  after `FLYWEIGHT_MTP_RECALIBRATE_TOKENS` decoded tokens (default 2048, 0
  keeps the first verdict) so a reading taken under a load spike does not
  last the whole process, and `FLYWEIGHT_MTP_ADAPTIVE=0` drafts
  unconditionally. `--mtp-model` supplies a draft GGUF overlay (DSpark for
  DeepSeek-V4-Flash)
- `--dense-requant auto|q8|off`: control temporary BF16 dense-weight Q8 upload
- `--parallel N`: independent sequence slots; `--scratch-context TOKENS`
  gives the slots past the first a smaller context
- `--cache auto|off|MIB` (alias `--prompt-cache-mib`): host cache for
  displaced conversation state
- `--prefill-checkpoint-interval N` (default 256) and
  `--prefill-checkpoint-slots N` (default 4): how often a mid-prefill
  prefix-reuse snapshot is taken and how many are kept (`serve`, `generate`)
- `--cpu-threads N`: CPU expert worker count (0, the default, picks the
  physical cores)
- `--hybrid-prefill split|cpu`: whether prompt processing splits routed
  experts between the resident GPU set and the host or runs them all on the
  host (default `cpu` under `--expert-mode auto`, `split` otherwise)
- `--expert-residency mutable|immutable`: whether the GPU hot set may move
  during decode
- `--routed-moe`: run prompt processing's routed experts through the
  block-table MMQ kernels, and refuse to start rather than quietly not engage
- `--prefill-cache-seed auto|off|N`: post-prefill hot-expert placement
- `--expert-paging auto|staged|direct`: legacy paging transfer policy
- `--cpu-prefetch-auto` / `--cpu-prefetch-mib MIB`: warm prompt-relevant
  expert pages when beneficial, or under an explicit budget
- `--next-layer-prefetch N`: experts to page-hint per layer from observed
  layer-to-layer routing (0-64)
- `--swa-full`: trade VRAM for unrestricted sliding-layer rollback

Server options (`serve` only):

- `--model-name NAME`, `--cors-origin ORIGIN`, `--api-key KEY`,
  `--strict-model`
- `--reasoning-effort low|medium|high|xhigh`: server-wide default effort
- `--thinking-budget N`: cap for requests that think without naming a budget;
  unset it guards only `/v1/messages` (at 2048), a value applies everywhere,
  0 disables it everywhere
- `--temperature`, `--top-k`, `--top-p`, `--min-p`, `--repetition-penalty`,
  `--presence-penalty`, `--frequency-penalty`, `--penalty-window`:
  server-wide sampling defaults
- `--concurrency N` (alias `--max-concurrent-requests`, default 64): requests
  admitted to inference at once; the rest get HTTP 429 with `Retry-After`
- `--max-connections N` (default 128): cap simultaneous HTTP connection
  threads
- `--request-timeout-seconds N` (default 30): how long a client may take to
  send its request before the connection is dropped
- `--sse-keepalive-seconds S` (default 10): interval between keepalive
  comments on an idle stream
- `--max-tool-call-tokens N`: bound a runaway tool call (0 = unbounded)
- `--freeze-total-tokens`: pin the `<total_tokens>N tokens left</total_tokens>`
  counter Claude Code rewrites in its history on every request, so
  `/v1/messages` prompts stay cache-identical across turns instead of
  re-evaluating everything after the counter
- `--quiet` / `--verbose` (`-q` / `-v`): see "Reading the server log"

Prefill expert streaming (staging routed experts to the GPU for the batched
prefill kernels) is on by default with an automatically sized budget and has
no CLI flag; `FLYWEIGHT_PREFILL_EXPERT_STREAM_MIB` overrides the budget in MiB
(`0` disables). `FLYWEIGHT_PREFILL_PIPELINE=0` restores the serial prefill and
`FLYWEIGHT_CUDA_GRAPHS=0` disables graph replay, both for comparison only.

Runtime diagnostics are exposed through `/health`, including prefix-cache
counters and the sampler-grammar counters
(`grammar_constrained_steps`, `grammar_rejected_candidates`,
`grammar_empty_candidate_sets`); `FLYWEIGHT_ROUTE_RECURRENCE=1` adds routing
recurrence statistics. A few more environment switches are worth knowing:
`FLYWEIGHT_HF_CACHE` relocates (or, set to `off`, disables) the packed
safetensors cache; `FLYWEIGHT_DS4_EXPERT_CACHE_MIB` opts DeepSeek-V4 into a
GPU expert cache of that size; `FLYWEIGHT_QSA=1` enables the experimental
qwen4exp sparse-attention indexer; `FLYWEIGHT_V2_MLOCK=1` populates and locks
the mapped model in RAM; `FLYWEIGHT_CPU_THREADS` overrides the CPU-backend
team size. Beyond those, detailed profiling and experimental kernel switches
use `FLYWEIGHT_*` environment variables named in the source; unset profiling
variables for production serving.

## Quantization

A GGUF arrives quantized; a safetensors checkpoint does not, so the first
open packs it and caches the result beside the checkpoint. On a terminal the
CLI asks which quantization to pack, listing the exact size of each and
marking the ones already cached -- picking a cached one opens in about a
second, an uncached one costs a repack and the disk to store it. Anything
non-interactive keeps the default (`Q6_K`), and `--quant`, or
`FLYWEIGHT_HF_QUANT`, answers ahead of time:

~~~
Qwen3.8-27B is a safetensors checkpoint. Choose how to quantize it:
  1) IQ2_XS          --   unavailable: needs an importance matrix
  2) Q2_K        9.5 GiB   packs on first open, writes 9.5 GiB
  3) IQ3_XXS    10.8 GiB   packs on first open, writes 10.8 GiB
  4) Q3_K       11.9 GiB   packs on first open, writes 11.9 GiB
  5) IQ4_XS     14.1 GiB   packs on first open, writes 14.1 GiB
  6) Q4_K       14.9 GiB   cached, opens immediately
  7) Q5_K       17.8 GiB   packs on first open, writes 17.8 GiB
  8) Q6_K       20.9 GiB   cached, opens immediately  [default]
  9) Q8_0       26.5 GiB   packs on first open, writes 26.5 GiB
 10) F32       101.8 GiB   packs on first open, writes 101.8 GiB
quantization [Q6_K]:
~~~

Below `Q6_K` the tradeoff is accuracy against fit, and fit is what dominates:
a dense block that does not fit in VRAM is executed on the CPU, at about 3 ms
per token in decode -- prefill batches those blocks and pays less per token,
but not little enough to ignore. On a 12 GB card the 27B above spills 51 of
64 dense blocks at `Q6_K` and none at `Q2_K`, which is the difference between
4 and 36 tokens/s of decode. Pick the largest target that still fits, not the
largest you can pack.

Two things to know about spilled blocks. Which blocks spill is decided from
the VRAM free at startup, so on a card shared with a desktop the split can
differ from one launch to the next; pass `--gpu-cache-mib` to pin it. And a
spilled block whose weights are in a codebook format (IQ2/IQ3) is re-encoded
to Q3_K for the host kernels, which is lossy: `FLYWEIGHT_HOST_FFN_FORMAT`
picks `q2_k`, `q3_k` (default), `q8_0` or `off`, and
`FLYWEIGHT_HOST_FFN_Q8_MIB` caps the re-encoded bytes (default 8192).

`Q2_K` and `Q3_K` are dense-only: no GPU routed-expert kernel decodes either,
so a mixture-of-experts checkpoint packed to one would run every routed layer
on the CPU. Both are refused there rather than silently doing that -- `Q4_K`
is the smallest a MoE checkpoint can be packed to -- and the menu marks them
unavailable on such a model. `IQ3_XXS` has grouped expert kernels and no such
restriction.

`IQ3_XXS` is a codebook format -- 3.06 bits per weight, against Q3_K's 3.44
-- and quantizing to it searches 256 patterns per four weights rather than
rounding to a lattice, so packing the 27B above takes ~5 minutes against ~40
seconds for a K-quant. It is a one-time cost, cached like any other. It also
prefills fastest of the lot on the checkpoint above (196 tok/s at 1k context,
against 273 for Q2_K only because Q2_K is 1.5 GiB smaller and spills
nothing).

The search accepts an importance matrix -- per-channel activation statistics
gathered over calibration data, the `imatrix.dat` the ecosystem publishes
beside checkpoints. An `imatrix.dat` in the checkpoint directory is picked up
automatically, `--imatrix path` (or `FLYWEIGHT_HF_IMATRIX`) names one
elsewhere, and `off` disables the probe. With a matrix the codebook search
weights each channel by how hard the model actually drives it, which is what
lifts IQ3_XXS above the K-quant accuracy curve; without one it uses
llama.cpp's own no-matrix fallback weighting and lands on that curve, buying
size only. The matrix is part of the cache fingerprint, so switching it packs
a distinct cache.

The runtime can also gather its own matrix, over any Qwen-family model it
serves:

~~~bash
flyweight imatrix model.gguf \
  --text calibration.txt --output imatrix.dat
~~~

Calibration prefills the text in chunks and accumulates activation energy at
every projection's input -- dense projections on either backend, routed
experts pinned to the CPU path for the run so no layer goes uncounted. The
output is llama.cpp's legacy `.dat` layout, readable by both this packer and
`llama-quantize`.

`IQ4_XS` (4.25 bits against Q4_K's 4.5) packs through a 16-level nonlinear
table rather than a codebook search, so it costs K-quant packing time, reads
the importance matrix, and keeps grouped routed-expert GPU kernels -- on a
mixture-of-experts checkpoint it is the smallest target that serves every
routed layer on the GPU below Q4_K.

`IQ2_XS` (2.31 bits) is offered **only with an importance matrix** -- the
menu marks it unavailable and the loader refuses it otherwise. This mirrors
llama.cpp's own policy, and the measurement behind it is pinned in the test
suite: packed unweighted it round-trips *worse* than Q2_K, because at two
bits the search's entire job is knowing which channels can afford to be
wrong, and only calibration data can say. With a matrix it is the smallest
pack whose routed experts still run on grouped GPU kernels. The remaining
sub-3-bit formats (IQ2_XXS, IQ1_M) are still unoffered: no encoders yet.

For GGUFs that arrive already quantized, the dense GPU kernels cover F32,
F16, BF16, the K quants, Q8_0, IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS/
IQ4_NL, and the 1-bit IQ1_S and IQ1_M; grouped routed-expert GPU kernels
exist for Q4_K, Q5_K, Q6_K, Q8_0, IQ1_S, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S,
IQ4_XS, IQ4_NL, and NVFP4, and other formats (IQ2_S among them) run their
experts on the CPU path. IQ1_M has neither an expert kernel on either side
nor a readable LM head: an IQ1_M head is requantized to Q8_0 on upload, an
IQ1_M embedding table is refused, and IQ1_M routed experts are unsupported.

`--dense-requant auto` keeps the GGUF unchanged and chooses the temporary GPU
representation from the requested or available VRAM budget. It converts BF16
dense tensors to Q8_0 when the BF16 working set plus useful routed-expert
cache would exceed that budget. Use `q8` to force the memory-saving
representation or `off` to preserve the checkpoint's dense precision exactly.

`--cache-type-k` / `--cache-type-v` default to `f16`, and `auto` only reaches
for `turbo4` on a checkpoint with routed experts, above 32K context, whose
attention `head_dim` is a power of two between 32 and 512. A
*dense* checkpoint with a wide `head_dim` is the case that default serves
badly, and it has to be set by hand. Qwen3.8-27B (`qwen35`) is the worked
example: 16 full attention layers x 4 KV heads x head_dim 256 is 64 KiB of KV
per token, so KV competes with the weights for VRAM, and every dense block
that loses is re-read over PCIe on every token. On a 12 GB card with the
UD-IQ2\_XXS build:

| context | KV       | dense blocks spilled | decode      |
| ------- | -------- | -------------------- | ----------- |
| 16K     | `f16`    | 5 of 64 (408 MiB)    | 16.6 tok/s  |
| 16K     | `q8_0`   | none                 | 23.2 tok/s  |
| 16K     | `turbo4` | none                 | 24.0 tok/s  |
| 32K     | `f16`    | 16 of 64 (1306 MiB)  | 10.8 tok/s  |
| 32K     | `q8_0`   | 5 of 64 (408 MiB)    | 15.3 tok/s  |
| 32K     | `turbo4` | none                 | 22.9 tok/s  |

`q8_0` halves the cache and `turbo4` quarters it, which is why `q8_0` is
enough to clear the spill at 16K but not at 32K. Needle retrieval stays exact
under `turbo4` at 32K. The rule of thumb: if `prepare` reports dense blocks
on CPU, spend KV precision to buy them back before anything else.

Dense projections and the LM head take Q8-activation group-decode kernels
(`dp4a` on the K-quants, IQ formats and, since this release, Q8_0 -- which is
also the type an NVFP4 build requantizes its LM head to). `FLYWEIGHT_IQ2_Q8_DECODE=0`
switches every one of them, decode and chunked prefill alike, back to the
reconstruct-in-float kernels: slower, but bit-identical between the paths,
which is what the path-parity tests pin.

Qwen sampling with `top_k <= 256` reduces candidates on the GPU by default
(a grammar or penalty widens the candidate set it asks for, but the ceiling
is the same).
`sampling_gpu_topk_*`, `sampling_full_download_bytes`, and
`sampling_nanoseconds` expose its behavior; set `FLYWEIGHT_SAMPLING_GPU_TOPK=0`
only when comparing against the full-vocabulary host fallback.

## Testing

The default suite builds synthetic fixtures and does not require model
weights:

~~~bash
pip install ruff mypy      # CI installs these ad hoc; they are in no extra
ruff check src tests setup.py
mypy src/flyweight
pytest -q
~~~

The `check_*.py` scripts at the repository root need a checkout and real
weights. `check_vision_parity.py --mmproj PATH` runs the native vision tower
against the NumPy reference in `native/tools/qwen_vision_reference.py`
(`--backend cpu` for the host kernels) and needs no language model;
`check_greedy_determinism.py`, `check_q8_decode_parity.py`,
`check_attention_parity.py` and `check_expert_path_divergence.py` pin the
decode paths against each other on a model of your choosing.

Set `FLYWEIGHT_TEST_MODEL=/path/to/model.gguf` to opt into the real Qwen
reference tests. A configured model path that is missing or fails to load is
treated as a test failure; only an unset opt-in and an unavailable CUDA
device are skipped.

## Current limitations

- CUDA is the only model-execution accelerator; `--backend cpu` serves
  everything on the CPU kernels instead.
- Qwen3.8-Flash-Next (qwen4exp) runs its 12 sparse-attention layers as dense
  GQA by default: exact while the context fits the trained 2048-token
  selection budget, an approximation beyond it. `FLYWEIGHT_QSA=1` opts into
  the learned indexer, which is experimental. MTP needs a draft block: the
  Q4_K_XL release carries one, the standalone MTP file attaches through
  `--mtp-model`, and `--mtp-drafts` is rejected on UD-IQ1_S, which has none.
  The n-gram embedding table stays in host memory (16 row reads per token).
  Its IQ1_S/IQ4_NL experts have grouped GPU kernels, so under
  `--expert-mode cpu` prefill is expert-decode-bound on the host.
- Gemma 4: MTP, per-layer embeddings, shared-KV tail layers and next-layer
  prefetch are unimplemented, and expert placement is restricted to
  `cpu`/`hybrid`. The routed experts must be Q4_0 (the QAT release).
- Vision covers still images through a GGUF `mmproj` on the Qwen 3.5 family:
  no video, and the safetensors loader still reads only `text_config`. An
  mmproj whose tower has deepstack layers (`clip.vision.is_deepstack_layers`)
  is refused at attach until the decoder-side injection lands. The tower's
  attention and GEMM kernels are plain CUDA rather than tensor-core paths, so
  a 1024-token image costs a few seconds to encode.
- BailingMoE3 decodes its slots by interleaving rather than batching them, so
  `--parallel` removes the waiting but does not multiply throughput the way a
  batched forward would. Its prompt evaluation also runs at admission, so a
  very long prompt still holds the other slots for its duration. It has no
  expert paging: a model that does not fit falls back to the host entirely
  rather than keeping part of itself on the GPU.
- BailingMoE3's grouped routed-expert GPU kernels cover Q4_K and Q6_K only.
  Every other format its dispatch decodes -- the IQ formats among them --
  runs the routed experts one expert at a time instead, which is correct but
  much slower. Pack Ling to Q4_K or Q6_K unless the checkpoint does not
  otherwise fit.
- HF safetensors loading covers the Qwen 3.5 family and BailingMoE3 only;
  other architectures are GGUF-only.
- Laguna has no MTP, and supports only the per-head attention gate, so the
  per-element gate the larger Laguna checkpoints use is rejected at load.
- Laguna prefill uses the warp-online attention kernel. The tensor-core
  prefill routines fold Qwen's per-channel sigmoid gate in themselves, so
  Laguna's per-head softplus gate cannot use them and it forgoes that
  long-context path.
- Laguna's pre-tokenizer classifies non-ASCII letters by Unicode block rather
  than by a full category table, so non-Latin prose can split differently
  from the reference tokenizer.
- Laguna (with IQ experts) and Gemma 4 concentrate available expert-cache
  VRAM into a contiguous suffix of complete layers and pin every expert in
  those layers, using the CPU path for earlier layers. Set
  `FLYWEIGHT_LAGUNA_WHOLE_LAYERS=0` to restore per-expert placement for
  comparison, or to a positive integer to cap the number of complete GPU
  layers.
- Laguna prefill over IQ2_XS, IQ3_XXS or IQ4_XS experts uses the direct
  quantized 8-token CPU kernel by default instead of expanding expert rows to
  f32. Set `FLYWEIGHT_PREFILL_DIRECT_QUANT=0` only for comparison; `=1`
  continues to opt other supported architectures into the same path.
- On AVX-512 hosts, IQ2_XS decode widens a complete 16-value scale group at a
  time and fuses the gate/up projections so they share each activation load.
  Set `FLYWEIGHT_IQ_AVX512=0` to compare with the AVX2 kernel, or
  `FLYWEIGHT_FUSED_MOE_GATE_UP=0` to disable only the automatic IQ2_XS fusion.
- IQ expert decode is sensitive to memory bandwidth, clock sharing and thread
  placement. The default uses physical cores; tune `--cpu-threads` for the
  machine rather than assuming SMT helps (14 workers beat 8, 16 and 32 on the
  reference 16-core Laguna host).
- The tool-call grammar constrains the generic Hermes markup; DeepSeek-V4,
  BailingMoE3, K2-Horizon and Muse Glimmer emit their own formats, which are
  parsed tolerantly but not sampler-enforced. Muse Glimmer also has no
  sampler-enforced JSON response mode and no thinking budget or
  `stop_thinking`.
- Qwen sampled decoding currently transfers the vocabulary logits to the host
  when `top_k > 256`.
- Dynamic MoE routing still has host synchronization points.
- Special-token spellings inside message content (`<|im_start|>`,
  `<tool_call>`, `<think>`, ...) are tokenized as the control tokens, as
  they are by the HF and llama.cpp tokenizers: the rendered prompt is one
  flat string. A client that relays untrusted text should strip them.
- `logprobs`, `top_logprobs` and a non-empty `logit_bias` are rejected with
  400 rather than ignored; `parallel_tool_calls: false` (and Anthropic's
  `disable_parallel_tool_use`) cap a turn at one tool call.
- Usage detail: `cached_tokens` / `cache_read_input_tokens` is the prompt
  prefix the runtime reused; `reasoning_tokens` is counted by re-encoding the
  chain-of-thought split out of the answer, so it is exact wherever the
  tokenizer round-trips its own output (BPE does) and an estimate otherwise.
- Persistent fused layer kernels are incomplete.
- Image, audio, embedding, fine-tuning, and hosted-tool APIs are out of
  scope.
- Response records and prompt caches are process-local.

## Architecture

- `native/src/v2_runtime.cpp`: GGUF parsing, memory planning, scheduling,
  model orchestration, prefix reuse, sampling, and the native runtime ABI;
  `native/src/v2_mtp_verifier.inc` (the prefill driver and MTP verifier) and
  `native/src/v2_vision.inc` (the mmproj tower) are compiled into it
- `native/src/gpu_driver.cpp`: CUDA driver, NVRTC, cuBLAS/cuBLASLt, graph,
  and transfer integration
- `native/include/flyweight_v2_qwen_kernels.hpp`: the CUDA kernel source,
  JIT-compiled by NVRTC at startup and compiled as host C++ for
  `--backend cpu`; `native/src/cpu_backend.cpp` and the `cpu_*`, `q4_*` and
  `qwen_cpu_*` files are the host kernels
- `native/include/flyweight_v2_format_dispatch.hpp`: which kernel reads which
  tensor format, on each side
- `native/include/flyweight_v2_hf.hpp`, `_hf_quantize.hpp`, `_hf_cache.hpp`,
  `_imatrix.hpp`: the safetensors loader, packer and cache
- `native/include/flyweight_v2_bailing.hpp`, `flyweight_v2_deepseek4*.hpp`:
  the BailingMoE3 and DeepSeek-V4 runtimes
- `native/include/flyweight_v2_tool_grammar.hpp`: sampler-side tool and JSON
  response constraints
- `src/flyweight/cli.py`: the command line, `doctor`, and the quantization
  menu; `src/flyweight/native_build.py`: the CMake driver `pip install` uses
- `src/flyweight/v2.py`: Python bindings for the native ABI
- `src/flyweight/v2_server.py`: tokenizer, cooperative engine thread, and
  native inference service; `src/flyweight/vision.py`: image decoding and
  the encoded-image cache
- `src/flyweight/deepseek4_server.py`, `deepseek4.py`, `dspark.py`: the
  dedicated DeepSeek-V4 service and its DSpark drafter
- `src/flyweight/server.py`: shared HTTP protocol implementation
- `src/flyweight/sampling.py`: the sampling settings every surface shares
- `src/flyweight/transcript_audit.py`: request dumps and the
  `transcript-audit` command
- `src/flyweight/runtime_benchmark.py`: benchmark capture and comparison
- `plans/`: design notes for the deliberate omissions and the semantics of
  each architecture, referenced from the code

See [CONTRIBUTING.md](CONTRIBUTING.md) for how changes are expected to arrive
and [SECURITY.md](SECURITY.md) for reporting a vulnerability.

## License

Apache-2.0.
