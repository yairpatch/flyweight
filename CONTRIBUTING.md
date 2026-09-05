# Contributing to Flyweight

Thanks for taking an interest. Flyweight is a native C++/CUDA GGUF inference
runtime, so contributing to it means compiling the runtime locally — this
document is mostly about getting that far.

## Before you start

Open an issue before writing anything non-trivial. A lot of what looks like a
missing feature here is a deliberate omission recorded in
[Current limitations](README.md#current-limitations) or worked through in
`plans/`, and a kernel that runs 10% faster on one card is not automatically
faster on another. Small fixes — a stale doc line, a mis-sized buffer, a
format the dispatch misses — need no discussion; send them.

## Setting up

A checkout compiles the native runtime from source (the published
`flyweight-llm` wheels are for using Flyweight, not changing it), so a C++
toolchain is required once, at install time:

| | Needed |
| --- | --- |
| Python | 3.11 or newer, 64-bit |
| CMake | 3.24 or newer |
| Compiler | MSVC v143 (Windows) or GCC 13+ / Clang 16+ (Linux) |
| GPU | Any current NVIDIA driver — **no CUDA toolkit** |
| Node | 22, only if you touch the web UI |

CUDA kernels are compiled at runtime through the driver API, which is why the
driver alone is enough and `nvcc` is never invoked. You do not need a GPU to
contribute: `--backend cpu` serves everything on the CPU kernels, and the
default test suite requires neither a GPU nor model weights.

```bash
pip install -e .          # builds the native library
flyweight doctor          # confirms the install reports itself healthy
```

`python -m flyweight.native_build` adds the contract tests and benchmarks to
the same build tree. The library's compile flags do not change, so this links
the extra targets rather than compiling the runtime a second time.

## Running the tests

CI runs exactly this, on `ubuntu-latest` and `windows-latest` against Python
3.11 and 3.13. Run it locally before opening a PR:

```bash
ruff check src tests setup.py
mypy src/flyweight
pytest -q
ctest --test-dir build/native --output-on-failure -C Release
```

The default suite builds synthetic GGUF and safetensors fixtures (see
`tests/*_fixture.py`) and needs no model weights. To exercise the real thing,
set `FLYWEIGHT_TEST_MODEL=/path/to/model.gguf` — note that a configured path
which is missing or fails to load is treated as a **failure**, not a skip.
Only an unset opt-in and an unavailable CUDA device are skipped.

If you change the web UI, rebuild the committed bundle or CI will fail on a
stale one:

```bash
cd web && pnpm install --frozen-lockfile && pnpm build
```

## Performance claims

This project is largely about throughput, so a performance change needs a
measurement, not an argument. Say which model, quantization, GPU and context
length you measured at, and give the before and after. `bench_runtime.py`
produces a comparable JSONL record:

```bash
python -m flyweight.runtime_benchmark run model.gguf --output after.jsonl
python -m flyweight.runtime_benchmark compare before.jsonl after.jsonl
```

Numbers from one card do not transfer to another. If a change helps your
hardware and might hurt someone else's, put it behind a `FLYWEIGHT_*`
environment variable defaulting to the old behaviour, and say so in the PR.

## Style

- **Match the surrounding code.** The native sources are dense and heavily
  commented in a particular voice; new code should be indistinguishable from
  what is already there.
- **Comments explain why, not what.** The useful comments in this codebase
  record what was measured, what was tried and failed, or what a future reader
  would otherwise break. `// increment the counter` is noise.
- **`ruff` and `mypy` must pass** on `src/flyweight`. There is no separate
  formatter to run.
- **Commit messages** follow `type(scope): summary in the imperative`, with
  the body explaining the reasoning. Types in use: `feat`, `perf`, `fix`,
  `docs`, `test`, `refactor`, `build`, `chore`, `plan`. Scopes are the area
  touched — `prefill`, `server`, `decode`, `moe`, `runtime`, `cpu`, `ui` and
  so on. Look at `git log` before writing one.

## Pull requests

`main` is protected: it takes no direct pushes, and every change lands through
a pull request with the four CI jobs green and one approving review.

A few things specific to this repo:

- **Your first PR needs a maintainer to release CI.** GitHub holds workflow
  runs on pull requests from first-time contributors until they are approved,
  so your checks will sit unstarted rather than failing. That is expected;
  it is not something you did wrong.
- **One topic per PR.** A kernel change and a server change in the same branch
  are two PRs.
- Say what hardware and which model you tested on. For this runtime that is
  not boilerplate — behaviour genuinely differs across architectures,
  quantization formats and cards.

## Licence

Flyweight is Apache-2.0. Contributions are accepted under the same terms, per
section 5 of the licence; there is no separate CLA to sign.
