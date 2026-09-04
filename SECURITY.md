# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it through
[GitHub's private vulnerability reporting](https://github.com/yairpatch/flyweight/security/advisories/new),
which keeps the report private until a fix is published. Include what you did,
what happened, and the model and command line involved if it is reachable
through the runtime.

This is a single-maintainer project, so please allow a reasonable window for a
response before disclosing publicly.

## Scope

Flyweight parses untrusted binary input (GGUF and safetensors files), decodes
weights into fixed device buffers, and exposes an HTTP API. Reports that are
clearly in scope:

- Memory corruption reachable from a malformed GGUF or safetensors checkpoint
- Out-of-bounds reads or writes in the native runtime, including the CUDA
  kernels, driven by attacker-influenced metadata (tensor shapes, block
  counts, vocabulary sizes)
- Authentication bypass on the HTTP server, or a request from one client
  reaching another client's cache, slot or response
- Path traversal or arbitrary file access through a model path, `--mmproj`, or
  an uploaded attachment

## Not in scope

- **A model file you supplied yourself doing something you did not expect.**
  Loading a checkpoint is equivalent to running its author's code; the runtime
  validates structure so it fails cleanly rather than corrupting memory, but a
  malicious checkpoint from an untrusted source is outside the threat model.
- **Serving on an untrusted network without authentication.** The server binds
  where you tell it to and is unauthenticated unless you set a bearer token.
- **Model output.** Wrong, biased or unsafe generated text is a model
  property, not a runtime vulnerability.
- **Special-token spellings in message content.** These are tokenized as
  control tokens by design, matching the HF and llama.cpp tokenizers. This is
  documented under
  [Current limitations](https://github.com/yairpatch/flyweight#current-limitations):
  a client relaying untrusted text should strip them.

## Supported versions

Fixes land on `main`. There is no long-term support branch and no backporting
to older tags.
