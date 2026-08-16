"""Generate the host-side kernel translation unit and dispatch registry.

The CUDA corpus in colibri_v2_qwen_kernels.hpp and colibri_v2_native_kernels.hpp
is the single source of truth for the runtime's numerics. The CPU backend
compiles that same text as C++ (see colibri_cpu_shim.hpp) rather than carrying a
hand-written second copy, so the two backends cannot drift.

This script extracts the text, applies the few transforms the host compiler
needs, and emits:

  colibri_cpu_kernels.inc        the corpus body, host-compilable
  colibri_cpu_kernel_table.inc   name -> launcher, unpacking void** arguments

Kernel names cannot be read off the raw source: several families (KV_STORE,
KV_APPEND, ...) are emitted by function-like macros, so the entry point names
only exist after expansion. The table is therefore built from a real
preprocessor pass over the transformed corpus, with __global__ redefined to a
marker token so kernels stay distinguishable from ordinary extern "C" helpers.

Both outputs are build artifacts; CMake regenerates them when a header changes.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RAW_SEGMENT = re.compile(r'R"COLIBRI_CUDA\((.*?)\)COLIBRI_CUDA"', re.DOTALL)

MARKER = "COLIBRI_CPU_KERNEL_MARKER"

# Device headers the host build supplies itself through the shim.
DROP_INCLUDES = re.compile(
    r"^\s*#\s*include\s*<(cuda_[A-Za-z0-9_]*\.h|cub/[^>]*)>\s*$",
    re.MULTILINE,
)

# `extern __shared__ float name[];` declares the dynamic shared allocation whose
# size is the launch's shared_bytes. The shim hands that block out per block, so
# the declaration becomes a pointer into it. `extern` cannot survive the
# `__shared__ -> static thread_local` macro, which is why this one is rewritten
# textually instead.
EXTERN_SHARED = re.compile(
    r"extern\s+__shared__\s+([A-Za-z_][A-Za-z0-9_ :]*?)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*\]\s*;"
)

# `__shared__ float warp_sums[8];` and `__shared__ typename S::T storage;`.
# extern __shared__ is rewritten earlier and no longer matches.
SHARED_DECL = re.compile(
    r"__shared__\s+(?:typename\s+)?[A-Za-z_][A-Za-z0-9_:<>, ]*?\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^;\]]*\])*\s*;"
)

KERNEL_HEAD = re.compile(
    r'extern\s+"C"\s+' + MARKER + r"\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?"
    r"void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.DOTALL,
)


# MSVC truncates a single string literal past 16384 bytes (C2026), which is why
# the corpus is split into adjacent literals rather than written as one. The
# concatenation is unbounded, so only the individual pieces matter. Checked here
# because the headers are edited on Linux, where nothing else would notice.
MAX_SEGMENT_BYTES = 16384


def extract(path: Path) -> str:
    text = path.read_text()
    segments = RAW_SEGMENT.findall(text)
    if not segments:
        raise SystemExit(f"no COLIBRI_CUDA raw segments found in {path}")
    for index, segment in enumerate(segments):
        if len(segment.encode()) < MAX_SEGMENT_BYTES:
            continue
        line = text[: text.index(segment)].count("\n") + 1
        raise SystemExit(
            f"{path}:{line}: COLIBRI_CUDA segment {index} is "
            f"{len(segment.encode())} bytes, over the {MAX_SEGMENT_BYTES} MSVC "
            'limit; close it with )COLIBRI_CUDA" and reopen with '
            'R"COLIBRI_CUDA( at a line boundary'
        )
    return "".join(segments)


def transform(cuda: str) -> str:
    cuda = DROP_INCLUDES.sub("", cuda)
    cuda = EXTERN_SHARED.sub(
        r"\1* \2 = reinterpret_cast<\1*>("
        r"::colibri::cpu::t_scheduler->dynamic_shared());",
        cuda,
    )
    # Every __shared__ declaration gets a zeroing call after it. __shared__ maps
    # to `static thread_local`, which persists across blocks running on the same
    # worker; CUDA hands each block uninitialized shared memory instead. Parts of
    # the corpus read shared slots they did not write in this launch (see
    # block_reduce_sum), which is benign-ish on a GPU and catastrophic against a
    # previous block's real values.
    cuda = SHARED_DECL.sub(
        r"\g<0> ::colibri::cpu::shared_zero_once(&\g<name>, sizeof(\g<name>));",
        cuda,
    )
    # #pragma unroll is a device-compiler hint with no host spelling; GCC and
    # MSVC both warn on the unknown pragma.
    cuda = re.sub(r"^[ \t]*#pragma[ \t]+unroll.*$", "", cuda, flags=re.MULTILINE)
    return cuda


def preprocessor_flags(compiler: str) -> list[str]:
    """Flags that make `compiler` preprocess C++ to stdout, no line markers.

    MSVC shares none of the GCC spellings: /EP is -E -P, /TP is -x c++, and the
    logo would otherwise land in the token stream we parse.
    """
    name = Path(compiler).stem.lower()
    if name in ("cl", "clang-cl"):
        return ["/nologo", "/EP", "/TP"]
    return ["-E", "-P", "-x", "c++"]


def find_preprocessor(explicit: str | None) -> list[str]:
    # A path, not a command line: MSVC lives under "Program Files (x86)", and
    # splitting on whitespace would tear that path apart.
    if explicit:
        return [explicit, *preprocessor_flags(explicit)]
    for candidate in ("cc", "gcc", "clang", "c++", "g++", "cl"):
        found = shutil.which(candidate)
        if found:
            return [found, *preprocessor_flags(found)]
    raise SystemExit("no C preprocessor found; pass --preprocessor")


def preprocess(cuda: str, command: list[str]) -> str:
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "corpus.cpp"
        # Expanding __global__ to a marker keeps kernels distinguishable once
        # the macro families have been expanded away.
        source.write_text(f"#define __global__ {MARKER}\n" + cuda)
        result = subprocess.run(
            [*command, str(source)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(
                "preprocessing the CUDA corpus failed:\n"
                + " ".join(command)
                + "\n"
                + result.stderr[:4000]
            )
        # cl echoes the input file name onto the output stream ahead of the
        # preprocessed text; it is not a token the table scanner should see.
        return result.stdout.replace(source.name + "\n", "", 1)


def scan_parameter_list(text: str, start: int) -> tuple[str, int]:
    """Return the balanced parameter text starting just after '(' at `start`."""
    depth = 1
    index = start
    while index < len(text) and depth:
        character = text[index]
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
            if depth == 0:
                return text[start:index], index + 1
        index += 1
    raise SystemExit("unterminated kernel parameter list")


def split_parameters(text: str) -> list[tuple[str, str]]:
    """Return (type, name) for each parameter, ignoring comments."""
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # Split on commas that are not inside template brackets or parens.
    parts: list[str] = []
    depth = 0
    current = ""
    for character in text:
        if character in "(<[":
            depth += 1
        elif character in ")>]":
            depth -= 1
        if character == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += character
    parts.append(current)

    parameters = []
    for raw in parts:
        item = raw.strip()
        if not item or item == "void":
            continue
        match = re.match(r"^(.*?)([A-Za-z_][A-Za-z0-9_]*)\s*$", item, re.DOTALL)
        if not match:
            raise SystemExit(f"cannot parse kernel parameter: {item!r}")
        parameter_type = " ".join(match.group(1).split())
        name = match.group(2)
        if not parameter_type:
            raise SystemExit(f"kernel parameter has no type: {item!r}")
        # `const` on a by-value parameter binds the callee's local only; keeping
        # it would make the unpacking variable unassignable.
        if parameter_type.startswith("const ") and "*" not in parameter_type:
            parameter_type = parameter_type[len("const "):]
        parameters.append((parameter_type, name))
    return parameters


# Control-flow keywords that look like calls but are not.
NOT_A_CALL = frozenset({
    "if", "for", "while", "switch", "return", "sizeof", "static_cast",
    "reinterpret_cast", "const_cast", "dynamic_cast", "do", "else", "catch",
    "alignof", "decltype", "typename", "template", "operator", "new", "delete",
})

# Anything that suspends a thread mid-kernel. A kernel that reaches one of these
# needs real fibers; a kernel that cannot reach one can run as a plain loop over
# thread indices, which is roughly two orders of magnitude cheaper.
#
# `cub::` is in the list because the block-wide sort is block-cooperative even
# though the corpus text shows no barrier: the synchronization lives inside the
# host replacement in cpu_kernels.cpp, which this script never sees. Any future
# shim that synchronizes internally has to be listed here too.
COOPERATIVE_TOKENS = ("__syncthreads", "__syncwarp", "__shfl_", "cub::")

CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^;{}()]*>)?\s*\(")

DEFINITION = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^;{}()]*>)?\s*\([^;{}]*\)\s*(?:const\s*)?\{"
)


def function_bodies(text: str) -> dict[str, str]:
    """Map function name -> body text for every definition in the corpus."""
    bodies: dict[str, str] = {}
    for match in DEFINITION.finditer(text):
        name = match.group(1)
        if name in NOT_A_CALL:
            continue
        depth = 1
        index = match.end()
        while index < len(text) and depth:
            character = text[index]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
            index += 1
        body = text[match.end():index]
        # Overloads and template instantiations share a name; concatenating is
        # the conservative choice, since it can only add cooperative edges.
        bodies[name] = bodies.get(name, "") + body
    return bodies


def cooperative_kernels(text: str, kernels: list[str]) -> set[str]:
    """Names that can reach a barrier or shuffle, directly or transitively.

    Errs toward marking a kernel cooperative: a false positive costs speed, a
    false negative would run a barrier without a fiber to suspend.
    """
    bodies = function_bodies(text)

    direct = {
        name
        for name, body in bodies.items()
        if any(token in body for token in COOPERATIVE_TOKENS)
    }

    callees = {
        name: {
            call
            for call in CALL.findall(body)
            if call not in NOT_A_CALL and call in bodies
        }
        for name, body in bodies.items()
    }

    cooperative = set(direct)
    changed = True
    while changed:  # transitive closure over the call graph
        changed = False
        for name, called in callees.items():
            if name not in cooperative and called & cooperative:
                cooperative.add(name)
                changed = True

    # A kernel we never found a body for is unknown, so assume the worst.
    return {name for name in kernels if name not in bodies or name in cooperative}


def emit_table(preprocessed: str) -> tuple[str, list[str]]:
    lines = [
        "// Generated by native/tools/generate_cpu_kernels.py -- do not edit.",
        "",
    ]
    entries: list[str] = []
    seen: set[str] = set()

    for match in KERNEL_HEAD.finditer(preprocessed):
        name = match.group(1)
        if name in seen:
            raise SystemExit(f"duplicate kernel name in corpus: {name}")
        seen.add(name)

        raw_parameters, _ = scan_parameter_list(preprocessed, match.end())
        parameters = split_parameters(raw_parameters)

        # CUDA passes kernel arguments as an array of pointers to the values.
        unpack = "".join(
            f"    {parameter_type} {parameter_name} = "
            f"*reinterpret_cast<{parameter_type}*>(arguments[{index}]);\n"
            for index, (parameter_type, parameter_name) in enumerate(parameters)
        )
        call = ", ".join(parameter_name for _, parameter_name in parameters)
        lines.append(
            f"static void colibri_cpu_launch_{name}(void** arguments) {{\n"
            f"{unpack}"
            f"    {name}({call});\n"
            f"}}\n"
        )
        entries.append(name)

    cooperative = cooperative_kernels(preprocessed, entries)

    lines.append("static const ColibriCpuKernelEntry kColibriCpuKernels[] = {")
    for name in entries:
        flag = "true" if name in cooperative else "false"
        lines.append(f'    {{"{name}", &colibri_cpu_launch_{name}, {flag}}},')
    lines.append("};")
    lines.append("")
    print(
        f"  {len(cooperative)} cooperative, {len(entries) - len(cooperative)} "
        "barrier-free",
        file=sys.stderr,
    )
    return "\n".join(lines), entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("body")
    parser.add_argument("table")
    parser.add_argument("headers", nargs="+")
    parser.add_argument("--preprocessor")
    arguments = parser.parse_args()

    headers = [Path(header) for header in arguments.headers]
    cuda = transform("\n".join(extract(header) for header in headers))

    body_path = Path(arguments.body)
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_path.write_text(
        "// Generated by native/tools/generate_cpu_kernels.py -- do not edit.\n"
        "// Source: " + ", ".join(header.name for header in headers) + "\n\n"
        + cuda
        + "\n"
    )

    preprocessed = preprocess(cuda, find_preprocessor(arguments.preprocessor))
    table, entries = emit_table(preprocessed)
    Path(arguments.table).write_text(table)
    print(f"generated {len(entries)} host kernels", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
