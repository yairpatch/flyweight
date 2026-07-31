from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def build_native(*, clean: bool = False) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    source = project_root / "native"
    build = project_root / "build" / "native"
    output = Path(__file__).with_name("_native")
    if clean and build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    environment = _build_environment()
    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output}",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={output}",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={output}",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={output}",
    ]
    if os.name == "nt":
        configure.extend(["-G", "NMake Makefiles"])
    subprocess.run(configure, check=True, env=environment)
    subprocess.run(
        ["cmake", "--build", str(build), "--config", "Release"],
        check=True,
        env=environment,
    )
    suffixes = (".dll", ".dylib", ".so") if os.name == "nt" else (".so", ".dylib", ".dll")
    candidates = tuple(output / f"colibri_native{suffix}" for suffix in suffixes) + tuple(
        build / f"colibri_native{suffix}" for suffix in suffixes
    )
    for candidate in candidates:
        if candidate.is_file():
            destination = output / candidate.name
            if candidate != destination:
                shutil.copy2(candidate, destination)
            # CMake builds the v2 target alongside v1. Returning the v1 path
            # preserves the existing build_native API.
            return destination
    raise FileNotFoundError("native build completed without a runtime library")


def _build_environment() -> dict[str, str]:
    environment = os.environ.copy()
    if os.name != "nt":
        return environment
    # Always prefer the x64 toolchain via vcvars64. An x86/Developer prompt puts
    # a 32-bit cl.exe on PATH, which cannot compile the F16C intrinsics used by
    # the AVX2 kernels (e.g. _cvtsh_ss), so trusting an existing cl is unsafe.
    candidates = (
        Path(
            r"C:\Program Files\Microsoft Visual Studio\2022\Community"
            r"\VC\Auxiliary\Build\vcvars64.bat"
        ),
        Path(
            r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools"
            r"\VC\Auxiliary\Build\vcvars64.bat"
        ),
        Path(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
            r"\VC\Auxiliary\Build\vcvars64.bat"
        ),
    )
    vcvars = next((candidate for candidate in candidates if candidate.is_file()), None)
    if vcvars is None:
        # Fall back to an existing x64 cl on PATH if one is already configured.
        if shutil.which("cl", path=environment.get("PATH")):
            return environment
        raise FileNotFoundError(
            "Visual Studio C++ Build Tools were not found; install the Desktop "
            "development with C++ workload"
        )
    result = subprocess.run(
        f'call "{vcvars}" >nul && set',
        shell=True,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.casefold() == "path":
            for existing in tuple(environment):
                if existing.casefold() == "path":
                    del environment[existing]
            name = "PATH"
        environment[name] = value
    return environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the Colibri native backend")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args(argv)
    print(build_native(clean=args.clean))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
