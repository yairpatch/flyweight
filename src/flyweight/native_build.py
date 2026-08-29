from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


LIBRARY_STEM = "flyweight_v2"


def source_root() -> Path | None:
    """The checkout's `native/` directory, or None when there is no source.

    A wheel install has the package but not the sources it was built from, so
    every "rebuild it" instruction has to check this first -- telling someone to
    run a build that cannot find a CMakeLists is worse than saying nothing.
    """
    native = Path(__file__).resolve().parents[2] / "native"
    return native if (native / "CMakeLists.txt").is_file() else None


def default_output() -> Path:
    """Where the runtime looks for the library: the package's own `_native`."""
    return Path(__file__).with_name("_native")


def build_native(
    *,
    output: Path | None = None,
    build_dir: Path | None = None,
    clean: bool = False,
    development_targets: bool = True,
) -> Path:
    """Configure and build the native runtime, returning the library path.

    One implementation, one build directory. `setup.py` calls this too rather
    than carrying a second copy of the cmake invocation: when the two drifted
    they wrote to different build trees, and whichever one you did not refresh
    became a stale library that silently answered every later question.

    `development_targets` off skips the contract tests and benchmarks, which a
    wheel does not ship and which roughly halve the build.
    """
    source = source_root()
    if source is None:
        raise FileNotFoundError(
            "no native sources next to this package: "
            f"{Path(__file__).resolve().parents[2] / 'native'} does not exist. "
            "This is an installed copy without the checkout it was built from; "
            "install a wheel that bundles the library, or build from a clone."
        )
    project_root = source.parent
    build = Path(build_dir) if build_dir else project_root / "build" / "native"
    output = Path(output) if output else default_output()
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
        f"-DFLYWEIGHT_BUILD_DEVELOPMENT_TARGETS={'ON' if development_targets else 'OFF'}",
        f"-DBUILD_TESTING={'ON' if development_targets else 'OFF'}",
    ]
    # Everything is built inside `build`, and only the library is copied out.
    # Pointing cmake's output directories at the package instead dumped every
    # contract test and benchmark executable into it -- two dozen binaries
    # shipped beside the runtime, and stale ones left behind after a rename.
    if os.name == "nt":
        configure.extend(["-G", "NMake Makefiles"])
    subprocess.run(configure, check=True, env=environment)
    subprocess.run(
        ["cmake", "--build", str(build), "--config", "Release"],
        check=True,
        env=environment,
    )
    suffixes = (".dll", ".dylib", ".so") if os.name == "nt" else (".so", ".dylib", ".dll")
    # The build tree is searched FIRST and the package second. The other order
    # returns a library already sitting in the package without copying the one
    # just built -- every later run then answers from a build nobody made.
    for directory in (build, output):
        for suffix in suffixes:
            candidate = directory / f"{LIBRARY_STEM}{suffix}"
            if not candidate.is_file():
                continue
            destination = output / candidate.name
            if candidate != destination:
                shutil.copy2(candidate, destination)
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
    parser = argparse.ArgumentParser(description="build the Flyweight native backend")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args(argv)
    print(build_native(clean=args.clean))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
