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


def _replace_library(built: Path, destination: Path) -> None:
    """Install the freshly built library WITHOUT touching the old file's inode.

    A running server has this library mmap'd. Copying onto it in place --
    shutil.copy2 opens the destination with O_TRUNC and writes through the same
    inode -- pulls the mapped pages out from under that process, and its next
    call into the runtime dies with SIGBUS or SEGV_ACCERR somewhere with no
    symbols left to name. That is a genuinely baffling crash to be handed by a
    rebuild in another terminal, and it cost a live 45k-token session to
    diagnose.

    Writing a sibling and renaming is atomic: the old inode stays alive and
    mapped for as long as the server holds it, so the running process keeps
    working on the code it started with and picks the new library up when it is
    next restarted. On Windows, where a loaded DLL cannot be replaced at all,
    the rename fails and the old library is left in place rather than
    half-overwritten -- so the message says which process to stop.
    """
    staged = destination.with_name(destination.name + ".new")
    shutil.copy2(built, staged)
    try:
        os.replace(staged, destination)
    except OSError as error:
        staged.unlink(missing_ok=True)
        raise OSError(
            f"could not install the native library over {destination}: {error}. "
            "Stop whatever has it loaded (a running `flyweight serve`) and "
            "build again."
        ) from error


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
    output.mkdir(parents=True, exist_ok=True)
    environment = _build_environment()
    generator = _generator(environment)
    # cmake refuses to reconfigure a build tree with a different generator, and
    # says so in terms of a cache the reader never chose. Switching from the old
    # NMake default to Ninja must not turn every existing checkout's next build
    # into that error, so the mismatch is resolved here by starting over.
    if generator is not None and _cached_generator(build) not in (None, generator):
        shutil.rmtree(build)
    build.mkdir(parents=True, exist_ok=True)
    # Everything is built inside `build`, and only the library is copied out.
    # Pointing cmake's output directories at the package instead dumped every
    # contract test and benchmark executable into it -- two dozen binaries
    # shipped beside the runtime, and stale ones left behind after a rename.
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
    if generator is not None:
        configure.extend(["-G", generator])
    subprocess.run(configure, check=True, env=environment)
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--config",
            "Release",
            # NMake Makefiles has no parallel mode at all, so the Windows build
            # compiled one heavy AVX/kernel translation unit at a time while the
            # other fifteen cores idled. Ninja above fixes that; the explicit job
            # count keeps make from spawning an unbounded -j on the way there.
            "--parallel",
            str(os.cpu_count() or 4),
        ],
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
                _replace_library(candidate, destination)
            return destination
    raise FileNotFoundError("native build completed without a runtime library")


MSVC_MISSING = (
    "Visual Studio's C++ build tools were not found. Install them with:\n"
    "    winget install --id Microsoft.VisualStudio.2022.BuildTools "
    '--override "--quiet --wait --add '
    'Microsoft.VisualStudio.Workload.VCTools --includeRecommended"\n'
    "or tick 'Desktop development with C++' in the Visual Studio Installer. "
    "Any edition works -- Community, Professional, Enterprise, or the "
    "standalone Build Tools -- and no CUDA toolkit is required."
)


def find_msvc() -> Path | None:
    """The `vcvars64.bat` of the newest VS install carrying the x64 C++ tools.

    Asking vswhere is the only supported way to find Visual Studio: it is
    installed at a fixed path by every edition, and it knows about installs this
    code cannot guess -- Professional and Enterprise, Preview channels, future
    year releases, and anything the user put on another drive. A hardcoded list
    of three Community/BuildTools paths told those machines to install a
    compiler they already had.

    Both the build and `flyweight doctor` resolve the toolchain through here, so
    the diagnosis and the build can never disagree about what is installed.
    """
    for arguments in (["-latest"], ["-latest", "-prerelease"]):
        for install in _vswhere(arguments):
            vcvars = install / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if vcvars.is_file():
                return vcvars
    # vswhere ships with the installer, not with the tools, so it can be absent
    # on a machine whose VS was imaged or hand-copied. Globbing the standard
    # roots covers every edition and year without naming any of them.
    found = sorted(
        (
            vcvars
            for root in _program_files()
            for vcvars in root.glob(
                "Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat"
            )
            if vcvars.is_file()
        ),
        reverse=True,
    )
    return found[0] if found else None


def _vswhere(arguments: list[str]) -> list[Path]:
    """Installation paths reported by vswhere, or nothing if it cannot answer."""
    vswhere = None
    for root in _program_files():
        candidate = root / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if candidate.is_file():
            vswhere = candidate
            break
    if vswhere is None:
        return []
    command = [
        str(vswhere),
        "-products",
        "*",
        # The workload alone is not enough: it can be installed without the x64
        # compiler, which is the one component this build actually needs.
        "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "-property",
        "installationPath",
        "-utf8",
        *arguments,
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False, timeout=60,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []
    return [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def _program_files() -> tuple[Path, ...]:
    roots = (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"))
    return tuple(Path(root) for root in roots if root) or (
        Path(r"C:\Program Files (x86)"),
        Path(r"C:\Program Files"),
    )


def find_ninja(environment: dict[str, str] | None = None) -> Path | None:
    """Ninja from PATH, else the copy every Visual Studio C++ install ships.

    Worth looking for: NMake Makefiles, the previous Windows generator, builds
    strictly one file at a time no matter what `--parallel` says, and this
    runtime is a few dozen heavyweight AVX-512 and kernel translation units.
    """
    path = (environment or os.environ).get("PATH")
    on_path = shutil.which("ninja", path=path)
    if on_path:
        return Path(on_path)
    vcvars = find_msvc()
    if vcvars is None:
        return None
    bundled = (
        vcvars.parents[3]
        / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "Ninja"
        / "ninja.exe"
    )
    return bundled if bundled.is_file() else None


def _generator(environment: dict[str, str]) -> str | None:
    """The cmake generator to force, or None to accept the platform default."""
    if os.name != "nt":
        return None
    ninja = find_ninja(environment)
    if ninja is None:
        return "NMake Makefiles"
    # The bundled copy is not on PATH, and cmake resolves the build program by
    # name; prepending its directory is what makes -G Ninja usable at all.
    _prepend_path(environment, ninja.parent)
    return "Ninja"


def _prepend_path(environment: dict[str, str], directory: Path) -> None:
    for name in tuple(environment):
        if name.casefold() == "path":
            environment["PATH"] = f"{directory}{os.pathsep}{environment.pop(name)}"
            return
    environment["PATH"] = str(directory)


def _cached_generator(build: Path) -> str | None:
    """The generator a configured build tree already uses, if it is configured."""
    cache = build / "CMakeCache.txt"
    if not cache.is_file():
        return None
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("CMAKE_GENERATOR:"):
            return line.split("=", 1)[1].strip() if "=" in line else None
    return None


def _build_environment() -> dict[str, str]:
    environment = os.environ.copy()
    if os.name != "nt":
        return environment
    # Always prefer the x64 toolchain via vcvars64. An x86/Developer prompt puts
    # a 32-bit cl.exe on PATH, which cannot compile the F16C intrinsics used by
    # the AVX2 kernels (e.g. _cvtsh_ss), so trusting an existing cl is unsafe.
    vcvars = find_msvc()
    if vcvars is None:
        # Fall back to an existing x64 cl on PATH if one is already configured.
        if shutil.which("cl", path=environment.get("PATH")):
            return environment
        raise FileNotFoundError(MSVC_MISSING)
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
