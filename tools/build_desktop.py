"""Bundle Flyweight Desktop into a standalone app folder with PyInstaller.

The result needs no Python on the target machine: unzip, run `Flyweight`
(`Flyweight.exe` on Windows). It is built from an INSTALLED flyweight -- the
wheel, with its compiled native library -- never from a bare checkout, so what
ships is exactly what pip would have installed.

    pip install flyweight_llm-*.whl pyinstaller
    python tools/build_desktop.py   # -> dist/Flyweight/ and a .zip/.tar.gz

The engine runs as a second copy of the same executable (`--engine serve
...`), so the one bundle is both the window and the server. GPU drivers and
the CUDA toolkit are still the machine's own, exactly as for the wheel.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    spec = importlib.util.find_spec("flyweight")
    if spec is None or spec.origin is None:
        raise SystemExit("install the flyweight wheel first (pip install flyweight-llm)")
    package = Path(spec.origin).parent
    if package.is_relative_to(ROOT / "src"):
        raise SystemExit("this is the source tree; install the built wheel into a "
                         "clean environment and run from there")
    if not any((package / "_native").glob("*flyweight*")):
        raise SystemExit(f"no native library under {package / '_native'}")
    if importlib.util.find_spec("PyInstaller") is None:
        raise SystemExit("pip install pyinstaller")

    work = ROOT / "build" / "desktop"
    shutil.rmtree(work, ignore_errors=True)
    entry = work / "flyweight_desktop.py"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("from flyweight.desktop import main\nraise SystemExit(main())\n",
                     encoding="utf-8")

    command = [
        sys.executable, "-m", "PyInstaller", str(entry),
        "--name", "Flyweight", "--noconfirm", "--clean", "--onedir",
        "--distpath", str(args.dist), "--workpath", str(work / "pyinstaller"),
        "--specpath", str(work),
        # The package is loaded lazily by the CLI (the runtime, the per-model
        # servers, the UI bundle and the native library), so name all of it.
        "--collect-all", "flyweight",
        "--collect-all", "jinja2",
        "--hidden-import", "tkinter",
    ]
    # `_native` is not a Python package, and --collect-all skips the library
    # inside it; without this line the bundle opens and cannot load a model.
    for library in sorted((package / "_native").iterdir()):
        if library.is_file():
            command += ["--add-binary", f"{library}{os.pathsep}flyweight/_native"]
    if importlib.util.find_spec("PIL") is not None:
        command += ["--collect-submodules", "PIL"]
    if sys.platform == "win32":
        command.append("--windowed")  # no console window behind the app
    subprocess.run(command, check=True)

    bundle = args.dist / "Flyweight"
    if args.no_zip:
        print(bundle)
        return 0
    from flyweight.cli import _version

    system = {"win32": "windows", "linux": "linux"}.get(sys.platform, sys.platform)
    machine = platform.machine().lower().replace("amd64", "x86_64")
    archive = args.dist / f"Flyweight-{_version()}-{system}-{machine}"
    # tar keeps the executable bits a Linux bundle needs; Windows gets a zip.
    kind = "zip" if sys.platform == "win32" else "gztar"
    print(shutil.make_archive(str(archive), kind, root_dir=args.dist, base_dir="Flyweight"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
