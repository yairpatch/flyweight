"""Build the Flyweight Desktop downloads: a Windows installer or a Linux AppImage.

Nothing here is needed to use Flyweight from pip. It turns an INSTALLED
flyweight (the wheel, with its compiled native library) into the files a
person without Python downloads from the release page:

    Windows  dist/Flyweight-Setup-windows-x86_64.exe   (Inno Setup 6, per-user)
    Linux    dist/Flyweight-linux-x86_64.AppImage       (double-click to run)

    pip install flyweight_llm-*.whl pyinstaller
    python tools/build_desktop.py             # the bundle in dist/Flyweight/
    python tools/build_desktop.py --installer # ...and the installer / AppImage

The bundle is one PyInstaller folder that is both the window and the engine
(the app re-enters itself with `--engine serve ...`). The names carry no
version so that .../releases/latest/download/<name> is a stable link.

--installer needs Inno Setup's ISCC.exe on Windows (preinstalled on GitHub's
Windows runners) and downloads appimagetool on Linux unless --appimagetool
names one.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging"
APPIMAGETOOL_URL = ("https://github.com/AppImage/appimagetool/releases/download/"
                    "continuous/appimagetool-x86_64.AppImage")


def installed_package() -> Path:
    spec = importlib.util.find_spec("flyweight")
    if spec is None or spec.origin is None:
        raise SystemExit("install the flyweight wheel first (pip install flyweight-llm)")
    package = Path(spec.origin).parent
    if package.is_relative_to(ROOT / "src"):
        raise SystemExit("this is the source tree; install the built wheel into a "
                         "clean environment and run from outside the checkout")
    if not any((package / "_native").glob("*flyweight*")):
        raise SystemExit(f"no native library under {package / '_native'}")
    return package


def bundle(package: Path, dist: Path, work: Path) -> Path:
    if importlib.util.find_spec("PyInstaller") is None:
        raise SystemExit("pip install pyinstaller")
    entry = work / "flyweight_desktop.py"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("from flyweight.desktop import main\nraise SystemExit(main())\n",
                     encoding="utf-8")
    command = [
        sys.executable, "-m", "PyInstaller", str(entry),
        "--name", "Flyweight", "--noconfirm", "--clean", "--onedir",
        "--distpath", str(dist), "--workpath", str(work / "pyinstaller"),
        "--specpath", str(work),
        # The package is loaded lazily by the CLI (the runtime, the per-model
        # servers), so name all of it.
        "--collect-all", "flyweight",
        "--collect-all", "jinja2",
        # The version the app reports and compares against new releases.
        "--copy-metadata", "flyweight-llm",
        # The last-resort file dialog (`--pick`) when no zenity/kdialog exists.
        "--hidden-import", "tkinter",
    ]
    # `_native`, the chat UI and the desktop page are plain directories, which
    # --collect-all does not reliably pick up; a bundle without them starts
    # and then cannot load a model or show anything.
    for library in sorted((package / "_native").iterdir()):
        if library.is_file():
            command += ["--add-binary", f"{library}{os.pathsep}flyweight/_native"]
    for directory in ("ui", "desktop_ui"):
        if not (package / directory).is_dir():
            raise SystemExit(f"{package / directory} is missing from this install")
        command += ["--add-data", f"{package / directory}{os.pathsep}flyweight/{directory}"]
    if importlib.util.find_spec("PIL") is not None:
        command += ["--collect-submodules", "PIL"]
    if sys.platform == "win32":
        command += ["--windowed", "--icon", str(PACKAGING / "icons" / "flyweight.ico")]
    subprocess.run(command, check=True)
    return dist / "Flyweight"


def windows_installer(app: Path, dist: Path, version: str, iscc: str | None) -> Path:
    candidates = [iscc] if iscc else [
        shutil.which("iscc"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"),
        os.path.expandvars(r"%ProgramFiles%\Inno Setup 6\ISCC.exe"),
        os.path.expandvars(r"%LocalAppData%\Programs\Inno Setup 6\ISCC.exe"),
    ]
    compiler = next((c for c in candidates if c and Path(c).is_file()), None)
    if compiler is None:
        raise SystemExit("Inno Setup 6 not found (choco install innosetup), or pass --iscc")
    subprocess.run([compiler, f"/DAppVersion={version}", f"/DSourceDir={app}",
                    f"/DOutputDir={dist}", str(PACKAGING / "windows" / "flyweight.iss")],
                   check=True)
    return dist / "Flyweight-Setup-windows-x86_64.exe"


def make_appdir(app: Path, work: Path) -> Path:
    """The AppImage's contents: the bundle, AppRun, the menu entry and icon."""
    appdir = work / "Flyweight.AppDir"
    shutil.rmtree(appdir, ignore_errors=True)
    shutil.copytree(app, appdir / "usr" / "lib" / "flyweight", symlinks=True)
    shutil.copy2(PACKAGING / "linux" / "AppRun", appdir / "AppRun")
    (appdir / "AppRun").chmod(0o755)
    shutil.copy2(PACKAGING / "linux" / "flyweight.desktop", appdir / "flyweight.desktop")
    shutil.copy2(PACKAGING / "icons" / "flyweight.png", appdir / "flyweight.png")
    icons = appdir / "usr" / "share" / "icons" / "hicolor" / "512x512" / "apps"
    icons.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PACKAGING / "icons" / "flyweight.png", icons / "flyweight.png")
    return appdir


def appimage(app: Path, dist: Path, work: Path, tool: str | None) -> Path:
    appdir = make_appdir(app, work)
    if tool is None:
        tool = str(work / "appimagetool")
        if not Path(tool).is_file():
            print(f"downloading {APPIMAGETOOL_URL}")
            urllib.request.urlretrieve(APPIMAGETOOL_URL, tool)
        Path(tool).chmod(Path(tool).stat().st_mode | stat.S_IEXEC)
    output = dist / "Flyweight-linux-x86_64.AppImage"
    # CI containers have no FUSE; the tool can run itself unpacked instead.
    env = dict(os.environ, ARCH="x86_64", APPIMAGE_EXTRACT_AND_RUN="1")
    subprocess.run([tool, "--no-appstream", str(appdir), str(output)], check=True, env=env)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    parser.add_argument("--installer", action="store_true",
                        help="also build the Windows installer or the Linux AppImage")
    parser.add_argument("--iscc", help="path to Inno Setup's ISCC.exe")
    parser.add_argument("--appimagetool", help="path to appimagetool")
    args = parser.parse_args()

    package = installed_package()
    dist = args.dist.resolve()
    work = ROOT / "build" / "desktop"
    shutil.rmtree(work / "pyinstaller", ignore_errors=True)
    app = bundle(package, dist, work)
    print(app)
    if not args.installer:
        return 0
    from flyweight.cli import _version

    if sys.platform == "win32":
        print(windows_installer(app, dist, _version(), args.iscc))
    elif sys.platform.startswith("linux"):
        print(appimage(app, dist, work, args.appimagetool))
    else:
        raise SystemExit(f"no installer format for {sys.platform}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
