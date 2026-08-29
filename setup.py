from __future__ import annotations

import importlib.util
from pathlib import Path

from setuptools import Distribution, setup
from setuptools._distutils.util import get_platform
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py


class BinaryDistribution(Distribution):
    """Mark the distribution as platform-specific without a CPython extension."""

    def has_ext_modules(self) -> bool:
        return True


def _native_build_module():
    """Load `flyweight.native_build` by path, without importing the package.

    Importing `flyweight` would pull in the runtime and its dependencies, which
    a build has no business requiring. The module itself is dependency-free.
    """
    path = Path(__file__).resolve().parent / "src" / "flyweight" / "native_build.py"
    spec = importlib.util.spec_from_file_location("_flyweight_native_build", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the native build helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CMakeBuildPy(build_py):
    """Compile the native runtime into whichever tree is being installed."""

    def run(self) -> None:
        super().run()
        native_build = _native_build_module()
        root = Path(__file__).resolve().parent
        # An editable install must land the library in the SOURCE tree. Its
        # build_lib is a temporary directory that pip deletes, so building into
        # the staging path compiled the whole runtime and then threw it away --
        # `pip install -e .` appeared to succeed and left nothing importable.
        if getattr(self, "editable_mode", False):
            output_dir = root / "src" / "flyweight" / "_native"
        else:
            output_dir = (Path(self.build_lib) / "flyweight" / "_native").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        # build_py may have copied a locally built library from the source
        # tree. Remove every platform variant before compiling so a reused
        # build directory can never produce a mixed-OS wheel.
        for suffix in (".so", ".dylib", ".dll"):
            candidate = output_dir / f"{native_build.LIBRARY_STEM}{suffix}"
            if candidate.exists():
                candidate.unlink()
        native_build.build_native(
            output=output_dir,
            build_dir=root / "build" / "native",
            development_targets=False,
        )


class PlatformWheel(bdist_wheel):
    """The ctypes library is Python-ABI independent, but not OS independent."""

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        return "py3", "none", get_platform().replace("-", "_")


setup(
    distclass=BinaryDistribution,
    cmdclass={"build_py": CMakeBuildPy, "bdist_wheel": PlatformWheel},
)
