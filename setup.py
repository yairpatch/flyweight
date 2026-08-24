from __future__ import annotations

import os
import subprocess
from pathlib import Path

from setuptools import Distribution, setup
from setuptools._distutils.util import get_platform
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py


class BinaryDistribution(Distribution):
    """Mark the distribution as platform-specific without a CPython extension."""

    def has_ext_modules(self) -> bool:
        return True


class CMakeBuildPy(build_py):
    """Compile the native runtime directly into the wheel staging directory."""

    def run(self) -> None:
        super().run()
        root = Path(__file__).resolve().parent
        build_dir = (Path(self.build_lib).parent / "cmake-native").resolve()
        output_dir = (
            Path(self.build_lib) / "colibri_next" / "_native"
        ).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        # build_py may have copied a locally built library from the source
        # tree. Remove every platform variant before compiling so a reused
        # build directory can never produce a mixed-OS wheel.
        for suffix in (".so", ".dylib", ".dll"):
            candidate = output_dir / f"colibri_v2{suffix}"
            if candidate.exists():
                candidate.unlink()
        configure = [
            "cmake",
            "-S",
            str(root / "native"),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
            "-DCOLIBRI_BUILD_DEVELOPMENT_TARGETS=OFF",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={output_dir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={output_dir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={output_dir}",
        ]
        if os.name == "nt":
            configure.extend(["-G", "NMake Makefiles"])
        subprocess.run(configure, check=True)
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--config", "Release"],
            check=True,
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
