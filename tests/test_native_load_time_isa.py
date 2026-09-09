"""No initializer in the native runtime may need more than the x86-64 baseline.

The runtime dispatches AVX2 and AVX-512 kernels after probing the CPU, but a
static initializer runs before any probe: at dlopen on Linux and in DllMain on
Windows. A namespace-scope `__m512i` built with `_mm512_set_*` gave both
compilers a dynamic initializer full of zmm moves, and every Intel consumer
part since Alder Lake failed to load the library with an illegal instruction
(WinError 1114 on Windows, SIGILL on Linux). CI never saw it because the
hosted runners have AVX-512.

This walks the startup code of every object in the build tree and fails on
any ymm, zmm, or mask-register use there. It needs the build tree and objdump,
so it skips where either is absent.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path

from flyweight import native_build

WIDE_REGISTER = re.compile(r"%(ymm|zmm|k)\d")


def _objects() -> list[Path]:
    source = native_build.source_root()
    if source is None:
        return []
    build = source.parent / "build" / "native" / "CMakeFiles" / "flyweight_v2.dir"
    return sorted(build.rglob("*.o"))


class LoadTimeIsaTests(unittest.TestCase):
    def test_static_initializers_use_no_wide_vector_registers(self) -> None:
        objdump = shutil.which("objdump")
        objects = _objects()
        if objdump is None or not objects:
            self.skipTest("needs objdump and a native build tree")
        offenders: list[str] = []
        for path in objects:
            # GCC and Clang place dynamic initializers in .text.startup.
            listing = subprocess.run(
                [objdump, "-d", "--section=.text.startup", str(path)],
                capture_output=True, text=True, check=False,
            ).stdout
            wide = [line.strip() for line in listing.splitlines() if WIDE_REGISTER.search(line)]
            if wide:
                offenders.append(f"{path.name}: {wide[0]}")
        self.assertEqual(
            offenders, [],
            "static initializers that need AVX2/AVX-512 run before the CPU is probed; "
            "hold such constants as plain integer arrays and load them in the kernel",
        )
