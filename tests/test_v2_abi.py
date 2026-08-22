"""Every native entry point the bindings call must exist in the library.

A ctypes binding is resolved lazily, at the call: a function declared in
`v2.py` with no implementation behind it builds, imports, passes every test
that does not reach it, and then fails in production with

    undefined symbol: colibri_v2_bailing_set_progress

That is exactly what shipped -- the BailingMoE3 progress hook and, behind it,
the whole snapshot half of its prefix cache. Nothing in the suite could catch
it, because the stubs the server tests run against answer for the runtime.

So check the real library instead: pull every `colibri_v2_*` spelling out of
the bindings and demand the loaded library define it.
"""

from __future__ import annotations

import ctypes
import re
import unittest
from pathlib import Path

from colibri_next import v2

SOURCE = Path(v2.__file__)


class NativeAbiTests(unittest.TestCase):
    def test_every_referenced_entry_point_is_defined(self) -> None:
        names = sorted(set(re.findall(r"colibri_v2_\w+", SOURCE.read_text())))
        # A guard that guards nothing would pass just as quietly.
        self.assertGreater(len(names), 20)

        library = v2._library()
        missing = []
        for name in names:
            try:
                getattr(library, name)
            except AttributeError:
                missing.append(name)
        self.assertEqual(
            missing, [],
            "the bindings call native entry points the library does not "
            "define; these fail at the call, not at import",
        )

    def test_the_check_would_notice_an_absent_symbol(self) -> None:
        # The property above is only worth having if it can fail, and a
        # library lookup that never raises would make it vacuous.
        library = v2._library()
        with self.assertRaises(AttributeError):
            getattr(library, "colibri_v2_a_function_that_does_not_exist")
        self.assertIsInstance(library, ctypes.CDLL)


if __name__ == "__main__":
    unittest.main()
