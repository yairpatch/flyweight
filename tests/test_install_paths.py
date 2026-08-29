"""The install plumbing: one build entry point, and honest failure messages.

These guard the two ways this went wrong. `pip install -e .` compiled the whole
runtime into a temporary directory pip then deleted, so the install appeared to
succeed and left nothing importable -- and CI never ran that path, so nothing
noticed. And the "not built" error told everyone to run a build command that
cannot work from an installed copy, because there is no `native/` beside it.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from flyweight import native_build
from flyweight.cli import main


class NativeBuildTests(unittest.TestCase):
    def test_the_checkout_finds_its_own_sources(self) -> None:
        source = native_build.source_root()
        self.assertIsNotNone(source)
        assert source is not None
        self.assertTrue((source / "CMakeLists.txt").is_file())

    def test_building_without_sources_says_so_instead_of_failing_in_cmake(self) -> None:
        # An installed copy has the package and no `native/`. Reaching cmake at
        # all would surface a missing-CMakeLists error naming a path the reader
        # never chose, so the refusal happens here with the reason attached.
        with patch.object(native_build, "source_root", return_value=None):
            with self.assertRaises(FileNotFoundError) as raised:
                native_build.build_native()
        message = str(raised.exception)
        self.assertIn("no native sources", message)
        self.assertIn("wheel", message)


class DoctorTests(unittest.TestCase):
    def _run(self) -> tuple[int, str]:
        printed: list[str] = []
        with patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
            code = main(["doctor"])
        return code, "\n".join(printed)

    def test_a_missing_library_fails_and_names_the_build_command(self) -> None:
        with patch.object(native_build, "default_output", return_value=Path("/nonexistent")):
            code, output = self._run()
        self.assertEqual(code, 1)
        self.assertIn("native runtime: not built", output)
        self.assertIn("python -m flyweight.native_build", output)
        # Without a library the GPU cannot be probed; saying so beats repeating
        # the same failure as a second, unrelated-looking problem.
        self.assertIn("not probed", output)

    def test_an_installed_copy_is_not_told_to_run_a_build_it_cannot(self) -> None:
        with (
            patch.object(native_build, "default_output", return_value=Path("/nonexistent")),
            patch.object(native_build, "source_root", return_value=None),
        ):
            code, output = self._run()
        self.assertEqual(code, 1)
        self.assertNotIn("python -m flyweight.native_build", output)
        self.assertIn("wheel", output)

    def test_a_healthy_checkout_reports_that_it_can_serve(self) -> None:
        if not any(
            (native_build.default_output() / f"{native_build.LIBRARY_STEM}{suffix}").is_file()
            for suffix in (".so", ".dylib", ".dll")
        ):
            self.skipTest("native library not built in this checkout")
        code, output = self._run()
        self.assertEqual(code, 0, output)
        self.assertIn("this install can serve", output)


if __name__ == "__main__":
    unittest.main()
