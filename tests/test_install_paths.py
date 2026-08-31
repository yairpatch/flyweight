"""The install plumbing: one build entry point, and honest failure messages.

These guard the two ways this went wrong. `pip install -e .` compiled the whole
runtime into a temporary directory pip then deleted, so the install appeared to
succeed and left nothing importable -- and CI never ran that path, so nothing
noticed. And the "not built" error told everyone to run a build command that
cannot work from an installed copy, because there is no `native/` beside it.
"""

from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from flyweight import native_build
from flyweight.cli import main

# Bound before any patch replaces it. The doctor calls which() for cmake and the
# compiler too, so the tests below have to answer for "flyweight" and defer the
# rest -- deferring to the patched name instead recurses until the stack ends.
real_which = shutil.which


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


@unittest.skipUnless(os.name == "nt", "the Windows toolchain lookup")
class WindowsToolchainTests(unittest.TestCase):
    """Finding MSVC the way Visual Studio says to, not by guessing paths.

    The old lookup tried three hardcoded Community/BuildTools directories, so a
    machine with Professional, Enterprise, a Preview channel, or a non-C: drive
    was told to install a compiler it already had -- by the build as an error,
    and by `flyweight doctor` as a blocking failure.
    """

    def test_the_toolchain_is_found_on_a_machine_that_can_build(self) -> None:
        vcvars = native_build.find_msvc()
        if vcvars is None:
            self.skipTest("no Visual Studio C++ tools on this machine")
        self.assertTrue(vcvars.is_file())
        self.assertEqual(vcvars.name, "vcvars64.bat")

    def test_a_professional_install_is_found_through_vswhere(self) -> None:
        # vswhere reports an installation root; everything else is derived. The
        # edition and the drive are exactly what a hardcoded list cannot cover.
        install = Path(r"D:\VS\2022\Professional")
        with (
            patch.object(native_build, "_vswhere", return_value=[install]),
            patch.object(Path, "is_file", lambda self: True),
        ):
            found = native_build.find_msvc()
        self.assertEqual(
            found, install / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        )

    def test_no_toolchain_yields_an_instruction_rather_than_a_cmake_crash(self) -> None:
        with (
            patch.object(native_build, "find_msvc", return_value=None),
            patch.object(native_build.shutil, "which", return_value=None),
        ):
            with self.assertRaises(FileNotFoundError) as raised:
                native_build._build_environment()
        # The fix has to be runnable, not a description of a checkbox.
        self.assertIn("winget install", str(raised.exception))

    def test_ninja_is_preferred_so_the_build_is_not_serial(self) -> None:
        # NMake Makefiles ignores --parallel entirely, which is the difference
        # between one core and all of them over a few dozen AVX-512 units.
        with patch.object(native_build, "find_ninja", return_value=Path("ninja.exe")):
            environment = {"PATH": "C:\\existing"}
            self.assertEqual(native_build._generator(environment), "Ninja")
            self.assertTrue(environment["PATH"].startswith("."))
        with patch.object(native_build, "find_ninja", return_value=None):
            self.assertEqual(native_build._generator({}), "NMake Makefiles")

    def test_switching_generator_does_not_strand_an_existing_build_tree(self) -> None:
        # cmake refuses to reconfigure under a new generator. A checkout built
        # before Ninja must not have its next build turn into that error.
        self.assertIsNone(native_build._cached_generator(Path("/nonexistent")))


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

    def test_an_installed_copy_shadowing_the_checkout_is_reported(self) -> None:
        # Installing a built wheel into the same environment as a checkout
        # makes every import resolve to site-packages: edits look inert, and
        # the library check still passes because the installed copy brings a
        # library of its own. It happened twice in one afternoon.
        with patch.object(
            native_build, "default_output",
            return_value=Path("/elsewhere/flyweight/_native"),
        ):
            _, output = self._run()
        # Named, not spelled "not /": the checkout is reported as an absolute
        # path, which on Windows starts with a drive letter and failed here.
        self.assertIn(f"not {Path(__file__).resolve().parents[1]}", output)
        self.assertIn("shadowing the checkout", output)

    def test_a_console_script_off_path_is_a_warning_naming_the_module_form(self) -> None:
        # The exact end state of a successful Microsoft Store `pip install`:
        # flyweight.exe is written, pip warns that its directory is not on
        # PATH, and the shell then says the command does not exist. Not a
        # failure -- `python -m flyweight` is a complete answer -- but it has
        # to be *said*, because nothing else in the install does.
        script = Path("/scripts-off-path") / (
            "flyweight.exe" if os.name == "nt" else "flyweight")
        with (
            patch("flyweight.cli._console_script", return_value=script),
            patch("shutil.which", lambda name, **k: None if name == "flyweight"
                  else real_which(name, **k)),
        ):
            code, output = self._run()
        self.assertEqual(code, 0, output)
        self.assertIn("not on PATH", output)
        self.assertIn("python -m flyweight", output)

    def test_a_console_script_from_another_install_is_reported_as_shadowing(self) -> None:
        ours = Path("/ours") / "flyweight"
        with (
            patch("flyweight.cli._console_script", return_value=ours),
            patch("shutil.which", lambda name, **k: "/somewhere/else/flyweight"
                  if name == "flyweight" else real_which(name, **k)),
        ):
            _, output = self._run()
        self.assertIn("shadows this one on PATH", output)

    def test_the_user_scheme_scripts_directory_is_searched(self) -> None:
        # A Store Python's default scripts path is under Program
        # Files\WindowsApps, which pip cannot write to, so every console script
        # it installs goes to the nt_user scheme instead. Looking only at the
        # default reported a correctly installed command as missing.
        import sysconfig

        user_scheme = sysconfig.get_preferred_scheme("user")
        user_directory = Path(sysconfig.get_path("scripts", user_scheme))
        name = "flyweight.exe" if os.name == "nt" else "flyweight"
        real_is_file = Path.is_file

        def only_the_user_copy(self: Path) -> bool:
            return self == user_directory / name or real_is_file(self)

        with patch.object(Path, "is_file", only_the_user_copy):
            from flyweight.cli import _console_script

            self.assertEqual(_console_script(), user_directory / name)

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
