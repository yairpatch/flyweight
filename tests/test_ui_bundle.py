"""The web UI ships as a committed Vite build under src/flyweight/ui.

The source lives in web/; these tests check the build that the server
actually serves is present and self-consistent, and run the web unit tests
when a Node toolchain with installed dependencies is available.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
UI = ROOT / "src" / "flyweight" / "ui"
WEB = ROOT / "web"


class UIBundleTests(unittest.TestCase):
    def test_index_references_existing_hashed_assets(self) -> None:
        html = (UI / "index.html").read_text(encoding="utf-8")
        self.assertIn("Flyweight Chat", html)
        # No inline scripts: the server's CSP only allows same-origin files.
        self.assertNotRegex(html, r"<script(?![^>]*\bsrc=)")
        assets = re.findall(r'(?:src|href)="/assets/([^"]+)"', html)
        self.assertTrue(assets, "index.html names no /assets/ files; run `pnpm build` in web/")
        for asset in assets:
            self.assertTrue((UI / "assets" / asset).is_file(), asset)
        self.assertTrue(any(asset.endswith(".js") for asset in assets))
        self.assertTrue(any(asset.endswith(".css") for asset in assets))

    def test_preview_shell_and_icon_are_bundled(self) -> None:
        self.assertIn("location.hash", (UI / "preview.html").read_text(encoding="utf-8"))
        self.assertTrue((UI / "favicon.svg").is_file())

    def test_bundle_targets_every_server_endpoint(self) -> None:
        javascript = "".join(path.read_text(encoding="utf-8") for path in (UI / "assets").glob("*.js"))
        for endpoint in (
            "/v1/chat/completions",
            "/v1/messages",
            "/v1/messages/count_tokens",
            "/v1/responses",
            "/v1/responses/input_tokens",
            "/v1/completions",
            "/v1/models",
            "/health",
            "/props",
            "/slots",
            "/tokenize",
            "/detokenize",
        ):
            self.assertIn(endpoint, javascript, endpoint)

    def test_web_unit_tests(self) -> None:
        if shutil.which("node") is None or not (WEB / "node_modules").is_dir():
            raise unittest.SkipTest("web/node_modules not installed")
        runner = WEB / "node_modules" / ".bin" / "vitest"
        if not runner.exists():
            raise unittest.SkipTest("vitest not installed")
        result = subprocess.run(
            [str(runner), "run"], cwd=WEB, capture_output=True, text=True, timeout=300
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
