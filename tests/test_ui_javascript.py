from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path


APP = Path(__file__).parents[1] / "src" / "colibri_next" / "ui" / "app.js"


class UIJavaScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.node = shutil.which("node")
        if cls.node is None:
            raise unittest.SkipTest("Node.js is required for UI parser tests")

    def test_app_is_valid_javascript(self) -> None:
        subprocess.run(
            [self.node, "--check", str(APP)],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_runtime_badge_uses_resolved_expert_mode(self) -> None:
        source = APP.read_text(encoding="utf-8")
        self.assertIn('execution.backend === "native-v2-cpp-cuda"', source)
        self.assertIn("execution.expert_mode", source)
        self.assertIn("execution.expert_fallback_reason", source)

    @staticmethod
    def _slice(source: str, start_marker: str, end_marker: str) -> str:
        start = source.index(start_marker)
        return source[start : source.index(end_marker, start)]

    def _direction_helpers(self, source: str) -> str:
        """The bidi block, which the markdown parser calls into."""
        return self._slice(source, "const RTL_LETTERS", "\nfunction applyMessageDirection(")

    def test_markdown_block_parser_always_advances(self) -> None:
        source = APP.read_text(encoding="utf-8")
        parser = self._slice(
            source, "function appendMarkdownBlock(", "\nfunction appendInline("
        )
        cases = [
            "| name | value |\n| --- | --- |\n| a | b |",
            "#heading-without-space",
            ">quote-without-space",
            "normal paragraph\n| table-like continuation |",
            # Right-to-left input runs the same control flow through the
            # direction helpers the parser now calls on every block.
            "שלום עולם\n\n- פריט ראשון\n- פריט שני",
        ]
        harness = f"""
class FakeNode {{
  constructor(tag = "fragment") {{ this.tag = tag; this.children = []; }}
  append(...children) {{ this.children.push(...children); }}
}}
const document = {{
  createElement(tag) {{ return new FakeNode(tag); }}
}};
function appendInline(parent, text) {{ parent.append(String(text)); }}
{self._direction_helpers(source)}
{parser}
for (const value of {json.dumps(cases)}) {{
  const root = new FakeNode();
  appendMarkdownBlock(root, value);
  if (!root.children.length) throw new Error("parser emitted no nodes");
}}
"""
        subprocess.run(
            [self.node, "-e", harness],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )

    def test_direction_detection_weighs_the_whole_block(self) -> None:
        """A block is laid out by which script dominates it.

        `dir="auto"` decides on the first strong character alone, which puts a
        Hebrew answer that opens with a Latin product name entirely
        left-to-right. These cases pin the weighted behaviour that replaced it.
        """
        source = APP.read_text(encoding="utf-8")
        cases = [
            ("Hello world, this is English.", "ltr"),
            ("שלום עולם, זה טקסט בעברית.", "rtl"),
            # Hebrew prose quoting a Latin identifier stays right-to-left.
            ("שלום, אני משתמש ב-CUDA כדי להריץ מודל מקומי.", "rtl"),
            # ...and one Hebrew word inside English prose does not flip it.
            ("The Hebrew word for peace is שלום.", "ltr"),
            ("مرحبا بالعالم، هذا نص عربي.", "rtl"),
            # Nothing directional: inherit rather than assert a direction.
            ("1. 2. 3. 42 %", None),
            ("", None),
        ]
        harness = f"""
{self._direction_helpers(source)}
const cases = {json.dumps(cases)};
for (const [text, expected] of cases) {{
  const actual = detectDirection(text)?.dir ?? null;
  if (actual !== expected) {{
    throw new Error(`${{JSON.stringify(text)}}: expected ${{expected}}, got ${{actual}}`);
  }}
}}
const element = {{}};
applyDirection(element, "שלום עולם");
if (element.dir !== "rtl") throw new Error("applyDirection did not set dir");
const untouched = {{}};
applyDirection(untouched, "12345");
if ("dir" in untouched) throw new Error("a block with no letters must inherit direction");
"""
        subprocess.run(
            [self.node, "-e", harness],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )


if __name__ == "__main__":
    unittest.main()
