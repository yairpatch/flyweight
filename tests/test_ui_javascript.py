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

    def test_markdown_block_parser_always_advances(self) -> None:
        source = APP.read_text(encoding="utf-8")
        start = source.index("function appendMarkdownBlock(")
        end = source.index("\nfunction appendInline(", start)
        parser = source[start:end]
        cases = [
            "| name | value |\n| --- | --- |\n| a | b |",
            "#heading-without-space",
            ">quote-without-space",
            "normal paragraph\n| table-like continuation |",
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


if __name__ == "__main__":
    unittest.main()
