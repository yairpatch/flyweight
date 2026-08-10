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


    def _reveal_controller(self, source: str) -> str:
        """The smooth-reveal block, driven here off a fake clock."""
        return self._slice(
            source,
            "const REVEAL_DRAIN_SECONDS",
            "\n// Reveals whatever is left at once",
        )

    def _reveal_harness(self, source: str, body: str) -> str:
        """`smoothTick` under a stubbed DOM, clock and frame scheduler."""
        return f"""
let clock = 0;
const performance = {{ now: () => clock }};
let scheduled = null;
function requestAnimationFrame(fn) {{ scheduled = fn; return 1; }}
function cancelAnimationFrame() {{ scheduled = null; }}
function clampNumber(value, minimum, maximum, fallback) {{
  const number = Number(value);
  return Number.isFinite(number)
    ? Math.min(maximum, Math.max(minimum, number))
    : fallback;
}}
let painted = "";
function updateStreamingMessage(message, text) {{ painted = text; return true; }}
function isFollowingStream() {{ return false; }}
function scrollToBottom() {{}}
function flushSmoothStream() {{}}
function finishSmoothStream() {{
  const done = smooth.onDone;
  cancelAnimationFrame();
  smooth.message = null;
  smooth.onDone = null;
  done?.();
}}
{self._reveal_controller(source)}
// One 60 Hz frame. A hidden tab is modelled by advancing the clock without
// running the frame the browser never delivers.
function frame() {{
  clock += 1000 / 60;
  const pending = scheduled;
  scheduled = null;
  pending?.(clock);
}}
{body}
"""

    def test_reveal_does_not_decelerate_into_the_end_of_a_stream(self) -> None:
        """The tail of an answer must not crawl.

        Speed proportional to the unrevealed backlog decays exponentially once
        the backlog stops growing, so the last characters of every answer used
        to arrive in slow motion at the `REVEAL_MIN_CPS` floor. Finishing holds
        the pace the stream had reached instead.
        """
        source = APP.read_text(encoding="utf-8")
        body = """
const message = { content: "" };
beginSmoothStream(message);
// Two seconds of steady decode at 120 chars/second.
for (let i = 0; i < 120; i += 1) {
  message.content += "ab";
  frame();
}
const backlog = message.content.length - smooth.revealed;
if (backlog < 8) throw new Error(`expected a buffered backlog, got ${backlog}`);
let settled = false;
endSmoothStream(() => { settled = true; });
const finishingAt = clock;
for (let i = 0; i < 600 && !settled; i += 1) frame();
if (!settled) throw new Error("the reveal never settled");
const tail = (clock - finishingAt) / 1000;
// At the pace it was already running, the backlog takes about
// REVEAL_DRAIN_SECONDS to drain; the old decay took more than twice that.
if (tail > 0.35) throw new Error(`tail drained in ${tail}s, expected <= 0.35s`);
if (painted !== message.content) throw new Error("the full answer was not painted");
"""
        subprocess.run(
            [self.node, "-e", self._reveal_harness(source, body)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )

    def test_returning_to_a_hidden_page_does_not_replay_the_stream(self) -> None:
        """A stream that ended while the page was hidden is already over.

        requestAnimationFrame is suspended in a hidden tab while the SSE reader
        keeps filling the message, so the reveal would otherwise come back an
        entire answer behind and animate text the runtime finished long ago.
        """
        source = APP.read_text(encoding="utf-8")
        body = """
const message = { content: "" };
beginSmoothStream(message);
for (let i = 0; i < 30; i += 1) { message.content += "ab"; frame(); }
// The page is hidden: the reader keeps running, no frames are delivered.
const away = smooth.revealed;
for (let i = 0; i < 400; i += 1) { message.content += "xy"; clock += 1000 / 60; }
let settled = false;
endSmoothStream(() => { settled = true; });
clock += 30_000;
if (settled) throw new Error("a hidden page cannot have settled the request");
if (smooth.revealed !== away) throw new Error("a hidden page cannot have revealed text");
// ...and the reader comes back.
catchUpSmoothStream();
if (!settled) throw new Error("returning did not settle the finished request");
const remaining = message.content.length - smooth.revealed;
if (remaining > 0) throw new Error(`${remaining} characters were left to replay`);
"""
        subprocess.run(
            [self.node, "-e", self._reveal_harness(source, body)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )


if __name__ == "__main__":
    unittest.main()
