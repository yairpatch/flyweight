import { describe, expect, it } from "vitest";
import { executorDocument } from "./executor";

describe("executorDocument", () => {
  it("embeds the handler, arguments, and token as inert JSON", () => {
    const doc = executorDocument('return "hi " + args.name;', '{"name":"world"}', "exec-1");
    expect(doc).toContain(JSON.stringify('return "hi " + args.name;'));
    expect(doc).toContain(JSON.stringify('{"name":"world"}'));
    expect(doc).toContain('"exec-1"');
  });

  it("keeps a handler containing </script> from closing the script block", () => {
    const doc = executorDocument('return "</script><img src=x>";', "{}", "exec-2");
    // Only the template's own closing tag may appear; the handler's copy is escaped.
    expect(doc.match(/<\/script>/g)).toHaveLength(1);
    expect(doc).toContain("\\u003c/script>");
  });

  it("escapes < in the arguments text too", () => {
    const doc = executorDocument("return args;", '{"html":"</script>"}', "exec-3");
    expect(doc.match(/<\/script>/g)).toHaveLength(1);
  });
});
