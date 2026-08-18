// What the tool-call constraint must and must not forbid.
//
// The engine decides which tokens the sampler may pick once a call opens, so
// two failures matter in opposite directions. Too permissive and it does not
// prevent the malformed call it exists for. Too strict and it makes some
// legitimate continuation unsamplable -- which does not show up as a bad call,
// it shows up as the model being unable to finish a sentence.

#include "colibri_v2_tool_grammar.hpp"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (condition) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

// opencode's two most-used tools: one whose second parameter the model likes
// to skip, one whose value is long.
const char* kSpecification = R"([
  {"name": "bash", "parameters": [
    {"name": "command", "required": true},
    {"name": "description", "required": true}]},
  {"name": "write", "parameters": [
    {"name": "filePath", "required": true},
    {"name": "content", "required": true},
    {"name": "mode", "required": false}]}
])";

colibri::v2::tools::Grammar build() {
    return colibri::v2::tools::Grammar(
        colibri::v2::tools::parse_specification(kSpecification));
}

// Feed `text`, then ask whether `candidate` could come next.
bool allows(const std::string& text, const std::string& candidate) {
    auto grammar = build();
    grammar.observe(text);
    return grammar.accepts(candidate);
}

void check_prose_is_untouched() {
    auto grammar = build();
    grammar.observe("Sure, I can help with that. Here is what I found:");
    expect(!grammar.armed(), "prose does not arm the constraint");
    expect(grammar.accepts("anything at all"), "prose accepts any token");
}

void check_opening_commits_to_a_tool() {
    expect(allows("<tool_call>", "<function=bash>"), "a declared tool may open");
    expect(allows("<tool_call>", "<function=wr"), "a tool name may arrive in pieces");
    expect(!allows("<tool_call>", "<function=curl>"), "an undeclared tool may not open");
    expect(!allows("<tool_call>", "hello"), "prose may not follow the opening tag");
}

void check_required_parameters_cannot_be_skipped() {
    // The failure this whole file exists for: the model writes `command` and
    // then wants to close.
    const std::string opened =
        "<tool_call><function=bash><parameter=command>ls -la</parameter>";
    expect(!allows(opened, "</function>"),
           "closing with a required parameter outstanding is refused");
    expect(allows(opened, "<parameter=description>"),
           "the outstanding parameter may still be written");

    const std::string complete = opened +
        "<parameter=description>List files</parameter>";
    expect(allows(complete, "</function></tool_call>"),
           "closing is allowed once every required parameter is written");
}

void check_optional_parameters_are_optional() {
    const std::string written =
        "<tool_call><function=write><parameter=filePath>a.py</parameter>"
        "<parameter=content>print(1)</parameter>";
    expect(allows(written, "</function></tool_call>"),
           "an optional parameter may be left out");
    expect(allows(written, "<parameter=mode>"),
           "an optional parameter may be written");
}

void check_a_parameter_cannot_repeat() {
    const std::string written =
        "<tool_call><function=bash><parameter=command>ls</parameter>";
    expect(!allows(written, "<parameter=command>"),
           "a parameter already written may not be written again");
}

void check_values_are_free_text() {
    const std::string open =
        "<tool_call><function=write><parameter=filePath>a.py</parameter>"
        "<parameter=content>";
    expect(allows(open, "def main():\n    print(\"hi\")\n"),
           "a value takes arbitrary text");
    // A value containing something that looks like the start of the closing
    // tag must not derail the match.
    expect(allows(open, "if a < b and c </ d:"),
           "a value may contain the closing tag's opening characters");
    expect(allows(open + "x</parameter", ">"),
           "a closing tag may arrive one character at a time");
}

void check_the_constraint_disarms() {
    auto grammar = build();
    grammar.observe("<tool_call><function=bash><parameter=command>ls</parameter>"
                    "<parameter=description>list</parameter></function></tool_call>");
    expect(!grammar.armed(), "the constraint disarms when the call closes");
    expect(grammar.accepts("and then some prose"),
           "text after a closed call is unconstrained");
}

void check_a_bypassed_sampler_does_not_deadlock() {
    // Nothing should be able to leave the grammar with no positions: a
    // resumed conversation replays text this engine never sampled.
    auto grammar = build();
    grammar.observe("<tool_call><function=unknown_tool>");
    expect(!grammar.armed(), "an impossible call disarms rather than sticking");
    expect(grammar.accepts("whatever follows"), "and stops constraining");
}

void check_no_tools_means_no_constraint() {
    colibri::v2::tools::Grammar grammar{};
    expect(grammar.empty(), "an empty specification yields an empty grammar");
    grammar.observe("<tool_call>");
    expect(grammar.accepts("anything"), "an empty grammar constrains nothing");
}

}  // namespace

int main() {
    check_prose_is_untouched();
    check_opening_commits_to_a_tool();
    check_required_parameters_cannot_be_skipped();
    check_optional_parameters_are_optional();
    check_a_parameter_cannot_repeat();
    check_values_are_free_text();
    check_the_constraint_disarms();
    check_a_bypassed_sampler_does_not_deadlock();
    check_no_tools_means_no_constraint();

    if (failures) {
        std::printf("tool_grammar_contract: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("tool_grammar_contract: ok\n");
    return 0;
}
