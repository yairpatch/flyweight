// What the tool-call constraint must and must not forbid.
//
// The engine decides which tokens the sampler may pick once a call opens, so
// two failures matter in opposite directions. Too permissive and it does not
// prevent the malformed call it exists for. Too strict and it makes some
// legitimate continuation unsamplable -- which does not show up as a bad call,
// it shows up as the model being unable to finish a sentence.

#include "colibri_v2_tool_grammar.hpp"
#include "colibri_v2_byte_alphabet.hpp"

#include <cstdio>
#include <string>
#include <vector>

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
    {"name": "mode", "required": false}]},
  {"name": "question", "parameters": [
    {"name": "questions", "required": true, "type": "array"}]},
  {"name": "configure", "parameters": [
    {"name": "settings", "required": true, "type": "object"}]}
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

// The sampler does not see text, it sees candidate *tokens*, and a token is
// stored in the vocabulary's byte-level alphabet. Everything above feeds the
// grammar plain bytes; these two check the step that gets it there, because a
// wrong decode does not fail loudly -- it silently stops matching the literals
// and the constraint quietly permits everything.
// The layout the server's own tool prompt asks for, which is the one the model
// imitates: every tag on its own line. The literals here run together, so
// without whitespace between them the grammar dies on the first newline and
// stops constraining -- silently, because an unconstrained sampler looks
// exactly like a working one until you count malformed calls.
// A parameter the schema declares as an array. The observed failure is the
// model writing prose there: the client's validator then reports "expected
// array, received string", and no amount of tolerant parsing can invent the
// array. So the value is read as JSON rather than passed through.
void check_a_declared_array_must_be_written_as_json() {
    const std::string opened = "<tool_call>\n<function=question>\n<parameter=questions>\n";
    expect(!allows(opened, "What backend would you like"),
           "prose where an array belongs is refused");
    expect(!allows(opened, "{\"question\":"),
           "an object where an array belongs is refused");
    expect(allows(opened, "["), "an array may open");
    expect(allows(opened, "[{\"question\": \"Which backend?\", \"header\": \"Backend\"}"),
           "and may be filled in");
    expect(allows(opened, "[\n  {\n    \"question\": \"Which?\"\n  }\n]"),
           "pretty-printed JSON is fine");
    expect(!allows(opened, "[}"), "an impossible continuation is refused");
    expect(!allows(opened, "[{\"a\": tru}"), "a broken literal is refused");
}

void check_a_declared_array_closes_only_when_whole() {
    const std::string opened = "<tool_call>\n<function=question>\n<parameter=questions>\n";
    expect(!allows(opened + "[{\"question\": \"Which?\"}", "</parameter>"),
           "an unterminated array cannot close the parameter");
    expect(!allows(opened + "[", "</parameter>"),
           "nor can a bare opening bracket");
    expect(allows(opened + "[{\"question\": \"Which?\"}]", "</parameter>"),
           "a complete array can");
    expect(allows(opened + "[]", "</parameter>"), "an empty array is complete");
    expect(allows(opened + "[{\"question\": \"Which?\"}]\n", "</parameter>"),
           "and the layout's trailing newline is allowed before the tag");
}

void check_a_declared_object_is_held_to_the_same_rule() {
    const std::string opened = "<tool_call>\n<function=configure>\n<parameter=settings>\n";
    expect(!allows(opened, "["), "an array where an object belongs is refused");
    expect(allows(opened, "{\"retries\": 3, \"backoff\": \"exponential\"}"),
           "an object may be written");
    expect(!allows(opened + "{\"retries\": 3", "</parameter>"),
           "an unterminated object cannot close the parameter");
    expect(allows(opened + "{\"retries\": 3}", "</parameter>"),
           "a complete one can");
}

// The escape hatch: a string-typed or undeclared parameter is still free text,
// including text that looks like broken JSON. Constraining those would break
// every tool whose value is prose or code.
void check_untyped_values_stay_free_text() {
    const std::string opened =
        "<tool_call>\n<function=write>\n<parameter=content>\n";
    expect(allows(opened, "def main():\n    print(\"{[unbalanced\")"),
           "code with unbalanced brackets is still a legal value");
    expect(allows(opened + "anything at all", "</parameter>"),
           "and it may close whenever the model likes");
}

void check_the_prompted_multi_line_layout_is_accepted() {
    auto grammar = build();
    grammar.observe("<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n");
    expect(grammar.armed(), "a call laid out over lines stays armed");
    expect(!grammar.accepts("</function>"),
           "and still refuses to close with a required parameter outstanding");
    expect(grammar.accepts("<parameter=description>"),
           "and still allows the outstanding parameter");

    auto complete = build();
    complete.observe(
        "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n"
        "<parameter=description>\nList files\n</parameter>\n");
    expect(complete.accepts("</function>\n</tool_call>"),
           "the closing tags may be separated by a newline");
    complete.observe("</function>\n</tool_call>");
    expect(!complete.armed(), "and the call closes");
}

void check_the_byte_alphabet_round_trips() {
    const char* cases[] = {
        "<tool_call>", " leading space", "line\nbreak", "tab\there",
        "plain", "</parameter>", "unicode \xc3\xa9\xe2\x82\xac", " {\"a\": 1}",
    };
    for (const char* text : cases) {
        const std::string encoded = colibri::v2::alphabet::encode(text);
        expect(colibri::v2::alphabet::decode(encoded) == text,
               std::string("byte alphabet round-trips: ") + text);
    }
    // A space is the case that matters: it is the one character that appears in
    // most tokens and is spelled as U+0120 rather than 0x20.
    expect(colibri::v2::alphabet::encode(" ") == "\xc4\xa0",
           "a space encodes to U+0120");
    expect(colibri::v2::alphabet::decode("\xc4\xa0") == " ",
           "U+0120 decodes back to a space");
    // Special tokens are stored literally, not byte-encoded.
    expect(colibri::v2::alphabet::decode("<|im_start|>") == "<|im_start|>",
           "a special token passes through the decoder unchanged");
}

// The failure in the field, replayed the way the sampler meets it: the model
// has written the first parameter and the candidates are vocabulary pieces.
void check_candidate_tokens_are_judged_after_decoding() {
    auto grammar = build();
    const char* written[] = {
        "<tool_call>", "<function=bash>", "<parameter=command>", "ls",
        " -la", "</parameter>",
    };
    for (const char* piece : written)
        grammar.observe(colibri::v2::alphabet::decode(
            colibri::v2::alphabet::encode(piece)));
    expect(grammar.armed(), "the call is still open");

    // Encoded exactly as a vocabulary holds them, leading space and all.
    const auto encoded = [](const char* text) {
        return colibri::v2::alphabet::decode(colibri::v2::alphabet::encode(text));
    };
    expect(!grammar.accepts(encoded("</function>")),
           "closing early is refused when the candidate arrives as a token");
    expect(grammar.accepts(encoded("<parameter=description>")),
           "the outstanding parameter is allowed as a token");
    // Without the decode this one silently fails: the raw spelling of a token
    // carrying a leading space matches no literal, so it would be rejected and
    // the model would be unable to write a value that starts with a space.
    auto inside = build();
    inside.observe("<tool_call><function=write><parameter=content>");
    expect(inside.accepts(encoded(" indented line")),
           "a value token with a leading space is accepted");
}

void check_no_tools_means_no_constraint() {
    colibri::v2::tools::Grammar grammar{};
    expect(grammar.empty(), "an empty specification yields an empty grammar");
    grammar.observe("<tool_call>");
    expect(grammar.accepts("anything"), "an empty grammar constrains nothing");
}

}  // namespace

// --- response_format ------------------------------------------------------
//
// The same two failure directions as the tool constraint: too permissive and
// json_object mode is back to being a polite request; too strict and the model
// cannot finish -- or think -- at all.

colibri::v2::tools::ResponseGrammar build_response(
        colibri::v2::tools::ValueShape shape, bool thinking_open = false) {
    return colibri::v2::tools::ResponseGrammar(shape, thinking_open);
}

void check_response_prose_is_unsamplable() {
    auto grammar = build_response(colibri::v2::tools::ValueShape::json_object);
    expect(grammar.armed(), "the response constraint arms from the first token");
    expect(!grammar.accepts("Sure"), "prose cannot begin a JSON object");
    expect(!grammar.accepts("```json"), "a code fence cannot begin a JSON object");
    expect(grammar.accepts("{\"answer\":"), "the object itself can");
    expect(grammar.accepts("\n{"), "leading whitespace is fine");
}

void check_response_shape_is_enforced() {
    auto object = build_response(colibri::v2::tools::ValueShape::json_object);
    expect(!object.accepts("["), "an array cannot answer for an object");
    auto array = build_response(colibri::v2::tools::ValueShape::json_array);
    expect(array.accepts("[1,"), "an array answers for an array");
    auto any = build_response(colibri::v2::tools::ValueShape::text);
    expect(any.accepts("\"just a string\""), "shapeless mode takes any JSON value");
    expect(!any.accepts("just a string"), "but never bare prose");
}

void check_response_thinking_is_carved_out() {
    auto grammar = build_response(colibri::v2::tools::ValueShape::json_object);
    expect(grammar.accepts("<think>"), "the model may open a thinking block");
    grammar.observe("<think>let me consider { and ] freely");
    expect(!grammar.armed(), "reasoning is not constrained");
    expect(grammar.accepts("anything at all"), "reasoning text is free");
    grammar.observe(" done</think>\n");
    expect(grammar.armed(), "the constraint takes hold after the block");
    expect(!grammar.accepts("So the answer is"), "prose after thinking is out");
    expect(!grammar.accepts("<think>"), "a second thinking block is not on offer");
    expect(grammar.accepts("{\"a\""), "the value follows the thinking block");
}

void check_response_prompt_opened_thinking_starts_free() {
    auto grammar = build_response(
        colibri::v2::tools::ValueShape::json_object, /*thinking_open=*/true);
    expect(!grammar.armed(), "a prompt-opened block starts inside the carve-out");
    grammar.observe("reasoning with no opening tag</think>");
    expect(grammar.armed(), "the closing tag ends the carve-out");
    expect(!grammar.accepts("Sure"), "and the value is constrained after it");
}

void check_response_disarms_once_the_value_is_whole() {
    auto grammar = build_response(colibri::v2::tools::ValueShape::json_object);
    grammar.observe("{\"answer\": 42");
    expect(!grammar.accepts("done"), "prose cannot interrupt the value");
    grammar.observe("}");
    expect(grammar.empty(), "a whole value hands the tail back");
    expect(grammar.accepts("<|im_end|>"), "so end-of-turn is samplable");
}

void check_response_bypassed_sampler_does_not_deadlock() {
    auto grammar = build_response(colibri::v2::tools::ValueShape::json_object);
    grammar.observe("Sure, here you go: ");  // forced past the constraint
    expect(grammar.empty(), "text outside the language disarms it");
    expect(grammar.accepts("anything"), "and nothing is constrained after");
}

void check_constraint_specifications_parse_both_forms() {
    const auto bare = colibri::v2::tools::parse_constraints(kSpecification);
    expect(bare.tools.size() == 4, "the bare tool array still parses");
    expect(!bare.response_enabled, "and carries no response constraint");
    expect(!bare.tool_calls_forbidden, "and no ban");
    const auto combined = colibri::v2::tools::parse_constraints(
        R"({"tools": [{"name": "bash", "parameters": []}],
            "response_format": {"shape": "object", "thinking_open": true}})");
    expect(combined.tools.size() == 1, "the object form carries tools");
    expect(combined.response_enabled, "and the response constraint");
    expect(combined.response_shape == colibri::v2::tools::ValueShape::json_object,
           "with its shape");
    expect(combined.thinking_open, "and whether the prompt opened thinking");
    const auto forbidden = colibri::v2::tools::parse_constraints(
        R"({"tools": [], "tool_calls": "forbidden"})");
    expect(forbidden.tool_calls_forbidden, "the object form carries the ban");
    expect(forbidden.tools.empty(), "with no tools beside it");
}

void check_markup_ban_refuses_the_atomic_tag() {
    colibri::v2::tools::MarkupBan ban;
    expect(ban.accepts("<tool_call>"), "disabled, everything passes");
    ban.enable();
    expect(!ban.accepts("<tool_call>"), "the atomic opening tag is refused");
    expect(ban.accepts("Here is the summary."), "prose is untouched");
    expect(ban.accepts("</tool_call>"), "and only the opener is watched");
}

void check_markup_ban_refuses_a_split_spelling() {
    colibri::v2::tools::MarkupBan ban;
    ban.enable();
    ban.observe("First, let me confirm: <tool");
    expect(!ban.accepts("_call>"), "a tag split across tokens is refused");
    expect(!ban.accepts("_call>\n<function=bash>"), "however much follows it");
    expect(ban.accepts("s are great"), "but the prefix alone commits nothing");
    ban.observe("s are great");
    expect(ban.accepts("_call>"), "and once passed, the window has moved on");
}

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
    check_a_declared_array_must_be_written_as_json();
    check_a_declared_array_closes_only_when_whole();
    check_a_declared_object_is_held_to_the_same_rule();
    check_untyped_values_stay_free_text();
    check_the_prompted_multi_line_layout_is_accepted();
    check_the_byte_alphabet_round_trips();
    check_candidate_tokens_are_judged_after_decoding();
    check_response_prose_is_unsamplable();
    check_response_shape_is_enforced();
    check_response_thinking_is_carved_out();
    check_response_prompt_opened_thinking_starts_free();
    check_response_disarms_once_the_value_is_whole();
    check_response_bypassed_sampler_does_not_deadlock();
    check_constraint_specifications_parse_both_forms();
    check_markup_ban_refuses_the_atomic_tag();
    check_markup_ban_refuses_a_split_spelling();

    if (failures) {
        std::printf("tool_grammar_contract: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("tool_grammar_contract: ok\n");
    return 0;
}
