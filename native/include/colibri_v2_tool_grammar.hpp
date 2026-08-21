#pragma once

// Constrained decoding for tool calls.
//
// The tolerant parser in the server recovers what it can from a malformed
// call -- parameters emitted after the closing tag, a block that closed early,
// a value whose quotes never balanced. It recovers a lot, and it is still the
// wrong shape of fix: it runs *after* the tokens are chosen, so the failure it
// is repairing has already been sampled. Measured against Qwen3.8-27B at Q2_K,
// asked to run one shell command with a two-parameter schema, the model omits
// the parameter it reads as optional -- `description` beside a `command` --
// often enough that an agent loop stalls on it.
//
// This constrains the sampler instead: once the model opens a tool call, only
// tokens that keep the call well-formed may be chosen. A required parameter
// cannot be skipped because the token that would skip it is never a candidate.
//
// Scope, deliberately narrow:
//
// * Only the tag template this family emits is described --
//   `<tool_call><function=NAME><parameter=KEY>value</parameter>...</function>
//   </tool_call>`. A general GBNF engine would cover more and is a much larger
//   surface to get wrong; the failures worth preventing are all structural.
// * The constraint arms on `<tool_call>` and disarms when the call closes.
//   Prose is never constrained, so a model that answers in text is unaffected.
// * Parameter *values* are free text. Constraining them would mean describing
//   every tool's value domain, and nothing observed goes wrong inside a value.
//
// The engine is a small NFA over literals: at any point it holds the set of
// positions it could be at, and a candidate token survives if some position
// consumes it. That is what lets the sampler ask "could this token continue a
// valid call?" without committing to which branch the model is on.

#include <cstddef>
#include <string>
#include <vector>

#include "colibri_v2_json.hpp"

namespace colibri::v2::tools {

// What a parameter's value has to look like. Only the structural pair is
// described: a Hermes parameter carries text, and the server reconstructs a
// declared array or object by parsing that text as JSON, so a value that is not
// JSON arrives at the client as a string and fails its schema -- "expected
// array, received string". Scalars need nothing here, because text is already
// what they are and the server coerces "5" and "true" from it.
enum class ValueShape { text, json_array, json_object };

struct Parameter {
    std::string name;
    bool required = false;
    ValueShape shape = ValueShape::text;
};

struct Tool {
    std::string name;
    std::vector<Parameter> parameters;
};

inline std::vector<Tool> parse_tool_entries(const colibri::v2::json::Value& document) {
    std::vector<Tool> tools;
    for (std::size_t index = 0; index < document.size(); ++index) {
        const auto& entry = document[index];
        Tool tool;
        tool.name = entry["name"].as_string();
        if (tool.name.empty()) continue;
        const auto& parameters = entry["parameters"];
        for (std::size_t p = 0; p < parameters.size(); ++p) {
            Parameter parameter;
            parameter.name = parameters[p]["name"].as_string();
            if (parameter.name.empty()) continue;
            parameter.required = parameters[p]["required"].as_bool();
            const auto declared = parameters[p]["type"].as_string();
            if (declared == "array") parameter.shape = ValueShape::json_array;
            else if (declared == "object") parameter.shape = ValueShape::json_object;
            tool.parameters.push_back(std::move(parameter));
        }
        tools.push_back(std::move(tool));
    }
    return tools;
}

// `[{"name": "write", "parameters": [{"name": "filePath", "required": true,
//   "type": "string"}]}]`. An unknown or absent type is text.
inline std::vector<Tool> parse_specification(const std::string& text) {
    if (text.empty()) return {};
    return parse_tool_entries(colibri::v2::json::parse(text));
}

// Everything one request may constrain the sampler with. The wire form is
// either the historical bare tool array, or an object carrying both kinds:
// `{"tools": [...], "response_format": {"shape": "object", "thinking_open":
// true}}`. The array form stays parseable so an older Python layer keeps
// working against a newer native library, and vice versa.
struct RequestConstraints {
    std::vector<Tool> tools;
    bool response_enabled = false;
    ValueShape response_shape = ValueShape::text;
    bool thinking_open = false;
};

inline RequestConstraints parse_constraints(const std::string& text) {
    RequestConstraints constraints;
    if (text.empty()) return constraints;
    const auto document = colibri::v2::json::parse(text);
    if (document.kind == colibri::v2::json::Kind::Array) {
        constraints.tools = parse_tool_entries(document);
        return constraints;
    }
    constraints.tools = parse_tool_entries(document["tools"]);
    const auto& format = document["response_format"];
    if (!format.is_null()) {
        constraints.response_enabled = true;
        const auto shape = format["shape"].as_string();
        if (shape == "object") constraints.response_shape = ValueShape::json_object;
        else if (shape == "array") constraints.response_shape = ValueShape::json_array;
        constraints.thinking_open = format["thinking_open"].as_bool();
    }
    return constraints;
}

// Whether a byte stream is still a possible prefix of one JSON value, and
// whether it is already a whole one.
//
// This is what turns "the value is free text" into "the value is an array":
// the sampler asks it of every candidate token, so it must judge a *prefix*
// rather than a document -- `[{"a":` is not valid JSON but is a perfectly good
// beginning, while `[}` can never become one.
//
// Deliberately a real acceptor rather than a bracket counter. Counting brackets
// makes the type right and the contents arbitrary, and `[garbage]` still fails
// the client's schema, which would leave the same error with more steps.
class JsonCursor {
public:
    explicit JsonCursor(ValueShape shape) : shape_(shape) {}

    bool complete() const { return started_ && stack_.empty() && !in_token(); }

    // False when this byte cannot appear here in any JSON value.
    bool feed(char byte) {
        const unsigned char value = static_cast<unsigned char>(byte);
        if (in_string_) return feed_string(byte);
        if (in_number_) {
            if (is_number_byte(byte)) return true;
            in_number_ = false;
            if (!close_value()) return false;
            // fall through: this byte belongs to whatever follows the number
        }
        if (!literal_.empty()) {
            if (literal_expected_ >= literal_.size()) {
                literal_.clear();
                if (!close_value()) return false;
            } else {
                if (byte != literal_[literal_expected_]) return false;
                ++literal_expected_;
                if (literal_expected_ == literal_.size()) {
                    literal_.clear();
                    literal_expected_ = 0;
                    return close_value();
                }
                return true;
            }
        }
        if (is_space(byte)) return true;
        switch (state_) {
            case State::value: return open_value(byte);
            case State::key:
                if (byte == '"') { in_string_ = true; after_key_ = true; return true; }
                if (byte == '}' && !stack_.empty() && stack_.back() == '{' && empty_object_)
                    return pop('}');
                return false;
            case State::colon:
                if (byte != ':') return false;
                state_ = State::value;
                return true;
            case State::separator:
                if (byte == ',') {
                    state_ = stack_.empty() ? State::done
                        : (stack_.back() == '{' ? State::key : State::value);
                    return !stack_.empty();
                }
                if (byte == ']') return pop(']');
                if (byte == '}') return pop('}');
                return false;
            case State::done:
                return false;
        }
        return false;
    }

private:
    enum class State { value, key, colon, separator, done };

    static bool is_space(char byte) {
        return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r';
    }
    static bool is_number_byte(char byte) {
        return (byte >= '0' && byte <= '9') || byte == '-' || byte == '+' ||
               byte == '.' || byte == 'e' || byte == 'E';
    }
    bool in_token() const { return in_string_ || in_number_ || !literal_.empty(); }

    bool feed_string(char byte) {
        if (escape_) {
            escape_ = false;
            return true;  // \u is not validated further; it cannot end the value
        }
        if (byte == '\\') { escape_ = true; return true; }
        if (byte == '"') {
            in_string_ = false;
            if (after_key_) { after_key_ = false; state_ = State::colon; return true; }
            return close_value();
        }
        return true;
    }

    // The first value has to be the declared shape; nested ones are free.
    bool open_value(char byte) {
        if (!started_) {
            if (shape_ == ValueShape::json_array && byte != '[') return false;
            if (shape_ == ValueShape::json_object && byte != '{') return false;
        }
        started_ = true;
        switch (byte) {
            case '[': stack_.push_back('['); state_ = State::value; empty_array_ = true; return true;
            case '{': stack_.push_back('{'); state_ = State::key; empty_object_ = true; return true;
            case '"': in_string_ = true; empty_array_ = false; return true;
            case 't': literal_ = "true"; literal_expected_ = 1; empty_array_ = false; return true;
            case 'f': literal_ = "false"; literal_expected_ = 1; empty_array_ = false; return true;
            case 'n': literal_ = "null"; literal_expected_ = 1; empty_array_ = false; return true;
            case ']':
                // An empty array is the one place a close can arrive here.
                if (!stack_.empty() && stack_.back() == '[' && empty_array_) return pop(']');
                return false;
            default:
                if (byte == '-' || (byte >= '0' && byte <= '9')) {
                    in_number_ = true;
                    empty_array_ = false;
                    return true;
                }
                return false;
        }
    }

    // A value finished; what may follow depends on what encloses it.
    bool close_value() {
        empty_array_ = false;
        empty_object_ = false;
        state_ = stack_.empty() ? State::done : State::separator;
        return true;
    }

    bool pop(char closer) {
        if (stack_.empty()) return false;
        const char opener = stack_.back();
        if ((closer == ']' && opener != '[') || (closer == '}' && opener != '{'))
            return false;
        stack_.pop_back();
        return close_value();
    }

    ValueShape shape_ = ValueShape::text;
    std::vector<char> stack_;
    State state_ = State::value;
    std::string literal_;
    std::size_t literal_expected_ = 0;
    bool in_string_ = false, escape_ = false, after_key_ = false;
    bool in_number_ = false, started_ = false;
    bool empty_array_ = false, empty_object_ = false;
};

inline constexpr const char* kCallOpen = "<tool_call>";
inline constexpr const char* kCallClose = "</tool_call>";
inline constexpr const char* kFunctionOpen = "<function=";
inline constexpr const char* kFunctionClose = "</function>";
inline constexpr const char* kParameterOpen = "<parameter=";
inline constexpr const char* kParameterClose = "</parameter>";

// One position the model could be at inside a call.
//
// `literal` is what must come next; `consumed` is how much of it is already
// written. A free-text position instead reads anything until `literal` starts,
// which is how a parameter value is described without describing its contents.
struct Position {
    std::string literal;
    std::size_t consumed = 0;
    bool free_text = false;
    // A value the schema declared as an array or object reads as JSON rather
    // than as free text: the cursor judges each byte, and the closing tag only
    // becomes reachable once the value is whole.
    bool json_value = false;
    JsonCursor json{ValueShape::text};
    // Which tool this branch committed to, and which parameters it has
    // written. Indices into the tool list and its parameter list.
    std::size_t tool = 0;
    // Which parameter's value is being written, so the value position can look
    // up the shape the schema declared for it.
    std::size_t parameter = 0;
    std::vector<bool> written;
    // Where to go once `literal` completes.
    enum class Next {
        function_name,     // after "<function=": the tool names branch here
        parameter_or_end,  // after ">" or "</parameter>"
        parameter_name,    // after "<parameter="
        value,             // after a parameter's ">"
        call_close,        // after "</function>": "</tool_call>" is next
        closed,            // the call is finished
    } next = Next::function_name;
};

class Grammar {
public:
    Grammar() = default;
    explicit Grammar(std::vector<Tool> tools) : tools_(std::move(tools)) {}

    bool empty() const { return tools_.empty(); }
    // True once a call has opened and before it closes: only then does the
    // sampler consult `accepts`.
    bool armed() const { return armed_; }

    // Feed text the model actually produced. Outside a call this only watches
    // for the opening tag; inside one it advances the positions.
    void observe(const std::string& text) {
        for (const char byte : text) observe_byte(byte);
    }

    // Could `text` continue a well-formed call from here? Used to filter
    // sampling candidates, so it must not mutate the state.
    bool accepts(const std::string& text) const {
        if (!armed_) return true;
        auto positions = positions_;
        for (const char byte : text) {
            positions = step(positions, byte);
            if (positions.empty()) return false;
        }
        return true;
    }

    // The positions, for tests: a grammar with none is stuck, which is the one
    // state the sampler must never be left in.
    std::size_t position_count() const { return positions_.size(); }

private:
    void observe_byte(char byte) {
        // With nothing to constrain to, arming would leave the sampler with an
        // empty candidate set -- every token forbidden, which is worse than any
        // malformed call.
        if (tools_.empty()) return;
        if (!armed_) {
            pending_.push_back(byte);
            const std::string open = kCallOpen;
            if (pending_.size() > open.size())
                pending_.erase(0, pending_.size() - open.size());
            if (pending_ == open) {
                positions_ = opening_positions();
                if (positions_.empty()) return;
                armed_ = true;
                pending_.clear();
            }
            return;
        }
        positions_ = step(positions_, byte);
        if (positions_.empty()) {
            // Nothing survived, which means the model wrote something the
            // grammar could not have produced -- only reachable when the
            // sampler was bypassed (a forced token, a resumed conversation).
            // Disarm rather than deadlock: an unconstrained tail is what the
            // tolerant parser already handles.
            armed_ = false;
            pending_.clear();
            return;
        }
        for (const auto& position : positions_)
            if (position.next == Position::Next::closed &&
                position.consumed == position.literal.size()) {
                armed_ = false;
                pending_.clear();
                positions_.clear();
                return;
            }
    }

    // Branch once per tool: the name is what the model picks between.
    std::vector<Position> opening_positions() const {
        std::vector<Position> positions;
        for (std::size_t index = 0; index < tools_.size(); ++index) {
            Position position;
            position.literal = std::string(kFunctionOpen) + tools_[index].name + ">";
            position.tool = index;
            position.written.assign(tools_[index].parameters.size(), false);
            position.next = Position::Next::parameter_or_end;
            positions.push_back(std::move(position));
        }
        return positions;
    }

    // What may follow a completed literal: the parameters not yet written, and
    // the closing tags once nothing required is outstanding.
    std::vector<Position> expand(const Position& position) const {
        std::vector<Position> out;
        const auto& tool = tools_[position.tool];
        switch (position.next) {
            case Position::Next::parameter_or_end: {
                for (std::size_t index = 0; index < tool.parameters.size(); ++index) {
                    if (position.written[index]) continue;
                    Position next = position;
                    next.literal = std::string(kParameterOpen) +
                        tool.parameters[index].name + ">";
                    next.consumed = 0;
                    next.free_text = false;
                    next.json_value = false;
                    next.next = Position::Next::value;
                    next.parameter = index;
                    next.written[index] = true;
                    out.push_back(std::move(next));
                }
                bool outstanding = false;
                for (std::size_t index = 0; index < tool.parameters.size(); ++index)
                    if (tool.parameters[index].required && !position.written[index])
                        outstanding = true;
                if (!outstanding) {
                    // Closing is only a choice once every required parameter
                    // has been written. This is the whole point of the
                    // exercise: the token that would close early is not a
                    // candidate.
                    //
                    // The two closing tags are separate positions rather than
                    // one literal because the format the server prompts for
                    // puts a newline between them, and whitespace is only
                    // skippable ahead of a literal, not inside one.
                    Position next = position;
                    next.literal = kFunctionClose;
                    next.consumed = 0;
                    next.free_text = false;
                    next.next = Position::Next::call_close;
                    out.push_back(std::move(next));
                }
                break;
            }
            case Position::Next::call_close: {
                Position next = position;
                next.literal = kCallClose;
                next.consumed = 0;
                next.free_text = false;
                next.next = Position::Next::closed;
                out.push_back(std::move(next));
                break;
            }
            case Position::Next::value: {
                Position next = position;
                next.literal = kParameterClose;
                next.consumed = 0;
                next.next = Position::Next::parameter_or_end;
                const auto shape = position.parameter < tool.parameters.size()
                    ? tool.parameters[position.parameter].shape
                    : ValueShape::text;
                if (shape == ValueShape::text) {
                    next.free_text = true;
                    next.json_value = false;
                } else {
                    // The value has a declared structure, so it is read rather
                    // than passed through. `</parameter>` stays unreachable
                    // until the JSON is whole, which is what makes a half
                    // array -- or prose where an array belongs -- unsamplable.
                    next.free_text = false;
                    next.json_value = true;
                    next.json = JsonCursor(shape);
                }
                out.push_back(std::move(next));
                break;
            }
            default:
                break;
        }
        return out;
    }

    std::vector<Position> step(const std::vector<Position>& positions,
                               char byte) const {
        std::vector<Position> out;
        for (const auto& position : positions) {
            if (position.consumed == position.literal.size()) {
                // Sitting on a completed literal: expand and retry the byte
                // against whatever may follow.
                for (const auto& expanded : expand(position)) {
                    auto stepped = step({expanded}, byte);
                    out.insert(out.end(), stepped.begin(), stepped.end());
                }
                continue;
            }
            // A structured value: the byte either continues the JSON, or --
            // once the JSON is a whole value -- begins the closing tag. Both
            // are offered, and the set keeps whichever survives.
            if (position.json_value && position.consumed == 0) {
                if (position.json.complete() &&
                    position.literal[0] == byte) {
                    Position closing = position;
                    closing.json_value = false;
                    closing.consumed = 1;
                    out.push_back(std::move(closing));
                }
                Position advanced = position;
                if (advanced.json.feed(byte)) out.push_back(std::move(advanced));
                continue;
            }
            // Whitespace ahead of a structural tag is skipped rather than
            // matched. The tool prompt this server injects lays a call out over
            // several lines -- `<tool_call>\n<function=name>\n<parameter=key>\n`
            // -- so a grammar whose literals run together would die on the
            // first newline and silently stop constraining anything. Only
            // *before* a literal: inside one, and inside a value, every byte
            // still counts.
            if (position.consumed == 0 && !position.free_text &&
                (byte == '\n' || byte == '\r' || byte == ' ' || byte == '\t')) {
                out.push_back(position);
                continue;
            }
            if (position.literal[position.consumed] == byte) {
                Position advanced = position;
                ++advanced.consumed;
                if (advanced.consumed == advanced.literal.size() &&
                    advanced.next != Position::Next::closed) {
                    for (auto&& expanded : expand(advanced))
                        out.push_back(std::move(expanded));
                    // The completed literal is kept too, so a closing tag that
                    // is also a prefix of something else stays reachable.
                    out.push_back(std::move(advanced));
                } else {
                    out.push_back(std::move(advanced));
                }
                continue;
            }
            if (position.free_text) {
                // Inside a value: any byte is fine, but a partial match of the
                // closing tag has to restart rather than be lost.
                Position restarted = position;
                restarted.consumed = 0;
                if (restarted.literal[0] == byte) restarted.consumed = 1;
                out.push_back(std::move(restarted));
            }
        }
        // Deduplicate: expansion can reach the same position by two routes and
        // the set would otherwise grow with the value's length.
        std::vector<Position> unique;
        for (auto& position : out) {
            bool seen = false;
            // A JSON value carries reader state that these fields do not
            // describe, so two such positions are never interchangeable and
            // must not be collapsed into one.
            if (!position.json_value)
                for (const auto& other : unique)
                    if (!other.json_value &&
                        other.consumed == position.consumed &&
                        other.literal == position.literal &&
                        other.tool == position.tool &&
                        other.written == position.written &&
                        other.next == position.next &&
                        other.free_text == position.free_text) {
                        seen = true;
                        break;
                    }
            if (!seen) unique.push_back(std::move(position));
        }
        return unique;
    }

    std::vector<Tool> tools_;
    std::vector<Position> positions_;
    std::string pending_;
    bool armed_ = false;
};

// Constrained decoding for response_format: the visible answer must be one
// JSON value of the declared shape.
//
// The server already renders the requirement into the system prompt; this is
// what makes it a guarantee rather than a request. Prompt-level JSON mode is
// exactly the failure class the tool grammar was built against -- the option
// is accepted, nothing enforces it, and the client discovers that at parse
// time on the far side.
//
// Reasoning is carved out, because a checkpoint asked to think must be allowed
// to: a leading `<think>...</think>` block is free text, and the constraint
// takes hold on the first byte after it. When the prompt itself opened the
// block (`thinking_open`), the turn *starts* inside that carve-out. Anything
// else before the value -- prose, a code fence -- is never a candidate. Once
// the value is whole the constraint disarms, the same hand-back the tool
// grammar does after a call closes, so end-of-turn is the model's own choice
// (in practice EOS, which the constrained tail has made the only likely token).
class ResponseGrammar {
public:
    ResponseGrammar() = default;
    ResponseGrammar(ValueShape shape, bool thinking_open)
        : enabled_(true),
          state_(thinking_open ? State::thinking : State::start),
          cursor_(shape) {}

    bool empty() const { return !enabled_; }
    // Inside the thinking carve-out nothing is constrained, so the sampler can
    // skip candidate filtering for the whole of a long reasoning block.
    bool armed() const { return enabled_ && state_ != State::thinking; }

    // Feed text the model actually produced.
    void observe(const std::string& text) {
        if (!enabled_) return;
        for (const char byte : text)
            if (!feed(byte)) {
                // Only reachable when the sampler was bypassed (a forced
                // token, a resumed conversation). Disarm rather than deadlock,
                // exactly as the tool grammar does.
                enabled_ = false;
                return;
            }
        if (state_ == State::value && cursor_.complete()) enabled_ = false;
    }

    // Could `text` continue a well-formed response from here?
    bool accepts(const std::string& text) const {
        if (!armed()) return true;
        ResponseGrammar copy = *this;
        for (const char byte : text)
            if (!copy.feed(byte)) return false;
        return true;
    }

private:
    // start: leading whitespace, then either `<think>` or the value's first
    // byte. lead: after a thinking block -- whitespace, then the value; a
    // second thinking block is not on offer. value: inside the JSON.
    enum class State { start, think_tag, thinking, lead, value };

    static bool is_space(char byte) {
        return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r';
    }

    bool feed(char byte) {
        switch (state_) {
            case State::start:
                if (is_space(byte)) return true;
                if (byte == '<') {
                    // The value cannot begin with '<', so the only legal
                    // reading is the opening of a thinking block.
                    state_ = State::think_tag;
                    consumed_ = 1;
                    return true;
                }
                state_ = State::value;
                return cursor_.feed(byte);
            case State::think_tag: {
                static const std::string open = "<think>";
                if (consumed_ >= open.size() || byte != open[consumed_]) return false;
                ++consumed_;
                if (consumed_ == open.size()) {
                    state_ = State::thinking;
                    recent_.clear();
                }
                return true;
            }
            case State::thinking: {
                static const std::string close = "</think>";
                recent_.push_back(byte);
                if (recent_.size() > close.size())
                    recent_.erase(0, recent_.size() - close.size());
                if (recent_ == close) state_ = State::lead;
                return true;
            }
            case State::lead:
                if (is_space(byte)) return true;
                state_ = State::value;
                return cursor_.feed(byte);
            case State::value:
                return cursor_.feed(byte);
        }
        return false;
    }

    bool enabled_ = false;
    State state_ = State::start;
    std::size_t consumed_ = 0;
    std::string recent_;
    JsonCursor cursor_{ValueShape::text};
};

}  // namespace colibri::v2::tools
