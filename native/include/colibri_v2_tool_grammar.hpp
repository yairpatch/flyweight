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

struct Parameter {
    std::string name;
    bool required = false;
};

struct Tool {
    std::string name;
    std::vector<Parameter> parameters;
};

// `[{"name": "write", "parameters": [{"name": "filePath", "required": true}]}]`
inline std::vector<Tool> parse_specification(const std::string& text) {
    std::vector<Tool> tools;
    if (text.empty()) return tools;
    const auto document = colibri::v2::json::parse(text);
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
            tool.parameters.push_back(std::move(parameter));
        }
        tools.push_back(std::move(tool));
    }
    return tools;
}

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
    // Which tool this branch committed to, and which parameters it has
    // written. Indices into the tool list and its parameter list.
    std::size_t tool = 0;
    std::vector<bool> written;
    // Where to go once `literal` completes.
    enum class Next {
        function_name,     // after "<function=": the tool names branch here
        parameter_or_end,  // after ">" or "</parameter>"
        parameter_name,    // after "<parameter="
        value,             // after a parameter's ">"
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
                    next.next = Position::Next::value;
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
                    Position next = position;
                    next.literal = std::string(kFunctionClose) + kCallClose;
                    next.consumed = 0;
                    next.free_text = false;
                    next.next = Position::Next::closed;
                    out.push_back(std::move(next));
                }
                break;
            }
            case Position::Next::value: {
                Position next = position;
                next.literal = kParameterClose;
                next.consumed = 0;
                next.free_text = true;
                next.next = Position::Next::parameter_or_end;
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
            for (const auto& other : unique)
                if (other.consumed == position.consumed &&
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

}  // namespace colibri::v2::tools
