#pragma once

// Minimal JSON reader for the HF checkpoint path.
//
// The native side has no JSON dependency and does not want one: the three
// documents this has to read -- a safetensors header, `config.json`, and
// `model.safetensors.index.json` -- are machine-written, flat, and small. What
// they are not is trusted, so this parses defensively (explicit depth limit,
// every bound checked) and reports the byte offset on failure.
//
// Deliberately not a general JSON library: no comments, no NaN/Infinity, and
// numbers are kept as both a double and the original text so a 64-bit tensor
// offset survives a round trip that a double would round.

#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace colibri::v2::json {

class Value;
using Object = std::map<std::string, Value>;
using Array = std::vector<Value>;

enum class Kind { Null, Bool, Number, String, Array, Object };

class Value {
public:
    Kind kind = Kind::Null;
    bool boolean = false;
    double number = 0.0;
    // Original number text. `offset` fields in a safetensors header routinely
    // exceed 2^53 on large checkpoints, where `number` has already lost bits.
    std::string literal;
    std::string string;
    Array array;
    Object object;

    bool is_null() const { return kind == Kind::Null; }
    bool contains(const std::string& key) const {
        return kind == Kind::Object && object.find(key) != object.end();
    }

    // Missing key yields a null Value rather than throwing: HF configs omit
    // optional fields constantly, and callers want a default, not a branch.
    const Value& operator[](const std::string& key) const {
        static const Value absent;
        if (kind != Kind::Object) return absent;
        const auto found = object.find(key);
        return found == object.end() ? absent : found->second;
    }
    const Value& operator[](std::size_t index) const {
        static const Value absent;
        return kind == Kind::Array && index < array.size() ? array[index] : absent;
    }
    std::size_t size() const {
        return kind == Kind::Array ? array.size()
             : kind == Kind::Object ? object.size() : 0;
    }

    std::string as_string(const std::string& fallback = {}) const {
        return kind == Kind::String ? string : fallback;
    }
    bool as_bool(bool fallback = false) const {
        return kind == Kind::Bool ? boolean : fallback;
    }
    double as_double(double fallback = 0.0) const {
        return kind == Kind::Number ? number : fallback;
    }
    // Parsed from the literal so the full 64-bit range survives.
    std::uint64_t as_uint(std::uint64_t fallback = 0) const {
        if (kind != Kind::Number) return fallback;
        if (!literal.empty() && literal[0] == '-') return fallback;
        return std::strtoull(literal.c_str(), nullptr, 10);
    }
    std::int64_t as_int(std::int64_t fallback = 0) const {
        return kind == Kind::Number ? std::strtoll(literal.c_str(), nullptr, 10)
                                    : fallback;
    }
};

namespace detail {

// 64 is far past anything these three documents nest to; it exists so a
// malformed file cannot recurse the parser off the stack.
constexpr int kMaxDepth = 64;

struct Parser {
    const char* p;
    const char* end;
    const char* begin;

    [[noreturn]] void fail(const std::string& what) const {
        throw std::runtime_error("invalid JSON at byte " +
                                 std::to_string(p - begin) + ": " + what);
    }
    void skip_space() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }
    char peek() const {
        if (p >= end) throw std::runtime_error("truncated JSON");
        return *p;
    }
    void expect(char c) {
        if (p >= end || *p != c) fail(std::string("expected '") + c + "'");
        ++p;
    }
    void literal(const char* text, std::size_t length) {
        if (static_cast<std::size_t>(end - p) < length ||
            std::string(p, length) != text)
            fail("unknown literal");
        p += length;
    }

    void encode_utf8(std::uint32_t codepoint, std::string& out) const {
        if (codepoint < 0x80) {
            out.push_back(static_cast<char>(codepoint));
        } else if (codepoint < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else if (codepoint < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
    }

    std::uint32_t hex4() {
        if (end - p < 4) fail("truncated \\u escape");
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = *p++;
            value <<= 4;
            if (c >= '0' && c <= '9') value |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') value |= static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= static_cast<std::uint32_t>(c - 'A' + 10);
            else fail("bad hex digit in \\u escape");
        }
        return value;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (p >= end) fail("unterminated string");
            const char c = *p++;
            if (c == '"') break;
            if (c != '\\') { out.push_back(c); continue; }
            if (p >= end) fail("unterminated escape");
            switch (const char esc = *p++) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    std::uint32_t codepoint = hex4();
                    // Surrogate pair. A lone or malformed half becomes U+FFFD
                    // rather than an error: tokenizer.json legitimately carries
                    // unpaired surrogates for byte-level BPE symbols.
                    if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
                        if (end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                            const char* rewind = p;
                            p += 2;
                            const std::uint32_t low = hex4();
                            if (low >= 0xDC00 && low <= 0xDFFF)
                                codepoint = 0x10000 +
                                            ((codepoint - 0xD800) << 10) +
                                            (low - 0xDC00);
                            else { p = rewind; codepoint = 0xFFFD; }
                        } else {
                            codepoint = 0xFFFD;
                        }
                    } else if (codepoint >= 0xDC00 && codepoint <= 0xDFFF) {
                        codepoint = 0xFFFD;
                    }
                    encode_utf8(codepoint, out);
                    break;
                }
                default:
                    (void)esc;
                    fail("unknown string escape");
            }
        }
        return out;
    }

    Value parse(int depth) {
        if (depth > kMaxDepth) fail("JSON nests too deeply");
        skip_space();
        Value value;
        switch (peek()) {
            case 'n': literal("null", 4); value.kind = Kind::Null; return value;
            case 't': literal("true", 4); value.kind = Kind::Bool; value.boolean = true; return value;
            case 'f': literal("false", 5); value.kind = Kind::Bool; value.boolean = false; return value;
            case '"': value.kind = Kind::String; value.string = parse_string(); return value;
            case '[': {
                ++p;
                value.kind = Kind::Array;
                skip_space();
                if (peek() == ']') { ++p; return value; }
                while (true) {
                    value.array.push_back(parse(depth + 1));
                    skip_space();
                    const char c = peek();
                    if (c == ',') { ++p; continue; }
                    if (c == ']') { ++p; return value; }
                    fail("expected ',' or ']'");
                }
            }
            case '{': {
                ++p;
                value.kind = Kind::Object;
                skip_space();
                if (peek() == '}') { ++p; return value; }
                while (true) {
                    skip_space();
                    std::string key = parse_string();
                    skip_space();
                    expect(':');
                    value.object.emplace(std::move(key), parse(depth + 1));
                    skip_space();
                    const char c = peek();
                    if (c == ',') { ++p; continue; }
                    if (c == '}') { ++p; return value; }
                    fail("expected ',' or '}'");
                }
            }
            default: {
                const char* start = p;
                if (p < end && (*p == '-' || *p == '+')) ++p;
                bool digits = false;
                while (p < end && ((*p >= '0' && *p <= '9') || *p == '.' ||
                                   *p == 'e' || *p == 'E' || *p == '-' || *p == '+')) {
                    if (*p >= '0' && *p <= '9') digits = true;
                    ++p;
                }
                if (!digits) fail("expected a value");
                value.kind = Kind::Number;
                value.literal.assign(start, static_cast<std::size_t>(p - start));
                value.number = std::strtod(value.literal.c_str(), nullptr);
                return value;
            }
        }
    }
};

}  // namespace detail

// Parses one JSON document. Trailing whitespace is fine; trailing content is
// not -- a safetensors header declares its own length, and a mismatch there
// means the file is not what it claims.
inline Value parse(const char* data, std::size_t size) {
    detail::Parser parser{data, data + size, data};
    Value value = parser.parse(0);
    parser.skip_space();
    if (parser.p != parser.end)
        parser.fail("trailing content after the JSON document");
    return value;
}

inline Value parse(const std::string& text) {
    return parse(text.data(), text.size());
}

}  // namespace colibri::v2::json
