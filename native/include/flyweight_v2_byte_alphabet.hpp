#pragma once

// The GGUF byte-level alphabet, in both directions.
//
// Byte-level BPE vocabularies do not store raw bytes: every byte is mapped to a
// printable codepoint first, so a space is U+0120 and a newline U+010A. The
// table is GPT-2's and every checkpoint here inherits it.
//
// Both directions live together because they are the same table read two ways,
// and they had drifted apart once already -- tokenization encoded, while the
// only decoder was in Python. The tool grammar needs the decode natively: it
// matches literal bytes (`<parameter=`, `</function>`) against candidate
// tokens, and a candidate carrying a leading space is spelled `Ġ<parameter=` in
// the vocabulary, which matches nothing.

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>

namespace flyweight::v2::alphabet {

// The bytes that stand for themselves; everything else is pushed above 256 in
// the order it appears, which is what GPT-2's `bytes_to_unicode` does.
inline const std::array<int, 256>& encoding() {
    static const std::array<int, 256> table = [] {
        static const int direct[] = {
            33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,
            55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,
            77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,
            99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,
            116,117,118,119,120,121,122,123,124,125,126,161,162,163,164,165,
            166,167,168,169,170,171,172,174,175,176,177,178,179,180,181,182,
            183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,
            199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,
            215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,
            231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,
            247,248,249,250,251,252,253,254,255};
        std::array<int, 256> map{};
        for (int index = 0; index < 256; ++index) map[index] = -1;
        for (const int byte : direct) map[byte] = byte;
        int extra = 0;
        for (int index = 0; index < 256; ++index)
            if (map[index] < 0) map[index] = 256 + extra++;
        return map;
    }();
    return table;
}

inline std::string encode(const char* text) {
    const auto& map = encoding();
    std::string out;
    for (const unsigned char* cursor = reinterpret_cast<const unsigned char*>(text);
         *cursor; ++cursor) {
        const int codepoint = map[*cursor];
        if (codepoint < 128) {
            out.push_back(static_cast<char>(codepoint));
        } else if (codepoint < 2048) {
            out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 63)));
        } else {
            out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 63)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 63)));
        }
    }
    return out;
}

// One vocabulary piece back to the bytes it stands for.
//
// Special tokens (`<|im_start|>` and friends) are stored literally rather than
// byte-encoded and pass through unchanged, which is what the Python detokenizer
// does with them too. A codepoint outside the table is copied as written: a
// vocabulary that is not byte-encoded at all then decodes to itself.
inline std::string decode(const std::string& piece) {
    static const std::unordered_map<int, unsigned char> inverse = [] {
        std::unordered_map<int, unsigned char> out;
        const auto& map = encoding();
        for (int byte = 0; byte < 256; ++byte)
            out.emplace(map[byte], static_cast<unsigned char>(byte));
        return out;
    }();
    if (piece.size() >= 4 && piece.compare(0, 2, "<|") == 0 &&
        piece.compare(piece.size() - 2, 2, "|>") == 0)
        return piece;
    std::string out;
    for (std::size_t index = 0; index < piece.size();) {
        const unsigned char lead = static_cast<unsigned char>(piece[index]);
        std::size_t width = (lead < 0x80) ? 1 : (lead < 0xE0 ? 2 : (lead < 0xF0 ? 3 : 4));
        if (index + width > piece.size()) width = 1;
        int codepoint = 0;
        if (width == 1) {
            codepoint = lead;
        } else {
            codepoint = lead & (0xFF >> (width + 1));
            for (std::size_t offset = 1; offset < width; ++offset)
                codepoint = (codepoint << 6) |
                    (static_cast<unsigned char>(piece[index + offset]) & 0x3F);
        }
        const auto found = inverse.find(codepoint);
        if (found != inverse.end()) out.push_back(static_cast<char>(found->second));
        else out.append(piece, index, width);
        index += width;
    }
    return out;
}

}  // namespace flyweight::v2::alphabet
