#pragma once

// safetensors header parsing.
//
// The container is trivial by design: 8 bytes of little-endian header length N,
// N bytes of JSON, then a flat data buffer. Every entry names a dtype, a shape,
// and a [start, end) byte range relative to the start of that buffer. That is
// the same information a GGUF tensor descriptor carries, which is why this can
// feed flyweight::v2::WeightProvider without touching anything downstream.
//
// This only reads the header. Mapping, shard merging and name translation are
// the loader's job.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "flyweight_v2_json.hpp"

namespace flyweight::v2::safetensors {

struct Entry {
    std::string name;
    // GGUF order: [inputs, outputs]. See `parse_header` for why this is the
    // reverse of what the file says and why that costs nothing.
    std::vector<std::uint64_t> shape;
    std::uint32_t type = 0;
    std::uint64_t offset = 0;  // absolute within the file
    std::uint64_t size = 0;
    // Rank above what the v2 tensor ABI can describe. Recorded rather than
    // rejected here: a checkpoint may carry tensors its consumer intends to
    // drop -- Qwen3.5's vision patch embedding is a rank-5 conv -- and the
    // reader cannot know which. The consumer raises this if it wants the
    // tensor; see build_tensors.
    bool rank_exceeded = false;
};

struct Header {
    std::vector<Entry> entries;
    std::uint64_t data_offset = 0;  // where the tensor buffer starts
};

// safetensors dtype -> the runtime's tensor type code. Only the three float
// widths the kernels can actually consume are accepted; anything else is
// rejected by name so the error says what the checkpoint holds rather than
// failing later as a size mismatch.
inline std::uint32_t type_from_dtype(const std::string& dtype) {
    if (dtype == "BF16") return 30;
    if (dtype == "F16") return 1;
    if (dtype == "F32") return 0;
    throw std::runtime_error(
        "unsupported safetensors dtype \"" + dtype +
        "\" (the runtime reads BF16, F16 and F32; quantized HF checkpoints are "
        "not supported on this path)");
}

inline std::uint64_t dtype_bytes(std::uint32_t type) {
    return type == 0 ? 4 : 2;
}

// `size` may be the whole mapping; only the header is read.
inline Header parse_header(const std::uint8_t* data, std::uint64_t size) {
    if (size < 8) throw std::runtime_error("not a safetensors file: too short");
    std::uint64_t header_length = 0;
    std::memcpy(&header_length, data, sizeof(header_length));
    // Guard before the add so a hostile length cannot wrap past `size`.
    if (header_length > size - 8)
        throw std::runtime_error("safetensors header runs past the end of the file");
    const std::uint64_t data_offset = 8 + header_length;

    const auto document = json::parse(
        reinterpret_cast<const char*>(data) + 8,
        static_cast<std::size_t>(header_length));
    if (document.kind != json::Kind::Object)
        throw std::runtime_error("safetensors header is not a JSON object");

    Header header;
    header.data_offset = data_offset;
    const std::uint64_t payload = size - data_offset;

    for (const auto& [name, value] : document.object) {
        // Free-form string map the format reserves for producers.
        if (name == "__metadata__") continue;
        if (value.kind != json::Kind::Object)
            throw std::runtime_error("safetensors entry is not an object: " + name);

        Entry entry;
        entry.name = name;
        entry.type = type_from_dtype(value["dtype"].as_string());

        const auto& shape = value["shape"];
        if (shape.kind != json::Kind::Array)
            throw std::runtime_error("safetensors entry has no shape: " + name);
        entry.rank_exceeded = shape.size() > 4;

        // A row-major [out_features, in_features] HF matrix and a GGUF
        // [inputs, outputs] descriptor describe the *same bytes*: GGUF counts
        // dimensions fastest-varying first. So reversing the shape vector
        // converts between them exactly, with no transpose and no copy.
        std::uint64_t elements = 1;
        for (std::size_t i = 0; i < shape.size(); ++i) {
            const auto extent = shape[i].as_uint();
            if (shape[i].kind != json::Kind::Number)
                throw std::runtime_error("safetensors shape is not numeric: " + name);
            // Overflow-safe: reject before multiplying.
            if (extent && elements > (~0ull) / extent)
                throw std::runtime_error("safetensors shape overflows: " + name);
            elements *= extent;
            entry.shape.push_back(extent);
        }
        std::reverse(entry.shape.begin(), entry.shape.end());

        const auto& offsets = value["data_offsets"];
        if (offsets.kind != json::Kind::Array || offsets.size() != 2)
            throw std::runtime_error("safetensors entry has no data_offsets: " + name);
        const std::uint64_t start = offsets[0].as_uint();
        const std::uint64_t stop = offsets[1].as_uint();
        if (stop < start)
            throw std::runtime_error("safetensors data_offsets are inverted: " + name);
        if (start > payload || stop > payload)
            throw std::runtime_error(
                "safetensors tensor lies outside the file: " + name);

        entry.size = stop - start;
        if (entry.size != elements * dtype_bytes(entry.type))
            throw std::runtime_error(
                "safetensors tensor byte range does not match its shape: " + name);
        entry.offset = data_offset + start;
        header.entries.push_back(std::move(entry));
    }
    return header;
}

}  // namespace flyweight::v2::safetensors
