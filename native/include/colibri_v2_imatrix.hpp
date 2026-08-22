#pragma once

// Importance-matrix loader: llama.cpp's legacy `imatrix.dat`.
//
// An importance matrix is per-input-channel activation energy -- sums of x^2
// over calibration tokens -- gathered per weight tensor. The IQ packer uses it
// to weight its codebook search: a channel the model actually drives hard is
// expensive to misquantize, a dormant one is nearly free. Without one the
// packer falls back to sqrt(sigma2 + x^2), which is exactly llama.cpp's own
// no-matrix fallback, and why sub-3-bit formats are not offered without one.
//
// The legacy format is what the ecosystem publishes beside checkpoints:
//
//   int32 n_entries
//   n_entries x { int32 name_len; char name[]; int32 ncall; int32 nval;
//                 float values[nval] }
//   int32 last_call                      (optional trailer)
//   int32 dataset_len; char dataset[]    (optional, newer writers)
//
// There is no magic number, so a GGUF-format imatrix (llama.cpp's newer
// output) is recognized by its "GGUF" magic and refused with a pointer at the
// legacy format rather than misparsed.
//
// Entry names are GGUF tensor names ("blk.0.ffn_down.weight"), which is also
// what the HF loader renames tensors to before quantizing -- so a matrix
// gathered by llama.cpp over the GGUF conversion of a checkpoint applies
// directly to packing that checkpoint's safetensors.
//
// The stored values may be raw sums or sums scaled by call count depending on
// the writer's vintage. It does not matter here: the packer's error is
// sum(w_i * d_i^2) minimized per group, and scaling every weight in a tensor
// by one constant moves the error, not the argmin.

#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace colibri::v2::hf {

struct ImportanceMatrix {
    // Tensor name -> per-channel importance. A stacked expert tensor may carry
    // row_width * expert_count values (one row of channels per expert).
    std::map<std::string, std::vector<float>> entries;
};

// How the packer indexes importance for one tensor. `values` has length `row`
// (one weight per input column, every row alike) or `chunk`-strided per-expert
// rows when llama.cpp gathered the experts separately.
//
//   per-column:  qw(e) = values[e % row]
//   per-expert:  qw(e) = values[(e / chunk) * row + e % row]
struct ImportanceView {
    const float* values = nullptr;
    std::uint64_t row = 0;
    std::uint64_t chunk = 0;  // 0 means per-column
};

inline ImportanceMatrix load_imatrix(const std::string& path) {
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (!file) throw std::runtime_error("cannot open imatrix file " + path);
    struct Closer {
        std::FILE* file;
        ~Closer() { std::fclose(file); }
    } closer{file};

    const auto read_u32 = [&](std::uint32_t& out) {
        return std::fread(&out, sizeof(out), 1, file) == 1;
    };

    std::uint32_t count = 0;
    if (!read_u32(count))
        throw std::runtime_error("imatrix file is empty: " + path);
    if (count == 0x46554747u)  // "GGUF"
        throw std::runtime_error(
            path + " is a GGUF-format imatrix; this loader reads the legacy "
            ".dat format (llama-imatrix --output-format dat)");
    // A real file holds one entry per quantizable weight tensor; six digits is
    // already absurd and anything larger is a misparse, not a matrix.
    if (count == 0 || count > 1000000)
        throw std::runtime_error(path + " does not look like an imatrix file");

    ImportanceMatrix matrix;
    for (std::uint32_t entry = 0; entry < count; ++entry) {
        std::uint32_t name_length = 0;
        if (!read_u32(name_length) || name_length == 0 || name_length > 4096)
            throw std::runtime_error("truncated imatrix entry in " + path);
        std::string name(name_length, '\0');
        if (std::fread(name.data(), 1, name_length, file) != name_length)
            throw std::runtime_error("truncated imatrix entry in " + path);
        std::uint32_t calls = 0, values = 0;
        if (!read_u32(calls) || !read_u32(values) || values == 0)
            throw std::runtime_error("truncated imatrix entry in " + path);
        std::vector<float> data(values);
        if (std::fread(data.data(), sizeof(float), values, file) != values)
            throw std::runtime_error("truncated imatrix entry in " + path);
        // A negative or non-finite importance is a writer bug; clamping keeps
        // one bad channel from poisoning the whole tensor's search.
        for (auto& value : data)
            if (!(value >= 0.0f) || value != value) value = 0.0f;
        matrix.entries[std::move(name)] = std::move(data);
    }
    // The trailer (last_call, dataset name) is informational and optional;
    // nothing here reads it.
    return matrix;
}

// The view for one tensor, or an empty view when the matrix has no matching
// entry. `shape` is GGUF order: shape[0] is the contiguous row width, and a
// stacked expert tensor is [row, rows_per_expert, experts].
inline ImportanceView importance_for(const ImportanceMatrix* matrix,
                                     const std::string& name,
                                     const std::vector<std::uint64_t>& shape) {
    ImportanceView view;
    if (!matrix || shape.empty()) return view;
    const auto found = matrix->entries.find(name);
    if (found == matrix->entries.end()) return view;
    const std::uint64_t row = shape[0];
    const auto& values = found->second;
    if (values.size() == row) {
        view.values = values.data();
        view.row = row;
        return view;
    }
    if (shape.size() == 3 && values.size() == row * shape[2]) {
        view.values = values.data();
        view.row = row;
        view.chunk = row * shape[1];
        return view;
    }
    // A size that matches neither layout is a matrix for a different geometry
    // (wrong checkpoint, or a pruned conversion); weighting with it would be
    // worse than not weighting, so the mismatch reads as absent.
    return view;
}

}  // namespace colibri::v2::hf
