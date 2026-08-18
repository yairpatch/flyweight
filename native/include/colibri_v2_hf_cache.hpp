#pragma once

// A sidecar for the quantized arena.
//
// Quantizing an HF checkpoint is not cheap: the K-quant packers run at ~30M
// elements/s/core for Q4_K, and Ling-3.0-tiny is ~7.9G elements, so every open
// spends a few hundred core-seconds redoing work whose inputs never change.
// The result is a plain byte arena plus a descriptor table, which is exactly
// the shape of a file -- so write it once and map it thereafter.
//
// Two properties make this safe to keep beside the checkpoint:
//
// 1. The fingerprint covers everything the arena depends on -- the quantization
//    policy, config.json's bytes, and every shard's size and mtime -- plus a
//    format and a packer version. A stale cache is therefore not a risk to be
//    managed, it is a miss. `kPackerVersion` MUST be bumped whenever the
//    packers change what they emit, because nothing else can see that.
//
// 2. Reading validates before it trusts. The file is mapped, and a truncated
//    or corrupt one must be rejected rather than faulted on, so every offset in
//    the table is bounds-checked against the mapping before a descriptor is
//    built from it.
//
// The file is machine-local: it stores native-endian words and is rejected on a
// host that disagrees. It is a cache, not an interchange format.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "colibri_v2_hf_quantize.hpp"

namespace colibri::v2::hf::cache {

inline constexpr char kMagic[8] = {'C', 'O', 'L', 'I', 'B', 'R', 'I', 'Q'};
inline constexpr std::uint32_t kFormatVersion = 1;
// Bump when the packers change their output for the same input. The arena is
// bytes; nothing downstream can tell a new packing from an old one.
//
// 3: narrow-row 2-D weights (the MLA up-projections) became Q8_0 instead of
//    f32, so the GPU decode path can matvec them.
// 2: the packers moved into a translation unit built with `-ffp-contract=off`.
//    Before that GCC fused their multiply-adds, so the bytes depended on the
//    build rather than on the input -- a checkpoint quantized on a machine
//    without FMA did not match one quantized with it.
// 5: the same guarantee reached MSVC. CMake applies -ffp-contract=off only
//    when NOT MSVC, so the Windows build was free to fuse and emitted
//    different bytes than every other platform for the same input; a
//    #pragma fp_contract(off) in qwen_kquant_pack.h now covers it. Any cache
//    written by an older Windows build was packed by the contracted
//    arithmetic and must miss.
// 6: Qwen3.5's ssm_conv1d lost the singleton dimension torch's grouped Conv1d
//    carries -- the arena bytes are the same, but the cache stores descriptors
//    too, and a cached [kernel, 1, channels] would still size the convolution
//    state as `kernel` floats -- and the quantization target now requires a
//    whole block per row rather than per tensor, which changes the type of any
//    embedding table whose rows are not a multiple of the K-quant block.
inline constexpr std::uint32_t kPackerVersion = 6;
inline constexpr std::uint32_t kByteOrderProbe = 0x01020304u;
// The arena starts on a page boundary so a mapped arena keeps the alignment the
// in-memory one had, and so the mapping's first arena page is not shared with
// the table.
inline constexpr std::uint64_t kArenaAlignment = 4096;

// Fixed-size prologue. Everything after it is the table, then padding, then the
// arena.
struct Header {
    char magic[8];
    std::uint32_t format_version;
    std::uint32_t byte_order;
    std::uint64_t fingerprint;
    std::uint64_t tensor_count;
    std::uint64_t table_bytes;   // table length, starting at sizeof(Header)
    std::uint64_t arena_offset;  // from the start of the file
    std::uint64_t arena_bytes;
};
static_assert(sizeof(Header) == 56, "cache header must stay packed as written");

// ---------------------------------------------------------------------------
// fingerprint
// ---------------------------------------------------------------------------

// One source file as the fingerprint sees it. Content is never hashed -- these
// are multi-gigabyte files and reading them would cost more than the packing
// this avoids -- so identity is (name, size, mtime), the same bargain every
// build system makes.
//
// `name` is the file name alone, not its path: a checkpoint that is moved is
// still the same checkpoint, and keying on the path would throw the cache away
// on a rename. What keeps two different models apart is config.json's bytes,
// which are hashed whole.
struct SourceFile {
    std::string name;
    std::uint64_t size = 0;
    std::int64_t modified_ns = 0;
};

inline void hash_bytes(std::uint64_t& state, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        state ^= bytes[i];
        state *= 0x00000100000001B3ull;  // FNV-1a
    }
}

inline void hash_u64(std::uint64_t& state, std::uint64_t value) {
    hash_bytes(state, &value, sizeof(value));
}

// `config_text` is hashed whole rather than through ModelConfig: the parsed
// struct is a lossy view, and a field it ignores today may matter tomorrow.
inline std::uint64_t fingerprint(const std::string& config_text,
                                 const std::vector<SourceFile>& sources,
                                 const Policy& policy) {
    std::uint64_t state = 0xcbf29ce484222325ull;
    hash_u64(state, kFormatVersion);
    hash_u64(state, kPackerVersion);
    hash_u64(state, static_cast<std::uint64_t>(policy.weights));
    hash_u64(state, static_cast<std::uint64_t>(policy.embedding));
    hash_u64(state, static_cast<std::uint64_t>(policy.small));
    // The head target is hashed only when it differs from the embedding's.
    //
    // It is a later field: before it existed the two were one, and a cache
    // written then holds exactly what this policy would produce now whenever
    // they agree. Hashing it unconditionally would invalidate every such cache
    // -- tens of gigabytes on a large checkpoint -- to describe a difference
    // that is not there.
    if (policy.head != policy.embedding)
        hash_u64(state, static_cast<std::uint64_t>(policy.head));
    hash_bytes(state, config_text.data(), config_text.size());
    hash_u64(state, sources.size());
    for (const auto& source : sources) {
        hash_bytes(state, source.name.data(), source.name.size());
        hash_u64(state, source.size);
        hash_u64(state, static_cast<std::uint64_t>(source.modified_ns));
    }
    return state;
}

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------

inline void append(std::vector<std::uint8_t>& out, const void* data,
                   std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    out.insert(out.end(), bytes, bytes + size);
}

template <typename T>
inline void append_word(std::vector<std::uint8_t>& out, T value) {
    append(out, &value, sizeof(value));
}

// The descriptor table, in the order the tensors are listed. Names are
// length-prefixed rather than terminated so the reader never scans.
inline std::vector<std::uint8_t> encode_table(
    const std::vector<QuantizedTensor>& tensors) {
    std::vector<std::uint8_t> table;
    for (const auto& tensor : tensors) {
        append_word<std::uint32_t>(table,
                                   static_cast<std::uint32_t>(tensor.name.size()));
        append(table, tensor.name.data(), tensor.name.size());
        append_word<std::uint32_t>(table,
                                   static_cast<std::uint32_t>(tensor.shape.size()));
        for (const auto extent : tensor.shape) append_word<std::uint64_t>(table, extent);
        append_word<std::uint32_t>(table, tensor.type);
        append_word<std::uint64_t>(table, tensor.offset);
        append_word<std::uint64_t>(table, tensor.size);
    }
    return table;
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------

// A bounds-checked cursor over the mapped table. Every read goes through it, so
// a truncated file returns false instead of walking off the mapping.
struct Reader {
    const std::uint8_t* data = nullptr;
    std::uint64_t size = 0;
    std::uint64_t at = 0;

    bool take(void* out, std::uint64_t bytes) {
        if (bytes > size - at) return false;
        std::memcpy(out, data + at, static_cast<std::size_t>(bytes));
        at += bytes;
        return true;
    }
    template <typename T>
    bool word(T& out) {
        return take(&out, sizeof(out));
    }
    bool string(std::string& out, std::uint64_t bytes) {
        if (bytes > size - at) return false;
        out.assign(reinterpret_cast<const char*>(data + at),
                   static_cast<std::size_t>(bytes));
        at += bytes;
        return true;
    }
};

struct Contents {
    std::vector<QuantizedTensor> tensors;  // offsets are into the arena
    std::uint64_t arena_offset = 0;        // of the arena within the mapping
    std::uint64_t arena_bytes = 0;
};

// Validates a mapped cache file against `expected` and decodes its table.
// Returns false for every kind of miss -- wrong magic, wrong version, wrong
// fingerprint, truncation, a descriptor that does not fit the arena -- because
// the caller's response to all of them is the same: quantize.
inline bool decode(const std::uint8_t* data, std::uint64_t size,
                   std::uint64_t expected, Contents& out) {
    if (!data || size < sizeof(Header)) return false;
    Header header{};
    std::memcpy(&header, data, sizeof(header));
    if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0) return false;
    if (header.format_version != kFormatVersion) return false;
    if (header.byte_order != kByteOrderProbe) return false;
    if (header.fingerprint != expected) return false;

    if (header.arena_offset > size) return false;
    if (header.arena_bytes > size - header.arena_offset) return false;
    if (header.table_bytes > size - sizeof(Header)) return false;
    if (sizeof(Header) + header.table_bytes > header.arena_offset) return false;
    // A tensor count large enough to overflow the reserve below is a corrupt
    // file, not a big model: each entry costs at least 25 bytes on the wire.
    if (header.tensor_count > header.table_bytes / 24) return false;

    Reader reader{data + sizeof(Header), header.table_bytes, 0};
    std::vector<QuantizedTensor> tensors;
    tensors.reserve(static_cast<std::size_t>(header.tensor_count));
    for (std::uint64_t i = 0; i < header.tensor_count; ++i) {
        QuantizedTensor tensor;
        std::uint32_t name_length = 0;
        if (!reader.word(name_length)) return false;
        if (!reader.string(tensor.name, name_length)) return false;
        std::uint32_t rank = 0;
        if (!reader.word(rank)) return false;
        // GGUF tops out at 4 dimensions and the stacked experts use 3.
        if (rank > 8) return false;
        tensor.shape.resize(rank);
        for (std::uint32_t d = 0; d < rank; ++d)
            if (!reader.word(tensor.shape[d])) return false;
        if (!reader.word(tensor.type)) return false;
        if (!reader.word(tensor.offset)) return false;
        if (!reader.word(tensor.size)) return false;
        // The descriptor must address bytes this file actually has.
        if (tensor.offset > header.arena_bytes) return false;
        if (tensor.size > header.arena_bytes - tensor.offset) return false;
        tensors.push_back(std::move(tensor));
    }

    out.tensors = std::move(tensors);
    out.arena_offset = header.arena_offset;
    out.arena_bytes = header.arena_bytes;
    return true;
}

// ---------------------------------------------------------------------------
// write
// ---------------------------------------------------------------------------

// Writes through a temporary and renames, so a cache file is either complete or
// absent. A half-written one that happened to pass the fingerprint would be
// undetectable -- the fingerprint describes the *inputs*, not the file.
//
// Returns false on any I/O failure. A cache that cannot be written is a
// missed optimization, never an error: the caller already holds the arena.
inline bool write(const std::string& path, const QuantizedModel& model,
                  std::uint64_t print) {
    const auto table = encode_table(model.tensors);
    const std::uint64_t table_end = sizeof(Header) + table.size();
    const std::uint64_t arena_offset =
        (table_end + kArenaAlignment - 1) / kArenaAlignment * kArenaAlignment;

    Header header{};
    std::memcpy(header.magic, kMagic, sizeof(kMagic));
    header.format_version = kFormatVersion;
    header.byte_order = kByteOrderProbe;
    header.fingerprint = print;
    header.tensor_count = model.tensors.size();
    header.table_bytes = table.size();
    header.arena_offset = arena_offset;
    header.arena_bytes = model.arena.size();

    const std::string temporary = path + ".tmp";
    std::FILE* file = std::fopen(temporary.c_str(), "wb");
    if (!file) return false;

    bool ok = std::fwrite(&header, sizeof(header), 1, file) == 1;
    if (ok && !table.empty())
        ok = std::fwrite(table.data(), 1, table.size(), file) == table.size();
    if (ok) {
        static const std::uint8_t padding[kArenaAlignment] = {};
        const std::uint64_t pad = arena_offset - table_end;
        if (pad) ok = std::fwrite(padding, 1, pad, file) == pad;
    }
    if (ok && !model.arena.empty()) {
        // One fwrite of 4 GB is within the API but not within every libc's
        // comfort; chunk it so a short write is visible rather than wrapped.
        const std::uint64_t chunk = 64ull << 20;
        std::uint64_t written = 0;
        while (ok && written < model.arena.size()) {
            const std::uint64_t take =
                std::min<std::uint64_t>(chunk, model.arena.size() - written);
            ok = std::fwrite(model.arena.data() + written, 1,
                             static_cast<std::size_t>(take), file) == take;
            written += take;
        }
    }
    if (std::fclose(file) != 0) ok = false;
    if (!ok) {
        std::remove(temporary.c_str());
        return false;
    }
#if defined(_WIN32)
    // POSIX rename replaces; the Windows CRT's does not.
    std::remove(path.c_str());
#endif
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        std::remove(temporary.c_str());
        return false;
    }
    return true;
}

}  // namespace colibri::v2::hf::cache
