#pragma once

// Flyweight Paged KV-Cache Addressing ($P = 64$)
//
// Defines physical page table indirection primitives for growing KV cache,
// decoupling sequence context length from physical memory allocation.
//
// Key Invariants:
// 1. Fixed token page size: P = 64 (1 << 6). Modulo is bitwise AND (& 63),
//    division is bitwise right shift (>> 6).
// 2. Closed PageMajor layout: [physical_page, kv_head, page_offset, row_stride].
//    Within each page, tokens 0..63 for a given head are strictly contiguous.
// 3. Backward Compatibility: When block_table == nullptr, addressing falls back
//    transparently to contiguous layout: [kv_head, capacity, row_stride].
// 4. Memory Footprint: Eliminates per-slot reservation waste (reclaiming up to
//    70-90% of idle VRAM) with zero fragmentation.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stddef.h>
#include <stdint.h>
#include <stdexcept>
#include <vector>

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#define FLYWEIGHT_PAGED_HD __host__ __device__ __forceinline__
#define FLYWEIGHT_PAGED_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define FLYWEIGHT_PAGED_HD inline
#define FLYWEIGHT_PAGED_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define FLYWEIGHT_PAGED_HD inline
#define FLYWEIGHT_PAGED_RESTRICT __restrict__
#else
#define FLYWEIGHT_PAGED_HD inline
#define FLYWEIGHT_PAGED_RESTRICT
#endif

namespace flyweight::paged_kv {

using std::int32_t;
using std::int64_t;
using std::uint32_t;
using std::size_t;

inline constexpr std::int32_t kPagedKVPageSize = 64;
inline constexpr std::int32_t kPagedKVPageShift = 6;
inline constexpr std::int32_t kPagedKVPageMask = kPagedKVPageSize - 1;

static_assert((1 << kPagedKVPageShift) == kPagedKVPageSize, "Page shift mismatch");
static_assert((kPagedKVPageMask & kPagedKVPageSize) == 0, "Page mask mismatch");

// Physical page lookup: translates logical token index to physical page ID
FLYWEIGHT_PAGED_HD std::int32_t paged_kv_physical_page(const std::int32_t* FLYWEIGHT_PAGED_RESTRICT block_table,
                                                       std::int32_t token) {
    return block_table[token >> kPagedKVPageShift];
}

// Logical token offset within its physical page (0..63)
FLYWEIGHT_PAGED_HD std::int32_t paged_kv_page_offset(std::int32_t token) {
    return token & kPagedKVPageMask;
}

// Compute row offset (in elements or bytes, matching row_stride)
// - If block_table == nullptr: contiguous layout: (kv_head * capacity + token) * row_stride
// - If block_table != nullptr: paged PageMajor layout:
//     ((physical_page * kv_heads + kv_head) * 64 + page_offset) * row_stride
FLYWEIGHT_PAGED_HD std::int64_t paged_kv_row_offset(const std::int32_t* FLYWEIGHT_PAGED_RESTRICT block_table,
                                                    std::int32_t token,
                                                    std::int32_t kv_head,
                                                    std::int32_t kv_heads,
                                                    std::int32_t capacity,
                                                    std::int32_t row_stride) {
    if (block_table == nullptr) {
        return (static_cast<std::int64_t>(kv_head) * capacity + token) * row_stride;
    }
    const std::int32_t physical_page = block_table[token >> kPagedKVPageShift];
    const std::int32_t offset = token & kPagedKVPageMask;
    return ((static_cast<std::int64_t>(physical_page) * kv_heads + kv_head) * kPagedKVPageSize + offset) *
           row_stride;
}

// Element offset for uncompressed floats/halves (FP32, FP16, BF16)
FLYWEIGHT_PAGED_HD std::int64_t paged_kv_element_offset(const std::int32_t* FLYWEIGHT_PAGED_RESTRICT block_table,
                                                        std::int32_t token,
                                                        std::int32_t kv_head,
                                                        std::int32_t kv_heads,
                                                        std::int32_t head_dim,
                                                        std::int32_t capacity,
                                                        std::int32_t d) {
    return paged_kv_row_offset(block_table, token, kv_head, kv_heads, capacity, head_dim) + d;
}

// Byte offset for Q8_0 quantized rows (34 bytes per 32 elements)
FLYWEIGHT_PAGED_HD std::int64_t paged_kv_q8_byte_offset(const std::int32_t* FLYWEIGHT_PAGED_RESTRICT block_table,
                                                         std::int32_t token,
                                                         std::int32_t kv_head,
                                                         std::int32_t kv_heads,
                                                         std::int32_t head_dim,
                                                         std::int32_t capacity) {
    const std::int32_t blocks = head_dim / 32;
    const std::int32_t row_bytes = blocks * 34;
    return paged_kv_row_offset(block_table, token, kv_head, kv_heads, capacity, row_bytes);
}

// Byte offset for Turbo4 quantized rows (18 bytes per 32 elements)
FLYWEIGHT_PAGED_HD std::int64_t paged_kv_turbo4_byte_offset(const std::int32_t* FLYWEIGHT_PAGED_RESTRICT block_table,
                                                             std::int32_t token,
                                                             std::int32_t kv_head,
                                                             std::int32_t kv_heads,
                                                             std::int32_t head_dim,
                                                             std::int32_t capacity) {
    const std::int32_t blocks = head_dim / 32;
    const std::int32_t row_bytes = blocks * 18;
    return paged_kv_row_offset(block_table, token, kv_head, kv_heads, capacity, row_bytes);
}

// Byte offset for NVFP4 quantized rows (9 bytes per 16 elements)
FLYWEIGHT_PAGED_HD std::int64_t paged_kv_nvfp4_byte_offset(const std::int32_t* FLYWEIGHT_PAGED_RESTRICT block_table,
                                                           std::int32_t token,
                                                           std::int32_t kv_head,
                                                           std::int32_t kv_heads,
                                                           std::int32_t head_dim,
                                                           std::int32_t capacity) {
    const std::int32_t blocks = head_dim / 16;
    const std::int32_t row_bytes = blocks * 9;
    return paged_kv_row_offset(block_table, token, kv_head, kv_heads, capacity, row_bytes);
}

// Host-side Page Pool Allocator
class PagedKVPool {
public:
    explicit PagedKVPool(std::uint32_t total_pages) : total_pages_(total_pages) {
        free_pages_.resize(total_pages);
        for (std::uint32_t i = 0; i < total_pages; ++i) {
            free_pages_[i] = static_cast<std::int32_t>(total_pages - 1 - i); // LIFO stack
        }
    }

    [[nodiscard]] std::int32_t allocate_page() {
        if (free_pages_.empty()) {
            throw std::runtime_error("PagedKVPool: out of physical KV pages");
        }
        const std::int32_t page = free_pages_.back();
        free_pages_.pop_back();
        return page;
    }

    void free_page(std::int32_t page) {
        assert(page >= 0 && static_cast<std::uint32_t>(page) < total_pages_);
        free_pages_.push_back(page);
    }

    [[nodiscard]] std::uint32_t total_pages() const noexcept { return total_pages_; }
    [[nodiscard]] std::uint32_t available_pages() const noexcept { return static_cast<std::uint32_t>(free_pages_.size()); }
    [[nodiscard]] std::uint32_t allocated_pages() const noexcept { return total_pages_ - available_pages(); }
    [[nodiscard]] std::size_t token_capacity() const noexcept { return static_cast<std::size_t>(total_pages_) * kPagedKVPageSize; }

private:
    std::uint32_t total_pages_ = 0;
    std::vector<std::int32_t> free_pages_;
};

// Sequence Block Table: manages logical-to-physical page mappings for a single sequence
class PagedKVSequenceTable {
public:
    PagedKVSequenceTable() = default;

    // Ensure mapping exists up to token_count
    void ensure_mapped_tokens(std::int32_t token_count, PagedKVPool& pool) {
        if (token_count <= 0) return;
        const std::int32_t needed_pages = (token_count + kPagedKVPageSize - 1) >> kPagedKVPageShift;
        while (static_cast<std::int32_t>(block_table_.size()) < needed_pages) {
            block_table_.push_back(pool.allocate_page());
        }
    }

    // Truncate mappings to token_count, returning excess pages to pool
    void truncate_tokens(std::int32_t token_count, PagedKVPool& pool) {
        const std::int32_t needed_pages = token_count <= 0 ? 0 : (token_count + kPagedKVPageSize - 1) >> kPagedKVPageShift;
        while (static_cast<std::int32_t>(block_table_.size()) > needed_pages) {
            pool.free_page(block_table_.back());
            block_table_.pop_back();
        }
    }

    // Release all pages back to pool
    void release(PagedKVPool& pool) {
        truncate_tokens(0, pool);
    }

    [[nodiscard]] const std::int32_t* data() const noexcept { return block_table_.data(); }
    [[nodiscard]] std::size_t num_pages() const noexcept { return block_table_.size(); }
    [[nodiscard]] std::int32_t page_at(std::size_t logical_page) const { return block_table_.at(logical_page); }

    // Direct access to table vector for tests/custom layouts
    std::vector<std::int32_t>& table() noexcept { return block_table_; }
    [[nodiscard]] const std::vector<std::int32_t>& table() const noexcept { return block_table_; }

private:
    std::vector<std::int32_t> block_table_;
};

} // namespace flyweight::paged_kv
