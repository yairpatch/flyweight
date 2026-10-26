#pragma once

// Direct, overlapped row reads from qwen4exp's n-gram table.
//
// The table is 320M rows of ~90 bytes (26.8 GiB) and a decode token needs 16
// of them, each on its own page, all decided by the token itself. Read through
// the mapping, that was 16 serial page faults in front of layer 0 -- 2-4.5 ms
// a token whenever the rows were not already cached -- and every fault parked
// a 4 KiB page in the page cache to use 90 bytes of it, at the expense of the
// expert pages decode reads densely.
//
// This reader hands a token's pages to a small pool of reader threads, so the
// reads are in flight together, and drops each page from the page cache once
// it has been copied out. `issue` returns at once; `collect` blocks until the rows are in.
// The decode loop issues as soon as the token id is known and collects just
// before the first PLE layer, which hides the reads behind the embedding and
// layer 0. Strata's src/ngram/ple_reader.cpp is the design this follows.
//
// Not O_DIRECT, and not Linux native AIO, which needs it: on btrfs io_submit
// of an O_DIRECT read does the I/O inside the call (2.9 ms a token for 16
// reads), and even from threads O_DIRECT reads waited 1.4 ms a token against
// 0.3-0.5 ms buffered (checksummed direct I/O is the slow path there).
//
// Rows already fetched are kept in a bounded direct-mapped row cache, since
// common bigrams recur. On Windows `open` fails and callers keep the mapping
// path.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace flyweight::v2 {

class PleReader {
public:
    static constexpr std::uint64_t kPage = 4096;
    static constexpr int kMaxRows = 64;
    static constexpr int kThreads = 16;

    PleReader() = default;
    PleReader(const PleReader&) = delete;
    PleReader& operator=(const PleReader&) = delete;
    ~PleReader() { close(); }

    // `table_offset` is the file offset of row 0 and `rows` the row count;
    // `cache_rows` sizes the row cache (0 disables it).
    bool open(const std::string& path, std::uint64_t table_offset,
              std::uint64_t row_bytes, std::uint64_t rows,
              std::uint64_t cache_rows) {
#if defined(__linux__)
        close();
        if (row_bytes == 0 || row_bytes > kPage) return false;
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) return false;
        // No readahead: a row is 90 bytes of a random page.
        (void)posix_fadvise(fd_, 0, 0, POSIX_FADV_RANDOM);
        // Two pages per row: a row may straddle a page boundary.
        if (posix_memalign(reinterpret_cast<void**>(&pages_), kPage,
                           2 * kPage * kMaxRows) != 0) {
            pages_ = nullptr;
            close();
            return false;
        }
        path_ = path;
        table_offset_ = table_offset;
        row_bytes_ = row_bytes;
        rows_ = rows;
        cache_rows_ = cache_rows;
        tags_.assign(cache_rows, 0);
        cache_.assign(cache_rows * row_bytes, 0);
        raw_.assign(kMaxRows * row_bytes, 0);
        stop_ = false;
        for (int index = 0; index < kThreads; ++index)
            workers_.emplace_back([this] { work(); });
        keepalive_ms_ = 2;
        if (const char* setting = std::getenv("FLYWEIGHT_PLE_KEEPALIVE_MS"))
            keepalive_ms_ = std::atoi(setting);
        if (keepalive_ms_ > 0) keeper_ = std::thread([this] { keep_awake(); });
        // Probe one read so a file that opens but will not read is found
        // here, not in the middle of a decode.
        const std::uint64_t probe = 0;
        issue(&probe, 1);
        if (!collect()) {
            close();
            return false;
        }
        requests_ = hits_ = reads_ = wait_ns_ = 0;
        return true;
#else
        (void)path; (void)table_offset; (void)row_bytes; (void)rows;
        (void)cache_rows;
        return false;
#endif
    }

    void close() {
#if defined(__linux__)
        drain();
        {
            std::lock_guard<std::mutex> guard(mutex_);
            stop_ = true;
        }
        work_ready_.notify_all();
        keeper_wake_.notify_all();
        for (auto& worker : workers_) worker.join();
        if (keeper_.joinable()) keeper_.join();
        workers_.clear();
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        std::free(pages_);
        pages_ = nullptr;
#endif
    }

    bool is_open() const { return fd_ >= 0; }

    // Start fetching `count` rows. A previous ticket that was never collected
    // is waited out first, since its buffers are about to be reused.
    void issue(const std::uint64_t* rows, int count) {
        drain();
        if (count > kMaxRows) count = kMaxRows;
        count_ = count;
        int blocks = 0;
        for (int index = 0; index < count; ++index) {
            const std::uint64_t row = rows[index];
            ++requests_;
            Want& want = wants_[index];
            want.row = row;
            want.slot = -1;
            if (row >= rows_) {  // the mapping path throws first; zero here
                std::memset(&raw_[index * row_bytes_], 0, row_bytes_);
                continue;
            }
            if (cache_rows_) {
                const std::uint64_t at = cache_index(row);
                if (tags_[at] == row + 1) {
                    std::memcpy(&raw_[index * row_bytes_],
                                &cache_[at * row_bytes_], row_bytes_);
                    ++hits_;
                    continue;
                }
            }
            const std::uint64_t offset = table_offset_ + row * row_bytes_;
            const std::uint64_t start = offset / kPage * kPage;
            const std::uint64_t end =
                (offset + row_bytes_ + kPage - 1) / kPage * kPage;
            // Rows sharing a page span share one read.
            int slot = -1;
            for (int other = 0; other < blocks; ++other)
                if (blocks_[other].start == start &&
                    blocks_[other].bytes == end - start) {
                    slot = other;
                    break;
                }
            if (slot < 0) {
                slot = blocks++;
                blocks_[slot].start = start;
                blocks_[slot].bytes = end - start;
            }
            want.slot = slot;
            want.within = offset - start;
        }
        reads_ += static_cast<std::uint64_t>(blocks);
        {
            std::lock_guard<std::mutex> guard(mutex_);
            last_issue_ = std::chrono::steady_clock::now();
            last_io_ = last_issue_;
            failed_ = false;
            next_ = 0;
            blocks_total_ = blocks;
            pending_ = blocks;
        }
        if (blocks) work_ready_.notify_all();
    }

    // Wait for the ticket. On success `row(i)` holds row i's raw bytes until
    // the next `issue`.
    bool collect() {
        const auto started = std::chrono::steady_clock::now();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_done_.wait(lock, [this] { return pending_ == 0; });
        }
        wait_ns_ += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
        if (failed_) return false;
        for (int index = 0; index < count_; ++index) {
            const Want& want = wants_[index];
            if (want.slot < 0) continue;
            const std::uint8_t* source = pages_
                + static_cast<std::uint64_t>(want.slot) * 2 * kPage + want.within;
            std::memcpy(&raw_[index * row_bytes_], source, row_bytes_);
            if (cache_rows_) {
                const std::uint64_t at = cache_index(want.row);
                tags_[at] = want.row + 1;
                std::memcpy(&cache_[at * row_bytes_], source, row_bytes_);
            }
        }
        // Collected once: a repeat collect must not recache stale pages.
        for (int index = 0; index < count_; ++index) wants_[index].slot = -1;
        return true;
    }

    const std::uint8_t* row(int index) const {
        return raw_.data() + static_cast<std::uint64_t>(index) * row_bytes_;
    }

    std::uint64_t requests() const { return requests_; }
    std::uint64_t hits() const { return hits_; }
    std::uint64_t reads() const { return reads_; }
    std::uint64_t wait_nanoseconds() const { return wait_ns_; }

private:
    struct Want {
        std::uint64_t row = 0, within = 0;
        int slot = -1;
    };
    struct Block {
        std::uint64_t start = 0, bytes = 0;
    };

    std::uint64_t cache_index(std::uint64_t row) const {
        return (row * 0x9E3779B97F4A7C15ull >> 20) % cache_rows_;
    }

    void drain() {
        std::unique_lock<std::mutex> lock(mutex_);
        work_done_.wait(lock, [this] { return pending_ == 0; });
    }

    // Keep the SSD out of its low-power states while decode is reading.
    // A token issues its reads ~25 ms after the last ones, which is long
    // enough for APST to park this box's 990 EVO Plus: a read after a 25 ms
    // gap took 1.7-1.9 ms against 0.15-0.2 ms after a gap of 10 ms or less,
    // and all of a token's reads paid it together. So while reads have been
    // issued within the last second, one page is read whenever the drive has
    // been idle for FLYWEIGHT_PLE_KEEPALIVE_MS (default 2, 0 turns this off).
    // In decode that took a token's reads from 1.8 ms to 0.4 ms, under the
    // embedding plus layer 0. Strata's reader keeps its SSD awake the same way.
    void keep_awake() {
#if defined(__linux__)
        std::uint8_t* page = nullptr;
        if (posix_memalign(reinterpret_cast<void**>(&page), kPage, kPage) != 0)
            return;
        // A page of the table no decode path maps, at a stride that walks
        // the table, read O_DIRECT where the filesystem allows it: a page
        // that is resident -- or mapped, which DONTNEED cannot drop -- would
        // be served from memory and never reach the drive.
        const int direct = ::open(path_.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        const int source = direct >= 0 ? direct : fd_;
        const std::uint64_t first = (table_offset_ + kPage - 1) / kPage * kPage;
        const std::uint64_t span = (rows_ * row_bytes_) / kPage;
        std::uint64_t walk = 0;
        const auto period = std::chrono::milliseconds(keepalive_ms_);
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stop_) {
            keeper_wake_.wait_for(lock, period);
            if (stop_) break;
            const auto now = std::chrono::steady_clock::now();
            if (now - last_issue_ > std::chrono::seconds(1) ||
                now - last_io_ < period)
                continue;
            last_io_ = now;
            lock.unlock();
            walk = (walk + 7919) % (span > 1 ? span - 1 : 1);
            const std::uint64_t at = first + walk * kPage;
            if (::pread(source, page, kPage, static_cast<off_t>(at)) > 0 &&
                source == fd_)
                (void)posix_fadvise(fd_, static_cast<off_t>(at),
                                    static_cast<off_t>(kPage),
                                    POSIX_FADV_DONTNEED);
            lock.lock();
        }
        if (direct >= 0) ::close(direct);
        std::free(page);
#endif
    }

    void work() {
#if defined(__linux__)
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            work_ready_.wait(lock, [&] { return stop_ || next_ < blocks_total_; });
            if (stop_) return;
            const int slot = next_++;
            const Block block = blocks_[slot];
            lock.unlock();
            std::uint8_t* target =
                pages_ + static_cast<std::uint64_t>(slot) * 2 * kPage;
            std::uint64_t done = 0;
            bool ok = true;
            while (done < block.bytes) {
                const ssize_t got = ::pread(fd_, target + done,
                    block.bytes - done, static_cast<off_t>(block.start + done));
                if (got < 0 && errno == EINTR) continue;
                if (got <= 0) { ok = false; break; }
                done += static_cast<std::uint64_t>(got);
            }
            // The page is in the buffer now; dropping it keeps the table out
            // of the page cache, which belongs to the experts. Reuse is the
            // row cache's job.
            if (ok)
                (void)posix_fadvise(fd_, static_cast<off_t>(block.start),
                                    static_cast<off_t>(block.bytes),
                                    POSIX_FADV_DONTNEED);
            lock.lock();
            last_io_ = std::chrono::steady_clock::now();
            if (!ok) failed_ = true;
            if (--pending_ == 0) work_done_.notify_all();
        }
#endif
    }

    int fd_ = -1;
    std::uint8_t* pages_ = nullptr;
    Want wants_[kMaxRows]{};
    Block blocks_[kMaxRows]{};
    int count_ = 0;
    std::uint64_t table_offset_ = 0, row_bytes_ = 0, rows_ = 0, cache_rows_ = 0;
    std::vector<std::uint64_t> tags_;
    std::vector<std::uint8_t> cache_, raw_;
    std::uint64_t requests_ = 0, hits_ = 0, reads_ = 0, wait_ns_ = 0;

    std::mutex mutex_;
    std::condition_variable work_ready_, work_done_, keeper_wake_;
    std::vector<std::thread> workers_;
    std::thread keeper_;
    int keepalive_ms_ = 0;
    std::string path_;
    std::chrono::steady_clock::time_point last_issue_{}, last_io_{};
    bool stop_ = false, failed_ = false;
    int next_ = 0, blocks_total_ = 0, pending_ = 0;
};

}  // namespace flyweight::v2
