// Raon-1 batched prefill.
//
// The per-token path (raon::decoder_step) is a decode loop: every projection
// is a matvec, so a prompt of T tokens reads every weight T times and pays a
// host<->device round trip per projection per token. Prefill here runs a chunk
// of tokens layer-major instead. Every projection becomes one matmul over the
// chunk -- a GPU GEMM when the weight is resident (attention projections,
// pinned expert layers), the register-blocked CPU matmul otherwise -- and the
// MoE groups the chunk's routes by expert so each expert's weights are read
// once per chunk. Only what is inherently sequential stays per token: the
// short convolutions, the RDA recurrence (one fork per layer per chunk, each
// thread walking the chunk for its heads with the state hot in cache) and the
// mHC mixing.
//
// The math is the decode path's: the RDA head update, the mHC coefficients and
// the routing are the same functions decoder_step calls. What differs is GEMM
// accumulation order and, on the GPU MMQ path, Q8 activations.
//
// Included after v2_mtp_verifier.inc: the MMQ launches size themselves from
// qwen_mmq_shape() and the kQ8Mmq* constants defined there.

namespace raon_prefill {

namespace raon = flyweight::v2::raon;
using raon::Matrix;

// FLYWEIGHT_RAON_PREFILL_TRACE=1: wall time per prefill phase, printed after
// each prefill call. Phases nest, so the rows do not sum to the total.
struct Trace {
    std::vector<std::pair<const char*, double>> rows;
    static bool on() {
        static const bool enabled = [] {
            const char* value = std::getenv("FLYWEIGHT_RAON_PREFILL_TRACE");
            return value && value[0] == '1';
        }();
        return enabled;
    }
    void add(const char* name, double seconds) {
        for (auto& row : rows) if (row.first == name) { row.second += seconds; return; }
        rows.push_back({name, seconds});
    }
};
inline Trace trace_rows;
class Phase {
public:
    explicit Phase(const char* name) : name_(name), start_(Trace::on() ? now() : 0.0) {}
    ~Phase() { if (Trace::on()) trace_rows.add(name_, now() - start_); }
private:
    static double now() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    const char* name_;
    double start_;
};

static void launch(const char* name, std::uint32_t grid_x, std::uint32_t grid_y,
                   std::uint32_t block, void** arguments, std::uint32_t shared = 0) {
    if (flyweight_gpu_launch_named(name, grid_x, grid_y, block, shared, 0, arguments) != 0)
        throw std::runtime_error(std::string("raon prefill: launch failed: ") + name);
}

// output[rows][out] = input[rows][in] * W^T on the device. Quantized weights
// whose width tiles the MMQ take Q8 activations on the int8 tensor cores;
// everything else takes the format's reconstruct-in-float rows kernel.
static void gemm(const FlyweightV2RaonRuntime& rt, std::uint32_t type, std::uint64_t matrix,
                 std::uint64_t input, std::uint64_t output, int rows, int in, int out) {
    if (rows <= 0) return;
    const auto* format = flyweight::v2::qwen_format(type);
    if (!format) throw std::runtime_error("raon prefill: no GPU kernels for weight type " + std::to_string(type));
    // The MMQ tile is 128 tokens by 128 rows: a small expert group fills a
    // sliver of it and launches only out/128 blocks, so below
    // mmq_min_rows the rows kernel (thousands of blocks) wins.
    static const int mmq_min_rows = [] {
        const char* value = std::getenv("FLYWEIGHT_RAON_MMQ_MIN_ROWS");
        return value ? std::atoi(value) : 0;
    }();
    if (format->matmul_q8_mmq && rt.int8_tensor_cores && (in & 255) == 0 && rows >= mmq_min_rows) {
        int scale_stride = (in / 32) * 2;
        std::uint64_t q8 = rt.prefill_q8, scales = rt.prefill_scales;
        {
            void* arguments[] = {&input, &q8, &scales, &in, &scale_stride};
            launch("quantize_q8_blocks_rows", static_cast<std::uint32_t>((in + 255) / 256),
                   static_cast<std::uint32_t>(rows), 256, arguments);
        }
        const auto shape = qwen_mmq_shape();
        const int mmq_rows = format->mmq_min ? kQ8MmqMinRows : shape.rows();
        const int mmq_threads = format->mmq_min ? kQ8MmqMinThreads : shape.threads();
        const int mmq_tokens = format->mmq_min ? kQ8MmqMinTokens : shape.tokens();
        const std::uint32_t shared =
            (format->mmq_dynamic_shared && !format->mmq_min) ? qwen_mmq_dynamic_shared(shape) : 0u;
        for (int start = 0; start < rows; start += mmq_tokens) {
            int batch = std::min(mmq_tokens, rows - start);
            std::uint64_t batch_q8 = q8 + static_cast<std::uint64_t>(start) * in;
            std::uint64_t batch_scales = scales + static_cast<std::uint64_t>(start) * (in / 32) * sizeof(float);
            std::uint64_t batch_output = output + static_cast<std::uint64_t>(start) * out * sizeof(float);
            void* arguments[] = {&matrix, &batch_q8, &batch_scales, &batch_output,
                                 &in, &out, &batch, &scale_stride};
            launch(format->matmul_q8_mmq, static_cast<std::uint32_t>((out + mmq_rows - 1) / mmq_rows), 1,
                   static_cast<std::uint32_t>(mmq_threads), arguments, shared);
        }
        return;
    }
    if (!format->matmul_rows)
        throw std::runtime_error("raon prefill: weight type " + std::to_string(type) + " has no rows kernel");
    void* arguments[] = {&matrix, &input, &output, &in, &out, &rows};
    switch (format->matmul_rows_grid) {
    case flyweight::v2::RowsMatmulGrid::tiled32:
        launch(format->matmul_rows, static_cast<std::uint32_t>((out + 31) / 32),
               static_cast<std::uint32_t>((rows + 31) / 32), 256, arguments);
        break;
    case flyweight::v2::RowsMatmulGrid::quad_pack:
        launch(format->matmul_rows, static_cast<std::uint32_t>(out),
               static_cast<std::uint32_t>((rows + 3) / 4), 256, arguments);
        break;
    default:
        launch(format->matmul_rows, static_cast<std::uint32_t>(out),
               static_cast<std::uint32_t>(rows), 256, arguments);
        break;
    }
}

static void parallel_copy(void* target, const void* source, std::size_t bytes) {
    constexpr std::size_t piece = 1u << 20;
    const auto pieces = static_cast<std::int64_t>((bytes + piece - 1) / piece);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(std::min(8, flyweight::v2::bailing::matvec_threads())) \
        if (pieces >= 4)
#endif
    for (std::int64_t i = 0; i < pieces; ++i) {
        const std::size_t begin = static_cast<std::size_t>(i) * piece;
        std::memcpy(static_cast<std::uint8_t*>(target) + begin,
                    static_cast<const std::uint8_t*>(source) + begin, std::min(piece, bytes - begin));
    }
}

// Activation transfers go through the pinned staging buffer in two halves,
// ping-ponged on events, so the DMA runs at pinned speed (pageable copies
// measured 3.6 GB/s against 28.8). Uploads are ordered on stream 0.
static FlyweightV2RaonRuntime* transfer_runtime = nullptr;

static void upload_bytes(std::uint64_t device, const void* host, std::size_t bytes) {
    auto* rt = transfer_runtime;
    if (!rt || !rt->transfer_host) {
        if (flyweight_gpu_upload(device, host, bytes, 0) != 0)
            throw std::runtime_error("raon prefill: upload failed");
        return;
    }
    const std::size_t half = rt->transfer_bytes / 2;
    static int turn = 0;
    for (std::size_t done = 0; done < bytes; done += half) {
        const std::size_t size = std::min(half, bytes - done);
        auto* staging = static_cast<std::uint8_t*>(rt->transfer_host) + turn * half;
        if (flyweight_gpu_event_sync(rt->transfer_events[turn]) != 0)
            throw std::runtime_error("raon prefill: staging wait failed");
        parallel_copy(staging, static_cast<const std::uint8_t*>(host) + done, size);
        if (flyweight_gpu_upload(device + done, staging, size, 0) != 0 ||
            flyweight_gpu_event_record(rt->transfer_events[turn], 0) != 0)
            throw std::runtime_error("raon prefill: upload failed");
        turn ^= 1;
    }
}

static void upload(std::uint64_t device, const float* host, std::size_t elements) {
    upload_bytes(device, host, elements * sizeof(float));
}

static void download(float* host, std::uint64_t device, std::size_t elements) {
    Phase phase("download+sync");
    auto* rt = transfer_runtime;
    const std::size_t bytes = elements * sizeof(float);
    if (!rt || !rt->transfer_host) {
        if (flyweight_gpu_download(host, device, bytes, 0) != 0 || flyweight_gpu_stream_sync(0) != 0)
            throw std::runtime_error("raon prefill: download failed");
        return;
    }
    const std::size_t half = rt->transfer_bytes / 2;
    for (std::size_t done = 0; done < bytes; done += half) {
        const std::size_t size = std::min(half, bytes - done);
        if (flyweight_gpu_download(rt->transfer_host, device + done, size, 0) != 0 ||
            flyweight_gpu_stream_sync(0) != 0)
            throw std::runtime_error("raon prefill: download failed");
        parallel_copy(reinterpret_cast<std::uint8_t*>(host) + done, rt->transfer_host, size);
    }
}

// ---------------------------------------------------------------------------
// Weight streaming
// ---------------------------------------------------------------------------
//
// Prefill over a long chunk is GEMM-bound on the device but weight-bound on the
// host: a CPU-resident expert layer read once per chunk by the CPU matmul costs
// more than copying it over PCIe and running it on the GPU. So for chunks of
// at least stream_min_rows tokens, weights that are not resident stream
// through a small slot ring: the host copies an item (one matrix, or a few
// experts' gate/up/down) from the mapped file into a pinned slot, the copy
// stream DMAs it into the matching device slot, and stream 0 waits on that
// slot's upload event before the item's GEMMs. A slot is refilled one item
// behind the launches, so the host copy of item i+R-1 overlaps the GPU running
// item i. Pageable mmap uploads measured 3.6 GB/s against 28.8 GB/s pinned on
// the dev box, which is why the pinned hop is worth its memcpy.

struct StreamItem {
    std::vector<std::pair<const void*, std::size_t>> segments;
    std::function<void(std::uint64_t slot)> run;
};

static bool streaming(const FlyweightV2RaonRuntime& rt, std::size_t rows) {
    return rt.stream_slots >= 2 && rows >= rt.stream_min_rows;
}

static void stage(FlyweightV2RaonRuntime& rt, const StreamItem& item, std::uint64_t sequence) {
    const auto slot = static_cast<std::size_t>(sequence % rt.stream_slots);
    // The slot's previous item must be done on the device before its bytes go.
    {
        Phase wait("stream: slot wait");
        if (sequence >= rt.stream_slots && flyweight_gpu_event_sync(rt.stream_consumed[slot]) != 0)
            throw std::runtime_error("raon prefill: stream slot wait failed");
    }
    Phase copy("stream: host copy+issue");
    auto* host = static_cast<std::uint8_t*>(rt.stream_host) + slot * rt.stream_slot_bytes;
    const std::uint64_t device = rt.stream_device + slot * rt.stream_slot_bytes;
    // Segments already in page-locked memory DMA straight from where they
    // are; the rest go through one parallel copy into the pinned slot (over
    // 512 KiB pieces: an item is a few 2-3 MB expert slices, each too small to
    // split well on its own) and up from there.
    const auto* pinned_begin = static_cast<const std::uint8_t*>(rt.pinned_experts);
    const auto* pinned_end = pinned_begin ? pinned_begin + rt.pinned_expert_bytes : nullptr;
    struct Piece { const std::uint8_t* source; std::uint8_t* target; std::size_t bytes; };
    std::vector<Piece> pieces;
    std::vector<std::pair<std::size_t, std::size_t>> staged;   // (offset, bytes) in the slot
    std::size_t total = 0;
    for (const auto& segment : item.segments) {
        if (total + segment.second > rt.stream_slot_bytes)
            throw std::runtime_error("raon prefill: stream item larger than a slot");
        const auto* source = static_cast<const std::uint8_t*>(segment.first);
        if (pinned_begin && source >= pinned_begin && source + segment.second <= pinned_end) {
            if (flyweight_gpu_upload(device + total, source, segment.second, rt.copy_stream) != 0)
                throw std::runtime_error("raon prefill: stream upload failed");
        } else {
            constexpr std::size_t piece = 512u << 10;
            for (std::size_t begin = 0; begin < segment.second; begin += piece)
                pieces.push_back({source + begin, host + total + begin,
                                  std::min(piece, segment.second - begin)});
            staged.push_back({total, segment.second});
        }
        total += segment.second;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(std::min(12, flyweight::v2::bailing::matvec_threads())) \
        if (pieces.size() >= 4)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(pieces.size()); ++i)
        std::memcpy(pieces[i].target, pieces[i].source, pieces[i].bytes);
    for (const auto& [offset, bytes] : staged)
        if (flyweight_gpu_upload(device + offset, host + offset, bytes, rt.copy_stream) != 0)
            throw std::runtime_error("raon prefill: stream upload failed");
    if (flyweight_gpu_event_record(rt.stream_uploaded[slot], rt.copy_stream) != 0)
        throw std::runtime_error("raon prefill: stream upload failed");
}

static void execute(FlyweightV2RaonRuntime& rt, const std::vector<StreamItem>& items) {
    Phase phase("stream: execute");
    const std::size_t count = items.size(), slots = rt.stream_slots;
    const std::uint64_t base = rt.stream_sequence;
    std::size_t staged = 0;
    for (; staged < std::min(count, slots); ++staged) stage(rt, items[staged], base + staged);
    for (std::size_t i = 0; i < count; ++i) {
        const auto slot = static_cast<std::size_t>((base + i) % slots);
        if (flyweight_gpu_stream_wait_event(0, rt.stream_uploaded[slot]) != 0)
            throw std::runtime_error("raon prefill: stream wait failed");
        items[i].run(rt.stream_device + slot * rt.stream_slot_bytes);
        if (flyweight_gpu_event_record(rt.stream_consumed[slot], 0) != 0)
            throw std::runtime_error("raon prefill: stream record failed");
        // Refill the slot of the item before this one: the device still has
        // item i queued while the host copies.
        if (i >= 1 && staged < count) {
            stage(rt, items[staged], base + staged);
            ++staged;
        }
    }
    rt.stream_sequence = base + count;
}

// One projection over `rows` tokens: the device GEMM when `device` holds the
// weight, the CPU matmul otherwise.
static void linear(FlyweightV2RaonRuntime& rt, const Matrix& cpu, std::uint64_t device,
                   std::uint32_t device_type, const float* input, std::size_t rows,
                   std::size_t in, std::size_t out, float* output) {
    const std::size_t bytes = out * flyweight::v2::bailing::row_bytes(cpu.type, in);
    if (!device && streaming(rt, rows) && bytes <= rt.stream_slot_bytes &&
        flyweight::v2::qwen_format(cpu.type)) {
        upload(rt.prefill_in, input, rows * in);
        std::vector<StreamItem> items(1);
        items[0].segments.push_back({cpu.data, bytes});
        items[0].run = [&](std::uint64_t slot) {
            gemm(rt, cpu.type, slot, rt.prefill_in, rt.prefill_out, static_cast<int>(rows),
                 static_cast<int>(in), static_cast<int>(out));
        };
        execute(rt, items);
        download(output, rt.prefill_out, rows * out);
        return;
    }
    if (device) {
        upload(rt.prefill_in, input, rows * in);
        gemm(rt, device_type, device, rt.prefill_in, rt.prefill_out,
             static_cast<int>(rows), static_cast<int>(in), static_cast<int>(out));
        download(output, rt.prefill_out, rows * out);
        return;
    }
    flyweight::v2::bailing::matmul(cpu, input, rows, in, out, output);
}

// mHC pre-mixing for every token of the chunk: one matmul for the mixes, then
// the per-token coefficients and branch input (raon_mhc_pre, batched).
static void mhc_pre_rows(const raon::RaonMhcWeights& w, const raon::Geometry& g,
                         const float* streams, std::size_t rows, float* mixes,
                         float* branch, float* post, float* comb) {
    const std::size_t hc = g.mhc_streams, width = hc * g.hidden;
    const std::size_t mix = 2 * hc + (hc - 1) * (hc - 1);
    flyweight::v2::bailing::matmul(w.fn, streams, rows, width, mix, mixes);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(flyweight::v2::bailing::matvec_threads())
#endif
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(rows); ++index) {
        const auto t = static_cast<std::size_t>(index);
        const float* x = streams + t * width;
        double total = 0.0;
        for (std::size_t i = 0; i < width; ++i) total += static_cast<double>(x[i]) * x[i];
        const float inv_rms = 1.0f / std::sqrt(
            static_cast<float>(total / static_cast<double>(width)) + g.mhc_eps);
        float* m = mixes + t * mix;
        for (std::size_t r = 0; r < mix; ++r) m[r] *= inv_rms;
        float pre[4];
        raon::raon_mhc_coefficients(m, w, g, pre, post + t * hc, comb + t * hc * hc);
        float* out = branch + t * g.hidden;
        for (std::size_t j = 0; j < g.hidden; ++j)
            out[j] = pre[0] * x[j] + pre[1] * x[g.hidden + j] +
                     pre[2] * x[2 * g.hidden + j] + pre[3] * x[3 * g.hidden + j];
    }
}

static void mhc_post_rows(const raon::Geometry& g, float* streams, const float* branch,
                          const float* post, const float* comb, std::size_t rows) {
    const std::size_t hc = g.mhc_streams, width = hc * g.hidden;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(flyweight::v2::bailing::matvec_threads())
#endif
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(rows); ++index) {
        const auto t = static_cast<std::size_t>(index);
        std::vector<float> next(width);
        raon::raon_mhc_post(streams + t * width, branch + t * g.hidden, post + t * hc,
                            comb + t * hc * hc, g, next.data());
        std::memcpy(streams + t * width, next.data(), width * sizeof(float));
    }
}

static void rms_rows(const float* input, const float* gain, std::size_t rows, std::size_t width,
                     float epsilon, float* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(flyweight::v2::bailing::matvec_threads()) \
        if (rows >= 16)
#endif
    for (std::int64_t t = 0; t < static_cast<std::int64_t>(rows); ++t)
        flyweight::v2::bailing::rms_norm(input + t * width, gain, width, epsilon, output + t * width);
}

// A [rows][width] view into a packed projection output.
struct Rows {
    const float* base = nullptr;
    std::size_t stride = 0;
    const float* operator[](std::size_t t) const { return base + t * stride; }
};

struct Workspace {
    std::vector<float> streams, mixes, branch, post, comb, normalized, branch_out;
    std::vector<float> a, b, c, d, e, f;   // per-layer scratch, reused
};

// The small RDA tensors as one f32 block the device path streams per layer,
// built on first use: [f_a | f_b | beta | rda | q_conv | k_conv | v_conv |
// dt_bias | a | o_norm].
struct RdaPack {
    std::size_t f_a, f_b, beta, rda, q_conv, k_conv, v_conv, dt_bias, a, o_norm, total;
};

static RdaPack rda_pack_layout(const raon::Geometry& g) {
    const std::size_t ch = g.heads * g.head_dim;
    RdaPack p{};
    std::size_t at = 0;
    p.f_a = at; at += g.ssm_dt_rank * g.hidden;
    p.f_b = at; at += ch * g.ssm_dt_rank;
    p.beta = at; at += g.heads * g.hidden;
    p.rda = at; at += g.heads * g.hidden;
    p.q_conv = at; at += ch * g.conv_width;
    p.k_conv = at; at += ch * g.conv_width;
    p.v_conv = at; at += ch * g.conv_width;
    p.dt_bias = at; at += ch;
    p.a = at; at += g.heads;
    p.o_norm = at; at += g.head_dim;
    p.total = at;
    return p;
}

static const std::vector<float>& rda_pack(FlyweightV2RaonRuntime& rt, std::size_t l) {
    if (rt.rda_pack.size() < rt.layers.size()) rt.rda_pack.resize(rt.layers.size());
    auto& pack = rt.rda_pack[l];
    if (!pack.empty()) return pack;
    const auto& g = rt.geometry;
    const auto& w = rt.layers[l].rda;
    const auto p = rda_pack_layout(g);
    const std::size_t ch = g.heads * g.head_dim;
    pack.assign(p.total, 0.0f);
    flyweight::v2::bailing::matrix_decode(w.f_a, g.hidden, g.ssm_dt_rank, pack.data() + p.f_a);
    flyweight::v2::bailing::matrix_decode(w.f_b, g.ssm_dt_rank, ch, pack.data() + p.f_b);
    flyweight::v2::bailing::matrix_decode(w.beta, g.hidden, g.heads, pack.data() + p.beta);
    flyweight::v2::bailing::matrix_decode(w.rda, g.hidden, g.heads, pack.data() + p.rda);
    std::memcpy(pack.data() + p.q_conv, w.q_conv, ch * g.conv_width * sizeof(float));
    std::memcpy(pack.data() + p.k_conv, w.k_conv, ch * g.conv_width * sizeof(float));
    std::memcpy(pack.data() + p.v_conv, w.v_conv, ch * g.conv_width * sizeof(float));
    if (w.dt_bias) std::memcpy(pack.data() + p.dt_bias, w.dt_bias, ch * sizeof(float));
    std::memcpy(pack.data() + p.a, w.a, g.heads * sizeof(float));
    std::memcpy(pack.data() + p.o_norm, w.o_norm, g.head_dim * sizeof(float));
    return pack;
}

// output[rows][N] = input[rows][K] * W^T for an f32 W [N][K] on the device
// (raon_head_gemm with one head).
static void gemm_f32(std::uint64_t input, std::uint64_t weights, std::uint64_t output,
                     int rows, int K, int N) {
    long long a_row = K, b_head = 0, c_row = N;
    int a_head = 0, b_k = 1, b_n = K, c_head = 0, b_half = 0;
    void* arguments[] = {&input, &weights, &output, &rows, &K, &N, &a_row, &a_head, &b_head,
                         &b_k, &b_n, &c_row, &c_head, &b_half};
    launch("raon_head_gemm", static_cast<std::uint32_t>((N + 63) / 64),
           static_cast<std::uint32_t>((rows + 63) / 64), 256, arguments);
}

static bool rda_device_ready(const FlyweightV2RaonRuntime& rt, std::size_t rows) {
    const auto& g = rt.geometry;
    // In device decode the state lives on the device, so every chunk runs here.
    return (rt.device_decode || streaming(rt, rows)) && rt.stream_slots >= 2 && rt.prefill_rda_state &&
           g.head_dim == 128 && g.conv_width == 4 &&
           rda_pack_layout(g).total * sizeof(float) <= rt.stream_slot_bytes;
}

// A whole RDA layer of a chunk on the device: the four projections, the
// convolutions, the dt/beta projections, the recurrence and the output
// projection, with only x going up and the layer output (plus the state and
// windows) coming back. Device layout over the prefill scratch:
//   prefill_in:  x [T][hidden] | f_b [T][ch] | attended [T][ch] | f_a [T][rank]
//                | beta [T][heads] | delta [T][heads]
//   prefill_out: the projections (packed [T][4ch], else four [T][ch])
//   prefill_gate / up / act: the convolved q / k / v [T][ch]
static void rda_rows_device(FlyweightV2RaonRuntime& rt, std::size_t l,
                            const raon::GpuLayerDispatch* gl, RaonSlot& slot,
                            const float* x, std::size_t rows, float* branch_out) {
    raon::LayerCache& cache = slot.caches[l];
    // Device decode keeps the state in the slot; otherwise it makes the round
    // trip through the prefill state scratch.
    const bool resident_state = rt.device_decode;
    const std::uint64_t state_dev = resident_state ? slot.rda_state[l] : rt.prefill_rda_state;
    const std::uint64_t window_dev = resident_state ? slot.rda_window[l] : rt.prefill_rda_window;
    const auto& g = rt.geometry;
    const auto& layer = rt.layers[l];
    const auto& w = layer.rda;
    const std::size_t ch = g.heads * g.head_dim, T = rows;
    const int n = static_cast<int>(rows);
    const auto f32 = sizeof(float);
    const std::uint64_t x_dev = rt.prefill_in;
    const std::uint64_t fb_dev = x_dev + T * g.hidden * f32;
    const std::uint64_t att_dev = fb_dev + T * ch * f32;
    const std::uint64_t fa_dev = att_dev + T * ch * f32;
    const std::uint64_t beta_dev = fa_dev + T * g.ssm_dt_rank * f32;
    const std::uint64_t delta_dev = beta_dev + T * g.heads * f32;
    const std::uint64_t proj = rt.prefill_out;

    upload(x_dev, x, T * g.hidden);
    const std::size_t history = g.conv_width - 1;
    if (!resident_state) {
        std::vector<float> windows(3 * ch * history);
        std::memcpy(windows.data(), cache.q_window.data(), ch * history * f32);
        std::memcpy(windows.data() + ch * history, cache.k_window.data(), ch * history * f32);
        std::memcpy(windows.data() + 2 * ch * history, cache.v_window.data(), ch * history * f32);
        upload(rt.prefill_rda_window, windows.data(), windows.size());
        upload(rt.prefill_rda_state, cache.rda_state.data(), cache.rda_state.size());
    }

    // Projections: q, k, v, gate with their row strides.
    std::uint64_t q_dev, k_dev, v_dev, gate_dev;
    long long stride;
    const bool device = gl && gl->attn_gpu;
    std::vector<StreamItem> items;
    if (device && gl->attn_packed) {
        gemm(rt, gl->attn_type[0], gl->attn_weight[0], x_dev, proj, n, static_cast<int>(g.hidden),
             static_cast<int>(4 * ch));
        q_dev = proj; k_dev = proj + ch * f32; v_dev = proj + 2 * ch * f32; gate_dev = proj + 3 * ch * f32;
        stride = static_cast<long long>(4 * ch);
    } else {
        const Matrix* cpu[4] = {&w.q, &w.k, &w.v, &w.gate};
        for (std::size_t slot = 0; slot < 4; ++slot) {
            const std::uint64_t target = proj + slot * T * ch * f32;
            if (device && gl->attn_weight[slot]) {
                gemm(rt, gl->attn_type[slot], gl->attn_weight[slot], x_dev, target, n,
                     static_cast<int>(g.hidden), static_cast<int>(ch));
                continue;
            }
            StreamItem item;
            const auto type = cpu[slot]->type;
            item.segments.push_back({cpu[slot]->data, ch * flyweight::v2::bailing::row_bytes(type, g.hidden)});
            item.run = [&rt, type, x_dev, target, n, &g, ch](std::uint64_t weights) {
                gemm(rt, type, weights, x_dev, target, n, static_cast<int>(g.hidden), static_cast<int>(ch));
            };
            items.push_back(std::move(item));
        }
        q_dev = proj; k_dev = proj + T * ch * f32; v_dev = proj + 2 * T * ch * f32;
        gate_dev = proj + 3 * T * ch * f32;
        stride = static_cast<long long>(ch);
    }

    // The small tensors, then everything that reads them.
    const auto layout = rda_pack_layout(g);
    const auto& pack = rda_pack(rt, l);
    StreamItem small;
    small.segments.push_back({pack.data(), pack.size() * f32});
    small.run = [&](std::uint64_t base) {
        const auto at = [&](std::size_t offset) { return base + offset * f32; };
        gemm_f32(x_dev, at(layout.f_a), fa_dev, n, static_cast<int>(g.hidden), static_cast<int>(g.ssm_dt_rank));
        gemm_f32(fa_dev, at(layout.f_b), fb_dev, n, static_cast<int>(g.ssm_dt_rank), static_cast<int>(ch));
        gemm_f32(x_dev, at(layout.beta), beta_dev, n, static_cast<int>(g.hidden), static_cast<int>(g.heads));
        gemm_f32(x_dev, at(layout.rda), delta_dev, n, static_cast<int>(g.hidden), static_cast<int>(g.heads));
        const std::uint64_t conv_in[3] = {q_dev, k_dev, v_dev};
        const std::uint64_t conv_out[3] = {rt.prefill_gate, rt.prefill_up, rt.prefill_act};
        const std::size_t conv_weights[3] = {layout.q_conv, layout.k_conv, layout.v_conv};
        for (int c = 0; c < 3; ++c) {
            std::uint64_t input = conv_in[c], weights = at(conv_weights[c]), output = conv_out[c];
            std::uint64_t window = window_dev + c * ch * history * f32;
            int tokens = n, channels = static_cast<int>(ch);
            void* arguments[] = {&input, &stride, &weights, &window, &output, &tokens, &channels};
            launch("raon_rda_conv", static_cast<std::uint32_t>((ch + 127) / 128), 1, 128, arguments);
        }
        std::uint64_t qc = rt.prefill_gate, kc = rt.prefill_up, vc = rt.prefill_act;
        std::uint64_t fb = fb_dev, beta = beta_dev, delta = delta_dev, gate = gate_dev;
        std::uint64_t a = at(layout.a), dt = at(layout.dt_bias), norm = at(layout.o_norm);
        std::uint64_t state = state_dev, out = att_dev;
        int tokens = n, heads = static_cast<int>(g.heads), kmajor = resident_state ? 1 : 0;
        float lower = g.kda_gate_lower_bound, epsilon = g.epsilon;
        void* arguments[] = {&qc, &kc, &vc, &fb, &beta, &delta, &gate, &stride, &a, &dt, &norm,
                             &state, &out, &tokens, &heads, &lower, &epsilon, &kmajor};
        const std::uint32_t shared = (128 * 128 + 3 * 128 + 4) * f32;
        launch("raon_rda_recurrence", static_cast<std::uint32_t>(g.heads), 1, 128, arguments, shared);
    };
    items.push_back(std::move(small));

    // The output projection reads the attended rows where they are.
    const std::uint64_t out_dev = rt.prefill_out;
    if (device && gl->attn_weight[4]) {
        execute(rt, items);
        gemm(rt, gl->attn_type[4], gl->attn_weight[4], att_dev, out_dev, n, static_cast<int>(ch),
             static_cast<int>(g.hidden));
    } else {
        StreamItem item;
        const auto type = w.output.type;
        item.segments.push_back({w.output.data, g.hidden * flyweight::v2::bailing::row_bytes(type, ch)});
        item.run = [&rt, type, att_dev, out_dev, n, &g, ch](std::uint64_t weights) {
            gemm(rt, type, weights, att_dev, out_dev, n, static_cast<int>(ch), static_cast<int>(g.hidden));
        };
        items.push_back(std::move(item));
        execute(rt, items);
    }
    download(branch_out, out_dev, T * g.hidden);
    if (!resident_state) {
        std::vector<float> windows(3 * ch * history);
        download(windows.data(), rt.prefill_rda_window, windows.size());
        std::memcpy(cache.q_window.data(), windows.data(), ch * history * f32);
        std::memcpy(cache.k_window.data(), windows.data() + ch * history, ch * history * f32);
        std::memcpy(cache.v_window.data(), windows.data() + 2 * ch * history, ch * history * f32);
        download(cache.rda_state.data(), rt.prefill_rda_state, cache.rda_state.size());
    }
}

static void rda_rows(FlyweightV2RaonRuntime& rt, const raon::LayerWeights& layer,
                     const raon::GpuLayerDispatch* gl, raon::LayerCache& cache,
                     const float* x, std::size_t rows, Workspace& ws, float* branch_out) {
    const auto& g = rt.geometry;
    const auto& w = layer.rda;
    const std::size_t ch = g.heads * g.head_dim;
    raon::ProfileScope projection(raon::profile().rda_projection);

    Rows q, k, v, gate;
    const bool device = gl && gl->attn_gpu;
    if (device && gl->attn_packed) {
        ws.a.resize(rows * 4 * ch);
        upload(rt.prefill_in, x, rows * g.hidden);
        gemm(rt, gl->attn_type[0], gl->attn_weight[0], rt.prefill_in, rt.prefill_out,
             static_cast<int>(rows), static_cast<int>(g.hidden), static_cast<int>(4 * ch));
        download(ws.a.data(), rt.prefill_out, rows * 4 * ch);
        q = {ws.a.data(), 4 * ch};
        k = {ws.a.data() + ch, 4 * ch};
        v = {ws.a.data() + 2 * ch, 4 * ch};
        gate = {ws.a.data() + 3 * ch, 4 * ch};
    } else {
        ws.a.resize(rows * 4 * ch);
        const Matrix* cpu[4] = {&w.q, &w.k, &w.v, &w.gate};
        for (std::size_t s = 0; s < 4; ++s)
            linear(rt, *cpu[s], device ? gl->attn_weight[s] : 0, device ? gl->attn_type[s] : 0,
                   x, rows, g.hidden, ch, ws.a.data() + s * rows * ch);
        q = {ws.a.data(), ch};
        k = {ws.a.data() + rows * ch, ch};
        v = {ws.a.data() + 2 * rows * ch, ch};
        gate = {ws.a.data() + 3 * rows * ch, ch};
    }

    std::optional<Phase> small(std::in_place, "rda: conv+small matmuls");
    // Short convolutions advance the window token by token.
    ws.b.resize(rows * 3 * ch);
    float* qc = ws.b.data();
    float* kc = qc + rows * ch;
    float* vc = kc + rows * ch;
    for (std::size_t t = 0; t < rows; ++t) {
        flyweight::v2::bailing::short_conv_step(q[t], w.q_conv, ch, g.conv_width,
                                                cache.q_window.data(), qc + t * ch);
        flyweight::v2::bailing::short_conv_step(k[t], w.k_conv, ch, g.conv_width,
                                                cache.k_window.data(), kc + t * ch);
        flyweight::v2::bailing::short_conv_step(v[t], w.v_conv, ch, g.conv_width,
                                                cache.v_window.data(), vc + t * ch);
    }

    ws.c.resize(rows * g.ssm_dt_rank);
    ws.d.resize(rows * ch);
    ws.e.resize(rows * 2 * g.heads);
    flyweight::v2::bailing::matmul(w.f_a, x, rows, g.hidden, g.ssm_dt_rank, ws.c.data());
    flyweight::v2::bailing::matmul(w.f_b, ws.c.data(), rows, g.ssm_dt_rank, ch, ws.d.data());
    float* beta = ws.e.data();
    float* delta = beta + rows * g.heads;
    flyweight::v2::bailing::matmul(w.beta, x, rows, g.hidden, g.heads, beta);
    flyweight::v2::bailing::matmul(w.rda, x, rows, g.hidden, g.heads, delta);

    small.reset();
    {
        Phase phase("rda: recurrence");
        raon::ProfileScope recurrence(raon::profile().rda_recurrence);
        ws.f.resize(rows * ch);
        float* attended = ws.f.data();
        const float* f_b = ws.d.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(flyweight::v2::bailing::matvec_threads())
#endif
        for (std::int64_t head_index = 0; head_index < static_cast<std::int64_t>(g.heads); ++head_index) {
            const auto h = static_cast<std::size_t>(head_index);
            const std::size_t base = h * g.head_dim;
            float* state = cache.rda_state.data() + h * g.head_dim * g.head_dim;
            for (std::size_t t = 0; t < rows; ++t)
                raon::rda_head_token(qc + t * ch + base, kc + t * ch + base, vc + t * ch + base,
                                     f_b + t * ch + base, beta[t * g.heads + h],
                                     delta[t * g.heads + h], gate[t] + base, w, g, h, state,
                                     attended + t * ch + base);
        }
    }

    linear(rt, w.output, device ? gl->attn_weight[4] : 0, device ? gl->attn_type[4] : 0,
           ws.f.data(), rows, ch, g.hidden, branch_out);
}

// One head of the absorbed MLA attention for one query over `positions`
// cached latents (bailing::mla_attention_absorbed, per head).
static void mla_head(const float* query_nope, const float* query_rope, const float* head_weights,
                     const float* latents, const float* rope_keys, std::size_t positions,
                     const raon::Geometry& g, float* projected, float* accumulated,
                     float* scores, float* target) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(g.qk_nope + g.qk_rope));
    for (std::size_t i = 0; i < g.kv_lora; ++i) projected[i] = 0.0f;
    for (std::size_t row = 0; row < g.qk_nope; ++row) {
        const float weight = query_nope[row];
        const float* source = head_weights + row * g.kv_lora;
        for (std::size_t i = 0; i < g.kv_lora; ++i) projected[i] += weight * source[i];
    }
    for (std::size_t p = 0; p < positions; ++p) {
        const float* latent = latents + p * g.kv_lora;
        float total = 0.0f;
        for (std::size_t i = 0; i < g.kv_lora; ++i) total += projected[i] * latent[i];
        const float* rope = rope_keys + p * g.qk_rope;
        for (std::size_t i = 0; i < g.qk_rope; ++i) total += query_rope[i] * rope[i];
        scores[p] = total * scale;
    }
    flyweight::v2::bailing::softmax_in_place(scores, positions);
    for (std::size_t i = 0; i < g.kv_lora; ++i) accumulated[i] = 0.0f;
    for (std::size_t p = 0; p < positions; ++p) {
        const float weight = scores[p];
        const float* latent = latents + p * g.kv_lora;
        for (std::size_t i = 0; i < g.kv_lora; ++i) accumulated[i] += latent[i] * weight;
    }
    for (std::size_t row = 0; row < g.v_head_dim; ++row) {
        const float* source = head_weights + (g.qk_nope + row) * g.kv_lora;
        float total = 0.0f;
        for (std::size_t i = 0; i < g.kv_lora; ++i) total += source[i] * accumulated[i];
        target[row] = total;
    }
}

// Device cache rows the MLA attention reads, in their storage type.
struct MlaKeys { std::uint64_t latents = 0, rope = 0; int type = 0; };

// raon_head_gemm launch; B is f16 when b_half, else f32.
static void head_gemm(std::uint64_t A, std::uint64_t B, std::uint64_t C, int n, int heads, int K, int N,
                      long long a_row, int a_head, long long b_head, int b_k, int b_n,
                      long long c_row, int c_head, int b_half) {
    void* arguments[] = {&A, &B, &C, &n, &K, &N, &a_row, &a_head, &b_head, &b_k, &b_n,
                         &c_row, &c_head, &b_half};
    launch("raon_head_gemm", static_cast<std::uint32_t>((N + 63) / 64),
           static_cast<std::uint32_t>((n + 63) / 64 * heads), 256, arguments);
}

// Attention of `n` tokens (n * heads query rows at `queries`, the first at
// position `first`) into ctx [rows][512]. Few rows (decode) split the keys
// across blocks through `partial` / `partial_ml` scratch and fold them after.
static void mla_attention(const FlyweightV2RaonRuntime& rt, std::uint64_t queries, const MlaKeys& keys,
                          std::uint64_t ctx, int n, std::size_t first, std::uint64_t partial = 0,
                          std::uint64_t partial_ml = 0, std::size_t max_splits = 0) {
    const auto& g = rt.geometry;
    int rows = n * static_cast<int>(g.heads), heads = static_cast<int>(g.heads);
    int first_position = static_cast<int>(first), type = keys.type;
    float scale = 1.0f / std::sqrt(static_cast<float>(g.qk_nope + g.qk_rope));
    std::uint64_t latents = keys.latents, rope = keys.rope;
    const std::uint32_t shared = (32 * 65 + 32 * 65 + 32 * 33 + 32 * 512) * sizeof(float);
    const std::size_t keys_total = first + static_cast<std::size_t>(n);
    constexpr int split_keys = 256;
    const std::size_t splits = partial ? std::min(max_splits, (keys_total + split_keys - 1) / split_keys) : 1;
    // Many rows (prefill): the tensor-core kernel. FLYWEIGHT_RAON_MLA_MMA=0
    // keeps the f32 one.
    static const bool mma_enabled = [] {
        const char* value = std::getenv("FLYWEIGHT_RAON_MLA_MMA");
        return !value || value[0] != '0';
    }();
    if (!partial && mma_enabled && rt.int8_tensor_cores && type == 1) {
        // f16 cache: queries in registers, key tiles double-buffered.
        void* arguments[] = {&queries, &latents, &rope, &ctx, &rows, &heads, &first_position, &scale};
        const std::uint32_t shared = 2 * 16 * 584 * 2 + 4 * 32 * 17 * 4 + 32 * 24 * 2 + 3 * 32 * 4;
        launch("raon_mla_attention_mma_f16", static_cast<std::uint32_t>((rows + 31) / 32), 1, 256, arguments, shared);
        return;
    }
    if (!partial && mma_enabled && rt.int8_tensor_cores) {
        void* arguments[] = {&queries, &latents, &rope, &type, &ctx, &rows, &heads, &first_position, &scale};
        const std::uint32_t mma_shared = 2 * 32 * 584 * 2 + 32 * 33 * 4 + 32 * 40 * 2 + 3 * 32 * 4;
        launch("raon_mla_attention_mma", static_cast<std::uint32_t>((rows + 31) / 32), 1, 256, arguments, mma_shared);
        return;
    }
    if (splits <= 1) {
        std::uint64_t none = 0;
        int per_split = 1 << 30;
        void* arguments[] = {&queries, &latents, &rope, &type, &ctx, &none, &rows, &heads,
                             &first_position, &scale, &per_split};
        launch("raon_mla_attention", static_cast<std::uint32_t>((rows + 31) / 32), 1, 256, arguments, shared);
        return;
    }
    int per_split = static_cast<int>((keys_total + splits - 1) / splits + 31) / 32 * 32;
    const std::size_t used = (keys_total + per_split - 1) / per_split;
    {
        void* arguments[] = {&queries, &latents, &rope, &type, &partial, &partial_ml, &rows, &heads,
                             &first_position, &scale, &per_split};
        launch("raon_mla_attention", static_cast<std::uint32_t>((rows + 31) / 32),
               static_cast<std::uint32_t>(used), 256, arguments, shared);
    }
    int split_count = static_cast<int>(used);
    void* arguments[] = {&partial, &partial_ml, &ctx, &rows, &split_count};
    launch("raon_mla_combine", static_cast<std::uint32_t>(rows), 1, 128, arguments);
}

// The absorbed MLA attention of `rows` tokens on the device, from queries
// [rows][heads * (nope + rope)] at `q_dev` to [rows][heads * v_head] at
// `out_dev`: absorb q_nope through kv_b's key half -> [t][head][576] queries,
// attention -> [t][head][512] latent context, decompress through the value
// half. Worked in token groups sized to the expert scratch it borrows (the
// queries in prefill_gate, the context in prefill_up).
static void mla_core_rows(FlyweightV2RaonRuntime& rt, std::uint64_t kv_b, int kv_b_half,
                          const MlaKeys& keys, std::size_t first_position, std::size_t rows,
                          std::uint64_t q_dev, std::uint64_t out_dev) {
    const auto& g = rt.geometry;
    const std::size_t heads = g.heads, qk = g.qk_nope + g.qk_rope;
    const std::size_t width = g.kv_lora + g.qk_rope;   // 576
    const std::size_t stride = g.qk_nope + g.v_head_dim;
    const std::size_t element = kv_b_half ? 2 : 4;
    const std::size_t scratch = rt.prefill_chunk * g.experts_used * g.expert_size +
                                rt.prefill_chunk * g.shared_size;
    const std::size_t group = std::max<std::size_t>(1, std::min(rows, scratch / (heads * width)));
    for (std::size_t t0 = 0; t0 < rows; t0 += group) {
        int n = static_cast<int>(std::min(group, rows - t0));
        int nh = n * static_cast<int>(heads);
        const std::uint64_t q = q_dev + t0 * heads * qk * sizeof(float);
        std::uint64_t queries = rt.prefill_gate, ctx = rt.prefill_up;
        const std::uint64_t out = out_dev + t0 * heads * g.v_head_dim * sizeof(float);
        // q_nope [t][head][nope] through kv_b's key half [head][nope][latent].
        head_gemm(q, kv_b, queries, n, static_cast<int>(heads), static_cast<int>(g.qk_nope),
                  static_cast<int>(g.kv_lora), static_cast<long long>(heads * qk), static_cast<int>(qk),
                  static_cast<long long>(stride * g.kv_lora), static_cast<int>(g.kv_lora), 1,
                  static_cast<long long>(heads * width), static_cast<int>(width), kv_b_half);
        {
            std::uint64_t source = q;
            int tokens = n, h = static_cast<int>(heads), nope = static_cast<int>(g.qk_nope),
                rope = static_cast<int>(g.qk_rope), latent = static_cast<int>(g.kv_lora);
            void* arguments[] = {&source, &queries, &tokens, &h, &nope, &rope, &latent};
            launch("raon_mla_rope_queries",
                   static_cast<std::uint32_t>((static_cast<std::size_t>(nh) * g.qk_rope + 255) / 256), 1,
                   256, arguments);
        }
        mla_attention(rt, queries, keys, ctx, n, first_position + t0);
        // Latent context [t][head][latent] through the value half
        // [head][v][latent]: B(k = latent, n = v) at v * latent + k.
        head_gemm(ctx, kv_b + g.qk_nope * g.kv_lora * element, out, n, static_cast<int>(heads),
                  static_cast<int>(g.kv_lora), static_cast<int>(g.v_head_dim),
                  static_cast<long long>(heads * g.kv_lora), static_cast<int>(g.kv_lora),
                  static_cast<long long>(stride * g.kv_lora), 1, static_cast<int>(g.kv_lora),
                  static_cast<long long>(heads * g.v_head_dim), static_cast<int>(g.v_head_dim), kv_b_half);
    }
}

// The host-cache form: the cache rows go up into the prefill key scratch
// (f32; 16-bit rows are widened on the device), kv_b (decoded f32) streams
// through the slot ring.
static void mla_core_device(FlyweightV2RaonRuntime& rt, const raon::LayerWeights& layer,
                            const raon::LayerCache& cache, std::size_t first_position,
                            std::size_t rows, const float* query, float* attended) {
    const auto& g = rt.geometry;
    const std::size_t heads = g.heads, qk = g.qk_nope + g.qk_rope;
    const std::size_t width = g.kv_lora + g.qk_rope;   // 576
    const std::size_t positions = first_position + rows;
    const std::size_t stride = g.qk_nope + g.v_head_dim;
    upload(rt.prefill_in, query, rows * heads * qk);
    if (g.cache_type == 0) {
        upload(rt.prefill_latents, cache.latents.data(), positions * g.kv_lora);
        upload(rt.prefill_rope_keys, cache.rope_keys.data(), positions * g.qk_rope);
    } else if ((positions * width) * 2 <= rt.prefill_out_elements * sizeof(float)) {
        // 16-bit rows go up as they are (half the bytes) into prefill_out,
        // free until the attention output, and are widened on the device.
        const std::uint64_t staged_latents = rt.prefill_out;
        const std::uint64_t staged_rope = rt.prefill_out + positions * g.kv_lora * 2;
        upload_bytes(staged_latents, cache.latents16.data(), positions * g.kv_lora * 2);
        upload_bytes(staged_rope, cache.rope16.data(), positions * g.qk_rope * 2);
        const std::pair<std::uint64_t, std::uint64_t> widen[2] = {
            {staged_latents, rt.prefill_latents}, {staged_rope, rt.prefill_rope_keys}};
        const std::size_t counts[2] = {positions * g.kv_lora, positions * g.qk_rope};
        for (int i = 0; i < 2; ++i) {
            std::uint64_t source = widen[i].first, target = widen[i].second;
            long long count = static_cast<long long>(counts[i]);
            int bf16 = g.cache_type == 2 ? 1 : 0;
            void* arguments[] = {&source, &target, &count, &bf16};
            launch("raon_widen_half", static_cast<std::uint32_t>((counts[i] + 255) / 256), 1, 256, arguments);
        }
    } else {
        std::vector<float> wide_latents(positions * g.kv_lora), wide_rope(positions * g.qk_rope);
        raon::cache_widen(cache, g, 0, positions, wide_latents.data(), wide_rope.data());
        upload(rt.prefill_latents, wide_latents.data(), wide_latents.size());
        upload(rt.prefill_rope_keys, wide_rope.data(), wide_rope.size());
    }
    const MlaKeys keys{rt.prefill_latents, rt.prefill_rope_keys, 0};
    std::vector<StreamItem> items(1);
    items[0].segments.push_back({layer.mla.kv_b, heads * stride * g.kv_lora * sizeof(float)});
    items[0].run = [&](std::uint64_t kv_b) {
        mla_core_rows(rt, kv_b, 0, keys, first_position, rows, rt.prefill_in, rt.prefill_out);
    };
    execute(rt, items);
    download(attended, rt.prefill_out, rows * heads * g.v_head_dim);
}

static void mla_rows(FlyweightV2RaonRuntime& rt, std::size_t l, const raon::GpuLayerDispatch* gl,
                     RaonSlot& slot, std::size_t first_position, const float* x, std::size_t rows,
                     Workspace& ws, float* branch_out) {
    const raon::LayerWeights& layer = rt.layers[l];
    raon::LayerCache& cache = slot.caches[l];
    const auto& g = rt.geometry;
    const auto& w = layer.mla;
    raon::ProfileScope scoped(raon::profile().mla);
    const std::size_t gate_width = g.heads * g.v_head_dim;
    const std::size_t kv_width = g.kv_lora + g.qk_rope;
    const std::size_t qk = g.qk_nope + g.qk_rope;

    Rows q_low, gate, kv;
    const bool device = gl && gl->attn_gpu;
    if (device && gl->attn_packed) {
        const std::size_t width = g.q_lora + gate_width + kv_width;
        ws.a.resize(rows * width);
        upload(rt.prefill_in, x, rows * g.hidden);
        gemm(rt, gl->attn_type[0], gl->attn_weight[0], rt.prefill_in, rt.prefill_out,
             static_cast<int>(rows), static_cast<int>(g.hidden), static_cast<int>(width));
        download(ws.a.data(), rt.prefill_out, rows * width);
        q_low = {ws.a.data(), width};
        gate = {ws.a.data() + g.q_lora, width};
        kv = {ws.a.data() + g.q_lora + gate_width, width};
    } else {
        ws.a.resize(rows * (g.q_lora + gate_width + kv_width));
        float* q_out = ws.a.data();
        float* gate_out = q_out + rows * g.q_lora;
        float* kv_out = gate_out + rows * gate_width;
        linear(rt, w.q_a, device ? gl->attn_weight[0] : 0, device ? gl->attn_type[0] : 0,
               x, rows, g.hidden, g.q_lora, q_out);
        linear(rt, w.gate, device ? gl->attn_weight[3] : 0, device ? gl->attn_type[3] : 0,
               x, rows, g.hidden, gate_width, gate_out);
        linear(rt, w.kv_a_mqa, device ? gl->attn_weight[2] : 0, device ? gl->attn_type[2] : 0,
               x, rows, g.hidden, kv_width, kv_out);
        q_low = {q_out, g.q_lora};
        gate = {gate_out, gate_width};
        kv = {kv_out, kv_width};
    }

    if (rt.device_decode) {
        // Normalized rows up through prefill_out (free until the attention
        // output) and into the slot's device cache in its type.
        std::vector<float> rows_f32(rows * (g.kv_lora + g.qk_rope));
        float* latents = rows_f32.data();
        float* ropes = latents + rows * g.kv_lora;
        for (std::size_t t = 0; t < rows; ++t) {
            flyweight::v2::bailing::rms_norm(kv[t], w.kv_a_norm, g.kv_lora, g.latent_epsilon,
                                             latents + t * g.kv_lora);
            std::memcpy(ropes + t * g.qk_rope, kv[t] + g.kv_lora, g.qk_rope * sizeof(float));
        }
        upload(rt.prefill_out, rows_f32.data(), rows_f32.size());
        const std::uint64_t sources[2] = {rt.prefill_out, rt.prefill_out + rows * g.kv_lora * sizeof(float)};
        const std::uint64_t targets[2] = {slot.mla_latents[l], slot.mla_rope[l]};
        const std::size_t widths[2] = {g.kv_lora, g.qk_rope};
        for (int part = 0; part < 2; ++part) {
            std::uint64_t source = sources[part], target = targets[part];
            long long offset = static_cast<long long>(first_position * widths[part]);
            int count = static_cast<int>(rows * widths[part]), type = g.cache_type;
            void* arguments[] = {&source, &target, &offset, &count, &type};
            launch("raon_cache_store", static_cast<std::uint32_t>((count + 255) / 256), 1, 256, arguments);
        }
    } else {
        std::vector<float> latent(g.kv_lora);
        for (std::size_t t = 0; t < rows; ++t) {
            flyweight::v2::bailing::rms_norm(kv[t], w.kv_a_norm, g.kv_lora, g.latent_epsilon, latent.data());
            raon::cache_store(cache, g, first_position + t, latent.data(), kv[t] + g.kv_lora);
        }
    }
    cache.positions = first_position + rows;

    ws.b.resize(rows * g.q_lora);
    for (std::size_t t = 0; t < rows; ++t)
        flyweight::v2::bailing::rms_norm(q_low[t], w.q_a_norm, g.q_lora, g.latent_epsilon,
                                         ws.b.data() + t * g.q_lora);
    ws.c.resize(rows * g.heads * qk);
    linear(rt, w.q_b, device ? gl->attn_weight[1] : 0, device ? gl->attn_type[1] : 0,
           ws.b.data(), rows, g.q_lora, g.heads * qk, ws.c.data());

    ws.f.resize(rows * gate_width);
    float* attended = ws.f.data();
    const std::size_t work = rows * g.heads;
    std::optional<Phase> core(std::in_place, "mla: attention core");
    // raon_mla_attention is laid out for Raon-1's 512-wide latent and 64-wide
    // rope (eight threads x 64 dims per row, nine 64-wide score slices).
    if (rt.device_decode) {
        const MlaKeys keys{slot.mla_latents[l], slot.mla_rope[l], g.cache_type};
        upload(rt.prefill_in, ws.c.data(), rows * g.heads * qk);
        mla_core_rows(rt, rt.decode_layers[l].kv_b, 1, keys, first_position, rows, rt.prefill_in,
                      rt.prefill_out);
        download(attended, rt.prefill_out, rows * gate_width);
        for (std::size_t t = 0; t < rows; ++t) {
            float* out = attended + t * gate_width;
            const float* gate_row = gate[t];
            for (std::size_t i = 0; i < gate_width; ++i)
                out[i] *= 1.0f / (1.0f + std::exp(-gate_row[i]));
        }
    } else if (streaming(rt, rows) && first_position + rows <= rt.prefill_key_capacity &&
        g.kv_lora == 512 && g.qk_rope == 64 &&
        g.heads * (g.qk_nope + g.v_head_dim) * g.kv_lora * sizeof(float) <= rt.stream_slot_bytes) {
        mla_core_device(rt, layer, cache, first_position, rows, ws.c.data(), attended);
        for (std::size_t t = 0; t < rows; ++t) {
            float* out = attended + t * gate_width;
            const float* gate_row = gate[t];
            for (std::size_t i = 0; i < gate_width; ++i)
                out[i] *= 1.0f / (1.0f + std::exp(-gate_row[i]));
        }
    } else {
    // A 16-bit cache is widened once for the chunk.
    std::vector<float> wide_latents, wide_rope;
    const float* latents = cache.latents.data();
    const float* rope_keys = cache.rope_keys.data();
    if (g.cache_type != 0) {
        wide_latents.resize((first_position + rows) * g.kv_lora);
        wide_rope.resize((first_position + rows) * g.qk_rope);
        raon::cache_widen(cache, g, 0, first_position + rows, wide_latents.data(), wide_rope.data());
        latents = wide_latents.data();
        rope_keys = wide_rope.data();
    }
#ifdef _OPENMP
#pragma omp parallel num_threads(flyweight::v2::bailing::matvec_threads())
#endif
    {
        std::vector<float> projected(g.kv_lora), accumulated(g.kv_lora);
        std::vector<float> scores(first_position + rows);
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 4)
#endif
        for (std::int64_t item = 0; item < static_cast<std::int64_t>(work); ++item) {
            const std::size_t t = static_cast<std::size_t>(item) / g.heads;
            const std::size_t h = static_cast<std::size_t>(item) % g.heads;
            const float* query = ws.c.data() + t * g.heads * qk + h * qk;
            float* target = attended + t * gate_width + h * g.v_head_dim;
            mla_head(query, query + g.qk_nope, w.kv_b + h * (g.qk_nope + g.v_head_dim) * g.kv_lora,
                     latents, rope_keys, first_position + t + 1, g,
                     projected.data(), accumulated.data(), scores.data(), target);
            const float* gate_row = gate[t] + h * g.v_head_dim;
            for (std::size_t i = 0; i < g.v_head_dim; ++i)
                target[i] *= 1.0f / (1.0f + std::exp(-gate_row[i]));
        }
    }
    }

    core.reset();
    linear(rt, w.output, device ? gl->attn_weight[4] : 0, device ? gl->attn_type[4] : 0,
           attended, rows, gate_width, g.hidden, branch_out);
}

static void situ_rows(const raon::Geometry& g, const float* gate, const float* up,
                      std::size_t elements, float* output) {
    raon::situ_glu(gate, up, elements, g.situ_beta, g.situ_linear_beta, output);
}

static void dense_rows(FlyweightV2RaonRuntime& rt, const raon::LayerWeights& layer,
                       const float* x, std::size_t rows, Workspace& ws, float* branch_out) {
    const auto& g = rt.geometry;
    raon::ProfileScope dense(raon::profile().dense_ffn);
    ws.a.resize(rows * g.dense_size);
    ws.b.resize(rows * g.dense_size);
    const std::size_t l = static_cast<std::size_t>(&layer - rt.layers.data());
    const RaonDecodeLayer* d = rt.device_decode ? &rt.decode_layers[l] : nullptr;
    linear(rt, layer.ffn.gate, d ? d->dense_gate : 0, layer.ffn.gate.type, x, rows, g.hidden, g.dense_size, ws.a.data());
    linear(rt, layer.ffn.up, d ? d->dense_up : 0, layer.ffn.up.type, x, rows, g.hidden, g.dense_size, ws.b.data());
    situ_rows(g, ws.a.data(), ws.b.data(), rows * g.dense_size, ws.a.data());
    linear(rt, layer.ffn.down, d ? d->dense_down : 0, layer.ffn.down.type, ws.a.data(), rows, g.dense_size,
           g.hidden, branch_out);
}

// Device MoE for a chunk whose rows are already gathered expert by expert in
// prefill_in ([routes][hidden]) with the chunk's own rows after them: the
// routed experts write [routes][hidden] to prefill_out and the shared expert
// writes its [rows][hidden] after them. Experts run from the pinned stacked
// tensors when the layer is resident, else stream through the slot ring a few
// per item. Expert scratch: gate/up/act [routes][E], then the shared [rows][S].
static void moe_experts_device(FlyweightV2RaonRuntime& rt, const raon::LayerWeights& layer,
                               const raon::GpuLayerDispatch* gl,
                               const std::vector<std::size_t>& offsets, std::size_t rows) {
    const auto& g = rt.geometry;
    const auto& f = layer.ffn;
    const auto f32 = sizeof(float);
    const std::size_t E = g.expert_size, routes = rows * g.experts_used;
    const bool resident = gl && gl->pinned;
    const auto gate_type = resident ? gl->gate_type : f.gate_experts.type;
    const auto up_type = resident ? gl->up_type : f.up_experts.type;
    const auto down_type = resident ? gl->down_type : f.down_experts.type;
    const std::size_t gate_bytes = E * flyweight::v2::bailing::row_bytes(gate_type, g.hidden);
    const std::size_t up_bytes = E * flyweight::v2::bailing::row_bytes(up_type, g.hidden);
    const std::size_t down_bytes = g.hidden * flyweight::v2::bailing::row_bytes(down_type, E);
    const std::uint64_t shared_in = rt.prefill_in + routes * g.hidden * f32;

    // One expert over its rows; the three matrices at independent addresses.
    const auto expert_rows = [&](std::size_t e, std::uint64_t gate_w, std::uint64_t up_w,
                                 std::uint64_t down_w) {
        const int n = static_cast<int>(offsets[e + 1] - offsets[e]);
        const auto first = offsets[e];
        const std::uint64_t in = rt.prefill_in + first * g.hidden * f32;
        const std::uint64_t gate = rt.prefill_gate + first * E * f32;
        const std::uint64_t up = rt.prefill_up + first * E * f32;
        const std::uint64_t act = rt.prefill_act + first * E * f32;
        gemm(rt, gate_type, gate_w, in, gate, n, static_cast<int>(g.hidden), static_cast<int>(E));
        gemm(rt, up_type, up_w, in, up, n, static_cast<int>(g.hidden), static_cast<int>(E));
        if (rt.gpu_scratch.launch_situ_glu(gate, up, act, n * static_cast<int>(E), g.situ_beta,
                                           g.situ_linear_beta) != 0)
            throw std::runtime_error("raon prefill: situ launch failed");
        gemm(rt, down_type, down_w, act, rt.prefill_out + first * g.hidden * f32, n,
             static_cast<int>(E), static_cast<int>(g.hidden));
    };
    // The shared expert over the chunk's own rows.
    const auto shared_rows = [&](std::uint32_t gt, std::uint64_t gate_w, std::uint32_t ut,
                                 std::uint64_t up_w, std::uint32_t dt, std::uint64_t down_w) {
        const std::size_t S = g.shared_size;
        const std::uint64_t gate = rt.prefill_gate + routes * E * f32;
        const std::uint64_t up = rt.prefill_up + routes * E * f32;
        const std::uint64_t act = rt.prefill_act + routes * E * f32;
        const int n = static_cast<int>(rows);
        gemm(rt, gt, gate_w, shared_in, gate, n, static_cast<int>(g.hidden), static_cast<int>(S));
        gemm(rt, ut, up_w, shared_in, up, n, static_cast<int>(g.hidden), static_cast<int>(S));
        if (rt.gpu_scratch.launch_situ_glu(gate, up, act, static_cast<int>(rows * S), g.situ_beta,
                                           g.situ_linear_beta) != 0)
            throw std::runtime_error("raon prefill: situ launch failed");
        gemm(rt, dt, down_w, act, rt.prefill_out + routes * g.hidden * f32, n, static_cast<int>(S),
             static_cast<int>(g.hidden));
    };

    std::vector<std::size_t> active;
    for (std::size_t e = 0; e < g.experts; ++e)
        if (offsets[e + 1] > offsets[e]) active.push_back(e);
    const bool shared = g.shared_experts > 0 && f.shared_gate;

    if (resident) {
        for (auto e : active)
            expert_rows(e, gl->gate_exps + e * gate_bytes, gl->up_exps + e * up_bytes,
                        gl->down_exps + e * down_bytes);
        if (shared && gl->shared_gate) {
            shared_rows(gl->shared_gate_type, gl->shared_gate, gl->shared_up_type, gl->shared_up,
                        gl->shared_down_type, gl->shared_down);
            return;
        }
    }

    std::vector<StreamItem> items;
    if (!resident) {
        const std::size_t expert_bytes = gate_bytes + up_bytes + down_bytes;
        const std::size_t per_item = std::max<std::size_t>(1, rt.stream_slot_bytes / expert_bytes);
        const auto* gate_base = static_cast<const std::uint8_t*>(f.gate_experts.data);
        const auto* up_base = static_cast<const std::uint8_t*>(f.up_experts.data);
        const auto* down_base = static_cast<const std::uint8_t*>(f.down_experts.data);
        for (std::size_t i = 0; i < active.size(); i += per_item) {
            StreamItem item;
            std::vector<std::size_t> group(active.begin() + i,
                                           active.begin() + std::min(active.size(), i + per_item));
            for (auto e : group) {
                item.segments.push_back({gate_base + e * gate_bytes, gate_bytes});
                item.segments.push_back({up_base + e * up_bytes, up_bytes});
                item.segments.push_back({down_base + e * down_bytes, down_bytes});
            }
            item.run = [group, &expert_rows, gate_bytes, up_bytes, expert_bytes](std::uint64_t slot) {
                for (std::size_t j = 0; j < group.size(); ++j) {
                    const std::uint64_t base = slot + j * expert_bytes;
                    expert_rows(group[j], base, base + gate_bytes, base + gate_bytes + up_bytes);
                }
            };
            items.push_back(std::move(item));
        }
    }
    if (shared) {
        const std::size_t S = g.shared_size;
        const std::size_t sg = S * flyweight::v2::bailing::row_bytes(f.shared_gate.type, g.hidden);
        const std::size_t su = S * flyweight::v2::bailing::row_bytes(f.shared_up.type, g.hidden);
        const std::size_t sd = g.hidden * flyweight::v2::bailing::row_bytes(f.shared_down.type, S);
        StreamItem item;
        item.segments = {{f.shared_gate.data, sg}, {f.shared_up.data, su}, {f.shared_down.data, sd}};
        item.run = [&shared_rows, &f, sg, su](std::uint64_t slot) {
            shared_rows(f.shared_gate.type, slot, f.shared_up.type, slot + sg, f.shared_down.type,
                        slot + sg + su);
        };
        items.push_back(std::move(item));
    }
    execute(rt, items);
}

// The routed experts and the shared expert of a chunk through the block-table
// MMQ (qwen's *_q8_mmq_routed): routes grouped by expert in 32-route blocks,
// one launch per projection per tile instead of a handful per expert. A
// resident layer runs every block against its pinned stacks; a streamed layer
// runs, per ring item, the contiguous block range of that item's experts
// (active experts are grouped in ascending order, which is also the block
// order) against a pointer table aimed into the item's slot. Each tile's
// down outputs fold into the token rows with qwen_slot_accumulate_rows, after
// the shared expert has written them.
static bool routed_available(const FlyweightV2RaonRuntime& rt, const raon::LayerWeights& layer,
                             const raon::GpuLayerDispatch* gl) {
    const auto& g = rt.geometry;
    const bool resident = gl && gl->pinned;
    const auto& f = layer.ffn;
    const auto gate = resident ? gl->gate_type : f.gate_experts.type;
    const auto up = resident ? gl->up_type : f.up_experts.type;
    const auto down = resident ? gl->down_type : f.down_experts.type;
    static const bool enabled = [] {
        const char* value = std::getenv("FLYWEIGHT_RAON_ROUTED_MMQ");
        return !value || value[0] != '0';
    }();
    // A resident layer must have its shared expert resident too: it writes
    // the token rows before the routed folds add to them.
    if (resident && g.shared_experts > 0 && f.shared_gate && !gl->shared_gate) return false;
    return enabled && rt.int8_tensor_cores && rt.prefill_route_tables &&
           !qwen_routed_mmq_kernel(gate, static_cast<int>(g.hidden)).empty() &&
           !qwen_routed_mmq_kernel(up, static_cast<int>(g.hidden)).empty() &&
           !qwen_routed_mmq_kernel(down, static_cast<int>(g.expert_size)).empty();
}

static void quantize_rows(std::uint64_t input, std::uint64_t q8, std::uint64_t scales, int width, int rows) {
    int scale_stride = (width / 32) * 2;
    void* arguments[] = {&input, &q8, &scales, &width, &scale_stride};
    launch("quantize_q8_blocks_rows", static_cast<std::uint32_t>((width + 255) / 256),
           static_cast<std::uint32_t>(rows), 256, arguments);
}

static void routed_experts(FlyweightV2RaonRuntime& rt, const raon::LayerWeights& layer,
                           const raon::GpuLayerDispatch* gl, const std::vector<std::int32_t>& chosen,
                           const std::vector<float>& weights, const float* x, std::size_t rows,
                           float* branch_out) {
    const auto& g = rt.geometry;
    const auto& f = layer.ffn;
    const auto f32 = sizeof(float);
    const std::size_t used = g.experts_used, routes = rows * used, E = g.expert_size, H = g.hidden;
    constexpr int block_size = flyweight::v2::moe::kMmqBlockSize;
    const bool resident = gl && gl->pinned;
    const auto gate_type = resident ? gl->gate_type : f.gate_experts.type;
    const auto up_type = resident ? gl->up_type : f.up_experts.type;
    const auto down_type = resident ? gl->down_type : f.down_experts.type;
    const std::size_t gate_bytes = E * flyweight::v2::bailing::row_bytes(gate_type, H);
    const std::size_t up_bytes = E * flyweight::v2::bailing::row_bytes(up_type, H);
    const std::size_t down_bytes = H * flyweight::v2::bailing::row_bytes(down_type, E);

    // Experts the decode cache already holds need no streaming: renumber the
    // layer's experts so the cached ones come first. Their route blocks then
    // form one leading range, run against the cache slots, and only the rest
    // stream. `original[id]` maps a renumbered id back.
    const std::size_t l = static_cast<std::size_t>(&layer - rt.layers.data());
    const auto& cache = rt.expert_cache;
    std::vector<std::int32_t> renumber(g.experts), original(g.experts);
    std::size_t cached = 0;
    {
        std::vector<std::uint8_t> in_cache(g.experts, 0);
        if (!resident && cache.slots)
            for (std::size_t e = 0; e < g.experts; ++e) {
                const auto slot = cache.slot_of[l * g.experts + e];
                in_cache[e] = slot >= 0 && cache.ready[slot];
            }
        std::size_t next = 0;
        for (int pass = 0; pass < 2; ++pass)
            for (std::size_t e = 0; e < g.experts; ++e)
                if (in_cache[e] == (pass == 0 ? 1 : 0)) {
                    renumber[e] = static_cast<std::int32_t>(next);
                    original[next] = static_cast<std::int32_t>(e);
                    ++next;
                }
        for (auto flag : in_cache) cached += flag;
    }
    std::vector<std::int32_t> chosen_ids(routes);
    for (std::size_t r = 0; r < routes; ++r) chosen_ids[r] = renumber[static_cast<std::size_t>(chosen[r])];

    flyweight::v2::moe::AlignedRoutes aligned;
    flyweight::v2::moe::align_blocks(chosen_ids.data(), weights.data(), static_cast<int>(routes),
                                     static_cast<int>(g.experts), block_size, aligned);
    const std::size_t blocks = aligned.block_experts.size();
    std::vector<std::int32_t> slot_of_route(routes, -1);
    for (std::size_t slot = 0; slot < aligned.sorted_routes.size(); ++slot)
        if (aligned.sorted_routes[slot] >= 0)
            slot_of_route[static_cast<std::size_t>(aligned.sorted_routes[slot])] = static_cast<std::int32_t>(slot);
    std::vector<std::size_t> first_block(g.experts + 1, blocks);
    for (std::size_t b = blocks; b-- > 0;) first_block[aligned.block_experts[b]] = b;
    for (std::size_t e = g.experts; e-- > 0;)
        if (first_block[e] == blocks) first_block[e] = first_block[e + 1];

    // Device tables.
    const std::uint64_t t_blocks = rt.prefill_route_tables;
    const std::uint64_t t_sorted = t_blocks + 4 * (routes / block_size + g.experts + 1);
    const std::uint64_t t_slot = t_sorted + 4 * (routes + block_size * g.experts);
    const std::uint64_t t_weights = t_slot + 4 * routes;
    const std::uint64_t t_ptrs = (t_weights + 4 * routes + 7) & ~std::uint64_t(7);
    if (blocks) upload_bytes(t_blocks, aligned.block_experts.data(), blocks * 4);
    if (!aligned.sorted_routes.empty())
        upload_bytes(t_sorted, aligned.sorted_routes.data(), aligned.sorted_routes.size() * 4);
    upload_bytes(t_slot, slot_of_route.data(), routes * 4);
    upload_bytes(t_weights, weights.data(), routes * f32);

    // Activations: x [rows][H] in prefill_in; q8 layout: gemm scratch | x | act.
    upload(rt.prefill_in, x, rows * H);
    const std::uint64_t x_q8 = rt.prefill_q8 + rows * H;
    const std::uint64_t x_scales = rt.prefill_scales + rows * (H / 32) * f32;
    const std::uint64_t act_q8 = rt.prefill_q8 + 2 * rows * H;
    const std::uint64_t act_scales = rt.prefill_scales + 2 * rows * (H / 32) * f32;
    const std::uint64_t destination = rt.prefill_act;
    const bool shared = g.shared_experts > 0 && f.shared_gate;
    if (!shared && flyweight_gpu_memset(destination, 0, rows * H * f32, 0) != 0)
        throw std::runtime_error("raon prefill: memset failed");

    // Tile so every per-slot buffer fits its scratch: [slots][E] gate+up in
    // prefill_gate, the activation in prefill_up, [slots][H] down in prefill_out.
    const std::size_t q8_left = rt.prefill_in_elements - 2 * rows * H;
    const std::size_t tile_slots = std::min({rt.prefill_expert_elements / (2 * E),
                                             rt.prefill_out_elements / H, q8_left / E});
    const std::size_t tile_blocks = std::max<std::size_t>(1, tile_slots / block_size);
    const std::string gate_kernel = qwen_routed_mmq_kernel(gate_type, static_cast<int>(H));
    const std::string up_kernel = qwen_routed_mmq_kernel(up_type, static_cast<int>(H));
    const std::string down_kernel = qwen_routed_mmq_kernel(down_type, static_cast<int>(E));
    const auto is_min = [](std::uint32_t type) { return type == 10 || type == 12 || type == 13; };

    const auto routed_launch = [&](const std::string& name, std::uint32_t type, std::uint64_t table,
                                   std::uint64_t tile_experts, std::uint64_t tile_sorted,
                                   std::uint64_t input, std::uint64_t scales, std::uint64_t output,
                                   int in, int out, int top_k, int count) {
        int scale_stride = (in / 32) * 2;
        void* arguments[] = {&table, &tile_experts, &tile_sorted, &input, &scales, &output,
                             &in, &out, &top_k, &scale_stride, &count};
        const std::uint32_t row_tile = is_min(type) ? 64u : 128u;
        launch(name.c_str(), (static_cast<std::uint32_t>(out) + row_tile - 1) / row_tile,
               static_cast<std::uint32_t>(count), is_min(type) ? 512u : 256u, arguments);
    };
    const auto run_blocks = [&](std::size_t b0, std::size_t b1, std::uint64_t table) {
        for (std::size_t begin = b0; begin < b1; begin += tile_blocks) {
            const int count = static_cast<int>(std::min(tile_blocks, b1 - begin));
            const std::size_t slots = static_cast<std::size_t>(count) * block_size;
            const std::uint64_t tile_experts = t_blocks + begin * 4;
            const std::uint64_t tile_sorted = t_sorted + begin * block_size * 4;
            const std::uint64_t gate = rt.prefill_gate, up = rt.prefill_gate + slots * E * f32;
            routed_launch(gate_kernel, gate_type, table, tile_experts, tile_sorted, x_q8, x_scales,
                          gate, static_cast<int>(H), static_cast<int>(E), static_cast<int>(used), count);
            routed_launch(up_kernel, up_type, table + 8 * g.experts, tile_experts, tile_sorted, x_q8,
                          x_scales, up, static_cast<int>(H), static_cast<int>(E), static_cast<int>(used), count);
            if (rt.gpu_scratch.launch_situ_glu(gate, up, rt.prefill_up, static_cast<int>(slots * E),
                                               g.situ_beta, g.situ_linear_beta) != 0)
                throw std::runtime_error("raon prefill: situ launch failed");
            quantize_rows(rt.prefill_up, act_q8, act_scales, static_cast<int>(E), static_cast<int>(slots));
            routed_launch(down_kernel, down_type, table + 16 * g.experts, tile_experts, tile_sorted,
                          act_q8, act_scales, rt.prefill_out, static_cast<int>(E), static_cast<int>(H), 0,
                          count);
            std::uint64_t source = rt.prefill_out, slot_table = t_slot, weight = t_weights,
                          target = destination;
            int width = static_cast<int>(H), tokens = static_cast<int>(rows), top_k = static_cast<int>(used),
                slot_begin = static_cast<int>(begin * block_size), slot_count = static_cast<int>(slots);
            void* arguments[] = {&source, &slot_table, &weight, &target, &width, &tokens, &top_k,
                                 &slot_begin, &slot_count};
            launch("qwen_slot_accumulate_rows", static_cast<std::uint32_t>((H + 255) / 256),
                   static_cast<std::uint32_t>(rows), 256, arguments);
        }
    };
    const auto shared_rows = [&](std::uint32_t gt, std::uint64_t gate_w, std::uint32_t ut,
                                 std::uint64_t up_w, std::uint32_t dt, std::uint64_t down_w) {
        const std::size_t S = g.shared_size;
        const int n = static_cast<int>(rows);
        gemm(rt, gt, gate_w, rt.prefill_in, rt.prefill_gate, n, static_cast<int>(H), static_cast<int>(S));
        gemm(rt, ut, up_w, rt.prefill_in, rt.prefill_up, n, static_cast<int>(H), static_cast<int>(S));
        if (rt.gpu_scratch.launch_situ_glu(rt.prefill_gate, rt.prefill_up, rt.prefill_gate,
                                           static_cast<int>(rows * S), g.situ_beta, g.situ_linear_beta) != 0)
            throw std::runtime_error("raon prefill: situ launch failed");
        gemm(rt, dt, down_w, rt.prefill_gate, destination, n, static_cast<int>(S), static_cast<int>(H));
    };
    // Pointer tables, [gate | up | down] x experts each. They are built for
    // every item up front and uploaded once: an upload per item went through
    // the staging buffer's sync and stalled the stream pipeline.
    const std::size_t table_entries = 3 * g.experts;
    std::vector<std::uint64_t> tables;
    const auto add_table = [&](const std::vector<std::size_t>& experts, const std::uint64_t* gate_at,
                               const std::uint64_t* up_at, const std::uint64_t* down_at) {
        const std::size_t index = tables.size() / table_entries;
        tables.resize(tables.size() + table_entries, 0);
        std::uint64_t* table = tables.data() + index * table_entries;
        for (std::size_t j = 0; j < experts.size(); ++j) {
            table[experts[j]] = gate_at[j];
            table[g.experts + experts[j]] = up_at[j];
            table[2 * g.experts + experts[j]] = down_at[j];
        }
        return t_ptrs + index * table_entries * 8;
    };

    if (resident) {
        if (shared && gl->shared_gate)
            shared_rows(gl->shared_gate_type, gl->shared_gate, gl->shared_up_type, gl->shared_up,
                        gl->shared_down_type, gl->shared_down);
        quantize_rows(rt.prefill_in, x_q8, x_scales, static_cast<int>(H), static_cast<int>(rows));
        std::vector<std::size_t> all(g.experts);
        std::vector<std::uint64_t> ga(g.experts), ua(g.experts), da(g.experts);
        for (std::size_t e = 0; e < g.experts; ++e) {
            all[e] = e;
            ga[e] = gl->gate_exps + e * gate_bytes;
            ua[e] = gl->up_exps + e * up_bytes;
            da[e] = gl->down_exps + e * down_bytes;
        }
        const std::uint64_t table = add_table(all, ga.data(), ua.data(), da.data());
        upload_bytes(t_ptrs, tables.data(), tables.size() * 8);
        run_blocks(0, blocks, table);
    }

    std::vector<StreamItem> items;
    if (shared && !(resident && gl->shared_gate)) {
        const std::size_t S = g.shared_size;
        const std::size_t sg = S * flyweight::v2::bailing::row_bytes(f.shared_gate.type, H);
        const std::size_t su = S * flyweight::v2::bailing::row_bytes(f.shared_up.type, H);
        const std::size_t sd = H * flyweight::v2::bailing::row_bytes(f.shared_down.type, S);
        StreamItem item;
        item.segments = {{f.shared_gate.data, sg}, {f.shared_up.data, su}, {f.shared_down.data, sd}};
        item.run = [&shared_rows, &f, sg, su](std::uint64_t slot) {
            shared_rows(f.shared_gate.type, slot, f.shared_up.type, slot + sg, f.shared_down.type, slot + sg + su);
        };
        items.push_back(std::move(item));
    }
    if (!resident) {
        StreamItem quantize;   // after the shared expert's GEMMs reused the q8 scratch
        quantize.run = [&](std::uint64_t) {
            quantize_rows(rt.prefill_in, x_q8, x_scales, static_cast<int>(H), static_cast<int>(rows));
        };
        items.push_back(std::move(quantize));
        const std::size_t expert_bytes = gate_bytes + up_bytes + down_bytes;
        const std::size_t per_item = std::max<std::size_t>(1, rt.stream_slot_bytes / expert_bytes);
        const auto* gate_base = static_cast<const std::uint8_t*>(f.gate_experts.data);
        const auto* up_base = static_cast<const std::uint8_t*>(f.up_experts.data);
        const auto* down_base = static_cast<const std::uint8_t*>(f.down_experts.data);
        // The cached experts first, straight from their cache slots.
        if (cached && first_block[cached] > first_block[0]) {
            std::vector<std::size_t> ids;
            std::vector<std::uint64_t> ga, ua, da;
            for (std::size_t id = 0; id < cached; ++id) {
                const auto slot = cache.slot_of[l * g.experts + static_cast<std::size_t>(original[id])];
                const std::uint64_t base = cache.pool + static_cast<std::uint64_t>(slot) * cache.slot_bytes;
                ids.push_back(id);
                ga.push_back(base);
                ua.push_back(base + gate_bytes);
                da.push_back(base + gate_bytes + up_bytes);
            }
            const std::uint64_t table = add_table(ids, ga.data(), ua.data(), da.data());
            const std::size_t b0 = first_block[0], b1 = first_block[cached];
            StreamItem item;
            item.run = [b0, b1, table, &run_blocks](std::uint64_t) { run_blocks(b0, b1, table); };
            items.push_back(std::move(item));
        }
        std::vector<std::size_t> active;
        for (std::size_t id = cached; id < g.experts; ++id)
            if (first_block[id + 1] > first_block[id]) active.push_back(id);
        for (std::size_t i = 0; i < active.size(); i += per_item) {
            StreamItem item;
            std::vector<std::size_t> group(active.begin() + i,
                                           active.begin() + std::min(active.size(), i + per_item));
            // Stacked by role so the slot holds gate | up | down runs.
            for (auto id : group) {
                const auto e = static_cast<std::size_t>(original[id]);
                item.segments.push_back({gate_base + e * gate_bytes, gate_bytes});
            }
            for (auto id : group) {
                const auto e = static_cast<std::size_t>(original[id]);
                item.segments.push_back({up_base + e * up_bytes, up_bytes});
            }
            for (auto id : group) {
                const auto e = static_cast<std::size_t>(original[id]);
                item.segments.push_back({down_base + e * down_bytes, down_bytes});
            }
            const std::size_t b0 = first_block[group.front()], b1 = first_block[group.back() + 1];
            // The item's slot is known from its position in the ring, so its
            // table can point there before anything runs.
            const std::size_t item_index = items.size();
            const std::uint64_t slot = rt.stream_device +
                ((rt.stream_sequence + item_index) % rt.stream_slots) * rt.stream_slot_bytes;
            const std::size_t n = group.size();
            std::vector<std::uint64_t> ga(n), ua(n), da(n);
            for (std::size_t j = 0; j < n; ++j) {
                ga[j] = slot + j * gate_bytes;
                ua[j] = slot + n * gate_bytes + j * up_bytes;
                da[j] = slot + n * (gate_bytes + up_bytes) + j * down_bytes;
            }
            const std::uint64_t table = add_table(group, ga.data(), ua.data(), da.data());
            item.run = [b0, b1, table, slot, &run_blocks](std::uint64_t given) {
                if (given != slot) throw std::runtime_error("raon prefill: stream slot out of order");
                run_blocks(b0, b1, table);
            };
            items.push_back(std::move(item));
        }
        upload_bytes(t_ptrs, tables.data(), tables.size() * 8);
    }
    execute(rt, items);
    download(branch_out, destination, rows * H);
}

static void moe_rows(FlyweightV2RaonRuntime& rt, const raon::LayerWeights& layer,
                     const raon::GpuLayerDispatch* gl, const float* x, std::size_t rows,
                     Workspace& ws, float* branch_out) {
    const auto& g = rt.geometry;
    const auto& f = layer.ffn;
    raon::ProfileScope moe(raon::profile().moe);
    const std::size_t used = g.experts_used, routes = rows * used;
    const std::size_t E = g.expert_size;

    // Route every token, then lay the routes out expert by expert.
    std::vector<std::int32_t> chosen(rows * used);
    std::vector<float> weights(rows * used);
    {
        Phase phase("moe: route");
        raon::ProfileScope route(raon::profile().moe_route);
        std::vector<float> logits(rows * g.experts);
        flyweight::v2::bailing::matmul(f.router, x, rows, g.hidden, g.experts, logits.data());
#ifdef _OPENMP
#pragma omp parallel num_threads(flyweight::v2::bailing::matvec_threads()) if (rows >= 16)
#endif
        {
            std::vector<std::int32_t> order(g.experts);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (std::int64_t index = 0; index < static_cast<std::int64_t>(rows); ++index) {
                const auto t = static_cast<std::size_t>(index);
                raon::route_token(logits.data() + t * g.experts, f.router_bias, g, order.data(),
                                  weights.data() + t * used);
                std::copy(order.begin(), order.begin() + used, chosen.begin() + t * used);
            }
        }
    }
    std::vector<std::size_t> offsets(g.experts + 1, 0);
    for (auto e : chosen) offsets[static_cast<std::size_t>(e) + 1]++;
    for (std::size_t e = 0; e < g.experts; ++e) offsets[e + 1] += offsets[e];
    std::vector<std::size_t> sorted_row(routes);   // route (t*used+k) -> sorted row
    {
        std::vector<std::size_t> fill(offsets.begin(), offsets.end() - 1);
        for (std::size_t r = 0; r < routes; ++r)
            sorted_row[r] = fill[static_cast<std::size_t>(chosen[r])]++;
    }
    const auto f32 = sizeof(float);
    if (((gl && gl->pinned) || streaming(rt, rows)) && routed_available(rt, layer, gl)) {
        raon::ProfileScope experts(raon::profile().moe_experts);
        Phase phase(gl && gl->pinned ? "moe: resident experts" : "moe: streamed experts");
        routed_experts(rt, layer, gl, chosen, weights, x, rows, branch_out);
        return;
    }
    if ((gl && gl->pinned) || streaming(rt, rows)) {
        raon::ProfileScope experts(raon::profile().moe_experts);
        Phase phase(gl && gl->pinned ? "moe: resident experts" : "moe: streamed experts");
        // Route tables: the token of each sorted row, the sorted row and the
        // weight of each (token, slot).
        std::vector<std::int32_t> tables(3 * routes);
        std::int32_t* token_of_row = tables.data();
        std::int32_t* row_of_route = token_of_row + routes;
        for (std::size_t r = 0; r < routes; ++r) {
            token_of_row[sorted_row[r]] = static_cast<std::int32_t>(r / used);
            row_of_route[r] = static_cast<std::int32_t>(sorted_row[r]);
        }
        std::memcpy(row_of_route + routes, weights.data(), routes * f32);
        const std::uint64_t x_dev = rt.prefill_in + routes * g.hidden * f32;
        upload(x_dev, x, rows * g.hidden);
        upload_bytes(rt.prefill_routes, tables.data(), tables.size() * f32);
        {
            std::uint64_t source = x_dev, token = rt.prefill_routes, target = rt.prefill_in;
            int count = static_cast<int>(routes), width = static_cast<int>(g.hidden);
            void* arguments[] = {&source, &token, &target, &count, &width};
            launch("raon_gather_rows", static_cast<std::uint32_t>((routes * g.hidden + 255) / 256), 1,
                   256, arguments);
        }
        moe_experts_device(rt, layer, gl, offsets, rows);
        {
            const bool shared = g.shared_experts > 0 && f.shared_gate;
            std::uint64_t expert_out = rt.prefill_out;
            std::uint64_t route_row = rt.prefill_routes + routes * f32;
            std::uint64_t weight = rt.prefill_routes + 2 * routes * f32;
            std::uint64_t shared_out = shared ? rt.prefill_out + routes * g.hidden * f32 : 0;
            std::uint64_t out = rt.prefill_act;
            int tokens = static_cast<int>(rows), slots = static_cast<int>(used),
                width = static_cast<int>(g.hidden);
            void* arguments[] = {&expert_out, &route_row, &weight, &shared_out, &out, &tokens, &slots,
                                 &width};
            launch("raon_combine_routes", static_cast<std::uint32_t>((rows * g.hidden + 255) / 256), 1,
                   256, arguments);
            download(branch_out, rt.prefill_act, rows * g.hidden);
        }
        return;
    }

    // CPU: gather, experts, shared expert, combine.
    ws.a.resize(routes * g.hidden);
    float* gathered = ws.a.data();
    for (std::size_t r = 0; r < routes; ++r)
        std::memcpy(gathered + sorted_row[r] * g.hidden, x + (r / used) * g.hidden,
                    g.hidden * f32);
    ws.d.resize(routes * g.hidden);
    ws.e.assign(rows * g.hidden, 0.0f);
    {
        raon::ProfileScope experts(raon::profile().moe_experts);
        Phase phase("moe: cpu experts");
        const auto narrow = E * flyweight::v2::bailing::row_bytes(f.gate_experts.type, g.hidden);
        const auto wide = g.hidden * flyweight::v2::bailing::row_bytes(f.down_experts.type, E);
        const auto* gate_base = static_cast<const std::uint8_t*>(f.gate_experts.data);
        const auto* up_base = static_cast<const std::uint8_t*>(f.up_experts.data);
        const auto* down_base = static_cast<const std::uint8_t*>(f.down_experts.data);
        ws.b.resize(routes * E);
        ws.c.resize(routes * E);
        for (std::size_t e = 0; e < g.experts; ++e) {
            const auto n = offsets[e + 1] - offsets[e];
            if (!n) continue;
            const auto first = offsets[e];
            flyweight::v2::bailing::matmul({gate_base + e * narrow, f.gate_experts.type},
                                           gathered + first * g.hidden, n, g.hidden, E,
                                           ws.b.data() + first * E);
            flyweight::v2::bailing::matmul({up_base + e * narrow, f.up_experts.type},
                                           gathered + first * g.hidden, n, g.hidden, E,
                                           ws.c.data() + first * E);
        }
        situ_rows(g, ws.b.data(), ws.c.data(), routes * E, ws.b.data());
        for (std::size_t e = 0; e < g.experts; ++e) {
            const auto n = offsets[e + 1] - offsets[e];
            if (!n) continue;
            const auto first = offsets[e];
            flyweight::v2::bailing::matmul({down_base + e * wide, f.down_experts.type},
                                           ws.b.data() + first * E, n, E, g.hidden,
                                           ws.d.data() + first * g.hidden);
        }
    }
    if (g.shared_experts > 0 && f.shared_gate) {
        raon::ProfileScope shared(raon::profile().moe_shared);
        const std::size_t S = g.shared_size;
        ws.b.resize(rows * S);
        ws.c.resize(rows * S);
        flyweight::v2::bailing::matmul(f.shared_gate, x, rows, g.hidden, S, ws.b.data());
        flyweight::v2::bailing::matmul(f.shared_up, x, rows, g.hidden, S, ws.c.data());
        situ_rows(g, ws.b.data(), ws.c.data(), rows * S, ws.b.data());
        flyweight::v2::bailing::matmul(f.shared_down, ws.b.data(), rows, S, g.hidden, ws.e.data());
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(flyweight::v2::bailing::matvec_threads())
#endif
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(rows); ++index) {
        const auto t = static_cast<std::size_t>(index);
        float* out = branch_out + t * g.hidden;
        const float* shared = ws.e.data() + t * g.hidden;
        for (std::size_t j = 0; j < g.hidden; ++j) out[j] = shared[j];
        for (std::size_t k = 0; k < used; ++k) {
            const float weight = weights[t * used + k];
            const float* y = ws.d.data() + sorted_row[t * used + k] * g.hidden;
            for (std::size_t j = 0; j < g.hidden; ++j) out[j] += weight * y[j];
        }
    }
}

// Runs `count` tokens of `slot` through the model, chunk by chunk, and leaves
// the last token's final hidden state (the stream mean) in `final_hidden`.
static void run(FlyweightV2RaonRuntime& rt, RaonSlot& slot, const std::uint32_t* tokens,
                std::uint32_t count, float* final_hidden) {
    const auto started = std::chrono::steady_clock::now();
    trace_rows.rows.clear();
    transfer_runtime = &rt;
    const auto& g = rt.geometry;
    const std::size_t hc = g.mhc_streams, width = hc * g.hidden;
    const std::size_t mix = 2 * hc + (hc - 1) * (hc - 1);
    const auto embedding_stride = flyweight::v2::bailing::row_bytes(rt.embedding_matrix.type, g.hidden);
    const auto* embedding = static_cast<const std::uint8_t*>(rt.embedding_matrix.data);

    Workspace ws;
    std::uint32_t done = 0;
    while (done < count) {
        const std::size_t rows = std::min<std::size_t>(rt.prefill_chunk, count - done);
        const auto chunk_started = std::chrono::steady_clock::now();
        const double mla_before = [] {
            for (const auto& row : trace_rows.rows)
                if (std::strcmp(row.first, "mla: total") == 0) return row.second;
            return 0.0;
        }();
        const std::size_t chunk_position = slot.position;
        if (raon::profiling()) raon::profile().tokens += rows;
        ws.streams.resize(rows * width);
        ws.mixes.resize(rows * mix);
        ws.branch.resize(rows * g.hidden);
        ws.normalized.resize(rows * g.hidden);
        ws.branch_out.resize(rows * g.hidden);
        ws.post.resize(rows * hc);
        ws.comb.resize(rows * hc * hc);
        for (std::size_t t = 0; t < rows; ++t) {
            float* streams = ws.streams.data() + t * width;
            flyweight::v2::bailing::row_decode(
                embedding + static_cast<std::size_t>(tokens[done + t]) * embedding_stride,
                rt.embedding_matrix.type, g.hidden, streams);
            for (std::size_t s = 1; s < hc; ++s)
                std::memcpy(streams + s * g.hidden, streams, g.hidden * sizeof(float));
        }

        for (std::size_t l = 0; l < rt.layers.size(); ++l) {
            const auto& layer = rt.layers[l];
            const raon::GpuLayerDispatch* gl =
                rt.uses_gpu && l < rt.gpu_layers.size() ? &rt.gpu_layers[l] : nullptr;
            auto& cache = slot.caches[l];

            {
                Phase phase("mhc+norm");
                raon::ProfileScope mixing(raon::profile().mhc);
                mhc_pre_rows(layer.mhc_attn, g, ws.streams.data(), rows, ws.mixes.data(),
                             ws.branch.data(), ws.post.data(), ws.comb.data());
                rms_rows(ws.branch.data(), layer.attn_norm, rows, g.hidden, g.epsilon,
                         ws.normalized.data());
            }
            std::optional<Phase> attention(std::in_place, layer.is_rda ? "rda: total" : "mla: total");
            if (layer.is_rda && rda_device_ready(rt, rows))
                rda_rows_device(rt, l, gl, slot, ws.normalized.data(), rows, ws.branch_out.data());
            else if (layer.is_rda)
                rda_rows(rt, layer, gl, cache, ws.normalized.data(), rows, ws, ws.branch_out.data());
            else
                mla_rows(rt, l, gl, slot, slot.position, ws.normalized.data(), rows, ws,
                         ws.branch_out.data());
            attention.reset();
            {
                Phase phase("mhc+norm");
                raon::ProfileScope mixing(raon::profile().mhc);
                mhc_post_rows(g, ws.streams.data(), ws.branch_out.data(), ws.post.data(),
                              ws.comb.data(), rows);
                mhc_pre_rows(layer.mhc_ffn, g, ws.streams.data(), rows, ws.mixes.data(),
                             ws.branch.data(), ws.post.data(), ws.comb.data());
                rms_rows(ws.branch.data(), layer.ffn_norm, rows, g.hidden, g.epsilon,
                         ws.normalized.data());
            }
            std::optional<Phase> ffn(std::in_place, layer.is_moe ? "moe: total" : "dense ffn");
            if (layer.is_moe)
                moe_rows(rt, layer, gl, ws.normalized.data(), rows, ws, ws.branch_out.data());
            else
                dense_rows(rt, layer, ws.normalized.data(), rows, ws, ws.branch_out.data());
            ffn.reset();
            {
                Phase phase("mhc+norm");
                raon::ProfileScope mixing(raon::profile().mhc);
                mhc_post_rows(g, ws.streams.data(), ws.branch_out.data(), ws.post.data(),
                              ws.comb.data(), rows);
            }
        }

        slot.position += rows;
        done += static_cast<std::uint32_t>(rows);
        const float* last = ws.streams.data() + (rows - 1) * width;
        const float inverse = 1.0f / static_cast<float>(hc);
        for (std::size_t j = 0; j < g.hidden; ++j) {
            float total = 0.0f;
            for (std::size_t s = 0; s < hc; ++s) total += last[s * g.hidden + j];
            final_hidden[j] = total * inverse;
        }
        std::memcpy(slot.streams.data(), last, width * sizeof(float));
        if (Trace::on()) {
            double mla_after = 0.0;
            for (const auto& row : trace_rows.rows)
                if (std::strcmp(row.first, "mla: total") == 0) mla_after = row.second;
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - chunk_started).count();
            std::fprintf(stderr, "[flyweight] raon prefill chunk at %zu: %zu tokens in %.3f s (%.0f tok/s), mla %.3f s\n",
                         chunk_position, rows, seconds, rows / seconds, mla_after - mla_before);
        }
        if (rt.progress_callback && rt.progress_callback(rt.progress_user, done, count) != 0)
            throw std::runtime_error("raon eval aborted by progress callback");
    }
    if (Trace::on()) {
        const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::fprintf(stderr, "[flyweight] raon prefill %u tokens in %.3f s (%.1f tok/s), chunk %u\n",
                     count, total, count / total, rt.prefill_chunk);
        for (const auto& row : trace_rows.rows)
            std::fprintf(stderr, "    %-28s %8.3f s\n", row.first, row.second);
    }
}

// ---------------------------------------------------------------------------
// Device decode: one token
// ---------------------------------------------------------------------------
//
// decoder_step's shape -- mHC mixing and the norms on the host, which are a
// few microseconds -- with each layer's attention as one upload of the branch
// norm and one download of its output, everything between on the device
// against the slot's device state (see raon_device_decode_setup). A MoE layer
// not resident on the device queues its shared expert on the GPU and runs the
// routed experts on the CPU meanwhile.

static void matvec_device(std::uint32_t type, std::uint64_t matrix, std::uint64_t input,
                          std::uint64_t output, std::size_t in, std::size_t out) {
    if (qwen_gpu_matvec_by_type(type, matrix, input, output, static_cast<int>(in), static_cast<int>(out), 0) != 0)
        throw std::runtime_error("raon decode: matvec failed for type " + std::to_string(type));
}

// Stream-ordered: the launches after it see the row, and the host buffer is
// staged before the call returns.
static void put_row(std::uint64_t device, const float* host, std::size_t elements) {
    if (flyweight_gpu_upload(device, host, elements * sizeof(float), 0) != 0)
        throw std::runtime_error("raon decode: upload failed");
}

static void get_row(float* host, std::uint64_t device, std::size_t elements) {
    if (flyweight_gpu_download(host, device, elements * sizeof(float), 0) != 0 ||
        flyweight_gpu_stream_sync(0) != 0)
        throw std::runtime_error("raon decode: download failed");
}

// The branch norm is in D.x; the layer output is left in D.hidden.
static void rda_decode(FlyweightV2RaonRuntime& rt, std::size_t l, RaonSlot& slot) {
    const auto& g = rt.geometry;
    const auto& w = rt.layers[l].rda;
    const auto& gl = rt.gpu_layers[l];
    const auto& d = rt.decode_layers[l];
    const auto& D = rt.decode;
    const std::size_t ch = g.heads * g.head_dim, f32 = sizeof(float);
    // q | k | v | gate, ch each: the fused matrix fills it in one launch.
    if (gl.attn_packed)
        matvec_device(gl.attn_type[0], gl.attn_weight[0], D.x, D.pack, g.hidden, 4 * ch);
    else
        for (std::size_t slot_index = 0; slot_index < 4; ++slot_index)
            matvec_device(gl.attn_type[slot_index], gl.attn_weight[slot_index], D.x,
                          D.pack + slot_index * ch * f32, g.hidden, ch);
    if (d.f_a_beta_rda) {
        matvec_device(w.f_a.type, d.f_a_beta_rda, D.x, D.fa, g.hidden, g.ssm_dt_rank + 2 * g.heads);
    } else {
        matvec_device(w.f_a.type, d.f_a, D.x, D.fa, g.hidden, g.ssm_dt_rank);
        matvec_device(w.beta.type, d.beta, D.x, D.beta, g.hidden, g.heads);
        matvec_device(w.rda.type, d.rda, D.x, D.delta, g.hidden, g.heads);
    }
    matvec_device(w.f_b.type, d.f_b, D.fa, D.fb, g.ssm_dt_rank, ch);
    const std::size_t history = g.conv_width - 1;
    for (std::size_t c = 0; c < 3; ++c) {
        std::uint64_t input = D.pack + c * ch * f32, weights = d.small + c * ch * g.conv_width * f32;
        std::uint64_t window = slot.rda_window[l] + c * ch * history * f32, output = D.qkv + c * ch * f32;
        long long stride = static_cast<long long>(4 * ch);
        int tokens = 1, channels = static_cast<int>(ch);
        void* arguments[] = {&input, &stride, &weights, &window, &output, &tokens, &channels};
        launch("raon_rda_conv", static_cast<std::uint32_t>((ch + 127) / 128), 1, 128, arguments);
    }
    {
        const std::uint64_t small = d.small;
        std::uint64_t q = D.qkv, k = D.qkv + ch * f32, v = D.qkv + 2 * ch * f32, fb = D.fb;
        std::uint64_t beta = D.beta, delta = D.delta, gate = D.pack + 3 * ch * f32;
        long long gate_stride = static_cast<long long>(4 * ch);
        std::uint64_t dt = small + 3 * ch * g.conv_width * f32;
        std::uint64_t a = dt + ch * f32, norm = a + g.heads * f32;
        std::uint64_t state = slot.rda_state[l], output = D.att;
        int tokens = 1, heads = static_cast<int>(g.heads), kmajor = 1;
        float lower = g.kda_gate_lower_bound, epsilon = g.epsilon;
        void* arguments[] = {&q, &k, &v, &fb, &beta, &delta, &gate, &gate_stride, &a, &dt, &norm,
                             &state, &output, &tokens, &heads, &lower, &epsilon, &kmajor};
        launch("raon_rda_recurrence", static_cast<std::uint32_t>(g.heads), 1, 128, arguments,
               (128 * 128 + 3 * 128 + 4) * f32);
    }
    matvec_device(gl.attn_type[4], gl.attn_weight[4], D.att, D.hidden, ch, g.hidden);
}

// The branch norm is in D.x; the layer output is left in D.hidden.
static void mla_decode(FlyweightV2RaonRuntime& rt, std::size_t l, RaonSlot& slot, std::size_t position) {
    const auto& g = rt.geometry;
    const auto& gl = rt.gpu_layers[l];
    const auto& d = rt.decode_layers[l];
    const auto& D = rt.decode;
    const std::size_t f32 = sizeof(float), qk = g.qk_nope + g.qk_rope;
    const std::size_t gate_width = g.heads * g.v_head_dim, kv_width = g.kv_lora + g.qk_rope;
    const std::size_t stride = g.qk_nope + g.v_head_dim;
    // [q_a | gate | kv], the layout of mla_step and of the fused matrix.
    if (gl.attn_packed) {
        matvec_device(gl.attn_type[0], gl.attn_weight[0], D.x, D.pack, g.hidden, g.q_lora + gate_width + kv_width);
    } else {
        matvec_device(gl.attn_type[0], gl.attn_weight[0], D.x, D.pack, g.hidden, g.q_lora);
        matvec_device(gl.attn_type[3], gl.attn_weight[3], D.x, D.pack + g.q_lora * f32, g.hidden, gate_width);
        matvec_device(gl.attn_type[2], gl.attn_weight[2], D.x, D.pack + (g.q_lora + gate_width) * f32,
                      g.hidden, kv_width);
    }
    const std::uint64_t gate = D.pack + g.q_lora * f32;
    const std::uint64_t kv = D.pack + (g.q_lora + gate_width) * f32;
    const auto norm_row = [&](std::uint64_t input, std::uint64_t gain, std::uint64_t output, int width) {
        int rows = 1, one_centered = 0;
        float epsilon = g.latent_epsilon;
        void* arguments[] = {&input, &gain, &output, &rows, &width, &epsilon, &one_centered};
        launch("rms_norm_rows", 1, 1, 256, arguments);
    };
    const auto store = [&](std::uint64_t source, std::uint64_t target, std::size_t width) {
        long long offset = static_cast<long long>(position * width);
        int count = static_cast<int>(width), type = g.cache_type;
        void* arguments[] = {&source, &target, &offset, &count, &type};
        launch("raon_cache_store", static_cast<std::uint32_t>((width + 255) / 256), 1, 256, arguments);
    };
    norm_row(kv, d.norms + g.q_lora * f32, D.latent, static_cast<int>(g.kv_lora));
    store(D.latent, slot.mla_latents[l], g.kv_lora);
    store(kv + g.kv_lora * f32, slot.mla_rope[l], g.qk_rope);
    norm_row(D.pack, d.norms, D.qn, static_cast<int>(g.q_lora));
    matvec_device(gl.attn_type[1], gl.attn_weight[1], D.qn, D.query, g.q_lora, g.heads * qk);

    const int heads = static_cast<int>(g.heads);
    const auto head_gemv = [&](std::uint64_t A, std::uint64_t B, std::uint64_t C, int K, int N, int a_head,
                               long long b_head, int b_k, int b_n, int c_head) {
        int half = 1;
        void* arguments[] = {&A, &B, &C, &K, &N, &a_head, &b_head, &b_k, &b_n, &c_head, &half};
        if (b_n == 1)
            launch("raon_head_gemv", static_cast<std::uint32_t>((N + 127) / 128), static_cast<std::uint32_t>(heads),
                   128, arguments);
        else
            launch("raon_head_gemv", static_cast<std::uint32_t>((N + 7) / 8), static_cast<std::uint32_t>(heads),
                   256, arguments);
    };
    // q_nope through kv_b's key half [head][nope][latent].
    head_gemv(D.query, d.kv_b, D.queries, static_cast<int>(g.qk_nope), static_cast<int>(g.kv_lora),
              static_cast<int>(qk), static_cast<long long>(stride * g.kv_lora), static_cast<int>(g.kv_lora), 1,
              static_cast<int>(kv_width));
    {
        std::uint64_t source = D.query, queries = D.queries;
        int tokens = 1, h = heads, nope = static_cast<int>(g.qk_nope), rope = static_cast<int>(g.qk_rope),
            latent = static_cast<int>(g.kv_lora);
        void* arguments[] = {&source, &queries, &tokens, &h, &nope, &rope, &latent};
        launch("raon_mla_rope_queries", static_cast<std::uint32_t>((g.heads * g.qk_rope + 255) / 256), 1, 256,
               arguments);
    }
    {
        // Split-K over the cache, folded by raon_mla_combine.
        constexpr int split_keys = 256;
        int positions = static_cast<int>(position + 1);
        const std::size_t splits = std::min<std::size_t>(D.max_splits, (position + split_keys) / split_keys);
        int per_split = static_cast<int>((position + 1 + splits - 1) / splits);
        std::uint64_t queries = D.queries, latents = slot.mla_latents[l], rope = slot.mla_rope[l];
        std::uint64_t partial = D.partial, partial_ml = D.partial_ml;
        int type = g.cache_type, h = heads;
        float scale = 1.0f / std::sqrt(static_cast<float>(qk));
        void* arguments[] = {&queries, &latents, &rope, &type, &partial, &partial_ml, &h, &positions,
                             &per_split, &scale};
        launch("raon_mla_decode_attention", static_cast<std::uint32_t>((g.heads + 3) / 4),
               static_cast<std::uint32_t>(splits), 256, arguments, (8 * 4 * 512 + 8 * 4 * 2) * sizeof(float));
        std::uint64_t ctx = D.ctx;
        int split_count = static_cast<int>(splits);
        void* fold[] = {&partial, &partial_ml, &ctx, &h, &split_count};
        launch("raon_mla_combine", static_cast<std::uint32_t>(g.heads), 1, 128, fold);
    }
    // Latent context through the value half [head][v][latent]: n = v, k = latent.
    head_gemv(D.ctx, d.kv_b + g.qk_nope * g.kv_lora * 2, D.mla_out, static_cast<int>(g.kv_lora),
              static_cast<int>(g.v_head_dim), static_cast<int>(g.kv_lora),
              static_cast<long long>(stride * g.kv_lora), 1, static_cast<int>(g.kv_lora),
              static_cast<int>(g.v_head_dim));
    {
        std::uint64_t values = D.mla_out, gates = gate;
        int count = static_cast<int>(gate_width);
        void* arguments[] = {&values, &gates, &count};
        launch("raon_sigmoid_mul", static_cast<std::uint32_t>((gate_width + 255) / 256), 1, 256, arguments);
    }
    matvec_device(gl.attn_type[4], gl.attn_weight[4], D.mla_out, D.hidden, gate_width, g.hidden);
    slot.caches[l].positions = position + 1;
}

// ---------------------------------------------------------------------------
// Expert cache (see RaonExpertCache)
// ---------------------------------------------------------------------------

static void cache_unlink(RaonExpertCache& c, std::int32_t slot) {
    const auto newer = c.newer[slot], older = c.older[slot];
    if (newer >= 0) c.older[newer] = older; else if (c.head == slot) c.head = older;
    if (older >= 0) c.newer[older] = newer; else if (c.tail == slot) c.tail = newer;
    c.newer[slot] = c.older[slot] = -1;
}

static void cache_push_front(RaonExpertCache& c, std::int32_t slot) {
    c.newer[slot] = -1;
    c.older[slot] = c.head;
    if (c.head >= 0) c.newer[c.head] = slot;
    c.head = slot;
    if (c.tail < 0) c.tail = slot;
}

// Admissions whose upload stream has drained become usable.
static void cache_poll(RaonExpertCache& c) {
    if (c.pending.empty() || flyweight_gpu_stream_query(c.stream) != 0) return;
    for (auto slot : c.pending) c.ready[slot] = 1;
    c.pending.clear();
}

// Upload expert `e` of layer `l` from its (pinned) host copy into a free or
// least-recently-used slot. Called after the decode step has synced the
// compute stream at this layer, so no queued kernel still reads the victim;
// the slot is not used until the upload is seen to land.
static void cache_admit(FlyweightV2RaonRuntime& rt, std::size_t l, std::int32_t e) {
    auto& c = rt.expert_cache;
    const auto& g = rt.geometry;
    const auto& f = rt.layers[l].ffn;
    if (c.admitted >= c.admit_budget) return;
    std::int32_t slot;
    if (c.unused < static_cast<std::int32_t>(c.slots)) {
        slot = c.unused++;
    } else {
        slot = c.tail;
        if (slot < 0) return;
        cache_unlink(c, slot);
        c.slot_of[static_cast<std::size_t>(c.owner[slot])] = -1;
    }
    const std::size_t gate_bytes = g.expert_size * flyweight::v2::bailing::row_bytes(f.gate_experts.type, g.hidden);
    const std::size_t up_bytes = g.expert_size * flyweight::v2::bailing::row_bytes(f.up_experts.type, g.hidden);
    const std::size_t down_bytes = g.hidden * flyweight::v2::bailing::row_bytes(f.down_experts.type, g.expert_size);
    const std::uint64_t target = c.pool + static_cast<std::uint64_t>(slot) * c.slot_bytes;
    const auto* gate = static_cast<const std::uint8_t*>(f.gate_experts.data) + e * gate_bytes;
    const auto* up = static_cast<const std::uint8_t*>(f.up_experts.data) + e * up_bytes;
    const auto* down = static_cast<const std::uint8_t*>(f.down_experts.data) + e * down_bytes;
    if (flyweight_gpu_upload(target, gate, gate_bytes, c.stream) != 0 ||
        flyweight_gpu_upload(target + gate_bytes, up, up_bytes, c.stream) != 0 ||
        flyweight_gpu_upload(target + gate_bytes + up_bytes, down, down_bytes, c.stream) != 0)
        throw std::runtime_error("raon decode: expert upload failed");
    const auto key = static_cast<std::int32_t>(l * g.experts + static_cast<std::size_t>(e));
    c.owner[slot] = key;
    c.slot_of[static_cast<std::size_t>(key)] = slot;
    c.ready[slot] = 0;
    c.pending.push_back(slot);
    cache_push_front(c, slot);
    ++c.admitted;
    ++c.admissions;
}

// The branch norm is in D.x; the block output is left in D.hidden. Only a
// MoE layer whose experts stay on the CPU goes back to the host: the norm
// comes down, the routed experts run while the GPU does the shared expert,
// and their sum goes back up to be added to it.
static void ffn_decode(FlyweightV2RaonRuntime& rt, std::size_t l) {
    const auto& g = rt.geometry;
    const auto& layer = rt.layers[l];
    const auto& gl = rt.gpu_layers[l];
    const auto& d = rt.decode_layers[l];
    const auto& D = rt.decode;
    const auto& gs = rt.gpu_scratch;
    const auto situ = [&](std::uint64_t gate, std::uint64_t up, std::uint64_t act, std::size_t size) {
        if (gs.launch_situ_glu(gate, up, act, static_cast<int>(size), g.situ_beta, g.situ_linear_beta) != 0)
            throw std::runtime_error("raon decode: situ launch failed");
    };
    const auto add = [&](std::uint64_t target, std::uint64_t source, float weight) {
        if (flyweight_gpu_scaled_add(target, source, weight, static_cast<int>(g.hidden)) != 0)
            throw std::runtime_error("raon decode: add failed");
    };
    if (!layer.is_moe) {
        raon::ProfileScope dense(raon::profile().dense_ffn);
        matvec_device(layer.ffn.gate.type, d.dense_gate, D.x, D.ffn_gate, g.hidden, g.dense_size);
        matvec_device(layer.ffn.up.type, d.dense_up, D.x, D.ffn_up, g.hidden, g.dense_size);
        situ(D.ffn_gate, D.ffn_up, D.ffn_act, g.dense_size);
        matvec_device(layer.ffn.down.type, d.dense_down, D.ffn_act, D.hidden, g.dense_size, g.hidden);
        return;
    }
    const bool shared = g.shared_experts > 0 && d.shared_gate;
    const auto run_shared = [&](std::uint64_t target) {
        matvec_device(layer.ffn.shared_gate.type, d.shared_gate, D.x, D.ffn_gate, g.hidden, g.shared_size);
        matvec_device(layer.ffn.shared_up.type, d.shared_up, D.x, D.ffn_up, g.hidden, g.shared_size);
        situ(D.ffn_gate, D.ffn_up, D.ffn_act, g.shared_size);
        matvec_device(layer.ffn.shared_down.type, d.shared_down, D.ffn_act, target, g.shared_size, g.hidden);
    };
    if (gl.pinned) {
        // raon::ffn_step's GPU route with the norm already on the device.
        raon::ProfileScope moe(raon::profile().moe);
        std::vector<float> logits(g.experts);
        std::vector<std::int32_t> order(g.experts);
        std::vector<float> weights(g.experts_used);
        matvec_device(gl.router_type, gl.router, D.x, gs.router_logits, g.hidden, g.experts);
        get_row(logits.data(), gs.router_logits, g.experts);
        raon::route_token(logits.data(), gl.router_bias, g, order.data(), weights.data());
        if (shared) run_shared(D.hidden);
        else if (flyweight_gpu_memset(D.hidden, 0, g.hidden * sizeof(float), 0) != 0)
            throw std::runtime_error("raon decode: memset failed");
        const auto narrow = g.expert_size * flyweight::v2::bailing::row_bytes(gl.gate_type, g.hidden);
        const auto wide = g.hidden * flyweight::v2::bailing::row_bytes(gl.down_type, g.expert_size);
        for (std::size_t k = 0; k < g.experts_used; ++k) {
            const auto e = static_cast<std::size_t>(order[k]);
            matvec_device(gl.gate_type, gl.gate_exps + e * narrow, D.x, gs.expert_gate, g.hidden, g.expert_size);
            matvec_device(gl.up_type, gl.up_exps + e * narrow, D.x, gs.expert_up, g.hidden, g.expert_size);
            situ(gs.expert_gate, gs.expert_up, gs.expert_act, g.expert_size);
            matvec_device(gl.down_type, gl.down_exps + e * wide, gs.expert_act, gs.expert_down, g.expert_size,
                          g.hidden);
            add(D.hidden, gs.expert_down, weights[k]);
        }
        return;
    }
    // The norm comes down first -- a download waits for everything queued
    // before it -- and only then is the shared expert queued, so it runs while
    // the CPU does the routed experts.
    std::vector<float> x(g.hidden), routed(g.hidden);
    get_row(x.data(), D.x, g.hidden);
    if (rt.expert_cache.slots) {
        // Routed experts found in the cache run on the GPU behind the shared
        // expert (which writes D.hidden first); the rest run on the CPU at the
        // same time, and their sum is added once it is up.
        auto& c = rt.expert_cache;
        cache_poll(c);
        std::vector<std::int32_t> order(g.experts);
        std::vector<float> weights(g.experts_used);
        raon::moe_route_cpu(x.data(), layer, g, order.data(), weights.data());
        if (shared) run_shared(D.hidden);
        else if (flyweight_gpu_memset(D.hidden, 0, g.hidden * sizeof(float), 0) != 0)
            throw std::runtime_error("raon decode: memset failed");
        const auto& f = layer.ffn;
        const std::size_t gate_bytes = g.expert_size * flyweight::v2::bailing::row_bytes(f.gate_experts.type, g.hidden);
        const std::size_t up_bytes = g.expert_size * flyweight::v2::bailing::row_bytes(f.up_experts.type, g.hidden);
        std::vector<std::int32_t> missed;
        std::vector<float> missed_weights;
        for (std::size_t k = 0; k < g.experts_used; ++k) {
            const auto key = l * g.experts + static_cast<std::size_t>(order[k]);
            const auto slot = c.slot_of[key];
            ++c.lookups;
            if (slot >= 0 && c.ready[slot]) {
                ++c.hits;
                cache_unlink(c, slot);
                cache_push_front(c, slot);
                const std::uint64_t base = c.pool + static_cast<std::uint64_t>(slot) * c.slot_bytes;
                matvec_device(f.gate_experts.type, base, D.x, gs.expert_gate, g.hidden, g.expert_size);
                matvec_device(f.up_experts.type, base + gate_bytes, D.x, gs.expert_up, g.hidden, g.expert_size);
                situ(gs.expert_gate, gs.expert_up, gs.expert_act, g.expert_size);
                matvec_device(f.down_experts.type, base + gate_bytes + up_bytes, gs.expert_act, gs.expert_down,
                              g.expert_size, g.hidden);
                add(D.hidden, gs.expert_down, weights[k]);
            } else {
                missed.push_back(order[k]);
                missed_weights.push_back(weights[k]);
            }
        }
        for (auto e : missed)
            if (c.slot_of[l * g.experts + static_cast<std::size_t>(e)] < 0) cache_admit(rt, l, e);
        if (!missed.empty()) {
            raon::moe_experts_cpu(x.data(), layer, g, missed.data(), missed_weights.data(), missed.size(),
                                  routed.data());
            put_row(D.routed, routed.data(), g.hidden);
            add(D.hidden, D.routed, 1.0f);
        }
        return;
    }
    if (shared) run_shared(D.shared_out);
    raon::moe_routed_cpu(x.data(), layer, g, routed.data());
    put_row(D.hidden, routed.data(), g.hidden);
    if (shared) add(D.hidden, D.shared_out, 1.0f);
}

static void decode_token(FlyweightV2RaonRuntime& rt, RaonSlot& slot, const float* token_embd,
                         float* final_hidden) {
    const auto& g = rt.geometry;
    const auto& D = rt.decode;
    if (raon::profiling()) raon::profile().tokens++;
    const std::size_t hc = g.mhc_streams, width = hc * g.hidden;
    // The token's four streams start as copies of its embedding and stay on
    // the device for the whole step.
    std::vector<float> streams(width);
    for (std::size_t s = 0; s < hc; ++s)
        std::memcpy(streams.data() + s * g.hidden, token_embd, g.hidden * sizeof(float));
    put_row(D.streams, streams.data(), width);
    rt.expert_cache.admitted = 0;
    const auto mhc_pre = [&](std::size_t l, int site) {
        const auto& d = rt.decode_layers[l];
        const auto& w = site == 0 ? rt.layers[l].mhc_attn : rt.layers[l].mhc_ffn;
        if (w.fn.type == 0 || w.fn.type == 1) {
            std::uint64_t matrix = d.mhc_fn[site], input = D.streams, output = D.mixes;
            int half = w.fn.type == 1 ? 1 : 0, columns = static_cast<int>(width);
            void* arguments[] = {&matrix, &half, &input, &output, &columns};
            launch("raon_long_rows_matvec", 17, 1, 256, arguments);
        } else {
            matvec_device(w.fn.type, d.mhc_fn[site], D.streams, D.mixes, width, 17);
        }
        std::uint64_t streams_dev = D.streams, mixes = D.mixes, base = d.mhc_params[site];
        std::uint64_t scale = base + 17 * sizeof(float), gain = base + 21 * sizeof(float);
        std::uint64_t x = D.x, coefficients = D.coefficients;
        int H = static_cast<int>(g.hidden);
        float mhc_eps = g.mhc_eps, magnitude = g.mhc_magnitude, epsilon = g.epsilon;
        void* arguments[] = {&streams_dev, &mixes, &base, &scale, &gain, &x, &coefficients, &H, &mhc_eps,
                             &magnitude, &epsilon};
        launch("raon_mhc_pre", 1, 1, 1024, arguments);
    };
    const auto mhc_post = [&]() {
        std::uint64_t streams_dev = D.streams, branch = D.hidden, coefficients = D.coefficients;
        int H = static_cast<int>(g.hidden);
        void* arguments[] = {&streams_dev, &branch, &coefficients, &H};
        launch("raon_mhc_post", static_cast<std::uint32_t>((g.hidden + 255) / 256), 1, 256, arguments);
    };
    for (std::size_t l = 0; l < rt.layers.size(); ++l) {
        const auto& layer = rt.layers[l];
        mhc_pre(l, 0);
        if (layer.is_rda) {
            raon::ProfileScope scoped(raon::profile().rda_projection);
            rda_decode(rt, l, slot);
        } else {
            raon::ProfileScope scoped(raon::profile().mla);
            mla_decode(rt, l, slot, slot.position);
        }
        mhc_post();
        mhc_pre(l, 1);
        raon::route_layer = static_cast<int>(l);
        ffn_decode(rt, l);
        raon::route_layer = -1;
        mhc_post();
    }
    get_row(streams.data(), D.streams, width);
    for (std::size_t j = 0; j < g.hidden; ++j) {
        float total = 0.0f;
        for (std::size_t s = 0; s < hc; ++s) total += streams[s * g.hidden + j];
        final_hidden[j] = total / static_cast<float>(hc);
    }
    std::memcpy(slot.streams.data(), streams.data(), width * sizeof(float));
}

}  // namespace raon_prefill

static void raon_prefill_run(FlyweightV2RaonRuntime& runtime, RaonSlot& slot,
                             const std::uint32_t* tokens, std::uint32_t count,
                             float* final_hidden) {
    raon_prefill::run(runtime, slot, tokens, count, final_hidden);
}

static void raon_decode_device(FlyweightV2RaonRuntime& runtime, RaonSlot& slot,
                               const float* token_embd, float* final_hidden) {
    raon_prefill::decode_token(runtime, slot, token_embd, final_hidden);
}
