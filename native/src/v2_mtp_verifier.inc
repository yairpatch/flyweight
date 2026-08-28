// Batched rows forward over the shared layer stack. Two callers:
//   - MTP verification (qwen_verify_target_rows): small row counts, needs the
//     per-row LM-head argmax and the final hidden states.
//   - Chunked prefill (qwen_prefill_rows): large row counts, skips the LM
//     head entirely (only the last prompt token needs logits, and it runs
//     through the regular single-token decode).
// Neither advances runtime.position/processed_tokens; callers own that.

// Rows per launch of the batched quantized matvec. Must match COLIBRI_Q8_ROWS
// in native/include/colibri_v2_qwen_kernels.hpp -- the kernel's accumulators
// and shared reduction slots are sized by it, and a larger batch would be read
// past silently.
static constexpr int kQ8RowBatch=8;

// Tokens per launch of the tiled prefill matmul. Must match
// COLIBRI_Q8_TILE_TOKENS in native/include/colibri_v2_qwen_kernels.hpp -- the
// kernel's per-warp accumulator slots are sized by it, and a larger batch would
// drop the excess rows silently.
static constexpr int kQ8TileTokens=32;

// Output rows per block of the tiled matmul. Must match COLIBRI_Q8_TILE_ROWS.
static constexpr int kQ8TileRows=16;

// The tensor-core matmul's block tile. Must match COLIBRI_MMQ_ROWS and
// COLIBRI_MMQ_TOKENS in native/include/colibri_v2_qwen_kernels.hpp.
static constexpr int kQ8MmqRows=128;
static constexpr int kQ8MmqTokens=128;
// The asymmetric K-quants (Q2_K/Q4_K/Q5_K) run the _MIN variant, which stages
// four more arrays and so cannot afford 64 rows in shared memory. Must match
// COLIBRI_MMQ_MIN_ROWS.
static constexpr int kQ8MmqMinRows=32;
// Must match COLIBRI_MMQ_MIN_TOKENS.
static constexpr int kQ8MmqMinTokens=64;
// Fragments per warp. A warp covers kQ8MmqRowFrags*16 rows by
// kQ8MmqTokenFrags*8 tokens, so the block tile no longer dictates the thread
// count. Must match COLIBRI_MMQ_ROW_FRAGS / COLIBRI_MMQ_TOKEN_FRAGS.
static constexpr int kQ8MmqRowFrags=4;
static constexpr int kQ8MmqTokenFrags=4;
// Threads per block: one warp per fragment *grid*.
static constexpr int kQ8MmqThreads=
    (kQ8MmqRows/16/kQ8MmqRowFrags)*(kQ8MmqTokens/8/kQ8MmqTokenFrags)*32;
// The wide shape (COLIBRI_MMQ_WIDE, GPU only): same 128x128 tile over 16
// warps at 2x4 fragments each. The tile and grid math above stay valid --
// only the block size changes. Must agree with the defines the GPU compile
// injects (see qwen_mmq_wide in v2_runtime.cpp); a wide launch against a
// default-shape kernel sends warps 8..15 past the tile.
static constexpr int kQ8MmqWideThreads=
    (kQ8MmqRows/16/2)*(kQ8MmqTokens/8/4)*32;
// The _MIN variant still runs one fragment per warp.
static constexpr int kQ8MmqMinThreads=(kQ8MmqMinRows/16)*(kQ8MmqMinTokens/8)*32;

// Launch tally for COLIBRI_PREFILL_LAUNCHES; see the `launch` lambda below.
struct QwenLaunchStat { std::uint64_t calls=0, blocks=0; double milliseconds=0.0; };
inline std::map<std::string,QwenLaunchStat>& qwen_launch_stats(){
    static std::map<std::string,QwenLaunchStat> stats; return stats;
}
inline void qwen_note_launch(const char* name,std::uint32_t gx,std::uint32_t gy){
    auto& e=qwen_launch_stats()[name];
    ++e.calls; e.blocks+=static_cast<std::uint64_t>(gx)*gy;
}
// COLIBRI_PREFILL_LAUNCH_TIME=1 adds each kernel's own GPU duration to the tally.
// It brackets every launch with events and syncs, so the WALL time of the run is
// meaningless -- but a kernel's measured duration is not affected by being
// serialized, which is the point: it answers "is this kernel slow here?" against
// the same kernel benchmarked alone.
inline bool qwen_launch_time_enabled(){
    static const bool on=[]{
        const char* v=std::getenv("COLIBRI_PREFILL_LAUNCH_TIME");
        return v&&v[0]=='1';
    }();
    return on;
}
inline void qwen_note_launch_time(const char* name,double milliseconds){
    qwen_launch_stats()[name].milliseconds+=milliseconds;
}
inline void qwen_prefill_launch_report(){
    auto& stats=qwen_launch_stats();
    if(stats.empty())return;
    std::uint64_t total=0;
    for(const auto& kv:stats)total+=kv.second.calls;
    std::vector<std::pair<std::string,QwenLaunchStat>> ordered(stats.begin(),stats.end());
    std::sort(ordered.begin(),ordered.end(),[](const auto&a,const auto&b){
        return a.second.calls>b.second.calls;});
    double total_ms=0.0;
    for(const auto& kv:stats)total_ms+=kv.second.milliseconds;
    std::fprintf(stderr,"[launches] %llu total, %.1f s of measured kernel time\n",
                 static_cast<unsigned long long>(total),total_ms/1000.0);
    for(const auto& kv:ordered)
        std::fprintf(stderr,"[launches] %-34s %8llu calls %5.1f%%  %8.2f s %5.1f%%  %10llu blocks  %6.1f blocks/call\n",
            kv.first.c_str(),static_cast<unsigned long long>(kv.second.calls),
            100.0*kv.second.calls/total,
            kv.second.milliseconds/1000.0,
            total_ms>0?100.0*kv.second.milliseconds/total_ms:0.0,
            static_cast<unsigned long long>(kv.second.blocks),
            static_cast<double>(kv.second.blocks)/kv.second.calls);
    stats.clear();
}

// Row count at or above which a spilled block streams its weights to the device
// instead of computing on the host. The upload is a fixed cost per block per
// chunk (~6.6 ms for an 86 MiB triple over PCIe) where the host compute scales
// with rows, so the crossover is low; 16 keeps MTP's small verification batches
// on the host path where they belong. COLIBRI_HOST_FFN_GPU_ROWS overrides it,
// and 0 disables the batch path entirely.
inline int qwen_host_ffn_batch_threshold(){
    static const int threshold=[]{
        if(const char* env=std::getenv("COLIBRI_HOST_FFN_GPU_ROWS"))
            return static_cast<int>(std::strtol(env,nullptr,10));
        return 16;
    }();
    return threshold;
}

void qwen_forward_rows(
    ColibriV2QwenRuntime& runtime, const std::uint32_t* input_tokens,
    int total_rows, std::uint32_t* output_tokens, std::uint64_t* final_hidden
) {
    // Chunk boundary: pick up any layers the background registration finished.
    qwen_absorb_registration(runtime);
    // The row range every launch below operates over. The phase driver near
    // the bottom installs a span's (begin, count) before invoking each phase
    // function, so `rows` is the current span's row count and `span_begin` its
    // row offset into the batch; outside the layer loop both cover the whole
    // batch. Today a single span covers all rows -- the split exists so a
    // two-half schedule can overlap CPU experts with queued GPU work.
    int rows=total_rows;
    int span_begin=0;
    if(rows<=0||rows>static_cast<int>(runtime.forward_rows_capacity))
        throw std::runtime_error("native Qwen rows forward batch exceeds the prepared capacity");
    if(runtime.cancelled)throw std::runtime_error("native Qwen runtime is cancelled");
    const int hidden_size=static_cast<int>(runtime.model->config.hidden_size);
    const int experts=static_cast<int>(runtime.model->config.expert_count);
    const int top_k=static_cast<int>(runtime.model->config.expert_used_count);
    const int intermediate=static_cast<int>(runtime.moe_intermediate);
    const auto expert_phase=output_tokens
        ?colibri::v2::ExpertExecutionPhase::verification
        :colibri::v2::ExpertExecutionPhase::prefill;
    auto expert_policy=qwen_expert_policy(runtime,expert_phase);
    // The flag must imply routed_gpu_execution_allowed()==false, which the
    // policy only derives in hybrid mode. Setting it in streamed-GPU mode let
    // the admission loop below zero resident experts out of weights_host into
    // GPU tables that the !hybrid_prefill_cpu launch gate never consumed, so
    // those experts' contributions vanished from every chunked-prefill token.
    expert_policy.hybrid_prefill_cpu=
        expert_policy.is_hybrid()&&
        expert_phase==colibri::v2::ExpertExecutionPhase::prefill&&
        runtime.options.hybrid_prefill_cpu!=0;
    // The two-half software pipeline (plans/prefill-pipeline.md): while the
    // engine thread runs one half's routed experts on the CPU, the GPU works
    // through the other half's queued core. Engaged wherever prefill has a
    // CPU expert share to hide -- the CPU expert modes, hybrid_prefill_cpu,
    // and the hybrid split path, whose seeded residents run on the GPU while
    // every other route falls to the host (prefill admission never uploads:
    // misses_may_be_admitted requires decode). Dense checkpoints and the
    // fully resident streamed-GPU mode keep the one-span driver and pay
    // nothing. The half is kept a multiple of the DeltaNet WY chunk; below
    // two chunks there is nothing to overlap. COLIBRI_PREFILL_PIPELINE=0
    // restores the serial driver for rollback and A/B.
    static const bool pipeline_enabled=[]{
        const char*setting=std::getenv("COLIBRI_PREFILL_PIPELINE");
        return !setting||setting[0]!='0';
    }();
    const int pipeline_half=
        total_rows/2/static_cast<int>(colibri::v2::workspace::kDeltaChunk)
        *static_cast<int>(colibri::v2::workspace::kDeltaChunk);
    const bool pipelined=pipeline_enabled&&pipeline_half>0&&experts>0&&
        expert_phase==colibri::v2::ExpertExecutionPhase::prefill&&
        !expert_policy.is_streamed_gpu();
    const float epsilon=runtime.model->config.rms_norm_epsilon?runtime.model->config.rms_norm_epsilon:1.0e-6f;
    for(int row=0;row<rows;++row)
        if(input_tokens[row]>=runtime.model->config.vocabulary_size)
            throw std::runtime_error("native Qwen input token is out of range");
    const auto&workspace_layout=runtime.rows_workspace_layout;
    const auto&host_layout=runtime.rows_host_layout;
    if(workspace_layout.bytes>runtime.workspace_bytes)
        throw std::runtime_error("native Qwen rows forward workspace is too small");
    if(host_layout.bytes>runtime.host_staging_bytes)
        throw std::runtime_error("native Qwen rows forward host workspace overflow");
    // Mutable rather than const: the phase driver re-derives every row-indexed
    // region address for the span it is about to run (base plus span_begin
    // rows of that region's stride), so two spans read and write disjoint
    // slices of the same regions.
    std::uint64_t hidden=workspace_layout.hidden.address(runtime.workspace);
    std::uint64_t residual=workspace_layout.residual.address(runtime.workspace);
    std::uint64_t normalized=workspace_layout.normalized.address(runtime.workspace);
    std::uint64_t first=workspace_layout.first.address(runtime.workspace);
    std::uint64_t second=workspace_layout.second.address(runtime.workspace);
    std::uint64_t third=workspace_layout.third.address(runtime.workspace);
    std::uint64_t fourth=workspace_layout.fourth.address(runtime.workspace);
    std::uint64_t router_logits=workspace_layout.router_logits.address(runtime.workspace);
    std::uint64_t selected_device=workspace_layout.selected_device.address(runtime.workspace);
    std::uint64_t route_weights=workspace_layout.route_weights.address(runtime.workspace);
    std::uint64_t gpu_activated=workspace_layout.gpu_activated.address(runtime.workspace);
    // Small MTP batches are decode-shaped rather than prefill-shaped.  Their
    // k-quant matrix kernels are dominated by reconstructing weights against
    // f32 activations, while decode's DP4A path quantizes each activation once
    // and reuses it across q/k/v and the FFN projections.  Borrow the routed
    // activation region as transient Q8 scratch; expert execution consumes it
    // only after all dense projections for the layer have finished.
    // Dedicated regions rather than borrowing `gpu_activated`, which is sized
    // off top_k and so is empty on a dense model.
    std::uint64_t small_q8=workspace_layout.rows_q8.address(runtime.workspace);
    std::uint64_t small_q8_scales=
        workspace_layout.rows_q8_scales.address(runtime.workspace);
    std::uint64_t q8_cached_input=0;
    int q8_cached_width=0;
    const auto gpu_table_entries=static_cast<std::uint64_t>(total_rows)*top_k;
    std::uint64_t gpu_gate_table=workspace_layout.gpu_gate_table.address(runtime.workspace);
    std::uint64_t gpu_up_table=workspace_layout.gpu_up_table.address(runtime.workspace);
    std::uint64_t gpu_down_table=workspace_layout.gpu_down_table.address(runtime.workspace);
    std::uint64_t gpu_weight_table=workspace_layout.gpu_weight_table.address(runtime.workspace);
    // NVFP4 weight_scale_2 per route, mirroring the decode path: gate/up before
    // SiLU, the linear down scale folded into `up`.
    std::uint64_t gpu_gate_scale_table=workspace_layout.gpu_gate_scale_table.address(runtime.workspace);
    std::uint64_t gpu_up_scale_table=workspace_layout.gpu_up_scale_table.address(runtime.workspace);
    std::uint64_t gpu_count_table=workspace_layout.gpu_count_table.address(runtime.workspace);
    const auto token_device=workspace_layout.token_device.address(runtime.workspace);
    const auto winners=workspace_layout.winners.address(runtime.workspace);
    const auto attention_scores=workspace_layout.attention_scores.address(runtime.workspace);
    const auto mtp_prompt_hidden=
        workspace_layout.mtp_prompt_hidden.address(runtime.workspace);
    // COLIBRI_PREFILL_LAUNCHES=1 tallies launches and total blocks per kernel
    // name over the whole rows forward, dumped by qwen_prefill_launch_report.
    // Launch *count* is the thing the phase trace cannot show: a phase that
    // looks cheap in milliseconds can still be thousands of 4-block grids, and
    // on this path that is the difference between a kernel problem and a
    // batching problem.
    static const bool trace_launches=std::getenv("COLIBRI_PREFILL_LAUNCHES")!=nullptr;
    auto launch=[&](const char*name,std::uint32_t gx,std::uint32_t gy,std::uint32_t bx,void**args,std::uint32_t shared=0){
        if(trace_launches)qwen_note_launch(name,gx,gy);
        if(qwen_launch_time_enabled()){
            static std::uint64_t start=0,stop=0;
            if(!start){colibri_gpu_timed_event_create(&start);
                       colibri_gpu_timed_event_create(&stop);}
            colibri_gpu_event_record(start,runtime.stream);
            if(colibri_gpu_launch_named(name,gx,gy,bx,shared,runtime.stream,args)!=0)
                throw std::runtime_error(std::string("native Qwen rows forward CUDA kernel failed: ")+name);
            colibri_gpu_event_record(stop,runtime.stream);
            colibri_gpu_event_sync(stop);
            float ms=0.0f;
            if(colibri_gpu_event_elapsed(start,stop,&ms)==0)
                qwen_note_launch_time(name,ms);
            if(!trace_launches)qwen_note_launch(name,gx,gy);
            return;
        }
        if(colibri_gpu_launch_named(name,gx,gy,bx,shared,runtime.stream,args)!=0)
            throw std::runtime_error(std::string("native Qwen rows forward CUDA kernel failed: ")+name);
    };
    auto rms_rows=[&](std::uint64_t input,std::uint64_t weights,std::uint64_t output){
        // Invalidate the Q8 activation cache, which is keyed on the source
        // address. `normalized` is rewritten twice per layer -- once before
        // attention and again before the feed-forward -- at the same address,
        // so a per-layer invalidation alone leaves the second set of
        // projections reading the first one's quantized activations. The
        // single-token path invalidates on every norm for the same reason.
        q8_cached_input=0;q8_cached_width=0;
        int one_centered=0;void*args[]={&input,&weights,&output,&rows,const_cast<int*>(&hidden_size),const_cast<float*>(&epsilon),&one_centered};
        launch("rms_norm_rows",rows,1,256,args);
    };
    const bool preserve_mtp_prompt_hidden=
        !output_tokens&&runtime.options.mtp_drafts&&mtp_prompt_hidden;
    const bool had_mtp_target_hidden=runtime.mtp_has_target_hidden;
    if(preserve_mtp_prompt_hidden&&had_mtp_target_hidden){
        auto source=runtime.state+runtime.mtp_target_hidden_offset;
        void*args[]={&source,const_cast<std::uint64_t*>(&mtp_prompt_hidden),
                     const_cast<int*>(&hidden_size)};
        launch("qwen_copy_vector",(hidden_size+255)/256,1,256,args);
    }
    auto q8_rows=[&](std::uint64_t matrix,std::uint64_t input,std::uint64_t output,int input_size,int output_size){
        void*args[]={&matrix,&input,&output,&input_size,&output_size,&rows};
        launch("q8_matmul_tiled",(output_size+31)/32,(static_cast<std::uint32_t>(rows)+31)/32,256,args);
    };
    auto f32_rows=[&](std::uint64_t matrix,std::uint64_t input,std::uint64_t output,int input_size,int output_size){
        void*args[]={&matrix,&input,&output,&input_size,&output_size,&rows};
        launch("qwen_f32_matmul_rows",output_size,rows,256,args);
    };
    // Dense projections carry the checkpoint's own type -- bf16 for the NVFP4
    // Qwen3.6 builds, Q8_0 for the older ones. See the matching dispatch in
    // qwen_decode_step; reading one type as the other silently corrupts every
    // activation instead of failing.
    // `staged` overrides the resident device pointer, for weights that live on
    // the host and were uploaded to a staging buffer for this batch. The type
    // still comes from the index: a spilled tensor is never requantized, so its
    // device type and the raw model bytes agree.
    auto dense_rows=[&](std::size_t index,std::uint64_t input,std::uint64_t output,int input_size,int output_size,std::uint64_t staged=0){
        // Importance-matrix capture: every dense projection funnels through
        // here, so one accumulation covers them all. Stream-ordered like the
        // matmul it precedes, so it reads exactly the activations the
        // projection reads.
        if(runtime.imatrix_sums){
            const auto slot=runtime.imatrix_offsets[index];
            if(slot!=~0ull){
                std::uint64_t sums=runtime.imatrix_sums+slot*sizeof(float);
                void*args[]={&input,&sums,&input_size,const_cast<int*>(&rows)};
                launch("qwen_imatrix_accumulate",
                       (static_cast<std::uint32_t>(input_size)+255)/256,1,256,args);
                runtime.imatrix_rows[index]+=static_cast<std::uint64_t>(rows);
            }
        }
        std::uint64_t matrix=staged?staged:runtime.device_tensors[index];
        const auto type=qwen_device_type(runtime,index);
        const auto*format=colibri::v2::qwen_format(type);
        // Tensor cores where the format has an MMQ kernel; the dp4a tile
        // kernel is the fallback for the rest. Both are resolved before the
        // gate below, because whether either exists is what decides if a wide
        // MoE batch may enter the block at all. (IQ1_S/IQ1_M land on the
        // symmetric MMQ rather than with the asymmetric _MIN three: their
        // per-group deltas fold into the int8 weights exactly.)
        // COLIBRI_DENSE_TRACE=1: which kernel each dense projection actually
        // takes. Reasoning about these gates has been wrong twice (IQ4_NL's
        // missing format row, --hybrid-prefill's silent override), so the
        // dispatch reports itself.
        auto dense_trace=[&](const char* picked){
            static const bool on=std::getenv("COLIBRI_DENSE_TRACE")!=nullptr;
            if(!on)return;
            static std::set<std::string> seen;
            char key[256];
            std::snprintf(key,sizeof(key),"type=%u %dx%d rows=%d -> %s",
                          type,input_size,output_size,rows,picked?picked:"float-fallback");
            if(seen.insert(key).second)std::fprintf(stderr,"[dense] %s\n",key);
        };
        const char* mmq_kernel=format?format->matmul_q8_mmq:nullptr;
        const bool mmq_min=format&&format->mmq_min;
        const char* tiled_kernel=format?format->matmul_q8_tiled:nullptr;
        const char* q8_kernel=nullptr;
        // On MoE the *per-row* path below re-reads the whole weight matrix per
        // row, which only pays off against the reconstruct-in-float matmul when
        // there are few rows. A dense model has no *_matmul_rows kernel that
        // uses DP4A at all, so the same per-row loop is the faster path at any
        // batch size -- prefill was running slower than decode without it.
        //
        // The MMQ and tiled kernels do not re-read per row: they cover
        // kQ8MmqTokens / kQ8TileTokens per weight pass. So a wide MoE batch is
        // admitted too when one of those is what will actually run. Without
        // this a K-quant dense projection on an MoE checkpoint fell past this
        // block to the bottom switch and took *_matmul_rows at 4 rows/block --
        // Q6_K dense measured 39% slower at pp2048 than an otherwise identical
        // Q8_0-dense checkpoint, whose type-8 path is already a 32x32 tile.
        const bool batched_kernel=rows>kQ8RowBatch&&
            ((mmq_kernel&&runtime.int8_tensor_cores)||tiled_kernel);
        // rows_q8_gate, not matvec_q8_warp alone: the table records which
        // formats this block has been measured on (see its drift note).
        if((experts>0?(rows<=8||batched_kernel):true)&&(input_size&255)==0&&
           format&&format->rows_q8_gate)
            q8_kernel=format->matvec_q8_warp;
        if(q8_kernel){
            // The scale rows sit on a float-sized stride (see rows_q8_scales in
            // colibri_v2_workspace.hpp) though the kernels write halves.
            int scale_stride=(input_size/32)*2;
            if(q8_cached_input!=input||q8_cached_width!=input_size){
                std::uint64_t all_q8=small_q8,all_scales=small_q8_scales;
                if(rows>1){
                    void*quant_args[]={&input,&all_q8,&all_scales,
                        &input_size,&scale_stride};
                    launch("quantize_q8_blocks_rows",(input_size+31)/32,
                           static_cast<std::uint32_t>(rows),32,quant_args);
                }else{
                    void*quant_args[]={&input,&all_q8,&all_scales,&input_size};
                    launch("quantize_q8_blocks",(input_size+31)/32,1,32,quant_args);
                }
                q8_cached_input=input;q8_cached_width=input_size;
            }
            // Formats with a batched matvec: one weight read serves 8 rows
            // here, 32 through the tiled kernel or 64 through MMQ. A format
            // without one still pays a full pass per row, which makes its
            // prefill a memory-bound repeat of decode -- the codebook formats
            // every K-quant now has one. The asymmetric three (Q2_K, Q4_K,
            // Q5_K) carry a per-sub-block minimum and go through the _MIN
            // kernels, which correct for it with the activation sums; there is
            // no tiled variant for those, so a pre-sm_75 device falls back to
            // the 8-row kernel rather than the 32-token tile.
            const char* rows_kernel=format->matvec_q8_rows;
            // Wide batches take the tiled kernel instead: it stages decoded
            // weights through shared memory and splits tokens across warps, so
            // it covers kQ8TileTokens per weight pass where the rows kernel
            // covers kQ8RowBatch. Only worth it once the batch is wider than
            // the rows kernel handles in one launch -- below that the tiled
            // kernel leaves most of its warps idle. MTP verification runs a
            // handful of rows and stays on the rows kernel; chunked prefill
            // runs prefill_rows (64 on a dense checkpoint) and takes this.
            // mmq_kernel and tiled_kernel are resolved above the gate.
            if(rows>kQ8RowBatch&&mmq_kernel&&runtime.int8_tensor_cores){
                const int mmq_rows=mmq_min?kQ8MmqMinRows:kQ8MmqRows;
                const int mmq_threads=mmq_min?kQ8MmqMinThreads
                    :qwen_mmq_wide()?kQ8MmqWideThreads:kQ8MmqThreads;
                const int mmq_tokens=mmq_min?kQ8MmqMinTokens:kQ8MmqTokens;
                for(int first=0;first<rows;first+=mmq_tokens){
                    int batch=std::min(mmq_tokens,rows-first);
                    auto batch_q8=small_q8+static_cast<std::uint64_t>(first)*input_size;
                    auto batch_scales=small_q8_scales+
                        static_cast<std::uint64_t>(first)*(input_size/32)*sizeof(float);
                    auto batch_output=output+
                        static_cast<std::uint64_t>(first)*output_size*sizeof(float);
                    void*mmq_args[]={&matrix,&batch_q8,&batch_scales,&batch_output,
                        &input_size,&output_size,&batch,&scale_stride};
                    launch(mmq_kernel,(output_size+mmq_rows-1)/mmq_rows,1,mmq_threads,mmq_args);
                }
                dense_trace(mmq_kernel);
                return;
            }
            // Reached when the device predates sm_75, where the MMQ kernels
            // would run their emulation instead of real tensor cores.
            if(rows>kQ8RowBatch&&tiled_kernel){
                for(int first=0;first<rows;first+=kQ8TileTokens){
                    int batch=std::min(kQ8TileTokens,rows-first);
                    auto batch_q8=small_q8+static_cast<std::uint64_t>(first)*input_size;
                    auto batch_scales=small_q8_scales+
                        static_cast<std::uint64_t>(first)*(input_size/32)*sizeof(float);
                    auto batch_output=output+
                        static_cast<std::uint64_t>(first)*output_size*sizeof(float);
                    void*tiled_args[]={&matrix,&batch_q8,&batch_scales,&batch_output,
                        &input_size,&output_size,&batch,&scale_stride};
                    launch(tiled_kernel,(output_size+kQ8TileRows-1)/kQ8TileRows,1,256,tiled_args);
                }
                dense_trace(tiled_kernel);
                return;
            }
            if(rows>1&&rows_kernel){
                for(int first=0;first<rows;first+=kQ8RowBatch){
                    int batch=std::min(kQ8RowBatch,rows-first);
                    auto batch_q8=small_q8+static_cast<std::uint64_t>(first)*input_size;
                    auto batch_scales=small_q8_scales+
                        static_cast<std::uint64_t>(first)*(input_size/32)*sizeof(float);
                    auto batch_output=output+
                        static_cast<std::uint64_t>(first)*output_size*sizeof(float);
                    void*rows_args[]={&matrix,&batch_q8,&batch_scales,&batch_output,
                        &input_size,&output_size,&batch,&scale_stride};
                    launch(rows_kernel,output_size,1,128,rows_args);
                }
                return;
            }
            for(int row=0;row<rows;++row){
                auto row_q8=small_q8+static_cast<std::uint64_t>(row)*input_size;
                auto row_scales=small_q8_scales+
                    static_cast<std::uint64_t>(row)*(input_size/32)*sizeof(float);
                auto row_output=output+static_cast<std::uint64_t>(row)*output_size*sizeof(float);
                void*matvec_args[]={&matrix,&row_q8,&row_scales,&row_output,
                    &input_size,&output_size};
                launch(q8_kernel,output_size,1,128,matvec_args);
            }
            return;
        }
        if(rows<=8&&type==8){
            if(rows==2){
                void*pair_args[]={&matrix,&input,&output,&input_size,&output_size};
                launch("q8_matvec_transposed_pair",(output_size+7)/8,1,256,
                       pair_args);
                return;
            }
            if(rows==3){
                void*triple_args[]={&matrix,&input,&output,&input_size,&output_size};
                launch("q8_matvec_transposed_triple",(output_size+7)/8,1,256,
                       triple_args);
                return;
            }
            for(int row=0;row<rows;++row){
                const auto row_input=input+
                    static_cast<std::uint64_t>(row)*input_size*sizeof(float);
                const auto row_output=output+
                    static_cast<std::uint64_t>(row)*output_size*sizeof(float);
                if(colibri_gpu_q8_matvec_transposed(
                       matrix,row_input,row_output,input_size,output_size,
                       runtime.stream)!=0)
                    throw std::runtime_error(
                        "native Qwen rows Q8 projection failed");
            }
            return;
        }
        // The reconstruct-in-float batch matmul, one launch for the whole
        // chunk. A format without one (a *future* format: every current type
        // has an entry) pays the per-token fallback below -- 64 launches of a
        // 48-block grid per projection per layer, pure launch latency, which
        // is what IQ1_M's ssm_alpha/ssm_beta measured as the bulk of the
        // prefill "proj" phase before it got its rows kernel.
        if(format&&format->matmul_rows){
            void*args[]={&matrix,&input,&output,&input_size,&output_size,&rows};
            using Grid=colibri::v2::RowsMatmulGrid;
            switch(format->matmul_rows_grid){
                case Grid::per_token:
                    launch(format->matmul_rows,output_size,rows,256,args);return;
                case Grid::tiled32:
                    launch(format->matmul_rows,(output_size+31)/32,
                           (static_cast<std::uint32_t>(rows)+31)/32,256,args);return;
                case Grid::quad_pack:
                    launch(format->matmul_rows,output_size,(rows+3)/4,256,args);return;
                case Grid::none:break;
            }
        }
        // Keep an explicit compatibility fallback for future tensor formats.
        for(int token=0;token<rows;++token){
            const auto row_input=input+static_cast<std::uint64_t>(token)*input_size*sizeof(float);
            const auto row_output=output+static_cast<std::uint64_t>(token)*output_size*sizeof(float);
            if(qwen_gpu_matvec_by_type(type,matrix,row_input,row_output,input_size,output_size,runtime.stream)!=0)
                throw std::runtime_error(
                    "native Qwen rows dense projection failed for type "+std::to_string(type)
                    +" ("+std::to_string(input_size)+"x"+std::to_string(output_size)
                    +", "+std::to_string(rows)+" rows"
                    +(matrix?"":", weights are not device-resident")+")");
        }
        return;
    };
    auto add_rows=[&](std::uint64_t target,std::uint64_t source){float scale=1.0f;int count=rows*hidden_size;void*args[]={&target,&source,&scale,&count};launch("scaled_add",(count+255)/256,1,256,args);};
    // Muse Glimmer's post-attention and post-FFN norms, which run at a tighter
    // epsilon than the model's own and are applied in place before the residual
    // join. rms_norm_rows reduces per row before writing, so in place is safe.
    const float post_norm_epsilon=1.0e-8f;
    auto post_rms_rows=[&](std::uint64_t values,std::uint64_t weights){
        q8_cached_input=0;q8_cached_width=0;
        int one_centered=0;
        void*args[]={&values,&weights,&values,&rows,const_cast<int*>(&hidden_size),
                     const_cast<float*>(&post_norm_epsilon),&one_centered};
        launch("rms_norm_rows",rows,1,256,args);
    };
    // qwen4exp gated-residual bookends, rows form. Same seam as the decode
    // path: hc_pre_rows replaces the pre-block rms (streams -> `normalized`,
    // raw inject logits parked per row), hc_post_rows replaces the residual
    // add. The buffers slice by span_begin exactly like hidden/residual do,
    // so the two-half pipeline addresses its own rows.
    const int hc_count=static_cast<int>(runtime.model->config.hyper_connection_count);
    const int hc_low_rank=static_cast<int>(runtime.model->config.hyper_connection_low_rank);
    const std::uint64_t hc_wide_size=static_cast<std::uint64_t>(hc_count)*hidden_size;
    const auto hc_streams_base=workspace_layout.streams.address(runtime.workspace);
    const auto hc_normed_base=workspace_layout.hc_normed.address(runtime.workspace);
    const auto hc_wide_base=workspace_layout.hc_wide.address(runtime.workspace);
    const auto hc_low_base=workspace_layout.hc_low.address(runtime.workspace);
    const auto hc_gates_base=workspace_layout.hc_gates.address(runtime.workspace);
    const auto ple_embed_base=workspace_layout.ple_embed.address(runtime.workspace);
    // QSA scratch (qwen4exp): NOT span-sliced -- each span's usage is
    // enqueue-local (projected, consumed, and free before the other span's
    // layer body queues), and the score/slot buffers are single-row anyway.
    const auto qsa_query_base=workspace_layout.qsa_query.address(runtime.workspace);
    const auto qsa_keys_base=workspace_layout.qsa_keys.address(runtime.workspace);
    const auto qsa_scores_base=workspace_layout.qsa_block_scores.address(runtime.workspace);
    const auto qsa_slots_base=workspace_layout.qsa_slots.address(runtime.workspace);
    auto hc_slice=[&](std::uint64_t base,std::uint64_t per_row){
        return base+static_cast<std::uint64_t>(span_begin)*per_row*sizeof(float);
    };
    auto hc_pre_rows=[&](std::uint64_t norm_weights,std::uint64_t down_index,
                         std::uint64_t up_index,std::uint64_t inject_index){
        const auto streams=hc_slice(hc_streams_base,hc_wide_size);
        const auto normed=hc_slice(hc_normed_base,hc_wide_size);
        const auto wide=hc_slice(hc_wide_base,hc_wide_size);
        const auto low=hc_slice(hc_low_base,hc_low_rank);
        const auto gates_raw=hc_slice(hc_gates_base,hc_count);
        void*norm_args[]={const_cast<std::uint64_t*>(&streams),&norm_weights,
            const_cast<std::uint64_t*>(&normed),const_cast<int*>(&hc_count),
            const_cast<int*>(&hidden_size),const_cast<float*>(&epsilon)};
        launch("qwen4_group_rms_rows",hc_count,rows,256,norm_args);
        dense_rows(down_index,normed,low,
                   static_cast<int>(hc_wide_size),hc_low_rank);
        float divisor=static_cast<float>(hc_count);
        int silu_count=rows*hc_low_rank;
        void*silu_args[]={const_cast<std::uint64_t*>(&low),&silu_count,&divisor};
        launch("qwen4_silu_scale",
               (static_cast<std::uint32_t>(silu_count)+255)/256,1,256,silu_args);
        dense_rows(up_index,low,wide,hc_low_rank,
                   static_cast<int>(hc_wide_size));
        if(inject_index!=std::numeric_limits<std::uint64_t>::max())
            dense_rows(inject_index,normed,gates_raw,
                       static_cast<int>(hc_wide_size),hc_count);
        void*mix_args[]={const_cast<std::uint64_t*>(&normed),
            const_cast<std::uint64_t*>(&wide),
            const_cast<std::uint64_t*>(&normalized),
            const_cast<int*>(&hc_count),const_cast<int*>(&hidden_size)};
        launch("qwen4_hc_mix_rows",
               (static_cast<std::uint32_t>(hidden_size)+255)/256,rows,256,mix_args);
        q8_cached_input=0;q8_cached_width=0;
    };
    auto hc_post_rows=[&](std::uint64_t block_output){
        const auto streams=hc_slice(hc_streams_base,hc_wide_size);
        const auto gates_raw=hc_slice(hc_gates_base,hc_count);
        void*inject_args[]={const_cast<std::uint64_t*>(&streams),
            const_cast<std::uint64_t*>(&block_output),
            const_cast<std::uint64_t*>(&gates_raw),
            const_cast<int*>(&hc_count),const_cast<int*>(&hidden_size)};
        launch("qwen4_hc_inject_rows",
               (static_cast<std::uint32_t>(hc_wide_size)+255)/256,rows,256,
               inject_args);
    };
    auto ple_rows=[&](const QwenLayerPlan&layer){
        const auto streams=hc_slice(hc_streams_base,hc_wide_size);
        const auto normed=hc_slice(hc_normed_base,hc_wide_size);
        const auto wide=hc_slice(hc_wide_base,hc_wide_size);
        const auto embed=hc_slice(ple_embed_base,hidden_size);
        const auto gates_raw=hc_slice(hc_gates_base,hc_count);
        dense_rows(layer.ple_key,embed,wide,hidden_size,
                   static_cast<int>(hc_wide_size));
        auto norm_key=runtime.device_tensors[layer.ple_norm_key];
        void*key_args[]={const_cast<std::uint64_t*>(&wide),&norm_key,
            const_cast<std::uint64_t*>(&wide),const_cast<int*>(&hc_count),
            const_cast<int*>(&hidden_size),const_cast<float*>(&epsilon)};
        launch("qwen4_group_rms_rows",hc_count,rows,256,key_args);
        auto norm_query=runtime.device_tensors[layer.ple_norm_query];
        void*query_args[]={const_cast<std::uint64_t*>(&streams),&norm_query,
            const_cast<std::uint64_t*>(&normed),const_cast<int*>(&hc_count),
            const_cast<int*>(&hidden_size),const_cast<float*>(&epsilon)};
        launch("qwen4_group_rms_rows",hc_count,rows,256,query_args);
        void*gate_args[]={const_cast<std::uint64_t*>(&wide),
            const_cast<std::uint64_t*>(&normed),
            const_cast<std::uint64_t*>(&gates_raw),
            const_cast<int*>(&hc_count),const_cast<int*>(&hidden_size)};
        launch("qwen4_ple_gate_rows",hc_count,rows,256,gate_args);
        // value borrows `normalized`; hc_pre_rows rewrites it right after.
        dense_rows(layer.ple_value,embed,normalized,hidden_size,hidden_size);
        void*gv_args[]={const_cast<std::uint64_t*>(&normalized),
            const_cast<std::uint64_t*>(&gates_raw),
            const_cast<std::uint64_t*>(&wide),
            const_cast<int*>(&hc_count),const_cast<int*>(&hidden_size)};
        launch("qwen4_ple_gv_rows",
               (static_cast<std::uint32_t>(hc_wide_size)+255)/256,rows,256,gv_args);
        auto norm_conv=runtime.device_tensors[layer.ple_norm_conv];
        void*gvn_args[]={const_cast<std::uint64_t*>(&wide),&norm_conv,
            const_cast<std::uint64_t*>(&normed),const_cast<int*>(&hc_count),
            const_cast<int*>(&hidden_size),const_cast<float*>(&epsilon)};
        launch("qwen4_group_rms_rows",hc_count,rows,256,gvn_args);
        const auto&conv=runtime.model->tensors[layer.ple_conv];
        int kernel_size=static_cast<int>(conv.shape[0]);
        int dilation=static_cast<int>(std::max<std::uint32_t>(
            runtime.model->config.ple_ngram_size,1));
        int channels=static_cast<int>(hc_wide_size);
        auto conv_weights=runtime.device_tensors[layer.ple_conv];
        std::uint64_t conv_state=runtime.state+layer.state_third;
        void*conv_args[]={const_cast<std::uint64_t*>(&normed),&conv_weights,
            &conv_state,const_cast<std::uint64_t*>(&normed),&rows,&channels,
            &kernel_size,&dilation};
        launch("qwen4_ple_conv_sequence",
               (static_cast<std::uint32_t>(channels)+255)/256,1,256,conv_args);
        int add_count=rows*static_cast<int>(hc_wide_size);
        void*add_args[]={const_cast<std::uint64_t*>(&streams),
            const_cast<std::uint64_t*>(&wide),
            const_cast<std::uint64_t*>(&normed),&add_count};
        launch("qwen4_ple_add",
               (static_cast<std::uint32_t>(add_count)+255)/256,1,256,add_args);
        q8_cached_input=0;q8_cached_width=0;
    };
    if(colibri_gpu_upload(token_device,input_tokens,rows*sizeof(std::uint32_t),runtime.stream)!=0)throw std::runtime_error("native Qwen rows token upload failed");
    const auto embedding=qwen_stage_embedding_rows(runtime,input_tokens,rows);
    // Staged rows arrive in chunk order, so they are gathered by the identity
    // permutation rather than by the real token ids in token_device.
    const auto embedding_index=runtime.embeddings_host_resident
        ?runtime.embedding_row_index:token_device;
    void*embedding_args[]={const_cast<std::uint64_t*>(&embedding),const_cast<std::uint64_t*>(&embedding_index),&hidden,&rows,const_cast<int*>(&hidden_size)};
    launch(qwen_embedding_kernel(qwen_device_type(runtime,runtime.token_embeddings),true),(hidden_size+255)/256,rows,256,embedding_args);
    if(runtime.qwen4exp){
        // Streams init for the whole batch, and the chunk's n-gram gathers
        // staged up front so the PLE layer finds them uploaded.
        void*init_args[]={const_cast<std::uint64_t*>(&hidden),
            const_cast<std::uint64_t*>(&hc_streams_base),
            const_cast<int*>(&hc_count),const_cast<int*>(&hidden_size)};
        launch("qwen4_hc_init_rows",
               (static_cast<std::uint32_t>(hc_wide_size)+255)/256,rows,256,
               init_args);
        if(runtime.ple_table!=std::numeric_limits<std::uint64_t>::max()&&
           !runtime.model->config.ple_layers.empty()){
            // Timed because it is the one large cost the prefill split could
            // not see: the gpu_* event pairs are unavailable under the
            // pipelined driver, so everything that was not the CPU expert
            // phase landed in one unattributed bucket. This is host work, so a
            // steady_clock reads it in either driver.
            const auto ple_started=std::chrono::steady_clock::now();
            qwen4_stage_ple_embed_rows(runtime,input_tokens,rows,ple_embed_base);
            runtime.prefill_ple_nanoseconds+=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now()-ple_started).count();
        }
    }
    // Muse Glimmer RMS-normalizes the raw embedding, with no learned weight,
    // before the first block. The decode path does the same; a row batch that
    // skipped it would disagree with token-by-token prefill immediately.
    if(runtime.muse){
        int norm_rows=rows,width=hidden_size;
        void*norm_args[]={const_cast<std::uint64_t*>(&hidden),
                          const_cast<std::uint64_t*>(&hidden),&norm_rows,&width,
                          const_cast<float*>(&epsilon)};
        launch("gemma_head_rms",rows,1,256,norm_args);
    }
    auto*staging=static_cast<std::uint8_t*>(runtime.host_staging);
    const bool prune_experts=expert_policy.route_pruning_allowed()&&
        (runtime.options.expert_top_k>0||(runtime.options.expert_top_p>0.0f&&runtime.options.expert_top_p<1.0f));
    std::vector<float>gpu_weights_store(gpu_table_entries);
    std::vector<float>gpu_gate_scales_store(gpu_table_entries),gpu_up_scales_store(gpu_table_entries);
    std::vector<std::uint64_t>gate_pointers_store(gpu_table_entries),up_pointers_store(gpu_table_entries),down_pointers_store(gpu_table_entries);
    std::vector<std::int32_t>gpu_counts_store(total_rows);
    // Span views over the stores and the host staging regions, re-sliced by
    // install_span below just like the device regions.
    float*gpu_weights=gpu_weights_store.data();
    float*gpu_gate_scales=gpu_gate_scales_store.data();
    float*gpu_up_scales=gpu_up_scales_store.data();
    std::uint64_t*gate_pointers=gate_pointers_store.data();
    std::uint64_t*up_pointers=up_pointers_store.data();
    std::uint64_t*down_pointers=down_pointers_store.data();
    std::int32_t*gpu_counts=gpu_counts_store.data();
    auto*selected_host=reinterpret_cast<std::int32_t*>(
        staging+host_layout.selected.offset);
    auto*weights_host=reinterpret_cast<float*>(
        staging+host_layout.weights.offset);
    auto*input_host=reinterpret_cast<float*>(
        staging+host_layout.input.offset);
    auto*activated_host=reinterpret_cast<float*>(
        staging+host_layout.activated.offset);
    auto*down_host=reinterpret_cast<float*>(
        staging+host_layout.down_input.offset);
    auto*output_host=reinterpret_cast<float*>(
        staging+host_layout.output.offset);
    // One contiguous slice of the row batch, carried through the three phase
    // functions below. The driver runs a single full-range span today; a
    // two-half schedule swaps two of these through the same functions, so a
    // span owns everything that is per-half state: its row range, which of the
    // two ping-pong buffers holds its current layer input, and the pager clock
    // that spans cpu_phase into queue_combine.
    struct RowsSpan{
        int begin;int count;
        std::uint64_t hidden;std::uint64_t residual;
        // Which route event this span records and waits on. The two-half
        // schedule gives each half its own event; sharing one would re-record
        // it from the other half's core and stretch this half's route wait
        // over work it does not depend on.
        std::uint64_t route_event;
        // This span's slice of the prefill expert-streaming arena (halves get
        // disjoint slices, reused layer to layer). Zero bytes disables
        // streaming for the span. Uploads ride the prefetch stream:
        // upload_ready fences the compute stream before the grouped kernels
        // read staged weights, consumed fences the prefetch stream before the
        // next layer's uploads overwrite the slice.
        std::uint64_t stream_base;std::uint64_t stream_bytes;
        std::uint64_t stream_scratch;
        std::uint64_t upload_ready_event;std::uint64_t consumed_event;
        // Whether cpu_phase queued GPU expert work whose accumulator
        // queue_combine must fold into the residual.
        bool gpu_experts;
        std::chrono::steady_clock::time_point pager_started;
    };
    const auto scratch_stride=static_cast<std::uint64_t>(runtime.scratch_elements);
    auto install_span=[&](const RowsSpan&span){
        rows=span.count;
        span_begin=span.begin;
        const auto row=static_cast<std::uint64_t>(span.begin);
        hidden=span.hidden+row*hidden_size*sizeof(float);
        residual=span.residual+row*hidden_size*sizeof(float);
        normalized=workspace_layout.normalized.address(runtime.workspace)+row*hidden_size*sizeof(float);
        first=workspace_layout.first.address(runtime.workspace)+row*scratch_stride*sizeof(float);
        second=workspace_layout.second.address(runtime.workspace)+row*scratch_stride*sizeof(float);
        third=workspace_layout.third.address(runtime.workspace)+row*scratch_stride*sizeof(float);
        fourth=workspace_layout.fourth.address(runtime.workspace)+row*scratch_stride*sizeof(float);
        router_logits=workspace_layout.router_logits.address(runtime.workspace)+row*experts*sizeof(float);
        selected_device=workspace_layout.selected_device.address(runtime.workspace)+row*top_k*sizeof(std::int32_t);
        route_weights=workspace_layout.route_weights.address(runtime.workspace)+row*top_k*sizeof(float);
        gpu_activated=workspace_layout.gpu_activated.address(runtime.workspace)+row*top_k*intermediate*sizeof(float);
        small_q8=workspace_layout.rows_q8.address(runtime.workspace)+row*scratch_stride;
        small_q8_scales=workspace_layout.rows_q8_scales.address(runtime.workspace)+row*(scratch_stride/32+1)*sizeof(float);
        gpu_gate_table=workspace_layout.gpu_gate_table.address(runtime.workspace)+row*top_k*sizeof(std::uint64_t);
        gpu_up_table=workspace_layout.gpu_up_table.address(runtime.workspace)+row*top_k*sizeof(std::uint64_t);
        gpu_down_table=workspace_layout.gpu_down_table.address(runtime.workspace)+row*top_k*sizeof(std::uint64_t);
        gpu_weight_table=workspace_layout.gpu_weight_table.address(runtime.workspace)+row*top_k*sizeof(float);
        gpu_gate_scale_table=workspace_layout.gpu_gate_scale_table.address(runtime.workspace)+row*top_k*sizeof(float);
        gpu_up_scale_table=workspace_layout.gpu_up_scale_table.address(runtime.workspace)+row*top_k*sizeof(float);
        gpu_count_table=workspace_layout.gpu_count_table.address(runtime.workspace)+row*sizeof(std::int32_t);
        gpu_weights=gpu_weights_store.data()+row*top_k;
        gpu_gate_scales=gpu_gate_scales_store.data()+row*top_k;
        gpu_up_scales=gpu_up_scales_store.data()+row*top_k;
        gate_pointers=gate_pointers_store.data()+row*top_k;
        up_pointers=up_pointers_store.data()+row*top_k;
        down_pointers=down_pointers_store.data()+row*top_k;
        gpu_counts=gpu_counts_store.data()+row;
        selected_host=reinterpret_cast<std::int32_t*>(
            staging+host_layout.selected.offset)+row*top_k;
        weights_host=reinterpret_cast<float*>(
            staging+host_layout.weights.offset)+row*top_k;
        input_host=reinterpret_cast<float*>(
            staging+host_layout.input.offset)+row*hidden_size;
        activated_host=reinterpret_cast<float*>(
            staging+host_layout.activated.offset)+row*top_k*intermediate;
        down_host=reinterpret_cast<float*>(
            staging+host_layout.down_input.offset)+row*top_k*hidden_size;
        output_host=reinterpret_cast<float*>(
            staging+host_layout.output.offset)+row*hidden_size;
    };
    // The same GPU-event split (core / router / transfer) answers "is
    // route_wait real GPU work or a bubble?" for MTP verification, which is
    // the phase that currently stalls more than decode. Opt in with
    // COLIBRI_MTP_PROFILE=1; it accumulates into the prefill_gpu_* counters,
    // so a probe must snapshot them between the prefill and decode phases.
    static const bool verify_profile=
        std::getenv("COLIBRI_MTP_PROFILE")!=nullptr;
    // The event split cannot describe two interleaved halves (each half would
    // re-record the other's start/end markers), so the pipelined driver keeps
    // only the host-side counters and leaves the prefill_gpu_* split at zero.
    const bool profile_prefill=!pipelined&&
        ((runtime.prefill_profile&&expert_policy.records_prefill_frequency())||
         (verify_profile&&
          expert_phase==colibri::v2::ExpertExecutionPhase::verification));
    static const bool trace_phases=std::getenv("COLIBRI_PREFILL_TRACE")
        &&std::getenv("COLIBRI_PREFILL_TRACE")[0]=='2';
    auto phase_clock=std::chrono::steady_clock::now();
    auto phase=[&](std::uint32_t layer_number,const char*name){
        if(!trace_phases)return;
        colibri_gpu_stream_sync(runtime.stream);
        const auto now=std::chrono::steady_clock::now();
        std::fprintf(stderr,"[phase] layer %2u %-10s %.1fms\n",layer_number,name,
            std::chrono::duration_cast<std::chrono::nanoseconds>(now-phase_clock).count()/1e6);
        phase_clock=now;
    };
    // Phase one of a layer: everything queued to the GPU with no host
    // synchronization -- norms, DeltaNet/attention, the router and its result
    // downloads, the route event, and the shared expert. Dense feed-forward
    // layers complete entirely here (including their residual join) and return
    // false; a MoE layer returns true with its routed-expert work pending for
    // cpu_phase and queue_combine.
    auto queue_core=[&](RowsSpan&span,std::uint32_t layer_number)->bool{
        install_span(span);
        if(runtime.cancelled)throw std::runtime_error("native Qwen runtime is cancelled");
        auto&layer=runtime.layers[layer_number];
        q8_cached_input=0;q8_cached_width=0;
        if(profile_prefill&&colibri_gpu_event_record(runtime.prefill_layer_start_event,runtime.stream)!=0)
            throw std::runtime_error("native Qwen prefill layer-start profiling event failed");
        auto tensor=[&](std::size_t role){return runtime.device_tensors[layer.static_tensors.at(role)];};
        auto dense=[&](std::size_t role,std::uint64_t input,std::uint64_t output,int input_size,int output_size,std::uint64_t staged=0){dense_rows(layer.static_tensors.at(role),input,output,input_size,output_size,staged);};
        phase(layer_number,"pre");
        if(runtime.qwen4exp){
            if(layer.ple_conv!=std::numeric_limits<std::uint64_t>::max())
                ple_rows(layer);
            hc_pre_rows(tensor(0),layer.hc_attn_down,layer.hc_attn_up,
                        layer.hc_attn_inject);
        }else rms_rows(hidden,tensor(0),normalized);
        std::size_t moe_base=0;
        if(!layer.attention){
            int channels=static_cast<int>(runtime.model->tensors[layer.static_tensors[1]].shape[1]);
            int gate_elements=static_cast<int>(runtime.model->tensors[layer.static_tensors[2]].shape[1]);
            int value_heads=static_cast<int>(runtime.model->tensors[layer.static_tensors[8]].shape[0]);
            int head_dim=static_cast<int>(runtime.model->tensors[layer.static_tensors[9]].shape[0]);
            int value_dim=value_heads*head_dim;
            int key_heads=(channels-value_dim)/(2*head_dim);
            // MTP fold capture: the qkv projection and the decay/beta logits
            // are the recurrence's only sequence-dependent inputs, so during
            // the batched verify they are written to the layer's retention
            // arena instead of the shared scratch. A rejected round then
            // rebuilds the DeltaNet state by replaying just the conv and
            // recurrence kernels over the accepted prefix from the restored
            // snapshot (qwen_mtp_fold_delta_state) -- bitwise identical to the
            // full re-forward it replaces, because it launches the same
            // kernels on the same inputs. The redirect costs no extra copies.
            const bool fold_capture=
                runtime.mtp_fold_capture&&span_begin==0&&rows<=8;
            if(runtime.mtp_fold_capture&&!fold_capture)
                runtime.mtp_fold_valid=false;
            const auto qkv=fold_capture
                ?runtime.state+layer.mtp_retain_qkv:first;
            const auto decay_logits=fold_capture
                ?runtime.state+layer.mtp_retain_logits:third;
            dense(1,normalized,qkv,hidden_size,channels);
            dense(2,normalized,second,hidden_size,gate_elements);
            dense(4,normalized,decay_logits,hidden_size,value_heads);
            auto beta_logits=decay_logits+static_cast<std::uint64_t>(rows)*value_heads*sizeof(float);
            dense(5,normalized,beta_logits,hidden_size,value_heads);
            int kernel_size=static_cast<int>(runtime.model->tensors[layer.static_tensors[6]].shape[0]);
            auto conv_state=runtime.state+layer.state_first;auto conv_weights=tensor(6);
            void*conv_args[]={const_cast<std::uint64_t*>(&qkv),&conv_weights,&conv_state,const_cast<std::uint64_t*>(&fourth),&rows,&channels,&kernel_size};
            phase(layer_number,"proj");
            if(kernel_size<=8)launch("delta_conv_chunk",(static_cast<std::uint32_t>(channels)+255)/256,1,256,conv_args);
            else launch("delta_conv_sequence",channels,1,1,conv_args);
            phase(layer_number,"conv");
            auto recurrent_state=runtime.state+layer.state_second;auto decay=tensor(8),dt=tensor(7),norm=tensor(9);
            // COLIBRI_DELTA_DUMP=<dir> writes one DeltaNet layer's recurrence
            // inputs so alternative kernels can be replayed on real activations
            // instead of synthetic ones. Diagnostic only: it dumps the first
            // such layer once and never alters what runs.
            if(const char*dump_dir=std::getenv("COLIBRI_DELTA_DUMP")){
                // COLIBRI_DELTA_DUMP_SKIP skips the first N opportunities. The
                // default first one is layer 0 of the first chunk, whose incoming
                // recurrent state is still all zeros; skipping past it is what
                // captures a live state carried in from earlier tokens.
                static long skip=[]{
                    const char*value=std::getenv("COLIBRI_DELTA_DUMP_SKIP");
                    return value?std::strtol(value,nullptr,10):0L;
                }();
                static bool dumped=false;
                if(!dumped&&skip-->0){
                    // fall through without dumping
                }else if(!dumped){
                    dumped=true;
                    colibri_gpu_stream_sync(runtime.stream);
                    auto save=[&](const char*what,std::uint64_t address,std::size_t count){
                        std::vector<float> host(count);
                        if(colibri_gpu_download(host.data(),address,count*sizeof(float),runtime.stream)!=0
                           ||colibri_gpu_stream_sync(runtime.stream)!=0)
                            throw std::runtime_error("native Qwen DeltaNet dump readback failed");
                        char path[1024];
                        std::snprintf(path,sizeof(path),"%s/delta_%s.f32",dump_dir,what);
                        std::FILE*file=std::fopen(path,"wb");
                        if(!file)throw std::runtime_error("native Qwen DeltaNet dump cannot open "+std::string(path));
                        std::fwrite(host.data(),sizeof(float),count,file);
                        std::fclose(file);
                    };
                    const std::size_t token_count=static_cast<std::size_t>(rows);
                    save("convolved",fourth,token_count*static_cast<std::size_t>(channels));
                    save("gates",second,token_count*static_cast<std::size_t>(gate_elements));
                    save("decay_logits",decay_logits,token_count*static_cast<std::size_t>(value_heads));
                    save("beta_logits",beta_logits,token_count*static_cast<std::size_t>(value_heads));
                    save("state",recurrent_state,
                         static_cast<std::size_t>(value_heads)*head_dim*head_dim);
                    save("a_log",decay,static_cast<std::size_t>(value_heads));
                    save("dt_bias",dt,static_cast<std::size_t>(value_heads));
                    save("norm",norm,static_cast<std::size_t>(head_dim));
                    char meta[1024];
                    std::snprintf(meta,sizeof(meta),"%s/delta_meta.txt",dump_dir);
                    if(std::FILE*file=std::fopen(meta,"w")){
                        std::fprintf(file,
                            "layer %u\nrows %d\nchannels %d\ngate_elements %d\n"
                            "key_heads %d\nvalue_heads %d\nhead_dim %d\nepsilon %.9g\n",
                            layer_number,rows,channels,gate_elements,
                            key_heads,value_heads,head_dim,epsilon);
                        std::fclose(file);
                    }
                    std::fprintf(stderr,"[delta-dump] layer %u rows %d -> %s\n",
                        layer_number,rows,dump_dir);
                }
            }
            int gate_sigmoid=runtime.qwen4exp?1:0;
            void*recurrent_args[]={const_cast<std::uint64_t*>(&fourth),const_cast<std::uint64_t*>(&second),&beta_logits,const_cast<std::uint64_t*>(&decay_logits),&decay,&dt,&norm,&recurrent_state,const_cast<std::uint64_t*>(&first),&rows,&key_heads,&value_heads,&head_dim,const_cast<float*>(&epsilon),&gate_sigmoid};
            // Chunked WY form: the same recurrence with the serial depth cut from
            // `rows` steps to rows/64 and the grid widened past `value_heads`.
            // Requires the workspace regions sized at prepare time, so the guard
            // is the same head_dim test that sized them.
            const bool chunked_delta=
                colibri::v2::workspace::use_chunked_delta(
                    static_cast<std::uint64_t>(rows),
                    static_cast<std::uint64_t>(head_dim),
                    static_cast<std::uint64_t>(value_heads),
                    runtime.delta_value_heads)&&
                !std::getenv("COLIBRI_DELTA_SEQUENTIAL");
            if(chunked_delta){
                const auto attn=workspace_layout.delta_attn.address(runtime.workspace);
                const auto pmat=workspace_layout.delta_pmat.address(runtime.workspace);
                const auto gcum=workspace_layout.delta_gcum.address(runtime.workspace);
                const auto beta_out=workspace_layout.delta_beta.address(runtime.workspace);
                const auto qinv=workspace_layout.delta_qinv.address(runtime.workspace);
                const auto kinv=workspace_layout.delta_kinv.address(runtime.workspace);
                const auto w_rows=workspace_layout.delta_w.address(runtime.workspace);
                const auto u_rows=workspace_layout.delta_u.address(runtime.workspace);
                // `first` held the qkv projection, which the causal conv has
                // already consumed into `fourth`; the epilogue then normalizes it
                // in place, which is where the sequential kernel leaves its output.
                const auto core=first;
                const std::uint32_t chunk_count=
                    (static_cast<std::uint32_t>(rows)
                     +static_cast<std::uint32_t>(colibri::v2::workspace::kDeltaChunk)-1)
                    /static_cast<std::uint32_t>(colibri::v2::workspace::kDeltaChunk);
                void*score_args[]={const_cast<std::uint64_t*>(&fourth),&beta_logits,
                    const_cast<std::uint64_t*>(&decay_logits),&decay,&dt,
                    const_cast<std::uint64_t*>(&attn),const_cast<std::uint64_t*>(&pmat),
                    const_cast<std::uint64_t*>(&gcum),const_cast<std::uint64_t*>(&beta_out),
                    const_cast<std::uint64_t*>(&qinv),const_cast<std::uint64_t*>(&kinv),
                    &rows,&key_heads,&value_heads};
                launch("qwen_delta_wy_scores",chunk_count,
                    static_cast<std::uint32_t>(value_heads),256,score_args);
                void*solve_args[]={const_cast<std::uint64_t*>(&fourth),
                    const_cast<std::uint64_t*>(&attn),const_cast<std::uint64_t*>(&gcum),
                    const_cast<std::uint64_t*>(&beta_out),const_cast<std::uint64_t*>(&kinv),
                    const_cast<std::uint64_t*>(&w_rows),const_cast<std::uint64_t*>(&u_rows),
                    &rows,&key_heads,&value_heads};
                launch("qwen_delta_wy_solve",chunk_count,
                    static_cast<std::uint32_t>(value_heads),256,solve_args);
                void*pass_args[]={const_cast<std::uint64_t*>(&fourth),
                    const_cast<std::uint64_t*>(&pmat),const_cast<std::uint64_t*>(&gcum),
                    const_cast<std::uint64_t*>(&qinv),const_cast<std::uint64_t*>(&kinv),
                    const_cast<std::uint64_t*>(&w_rows),const_cast<std::uint64_t*>(&u_rows),
                    &recurrent_state,const_cast<std::uint64_t*>(&core),
                    &rows,&key_heads,&value_heads};
                launch("qwen_delta_state_pass",static_cast<std::uint32_t>(value_heads),
                    static_cast<std::uint32_t>(colibri::v2::workspace::kDeltaDim)/32,
                    256,pass_args);
                void*gate_args[]={const_cast<std::uint64_t*>(&core),
                    const_cast<std::uint64_t*>(&second),&norm,
                    const_cast<std::uint64_t*>(&first),&value_heads,
                    const_cast<float*>(&epsilon),&gate_sigmoid};
                launch("qwen_delta_norm_gate",static_cast<std::uint32_t>(rows),
                    static_cast<std::uint32_t>(value_heads),
                    static_cast<std::uint32_t>(colibri::v2::workspace::kDeltaDim),
                    gate_args);
            }else if(head_dim==128)launch("qwen_delta_recurrent_chunk",value_heads,1,128,recurrent_args);
            else launch("qwen_delta_recurrent_rows",value_heads,1,256,recurrent_args);
            phase(layer_number,"recurrent");
            dense(3,first,residual,value_dim,hidden_size);
            if(runtime.qwen4exp)hc_post_rows(residual);
            else add_rows(residual,hidden);
            moe_base=10;
            phase(layer_number,"outproj");
        }else{
            // Laguna projects the query plainly and carries the attention gate
            // in its own n_head-wide tensor, where Qwen interleaves a per-channel
            // gate into the query projection. Head count and RoPE are per layer.
            const bool laguna=runtime.laguna;
            // Muse Glimmer projects the query plainly like Laguna, but its gate
            // is per-channel like Qwen's, so it takes Laguna's projection shape
            // and Qwen's gate kernels.
            const bool muse=runtime.muse;
            const int heads=laguna
                ?static_cast<int>(layer.attention_heads)
                :static_cast<int>(runtime.model->config.attention_heads);
            const int kv_heads=static_cast<int>(runtime.model->config.attention_kv_heads);
            const int head_dim=static_cast<int>(runtime.model->tensors[layer.static_tensors[2]].shape[1]/kv_heads);
            const int query_elements=heads*head_dim;
            const int q_size=(laguna||muse)?query_elements:query_elements*2;
            const int kv_size=kv_heads*head_dim;
            dense(1,normalized,first,hidden_size,q_size);
            dense(2,normalized,second,hidden_size,kv_size);
            dense(3,normalized,third,hidden_size,kv_size);
            // Both per-layer architectures carry their own rotary span and
            // theta. For Muse Glimmer the span is zero on full-attention
            // layers, which is how those run NoPE.
            const int rotary=(laguna||muse)
                ?static_cast<int>(layer.rotary_dim)
                :static_cast<int>(runtime.model->config.rotary_dimension?runtime.model->config.rotary_dimension:head_dim);
            const float theta=(laguna||muse)
                ?layer.rope_theta
                :(runtime.model->config.rope_freq_base?runtime.model->config.rope_freq_base:1000000.0f);
            // Laguna's full-attention layers extend with YaRN; its sliding-window
            // layers and every Qwen layer leave ext_factor at zero and so run the
            // identical plain rotation.
            const float freq_scale=laguna?layer.rope_freq_scale:1.0f;
            const float ext_factor=laguna?layer.rope_ext_factor:0.0f;
            const float mscale=laguna?layer.rope_attn_factor:1.0f;
            float corr_low=0.0f,corr_high=0.0f;
            if(ext_factor!=0.0f)
                qwen_yarn_correction_dims(
                    rotary,layer.rope_orig_context,theta,
                    layer.rope_beta_fast,layer.rope_beta_slow,corr_low,corr_high);
            auto qnorm=tensor(5),knorm=tensor(6);
            const auto queries=fourth;
            // One scalar per head for Laguna, one per query channel for Qwen.
            const auto gates=fourth+static_cast<std::uint64_t>(rows)*query_elements*sizeof(float);
            // Laguna's gate is one scalar per head, Muse Glimmer's one per
            // query channel; both come from a tensor of their own rather than
            // riding in the query projection the way Qwen's does.
            if(laguna)dense(7,normalized,gates,hidden_size,heads);
            else if(muse)dense(7,normalized,gates,hidden_size,query_elements);
            auto cache_keys=runtime.state+layer.state_first,cache_values=runtime.state+layer.state_second;
            int capacity=static_cast<int>(layer.cache_capacity);
            phase(layer_number,"qkvproj");
            for(int token=0;token<rows;++token){
                auto projected=first+static_cast<std::uint64_t>(token)*q_size*sizeof(float);
                auto query=queries+static_cast<std::uint64_t>(token)*query_elements*sizeof(float);
                auto gate=gates+static_cast<std::uint64_t>(token)*
                    (laguna?heads:query_elements)*sizeof(float);
                int position=static_cast<int>(runtime.position)+span_begin+token;
                auto projected_key=second+static_cast<std::uint64_t>(token)*kv_size*sizeof(float);
                auto key=first+static_cast<std::uint64_t>(token)*kv_size*sizeof(float);
                if(laguna){
                    void*q_args[]={&projected,&qnorm,&query,const_cast<int*>(&heads),
                                   const_cast<int*>(&head_dim),const_cast<int*>(&rotary),
                                   &position,const_cast<float*>(&theta),const_cast<float*>(&epsilon),
                                   const_cast<float*>(&freq_scale),const_cast<float*>(&ext_factor),
                                   const_cast<float*>(&mscale),&corr_low,&corr_high};
                    launch("laguna_head_norm_rope",heads,1,256,q_args);
                    void*k_args[]={&projected_key,&knorm,&key,const_cast<int*>(&kv_heads),
                                   const_cast<int*>(&head_dim),const_cast<int*>(&rotary),
                                   &position,const_cast<float*>(&theta),const_cast<float*>(&epsilon),
                                   const_cast<float*>(&freq_scale),const_cast<float*>(&ext_factor),
                                   const_cast<float*>(&mscale),&corr_low,&corr_high};
                    launch("laguna_head_norm_rope",kv_heads,1,256,k_args);
                }else if(muse){
                    // The interleaved-rope kernel is per token already, so the
                    // row batch reuses it unchanged.
                    void*q_args[]={&projected,&qnorm,&query,const_cast<int*>(&heads),
                                   const_cast<int*>(&head_dim),const_cast<int*>(&rotary),
                                   &position,const_cast<float*>(&theta),
                                   const_cast<float*>(&epsilon)};
                    launch("muse_head_norm_rope",heads,1,256,q_args);
                    void*k_args[]={&projected_key,&knorm,&key,const_cast<int*>(&kv_heads),
                                   const_cast<int*>(&head_dim),const_cast<int*>(&rotary),
                                   &position,const_cast<float*>(&theta),
                                   const_cast<float*>(&epsilon)};
                    launch("muse_head_norm_rope",kv_heads,1,256,k_args);
                }else{
                    void*q_args[]={&projected,&qnorm,&query,&gate,const_cast<int*>(&heads),const_cast<int*>(&head_dim),const_cast<int*>(&rotary),&position,const_cast<float*>(&theta),const_cast<float*>(&epsilon)};
                    launch("qwen_attention_query",heads,1,256,q_args);
                    void*k_args[]={&projected_key,&knorm,&key,const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),const_cast<int*>(&rotary),&position,const_cast<float*>(&theta),const_cast<float*>(&epsilon)};
                    launch("qwen_attention_key",kv_heads,1,256,k_args);
                }
                auto value=third+static_cast<std::uint64_t>(token)*kv_size*sizeof(float);
                int slot=static_cast<int>(static_cast<std::uint64_t>(position)%layer.cache_capacity);
                void*k_store_args[]={&key,&cache_keys,const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&slot,&capacity};
                launch(kv_store_kernel(runtime.options.cache_type_k,true),kv_heads,1,256,k_store_args);
                void*v_store_args[]={&value,&cache_values,const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&slot,&capacity};
                launch(kv_store_kernel(runtime.options.cache_type_v,false),kv_heads,1,256,v_store_args);
            }
            phase(layer_number,"ropestore");
            // QSA indexer upkeep for the whole span: batched raw-key
            // projection, then per-row ring writes + block closes (blocks
            // straddling the chunk boundary read the older raw keys from the
            // ring the previous chunk left behind). Runs from position 0 so
            // the block-key store is complete when selection first fires.
            const bool qsa_layer_on=qwen_qsa_layer(runtime,layer);
            const int qsa_heads=qsa_layer_on
                ?static_cast<int>(runtime.model->config.indexer_head_count):0;
            const int qsa_key_len=qsa_layer_on
                ?static_cast<int>(runtime.model->config.indexer_key_length):0;
            const int qsa_ratio=qsa_layer_on?static_cast<int>(layer.qsa_ratio):0;
            if(qsa_layer_on){
                const int qsa_rotary=std::min(rotary,qsa_key_len);
                std::uint64_t ring=runtime.state+layer.qsa_ring;
                std::uint64_t blocks=runtime.state+layer.qsa_blocks;
                auto knorm_w=runtime.device_tensors[layer.indexer_k_norm];
                dense_rows(layer.indexer_k,normalized,qsa_keys_base,
                           hidden_size,qsa_key_len);
                for(int token=0;token<rows;++token){
                    auto raw=qsa_keys_base+static_cast<std::uint64_t>(token)*qsa_key_len*sizeof(float);
                    int position=static_cast<int>(runtime.position)+span_begin+token;
                    if(!qwen_qsa_in_range(layer,static_cast<std::uint64_t>(position)))break;
                    void*store_args[]={&raw,&ring,&blocks,&knorm_w,&position,
                        const_cast<int*>(&qsa_ratio),const_cast<int*>(&qsa_key_len),
                        const_cast<int*>(&qsa_rotary),const_cast<float*>(&theta),
                        const_cast<float*>(&epsilon)};
                    launch("qsa_key_store",1,1,256,store_args);
                }
            }
            // Selection is per row; the chunk leaves the batched attention
            // tiers as soon as ANY of its rows prunes, because those tiers
            // compute dense attention for every row.
            const bool qsa_chunk_selecting=qsa_layer_on&&rows>0&&
                qwen_qsa_selecting(runtime,layer,
                    runtime.position+static_cast<std::uint64_t>(span_begin)+rows-1);
            float scale=1.0f/std::sqrt(static_cast<float>(head_dim));
            const char*cublas_prefill_env=
                std::getenv("COLIBRI_CUBLAS_PREFILL_ATTENTION");
            const std::uint64_t max_tokens=
                runtime.position+static_cast<std::uint64_t>(span_begin)+
                static_cast<std::uint64_t>(rows);
            // The warp-online kernel wins for short prefixes because it needs
            // one launch and no materialized score tile.  Once the visible
            // prefix reaches 4K, tensor-core QK/PV amortize the tiled cuBLAS
            // setup and pull progressively farther ahead.  1/0 force either
            // side for benchmarking and machine-specific tuning.
            const bool cublas_prefill_forced=
                cublas_prefill_env&&cublas_prefill_env[0]=='1';
            const bool cublas_prefill_disabled=
                cublas_prefill_env&&cublas_prefill_env[0]=='0';
            const bool cublas_prefill_requested=
                !cublas_prefill_disabled&&
                (cublas_prefill_forced||max_tokens>=4096);
            // Flash-blocked: the score tile is bounded by a KV position block
            // rather than the whole visible prefix, so the 16-row query tile
            // no longer shrinks as the context grows (it collapsed to 3 rows
            // at 70k, streaming the entire cache once per 3 queries -- the
            // long-context prefill decline). The block is whatever fits the
            // same score workspace at 16 rows.
            const int cublas_group=kv_heads?heads/kv_heads:0;
            const std::uint64_t score_workspace_bytes=
                static_cast<std::uint64_t>(rows)*runtime.scratch_elements*sizeof(float);
            const std::uint64_t block_capacity=cublas_group
                ?score_workspace_bytes/
                    (16ULL*cublas_group*sizeof(float)):0;
            const int cublas_block_tokens=static_cast<int>(
                std::min<std::uint64_t>(block_capacity,max_tokens));
            const int cublas_tile_rows=cublas_block_tokens>=256?16:0;
            const std::uint64_t packed_query_bytes=device_align(
                static_cast<std::uint64_t>(16)*heads*head_dim*
                sizeof(std::uint16_t));
            const std::uint64_t packed_output_bytes=device_align(
                static_cast<std::uint64_t>(16)*heads*head_dim*sizeof(float));
            const std::uint64_t flash_state_bytes=device_align(
                static_cast<std::uint64_t>(16)*heads*3*sizeof(float));
            const auto flash_state=
                attention_scores+packed_query_bytes+packed_output_bytes;
            const bool cublas_workspace_ok=
                packed_query_bytes+packed_output_bytes+flash_state_bytes<=
                static_cast<std::uint64_t>(heads)*
                    runtime.options.context_limit*sizeof(float);
            // The tensor-core prefill routines fold Qwen's per-channel sigmoid
            // gate in themselves, so Laguna, whose gate is a per-head softplus,
            // cannot use them and takes the warp path with its own gate kernel.
            const bool cublas_prefill_done=
                !qsa_chunk_selecting&&
                cublas_prefill_requested&&!laguna&&!layer.attention_window&&
                runtime.options.cache_type_k==1&&
                runtime.options.cache_type_v==1&&cublas_tile_rows>0&&
                cublas_workspace_ok&&
                colibri_gpu_attention_prefill_f16_cublas(
                    queries,gates,cache_keys,cache_values,
                    attention_scores,first,second,
                    attention_scores+packed_query_bytes,flash_state,third,
                    runtime.stream,heads,kv_heads,head_dim,rows,capacity,
                    static_cast<int>(runtime.position)+span_begin,
                    cublas_tile_rows,cublas_block_tokens,scale,1)==0;
            // COLIBRI_ATTN_PREFILL_TRACE=1: which attention path actually ran,
            // printed for the first few layers. The cuBLAS attempt fails
            // silently by design (missing library, kernel, workspace), so a
            // fallback is invisible without this -- and a dormant tensor-core
            // path looks exactly like a slow one.
            if(std::getenv("COLIBRI_ATTN_PREFILL_TRACE")){
                static int shown=0;
                if(shown<4){++shown;
                    std::fprintf(stderr,
                        "[attn-prefill] layer=%u cublas=%d requested=%d tile=%d block=%d rows=%d\n",
                        layer_number,(int)cublas_prefill_done,
                        (int)cublas_prefill_requested,cublas_tile_rows,
                        cublas_block_tokens,rows);}
            }
            if(qsa_chunk_selecting){
                // Per-row sparse attention: rows past the indexer budget score
                // the block keys, keep the top budget/ratio blocks + tail, and
                // attend over that slot list; rows still below it take the
                // dense ring pair -- identical math to the batched tiers.
                dense_rows(layer.indexer_q,normalized,qsa_query_base,
                           hidden_size,qsa_heads*qsa_key_len);
                const int qsa_rotary=std::min(rotary,qsa_key_len);
                auto qnorm_w=runtime.device_tensors[layer.indexer_q_norm];
                std::uint64_t blocks=runtime.state+layer.qsa_blocks;
                for(int token=0;token<rows;++token){
                    auto query=queries+static_cast<std::uint64_t>(token)*query_elements*sizeof(float);
                    auto gate=gates+static_cast<std::uint64_t>(token)*query_elements*sizeof(float);
                    int position=static_cast<int>(runtime.position)+span_begin+token;
                    auto attended=second+static_cast<std::uint64_t>(token)*query_elements*sizeof(float);
                    int n_sel=0;
                    if(qwen_qsa_selecting(runtime,layer,static_cast<std::uint64_t>(position))){
                        const int complete=(position+1)/qsa_ratio;
                        auto projected_q=qsa_query_base+static_cast<std::uint64_t>(token)*qsa_heads*qsa_key_len*sizeof(float);
                        // `first` is free per row (its roped key went to the
                        // cache in the store loop); it takes the normed+roped
                        // indexer queries.
                        auto prepped=first+static_cast<std::uint64_t>(token)*q_size*sizeof(float);
                        void*prep_args[]={&projected_q,&qnorm_w,&prepped,
                            const_cast<int*>(&qsa_heads),const_cast<int*>(&qsa_key_len),
                            const_cast<int*>(&qsa_rotary),&position,
                            const_cast<float*>(&theta),const_cast<float*>(&epsilon)};
                        launch("qwen_attention_key",qsa_heads,1,256,prep_args);
                        void*bscore_args[]={&prepped,&blocks,
                            const_cast<std::uint64_t*>(&qsa_scores_base),
                            const_cast<int*>(&complete),const_cast<int*>(&qsa_heads),
                            const_cast<int*>(&qsa_key_len)};
                        launch("qsa_block_scores",
                            (static_cast<std::uint32_t>(complete)+255)/256,1,256,bscore_args);
                        n_sel=qwen_qsa_select(runtime,layer,qsa_scores_base,
                            qsa_slots_base,static_cast<std::uint64_t>(position));
                    }
                    if(n_sel){
                        void*score_args[]={&query,&cache_keys,
                            const_cast<std::uint64_t*>(&qsa_slots_base),
                            const_cast<std::uint64_t*>(&attention_scores),
                            const_cast<int*>(&heads),const_cast<int*>(&kv_heads),
                            const_cast<int*>(&head_dim),&n_sel,&capacity,&scale};
                        launch(kv_scores_indexed_kernel(runtime),heads,
                            (static_cast<std::uint32_t>(n_sel)+255)/256,256,score_args);
                        void*value_args[]={const_cast<std::uint64_t*>(&attention_scores),
                            &cache_values,const_cast<std::uint64_t*>(&qsa_slots_base),
                            &attended,const_cast<int*>(&heads),const_cast<int*>(&kv_heads),
                            const_cast<int*>(&head_dim),&n_sel,&capacity};
                        launch(kv_values_indexed_kernel(runtime),heads,1,256,value_args);
                    }else{
                        const auto view=attention_cache_view(layer,static_cast<std::uint64_t>(position));
                        int tokens=view.tokens,first_slot=view.first;
                        void*score_args[]={&query,&cache_keys,const_cast<std::uint64_t*>(&attention_scores),const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&tokens,&capacity,&first_slot,&scale};
                        launch(kv_scores_ring_kernel(runtime),heads,(tokens+255)/256,256,score_args);
                        void*value_args[]={const_cast<std::uint64_t*>(&attention_scores),&cache_values,&attended,const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&tokens,&capacity,&first_slot};
                        launch(kv_values_ring_kernel(runtime),heads,1,256,value_args);
                    }
                    auto gated=third+static_cast<std::uint64_t>(token)*query_elements*sizeof(float);
                    int elements=query_elements;
                    void*token_gate_args[]={&attended,&gate,&gated,&elements};
                    launch("qwen_attention_gate",(elements+255)/256,1,256,token_gate_args);
                }
            }else if(cublas_prefill_done){
                // Tensor-core attention wrote the already-gated rows to third.
            }else if(!layer.attention_window&&head_dim%32==0&&head_dim<=256&&kv_fused_prefill_ok(runtime)){
                // One warp streams the KV cache once for 4 query rows: cache
                // traffic per chunk stops scaling with per-token passes, which
                // dominated long-context prefill.
                int base=static_cast<int>(runtime.position)+span_begin;
                void*attention_args[]={const_cast<std::uint64_t*>(&queries),&cache_keys,&cache_values,const_cast<std::uint64_t*>(&second),const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&base,&rows,&capacity,&scale};
                launch(kv_prefill_kernel(runtime),heads,(static_cast<std::uint32_t>(rows)+31)/32,256,attention_args);
                int elements=rows*query_elements;
                if(laguna){
                    void*gate_args[]={const_cast<std::uint64_t*>(&second),const_cast<std::uint64_t*>(&gates),const_cast<std::uint64_t*>(&third),&rows,const_cast<int*>(&heads),const_cast<int*>(&head_dim)};
                    launch("laguna_attention_gate_rows",(elements+255)/256,1,256,gate_args);
                }else{
                    void*gate_args[]={const_cast<std::uint64_t*>(&second),const_cast<std::uint64_t*>(&gates),const_cast<std::uint64_t*>(&third),&elements};
                    launch("qwen_attention_gate",(elements+255)/256,1,256,gate_args);
                }
            }else if(std::uint64_t stage_keys=0,stage_values=0;
                     !laguna&&
                     qwen_turbo_prefill_stage(
                         runtime,layer,queries,cache_keys,cache_values,
                         heads,kv_heads,head_dim,rows,
                         static_cast<int>(runtime.position)+span_begin,
                         stage_keys,stage_values)){
                // The staged window is dense and ring-free, so it is indexed
                // with capacity == window. cuBLAS runs ungated here because the
                // values are still rotated: the gate is a sigmoid and has to
                // wait until after the inverse rotation below.
                const int window=static_cast<int>(runtime.position)+span_begin+rows;
                int base=static_cast<int>(runtime.position)+span_begin,staged_capacity=window;
                int vectors=rows*heads,value_stream=1,elements=rows*query_elements;
                const bool staged_cublas=
                    cublas_prefill_requested&&cublas_tile_rows>0&&
                    cublas_workspace_ok&&
                    colibri_gpu_attention_prefill_f16_cublas(
                        queries,gates,stage_keys,stage_values,
                        attention_scores,first,second,
                        attention_scores+packed_query_bytes,flash_state,third,
                        runtime.stream,heads,kv_heads,head_dim,rows,window,
                        base,cublas_tile_rows,cublas_block_tokens,scale,0)==0;
                if(!staged_cublas){
                    void*attention_args[]={const_cast<std::uint64_t*>(&queries),&stage_keys,&stage_values,const_cast<std::uint64_t*>(&third),const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&base,&rows,&staged_capacity,&scale};
                    launch("kv_attention_prefill_f16",heads,(static_cast<std::uint32_t>(rows)+31)/32,256,attention_args);
                }
                void*unrotate_args[]={const_cast<std::uint64_t*>(&third),&vectors,const_cast<int*>(&head_dim),&value_stream};
                launch("turbo_unrotate_rows",static_cast<std::uint32_t>(vectors),1,256,unrotate_args);
                void*gate_args[]={const_cast<std::uint64_t*>(&third),const_cast<std::uint64_t*>(&gates),const_cast<std::uint64_t*>(&third),&elements};
                launch("qwen_attention_gate",(elements+255)/256,1,256,gate_args);
            }else for(int token=0;token<rows;++token){
                auto query=queries+static_cast<std::uint64_t>(token)*query_elements*sizeof(float);
                auto gate=gates+static_cast<std::uint64_t>(token)*
                    (laguna?heads:query_elements)*sizeof(float);
                int position=static_cast<int>(runtime.position)+span_begin+token;
                const auto view=attention_cache_view(layer,static_cast<std::uint64_t>(position));
                int tokens=view.tokens,first_slot=view.first;
                void*score_args[]={&query,&cache_keys,const_cast<std::uint64_t*>(&attention_scores),const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&tokens,&capacity,&first_slot,&scale};
                launch(kv_scores_ring_kernel(runtime),heads,(tokens+255)/256,256,score_args);
                auto attended=second+static_cast<std::uint64_t>(token)*query_elements*sizeof(float);
                void*value_args[]={const_cast<std::uint64_t*>(&attention_scores),&cache_values,&attended,const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&tokens,&capacity,&first_slot};
                launch(kv_values_ring_kernel(runtime),heads,1,256,value_args);
                auto gated=third+static_cast<std::uint64_t>(token)*query_elements*sizeof(float);int elements=query_elements;
                if(laguna){
                    void*token_gate_args[]={&attended,&gate,&gated,
                                            const_cast<int*>(&heads),const_cast<int*>(&head_dim)};
                    launch("laguna_attention_gate",(elements+255)/256,1,256,token_gate_args);
                }else{
                    void*token_gate_args[]={&attended,&gate,&gated,&elements};launch("qwen_attention_gate",(elements+255)/256,1,256,token_gate_args);
                }
            }
            dense(4,third,residual,query_elements,hidden_size);
            // Muse Glimmer normalizes the attention output before the residual
            // join rather than adding it raw.
            if(muse)post_rms_rows(residual,tensor(8));
            if(runtime.qwen4exp)hc_post_rows(residual);
            else add_rows(residual,hidden);
            moe_base=qwen_ffn_base(runtime,layer);
        }
        if(profile_prefill&&colibri_gpu_event_record(runtime.prefill_core_end_event,runtime.stream)!=0)
            throw std::runtime_error("native Qwen prefill core profiling event failed");
        if(runtime.qwen4exp)
            hc_pre_rows(tensor(moe_base),layer.hc_ffn_down,layer.hc_ffn_up,
                        layer.hc_ffn_inject);
        else rms_rows(residual,tensor(moe_base),normalized);
        phase(layer_number,"moerms");
        if(layer.dense_ffn){
            // Dense block over the whole row batch: one SwiGLU per token, no
            // router or experts. `first` holds gate and up back to back.
            // Laguna's leading dense blocks are far wider than its expert
            // blocks, so this width comes from the block's own gate projection.
            const int dense_intermediate=runtime.laguna
                ?static_cast<int>(runtime.model->tensors[layer.static_tensors.at(moe_base+1)].shape[1])
                :static_cast<int>(runtime.moe_intermediate);
            // A spilled block still has to produce this batch's feed-forward.
            // Doing it on the host is right for one token and badly wrong for a
            // batch: the host pays rows x 3 x hidden x intermediate MACs, and on
            // the IQ formats it pays them through a scalar decoder, while the
            // device sits idle. Past a threshold it is cheaper to send the
            // weights the other way -- one upload of the block's triple, then
            // the ordinary device path -- because the upload is a fixed cost
            // per block per chunk while the host compute scales with rows.
            // Measured on this checkpoint at 16 of 64 blocks spilled: the host
            // path gives 17.6 tok/s of prefill against 536 resident.
            const bool stage_ffn=layer.ffn_on_host&&runtime.host_ffn_stage&&
                rows>=qwen_host_ffn_batch_threshold();
            if(stage_ffn){
                const std::size_t ffn_base=qwen_ffn_base(runtime,layer);
                std::uint64_t offset=0;
                std::uint64_t staged[4]={0,0,0,0};
                for(std::size_t role=1;role<=3;++role){
                    const auto index=layer.static_tensors[ffn_base+role];
                    const auto&tensor=runtime.model->tensors[index];
                    staged[role]=runtime.host_ffn_stage+offset;
                    if(colibri_gpu_upload(staged[role],
                                          tensor_data(*runtime.model,tensor),
                                          tensor.size,runtime.stream)!=0)
                        throw std::runtime_error(
                            "native Qwen staged feed-forward upload failed");
                    offset+=device_align(tensor.size);
                }
                const auto up_half=first+
                    static_cast<std::uint64_t>(rows)*dense_intermediate*sizeof(float);
                dense(moe_base+1,normalized,first,hidden_size,dense_intermediate,staged[1]);
                dense(moe_base+2,normalized,up_half,hidden_size,dense_intermediate,staged[2]);
                int staged_count=rows*dense_intermediate;
                void*staged_silu[]={const_cast<std::uint64_t*>(&first),
                                    const_cast<std::uint64_t*>(&second),&staged_count};
                launch("silu_mul",(static_cast<std::uint32_t>(staged_count)+255)/256,1,256,staged_silu);
                dense(moe_base+3,second,third,dense_intermediate,hidden_size,staged[3]);
                if(runtime.muse)post_rms_rows(third,tensor(13));
                add_rows(residual,third);std::swap(span.hidden,span.residual);
                return false;
            }
            if(layer.ffn_on_host){
                // Spilled block: its weights are not on the device at all, so
                // the SwiGLU runs on the host.
                //
                // The whole batch goes over at once. Row at a time -- with a
                // stream sync per row -- meant this block re-read all of its
                // weights per token while the device path around it read them
                // once per 64-token tile, so the spilled fraction did not
                // shrink as prefill batched and became the entire cost of a
                // token. It also serialized: the device idled through every
                // host row and the host idled through every copy.
                const std::uint64_t batch_bytes=
                    static_cast<std::uint64_t>(rows)*hidden_size*sizeof(float);
                auto*scratch=static_cast<float*>(runtime.dense_host);
                float*host_input=scratch;
                float*host_output=host_input+
                    static_cast<std::size_t>(rows)*hidden_size;
                const auto host_started=std::chrono::steady_clock::now();
                if(colibri_gpu_download(host_input,normalized,batch_bytes,runtime.stream)!=0||
                   colibri_gpu_stream_sync(runtime.stream)!=0)
                    throw std::runtime_error("native dense host FFN rows input transfer failed");
                qwen_cpu_dense_ffn_rows(runtime,layer,host_input,host_output,rows);
                if(colibri_gpu_upload(third,host_output,batch_bytes,runtime.stream)!=0)
                    throw std::runtime_error("native dense host FFN rows output transfer failed");
                runtime.dense_host_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now()-host_started).count();
                if(runtime.muse)post_rms_rows(third,tensor(13));
                add_rows(residual,third);std::swap(span.hidden,span.residual);
                return false;
            }
            const auto up_half=first+static_cast<std::uint64_t>(rows)*dense_intermediate*sizeof(float);
            dense(moe_base+1,normalized,first,hidden_size,dense_intermediate);
            dense(moe_base+2,normalized,up_half,hidden_size,dense_intermediate);
            int dense_count=rows*dense_intermediate;
            void*dense_silu_args[]={const_cast<std::uint64_t*>(&first),const_cast<std::uint64_t*>(&second),&dense_count};
            launch("silu_mul",(static_cast<std::uint32_t>(dense_count)+255)/256,1,256,dense_silu_args);
            dense(moe_base+3,second,third,dense_intermediate,hidden_size);
            // Post-FFN norm, mirroring the post-attention norm above.
            if(runtime.muse)post_rms_rows(third,tensor(13));
            add_rows(residual,third);std::swap(span.hidden,span.residual);
            return false;
        }
        dense(moe_base+1,normalized,router_logits,hidden_size,experts);
        phase(layer_number,"router");
        if(runtime.laguna){
            auto bias=runtime.device_tensors[layer.router_bias];
            int normalize=runtime.model->config.expert_weights_norm?1:0;
            float weight_scale=runtime.model->config.expert_weights_scale;
            void*route_args[]={const_cast<std::uint64_t*>(&router_logits),&bias,
                               const_cast<std::uint64_t*>(&selected_device),
                               const_cast<std::uint64_t*>(&route_weights),&rows,
                               const_cast<int*>(&experts),const_cast<int*>(&top_k),
                               &normalize,&weight_scale};
            launch("route_topk_sigmoid_bias_rows",rows,1,256,route_args,
                   static_cast<std::uint32_t>(2*experts*sizeof(float)));
        }else{
            void*route_args[]={const_cast<std::uint64_t*>(&router_logits),const_cast<std::uint64_t*>(&selected_device),const_cast<std::uint64_t*>(&route_weights),&rows,const_cast<int*>(&experts),const_cast<int*>(&top_k)};
            launch("route_topk_rows",rows,1,256,route_args,experts*sizeof(float));
        }
        if(profile_prefill&&colibri_gpu_event_record(runtime.prefill_router_end_event,runtime.stream)!=0)
            throw std::runtime_error("native Qwen prefill router profiling event failed");
        const auto selected_bytes=
            static_cast<std::uint64_t>(rows)*top_k*sizeof(std::int32_t);
        const auto weights_bytes=
            static_cast<std::uint64_t>(rows)*top_k*sizeof(float);
        const auto input_bytes=
            static_cast<std::uint64_t>(rows)*hidden_size*sizeof(float);
        if(colibri_gpu_download(selected_host,selected_device,selected_bytes,runtime.stream)!=0||colibri_gpu_download(weights_host,route_weights,weights_bytes,runtime.stream)!=0||colibri_gpu_download(input_host,normalized,input_bytes,runtime.stream)!=0)
            throw std::runtime_error("native Qwen rows forward route transfer failed");
        if(colibri_gpu_event_record(span.route_event,runtime.stream)!=0)throw std::runtime_error("native Qwen rows forward route event failed");
        // Shared expert gate/up as two tiled GEMMs into the halves of the
        // `first` scratch (silu_mul expects gate at [0, n) and up at [n, 2n)
        // of one buffer), then the elementwise swiglu combine.
        const auto swiglu_elements=static_cast<std::uint64_t>(rows)*intermediate;
        const auto shexp_type=runtime.model->tensors[layer.static_tensors.at(moe_base+2)].type;
        if(shexp_type==40){
            // NVFP4 shared expert. weight_scale_2 rides in f32 (it is far below
            // E4M3's smallest subnormal) and gate/up must be scaled before the
            // silu_mul below, because SiLU is non-linear.
            auto up_half=first+swiglu_elements*sizeof(float);
            auto nvfp4_rows=[&](std::size_t role,std::uint64_t input,std::uint64_t output,int input_size,int output_size,float scale){
                std::uint64_t matrix=runtime.device_tensors[layer.static_tensors.at(role)];
                const char* tensor_core_setting =
                    std::getenv("COLIBRI_NVFP4_TENSOR_CORES");
                const bool tensor_cores =
                    (!tensor_core_setting || tensor_core_setting[0] != '0')
                    && rows >= 16;
                if (tensor_cores &&
                    colibri_gpu_nvfp4_matmul_cublas(
                        matrix, input, output, runtime.stream, input_size,
                        output_size, rows, scale) == 0)
                    return;
                void*args[]={&matrix,&input,&output,&input_size,&output_size,&rows,&scale};
                launch("nvfp4_matmul_rows",output_size,rows,256,args);
            };
            nvfp4_rows(moe_base+2,normalized,first,hidden_size,intermediate,layer.shared_gate_scale);
            nvfp4_rows(moe_base+3,normalized,up_half,hidden_size,intermediate,layer.shared_up_scale);
            int count=static_cast<int>(swiglu_elements);
            void*silu_args[]={const_cast<std::uint64_t*>(&first),const_cast<std::uint64_t*>(&second),&count};
            launch("silu_mul",(static_cast<std::uint32_t>(count)+255)/256,1,256,silu_args);
            nvfp4_rows(moe_base+4,second,third,intermediate,hidden_size,layer.shared_down_scale);
        }else{
            dense(moe_base+2,normalized,first,hidden_size,intermediate);
            dense(moe_base+3,normalized,first+swiglu_elements*sizeof(float),hidden_size,intermediate);
            int count=static_cast<int>(swiglu_elements);
            void*silu_args[]={const_cast<std::uint64_t*>(&first),const_cast<std::uint64_t*>(&second),&count};
            launch("silu_mul",(static_cast<std::uint32_t>(count)+255)/256,1,256,silu_args);
            dense(moe_base+4,second,third,intermediate,hidden_size);
        }
        // Qwen gates the shared expert on a learned projection; Laguna's is
        // always on, so its block ends at the down projection.
        if(!runtime.laguna){
            auto shared_scale=tensor(moe_base+5);
            const auto sg_type=runtime.model->tensors[layer.static_tensors.at(moe_base+5)].type;
            void*scale_args[]={const_cast<std::uint64_t*>(&normalized),&shared_scale,const_cast<std::uint64_t*>(&third),&rows,const_cast<int*>(&hidden_size)};
            launch(sg_type==30?"qwen_shared_scale_rows_bf16":"qwen_shared_scale_rows",rows,1,256,scale_args);
        }
        return true;
    };
    // Phase two of a layer: the host's turn. Wait on the route event, prune
    // and record routes, build the GPU-resident expert tables (queueing their
    // uploads and kernels), then run the span's routed experts on the CPU.
    // The GPU meanwhile executes whatever work is already queued behind the
    // route downloads.
    auto cpu_phase=[&](RowsSpan&span,std::uint32_t layer_number){
        install_span(span);
        auto&layer=runtime.layers[layer_number];
        const auto route_wait_started=std::chrono::steady_clock::now();
        if(colibri_gpu_event_sync(span.route_event)!=0)throw std::runtime_error("native Qwen rows forward route synchronization failed");
        if(profile_prefill){
            float core_ms=0.0f,router_ms=0.0f,transfer_ms=0.0f;
            if(colibri_gpu_event_elapsed(runtime.prefill_layer_start_event,runtime.prefill_core_end_event,&core_ms)!=0||
               colibri_gpu_event_elapsed(runtime.prefill_core_end_event,runtime.prefill_router_end_event,&router_ms)!=0||
               colibri_gpu_event_elapsed(runtime.prefill_router_end_event,runtime.route_event,&transfer_ms)!=0)
                throw std::runtime_error("native Qwen prefill profiling measurement failed");
            ++runtime.prefill_gpu_split_layers;
            runtime.prefill_gpu_core_nanoseconds+=static_cast<std::uint64_t>(core_ms*1.0e6f);
            runtime.prefill_gpu_router_nanoseconds+=static_cast<std::uint64_t>(router_ms*1.0e6f);
            runtime.prefill_gpu_transfer_nanoseconds+=static_cast<std::uint64_t>(transfer_ms*1.0e6f);
        }
        const auto route_wait_elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-route_wait_started).count();
        runtime.route_wait_nanoseconds+=route_wait_elapsed;
        if(expert_policy.records_prefill_frequency())
            runtime.prefill_route_wait_nanoseconds+=route_wait_elapsed;
        static const bool trace_layers=std::getenv("COLIBRI_PREFILL_TRACE")!=nullptr;
        if(trace_layers)std::fprintf(stderr,"[trace] layer %2u %s rows=%d sync=%.1fms\n",layer_number,layer.attention?"attn ":"delta",rows,route_wait_elapsed/1e6);
        span.pager_started=std::chrono::steady_clock::now();
        if(prune_experts){
            for(int token=0;token<rows;++token){
                const int keep=apply_expert_router_policy(
                    selected_host+static_cast<std::size_t>(token)*top_k,
                    weights_host+static_cast<std::size_t>(token)*top_k,
                    top_k,static_cast<int>(runtime.options.expert_top_k),
                    runtime.options.expert_top_p);
                for(int rank=keep;rank<top_k;++rank)weights_host[static_cast<std::size_t>(token)*top_k+rank]=0.0f;
                runtime.route_expert_sum+=static_cast<std::uint64_t>(keep);
            }
        }else runtime.route_expert_sum+=static_cast<std::uint64_t>(rows)*top_k;
        if(!expert_policy.routed_gpu_execution_allowed()&&
           expert_policy.records_prefill_frequency())
            for(int token=0;token<rows;++token)for(int rank=0;rank<top_k;++rank){
                const int route=token*top_k+rank;
                if(weights_host[route]!=0.0f)
                    record_expert_access(runtime,layer_number,
                        static_cast<std::uint32_t>(selected_host[route]));
            }
        const bool gpu_rows_allowed=
            expert_policy.routed_gpu_execution_allowed()&&
            !runtime.expert_slots.empty();
        // select_expert_cache_slot evaluates the policy at phase `decode`
        // internally, so the real phase has to be checked here: prefill rows
        // must keep training the history without touching residency.
        const bool rows_may_admit=expert_policy.misses_may_be_admitted();
        // Prefill expert streaming engages exactly where the two-half
        // pipeline does: configurations whose routed prefill experts run
        // wholly on the host. A budget of zero (the default, and every
        // config without the arena) leaves this path untouched.
        const bool stream_experts=span.stream_bytes>0&&
            runtime.dma_paging&&
            expert_phase==colibri::v2::ExpertExecutionPhase::prefill&&
            (expert_policy.hybrid_prefill_cpu||
             !expert_policy.routed_gpu_execution_allowed());
        if(std::getenv("COLIBRI_STREAM_TRACE")){
            static int shown=0;
            if(shown<3){++shown;
                std::fprintf(stderr,
                    "[stream-gate] stream_bytes=%llu dma=%d phase_prefill=%d "
                    "hybrid_prefill_cpu=%d routed_gpu_allowed=%d => %d\n",
                    (unsigned long long)span.stream_bytes,(int)runtime.dma_paging,
                    (int)(expert_phase==colibri::v2::ExpertExecutionPhase::prefill),
                    (int)expert_policy.hybrid_prefill_cpu,
                    (int)expert_policy.routed_gpu_execution_allowed(),
                    (int)stream_experts);}
        }
        const bool gpu_table_path=!expert_policy.hybrid_prefill_cpu;
        span.gpu_experts=gpu_table_path;
        const auto span_entries=static_cast<std::size_t>(rows)*top_k;
        if(gpu_table_path){
            std::fill(gpu_weights,gpu_weights+span_entries,0.0f);
            std::fill(gpu_gate_scales,gpu_gate_scales+span_entries,1.0f);
            std::fill(gpu_up_scales,gpu_up_scales+span_entries,1.0f);
            std::fill(gpu_counts,gpu_counts+rows,0);
        }
        if(gpu_rows_allowed){
            for(int token=0;token<rows;++token)for(int rank=0;rank<top_k;++rank){
                const int route=token*top_k+rank;
                if(weights_host[route]==0.0f)continue;
                const int expert=selected_host[route];
                if(expert<0||expert>=experts)
                    throw std::runtime_error("native Qwen rows forward selected an invalid expert");
                if(expert_policy.records_prefill_frequency())
                    record_expert_access(runtime,layer_number,expert);
                const auto cache_key=(static_cast<std::uint64_t>(layer_number)<<32)|static_cast<std::uint32_t>(expert);
                const auto resident=runtime.expert_residency.find(cache_key);
                std::size_t slot_index=kNoExpertSlot;
                if(resident!=runtime.expert_residency.end()){
                    slot_index=resident->second;
                    record_expert_cache_hit(runtime,runtime.expert_slots[slot_index]);
                }else{
                    ++runtime.expert_cache_misses;
                    // Under MTP this path *is* the decode path, so declining to
                    // admit here leaves the cache empty forever and runs every
                    // route on the CPU. Page in exactly as decode does. Only the
                    // DMA route is taken: the staged route would need
                    // rows*top_k expert bundles of host staging, and
                    // expert_staging_bytes is sized for top_k*2.
                    if(rows_may_admit&&layer.expert_dma)
                        slot_index=select_expert_cache_slot(
                            runtime,layer_number,expert,true);
                    if(slot_index==kNoExpertSlot)continue;
                    auto&admitted=runtime.expert_slots[slot_index];
                    admitted.key=cache_key;
                    admitted.valid=true;
                    admitted.native_valid=false;
                    runtime.expert_residency[cache_key]=slot_index;
                    const auto admitted_base=
                        runtime.expert_cache+slot_index*runtime.expert_slot_bytes;
                    std::uint64_t admitted_offset=0;
                    for(int role=0;role<3;++role){
                        const auto&source=runtime.model->tensors[layer.expert_tensors[role]];
                        const auto bytes=source.size/experts;
                        if(colibri_gpu_upload(
                               admitted_base+admitted_offset,
                               tensor_data(*runtime.model,source)+
                                   static_cast<std::uint64_t>(expert)*bytes,
                               bytes,runtime.stream)!=0)
                            throw std::runtime_error(
                                "native Qwen rows forward expert DMA upload failed");
                        admitted_offset+=bytes;
                    }
                }
                auto& slot=runtime.expert_slots[slot_index];
                slot.last_used=++runtime.expert_clock;
                const int compact=token*top_k+gpu_counts[token]++;
                const auto device_base=runtime.expert_cache+slot_index*runtime.expert_slot_bytes;
                std::uint64_t role_offset=0;
                for(int role=0;role<3;++role){
                    const auto bytes=runtime.model->tensors[layer.expert_tensors[role]].size/experts;
                    const auto pointer=device_base+role_offset;
                    if(role==0)gate_pointers[compact]=pointer;else if(role==1)up_pointers[compact]=pointer;else down_pointers[compact]=pointer;
                    role_offset+=bytes;
                }
                gpu_weights[compact]=weights_host[route];
                gpu_gate_scales[compact]=qwen_expert_role_scale(runtime,layer.expert_gate_scale,expert);
                gpu_up_scales[compact]=qwen_expert_role_scale(runtime,layer.expert_up_scale,expert)
                    *qwen_expert_role_scale(runtime,layer.expert_down_scale,expert);
                weights_host[route]=0.0f;
            }
        }
        // Prefill expert GEMM (plans/prefill-expert-gemm.md): stage the
        // densest routed experts of this half-layer into the span's arena
        // slice, pack each staged expert's tokens into a contiguous tile and
        // run the dense batch kernels over the staged weights -- decoded once
        // per token tile instead of once per routed token, which is what
        // killed the per-token grouped kernels for prefill-sized route
        // counts. Densest first because the upload is a fixed cost per
        // expert while the saving scales with its routed tokens; every pick
        // and every token list is a pure function of the routes, so a fixed
        // budget is deterministic. The scratch walk mirrors the prepare-time
        // sizing exactly.
        struct StreamGroup{int expert;std::uint64_t matrix;int offset;int count;};
        std::vector<StreamGroup> stream_groups;
        // Hoisted out of the staging block for the routed MMQ below: which
        // group holds an expert, and what the route weights were before
        // staging zeroed the ones it claimed.
        std::vector<std::int32_t> group_of;
        std::vector<float> routed_route_weights;
        std::uint64_t stream_role_bytes[3]={0,0,0};
        const std::uint64_t stream_cap=runtime.forward_rows_capacity;
        std::uint64_t stream_cursor=span.stream_scratch;
        auto stream_take=[&](std::uint64_t bytes){
            const auto at=stream_cursor;
            stream_cursor+=device_align(bytes);
            return at;
        };
        const auto stream_idx_dev=stream_take(stream_cap*top_k*sizeof(std::int32_t));
        const auto stream_wts_dev=stream_take(stream_cap*top_k*sizeof(float));
        const auto stream_packed=stream_take(stream_cap*hidden_size*sizeof(float));
        const auto stream_packed_q8=stream_take(stream_cap*hidden_size);
        const auto stream_packed_scales=
            stream_take(stream_cap*(hidden_size/32+1)*sizeof(float));
        const auto stream_act=stream_take(stream_cap*2*intermediate*sizeof(float));
        const auto stream_silu=stream_take(stream_cap*intermediate*sizeof(float));
        const auto stream_act_q8=stream_take(stream_cap*intermediate);
        const auto stream_act_scales=
            stream_take(stream_cap*(intermediate/32+1)*sizeof(float));
        const auto stream_down=stream_take(stream_cap*hidden_size*sizeof(float));
        // Routed MMQ tables. Same order and same worst cases as the sizing in
        // prepare; the two walks must not drift.
        const std::uint64_t routed_routes=stream_cap*top_k;
        const std::uint64_t routed_padded=
            routed_routes+static_cast<std::uint64_t>(experts)*
                (colibri::v2::moe::kMmqBlockSize-1);
        const std::uint64_t routed_blocks=
            routed_padded/colibri::v2::moe::kMmqBlockSize+1;
        std::uint64_t routed_role_ptrs[3];
        for(int role=0;role<3;++role)
            routed_role_ptrs[role]=
                stream_take(static_cast<std::uint64_t>(experts)*sizeof(std::uint64_t));
        const auto routed_block_experts=
            stream_take(routed_blocks*sizeof(std::int32_t));
        const auto routed_sorted=stream_take(routed_padded*sizeof(std::int32_t));
        const auto routed_slot_of_route=
            stream_take(routed_routes*sizeof(std::int32_t));
        const auto routed_weights=stream_take(routed_routes*sizeof(float));
        if(stream_experts){
            // Every role needs a batched kernel: MMQ over Q8 activations
            // where the format has one, else the reconstruct-in-float tile
            // kernel. A format with neither keeps its experts on the CPU.
            auto stream_role_ok=[&](int role,int in_size){
                const auto type=
                    runtime.model->tensors[layer.expert_tensors[role]].type;
                const auto*format=colibri::v2::qwen_format(type);
                if(!format)return false;
                if(format->matmul_q8_mmq&&runtime.int8_tensor_cores&&
                   (in_size&255)==0)return true;
                return format->matmul_rows!=nullptr&&
                    format->matmul_rows_grid!=
                        colibri::v2::RowsMatmulGrid::per_token;
            };
            if(stream_role_ok(0,hidden_size)&&stream_role_ok(1,hidden_size)&&
               stream_role_ok(2,intermediate)){
            std::uint64_t triple_bytes=0;
            for(int role=0;role<3;++role){
                stream_role_bytes[role]=
                    runtime.model->tensors[layer.expert_tensors[role]].size/experts;
                triple_bytes+=stream_role_bytes[role];
            }
            const auto stride=device_align(triple_bytes);
            std::vector<int> expert_tokens(experts,0);
            for(int token=0;token<rows;++token)for(int rank=0;rank<top_k;++rank){
                const int route=token*top_k+rank;
                if(weights_host[route]==0.0f)continue;
                const int expert=selected_host[route];
                if(expert<0||expert>=experts)
                    throw std::runtime_error("native Qwen rows forward selected an invalid expert");
                ++expert_tokens[expert];
            }
            std::vector<int> order;order.reserve(experts);
            for(int expert=0;expert<experts;++expert)
                if(expert_tokens[expert])order.push_back(expert);
            std::sort(order.begin(),order.end(),[&](int left,int right){
                return expert_tokens[left]!=expert_tokens[right]
                    ?expert_tokens[left]>expert_tokens[right]:left<right;});
            group_of.assign(static_cast<std::size_t>(experts),-1);
            std::uint64_t cursor=0;
            // The uploads ride the prefetch stream: interleaving thousands of
            // small copies with kernels on the compute stream serializes both.
            // The slice was last read by this span's previous layer's GEMMs,
            // so overwriting waits for that consumption first.
            if(colibri_gpu_stream_wait_event(
                   runtime.prefetch_stream,span.consumed_event)!=0)
                throw std::runtime_error(
                    "native Qwen prefill expert stream fence failed");
            const double t_upload=moe_now();
            for(const int expert:order){
                if(cursor+stride>span.stream_bytes)break;
                // NVFP4-style per-expert scales fold into the kernels the
                // grouped path runs; the GEMM path has no seam for them, so
                // scaled experts stay on the CPU.
                if(qwen_expert_role_scale(runtime,layer.expert_gate_scale,expert)!=1.0f||
                   qwen_expert_role_scale(runtime,layer.expert_up_scale,expert)!=1.0f||
                   qwen_expert_role_scale(runtime,layer.expert_down_scale,expert)!=1.0f)
                    continue;
                const auto base=span.stream_base+cursor;
                std::uint64_t offset=0;
                for(int role=0;role<3;++role){
                    const auto&source=
                        runtime.model->tensors[layer.expert_tensors[role]];
                    if(colibri_gpu_upload(base+offset,
                           tensor_data(*runtime.model,source)+
                               static_cast<std::uint64_t>(expert)*stream_role_bytes[role],
                           stream_role_bytes[role],runtime.prefetch_stream)!=0)
                        throw std::runtime_error(
                            "native Qwen prefill expert stream upload failed");
                    offset+=stream_role_bytes[role];
                }
                group_of[expert]=static_cast<std::int32_t>(stream_groups.size());
                stream_groups.push_back({expert,base,0,0});
                cursor+=stride;
                runtime.prefill_streamed_bytes+=triple_bytes;
            }
            if(moe_profile_enabled())
                qwen_moe_profile().stream_upload+=moe_now()-t_upload;
            if(cursor&&
               (colibri_gpu_event_record(
                    span.upload_ready_event,runtime.prefetch_stream)!=0||
                colibri_gpu_stream_wait_event(
                    runtime.stream,span.upload_ready_event)!=0))
                throw std::runtime_error(
                    "native Qwen prefill expert stream fence failed");
            // Token lists per group, tokens ascending. The streamed routes
            // leave weights_host so the CPU sweep skips them; their route
            // weights ride the scatter instead.
            for(int token=0;token<rows;++token)for(int rank=0;rank<top_k;++rank){
                const int route=token*top_k+rank;
                if(weights_host[route]==0.0f)continue;
                const int group=group_of[selected_host[route]];
                if(group>=0)++stream_groups[group].count;
            }
            int stream_total=0;
            for(auto&group:stream_groups){
                group.offset=stream_total;
                stream_total+=group.count;
                group.count=0;
            }
            std::vector<std::int32_t> stream_tokens(stream_total);
            std::vector<float> stream_route_weights(stream_total);
            // The loop below zeroes a claimed route's weight so the CPU sweep
            // skips it; the routed MMQ still needs the value, and the aligned
            // layout is built from a non-zero weight.
            routed_route_weights.assign(
                weights_host,weights_host+static_cast<std::size_t>(rows)*top_k);
            for(int token=0;token<rows;++token)for(int rank=0;rank<top_k;++rank){
                const int route=token*top_k+rank;
                if(weights_host[route]==0.0f)continue;
                const int index=group_of[selected_host[route]];
                if(index<0)continue;
                auto&group=stream_groups[index];
                stream_tokens[group.offset+group.count]=token;
                stream_route_weights[group.offset+group.count]=weights_host[route];
                ++group.count;
                weights_host[route]=0.0f;
            }
            if(stream_total&&
               (colibri_gpu_upload(stream_idx_dev,stream_tokens.data(),
                    stream_total*sizeof(std::int32_t),runtime.stream)!=0||
                colibri_gpu_upload(stream_wts_dev,stream_route_weights.data(),
                    stream_total*sizeof(float),runtime.stream)!=0))
                throw std::runtime_error(
                    "native Qwen prefill expert stream index upload failed");
            }
        }
        bool any_gpu=false;
        if(gpu_rows_allowed)
            for(int token=0;token<rows;++token)
                any_gpu=any_gpu||gpu_counts[token]>0;
        if(gpu_table_path){
            const auto output_bytes=
                static_cast<std::uint64_t>(rows)*hidden_size*sizeof(float);
            if(colibri_gpu_memset(fourth,0,output_bytes,runtime.stream)!=0)throw std::runtime_error("native Qwen rows forward GPU expert reset failed");
            const auto pointer_bytes=span_entries*sizeof(std::uint64_t);
            const auto scale_bytes=span_entries*sizeof(float);
            const auto table_row=static_cast<std::uint64_t>(span_begin)*top_k;
            auto*gate_host=staging+host_layout.gate_table.offset+table_row*sizeof(std::uint64_t);
            auto*up_host=staging+host_layout.up_table.offset+table_row*sizeof(std::uint64_t);
            auto*down_pointer_host=staging+host_layout.down_table.offset+table_row*sizeof(std::uint64_t);
            auto*gpu_weight_host=staging+host_layout.weight_table.offset+table_row*sizeof(float);
            auto*gpu_gate_scale_host=staging+host_layout.gate_scale_table.offset+table_row*sizeof(float);
            auto*gpu_up_scale_host=staging+host_layout.up_scale_table.offset+table_row*sizeof(float);
            auto*gpu_count_host=staging+host_layout.counts.offset+static_cast<std::uint64_t>(span_begin)*sizeof(std::int32_t);
            std::memcpy(gate_host,gate_pointers,pointer_bytes);
            std::memcpy(up_host,up_pointers,pointer_bytes);
            std::memcpy(down_pointer_host,down_pointers,pointer_bytes);
            std::memcpy(gpu_weight_host,gpu_weights,scale_bytes);
            std::memcpy(gpu_gate_scale_host,gpu_gate_scales,scale_bytes);
            std::memcpy(gpu_up_scale_host,gpu_up_scales,scale_bytes);
            std::memcpy(gpu_count_host,gpu_counts,rows*sizeof(std::int32_t));
            if(colibri_gpu_upload(gpu_gate_table,gate_host,pointer_bytes,runtime.stream)!=0||
               colibri_gpu_upload(gpu_up_table,up_host,pointer_bytes,runtime.stream)!=0||
               colibri_gpu_upload(gpu_down_table,down_pointer_host,pointer_bytes,runtime.stream)!=0||
               colibri_gpu_upload(gpu_weight_table,gpu_weight_host,scale_bytes,runtime.stream)!=0||
               colibri_gpu_upload(gpu_gate_scale_table,gpu_gate_scale_host,scale_bytes,runtime.stream)!=0||
               colibri_gpu_upload(gpu_up_scale_table,gpu_up_scale_host,scale_bytes,runtime.stream)!=0||
               colibri_gpu_upload(gpu_count_table,gpu_count_host,rows*sizeof(std::int32_t),runtime.stream)!=0)
                throw std::runtime_error("native Qwen rows forward pointer upload failed");
            if(any_gpu){
                void*gate_up_args[]={const_cast<std::uint64_t*>(&gpu_gate_table),const_cast<std::uint64_t*>(&gpu_up_table),const_cast<std::uint64_t*>(&gpu_count_table),const_cast<std::uint64_t*>(&normalized),const_cast<std::uint64_t*>(&gpu_activated),const_cast<int*>(&hidden_size),const_cast<int*>(&intermediate),const_cast<int*>(&top_k),&rows,const_cast<std::uint64_t*>(&gpu_gate_scale_table),const_cast<std::uint64_t*>(&gpu_up_scale_table)};
                const auto gate_type=runtime.model->tensors[layer.expert_tensors[0]].type;
                launch(qwen_grouped_swiglu_name(gate_type,false,true).c_str(),intermediate,rows*top_k,256,gate_up_args);
                const auto down_type=runtime.model->tensors[layer.expert_tensors[2]].type;
                void*down_args[]={const_cast<std::uint64_t*>(&gpu_down_table),const_cast<std::uint64_t*>(&gpu_activated),const_cast<std::uint64_t*>(&fourth),const_cast<std::uint64_t*>(&gpu_weight_table),const_cast<std::uint64_t*>(&gpu_count_table),const_cast<int*>(&intermediate),const_cast<int*>(&hidden_size),const_cast<int*>(&top_k),&rows};
                launch(qwen_grouped_accumulate_rows_name(down_type).c_str(),hidden_size,rows,256,down_args);
            }
        }
        // Expert-GEMM execution: pack, project, activate, project, fold back.
        if(!stream_groups.empty()){
            if(!gpu_table_path){
                const auto output_bytes=
                    static_cast<std::uint64_t>(rows)*hidden_size*sizeof(float);
                if(colibri_gpu_memset(fourth,0,output_bytes,runtime.stream)!=0)
                    throw std::runtime_error("native Qwen rows forward GPU expert reset failed");
            }
            const auto gate_type=runtime.model->tensors[layer.expert_tensors[0]].type;
            const auto up_type=runtime.model->tensors[layer.expert_tensors[1]].type;
            const auto down_type=runtime.model->tensors[layer.expert_tensors[2]].type;
            auto stream_needs_q8=[&](std::uint32_t type,int in_size){
                const auto*format=colibri::v2::qwen_format(type);
                return format->matmul_q8_mmq&&runtime.int8_tensor_cores&&
                    (in_size&255)==0;
            };
            auto expert_matmul=[&](std::uint32_t type,std::uint64_t matrix,
                                   std::uint64_t f32_input,std::uint64_t q8_input,
                                   std::uint64_t q8_scales,std::uint64_t output,
                                   int in_size,int out_size,int count){
                const auto*format=colibri::v2::qwen_format(type);
                if(stream_needs_q8(type,in_size)){
                    const int mmq_rows=format->mmq_min?kQ8MmqMinRows:kQ8MmqRows;
                    const int mmq_threads=format->mmq_min?kQ8MmqMinThreads
                        :qwen_mmq_wide()?kQ8MmqWideThreads:kQ8MmqThreads;
                    const int mmq_tokens=format->mmq_min?kQ8MmqMinTokens:kQ8MmqTokens;
                    int scale_stride=(in_size/32)*2;
                    for(int start=0;start<count;start+=mmq_tokens){
                        int batch=std::min(mmq_tokens,count-start);
                        auto batch_q8=q8_input+static_cast<std::uint64_t>(start)*in_size;
                        auto batch_scales=q8_scales+
                            static_cast<std::uint64_t>(start)*(in_size/32)*sizeof(float);
                        auto batch_output=output+
                            static_cast<std::uint64_t>(start)*out_size*sizeof(float);
                        void*args[]={&matrix,&batch_q8,&batch_scales,&batch_output,
                            &in_size,&out_size,&batch,&scale_stride};
                        launch(format->matmul_q8_mmq,
                               (out_size+mmq_rows-1)/mmq_rows,1,mmq_threads,args);
                    }
                    return;
                }
                void*args[]={&matrix,&f32_input,&output,&in_size,&out_size,&count};
                if(format->matmul_rows_grid==colibri::v2::RowsMatmulGrid::tiled32){
                    launch(format->matmul_rows,(out_size+31)/32,
                           (static_cast<std::uint32_t>(count)+31)/32,256,args);
                    return;
                }
                launch(format->matmul_rows,out_size,(count+3)/4,256,args);
            };
            auto stream_quantize=[&](std::uint64_t input,std::uint64_t q8,
                                     std::uint64_t scales,int width,int count){
                int scale_stride=(width/32)*2;
                void*args[]={&input,&q8,&scales,&width,&scale_stride};
                launch("quantize_q8_blocks_rows",(width+31)/32,
                       static_cast<std::uint32_t>(count),32,args);
            };
            const bool gate_up_q8=stream_needs_q8(gate_type,hidden_size)||
                stream_needs_q8(up_type,hidden_size);
            const bool down_q8=stream_needs_q8(down_type,intermediate);
            // The routed MMQ: one aligned layout per layer, then a handful of
            // launches per tile, instead of ~8 launches per staged expert. The
            // kernels measure 12.6x (gate/up) and 25x (down) over the octet
            // block-major pair, and 14.6x over the CPU expert phase -- see
            // plans/decode-device-dispatch.md. Opt-in for now: this is the
            // first path to run them end to end, and the project does not turn
            // on a numerics-affecting route silently.
            // The option is the supported switch; the env var stays as an
            // override so the path can be A/B'd against itself without
            // rebuilding a runtime.
            static const char*routed_env=std::getenv("COLIBRI_ROUTED_MOE");
            const bool routed_setting=routed_env
                ? routed_env[0]=='1'
                : runtime.options.routed_moe!=0;
            // A format reaches the routed kernel when it has one and the width
            // divides its blocking: 256 for the super-block formats, 32 for
            // IQ4_NL's flat blocks, which is what lets the 640-wide down
            // projection in at all.
            auto routed_kernel=[&](std::uint32_t type,int in_size){
                auto name=qwen_iq_grouped_kernel(type,"_q8_mmq_routed");
                const int unit=type==20?32:256;
                return (!name.empty()&&in_size%unit==0)?name:std::string();
            };
            const auto routed_gate=routed_kernel(gate_type,hidden_size);
            const auto routed_up=routed_kernel(up_type,hidden_size);
            const auto routed_down=routed_kernel(down_type,intermediate);
            const bool routed_moe=routed_setting&&runtime.int8_tensor_cores&&
                !routed_gate.empty()&&!routed_up.empty()&&!routed_down.empty()&&
                !routed_route_weights.empty();
            if(routed_moe){
                colibri::v2::moe::AlignedRoutes aligned;
                {
                    // Routes the staging claimed, keyed by group rather than by
                    // expert, so the kernel's block table indexes the compact
                    // pointer tables directly.
                    std::vector<std::int32_t> route_group(
                        static_cast<std::size_t>(rows)*top_k,-1);
                    for(int route=0;route<rows*top_k;++route)
                        if(routed_route_weights[route]!=0.0f)
                            route_group[route]=group_of[selected_host[route]];
                    colibri::v2::moe::align_blocks(
                        route_group.data(),routed_route_weights.data(),
                        rows*top_k,static_cast<int>(stream_groups.size()),
                        colibri::v2::moe::kMmqBlockSize,aligned);
                }
                const int blocks=static_cast<int>(aligned.block_experts.size());
                if(blocks){
                    std::vector<std::int32_t> slot_of_route(
                        static_cast<std::size_t>(rows)*top_k,-1);
                    for(std::size_t slot=0;slot<aligned.sorted_routes.size();++slot){
                        const auto route=aligned.sorted_routes[slot];
                        if(route>=0)
                            slot_of_route[static_cast<std::size_t>(route)]=
                                static_cast<std::int32_t>(slot);
                    }
                    const std::uint64_t role_offset[3]={
                        0,stream_role_bytes[0],
                        stream_role_bytes[0]+stream_role_bytes[1]};
                    std::vector<std::uint64_t> role_table[3];
                    for(const auto&group:stream_groups)
                        for(int role=0;role<3;++role)
                            role_table[role].push_back(
                                group.matrix+role_offset[role]);
                    for(int role=0;role<3;++role)
                        if(colibri_gpu_upload(routed_role_ptrs[role],
                               role_table[role].data(),
                               role_table[role].size()*sizeof(std::uint64_t),
                               runtime.stream)!=0)
                            throw std::runtime_error(
                                "native Qwen routed MoE pointer upload failed");
                    if(colibri_gpu_upload(routed_block_experts,
                           aligned.block_experts.data(),
                           aligned.block_experts.size()*sizeof(std::int32_t),
                           runtime.stream)!=0||
                       colibri_gpu_upload(routed_sorted,
                           aligned.sorted_routes.data(),
                           aligned.sorted_routes.size()*sizeof(std::int32_t),
                           runtime.stream)!=0||
                       colibri_gpu_upload(routed_slot_of_route,
                           slot_of_route.data(),
                           slot_of_route.size()*sizeof(std::int32_t),
                           runtime.stream)!=0||
                       colibri_gpu_upload(routed_weights,
                           routed_route_weights.data(),
                           routed_route_weights.size()*sizeof(float),
                           runtime.stream)!=0)
                        throw std::runtime_error(
                            "native Qwen routed MoE table upload failed");
                    // Once per layer, not once per expert: every routed token
                    // reads the same activation row, and the per-expert gather
                    // the streaming path needs disappears with it.
                    stream_quantize(normalized,stream_packed_q8,
                                    stream_packed_scales,hidden_size,rows);
                    const int block_size=colibri::v2::moe::kMmqBlockSize;
                    // Tiles are sized so every per-slot buffer below is the
                    // scratch the streaming path already owns.
                    const int tile_blocks=
                        std::max(1,static_cast<int>(stream_cap)/block_size);
                    int gate_scale_stride=(hidden_size/32)*2;
                    int down_scale_stride=(intermediate/32)*2;
                    // Mutable copies: the launch shim takes void* to each
                    // argument, so a const int cannot be handed over.
                    int slot_major=0,routed_top_k=top_k;
                    const double t_routed=moe_now();
                    for(int begin=0;begin<blocks;begin+=tile_blocks){
                        const int count=std::min(tile_blocks,blocks-begin);
                        int slots=count*block_size;
                        const int slot_begin=begin*block_size;
                        const auto tile_experts=routed_block_experts+
                            static_cast<std::uint64_t>(begin)*sizeof(std::int32_t);
                        const auto tile_sorted=routed_sorted+
                            static_cast<std::uint64_t>(slot_begin)*sizeof(std::int32_t);
                        const auto up_half=stream_act+
                            static_cast<std::uint64_t>(slots)*intermediate*sizeof(float);
                        int tile_count=count;
                        auto routed_launch=[&](const std::string&name,
                                               std::uint64_t table,
                                               std::uint64_t input,
                                               std::uint64_t scales,
                                               std::uint64_t output,
                                               int in_size,int out_size,
                                               int*token_map,int*scale_stride){
                            void*args[]={&table,
                                const_cast<std::uint64_t*>(&tile_experts),
                                const_cast<std::uint64_t*>(&tile_sorted),
                                const_cast<std::uint64_t*>(&input),
                                const_cast<std::uint64_t*>(&scales),
                                const_cast<std::uint64_t*>(&output),
                                &in_size,&out_size,token_map,scale_stride,
                                &tile_count};
                            launch(name.c_str(),
                                   (static_cast<std::uint32_t>(out_size)+127)/128,
                                   static_cast<std::uint32_t>(count),256,args);
                        };
                        routed_launch(routed_gate,routed_role_ptrs[0],
                            stream_packed_q8,stream_packed_scales,stream_act,
                            hidden_size,intermediate,&routed_top_k,&gate_scale_stride);
                        routed_launch(routed_up,routed_role_ptrs[1],
                            stream_packed_q8,stream_packed_scales,up_half,
                            hidden_size,intermediate,&routed_top_k,&gate_scale_stride);
                        int silu_count=slots*intermediate;
                        void*silu_args[]={const_cast<std::uint64_t*>(&stream_act),
                            const_cast<std::uint64_t*>(&stream_silu),&silu_count};
                        launch("silu_mul",
                               (static_cast<std::uint32_t>(silu_count)+255)/256,
                               1,256,silu_args);
                        stream_quantize(stream_silu,stream_act_q8,
                                        stream_act_scales,intermediate,slots);
                        // top_k 0: the down projection's input is the per-slot
                        // SwiGLU output, not a token row.
                        routed_launch(routed_down,routed_role_ptrs[2],
                            stream_act_q8,stream_act_scales,stream_down,
                            intermediate,hidden_size,&slot_major,
                            &down_scale_stride);
                        // Token-major fold: several experts in one tile means
                        // several of a token's routes land here, and the corpus
                        // has no atomicAdd. Each block owns one token's row.
                        int tile_slot_begin=slot_begin,tile_slots=slots;
                        void*fold_args[]={const_cast<std::uint64_t*>(&stream_down),
                            const_cast<std::uint64_t*>(&routed_slot_of_route),
                            const_cast<std::uint64_t*>(&routed_weights),
                            const_cast<std::uint64_t*>(&fourth),
                            const_cast<int*>(&hidden_size),&rows,
                            const_cast<int*>(&top_k),&tile_slot_begin,
                            &tile_slots};
                        launch("qwen_slot_accumulate_rows",
                               (static_cast<std::uint32_t>(hidden_size)+255)/256,
                               static_cast<std::uint32_t>(rows),256,fold_args);
                    }
                    if(moe_profile_enabled()){
                        auto&prof=qwen_moe_profile();
                        prof.stream_compute+=moe_now()-t_routed;
                        prof.stream_groups+=static_cast<std::uint64_t>(blocks);
                    }
                }
            }
            const double t_groups=moe_now();
            std::uint64_t group_launches=0,group_tokens=0;
            for(const auto&group:stream_groups){
                if(routed_moe)break;
                if(!group.count)continue;
                int count=group.count;
                if(moe_profile_enabled()){++group_launches;group_tokens+=count;}
                const auto indices=stream_idx_dev+
                    static_cast<std::uint64_t>(group.offset)*sizeof(std::int32_t);
                const auto route_weights=stream_wts_dev+
                    static_cast<std::uint64_t>(group.offset)*sizeof(float);
                void*gather_args[]={const_cast<std::uint64_t*>(&normalized),
                    const_cast<std::uint64_t*>(&indices),
                    const_cast<std::uint64_t*>(&stream_packed),
                    const_cast<int*>(&hidden_size),&count};
                launch("qwen_gather_rows",
                       (static_cast<std::uint32_t>(hidden_size)+255)/256,
                       static_cast<std::uint32_t>(count),256,gather_args);
                if(gate_up_q8)
                    stream_quantize(stream_packed,stream_packed_q8,
                                    stream_packed_scales,hidden_size,count);
                const auto up_half=stream_act+
                    static_cast<std::uint64_t>(count)*intermediate*sizeof(float);
                expert_matmul(gate_type,group.matrix,
                    stream_packed,stream_packed_q8,stream_packed_scales,
                    stream_act,hidden_size,intermediate,count);
                expert_matmul(up_type,group.matrix+stream_role_bytes[0],
                    stream_packed,stream_packed_q8,stream_packed_scales,
                    up_half,hidden_size,intermediate,count);
                int silu_count=count*intermediate;
                void*silu_args[]={const_cast<std::uint64_t*>(&stream_act),
                    const_cast<std::uint64_t*>(&stream_silu),&silu_count};
                launch("silu_mul",
                       (static_cast<std::uint32_t>(silu_count)+255)/256,1,256,
                       silu_args);
                if(down_q8)
                    stream_quantize(stream_silu,stream_act_q8,stream_act_scales,
                                    intermediate,count);
                expert_matmul(down_type,
                    group.matrix+stream_role_bytes[0]+stream_role_bytes[1],
                    stream_silu,stream_act_q8,stream_act_scales,
                    stream_down,intermediate,hidden_size,count);
                void*scatter_args[]={const_cast<std::uint64_t*>(&stream_down),
                    const_cast<std::uint64_t*>(&indices),
                    const_cast<std::uint64_t*>(&route_weights),
                    const_cast<std::uint64_t*>(&fourth),
                    const_cast<int*>(&hidden_size),&count};
                launch("qwen_scatter_add_rows",
                       (static_cast<std::uint32_t>(hidden_size)+255)/256,
                       static_cast<std::uint32_t>(count),256,scatter_args);
            }
            if(moe_profile_enabled()){
                auto&prof=qwen_moe_profile();
                prof.stream_compute+=moe_now()-t_groups;
                prof.stream_groups+=group_launches;
                prof.stream_group_tokens+=group_tokens;
            }
            // The GEMMs above are the last readers of this span's streamed
            // slice; the next layer's uploads wait on this marker.
            if(colibri_gpu_event_record(span.consumed_event,runtime.stream)!=0)
                throw std::runtime_error(
                    "native Qwen prefill expert stream fence failed");
            span.gpu_experts=true;
        }
        const auto cpu_expert_started=std::chrono::steady_clock::now();
        qwen_cpu_moe_rows(runtime,layer,selected_host,weights_host,rows,top_k,input_host,activated_host,down_host,output_host);
        if(expert_policy.records_prefill_frequency())
            runtime.prefill_expert_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now()-cpu_expert_started).count();
    };
    // Phase three of a layer: fold the experts back in -- upload the CPU
    // result, sum the GPU-expert and shared-expert contributions into the
    // residual, and swap the span's ping-pong buffers for the next layer.
    auto queue_combine=[&](RowsSpan&span){
        install_span(span);
        // Set by cpu_phase: whether the span queued GPU expert work (table
        // path or streamed) whose accumulator in `fourth` must fold in here.
        const bool gpu_table_path=span.gpu_experts;
        const auto output_bytes=
            static_cast<std::uint64_t>(rows)*hidden_size*sizeof(float);
        if(colibri_gpu_upload(first,output_host,output_bytes,runtime.stream)!=0)throw std::runtime_error("native Qwen rows forward expert upload failed");
        runtime.expert_page_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-span.pager_started).count();
        if(gpu_table_path){add_rows(fourth,first);add_rows(third,fourth);}
        else add_rows(third,first);
        if(runtime.qwen4exp)hc_post_rows(third);
        else add_rows(residual,third);
        std::swap(span.hidden,span.residual);
    };
    // The phase driver. Serial: one full-range span through core -> cpu ->
    // combine per layer, exactly the order the loop always ran. Pipelined:
    // two halves interleave so the GPU always holds one half-layer of queued
    // core work while the CPU runs the other half's experts -- half B reads
    // half A's KV/DeltaNet state through normal stream ordering because A's
    // core for a layer is always queued before B's.
    RowsSpan whole{0,total_rows,hidden,residual,runtime.route_event,
                   runtime.prefill_stream_arena,runtime.prefill_stream_bytes,
                   runtime.prefill_stream_scratch,
                   runtime.stream_upload_event,runtime.stream_consumed_event,
                   false,{}};
    if(pipelined){
        // Halves stage into disjoint slices of the streaming arena; round the
        // front's share down to the device alignment so the back's base stays
        // aligned.
        const std::uint64_t stream_half=
            runtime.prefill_stream_bytes/2
            /colibri::v2::workspace::kDeviceAlignment
            *colibri::v2::workspace::kDeviceAlignment;
        RowsSpan front{0,pipeline_half,hidden,residual,
                       runtime.route_event,
                       runtime.prefill_stream_arena,stream_half,
                       runtime.prefill_stream_scratch,
                       runtime.stream_upload_event,
                       runtime.stream_consumed_event,false,{}};
        RowsSpan back{pipeline_half,total_rows-pipeline_half,hidden,residual,
                      runtime.route_event_pipeline,
                      runtime.prefill_stream_arena+stream_half,
                      runtime.prefill_stream_bytes-stream_half,
                      runtime.prefill_stream_scratch+
                          runtime.prefill_stream_scratch_span,
                      runtime.stream_upload_event_pipeline,
                      runtime.stream_consumed_event_pipeline,false,{}};
        // Queue core phases until one is a MoE layer whose expert work is
        // pending; dense feed-forward layers complete inside queue_core.
        auto advance=[&](RowsSpan&span,std::uint32_t&next)->bool{
            while(next<runtime.layers.size()){
                if(queue_core(span,next))return true;
                ++next;
            }
            return false;
        };
        std::uint32_t layer_front=0,layer_back=0;
        bool pending_front=advance(front,layer_front);
        bool pending_back=advance(back,layer_back);
        while(pending_front||pending_back){
            if(pending_front){
                cpu_phase(front,layer_front);
                queue_combine(front);
                ++layer_front;
                pending_front=advance(front,layer_front);
            }
            if(pending_back){
                cpu_phase(back,layer_back);
                queue_combine(back);
                ++layer_back;
                pending_back=advance(back,layer_back);
            }
        }
        // Both halves swapped once per layer, so they must end in the same
        // ping-pong buffer; anything else means the halves diverged and the
        // full-range head below would read half-stale rows.
        if(front.hidden!=back.hidden)
            throw std::runtime_error(
                "native Qwen prefill pipeline halves diverged");
        whole.hidden=front.hidden;
        whole.residual=front.residual;
    }else for(std::uint32_t layer_number=0;layer_number<runtime.layers.size();++layer_number){
        if(!queue_core(whole,layer_number))continue;
        cpu_phase(whole,layer_number);
        queue_combine(whole);
    }
    // Reinstall the full range: the head below reads `hidden` from whichever
    // ping-pong buffer the last layer left the span in.
    install_span(whole);
    if(output_tokens){
        if(runtime.qwen4exp)
            hc_pre_rows(runtime.device_tensors[runtime.output_hc_norm],
                        runtime.output_hc_down,runtime.output_hc_up,
                        std::numeric_limits<std::uint64_t>::max());
        else rms_rows(hidden,runtime.device_tensors[runtime.final_norm],normalized);
        if(colibri_gpu_memset(winners,0,rows*sizeof(std::uint64_t),runtime.stream)!=0)throw std::runtime_error("native Qwen rows forward argmax reset failed");
        int vocabulary=static_cast<int>(runtime.model->config.vocabulary_size);auto lm_head=runtime.device_tensors[runtime.lm_head];
        if(runtime.lm_head_type==8||runtime.lm_head_type==30){
            void*argmax_args[]={&lm_head,const_cast<std::uint64_t*>(&normalized),const_cast<std::uint64_t*>(&winners),const_cast<int*>(&hidden_size),&vocabulary,&rows};
            launch(runtime.lm_head_type==30?"qwen_bf16_lm_head_argmax_rows":"qwen_q8_lm_head_argmax_rows",(vocabulary+7)/8,rows,256,argmax_args);
        }else{
            // Quantized heads (K-quants, IQ, F32) have no fused rows kernel, so
            // stand in with the per-row kernels the single-token decode uses,
            // one launch per row. They write the same packed winner with an
            // atomicMax on winners[0] of whatever pointer they are handed, so
            // the row's own slot is just an offset. `rows` is the MTP verify
            // batch (<= 9), and each launch does exactly the work the fused
            // kernel would do for that row, so this only costs the extra
            // launches. Without it an IQ/K-quant checkpoint served with
            // --mtp-draft failed on the first decoded token.
            const char*lm_q8_kernel=qwen_q8_lm_head_kernel(runtime.lm_head_type);
            const char*lm_setting=std::getenv("COLIBRI_IQ2_Q8_DECODE");
            const bool lm_q8_enabled=!lm_setting||lm_setting[0]!='0';
            if(lm_q8_kernel&&lm_q8_enabled&&(hidden_size&255)==0){
                // Same trade the decode path makes: quantize the activation to
                // Q8 blocks and run the group-decode head instead of
                // reconstructing every weight in float. The head is the largest
                // tensor read per row and verification reads it once per row,
                // so the float fallback alone made speculation slower than
                // ordinary decode on a Q3_K head.
                for(int row=0;row<rows;++row){
                    auto row_input=normalized+static_cast<std::uint64_t>(row)*hidden_size*sizeof(float);
                    auto row_q8=small_q8+static_cast<std::uint64_t>(row)*hidden_size;
                    auto row_scales=small_q8_scales+
                        static_cast<std::uint64_t>(row)*(hidden_size/32)*sizeof(float);
                    void*quant_args[]={&row_input,&row_q8,&row_scales,const_cast<int*>(&hidden_size)};
                    launch("quantize_q8_blocks",(hidden_size+31)/32,1,32,quant_args);
                }
                for(int row=0;row<rows;++row){
                    auto row_q8=small_q8+static_cast<std::uint64_t>(row)*hidden_size;
                    auto row_scales=small_q8_scales+
                        static_cast<std::uint64_t>(row)*(hidden_size/32)*sizeof(float);
                    auto winner=winners+static_cast<std::uint64_t>(row)*sizeof(std::uint64_t);
                    void*row_args[]={&lm_head,&row_q8,&row_scales,&winner,const_cast<int*>(&hidden_size),&vocabulary};
                    launch(lm_q8_kernel,(vocabulary+7)/8,1,256,row_args);
                }
            }else{
                const char*kernel=qwen_lm_head_argmax_kernel(runtime.lm_head_type);
                for(int row=0;row<rows;++row){
                    auto vector=normalized+static_cast<std::uint64_t>(row)*hidden_size*sizeof(float);
                    auto winner=winners+static_cast<std::uint64_t>(row)*sizeof(std::uint64_t);
                    void*row_args[]={&lm_head,&vector,&winner,const_cast<int*>(&hidden_size),&vocabulary};
                    launch(kernel,(vocabulary+7)/8,1,256,row_args);
                }
            }
        }
        auto*packed=reinterpret_cast<std::uint64_t*>(staging);
        if(colibri_gpu_download(packed,winners,rows*sizeof(std::uint64_t),runtime.stream)!=0||colibri_gpu_stream_sync(runtime.stream)!=0)throw std::runtime_error("native Qwen rows forward output transfer failed");
        for(int row=0;row<rows;++row)output_tokens[row]=0xffffffffu-static_cast<std::uint32_t>(packed[row]);
    }
    if(preserve_mtp_prompt_hidden){
        auto destination=mtp_prompt_hidden+
            static_cast<std::uint64_t>(hidden_size)*sizeof(float);
        int elements=rows*hidden_size;
        void*copy_rows_args[]={&hidden,&destination,&elements};
        launch("qwen_copy_vector",(elements+255)/256,1,256,copy_rows_args);
        auto last_hidden=hidden+
            static_cast<std::uint64_t>(rows-1)*hidden_size*sizeof(float);
        auto target_hidden=runtime.state+runtime.mtp_target_hidden_offset;
        void*copy_last_args[]={&last_hidden,&target_hidden,
                              const_cast<int*>(&hidden_size)};
        launch("qwen_copy_vector",(hidden_size+255)/256,1,256,copy_last_args);
        runtime.mtp_has_target_hidden=true;
    }
    if(final_hidden)*final_hidden=hidden;
}

void qwen_verify_target_rows(
    ColibriV2QwenRuntime& runtime, const std::uint32_t* input_tokens,
    int rows, std::uint32_t* output_tokens, std::uint64_t* final_hidden
) {
    if(rows>9)throw std::runtime_error("native MTP verifier batch is unsupported");
    qwen_forward_rows(runtime,input_tokens,rows,output_tokens,final_hidden);
}

// Chunked prompt prefill: advance the recurrent/KV state over a whole chunk
// of prompt tokens without computing logits for any of them.
void qwen_prefill_rows(
    ColibriV2QwenRuntime& runtime, const std::uint32_t* tokens, int rows
) {
    // Gemma 4 has its own layer-synchronous chunk driver (and no MTP), with
    // the same bookkeeping contract as the tail below.
    if(runtime.gemma4){gemma4_prefill_rows(runtime,tokens,rows);return;}
    const auto started=std::chrono::steady_clock::now();
    const bool had_mtp_target_hidden=runtime.mtp_has_target_hidden;
    qwen_forward_rows(runtime,tokens,rows,nullptr,nullptr);
    if(runtime.options.mtp_drafts){
        const auto preserved=
            runtime.rows_workspace_layout.mtp_prompt_hidden.address(runtime.workspace);
        const int hidden=static_cast<int>(runtime.model->config.hidden_size);
        for(int row=had_mtp_target_hidden?0:1;row<rows;++row)
            qwen_mtp_append_pair(
                runtime,tokens[row],preserved+
                    static_cast<std::uint64_t>(row)*hidden*sizeof(float));
    }
    runtime.processed_tokens.insert(runtime.processed_tokens.end(),tokens,tokens+rows);
    runtime.position+=static_cast<std::uint64_t>(rows);
    runtime.decode_calls+=static_cast<std::uint64_t>(rows);
    runtime.decode_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count();
    ++runtime.prefill_calls;
    runtime.prefill_tokens+=static_cast<std::uint64_t>(rows);
    runtime.prefill_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-started).count();
}
