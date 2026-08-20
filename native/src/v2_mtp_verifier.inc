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
// The _MIN variant still runs one fragment per warp.
static constexpr int kQ8MmqMinThreads=(kQ8MmqMinRows/16)*(kQ8MmqMinTokens/8)*32;

// Launch tally for COLIBRI_PREFILL_LAUNCHES; see the `launch` lambda below.
struct QwenLaunchStat { std::uint64_t calls=0, blocks=0; };
inline std::map<std::string,QwenLaunchStat>& qwen_launch_stats(){
    static std::map<std::string,QwenLaunchStat> stats; return stats;
}
inline void qwen_note_launch(const char* name,std::uint32_t gx,std::uint32_t gy){
    auto& e=qwen_launch_stats()[name];
    ++e.calls; e.blocks+=static_cast<std::uint64_t>(gx)*gy;
}
inline void qwen_prefill_launch_report(){
    auto& stats=qwen_launch_stats();
    if(stats.empty())return;
    std::uint64_t total=0;
    for(const auto& kv:stats)total+=kv.second.calls;
    std::vector<std::pair<std::string,QwenLaunchStat>> ordered(stats.begin(),stats.end());
    std::sort(ordered.begin(),ordered.end(),[](const auto&a,const auto&b){
        return a.second.calls>b.second.calls;});
    std::fprintf(stderr,"[launches] %llu total\n",
                 static_cast<unsigned long long>(total));
    for(const auto& kv:ordered)
        std::fprintf(stderr,"[launches] %-34s %8llu calls %5.1f%%  %10llu blocks  %6.1f blocks/call\n",
            kv.first.c_str(),static_cast<unsigned long long>(kv.second.calls),
            100.0*kv.second.calls/total,
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
    int rows, std::uint32_t* output_tokens, std::uint64_t* final_hidden
) {
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
    expert_policy.hybrid_prefill_cpu=
        expert_phase==colibri::v2::ExpertExecutionPhase::prefill&&
        runtime.options.hybrid_prefill_cpu!=0;
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
    std::uint64_t hidden=workspace_layout.hidden.address(runtime.workspace);
    std::uint64_t residual=workspace_layout.residual.address(runtime.workspace);
    const auto normalized=workspace_layout.normalized.address(runtime.workspace);
    const auto first=workspace_layout.first.address(runtime.workspace);
    const auto second=workspace_layout.second.address(runtime.workspace);
    const auto third=workspace_layout.third.address(runtime.workspace);
    const auto fourth=workspace_layout.fourth.address(runtime.workspace);
    const auto router_logits=workspace_layout.router_logits.address(runtime.workspace);
    const auto selected_device=workspace_layout.selected_device.address(runtime.workspace);
    const auto route_weights=workspace_layout.route_weights.address(runtime.workspace);
    const auto gpu_activated=workspace_layout.gpu_activated.address(runtime.workspace);
    // Small MTP batches are decode-shaped rather than prefill-shaped.  Their
    // k-quant matrix kernels are dominated by reconstructing weights against
    // f32 activations, while decode's DP4A path quantizes each activation once
    // and reuses it across q/k/v and the FFN projections.  Borrow the routed
    // activation region as transient Q8 scratch; expert execution consumes it
    // only after all dense projections for the layer have finished.
    // Dedicated regions rather than borrowing `gpu_activated`, which is sized
    // off top_k and so is empty on a dense model.
    const auto small_q8=workspace_layout.rows_q8.address(runtime.workspace);
    const auto small_q8_scales=
        workspace_layout.rows_q8_scales.address(runtime.workspace);
    std::uint64_t q8_cached_input=0;
    int q8_cached_width=0;
    const auto gpu_table_entries=static_cast<std::uint64_t>(rows)*top_k;
    const auto gpu_gate_table=workspace_layout.gpu_gate_table.address(runtime.workspace);
    const auto gpu_up_table=workspace_layout.gpu_up_table.address(runtime.workspace);
    const auto gpu_down_table=workspace_layout.gpu_down_table.address(runtime.workspace);
    const auto gpu_weight_table=workspace_layout.gpu_weight_table.address(runtime.workspace);
    // NVFP4 weight_scale_2 per route, mirroring the decode path: gate/up before
    // SiLU, the linear down scale folded into `up`.
    const auto gpu_gate_scale_table=workspace_layout.gpu_gate_scale_table.address(runtime.workspace);
    const auto gpu_up_scale_table=workspace_layout.gpu_up_scale_table.address(runtime.workspace);
    const auto gpu_count_table=workspace_layout.gpu_count_table.address(runtime.workspace);
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
        std::uint64_t matrix=staged?staged:runtime.device_tensors[index];
        const auto type=qwen_device_type(runtime,index);
        // Tensor cores where the format has an MMQ kernel; the dp4a tile
        // kernel is the fallback for the rest. Both are resolved before the
        // gate below, because whether either exists is what decides if a wide
        // MoE batch may enter the block at all.
        const char* mmq_kernel=nullptr;
        bool mmq_min=false;
        switch(type){
            case 16:mmq_kernel="iq2xxs_q8_mmq";break;
            case 17:mmq_kernel="iq2xs_q8_mmq";break;
            case 18:mmq_kernel="iq3xxs_q8_mmq";break;
            case 22:mmq_kernel="iq2s_q8_mmq";break;
            case 23:mmq_kernel="iq4xs_q8_mmq";break;
            case 11:mmq_kernel="q3k_q8_mmq";break;
            case 14:mmq_kernel="q6k_q8_mmq";break;
            case 10:mmq_kernel="q2k_q8_mmq";mmq_min=true;break;
            case 12:mmq_kernel="q4k_q8_mmq";mmq_min=true;break;
            case 13:mmq_kernel="q5k_q8_mmq";mmq_min=true;break;
            // IQ1_S carries a per-group delta and still lands here rather
            // than with the asymmetric three: its delta is +-1/8, so it
            // folds into the int8 weights exactly. See iq1s_q8_words.
            case 19:mmq_kernel="iq1s_q8_mmq";break;
            // IQ1_M folds the same way; its two sub-scales per 32 values
            // are the half-block split the symmetric kernels already carry.
            case 29:mmq_kernel="iq1m_q8_mmq";break;
            default:break;
        }
        const char* tiled_kernel=nullptr;
        switch(type){
            case 16:tiled_kernel="iq2xxs_q8_matmul_tiled";break;
            case 17:tiled_kernel="iq2xs_q8_matmul_tiled";break;
            case 18:tiled_kernel="iq3xxs_q8_matmul_tiled";break;
            case 22:tiled_kernel="iq2s_q8_matmul_tiled";break;
            case 23:tiled_kernel="iq4xs_q8_matmul_tiled";break;
            case 19:tiled_kernel="iq1s_q8_matmul_tiled";break;
            case 29:tiled_kernel="iq1m_q8_matmul_tiled";break;
            case 11:tiled_kernel="q3k_q8_matmul_tiled";break;
            case 14:tiled_kernel="q6k_q8_matmul_tiled";break;
            default:break;
        }
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
        if((experts>0?(rows<=8||batched_kernel):true)&&(input_size&255)==0){
            switch(type){
                case 10:q8_kernel="q2k_q8_matvec_transposed_warp";break;
                case 11:q8_kernel="q3k_q8_matvec_transposed_warp";break;
                case 12:q8_kernel="q4k_q8_matvec_transposed_warp";break;
                case 13:q8_kernel="q5k_q8_matvec_transposed_warp";break;
                case 14:q8_kernel="q6k_q8_matvec_transposed_warp";break;
                case 16:q8_kernel="iq2xxs_q8_matvec_transposed_warp";break;
                case 17:q8_kernel="iq2xs_q8_matvec_transposed_warp";break;
                case 18:q8_kernel="iq3xxs_q8_matvec_transposed_warp";break;
                case 22:q8_kernel="iq2s_q8_matvec_transposed_warp";break;
                case 19:q8_kernel="iq1s_q8_matvec_transposed_warp";break;
                case 29:q8_kernel="iq1m_q8_matvec_transposed_warp";break;
                default:break;
            }
        }
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
            const char* rows_kernel=nullptr;
            switch(type){
                case 16:rows_kernel="iq2xxs_q8_matvec_transposed_rows";break;
                case 17:rows_kernel="iq2xs_q8_matvec_transposed_rows";break;
                case 18:rows_kernel="iq3xxs_q8_matvec_transposed_rows";break;
                case 22:rows_kernel="iq2s_q8_matvec_transposed_rows";break;
                case 23:rows_kernel="iq4xs_q8_matvec_transposed_rows";break;
                case 11:rows_kernel="q3k_q8_matvec_transposed_rows";break;
                case 14:rows_kernel="q6k_q8_matvec_transposed_rows";break;
                case 10:rows_kernel="q2k_q8_matvec_transposed_rows";break;
                case 12:rows_kernel="q4k_q8_matvec_transposed_rows";break;
                case 13:rows_kernel="q5k_q8_matvec_transposed_rows";break;
                case 19:rows_kernel="iq1s_q8_matvec_transposed_rows";break;
                case 29:rows_kernel="iq1m_q8_matvec_transposed_rows";break;
                default:break;
            }
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
                const int mmq_threads=mmq_min?kQ8MmqMinThreads:kQ8MmqThreads;
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
        void*args[]={&matrix,&input,&output,&input_size,&output_size,&rows};
        switch(type){
            case 0:launch("qwen_f32_matmul_rows",output_size,rows,256,args);return;
            case 8:launch("q8_matmul_tiled",(output_size+31)/32,(static_cast<std::uint32_t>(rows)+31)/32,256,args);return;
            case 30:launch("bf16_matmul_rows",output_size,rows,256,args);return;
            case 10:launch("q2k_matmul_rows",output_size,(rows+3)/4,256,args);return;
            case 11:launch("q3k_matmul_rows",output_size,(rows+3)/4,256,args);return;
            case 12:launch("q4k_matmul_rows",output_size,(rows+3)/4,256,args);return;
            case 13:launch("q5k_matmul_rows",output_size,(rows+3)/4,256,args);return;
            case 14:launch("q6k_matmul_rows",output_size,(rows+3)/4,256,args);return;
            case 16:launch("iq2xxs_matmul_rows",output_size,(rows+3)/4,256,args);return;
            case 18:launch("iq3xxs_matmul_rows",output_size,(rows+3)/4,256,args);return;
            case 22:launch("iq2s_matmul_rows",output_size,(rows+3)/4,256,args);return;
            case 21:launch("iq3s_matmul_rows",output_size,(rows+3)/4,256,args);return;
            case 17:launch("iq2xs_matmul_rows",output_size,(rows+3)/4,256,args);return;
            case 23:launch("iq4xs_matmul_rows",output_size,(rows+3)/4,256,args);return;
            case 19:launch("iq1s_matmul_rows",output_size,(rows+3)/4,256,args);return;
            // IQ1_M: ssm_alpha/ssm_beta. Without this they fall to the
            // per-token fallback below -- 64 launches of a 48-block grid per
            // projection per layer -- which is pure launch latency and was the
            // bulk of the prefill "proj" phase. Batched, it is one launch.
            case 29:launch("iq1m_matmul_rows",output_size,(rows+3)/4,256,args);return;
            default:break;
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
    if(colibri_gpu_upload(token_device,input_tokens,rows*sizeof(std::uint32_t),runtime.stream)!=0)throw std::runtime_error("native Qwen rows token upload failed");
    const auto embedding=qwen_stage_embedding_rows(runtime,input_tokens,rows);
    // Staged rows arrive in chunk order, so they are gathered by the identity
    // permutation rather than by the real token ids in token_device.
    const auto embedding_index=runtime.embeddings_host_resident
        ?runtime.embedding_row_index:token_device;
    void*embedding_args[]={const_cast<std::uint64_t*>(&embedding),const_cast<std::uint64_t*>(&embedding_index),&hidden,&rows,const_cast<int*>(&hidden_size)};
    launch(qwen_embedding_kernel(qwen_device_type(runtime,runtime.token_embeddings),true),(hidden_size+255)/256,rows,256,embedding_args);
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
    std::vector<float>gpu_weights(gpu_table_entries);
    std::vector<float>gpu_gate_scales(gpu_table_entries),gpu_up_scales(gpu_table_entries);
    std::vector<std::uint64_t>gate_pointers(gpu_table_entries),up_pointers(gpu_table_entries),down_pointers(gpu_table_entries);
    std::vector<std::int32_t>gpu_counts(rows);
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
    for(std::uint32_t layer_number=0;layer_number<runtime.layers.size();++layer_number){
        if(runtime.cancelled)throw std::runtime_error("native Qwen runtime is cancelled");
        auto&layer=runtime.layers[layer_number];
        q8_cached_input=0;q8_cached_width=0;
        // The same GPU-event split (core / router / transfer) answers "is
        // route_wait real GPU work or a bubble?" for MTP verification, which is
        // the phase that currently stalls more than decode. Opt in with
        // COLIBRI_MTP_PROFILE=1; it accumulates into the prefill_gpu_* counters,
        // so a probe must snapshot them between the prefill and decode phases.
        static const bool verify_profile=
            std::getenv("COLIBRI_MTP_PROFILE")!=nullptr;
        const bool profile_prefill=
            (runtime.prefill_profile&&expert_policy.records_prefill_frequency())||
            (verify_profile&&
             expert_phase==colibri::v2::ExpertExecutionPhase::verification);
        if(profile_prefill&&colibri_gpu_event_record(runtime.prefill_layer_start_event,runtime.stream)!=0)
            throw std::runtime_error("native Qwen prefill layer-start profiling event failed");
        auto tensor=[&](std::size_t role){return runtime.device_tensors[layer.static_tensors.at(role)];};
        auto dense=[&](std::size_t role,std::uint64_t input,std::uint64_t output,int input_size,int output_size,std::uint64_t staged=0){dense_rows(layer.static_tensors.at(role),input,output,input_size,output_size,staged);};
        phase(layer_number,"pre");
        rms_rows(hidden,tensor(0),normalized);
        std::size_t moe_base=0;
        if(!layer.attention){
            int channels=static_cast<int>(runtime.model->tensors[layer.static_tensors[1]].shape[1]);
            int gate_elements=static_cast<int>(runtime.model->tensors[layer.static_tensors[2]].shape[1]);
            int value_heads=static_cast<int>(runtime.model->tensors[layer.static_tensors[8]].shape[0]);
            int head_dim=static_cast<int>(runtime.model->tensors[layer.static_tensors[9]].shape[0]);
            int value_dim=value_heads*head_dim;
            int key_heads=(channels-value_dim)/(2*head_dim);
            dense(1,normalized,first,hidden_size,channels);
            dense(2,normalized,second,hidden_size,gate_elements);
            dense(4,normalized,third,hidden_size,value_heads);
            auto beta_logits=third+static_cast<std::uint64_t>(rows)*value_heads*sizeof(float);
            dense(5,normalized,beta_logits,hidden_size,value_heads);
            int kernel_size=static_cast<int>(runtime.model->tensors[layer.static_tensors[6]].shape[0]);
            auto conv_state=runtime.state+layer.state_first;auto conv_weights=tensor(6);
            void*conv_args[]={const_cast<std::uint64_t*>(&first),&conv_weights,&conv_state,const_cast<std::uint64_t*>(&fourth),&rows,&channels,&kernel_size};
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
                    save("decay_logits",third,token_count*static_cast<std::size_t>(value_heads));
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
            void*recurrent_args[]={const_cast<std::uint64_t*>(&fourth),const_cast<std::uint64_t*>(&second),&beta_logits,const_cast<std::uint64_t*>(&third),&decay,&dt,&norm,&recurrent_state,const_cast<std::uint64_t*>(&first),&rows,&key_heads,&value_heads,&head_dim,const_cast<float*>(&epsilon)};
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
                    const_cast<std::uint64_t*>(&third),&decay,&dt,
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
                    const_cast<float*>(&epsilon)};
                launch("qwen_delta_norm_gate",static_cast<std::uint32_t>(rows),
                    static_cast<std::uint32_t>(value_heads),
                    static_cast<std::uint32_t>(colibri::v2::workspace::kDeltaDim),
                    gate_args);
            }else if(head_dim==128)launch("qwen_delta_recurrent_chunk",value_heads,1,128,recurrent_args);
            else launch("qwen_delta_recurrent_rows",value_heads,1,256,recurrent_args);
            phase(layer_number,"recurrent");
            dense(3,first,residual,value_dim,hidden_size);
            add_rows(residual,hidden);moe_base=10;
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
                int position=static_cast<int>(runtime.position)+token;
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
            float scale=1.0f/std::sqrt(static_cast<float>(head_dim));
            const char*cublas_prefill_env=
                std::getenv("COLIBRI_CUBLAS_PREFILL_ATTENTION");
            const std::uint64_t max_tokens=
                runtime.position+static_cast<std::uint64_t>(rows);
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
            const std::uint64_t score_row_bytes=
                static_cast<std::uint64_t>(heads)*max_tokens*sizeof(float);
            const std::uint64_t score_workspace_bytes=
                static_cast<std::uint64_t>(rows)*runtime.scratch_elements*sizeof(float);
            const int cublas_tile_rows=score_row_bytes
                ?static_cast<int>(std::min<std::uint64_t>(
                    16,score_workspace_bytes/score_row_bytes)):0;
            const std::uint64_t packed_query_bytes=device_align(
                static_cast<std::uint64_t>(16)*heads*head_dim*
                sizeof(std::uint16_t));
            const std::uint64_t packed_output_bytes=device_align(
                static_cast<std::uint64_t>(16)*heads*head_dim*sizeof(float));
            const bool cublas_workspace_ok=
                packed_query_bytes+packed_output_bytes<=
                static_cast<std::uint64_t>(heads)*
                    runtime.options.context_limit*sizeof(float);
            // The tensor-core prefill routines fold Qwen's per-channel sigmoid
            // gate in themselves, so Laguna, whose gate is a per-head softplus,
            // cannot use them and takes the warp path with its own gate kernel.
            const bool cublas_prefill_done=
                cublas_prefill_requested&&!laguna&&!layer.attention_window&&
                runtime.options.cache_type_k==1&&
                runtime.options.cache_type_v==1&&cublas_tile_rows>0&&
                cublas_workspace_ok&&
                colibri_gpu_attention_prefill_f16_cublas(
                    queries,gates,cache_keys,cache_values,
                    attention_scores,first,second,
                    attention_scores+packed_query_bytes,third,
                    runtime.stream,heads,kv_heads,head_dim,rows,capacity,
                    static_cast<int>(runtime.position),cublas_tile_rows,
                    scale,1)==0;
            if(cublas_prefill_done){
                // Tensor-core attention wrote the already-gated rows to third.
            }else if(!layer.attention_window&&head_dim%32==0&&head_dim<=256&&kv_fused_prefill_ok(runtime)){
                // One warp streams the KV cache once for 4 query rows: cache
                // traffic per chunk stops scaling with per-token passes, which
                // dominated long-context prefill.
                int base=static_cast<int>(runtime.position);
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
                         static_cast<int>(runtime.position),
                         stage_keys,stage_values)){
                // The staged window is dense and ring-free, so it is indexed
                // with capacity == window. cuBLAS runs ungated here because the
                // values are still rotated: the gate is a sigmoid and has to
                // wait until after the inverse rotation below.
                const int window=static_cast<int>(runtime.position)+rows;
                int base=static_cast<int>(runtime.position),staged_capacity=window;
                int vectors=rows*heads,value_stream=1,elements=rows*query_elements;
                const bool staged_cublas=
                    cublas_prefill_requested&&cublas_tile_rows>0&&
                    cublas_workspace_ok&&
                    colibri_gpu_attention_prefill_f16_cublas(
                        queries,gates,stage_keys,stage_values,
                        attention_scores,first,second,
                        attention_scores+packed_query_bytes,third,
                        runtime.stream,heads,kv_heads,head_dim,rows,window,
                        base,cublas_tile_rows,scale,0)==0;
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
                int position=static_cast<int>(runtime.position)+token;
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
            add_rows(residual,hidden);moe_base=qwen_ffn_base(runtime,layer);
        }
        if(profile_prefill&&colibri_gpu_event_record(runtime.prefill_core_end_event,runtime.stream)!=0)
            throw std::runtime_error("native Qwen prefill core profiling event failed");
        rms_rows(residual,tensor(moe_base),normalized);
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
                add_rows(residual,third);std::swap(hidden,residual);
                continue;
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
                add_rows(residual,third);std::swap(hidden,residual);
                continue;
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
            add_rows(residual,third);std::swap(hidden,residual);
            continue;
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
        const auto activated_bytes=
            static_cast<std::uint64_t>(rows)*top_k*intermediate*sizeof(float);
        const auto down_bytes=
            static_cast<std::uint64_t>(rows)*top_k*hidden_size*sizeof(float);
        const auto output_bytes=
            static_cast<std::uint64_t>(rows)*hidden_size*sizeof(float);
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
        if(colibri_gpu_download(selected_host,selected_device,selected_bytes,runtime.stream)!=0||colibri_gpu_download(weights_host,route_weights,weights_bytes,runtime.stream)!=0||colibri_gpu_download(input_host,normalized,input_bytes,runtime.stream)!=0)
            throw std::runtime_error("native Qwen rows forward route transfer failed");
        if(colibri_gpu_event_record(runtime.route_event,runtime.stream)!=0)throw std::runtime_error("native Qwen rows forward route event failed");
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
        const auto route_wait_started=std::chrono::steady_clock::now();
        if(colibri_gpu_event_sync(runtime.route_event)!=0)throw std::runtime_error("native Qwen rows forward route synchronization failed");
        if(profile_prefill){
            float core_ms=0.0f,router_ms=0.0f,transfer_ms=0.0f;
            if(colibri_gpu_event_elapsed(runtime.prefill_layer_start_event,runtime.prefill_core_end_event,&core_ms)!=0||
               colibri_gpu_event_elapsed(runtime.prefill_core_end_event,runtime.prefill_router_end_event,&router_ms)!=0||
               colibri_gpu_event_elapsed(runtime.prefill_router_end_event,runtime.route_event,&transfer_ms)!=0)
                throw std::runtime_error("native Qwen prefill profiling measurement failed");
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
        const auto pager_started=std::chrono::steady_clock::now();
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
        const bool gpu_table_path=!expert_policy.hybrid_prefill_cpu;
        if(gpu_table_path){
            std::fill(gpu_weights.begin(),gpu_weights.end(),0.0f);
            std::fill(gpu_gate_scales.begin(),gpu_gate_scales.end(),1.0f);
            std::fill(gpu_up_scales.begin(),gpu_up_scales.end(),1.0f);
            std::fill(gpu_counts.begin(),gpu_counts.end(),0);
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
                    if(rows_may_admit&&runtime.dma_paging)
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
        bool any_gpu=false;
        if(gpu_rows_allowed)
            for(int token=0;token<rows;++token)
                any_gpu=any_gpu||gpu_counts[token]>0;
        if(gpu_table_path){
            if(colibri_gpu_memset(fourth,0,output_bytes,runtime.stream)!=0)throw std::runtime_error("native Qwen rows forward GPU expert reset failed");
            const auto pointer_bytes=gpu_table_entries*sizeof(std::uint64_t);
            const auto scale_bytes=gpu_table_entries*sizeof(float);
            auto*gate_host=staging+host_layout.gate_table.offset;
            auto*up_host=staging+host_layout.up_table.offset;
            auto*down_pointer_host=staging+host_layout.down_table.offset;
            auto*gpu_weight_host=staging+host_layout.weight_table.offset;
            auto*gpu_gate_scale_host=staging+host_layout.gate_scale_table.offset;
            auto*gpu_up_scale_host=staging+host_layout.up_scale_table.offset;
            auto*gpu_count_host=staging+host_layout.counts.offset;
            std::memcpy(gate_host,gate_pointers.data(),pointer_bytes);
            std::memcpy(up_host,up_pointers.data(),pointer_bytes);
            std::memcpy(down_pointer_host,down_pointers.data(),pointer_bytes);
            std::memcpy(gpu_weight_host,gpu_weights.data(),scale_bytes);
            std::memcpy(gpu_gate_scale_host,gpu_gate_scales.data(),scale_bytes);
            std::memcpy(gpu_up_scale_host,gpu_up_scales.data(),scale_bytes);
            std::memcpy(gpu_count_host,gpu_counts.data(),rows*sizeof(std::int32_t));
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
        const auto cpu_expert_started=std::chrono::steady_clock::now();
        qwen_cpu_moe_rows(runtime,layer,selected_host,weights_host,rows,top_k,input_host,activated_host,down_host,output_host);
        if(expert_policy.records_prefill_frequency())
            runtime.prefill_expert_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now()-cpu_expert_started).count();
        if(colibri_gpu_upload(first,output_host,output_bytes,runtime.stream)!=0)throw std::runtime_error("native Qwen rows forward expert upload failed");
        runtime.expert_page_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-pager_started).count();
        if(gpu_table_path){add_rows(fourth,first);add_rows(third,fourth);}
        else add_rows(third,first);
        add_rows(residual,third);std::swap(hidden,residual);
    }
    if(output_tokens){
        rms_rows(hidden,runtime.device_tensors[runtime.final_norm],normalized);
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
