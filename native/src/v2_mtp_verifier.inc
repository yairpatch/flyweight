// Batched rows forward over the shared layer stack. Two callers:
//   - MTP verification (qwen_verify_target_rows): small row counts, needs the
//     per-row LM-head argmax and the final hidden states.
//   - Chunked prefill (qwen_prefill_rows): large row counts, skips the LM
//     head entirely (only the last prompt token needs logits, and it runs
//     through the regular single-token decode).
// Neither advances runtime.position/processed_tokens; callers own that.
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
    auto launch=[&](const char*name,std::uint32_t gx,std::uint32_t gy,std::uint32_t bx,void**args,std::uint32_t shared=0){
        if(colibri_gpu_launch_named(name,gx,gy,bx,shared,runtime.stream,args)!=0)
            throw std::runtime_error(std::string("native Qwen rows forward CUDA kernel failed: ")+name);
    };
    auto rms_rows=[&](std::uint64_t input,std::uint64_t weights,std::uint64_t output){
        int one_centered=0;void*args[]={&input,&weights,&output,&rows,const_cast<int*>(&hidden_size),const_cast<float*>(&epsilon),&one_centered};
        launch("rms_norm_rows",rows,1,256,args);
    };
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
    auto dense_rows=[&](std::size_t index,std::uint64_t input,std::uint64_t output,int input_size,int output_size){
        std::uint64_t matrix=runtime.device_tensors[index];
        const auto type=qwen_device_type(runtime,index);
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
    if(colibri_gpu_upload(token_device,input_tokens,rows*sizeof(std::uint32_t),runtime.stream)!=0)throw std::runtime_error("native Qwen rows token upload failed");
    const auto embedding=qwen_stage_embedding_rows(runtime,input_tokens,rows);
    // Staged rows arrive in chunk order, so they are gathered by the identity
    // permutation rather than by the real token ids in token_device.
    const auto embedding_index=runtime.embeddings_host_resident
        ?runtime.embedding_row_index:token_device;
    void*embedding_args[]={const_cast<std::uint64_t*>(&embedding),const_cast<std::uint64_t*>(&embedding_index),&hidden,&rows,const_cast<int*>(&hidden_size)};
    launch(qwen_embedding_kernel(qwen_device_type(runtime,runtime.token_embeddings),true),(hidden_size+255)/256,rows,256,embedding_args);
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
        const bool profile_prefill=
            runtime.prefill_profile&&expert_policy.records_prefill_frequency();
        if(profile_prefill&&colibri_gpu_event_record(runtime.prefill_layer_start_event,runtime.stream)!=0)
            throw std::runtime_error("native Qwen prefill layer-start profiling event failed");
        auto tensor=[&](std::size_t role){return runtime.device_tensors[layer.static_tensors.at(role)];};
        auto dense=[&](std::size_t role,std::uint64_t input,std::uint64_t output,int input_size,int output_size){dense_rows(layer.static_tensors.at(role),input,output,input_size,output_size);};
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
            void*recurrent_args[]={const_cast<std::uint64_t*>(&fourth),const_cast<std::uint64_t*>(&second),&beta_logits,const_cast<std::uint64_t*>(&third),&decay,&dt,&norm,&recurrent_state,const_cast<std::uint64_t*>(&first),&rows,&key_heads,&value_heads,&head_dim,const_cast<float*>(&epsilon)};
            if(head_dim==128)launch("qwen_delta_recurrent_chunk",value_heads,1,128,recurrent_args);
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
            const int heads=laguna
                ?static_cast<int>(layer.attention_heads)
                :static_cast<int>(runtime.model->config.attention_heads);
            const int kv_heads=static_cast<int>(runtime.model->config.attention_kv_heads);
            const int head_dim=static_cast<int>(runtime.model->tensors[layer.static_tensors[2]].shape[1]/kv_heads);
            const int query_elements=heads*head_dim;
            const int q_size=laguna?query_elements:query_elements*2;
            const int kv_size=kv_heads*head_dim;
            dense(1,normalized,first,hidden_size,q_size);
            dense(2,normalized,second,hidden_size,kv_size);
            dense(3,normalized,third,hidden_size,kv_size);
            const int rotary=laguna
                ?static_cast<int>(layer.rotary_dim)
                :static_cast<int>(runtime.model->config.rotary_dimension?runtime.model->config.rotary_dimension:head_dim);
            const float theta=laguna
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
            if(laguna)dense(7,normalized,gates,hidden_size,heads);
            auto cache_keys=runtime.state+layer.state_first,cache_values=runtime.state+layer.state_second;
            int capacity=static_cast<int>(layer.cache_capacity);
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
            add_rows(residual,hidden);moe_base=laguna?8:7;
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
            if(layer.ffn_on_host){
                // Spilled block: its weights are not on the device at all, so
                // run the SwiGLU on the host one row at a time through the
                // single-token scratch.
                auto*scratch=static_cast<float*>(runtime.dense_host);
                float*host_input=scratch;
                float*host_output=host_input+hidden_size;
                const auto host_started=std::chrono::steady_clock::now();
                for(int token=0;token<rows;++token){
                    const auto row_in=normalized+static_cast<std::uint64_t>(token)*hidden_size*sizeof(float);
                    const auto row_out=third+static_cast<std::uint64_t>(token)*hidden_size*sizeof(float);
                    if(colibri_gpu_download(host_input,row_in,hidden_size*sizeof(float),runtime.stream)!=0||
                       colibri_gpu_stream_sync(runtime.stream)!=0)
                        throw std::runtime_error("native dense host FFN rows input transfer failed");
                    qwen_cpu_dense_ffn(runtime,layer,host_input,host_output);
                    if(colibri_gpu_upload(row_out,host_output,hidden_size*sizeof(float),runtime.stream)!=0)
                        throw std::runtime_error("native dense host FFN rows output transfer failed");
                }
                runtime.dense_host_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now()-host_started).count();
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
                if(resident==runtime.expert_residency.end()){++runtime.expert_cache_misses;continue;}
                auto& slot=runtime.expert_slots[resident->second];
                record_expert_cache_hit(runtime,slot);
                const auto slot_index=resident->second;
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
        void*argmax_args[]={&lm_head,const_cast<std::uint64_t*>(&normalized),const_cast<std::uint64_t*>(&winners),const_cast<int*>(&hidden_size),&vocabulary,&rows};
        if(runtime.lm_head_type!=8&&runtime.lm_head_type!=30)
            throw std::runtime_error(
                "native Qwen rows LM head type is unsupported: "+std::to_string(runtime.lm_head_type));
        launch(runtime.lm_head_type==30?"qwen_bf16_lm_head_argmax_rows":"qwen_q8_lm_head_argmax_rows",(vocabulary+7)/8,rows,256,argmax_args);
        auto*packed=reinterpret_cast<std::uint64_t*>(staging);
        if(colibri_gpu_download(packed,winners,rows*sizeof(std::uint64_t),runtime.stream)!=0||colibri_gpu_stream_sync(runtime.stream)!=0)throw std::runtime_error("native Qwen rows forward output transfer failed");
        for(int row=0;row<rows;++row)output_tokens[row]=0xffffffffu-static_cast<std::uint32_t>(packed[row]);
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
    qwen_forward_rows(runtime,tokens,rows,nullptr,nullptr);
    runtime.processed_tokens.insert(runtime.processed_tokens.end(),tokens,tokens+rows);
    runtime.position+=static_cast<std::uint64_t>(rows);
    runtime.decode_calls+=static_cast<std::uint64_t>(rows);
    runtime.decode_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count();
    ++runtime.prefill_calls;
    runtime.prefill_tokens+=static_cast<std::uint64_t>(rows);
    runtime.prefill_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-started).count();
}
