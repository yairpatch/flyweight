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
    const float epsilon=runtime.model->config.rms_norm_epsilon?runtime.model->config.rms_norm_epsilon:1.0e-6f;
    for(int row=0;row<rows;++row)
        if(input_tokens[row]>=runtime.model->config.vocabulary_size)
            throw std::runtime_error("native Qwen input token is out of range");
    std::uint64_t cursor=runtime.workspace;
    const auto workspace_end=runtime.workspace+runtime.workspace_bytes;
    auto take=[&](std::uint64_t bytes){const auto result=cursor;cursor+=device_align(bytes);if(cursor>workspace_end)throw std::runtime_error("native Qwen rows forward workspace is too small");return result;};
    std::uint64_t hidden=take(static_cast<std::uint64_t>(rows)*hidden_size*sizeof(float));
    std::uint64_t residual=take(static_cast<std::uint64_t>(rows)*hidden_size*sizeof(float));
    const auto normalized=take(static_cast<std::uint64_t>(rows)*hidden_size*sizeof(float));
    const auto first=take(static_cast<std::uint64_t>(rows)*runtime.scratch_elements*sizeof(float));
    const auto second=take(static_cast<std::uint64_t>(rows)*runtime.scratch_elements*sizeof(float));
    const auto third=take(static_cast<std::uint64_t>(rows)*runtime.scratch_elements*sizeof(float));
    const auto fourth=take(static_cast<std::uint64_t>(rows)*runtime.scratch_elements*sizeof(float));
    const auto router_logits=take(static_cast<std::uint64_t>(rows)*experts*sizeof(float));
    const auto selected_device=take(static_cast<std::uint64_t>(rows)*top_k*sizeof(std::int32_t));
    const auto route_weights=take(static_cast<std::uint64_t>(rows)*top_k*sizeof(float));
    const auto gpu_activated=take(static_cast<std::uint64_t>(rows)*top_k*intermediate*sizeof(float));
    const auto gpu_table_entries=static_cast<std::uint64_t>(rows)*top_k;
    const auto gpu_gate_table=take(gpu_table_entries*sizeof(std::uint64_t));
    const auto gpu_up_table=take(gpu_table_entries*sizeof(std::uint64_t));
    const auto gpu_down_table=take(gpu_table_entries*sizeof(std::uint64_t));
    const auto gpu_weight_table=take(gpu_table_entries*sizeof(float));
    const auto gpu_count_table=take(static_cast<std::uint64_t>(rows)*sizeof(std::int32_t));
    const auto token_device=take(static_cast<std::uint64_t>(rows)*sizeof(std::uint32_t));
    const auto winners=take(static_cast<std::uint64_t>(rows)*sizeof(std::uint64_t));
    const auto attention_scores=take(static_cast<std::uint64_t>(runtime.model->config.attention_heads)*runtime.options.context_limit*sizeof(float));
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
    auto add_rows=[&](std::uint64_t target,std::uint64_t source){float scale=1.0f;int count=rows*hidden_size;void*args[]={&target,&source,&scale,&count};launch("scaled_add",(count+255)/256,1,256,args);};
    if(colibri_gpu_upload(token_device,input_tokens,rows*sizeof(std::uint32_t),runtime.stream)!=0)throw std::runtime_error("native Qwen rows token upload failed");
    const auto embedding=runtime.device_tensors[runtime.token_embeddings];
    void*embedding_args[]={const_cast<std::uint64_t*>(&embedding),const_cast<std::uint64_t*>(&token_device),&hidden,&rows,const_cast<int*>(&hidden_size)};
    launch("qwen_q8_embedding_rows",(hidden_size+255)/256,rows,256,embedding_args);
    auto*staging=static_cast<std::uint8_t*>(runtime.host_staging);
    const bool prune_experts=runtime.options.moe_device!=0&&
        (runtime.options.expert_top_k>0||(runtime.options.expert_top_p>0.0f&&runtime.options.expert_top_p<1.0f));
    std::vector<float>gpu_weights(gpu_table_entries);
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
        auto tensor=[&](std::size_t role){return runtime.device_tensors[layer.static_tensors.at(role)];};
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
            q8_rows(tensor(1),normalized,first,hidden_size,channels);
            q8_rows(tensor(2),normalized,second,hidden_size,gate_elements);
            f32_rows(tensor(4),normalized,third,hidden_size,value_heads);
            auto beta_logits=third+static_cast<std::uint64_t>(rows)*value_heads*sizeof(float);
            f32_rows(tensor(5),normalized,beta_logits,hidden_size,value_heads);
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
            q8_rows(tensor(3),first,residual,value_dim,hidden_size);
            add_rows(residual,hidden);moe_base=10;
            phase(layer_number,"outproj");
        }else{
            const int heads=static_cast<int>(runtime.model->config.attention_heads);
            const int kv_heads=static_cast<int>(runtime.model->config.attention_kv_heads);
            const int head_dim=static_cast<int>(runtime.model->tensors[layer.static_tensors[2]].shape[1]/kv_heads);
            const int query_elements=heads*head_dim,q_size=query_elements*2,kv_size=kv_heads*head_dim;
            q8_rows(tensor(1),normalized,first,hidden_size,q_size);
            q8_rows(tensor(2),normalized,second,hidden_size,kv_size);
            q8_rows(tensor(3),normalized,third,hidden_size,kv_size);
            const int rotary=static_cast<int>(runtime.model->config.rotary_dimension?runtime.model->config.rotary_dimension:head_dim);
            const float theta=runtime.model->config.rope_freq_base?runtime.model->config.rope_freq_base:1000000.0f;
            auto qnorm=tensor(5),knorm=tensor(6);
            const auto queries=fourth,gates=fourth+static_cast<std::uint64_t>(rows)*query_elements*sizeof(float);
            auto cache_keys=runtime.state+layer.state_first,cache_values=runtime.state+layer.state_second;
            int capacity=static_cast<int>(runtime.options.context_limit);
            for(int token=0;token<rows;++token){
                auto projected=first+static_cast<std::uint64_t>(token)*q_size*sizeof(float);
                auto query=queries+static_cast<std::uint64_t>(token)*query_elements*sizeof(float);
                auto gate=gates+static_cast<std::uint64_t>(token)*query_elements*sizeof(float);
                int position=static_cast<int>(runtime.position)+token;
                void*q_args[]={&projected,&qnorm,&query,&gate,const_cast<int*>(&heads),const_cast<int*>(&head_dim),const_cast<int*>(&rotary),&position,const_cast<float*>(&theta),const_cast<float*>(&epsilon)};
                launch("qwen_attention_query",heads,1,256,q_args);
                auto projected_key=second+static_cast<std::uint64_t>(token)*kv_size*sizeof(float);
                auto key=first+static_cast<std::uint64_t>(token)*kv_size*sizeof(float);
                void*k_args[]={&projected_key,&knorm,&key,const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),const_cast<int*>(&rotary),&position,const_cast<float*>(&theta),const_cast<float*>(&epsilon)};
                launch("qwen_attention_key",kv_heads,1,256,k_args);
                auto value=third+static_cast<std::uint64_t>(token)*kv_size*sizeof(float);
                void*append_args[]={&key,&value,&cache_keys,&cache_values,const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&position,&capacity};
                launch(kv_append_kernel(runtime),kv_heads,1,256,append_args);
            }
            float scale=1.0f/std::sqrt(static_cast<float>(head_dim));
            if(head_dim%32==0&&head_dim<=256){
                // One warp streams the KV cache once for 4 query rows: cache
                // traffic per chunk stops scaling with per-token passes, which
                // dominated long-context prefill.
                int base=static_cast<int>(runtime.position);
                void*attention_args[]={const_cast<std::uint64_t*>(&queries),&cache_keys,&cache_values,const_cast<std::uint64_t*>(&second),const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&base,&rows,&capacity,&scale};
                launch(kv_prefill_kernel(runtime),heads,(static_cast<std::uint32_t>(rows)+31)/32,256,attention_args);
                int elements=rows*query_elements;
                void*gate_args[]={const_cast<std::uint64_t*>(&second),const_cast<std::uint64_t*>(&gates),const_cast<std::uint64_t*>(&third),&elements};
                launch("qwen_attention_gate",(elements+255)/256,1,256,gate_args);
            }else for(int token=0;token<rows;++token){
                auto query=queries+static_cast<std::uint64_t>(token)*query_elements*sizeof(float);
                auto gate=gates+static_cast<std::uint64_t>(token)*query_elements*sizeof(float);
                int position=static_cast<int>(runtime.position)+token;
                int tokens=position+1;
                void*score_args[]={&query,&cache_keys,const_cast<std::uint64_t*>(&attention_scores),const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&tokens,&capacity,&scale};
                launch(kv_scores_kernel(runtime),heads,(tokens+255)/256,256,score_args);
                auto attended=second+static_cast<std::uint64_t>(token)*query_elements*sizeof(float);
                void*value_args[]={const_cast<std::uint64_t*>(&attention_scores),&cache_values,&attended,const_cast<int*>(&heads),const_cast<int*>(&kv_heads),const_cast<int*>(&head_dim),&tokens,&capacity};
                launch(kv_values_kernel(runtime),heads,1,256,value_args);
                auto gated=third+static_cast<std::uint64_t>(token)*query_elements*sizeof(float);int elements=query_elements;
                void*token_gate_args[]={&attended,&gate,&gated,&elements};launch("qwen_attention_gate",(elements+255)/256,1,256,token_gate_args);
            }
            q8_rows(tensor(4),third,residual,query_elements,hidden_size);
            add_rows(residual,hidden);moe_base=7;
        }
        rms_rows(residual,tensor(moe_base),normalized);
        phase(layer_number,"moerms");
        f32_rows(tensor(moe_base+1),normalized,router_logits,hidden_size,experts);
        phase(layer_number,"router");
        void*route_args[]={const_cast<std::uint64_t*>(&router_logits),const_cast<std::uint64_t*>(&selected_device),const_cast<std::uint64_t*>(&route_weights),&rows,const_cast<int*>(&experts),const_cast<int*>(&top_k)};
        launch("route_topk_rows",rows,1,256,route_args,experts*sizeof(float));
        const auto selected_bytes=static_cast<std::uint64_t>(rows)*top_k*sizeof(std::int32_t);
        const auto weights_host_offset=device_align(selected_bytes);
        const auto weights_bytes=static_cast<std::uint64_t>(rows)*top_k*sizeof(float);
        const auto input_host_offset=weights_host_offset+device_align(weights_bytes);
        const auto input_bytes=static_cast<std::uint64_t>(rows)*hidden_size*sizeof(float);
        const auto activated_host_offset=input_host_offset+device_align(input_bytes);
        const auto activated_bytes=static_cast<std::uint64_t>(rows)*top_k*intermediate*sizeof(float);
        const auto down_host_offset=activated_host_offset+device_align(activated_bytes);
        const auto down_bytes=static_cast<std::uint64_t>(rows)*top_k*hidden_size*sizeof(float);
        const auto output_host_offset=down_host_offset+device_align(down_bytes);
        const auto output_bytes=static_cast<std::uint64_t>(rows)*hidden_size*sizeof(float);
        if(output_host_offset+output_bytes>runtime.host_staging_bytes)throw std::runtime_error("native Qwen rows forward host workspace overflow");
        auto*selected_host=reinterpret_cast<std::int32_t*>(staging);
        auto*weights_host=reinterpret_cast<float*>(staging+weights_host_offset);
        auto*input_host=reinterpret_cast<float*>(staging+input_host_offset);
        auto*activated_host=reinterpret_cast<float*>(staging+activated_host_offset);
        auto*down_host=reinterpret_cast<float*>(staging+down_host_offset);
        auto*output_host=reinterpret_cast<float*>(staging+output_host_offset);
        if(colibri_gpu_download(selected_host,selected_device,selected_bytes,runtime.stream)!=0||colibri_gpu_download(weights_host,route_weights,weights_bytes,runtime.stream)!=0||colibri_gpu_download(input_host,normalized,input_bytes,runtime.stream)!=0)
            throw std::runtime_error("native Qwen rows forward route transfer failed");
        if(colibri_gpu_event_record(runtime.route_event,runtime.stream)!=0)throw std::runtime_error("native Qwen rows forward route event failed");
        // Shared expert gate/up as two tiled GEMMs into the halves of the
        // `first` scratch (silu_mul expects gate at [0, n) and up at [n, 2n)
        // of one buffer), then the elementwise swiglu combine.
        const auto swiglu_elements=static_cast<std::uint64_t>(rows)*intermediate;
        q8_rows(tensor(moe_base+2),normalized,first,hidden_size,intermediate);
        q8_rows(tensor(moe_base+3),normalized,first+swiglu_elements*sizeof(float),hidden_size,intermediate);
        {
            int count=static_cast<int>(swiglu_elements);
            void*silu_args[]={const_cast<std::uint64_t*>(&first),const_cast<std::uint64_t*>(&second),&count};
            launch("silu_mul",(static_cast<std::uint32_t>(count)+255)/256,1,256,silu_args);
        }
        q8_rows(tensor(moe_base+4),second,third,intermediate,hidden_size);
        auto shared_scale=tensor(moe_base+5);void*scale_args[]={const_cast<std::uint64_t*>(&normalized),&shared_scale,const_cast<std::uint64_t*>(&third),&rows,const_cast<int*>(&hidden_size)};
        launch("qwen_shared_scale_rows",rows,1,256,scale_args);
        const auto route_wait_started=std::chrono::steady_clock::now();
        if(colibri_gpu_event_sync(runtime.route_event)!=0)throw std::runtime_error("native Qwen rows forward route synchronization failed");
        const auto route_wait_elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-route_wait_started).count();
        runtime.route_wait_nanoseconds+=route_wait_elapsed;
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
        std::fill(gpu_weights.begin(),gpu_weights.end(),0.0f);
        std::fill(gpu_counts.begin(),gpu_counts.end(),0);
        if(runtime.options.moe_device!=1&&!runtime.expert_slots.empty()){
            for(int token=0;token<rows;++token)for(int rank=0;rank<top_k;++rank){
                const int route=token*top_k+rank;
                if(weights_host[route]==0.0f)continue;
                const int expert=selected_host[route];
                const auto cache_key=(static_cast<std::uint64_t>(layer_number)<<32)|static_cast<std::uint32_t>(expert);
                const auto resident=runtime.expert_residency.find(cache_key);
                if(resident==runtime.expert_residency.end()){++runtime.expert_cache_misses;continue;}
                ++runtime.expert_cache_hits;
                const auto slot_index=resident->second;
                runtime.expert_slots[slot_index].last_used=++runtime.expert_clock;
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
                weights_host[route]=0.0f;
            }
        }
        if(colibri_gpu_memset(fourth,0,output_bytes,runtime.stream)!=0)throw std::runtime_error("native Qwen rows forward GPU expert reset failed");
        const auto table_host_offset=output_host_offset+device_align(output_bytes);
        const auto pointer_bytes=gpu_table_entries*sizeof(std::uint64_t);
        const auto gpu_weights_host_offset=table_host_offset+3*device_align(pointer_bytes);
        const auto gpu_counts_host_offset=gpu_weights_host_offset+device_align(gpu_table_entries*sizeof(float));
        const auto table_host_end=gpu_counts_host_offset+static_cast<std::uint64_t>(rows)*sizeof(std::int32_t);
        if(table_host_end>runtime.host_staging_bytes)throw std::runtime_error("native Qwen rows forward pointer workspace overflow");
        auto*gate_host=staging+table_host_offset;
        auto*up_host=gate_host+device_align(pointer_bytes);
        auto*down_pointer_host=up_host+device_align(pointer_bytes);
        auto*gpu_weight_host=staging+gpu_weights_host_offset;
        auto*gpu_count_host=staging+gpu_counts_host_offset;
        std::memcpy(gate_host,gate_pointers.data(),pointer_bytes);
        std::memcpy(up_host,up_pointers.data(),pointer_bytes);
        std::memcpy(down_pointer_host,down_pointers.data(),pointer_bytes);
        std::memcpy(gpu_weight_host,gpu_weights.data(),gpu_table_entries*sizeof(float));
        std::memcpy(gpu_count_host,gpu_counts.data(),rows*sizeof(std::int32_t));
        if(colibri_gpu_upload(gpu_gate_table,gate_host,pointer_bytes,runtime.stream)!=0||
           colibri_gpu_upload(gpu_up_table,up_host,pointer_bytes,runtime.stream)!=0||
           colibri_gpu_upload(gpu_down_table,down_pointer_host,pointer_bytes,runtime.stream)!=0||
           colibri_gpu_upload(gpu_weight_table,gpu_weight_host,gpu_table_entries*sizeof(float),runtime.stream)!=0||
           colibri_gpu_upload(gpu_count_table,gpu_count_host,rows*sizeof(std::int32_t),runtime.stream)!=0)
            throw std::runtime_error("native Qwen rows forward pointer upload failed");
        bool any_gpu=false;for(int token=0;token<rows;++token)any_gpu=any_gpu||gpu_counts[token]>0;
        if(any_gpu){
            void*gate_up_args[]={const_cast<std::uint64_t*>(&gpu_gate_table),const_cast<std::uint64_t*>(&gpu_up_table),const_cast<std::uint64_t*>(&gpu_count_table),const_cast<std::uint64_t*>(&normalized),const_cast<std::uint64_t*>(&gpu_activated),const_cast<int*>(&hidden_size),const_cast<int*>(&intermediate),const_cast<int*>(&top_k),&rows};
            launch("q5k_grouped_swiglu_rows",intermediate,rows*top_k,256,gate_up_args);
            const auto down_type=runtime.model->tensors[layer.expert_tensors[2]].type;
            void*down_args[]={const_cast<std::uint64_t*>(&gpu_down_table),const_cast<std::uint64_t*>(&gpu_activated),const_cast<std::uint64_t*>(&fourth),const_cast<std::uint64_t*>(&gpu_weight_table),const_cast<std::uint64_t*>(&gpu_count_table),const_cast<int*>(&intermediate),const_cast<int*>(&hidden_size),const_cast<int*>(&top_k),&rows};
            launch(down_type==8?"q8_grouped_accumulate_rows":"q6k_grouped_accumulate_rows",hidden_size,rows,256,down_args);
        }
        qwen_cpu_moe_rows(runtime,layer,selected_host,weights_host,rows,top_k,input_host,activated_host,down_host,output_host);
        if(colibri_gpu_upload(first,output_host,output_bytes,runtime.stream)!=0)throw std::runtime_error("native Qwen rows forward expert upload failed");
        runtime.expert_page_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-pager_started).count();
        add_rows(fourth,first);add_rows(third,fourth);add_rows(residual,third);std::swap(hidden,residual);
    }
    if(output_tokens){
        rms_rows(hidden,runtime.device_tensors[runtime.final_norm],normalized);
        if(colibri_gpu_memset(winners,0,rows*sizeof(std::uint64_t),runtime.stream)!=0)throw std::runtime_error("native Qwen rows forward argmax reset failed");
        int vocabulary=static_cast<int>(runtime.model->config.vocabulary_size);auto lm_head=runtime.device_tensors[runtime.lm_head];
        void*argmax_args[]={&lm_head,const_cast<std::uint64_t*>(&normalized),const_cast<std::uint64_t*>(&winners),const_cast<int*>(&hidden_size),&vocabulary,&rows};
        launch("qwen_q8_lm_head_argmax_rows",(vocabulary+7)/8,rows,256,argmax_args);
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
}
