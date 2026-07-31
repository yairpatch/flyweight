import re
content = open('src/colibri_next/v2_server.py','r',encoding='utf-8').read()
lines = content.split('\n')
count = 0
for i, line in enumerate(lines):
    if any(k in line for k in ['gpu_cache','expert_pag','next_layer','cache_type','prefill_check','prefill_cache','parallel_seq','expert_top_k','moe_device','tp_size','tensor_split','n_gpu_layers']):
        print(i+1, line.strip()[:220])
        count += 1
        if count >= 30:
            break
