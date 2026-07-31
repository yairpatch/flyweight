import re
content = open('src/colibri_next/v2_server.py','r',encoding='utf-8').read()
lines = content.split('\n')
count = 0
for i, line in enumerate(lines):
    if any(k in line for k in ['decode_tokens','decode_token_bytes','forward_token','prefill_tokens','prompt_tokens','n_gpu_layers','n_ctx','gpu_layers','batch_tokens','batched_decode','prefill_checkpoint','max_new_tokens']):
        print(i+1, line.strip()[:220])
        count += 1
        if count >= 30:
            break
