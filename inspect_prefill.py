import re
content = open('src/colibri_next/v2_server.py','r',encoding='utf-8').read()
lines = content.split('\n')
count = 0
for i, line in enumerate(lines):
    if any(k in line for k in ['prefill','prefill_tokens','decode_token','batch_prompt','n_ctx','chunk']):
        print(i+1, line.strip()[:220])
        count += 1
        if count >= 30:
            break
