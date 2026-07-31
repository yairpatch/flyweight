from colibri_next.v2 import V2Model
from pathlib import Path

m = V2Model(Path(r'C:\Users\thegr\Downloads\Qwen3.6-35B-A3B-UD-Q6_K.gguf'))
cfg = m.config
print('layers:', cfg['layer_count'])
print('hidden_size:', cfg['hidden_size'])
print('intermediate_size:', cfg['intermediate_size'])
print('expert_count:', cfg['expert_count'])

for t in m.tensors():
    n = t['name']
    if 'ffn_down_exps' in n or 'ffn_gate_exps' in n or 'ffn_up_exps' in n:
        print(n, 'type=', t['ggml_type'], 'shape=', t['shape'])
