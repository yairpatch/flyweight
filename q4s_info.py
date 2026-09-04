from flyweight.v2 import V2Model
from pathlib import Path

m = V2Model(Path(r'C:\Users\thegr\Downloads\Qwen3.6-35B-A3B-UD-Q4_K_S.gguf'))
cfg = m.config
print('layers:', cfg['layer_count'])
print('hidden_size:', cfg['hidden_size'])
print('intermediate_size:', cfg['intermediate_size'])
print('expert_count:', cfg['expert_count'])
print('---')

types = {}
for t in m.tensors():
    n = t['name']
    if 'ffn_down_exps' in n:
        gt = t['ggml_type']
        types[gt] = types.get(gt, 0) + 1
print('down_expert types:', types)

types2 = {}
for t in m.tensors():
    n = t['name']
    if 'ffn_gate_exps' in n or 'ffn_up_exps' in n:
        gt = t['ggml_type']
        types2[gt] = types2.get(gt, 0) + 1
print('gate/up expert types:', types2)
