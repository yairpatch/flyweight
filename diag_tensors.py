import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from colibri_next.v2 import V2Model

model_path = Path(r'C:\Users\thegr\Downloads\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf')
model = V2Model(model_path)

# Print blk.0 tensors and output tensors
for t in model.tensors():
    name = t['name']
    if name.startswith('blk.0.') or name in ('output.weight', 'token_embd.weight', 'output_norm.weight'):
        print(f"{name}: type={t['ggml_type']} shape={t['shape']}")
