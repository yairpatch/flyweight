import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from colibri_next.v2 import V2Model
model = V2Model(Path(r'C:\Users\thegr\Downloads\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf'))
for t in model.tensors():
    name = t['name']
    if 'ffn_down_exps' in name:
        print(f"{name}: type={t['ggml_type']}")
