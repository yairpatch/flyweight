import json, sys, subprocess
result = subprocess.run(
    [sys.executable, "-m", "colibri_next.cli", "inspect-gguf-v2", sys.argv[1]],
    capture_output=True, text=True
)
d = json.loads(result.stdout)
types = {}
for t in d["tensors"]:
    if "ffn" in t["name"] or "attn_q" in t["name"] or "attn_k" in t["name"] or "attn_v" in t["name"] or "attn_output" in t["name"]:
        tt = t["ggml_type"]
        if tt not in types:
            types[tt] = []
        types[tt].append(t["name"])
for tt in sorted(types):
    print(f"Type {tt}: {len(types[tt])} tensors")
    for name in types[tt][:3]:
        print(f"  {name}")
    if len(types[tt]) > 3:
        print(f"  ... and {len(types[tt])-3} more")
