#!/usr/bin/env python3
"""Test real-model generation with Sylvester-Hadamard Q8 KV cache."""
import os
import sys
import time

# Ensure local src is in path
sys.path.insert(0, os.path.abspath("src"))

from flyweight.v2 import V2Model

MODEL_PATH = "/home/yair/Downloads/Qwen3.8-27B-UD-IQ2_XXS.gguf"
PROMPT = "The Hadamard transform is important in quantization because"
TOKENS = 30

def run_test(name: str, cache_type_k: str, cache_type_v: str, env_hadamard: str | None):
    env_backup = os.environ.get("FLYWEIGHT_KV_HADAMARD")
    if env_hadamard is not None:
        os.environ["FLYWEIGHT_KV_HADAMARD"] = env_hadamard
    elif "FLYWEIGHT_KV_HADAMARD" in os.environ:
        del os.environ["FLYWEIGHT_KV_HADAMARD"]

    print(f"\n--- Running: {name} (K={cache_type_k}, V={cache_type_v}, HADAMARD={env_hadamard}) ---")
    model = V2Model(MODEL_PATH)
    try:
        with model.native_runtime(context_limit=1024, cache_type_k=cache_type_k, cache_type_v=cache_type_v) as rt:
            rt.prepare()
            tokens = []
            t0 = time.perf_counter()
            prompt_tokens = list(model.tokenize(PROMPT))
            rt.generate(prompt_tokens, max_tokens=TOKENS, callback=lambda t: tokens.append(t))
            elapsed = time.perf_counter() - t0
            text = model.decode_tokens(tokens)
            tok_per_sec = len(tokens) / elapsed if elapsed > 0 else 0
            print(f"Generated {len(tokens)} tokens in {elapsed:.3f}s ({tok_per_sec:.1f} tok/s)")
            print(f"Output: {text!r}")
            return text
    finally:
        model.close()
        if env_backup is not None:
            os.environ["FLYWEIGHT_KV_HADAMARD"] = env_backup
        elif "FLYWEIGHT_KV_HADAMARD" in os.environ:
            del os.environ["FLYWEIGHT_KV_HADAMARD"]

def main():
    print("=== Testing Real Model End-to-End Inference ===")
    out_f16 = run_test("Baseline F16", "f16", "f16", None)
    out_q8_standard = run_test("Standard Q8 (No Hadamard)", "q8_0", "f16", "0")
    out_q8_hadamard = run_test("Sylvester-Hadamard Q8 (K=q8, V=f16)", "q8_0", "f16", "1")
    out_q8_hadamard_kv = run_test("Sylvester-Hadamard Q8 (K=q8, V=q8)", "q8_0", "q8_0", "1")

    print("\n=== Summary Comparison ===")
    print(f"F16 Reference:   {out_f16[:70]}...")
    print(f"Q8 Standard:     {out_q8_standard[:70]}...")
    print(f"Q8 Hadamard:     {out_q8_hadamard[:70]}...")
    print(f"Q8+Q8 Hadamard:  {out_q8_hadamard_kv[:70]}...")

if __name__ == "__main__":
    main()
