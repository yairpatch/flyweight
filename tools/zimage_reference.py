"""Dump a diffusers Z-Image-Turbo run as reference tensors for the native port.

Usage:
    python tools/zimage_reference.py --out /tmp/zimage_ref.npz \
        --prompt "a red bicycle leaning on a brick wall" --seed 7 --size 512

Saves, for one prompt/seed/size:
    input_ids            tokenized chat-templated prompt (no padding)
    prompt_embeds        Qwen3 hidden_states[-2] rows for the real tokens  [T][2560]
    sigmas               the scheduler's sigma ladder (includes trailing 0)
    latents_0            initial Gaussian latents                          [16][H/8][W/8]
    step{i}_out          transformer output (velocity, before negation)    [16][H/8][W/8]
    step{i}_latents      latents after the Euler step
    image                final decoded image, uint8 HWC
    tap_*                intermediate activations of the first transformer call
                         (t_emb, x after embed, after each refiner / layer, final)

Every tensor is stored as float32. The transformer runs in bf16 like a normal
diffusers session, so the native port is compared against bf16 numerics.
"""
from __future__ import annotations

import argparse
import contextlib

import sys

import numpy as np
import torch

# This box's torchao build does not match its torch nightly and diffusers'
# guard around it is itself broken; hiding torchao keeps diffusers importable.
sys.modules.setdefault("torchao", None)  # type: ignore[assignment]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--prompt", default="a red bicycle leaning on a brick wall, afternoon light")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=512, help="square output size in pixels")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--text-only", action="store_true", help="only dump the prompt embeddings")
    parser.add_argument("--no-taps", action="store_true", help="skip per-layer taps of the first step")
    args = parser.parse_args()

    from diffusers import ZImagePipeline

    taps: dict[str, np.ndarray] = {}
    out: dict[str, np.ndarray] = {}

    # Encode the prompt with the bare Qwen3 encoder first: the whole 8 GB
    # bf16 model fits this card on its own, while the pipeline's offload hooks
    # do not keep it under the limit next to the transformer.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    templated = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    ids = tok(templated, return_tensors="pt").input_ids
    out["input_ids"] = ids[0].numpy().astype(np.int64)
    out["templated_prompt"] = np.array(templated)
    encoder = AutoModelForCausalLM.from_pretrained(
        args.model, subfolder="text_encoder", torch_dtype=torch.bfloat16).to("cuda")
    with torch.no_grad():
        hidden = encoder(input_ids=ids.to("cuda"), output_hidden_states=True).hidden_states
    embeds = hidden[-2][0]
    out["prompt_embeds"] = embeds.float().cpu().numpy()
    del encoder, hidden
    torch.cuda.empty_cache()
    if args.text_only:
        np.savez(args.out, **out)
        print("wrote", args.out, {k: getattr(v, "shape", None) for k, v in out.items()})
        return

    pipe = ZImagePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, text_encoder=None, tokenizer=None)
    # The bf16 transformer alone is ~12.3 GB, more than this card holds, so
    # stream layers through the GPU instead of parking whole models there.
    pipe.enable_sequential_cpu_offload()

    transformer = pipe.transformer
    hooks = []
    first_call = {"done": False}

    def tap(name):
        def hook(_module, _inputs, output):
            if first_call["done"]:
                return
            value = output[0] if isinstance(output, tuple) else output
            taps[f"tap_{name}"] = value.detach().float().cpu().numpy()
        return hook

    if not args.no_taps:
        hooks.append(transformer.t_embedder.register_forward_hook(tap("t_emb")))
        hooks.append(transformer.cap_embedder.register_forward_hook(tap("cap_embed")))
        for key, module in transformer.all_x_embedder.items():
            hooks.append(module.register_forward_hook(tap(f"x_embed_{key}")))
        for i, block in enumerate(transformer.noise_refiner):
            hooks.append(block.register_forward_hook(tap(f"noise_refiner_{i}")))
        for i, block in enumerate(transformer.context_refiner):
            hooks.append(block.register_forward_hook(tap(f"context_refiner_{i}")))
        for i, block in enumerate(transformer.layers):
            hooks.append(block.register_forward_hook(tap(f"layer_{i}")))
        for key, module in transformer.all_final_layer.items():
            hooks.append(module.register_forward_hook(tap(f"final_{key}")))

    step_outputs = []
    step_latents = []

    original_forward = transformer.forward

    def recording_forward(*fargs, **fkwargs):
        result = original_forward(*fargs, **fkwargs)
        sample = result[0] if isinstance(result, tuple) else result.sample
        step_outputs.append(sample[0].squeeze(1).float().cpu().numpy())
        first_call["done"] = True
        return result

    transformer.forward = recording_forward

    def on_step_end(_pipe, _i, _t, kwargs):
        step_latents.append(kwargs["latents"][0].float().cpu().numpy())
        return {}

    generator = torch.Generator("cuda").manual_seed(args.seed)
    latents_0 = torch.randn((1, 16, args.size // 8, args.size // 8), generator=generator,
                            device="cuda", dtype=torch.float32)
    out["latents_0"] = latents_0[0].cpu().numpy()

    with torch.no_grad():
        result = pipe(
            prompt=None, prompt_embeds=[embeds], height=args.size, width=args.size,
            num_inference_steps=args.steps, guidance_scale=0.0,
            latents=latents_0, callback_on_step_end=on_step_end,
            output_type="np",
        )
    out["sigmas"] = pipe.scheduler.sigmas.float().cpu().numpy()
    for i, (o, stepped) in enumerate(zip(step_outputs, step_latents)):
        out[f"step{i}_out"] = o
        out[f"step{i}_latents"] = stepped
    out["image"] = (result.images[0] * 255).round().clip(0, 255).astype(np.uint8)
    out.update(taps)
    for h in hooks:
        h.remove()

    np.savez(args.out, **out)
    from PIL import Image
    Image.fromarray(out["image"]).save(args.out.replace(".npz", ".png"))
    print("wrote", args.out)
    for k, v in out.items():
        print(f"  {k:28s} {getattr(v, 'shape', v)}")


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
