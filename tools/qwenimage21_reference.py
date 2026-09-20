"""Dump a diffusers Qwen-Image-2.1 run as reference tensors for the native port.

Usage:
    python tools/qwenimage21_reference.py --out /tmp/qi_ref.npz \
        --prompt "a red bicycle leaning on a brick wall" --seed 7 --size 512

Saves, for one prompt/seed/size:
    input_ids            the raw prompt template tokenized, no padding
    drop_idx             leading system-role tokens the pipeline drops
    prompt_embeds        conditioning rows the transformer sees       [T][4096]
    sigmas               the scheduler's sigma ladder (includes trailing 0)
    latents_0            initial Gaussian latents                     [64][H/16][W/16]
    step{i}_out          transformer output for that step             [64][H/16][W/16]
    step{i}_latents      latents after the Euler step
    decoded              VAE decode of the final latents, float [4][H][W]
    image                final image, uint8 HWC (RGBA)

Every tensor is stored as float32. The transformer runs in bf16 like a normal
diffusers session, so the native port is compared against bf16 numerics.

The pipeline is driven with `use_kv_cache=False`, which is the mode the native
port implements: it recomputes the whole joint sequence every step instead of
caching the prefix. The reference itself notes the two are equally valid but
not bit-identical, so the comparison has to pick one.

diffusers must be new enough to carry QwenImage21Pipeline. `--diffusers DIR`
prepends a checkout or a `pip install --target` of diffusers main, so the
system install does not have to be upgraded:

    python -m pip install --target build/diffusers_main --no-deps \
        "git+https://github.com/huggingface/diffusers@main"
"""
from __future__ import annotations

import argparse
import contextlib
import math
import sys

import numpy as np
import torch

# This box's torchao build does not match its torch nightly and diffusers'
# guard around it is itself broken; hiding torchao keeps diffusers importable.
sys.modules.setdefault("torchao", None)  # type: ignore[assignment]

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
TEMPLATE = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--prompt", default="a red bicycle leaning on a brick wall, afternoon light")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=512, help="square output size in pixels")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--diffusers", help="a diffusers checkout to put ahead of the installed one")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--encoder-device", default="cpu", choices=("cuda", "cpu"),
                        help="the bf16 Qwen3-VL is 16 GB; a prompt of a few dozen tokens "
                             "runs in seconds on the CPU, so that is the default")
    parser.add_argument("--text-only", action="store_true", help="only dump the conditioning")
    parser.add_argument("--image", action="append", default=[],
                        help="a condition image (repeatable): the editing path, which also dumps "
                             "the resized RGBA pixels, the VAE latents and the joint layout")
    args = parser.parse_args()

    if args.diffusers:
        sys.path.insert(0, args.diffusers)

    from diffusers import QwenImage21Pipeline

    out: dict[str, np.ndarray] = {}
    device = args.device

    # Encode the prompt with the bare encoder first, on its own device: the
    # pipeline's offload hooks do not keep a 16 GB encoder under a 12 GB
    # card's limit next to the transformer.
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model, subfolder="processor")
    tokenizer = processor.tokenizer
    system_only = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    drop_idx = len(processor.apply_chat_template(system_only, tokenize=True, return_dict=False)[0])
    out["drop_idx"] = np.array(drop_idx, dtype=np.int64)

    # Condition images, prepared the way the pipeline does: resized to the
    # output area at their own aspect (multiples of 32), RGBA for the VAE,
    # alpha over white for the vision encoder.
    from PIL import Image as PILImage
    condition_rgba, condition_rgb = [], []
    for index, path in enumerate(args.image):
        picture = PILImage.open(path).convert("RGBA")
        ratio = picture.width / picture.height
        width = round(math.sqrt(args.size * args.size * ratio) / 32) * 32
        height = round(math.sqrt(args.size * args.size / ratio) / 32) * 32
        picture = picture.resize((width, height), PILImage.Resampling.LANCZOS)
        condition_rgba.append(picture)
        out[f"condition{index}_rgba"] = np.array(picture, dtype=np.uint8)
        white = PILImage.new("RGB", picture.size, (255, 255, 255))
        white.paste(picture, mask=picture.getchannel("A"))
        condition_rgb.append(white)

    if args.image:
        slots = "<image1><|vision_start|><|image_pad|><|vision_end|>"
        for index in range(2, len(args.image) + 1):
            slots += f" <image{index}><|vision_start|><|image_pad|><|vision_end|>"
        template = TEMPLATE.replace("<|im_start|>user\n{}", "<|im_start|>user\n" + slots + "{}")
        model_inputs = processor(text=[template.format(args.prompt)], images=condition_rgb,
                                 padding=True, return_tensors="pt")
        out["image_grid_thw"] = model_inputs.image_grid_thw.numpy().astype(np.int64)
        out["pixel_values_shape"] = np.array(model_inputs.pixel_values.shape, dtype=np.int64)
    else:
        model_inputs = tokenizer(TEMPLATE.format(args.prompt), return_tensors="pt")
    ids = model_inputs.input_ids
    out["input_ids"] = ids[0].numpy().astype(np.int64)
    image_token_id = tokenizer.encode("<|image_pad|>")[0]
    out["image_pad_mask"] = (ids[0] == image_token_id).numpy()[drop_idx:]

    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, subfolder="text_encoder", dtype=torch.bfloat16).to(args.encoder_device)
    # hidden_states[-1] has to be the last decoder layer's output, before the
    # final RMSNorm: that is what the transformer was trained on, and it is
    # what the pipeline arranges with this same hook.
    text_model = getattr(encoder.model, "language_model", encoder.model)
    handle = text_model.norm.register_forward_hook(lambda module, inputs, output: inputs[0])
    forward_kwargs = {"input_ids": ids.to(args.encoder_device), "output_hidden_states": True}
    if args.image:
        forward_kwargs["pixel_values"] = model_inputs.pixel_values.to(args.encoder_device, torch.bfloat16)
        forward_kwargs["image_grid_thw"] = model_inputs.image_grid_thw.to(args.encoder_device)
        if hasattr(model_inputs, "mm_token_type_ids"):
            forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids.to(args.encoder_device)
    try:
        with torch.no_grad():
            hidden = encoder(**forward_kwargs).hidden_states
    finally:
        handle.remove()
    embeds = hidden[-1][0][drop_idx:]
    out["prompt_embeds"] = embeds.float().cpu().numpy()
    del encoder, hidden
    if device == "cuda":
        torch.cuda.empty_cache()
    if args.text_only:
        np.savez(args.out, **out)
        print("wrote", args.out, {k: getattr(v, "shape", None) for k, v in out.items()})
        return

    pipe = QwenImage21Pipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, text_encoder=None)
    if device == "cuda":
        # The bf16 transformer alone is 13.5 GB, more than this card holds, so
        # stream layers through the GPU instead of parking whole models there.
        pipe.enable_sequential_cpu_offload()

    step_outputs: list[np.ndarray] = []
    step_latents: list[np.ndarray] = []
    transformer = pipe.transformer
    original_forward = transformer.forward
    latent_h, latent_w = args.size // 16, args.size // 16

    def recording_forward(*forward_args, **forward_kwargs):
        result = original_forward(*forward_args, **forward_kwargs)
        sample = result[0] if isinstance(result, tuple) else result.sample
        # The joint sequence ends with the target image's rows, which are what
        # the sampler steps; the text rows in front of them are not output.
        rows = sample[0, -latent_h * latent_w:]
        step_outputs.append(rows.transpose(0, 1).float().cpu().numpy().reshape(-1, latent_h, latent_w))
        return result

    transformer.forward = recording_forward

    def on_step_end(_pipe, _i, _t, kwargs):
        latents = kwargs["latents"][0]   # [tokens][channels]
        step_latents.append(
            latents.transpose(0, 1).float().cpu().numpy().reshape(-1, latent_h, latent_w))
        return {}

    generator = torch.Generator(device).manual_seed(args.seed)
    latents_0 = torch.randn((1, 1, 64, latent_h, latent_w), generator=generator,
                            device=device, dtype=torch.float32)
    out["latents_0"] = latents_0[0, 0].cpu().numpy()
    packed = latents_0.view(1, 64, latent_h * latent_w).transpose(1, 2).to(torch.bfloat16)

    if args.image:
        # The pipeline's own VAE encode of each condition image, normalized,
        # is what the transformer sees in front of the target latents.
        for index, picture in enumerate(condition_rgba):
            tensor = pipe.image_processor.preprocess(picture, width=picture.width, height=picture.height)
            with torch.no_grad():
                latents = pipe._encode_vae_image(tensor.unsqueeze(2).to(device, torch.bfloat16), None)
            out[f"condition{index}_latents"] = latents[0, :, 0].float().cpu().numpy()
    # The pipeline re-encodes the prompt itself when condition images are
    # passed (it needs the image_pad_mask its encoder produces), and the
    # encoder does not fit beside the transformer; hand it ours instead.
    conditioning = (embeds[None].to(torch.bfloat16).to(device), None,
                    torch.from_numpy(out["image_pad_mask"])[None].to(device))
    pipe.encode_prompt = lambda **kwargs: conditioning
    with torch.no_grad():
        result = pipe(
            prompt=args.prompt,
            image=condition_rgba or None,
            height=args.size, width=args.size,
            # Condition images are resized to this area at their own aspect;
            # it has to match the resize the vision copy above got.
            output_resolution=args.size,
            num_inference_steps=args.steps,
            latents=packed,
            use_kv_cache=False,
            callback_on_step_end=on_step_end,
            output_type="np",
        )
    out["sigmas"] = pipe.scheduler.sigmas.float().cpu().numpy()
    for i, (output, stepped) in enumerate(zip(step_outputs, step_latents)):
        out[f"step{i}_out"] = output
        out[f"step{i}_latents"] = stepped
    out["image"] = (result.images[0] * 255).round().clip(0, 255).astype(np.uint8)

    np.savez(args.out, **out)
    from PIL import Image
    Image.fromarray(out["image"]).save(args.out.replace(".npz", ".png"))
    print("wrote", args.out)
    for key, value in out.items():
        print(f"  {key:28s} {getattr(value, 'shape', value)}")


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
