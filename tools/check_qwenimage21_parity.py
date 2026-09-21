"""Compare the native Qwen-Image-2.1 tower against a diffusers reference dump.

    python tools/qwenimage21_reference.py --out /tmp/qi_ref512.npz --size 512 \
        --diffusers build/diffusers_main
    python tools/check_qwenimage21_parity.py --ref /tmp/qi_ref512.npz --model <snapshot dir>

Stages, each against the dump's tensors: the text encoder's conditioning rows,
one transformer forward at the first step (reference latents and conditioning
in), the VAE decode of the reference's final latents, and the whole pipeline
from the reference's initial noise. Reports max and RMS error relative to the
reference's own scale; the transformer runs Q8 weights against a bf16
reference, so agreement is close, not bit-exact.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from flyweight.v2 import V2Diffusion, V2Model


def compare(name: str, got: np.ndarray, want: np.ndarray) -> float:
    got = np.asarray(got, dtype=np.float32).reshape(-1)
    want = np.asarray(want, dtype=np.float32).reshape(-1)
    assert got.shape == want.shape, (name, got.shape, want.shape)
    diff = got - want
    scale = float(np.sqrt(np.mean(want * want))) or 1.0
    rms = float(np.sqrt(np.mean(diff * diff))) / scale
    worst = float(np.max(np.abs(diff))) / scale
    cos = float(np.dot(got, want) / (np.linalg.norm(got) * np.linalg.norm(want) + 1e-30))
    print(f"  {name:28s} rms {rms:.4%}  max {worst:.3%}  cos {cos:.6f}", flush=True)
    return rms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--model", required=True, help="Qwen-Image-2.1 snapshot directory")
    parser.add_argument("--stages", default="text,encode,step,decode,generate")
    parser.add_argument("--max-size", type=int, default=0)
    parser.add_argument("--weights", default="auto", choices=("device", "host", "auto"))
    parser.add_argument("--precision", default="balanced", choices=("fast", "balanced", "exact"))
    args = parser.parse_args()
    ref = np.load(args.ref)
    size = ref["image"].shape[0]
    stages = set(args.stages.split(","))

    root = args.model.rstrip("/") + "/"
    started = time.time()
    encoder = V2Model(root + "text_encoder")
    transformer = V2Model(root + "transformer")
    vae = V2Model(root + "vae")
    tower = V2Diffusion(encoder, transformer, vae, max_width=args.max_size or size,
                        max_height=args.max_size or size, max_prompt_tokens=512,
                        weights=args.weights, precision=args.precision)
    latent = size // tower.latent_stride
    channels = tower.output_channels
    print(f"tower ready in {time.time() - started:.1f}s", flush=True)

    ids = [int(t) for t in ref["input_ids"]]
    drop = int(ref["drop_idx"])
    caption = ref["prompt_embeds"]
    # The editing dump carries the condition images and where their tokens sit.
    conditions = []
    index = 0
    while f"condition{index}_rgba" in ref.files:
        rgba = ref[f"condition{index}_rgba"]
        conditions.append((rgba, rgba.shape[1], rgba.shape[0]))
        index += 1
    pad_mask = ref["image_pad_mask"] if "image_pad_mask" in ref.files else None
    offsets = []
    if conditions:
        full = np.concatenate([np.zeros(drop, dtype=bool), pad_mask.astype(bool)])
        starts = np.flatnonzero(full & ~np.concatenate([[False], full[:-1]]))
        offsets = [int(s) for s in starts]
        assert len(offsets) == len(conditions), (offsets, len(conditions))
    images = [(rgba.tobytes(), w, h, offset) for (rgba, w, h), offset in zip(conditions, offsets)]

    if "text" in stages:
        started = time.time()
        if images:
            rows = np.array(tower.encode_prompt(ids, images), dtype=np.float32).reshape(len(ids), -1)
        else:
            rows = np.array(tower.encode_text(ids), dtype=np.float32).reshape(len(ids), -1)
        got = rows[drop:]
        print(f"text encoder {time.time() - started:.2f}s")
        compare("prompt_embeds", got, caption)
        # A row-by-row view: the later rows accumulate the most error.
        for row in (0, len(got) // 2, len(got) - 1):
            compare(f"  row {row}", got[row], caption[row])
        if pad_mask is not None and pad_mask.any():
            compare("  image rows", got[pad_mask.astype(bool)], caption[pad_mask.astype(bool)])
            compare("  text rows", got[~pad_mask.astype(bool)], caption[~pad_mask.astype(bool)])
    else:
        got = caption

    if "encode" in stages and images:
        for index, (rgba, w, h) in enumerate(conditions):
            started = time.time()
            latents = np.array(tower.encode_image(rgba.tobytes(), w, h), dtype=np.float32)
            print(f"vae encode {w}x{h} {time.time() - started:.2f}s")
            want = ref[f"condition{index}_latents"]
            compare(f"condition{index}_latents", latents.reshape(want.shape), want)

    if "step" in stages and images:
        print("  (step stage skipped: the text-only transformer_step has no condition images)")
    if "step" in stages and not images:
        sigma = float(ref["sigmas"][0])
        started = time.time()
        out = tower.transformer_step(ref["latents_0"].reshape(-1), latent, latent,
                                     caption.reshape(-1), caption.shape[0], sigma)
        print(f"transformer step {time.time() - started:.2f}s")
        shape = (tower.latent_channels, latent, latent)
        compare("step0_out (ref caption)", np.array(out).reshape(shape), ref["step0_out"])
        if "text" in stages:
            out = tower.transformer_step(ref["latents_0"].reshape(-1), latent, latent,
                                         got.reshape(-1), got.shape[0], sigma)
            compare("step0_out (own caption)", np.array(out).reshape(shape), ref["step0_out"])

    if "decode" in stages:
        steps = sum(1 for k in ref.files if k.startswith("step") and k.endswith("_latents"))
        final = ref[f"step{steps - 1}_latents"]
        started = time.time()
        pixels = np.frombuffer(tower.decode_latents(final.reshape(-1), latent, latent), dtype=np.uint8)
        print(f"vae decode {time.time() - started:.2f}s")
        image = pixels.reshape(size, size, channels)
        compare("decoded image", image.astype(np.float32), ref["image"].astype(np.float32))
        from PIL import Image
        Image.fromarray(image).save("/tmp/qwenimage21_native_decode.png")

    if "generate" in stages:
        steps = sum(1 for k in ref.files if k.startswith("step") and k.endswith("_latents"))
        started = time.time()
        if images:
            pixels = np.frombuffer(
                tower.edit(ids, images, size, size, steps=steps, caption_drop=drop,
                           initial_latents=ref["latents_0"].reshape(-1)),
                dtype=np.uint8)
        else:
            pixels = np.frombuffer(
                tower.generate(ids, size, size, steps=steps, caption_drop=drop,
                               initial_latents=ref["latents_0"].reshape(-1)),
                dtype=np.uint8)
        print(f"generate {time.time() - started:.2f}s")
        image = pixels.reshape(size, size, channels)
        compare("generated image", image.astype(np.float32), ref["image"].astype(np.float32))
        from PIL import Image
        Image.fromarray(image).save("/tmp/qwenimage21_native_generate.png")
    tower.close()


if __name__ == "__main__":
    main()
