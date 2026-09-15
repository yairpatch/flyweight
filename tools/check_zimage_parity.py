"""Compare the native Z-Image tower against a diffusers reference dump.

    python tools/zimage_reference.py --out /tmp/zimage_ref512.npz --size 512
    python tools/check_zimage_parity.py --ref /tmp/zimage_ref512.npz --model <snapshot dir>

Stages, each against the dump's tensors: the text encoder's hidden_states[-2],
one transformer forward at the first step (reference latents and caption in),
the VAE decode of the reference's final latents, and the whole pipeline from
the reference's initial noise. Reports max and RMS error relative to the
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
    parser.add_argument("--model", required=True, help="Z-Image-Turbo snapshot directory")
    parser.add_argument("--stages", default="text,step,decode,generate")
    parser.add_argument("--max-size", type=int, default=0)
    parser.add_argument("--weights", default="device", choices=("device", "host", "auto"))
    parser.add_argument("--precision", default="fast", choices=("fast", "balanced", "exact"))
    args = parser.parse_args()
    ref = np.load(args.ref)
    size = ref["image"].shape[0]
    latent = size // 8
    stages = set(args.stages.split(","))

    root = args.model.rstrip("/") + "/"
    started = time.time()
    encoder = V2Model(root + "text_encoder")
    transformer = V2Model(root + "transformer")
    vae = V2Model(root + "vae")
    tower = V2Diffusion(encoder, transformer, vae, max_width=args.max_size or size,
                        max_height=args.max_size or size, max_prompt_tokens=512, weights=args.weights,
                        precision=args.precision)
    print(f"tower ready in {time.time() - started:.1f}s", flush=True)

    ids = [int(t) for t in ref["input_ids"]]
    own_ids = encoder.tokenize(str(ref["templated_prompt"]))
    if list(own_ids) != ids:
        print("  tokenizer mismatch:", own_ids, "vs", ids)
    caption = ref["prompt_embeds"]

    if "text" in stages:
        started = time.time()
        got = np.array(tower.encode_text(ids), dtype=np.float32).reshape(len(ids), -1)
        print(f"text encoder {time.time() - started:.2f}s")
        compare("prompt_embeds", got, caption)
        # A row-by-row view: the later rows accumulate the most error.
        for row in (0, len(ids) // 2, len(ids) - 1):
            compare(f"  row {row}", got[row], caption[row])

    if "step" in stages:
        sigma = float(ref["sigmas"][0])
        started = time.time()
        out = tower.transformer_step(ref["latents_0"].reshape(-1), latent, latent,
                                     caption.reshape(-1), caption.shape[0], 1.0 - sigma)
        print(f"transformer step {time.time() - started:.2f}s")
        compare("step0_out (ref caption)", np.array(out).reshape(16, latent, latent), ref["step0_out"])
        if "text" in stages:
            out = tower.transformer_step(ref["latents_0"].reshape(-1), latent, latent,
                                         got.reshape(-1), got.shape[0], 1.0 - sigma)
            compare("step0_out (own caption)", np.array(out).reshape(16, latent, latent), ref["step0_out"])

    if "decode" in stages:
        steps = sum(1 for k in ref.files if k.startswith("step") and k.endswith("_latents"))
        final = ref[f"step{steps - 1}_latents"]
        started = time.time()
        rgb = np.frombuffer(tower.decode_latents(final.reshape(-1), latent, latent), dtype=np.uint8)
        print(f"vae decode {time.time() - started:.2f}s")
        image = rgb.reshape(size, size, 3)
        compare("decoded image", image.astype(np.float32), ref["image"].astype(np.float32))
        from PIL import Image
        Image.fromarray(image).save("/tmp/zimage_native_decode.png")

    if "generate" in stages:
        started = time.time()
        rgb = np.frombuffer(tower.generate(ids, size, size, steps=8, initial_latents=ref["latents_0"].reshape(-1)),
                            dtype=np.uint8)
        print(f"generate {time.time() - started:.2f}s")
        image = rgb.reshape(size, size, 3)
        compare("generated image", image.astype(np.float32), ref["image"].astype(np.float32))
        from PIL import Image
        Image.fromarray(image).save("/tmp/zimage_native_generate.png")
    tower.close()


if __name__ == "__main__":
    main()
