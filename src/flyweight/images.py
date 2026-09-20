"""Image generation served beside the chat model, on the native tower.

``ImageGenerator`` owns the three component models of a diffusers snapshot
(``text_encoder``, ``transformer``, ``vae``) and the tower built over them,
turns an OpenAI-style ``/v1/images/generations`` request into a run, and
returns the picture as base64 PNG. One image renders at a time: the tower
has one workspace and the GPU is shared with the chat runtime.

Two models are recognised, and which one a snapshot holds decides the prompt
template, the size grid, the step count and whether the picture has an alpha
channel -- so everything here reads those off the tower rather than assuming.

Pillow encodes the PNG and is an optional dependency (``pip install
flyweight-llm[vision]``); without it the endpoint reports why.
"""
from __future__ import annotations

import base64
import io
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .server import APIError
from .v2 import V2Diffusion, V2Error, V2Model
from .vision import ImageError, expand_image_pads, fetch_image_url, image_token_offsets

# The Qwen3 chat template with add_generation_prompt=True and thinking left
# enabled, which is how Z-Image's reference pipeline conditions the DiT.
_PROMPT_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

# Qwen-Image-2.1 conditions on the same template behind a fixed system turn.
# Its pipeline builds this as a raw string rather than through
# apply_chat_template -- the two tokenize differently and the checkpoint
# expects this one -- and then drops the system turn's rows from the
# conditioning, which `caption_drop` below does.
_QWENIMAGE_SYSTEM = (
    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
)
# One slot per condition image in the editing template, in front of the
# prompt: "<image1>" is literal text the checkpoint expects, the pad expands
# to the picture's vision tokens.
_QWENIMAGE_SLOT = "<image{index}><|vision_start|><|image_pad|><|vision_end|>"
# Condition images are resized to this many pixels at their own aspect, and a
# request without a size renders at the last image's shape. The pipeline's
# `output_resolution`.
DEFAULT_EDIT_AREA_SIDE = 1024
MAX_CONDITION_IMAGES = 10

MAX_STEPS = 50
MAX_PROMPT_TOKENS = 512
# Reference images add their vision tokens (one per 32x32 pixels: 1024 for a
# 1024x1024 reference) on top of the text budget; four such, or sixteen at
# 512x512. The transformer then sees four latent rows per token.
MAX_IMAGE_TOKENS = 4096


def _int_option(payload: Mapping[str, Any], name: str, default: int, low: int, high: int) -> int:
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise APIError(400, f"{name} must be an integer", parameter=name)
    if value < low or value > high:
        raise APIError(400, f"{name} must be between {low} and {high}", parameter=name)
    return value


def parse_size(
    value: Any, default: tuple[int, int], limit: tuple[int, int], multiple: int = 16
) -> tuple[int, int]:
    """``"WxH"`` (or ``"auto"``) into pixels: on the model's grid, within the tower's limits."""
    if value is None or value == "auto":
        return default
    if not isinstance(value, str) or "x" not in value:
        raise APIError(400, "size must look like 1024x1024", parameter="size")
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except ValueError as error:
        raise APIError(400, "size must look like 1024x1024", parameter="size") from error
    if width <= 0 or height <= 0 or width % multiple or height % multiple:
        raise APIError(
            400,
            f"size sides must be positive multiples of {multiple}",
            parameter="size",
        )
    if width > limit[0] or height > limit[1]:
        raise APIError(
            400,
            f"size exceeds this server's limit of {limit[0]}x{limit[1]} "
            "(start with a larger --image-max-size)",
            parameter="size",
        )
    return width, height


def fit_to_area(width: int, height: int, side: int, multiple: int = 32) -> tuple[int, int]:
    """The pipeline's `calculate_dimensions`: the aspect of (width, height) at
    `side * side` pixels, both sides rounded to `multiple`."""
    ratio = width / height
    area = float(side * side)
    fitted_width = round((area * ratio) ** 0.5 / multiple) * multiple
    fitted_height = round((area / ratio) ** 0.5 / multiple) * multiple
    return max(multiple, fitted_width), max(multiple, fitted_height)


def snapshot_model_name(root: Path) -> str:
    """``org/name`` for a Hugging Face cache snapshot, else the directory name.

    A cache snapshot is ``models--org--name/snapshots/<hash>``; reporting the
    hash as the model name told nobody what was loaded.
    """
    parts = root.resolve().parts
    for index in range(len(parts) - 2, 0, -1):
        if parts[index] == "snapshots" and parts[index - 1].startswith("models--"):
            return parts[index - 1][len("models--"):].replace("--", "/")
    return root.name


class _Cancelled(Exception):
    """The streaming client went away; the render stops at the next step."""


class ImageGenerator:
    """The image model behind ``/v1/images/generations``."""

    def __init__(
        self,
        snapshot: Path | str,
        *,
        device: int = 0,
        max_width: int = 0,
        max_height: int = 0,
        weights: str = "auto",
        precision: str = "balanced",
        reserve: bool = False,
        model_name: str | None = None,
    ) -> None:
        root = Path(snapshot)
        for part in ("text_encoder", "transformer", "vae"):
            if not (root / part / "config.json").is_file():
                raise FileNotFoundError(
                    f"{root} is not a diffusers image snapshot: missing {part}/config.json"
                )
        self.path = root
        self.model_name = model_name or snapshot_model_name(root)
        self.encoder = V2Model(root / "text_encoder")
        self.transformer = V2Model(root / "transformer")
        self.vae = V2Model(root / "vae")
        edits = self.transformer.config["architecture"] == "qwenimage21-dit"
        try:
            self.tower = V2Diffusion(
                self.encoder, self.transformer, self.vae,
                device=device, max_width=int(max_width), max_height=int(max_height),
                max_prompt_tokens=MAX_PROMPT_TOKENS + (MAX_IMAGE_TOKENS if edits else 0),
                weights=weights, precision=precision, reserve=reserve,
            )
        except BaseException:
            self.close()
            raise
        # Zero asked for the model's own maximum; the tower says what that is.
        self.max_width = self.tower.max_width
        self.max_height = self.tower.max_height
        # Qwen-Image-2.1 is the model with a system turn in front of the prompt
        # and an alpha channel out of the decoder; Z-Image has neither.
        self.qwenimage = self.transformer.config["architecture"] == "qwenimage21-dit"
        self.caption_drop = (
            len(self.encoder.tokenize(_QWENIMAGE_SYSTEM)) if self.qwenimage else 0
        )
        self.default_steps = self.tower.default_steps
        self.size_multiple = self.tower.size_multiple
        self.pixel_mode = "RGBA" if self.tower.output_channels == 4 else "RGB"
        self._lock = threading.Lock()
        self.busy = False
        self.generated = 0

    def close(self) -> None:
        tower = getattr(self, "tower", None)
        if tower is not None:
            tower.close()
            self.tower = None  # type: ignore[assignment]
        for name in ("vae", "transformer", "encoder"):
            model = getattr(self, name, None)
            if model is not None:
                model.close()
                setattr(self, name, None)

    def describe(self) -> dict[str, Any]:
        info = self.tower.info
        return {
            "model": self.model_name,
            "path": str(self.path),
            "max_size": f"{self.max_width}x{self.max_height}",
            "default_steps": self.default_steps,
            "size_multiple": self.size_multiple,
            "alpha": self.pixel_mode == "RGBA",
            "edit": self.qwenimage,
            "weights": "host" if info["host_weights"] else "device",
            "precision": "exact" if info["exact"] else "balanced" if info["balanced"] else "fast",
            "device_mib": int(info["device_bytes"]) // (1024 * 1024),
            "host_mib": int(info["host_bytes"]) // (1024 * 1024),
            "busy": self.busy,
            "generated": self.generated,
        }

    def tokenize(self, prompt: str, image_tokens: Sequence[int] = ()) -> list[int]:
        """The prompt's tokens; with `image_tokens` (one count per condition
        image) the editing template, each pad expanded to its picture's count."""
        template = _PROMPT_TEMPLATE
        if self.qwenimage:
            slots = " ".join(_QWENIMAGE_SLOT.format(index=i + 1) for i in range(len(image_tokens)))
            template = _QWENIMAGE_SYSTEM + template.replace("{prompt}", slots + "{prompt}")
        tokens = list(self.encoder.tokenize(template.format(prompt=prompt)))
        if image_tokens:
            pad = self.encoder.tokenize("<|image_pad|>")
            if len(pad) != 1:
                raise APIError(500, "the tokenizer has no <|image_pad|> token", "server_error")
            try:
                tokens = expand_image_pads(tokens, pad[0], image_tokens)
            except ImageError as error:
                raise APIError(400, str(error), parameter="images") from None
        pictured = sum(image_tokens)
        if pictured > MAX_IMAGE_TOKENS:
            raise APIError(
                400,
                f"the reference images take {pictured} vision tokens; at most {MAX_IMAGE_TOKENS} "
                "(smaller or fewer references)",
                parameter="images",
            )
        if len(tokens) - pictured > MAX_PROMPT_TOKENS:
            raise APIError(
                400,
                f"prompt is {len(tokens) - pictured} tokens; the encoder reads at most {MAX_PROMPT_TOKENS}",
                parameter="prompt",
            )
        return tokens

    def condition_images(self, payload: Mapping[str, Any], side: int) -> list[tuple[bytes, int, int]]:
        """The request's `images` (data URLs or base64), decoded to RGBA and
        resized to `side * side` pixels at their own aspect, on the 32 grid."""
        values = payload.get("images")
        if values is None:
            return []
        if not self.qwenimage:
            raise APIError(400, "this image model cannot take reference images", parameter="images")
        if not isinstance(values, list) or not values:
            raise APIError(400, "images must be a non-empty list", parameter="images")
        if len(values) > MAX_CONDITION_IMAGES:
            raise APIError(400, f"at most {MAX_CONDITION_IMAGES} reference images", parameter="images")
        try:
            from PIL import Image
        except ImportError as error:
            raise APIError(500, "image input needs Pillow: pip install flyweight-llm[vision]", "server_error") from error
        out = []
        for value in values:
            if isinstance(value, Mapping):
                value = value.get("url") or value.get("b64_json") or value.get("image")
            if not isinstance(value, str) or not value:
                raise APIError(400, "each image must be a data URL or base64 string", parameter="images")
            if not value.startswith(("data:", "http://", "https://")):
                value = "data:image/*;base64," + value
            try:
                data = fetch_image_url(value, allow_remote=False).data
                with Image.open(io.BytesIO(data)) as opened:
                    picture = opened.convert("RGBA")
            except ImageError as error:
                raise APIError(400, str(error), parameter="images") from None
            except Exception:  # noqa: BLE001 - reported to the client
                raise APIError(400, "could not decode a reference image", parameter="images") from None
            width, height = fit_to_area(picture.width, picture.height, side, self.size_multiple)
            width = min(width, self.max_width)
            height = min(height, self.max_height)
            picture = picture.resize((width, height), Image.Resampling.LANCZOS)
            out.append((picture.tobytes(), width, height))
        return out

    def stream(self, payload: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
        """``generate`` as server-sent events: ``progress`` per denoising step,
        one ``image`` per picture, then ``done``. Closing the iterator (a client
        that went away) cancels the render at the next step."""
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        cancelled = threading.Event()
        outcome: dict[str, Any] = {}

        def on_progress(step: int, total: int) -> None:
            events.put({"type": "progress", "step": step, "steps": total})
            if cancelled.is_set():
                raise _Cancelled()

        def on_image(item: dict[str, Any], index: int, count: int) -> None:
            events.put({"type": "image", "index": index, "count": count, **item})

        def worker() -> None:
            try:
                outcome["result"] = self.generate(payload, progress=on_progress, on_image=on_image)
            except BaseException as error:  # noqa: BLE001 - handed to the stream
                outcome["error"] = error
            finally:
                events.put(None)

        thread = threading.Thread(target=worker, name="image-render", daemon=True)
        thread.start()
        try:
            while True:
                event = events.get()
                if event is None:
                    break
                yield event
            error = outcome.get("error")
            if isinstance(error, _Cancelled):
                return
            if error is not None:
                raise error
            result = outcome["result"]
            yield {"type": "done", **{k: v for k, v in result.items() if k != "data"}}
        finally:
            cancelled.set()

    def generate(
        self,
        payload: Mapping[str, Any],
        *,
        progress: Callable[[int, int], None] | None = None,
        on_image: Callable[[dict[str, Any], int, int], None] | None = None,
    ) -> dict[str, Any]:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise APIError(400, "prompt must be non-empty text", parameter="prompt")
        count = _int_option(payload, "n", 1, 1, 4)
        response_format = payload.get("response_format", "b64_json")
        if response_format != "b64_json":
            raise APIError(
                400, "only response_format b64_json is available", parameter="response_format"
            )
        # With reference images and no size, the picture takes the last
        # image's shape at the default area, as the pipeline does.
        side = min(DEFAULT_EDIT_AREA_SIDE, self.max_width, self.max_height)
        requested = payload.get("size")
        if isinstance(requested, str) and "x" in requested and requested != "auto":
            probe = parse_size(requested, (self.max_width, self.max_height),
                               (self.max_width, self.max_height), self.size_multiple)
            side = min(side, max(probe))
        conditions = self.condition_images(payload, side)
        if conditions and (requested is None or requested == "auto"):
            _, last_width, last_height = conditions[-1]
            default_size = (last_width, last_height)
        else:
            default_size = (self.max_width, self.max_height)
        width, height = parse_size(
            requested, default_size, (self.max_width, self.max_height), self.size_multiple,
        )
        steps = _int_option(payload, "steps", self.default_steps, 1, MAX_STEPS)
        seed_value = payload.get("seed")
        if seed_value is None:
            seed = int(time.time_ns()) & 0xFFFFFFFFFFFF
        elif isinstance(seed_value, bool) or not isinstance(seed_value, int) or seed_value < 0:
            raise APIError(400, "seed must be a non-negative integer", parameter="seed")
        else:
            seed = seed_value
        # Zero is the tower's "use the model's own schedule", which is how
        # Qwen-Image-2.1 samples: its shift follows the image's token count.
        shift = payload.get("shift", self.tower.default_shift)
        if isinstance(shift, bool) or not isinstance(shift, (int, float)) or shift < 0:
            raise APIError(400, "shift must be a non-negative number", parameter="shift")
        try:
            from PIL import Image
        except ImportError as error:
            raise APIError(
                500, "image output needs Pillow: pip install flyweight-llm[vision]", "server_error"
            ) from error

        token_side = self.tower.latent_stride * 2   # pixels per vision token
        image_tokens = [(w // token_side) * (h // token_side) for _, w, h in conditions]
        tokens = self.tokenize(prompt, image_tokens)
        images = []
        if conditions:
            pad = self.encoder.tokenize("<|image_pad|>")[0]
            offsets = image_token_offsets(tokens, pad, image_tokens)
            images = [(rgba, w, h, offset) for (rgba, w, h), offset in zip(conditions, offsets)]
        if not self._lock.acquire(timeout=0.0):
            raise APIError(
                429, "an image is already rendering; retry shortly", "rate_limit_error"
            )
        try:
            self.busy = True
            images = []
            for index in range(count):
                def on_progress(step: int, total: int, _index: int = index) -> bool:
                    if progress is not None:
                        try:
                            progress(_index * total + step, count * total)
                        except _Cancelled:
                            return True
                    return False

                started = time.monotonic()
                try:
                    if images:
                        rgb = self.tower.edit(
                            tokens, images, width, height, steps=steps, shift=float(shift),
                            seed=seed + index, caption_drop=self.caption_drop,
                            progress=on_progress,
                        )
                    else:
                        rgb = self.tower.generate(
                            tokens, width, height, steps=steps, shift=float(shift),
                            seed=seed + index, caption_drop=self.caption_drop,
                            progress=on_progress,
                        )
                except V2Error as error:
                    # The tower reports a cancel from the callback as an error.
                    if "cancelled" in str(error):
                        raise _Cancelled() from None
                    raise
                elapsed = time.monotonic() - started
                picture = Image.frombytes(self.pixel_mode, (width, height), rgb)
                buffer = io.BytesIO()
                picture.save(buffer, format="PNG")
                item = {
                    "b64_json": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "revised_prompt": prompt,
                    "seed": seed + index,
                    "seconds": round(elapsed, 2),
                }
                images.append(item)
                self.generated += 1
                if on_image is not None:
                    on_image(item, index, count)
        finally:
            self.busy = False
            self._lock.release()
        return {
            "created": int(time.time()),
            "model": self.model_name,
            "size": f"{width}x{height}",
            "steps": steps,
            "data": images,
        }
