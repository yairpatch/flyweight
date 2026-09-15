"""Image generation served beside the chat model: Z-Image-Turbo on the native tower.

``ImageGenerator`` owns the three component models of a diffusers snapshot
(``text_encoder``, ``transformer``, ``vae``) and the tower built over them,
turns an OpenAI-style ``/v1/images/generations`` request into a run, and
returns the picture as base64 PNG. One image renders at a time: the tower
has one workspace and the GPU is shared with the chat runtime.

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
from typing import Any, Callable, Iterator, Mapping

from .server import APIError
from .v2 import V2Diffusion, V2Error, V2Model

# The Qwen3 chat template with add_generation_prompt=True and thinking left
# enabled, which is how the reference pipeline conditions the DiT.
_PROMPT_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

DEFAULT_STEPS = 8
DEFAULT_SHIFT = 3.0
MAX_STEPS = 50
MAX_PROMPT_TOKENS = 512


def _int_option(payload: Mapping[str, Any], name: str, default: int, low: int, high: int) -> int:
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise APIError(400, f"{name} must be an integer", parameter=name)
    if value < low or value > high:
        raise APIError(400, f"{name} must be between {low} and {high}", parameter=name)
    return value


def parse_size(value: Any, default: tuple[int, int], limit: tuple[int, int]) -> tuple[int, int]:
    """``"WxH"`` (or ``"auto"``) into pixels: multiples of 16, within the tower's limits."""
    if value is None or value == "auto":
        return default
    if not isinstance(value, str) or "x" not in value:
        raise APIError(400, "size must look like 1024x1024", parameter="size")
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except ValueError as error:
        raise APIError(400, "size must look like 1024x1024", parameter="size") from error
    if width <= 0 or height <= 0 or width % 16 or height % 16:
        raise APIError(400, "size sides must be positive multiples of 16", parameter="size")
    if width > limit[0] or height > limit[1]:
        raise APIError(
            400,
            f"size exceeds this server's limit of {limit[0]}x{limit[1]} "
            "(start with a larger --image-max-size)",
            parameter="size",
        )
    return width, height


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
        max_width: int = 1024,
        max_height: int = 1024,
        weights: str = "auto",
        precision: str = "balanced",
        model_name: str | None = None,
    ) -> None:
        root = Path(snapshot)
        for part in ("text_encoder", "transformer", "vae"):
            if not (root / part / "config.json").is_file():
                raise FileNotFoundError(
                    f"{root} is not a Z-Image snapshot: missing {part}/config.json"
                )
        self.path = root
        self.model_name = model_name or snapshot_model_name(root)
        self.max_width = int(max_width)
        self.max_height = int(max_height)
        self.encoder = V2Model(root / "text_encoder")
        self.transformer = V2Model(root / "transformer")
        self.vae = V2Model(root / "vae")
        try:
            self.tower = V2Diffusion(
                self.encoder, self.transformer, self.vae,
                device=device, max_width=self.max_width, max_height=self.max_height,
                max_prompt_tokens=MAX_PROMPT_TOKENS, weights=weights, precision=precision,
            )
        except BaseException:
            self.close()
            raise
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
            "default_steps": DEFAULT_STEPS,
            "weights": "host" if info["host_weights"] else "device",
            "precision": "exact" if info["exact"] else "balanced" if info["balanced"] else "fast",
            "device_mib": int(info["device_bytes"]) // (1024 * 1024),
            "host_mib": int(info["host_bytes"]) // (1024 * 1024),
            "busy": self.busy,
            "generated": self.generated,
        }

    def tokenize(self, prompt: str) -> list[int]:
        tokens = list(self.encoder.tokenize(_PROMPT_TEMPLATE.format(prompt=prompt)))
        if len(tokens) > MAX_PROMPT_TOKENS:
            raise APIError(
                400,
                f"prompt is {len(tokens)} tokens; the encoder reads at most {MAX_PROMPT_TOKENS}",
                parameter="prompt",
            )
        return tokens

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
        width, height = parse_size(
            payload.get("size"), (self.max_width, self.max_height), (self.max_width, self.max_height)
        )
        steps = _int_option(payload, "steps", DEFAULT_STEPS, 1, MAX_STEPS)
        seed_value = payload.get("seed")
        if seed_value is None:
            seed = int(time.time_ns()) & 0xFFFFFFFFFFFF
        elif isinstance(seed_value, bool) or not isinstance(seed_value, int) or seed_value < 0:
            raise APIError(400, "seed must be a non-negative integer", parameter="seed")
        else:
            seed = seed_value
        shift = payload.get("shift", DEFAULT_SHIFT)
        if isinstance(shift, bool) or not isinstance(shift, (int, float)) or shift <= 0:
            raise APIError(400, "shift must be a positive number", parameter="shift")
        try:
            from PIL import Image
        except ImportError as error:
            raise APIError(
                500, "image output needs Pillow: pip install flyweight-llm[vision]", "server_error"
            ) from error

        tokens = self.tokenize(prompt)
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
                    rgb = self.tower.generate(
                        tokens, width, height, steps=steps, shift=float(shift),
                        seed=seed + index, progress=on_progress,
                    )
                except V2Error as error:
                    # The tower reports a cancel from the callback as an error.
                    if "cancelled" in str(error):
                        raise _Cancelled() from None
                    raise
                elapsed = time.monotonic() - started
                picture = Image.frombytes("RGB", (width, height), rgb)
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
