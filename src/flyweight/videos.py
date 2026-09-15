"""Video generation served beside the chat model: MiniMax-H3 on the native tower.

``VideoGenerator`` owns the three component models (the Qwen3-VL text
encoder, the H3 transformer, the video VAE) and the tower built over them,
turns a ``/v1/videos/generations`` request into a run, and returns the clip
as a base64 MP4. The denoising loop lives here: the tower exposes one
transformer forward and one VAE decode, and the rectified-flow schedule is a
few lines of numpy.

The model directory holds the GGUF weights and the configs beside them::

    text_encoder.gguf      Qwen3-VL-32B conditioner (unsloth/MiniMax-H3-GGUF)
    transformer.gguf       the DiT (unsloth/MiniMax-H3-GGUF)
    vae/                   model.safetensors + config.json (the fp16 video VAE)
    text_encoder/config.json, transformer/config.json, tokenizer/
                           from the MiniMaxAI/MiniMax-H3 release

PyAV writes the MP4 and is an optional dependency (``pip install
flyweight-llm[video]``); without it the endpoint reports why.
"""
from __future__ import annotations

import base64
import io
import math
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from .images import _Cancelled, _int_option, snapshot_model_name
from .server import APIError
from .v2 import V2Diffusion, V2Model

FPS = 24
DEFAULT_STEPS = 8
DEFAULT_SHIFT = 12.0
DEFAULT_AUDIO_SHIFT = 3.0
DEFAULT_FRAMES = 124
MAX_STEPS = 50
MAX_PROMPT_TOKENS = 512
CANVAS_MULTIPLE = 32
LATENT_CHANNELS = 24
AUDIO_CHANNELS = 2
AUDIO_LATENT_CHANNELS = 32
AUDIO_LATENTS_PER_SECOND = 40


def align_frames(frames: int) -> int:
    """The next ``17n + 5`` the video VAE can decode, at least 5."""
    return 17 * max(0, math.ceil((frames - 5) / 17)) + 5


def latent_frames(frames: int) -> int:
    """Latent frames for an aligned frame count: ``5n + 2``."""
    return 5 * ((frames - 5) // 17) + 2


def audio_latents(frames: int) -> int:
    return int(round(frames / FPS * AUDIO_LATENTS_PER_SECOND))


def sigma_schedule(steps: int, shift: float) -> list[float]:
    """``linspace(1, 0, steps)`` pushed through the exponential shift, duplicates collapsed."""
    sigmas: list[float] = []
    for index in range(steps):
        base = 1.0 - index / (steps - 1)
        sigma = shift * base / (1.0 + (shift - 1.0) * base)
        if not sigmas or sigma != sigmas[-1]:
            sigmas.append(sigma)
    return sigmas


def parse_video_size(value: Any, default: tuple[int, int], limit: tuple[int, int]) -> tuple[int, int]:
    """``"WxH"`` (or ``"auto"``) into pixels: multiples of 32, within the tower's limits."""
    if value is None or value == "auto":
        return default
    if not isinstance(value, str) or "x" not in value:
        raise APIError(400, "size must look like 640x384", parameter="size")
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except ValueError as error:
        raise APIError(400, "size must look like 640x384", parameter="size") from error
    if width <= 0 or height <= 0 or width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise APIError(400, f"size sides must be positive multiples of {CANVAS_MULTIPLE}", parameter="size")
    if width > limit[0] or height > limit[1]:
        raise APIError(
            400,
            f"size exceeds this server's limit of {limit[0]}x{limit[1]} "
            "(start with a larger --video-max-size)",
            parameter="size",
        )
    return width, height


def resolve_model_dir(root: Path) -> dict[str, Path]:
    """The component paths of a video model directory, with a message naming what is missing."""
    parts = {
        "text_encoder.gguf": root / "text_encoder.gguf",
        "transformer.gguf": root / "transformer.gguf",
        "vae/config.json": root / "vae" / "config.json",
        "text_encoder/config.json": root / "text_encoder" / "config.json",
        "transformer/config.json": root / "transformer" / "config.json",
        "tokenizer/tokenizer.json": root / "tokenizer" / "tokenizer.json",
    }
    missing = [name for name, path in parts.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{root} is not a MiniMax-H3 model directory: missing {', '.join(missing)} "
            "(see the README's video section for the layout)"
        )
    return {
        "encoder": parts["text_encoder.gguf"],
        "transformer": parts["transformer.gguf"],
        "vae": root / "vae",
        "encoder_config": parts["text_encoder/config.json"],
        "transformer_config": parts["transformer/config.json"],
        "tokenizer": root / "tokenizer",
    }


def encode_mp4(frames: Any, width: int, height: int, fps: int = FPS) -> bytes:
    """H.264 MP4 of uint8 frames ``[F][H][W][3]`` (numpy array)."""
    try:
        import av
    except ImportError as error:
        raise APIError(
            500, "video output needs PyAV: pip install flyweight-llm[video]", "server_error"
        ) from error
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium", "movflags": "+faststart"}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


class VideoGenerator:
    """The video model behind ``/v1/videos/generations``."""

    def __init__(
        self,
        model_dir: Path | str,
        *,
        device: int = 0,
        max_width: int = 640,
        max_height: int = 384,
        max_frames: int = DEFAULT_FRAMES,
        weights: str = "host",
        precision: str = "balanced",
        model_name: str | None = None,
    ) -> None:
        root = Path(model_dir)
        paths = resolve_model_dir(root)
        self.path = root
        self.model_name = model_name or snapshot_model_name(root)
        self.max_width = int(max_width)
        self.max_height = int(max_height)
        self.max_frames = align_frames(int(max_frames))
        self.encoder = V2Model(paths["encoder"])
        self.encoder.attach_config(paths["encoder_config"])
        self.encoder.attach_tokenizer(paths["tokenizer"])
        self.transformer = V2Model(paths["transformer"])
        self.transformer.attach_config(paths["transformer_config"])
        self.vae = V2Model(paths["vae"])
        try:
            self.tower = V2Diffusion.h3(
                self.encoder, self.transformer, self.vae,
                device=device, max_prompt_tokens=MAX_PROMPT_TOKENS, max_width=self.max_width,
                max_height=self.max_height, max_frames=self.max_frames, weights=weights, precision=precision,
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
            "max_frames": self.max_frames,
            "fps": FPS,
            "default_frames": min(DEFAULT_FRAMES, self.max_frames),
            "default_steps": DEFAULT_STEPS,
            "weights": "host" if info["host_weights"] else "device",
            "precision": "exact" if info["exact"] else "balanced" if info["balanced"] else "fast",
            "device_mib": int(info["device_bytes"]) // (1024 * 1024),
            "host_mib": int(info["host_bytes"]) // (1024 * 1024),
            "busy": self.busy,
            "generated": self.generated,
        }

    def tokenize(self, prompt: str) -> list[int]:
        # The conditioner reads the raw prompt: no chat template, no special tokens.
        tokens = list(self.encoder.tokenize(prompt))
        if not tokens:
            raise APIError(400, "prompt tokenized to nothing", parameter="prompt")
        if len(tokens) > MAX_PROMPT_TOKENS:
            raise APIError(
                400,
                f"prompt is {len(tokens)} tokens; the encoder reads at most {MAX_PROMPT_TOKENS}",
                parameter="prompt",
            )
        return tokens

    def stream(self, payload: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
        """``generate`` as server-sent events: ``progress`` per denoising step
        (plus one for the VAE decode), one ``video``, then ``done``. Closing
        the iterator cancels the run at the next step."""
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        cancelled = threading.Event()
        outcome: dict[str, Any] = {}

        def on_progress(step: int, total: int, stage: str) -> None:
            events.put({"type": "progress", "step": step, "steps": total, "stage": stage})
            if cancelled.is_set():
                raise _Cancelled()

        def worker() -> None:
            try:
                result = self.generate(payload, progress=on_progress)
                for index, item in enumerate(result["data"]):
                    events.put({"type": "video", "index": index, "count": len(result["data"]), **item})
                outcome["result"] = result
            except BaseException as error:  # noqa: BLE001 - handed to the stream
                outcome["error"] = error
            finally:
                events.put(None)

        thread = threading.Thread(target=worker, name="video-render", daemon=True)
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
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise APIError(400, "prompt must be non-empty text", parameter="prompt")
        response_format = payload.get("response_format", "b64_json")
        if response_format != "b64_json":
            raise APIError(
                400, "only response_format b64_json is available", parameter="response_format"
            )
        width, height = parse_video_size(
            payload.get("size"), (self.max_width, self.max_height), (self.max_width, self.max_height)
        )
        frames = align_frames(_int_option(payload, "frames", min(DEFAULT_FRAMES, self.max_frames), 5, self.max_frames))
        if frames > self.max_frames:
            raise APIError(
                400, f"frames rounds up to {frames}; this server decodes at most {self.max_frames}", parameter="frames"
            )
        steps = _int_option(payload, "steps", DEFAULT_STEPS, 2, MAX_STEPS)
        seed_value = payload.get("seed")
        if seed_value is None:
            seed = int(time.time_ns()) & 0xFFFFFFFFFFFF
        elif isinstance(seed_value, bool) or not isinstance(seed_value, int) or seed_value < 0:
            raise APIError(400, "seed must be a non-negative integer", parameter="seed")
        else:
            seed = seed_value
        shifts = []
        for name, default in (("shift", DEFAULT_SHIFT), ("audio_shift", DEFAULT_AUDIO_SHIFT)):
            value = payload.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise APIError(400, f"{name} must be a positive number", parameter=name)
            shifts.append(float(value))
        try:
            import numpy as np
        except ImportError as error:
            raise APIError(
                500, "video generation needs numpy: pip install flyweight-llm[video]", "server_error"
            ) from error

        tokens = self.tokenize(prompt)
        if not self._lock.acquire(timeout=0.0):
            raise APIError(429, "a video is already rendering; retry shortly", "rate_limit_error")
        try:
            self.busy = True
            started = time.monotonic()
            latent_t, latent_h, latent_w = latent_frames(frames), height // 16, width // 16
            audio_count = audio_latents(frames)
            video_sigmas = sigma_schedule(steps, shifts[0])
            audio_sigmas = sigma_schedule(steps, shifts[1])
            evaluations = min(len(video_sigmas), len(audio_sigmas)) - 1
            total = evaluations + 1

            def report(step: int, stage: str) -> None:
                if progress is not None:
                    progress(step, total, stage)

            report(0, "encode")
            caption = self.tower.encode_text(tokens)
            rng = np.random.default_rng(seed)
            video = rng.standard_normal((LATENT_CHANNELS, latent_t, latent_h, latent_w), dtype=np.float32)
            audio = rng.standard_normal((AUDIO_CHANNELS, audio_count, AUDIO_LATENT_CHANNELS), dtype=np.float32)
            for step in range(evaluations):
                t_video = np.float32(1.0 - np.float32(video_sigmas[step]))
                t_audio = np.float32(1.0 - np.float32(audio_sigmas[step]))
                velocity_v, velocity_a = self.tower.h3_step(
                    video.ravel(), latent_t, latent_h, latent_w, audio.ravel(), audio_count,
                    caption, len(tokens), float(t_video), float(t_audio),
                )
                velocity_v = np.frombuffer(velocity_v, dtype=np.float32).reshape(video.shape)
                velocity_a = np.frombuffer(velocity_a, dtype=np.float32).reshape(audio.shape)
                # Data-ward velocity: x0 = x_t + sigma v, then the Euler blend toward the next sigma.
                for sample, velocity, sigmas, t in (
                    (video, velocity_v, video_sigmas, t_video), (audio, velocity_a, audio_sigmas, t_audio),
                ):
                    denoised = sample + (np.float32(1.0) - t) * velocity
                    ratio = np.float32(sigmas[step + 1] / sigmas[step])
                    sample[...] = ratio * sample + (np.float32(1.0) - ratio) * denoised
                report(step + 1, "denoise")
            rgb, decoded = self.tower.h3_decode(video.ravel(), latent_t, latent_h, latent_w)
            pixels = np.frombuffer(rgb, dtype=np.uint8).reshape(decoded, height, width, 3)[:frames]
            clip = encode_mp4(pixels, width, height)
            elapsed = time.monotonic() - started
            self.generated += 1
        finally:
            self.busy = False
            self._lock.release()
        return {
            "created": int(time.time()),
            "model": self.model_name,
            "size": f"{width}x{height}",
            "frames": int(pixels.shape[0]),
            "fps": FPS,
            "steps": steps,
            "data": [{
                "b64_json": base64.b64encode(clip).decode("ascii"),
                "mime_type": "video/mp4",
                "revised_prompt": prompt,
                "seed": seed,
                "seconds": round(elapsed, 2),
            }],
        }
