"""Images on their way from an API request to the vision tower.

The HTTP layer turns an image part into an ``ImageInput`` (the encoded
bytes, however they arrived) and puts the model's placeholder text where
the picture sat in the message. The tokenizer facade then asks an
``ImagePreprocessor`` for the token count each picture occupies, expands
the placeholder to that many pad tokens, and hands the resized, normalized
pixels to the runtime beside the prompt.

Pillow does the decoding and resizing; it is an optional dependency
(``pip install flyweight[vision]``) so a text-only install carries nothing
extra.
"""
from __future__ import annotations

import base64
import binascii
import dataclasses
import hashlib
import io
import urllib.request
from collections import OrderedDict
from typing import Any, Mapping, Sequence

# What the Qwen chat template emits for one picture. Rendering the text
# directly rather than passing structured content keeps the template path
# identical for text-only requests; the tokenizer splits the control tokens
# back out, and the pad is expanded to the picture's token count afterwards.
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
IMAGE_PAD_TOKEN = "<|image_pad|>"

MAX_IMAGE_BYTES = 32 * 1024 * 1024
FETCH_TIMEOUT_SECONDS = 20.0


class ImageError(ValueError):
    """An image the request carried cannot be used; the message is client-facing."""


@dataclasses.dataclass(frozen=True)
class ImageInput:
    """An image as the request delivered it: encoded bytes plus their digest."""

    data: bytes
    digest: str

    @staticmethod
    def of(data: bytes) -> "ImageInput":
        return ImageInput(data, hashlib.blake2b(data, digest_size=16).hexdigest())


def _decode_data_url(url: str) -> bytes:
    header, _, payload = url.partition(",")
    if not payload:
        raise ImageError("data URL carries no payload")
    if ";base64" not in header:
        raise ImageError("data URL images must be base64 encoded")
    try:
        return base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as error:
        raise ImageError(f"data URL is not valid base64: {error}") from None


def fetch_image_url(url: str, *, allow_remote: bool) -> ImageInput:
    """Bytes behind an ``image_url``: a data URL, or http(s) when allowed."""
    if not isinstance(url, str) or not url:
        raise ImageError("image_url must be a non-empty string")
    if url.startswith("data:"):
        return ImageInput.of(_decode_data_url(url))
    if url.startswith(("http://", "https://")):
        if not allow_remote:
            raise ImageError(
                "remote image URLs are disabled on this server; send the image as a data URL"
            )
        request = urllib.request.Request(url, headers={"User-Agent": "flyweight"})
        try:
            with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
                data = response.read(MAX_IMAGE_BYTES + 1)
        except Exception as error:  # noqa: BLE001 - reported to the client
            raise ImageError(f"could not fetch image URL: {error}") from None
        if len(data) > MAX_IMAGE_BYTES:
            raise ImageError(f"image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MiB")
        return ImageInput.of(data)
    raise ImageError("image_url must be a data: URL or an http(s) URL")


def image_from_openai_part(part: Mapping[str, Any], *, allow_remote: bool) -> ImageInput:
    """An OpenAI ``image_url`` part (or Responses ``input_image``)."""
    value = part.get("image_url")
    url = value.get("url") if isinstance(value, Mapping) else value
    if not isinstance(url, str):
        raise ImageError("image_url must name a URL")
    return fetch_image_url(url, allow_remote=allow_remote)


def image_from_anthropic_block(block: Mapping[str, Any], *, allow_remote: bool) -> ImageInput:
    """An Anthropic ``image`` block: base64 source or URL source."""
    source = block.get("source")
    if not isinstance(source, Mapping):
        raise ImageError("image blocks need a source")
    kind = source.get("type")
    if kind == "base64":
        data = source.get("data")
        if not isinstance(data, str) or not data:
            raise ImageError("base64 image source carries no data")
        try:
            return ImageInput.of(base64.b64decode(data, validate=False))
        except (binascii.Error, ValueError) as error:
            raise ImageError(f"image source is not valid base64: {error}") from None
    if kind == "url":
        url = source.get("url")
        if not isinstance(url, str):
            raise ImageError("image URL source must name a URL")
        return fetch_image_url(url, allow_remote=allow_remote)
    raise ImageError(f"unsupported image source type {kind!r}")


@dataclasses.dataclass(frozen=True)
class PreparedImage:
    """Pixels as the tower wants them, plus the tokens they will occupy."""

    pixels: bytes          # width * height * 3 little-endian f32, HWC, normalized
    width: int
    height: int
    tokens: int
    grid_h: int
    grid_w: int
    hash: int              # 64-bit identity of the pixels, for prefix reuse


class ImagePreprocessor:
    """Decode, resize and normalize images for one model's tower.

    Results are cached by content digest and budget: a conversation replays
    its pictures on every turn, and the decode is the only part of the path
    that would otherwise repeat.
    """

    def __init__(self, model: Any, *, max_tokens: int = 1024, min_tokens: int = 4,
                 capacity: int = 32):
        self.model = model
        self.vision = model.vision
        self.max_tokens = int(max_tokens)
        self.min_tokens = int(min_tokens)
        self._cache: OrderedDict[tuple[str, int], PreparedImage] = OrderedDict()
        self._capacity = capacity

    @property
    def available(self) -> bool:
        return self.vision is not None

    def prepare(self, image: ImageInput) -> PreparedImage:
        if self.vision is None:
            raise ImageError(
                "this model has no vision tower attached; start the server with --mmproj"
            )
        key = (image.digest, self.max_tokens)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        try:
            from PIL import Image
        except ImportError:
            raise ImageError(
                "image input needs Pillow: pip install 'flyweight[vision]'"
            ) from None
        try:
            with Image.open(io.BytesIO(image.data)) as opened:
                picture = opened.convert("RGB")
        except Exception:  # noqa: BLE001 - reported to the client
            raise ImageError(
                "could not decode image: not a supported image format"
            ) from None
        resize = self.model.vision_resize(
            picture.width, picture.height,
            min_tokens=self.min_tokens, max_tokens=self.max_tokens,
        )
        resized = picture.resize(
            (resize["width"], resize["height"]), Image.Resampling.BICUBIC)
        mean = self.vision["image_mean"]
        std = self.vision["image_std"]
        pixels = _normalize(resized.tobytes(), resize["width"], resize["height"], mean, std)
        prepared = PreparedImage(
            pixels=pixels,
            width=resize["width"],
            height=resize["height"],
            tokens=resize["tokens"],
            grid_h=resize["grid_h"],
            grid_w=resize["grid_w"],
            hash=int.from_bytes(
                hashlib.blake2b(pixels, digest_size=8).digest(), "little"
            ),
        )
        self._cache[key] = prepared
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return prepared


def _normalize(rgb: bytes, width: int, height: int,
               mean: Sequence[float], std: Sequence[float]) -> bytes:
    """(x / 255 - mean) / std per channel, as little-endian f32 HWC."""
    try:
        import numpy as np
    except ImportError:
        return _normalize_pure(rgb, width, mean, std)
    raw = np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3).astype(np.float32)
    scaled = (raw / 255.0 - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return bytes(scaled.astype("<f4").tobytes())


def _normalize_pure(rgb: bytes, width: int,
                    mean: Sequence[float], std: Sequence[float]) -> bytes:
    import array
    scale = [1.0 / (255.0 * std[c]) for c in range(3)]
    offset = [-mean[c] / std[c] for c in range(3)]
    out = array.array("f", [0.0]) * len(rgb)
    for index, byte in enumerate(rgb):
        channel = index % 3
        out[index] = byte * scale[channel] + offset[channel]
    return out.tobytes()


def expand_image_pads(tokens: Sequence[int], pad_id: int,
                      counts: Sequence[int]) -> list[int]:
    """Replace each pad token, in order, with ``counts[i]`` copies of it.

    The number of pads must equal the number of images: a stray placeholder
    typed into a message would otherwise take an image's tokens.
    """
    pads = [index for index, token in enumerate(tokens) if token == pad_id]
    if len(pads) != len(counts):
        raise ImageError(
            f"prompt has {len(pads)} image placeholder(s) but {len(counts)} image(s)"
        )
    out: list[int] = []
    cursor = 0
    for position, count in zip(pads, counts):
        out.extend(tokens[cursor:position])
        out.extend([pad_id] * int(count))
        cursor = position + 1
    out.extend(tokens[cursor:])
    return out


def image_token_offsets(tokens: Sequence[int], pad_id: int,
                        counts: Sequence[int]) -> list[int]:
    """Where each image's run of pad tokens starts in an expanded prompt."""
    offsets: list[int] = []
    index = 0
    for count in counts:
        while index < len(tokens) and tokens[index] != pad_id:
            index += 1
        if index >= len(tokens):
            raise ImageError("expanded prompt lost an image placeholder")
        run = 0
        while index + run < len(tokens) and tokens[index + run] == pad_id and run < count:
            run += 1
        if run != count:
            raise ImageError("image placeholder run does not match its token count")
        offsets.append(index)
        index += run
    return offsets
