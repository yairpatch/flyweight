"""Image parts on their way from the API to the runtime: placeholders,
pad expansion, prefix-cache keys, and the errors a request without a tower
gets. No weights and no Pillow needed."""
from __future__ import annotations

import base64
import unittest

from flyweight import vision
from flyweight.server import _anthropic_request, _chat_messages, _validate_response_input
from flyweight.v2_server import NativeV2Tokenizer, _chat_key
from flyweight.vision import (
    IMAGE_PLACEHOLDER, ImageError, ImageInput, PreparedImage, expand_image_pads,
    image_token_offsets,
)

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
DATA_URL = "data:image/png;base64," + base64.b64encode(PNG).decode()


class PadHelperTests(unittest.TestCase):
    def test_expand_replaces_each_pad_in_order(self) -> None:
        self.assertEqual(expand_image_pads([1, 7, 2, 7, 3], 7, [3, 1]),
                         [1, 7, 7, 7, 2, 7, 3])

    def test_expand_refuses_mismatched_counts(self) -> None:
        with self.assertRaises(ImageError):
            expand_image_pads([1, 7, 2], 7, [3, 1])
        with self.assertRaises(ImageError):
            expand_image_pads([1, 2], 7, [3])

    def test_offsets_follow_the_pad_runs(self) -> None:
        tokens = expand_image_pads([1, 7, 2, 7, 3], 7, [3, 2])
        self.assertEqual(image_token_offsets(tokens, 7, [3, 2]), [1, 5])


class ContentParsingTests(unittest.TestCase):
    def test_openai_image_url_part_becomes_placeholder_and_image(self) -> None:
        messages, _ = _chat_messages({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": DATA_URL}},
            {"type": "text", "text": "what is this?"},
        ]}]})
        turn = messages[-1]
        self.assertEqual(turn["content"], IMAGE_PLACEHOLDER + "what is this?")
        self.assertEqual([image.data for image in turn["images"]], [PNG])

    def test_remote_url_can_be_denied(self) -> None:
        messages, _ = _chat_messages({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
        ]}]}, allow_remote_images=False)
        self.assertIn("remote image URLs are disabled", messages[-1]["content"])

    def test_bad_data_url_degrades_to_a_note(self) -> None:
        messages, _ = _chat_messages({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png,plain"}},
            {"type": "text", "text": "hi"},
        ]}]})
        self.assertIn("[image omitted:", messages[-1]["content"])
        self.assertNotIn("images", messages[-1])

    def test_anthropic_image_block(self) -> None:
        _, messages, _ = _anthropic_request({"max_tokens": 5, "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "look: "},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.b64encode(PNG).decode()}},
            ]},
        ]})
        turn = messages[-1]
        self.assertEqual(turn["content"], "look: " + IMAGE_PLACEHOLDER)
        self.assertEqual(turn["images"][0].data, PNG)

    def test_responses_input_image(self) -> None:
        messages = _validate_response_input([{"role": "user", "content": [
            {"type": "input_image", "image_url": DATA_URL},
            {"type": "input_text", "text": "hi"},
        ]}])
        self.assertEqual(messages[0]["content"], IMAGE_PLACEHOLDER + "hi")
        self.assertEqual(len(messages[0]["images"]), 1)

    def test_chat_key_distinguishes_pictures(self) -> None:
        a = [{"role": "user", "content": IMAGE_PLACEHOLDER, "images": [ImageInput.of(b"a")]}]
        b = [{"role": "user", "content": IMAGE_PLACEHOLDER, "images": [ImageInput.of(b"b")]}]
        self.assertNotEqual(_chat_key(a), _chat_key(b))
        self.assertEqual(_chat_key(a), _chat_key([dict(a[0])]))


class _StubModel:
    """Enough of V2Model for the tokenizer facade: a tower with 32-pixel
    tokens, and a byte-per-character tokenizer where <|image_pad|> is 7."""

    info = {"architecture": "qwen35"}
    config = {}
    chat_template = None
    vision = {
        "patch_size": 16, "spatial_merge_size": 2, "embedding_length": 8,
        "projection_dim": 4, "block_count": 1, "deepstack_layers": 0,
        "row_width": 4, "image_mean": (0.5, 0.5, 0.5), "image_std": (0.5, 0.5, 0.5),
    }

    def token_id(self, text):
        if text == "<|image_pad|>":
            return 7
        raise KeyError(text)

    def tokenize(self, text):
        tokens = []
        for piece in text.replace("<|image_pad|>", "\0<|image_pad|>\0").split("\0"):
            tokens.extend([7] if piece == "<|image_pad|>" else [ord(c) for c in piece])
        return tokens

    def vision_resize(self, width, height, *, min_tokens=1, max_tokens=4096):
        return {"width": 64, "height": 32, "grid_w": 2, "grid_h": 1, "tokens": 2}


class _NoTowerModel(_StubModel):
    vision = None


class TokenizerFacadeTests(unittest.TestCase):
    def _tokenizer(self, model):
        tokenizer = NativeV2Tokenizer(model)
        # The stub has no chat template: format_messages would fall back to
        # the architecture formatter, whose exact text is not under test
        # here, so turns render as their content.
        tokenizer.format_messages = lambda messages, **_: "".join(
            str(message["content"]) for message in messages)
        return tokenizer

    def test_pads_expand_to_the_image_token_count(self) -> None:
        tokenizer = self._tokenizer(_StubModel())
        prepared = PreparedImage(b"\0" * (64 * 32 * 12), 64, 32, 2, 1, 2, 42)
        tokenizer.images.prepare = lambda image: prepared  # type: ignore[union-attr]
        messages = [{"role": "user", "content": "a<|image_pad|>b",
                     "images": [ImageInput.of(b"x")]}]
        tokens = tokenizer.encode_messages(messages)
        self.assertEqual(tokens, [ord("a"), 7, 7, ord("b")])
        images = tokenizer.prompt_images(messages, tokens)
        self.assertEqual((images[0].token_offset, images[0].tokens, images[0].hash), (1, 2, 42))

    def test_text_only_prompt_is_untouched(self) -> None:
        tokenizer = self._tokenizer(_StubModel())
        self.assertEqual(tokenizer.encode_messages([{"role": "user", "content": "ab"}]),
                         [ord("a"), ord("b")])

    def test_images_without_a_tower_degrade_to_a_note(self) -> None:
        tokenizer = self._tokenizer(_NoTowerModel())
        tokens = tokenizer.encode_messages([{
            "role": "user", "content": "a" + IMAGE_PLACEHOLDER + "b",
            "images": [ImageInput.of(b"x")]}])
        self.assertEqual("".join(chr(t) for t in tokens),
                         "a[image omitted: no vision tower attached (--mmproj)]b")
        self.assertEqual(tokenizer.prompt_images(
            [{"role": "user", "content": IMAGE_PLACEHOLDER, "images": [ImageInput.of(b"x")]}],
            tokens), [])


class FetchTests(unittest.TestCase):
    def test_data_url_roundtrip(self) -> None:
        self.assertEqual(vision.fetch_image_url(DATA_URL, allow_remote=False).data, PNG)

    def test_unknown_scheme(self) -> None:
        with self.assertRaises(ImageError):
            vision.fetch_image_url("file:///etc/passwd", allow_remote=True)


if __name__ == "__main__":
    unittest.main()
