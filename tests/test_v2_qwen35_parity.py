"""Qwen3.5 safetensors loading, checked against transformers' own forward pass.

The name-translation tests next door pin what the loader *calls* each tensor.
This pins what it does to the values: the checkpoint is loaded unquantized
(``COLIBRI_HF_QUANT=F32``) into both this runtime and transformers'
``Qwen3_5ForCausalLM``, and the greedy token at every position must agree.

That is the only check here that could have caught the two bugs this file was
written for -- a fixture whose q_proj was half the width ``attn_output_gate``
makes it, and a convolution state sized from the first two extents of a
descriptor that carries torch's singleton channel dimension, which put the conv
state on top of the recurrent state and left the delta layers reading their own
convolution history as a recurrent state.

Skipped without torch and transformers, which are not runtime dependencies.
"""

from __future__ import annotations

import importlib.machinery
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from colibri_next.v2 import V2Model
from tests import qwen35_hf_fixture as fixture

TOKENS = [3, 9, 17, 4, 21, 33, 8, 12]


def _load_transformers():
    """Imports the reference model, or returns None when it is unavailable."""
    # transformers pulls in torchaudio through audio_utils, and an installation
    # whose torchaudio disagrees with torch about CUDA raises on import. The
    # decoder does not touch audio, so a stub keeps a broken side dependency
    # from deciding whether this test can run.
    if "torchaudio" not in sys.modules:
        stub = types.ModuleType("torchaudio")
        stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
        stub.__version__ = "0"
        sys.modules["torchaudio"] = stub
    try:
        import torch  # noqa: F401
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except Exception:  # pragma: no cover - depends on the environment
        return None
    # The hub/fla kernels refuse float32 and are not the reference; the plain
    # torch implementations the decorators wrap are.
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update"):
        wrapped = getattr(modeling_qwen3_5, name, None)
        if wrapped is not None and hasattr(wrapped, "__wrapped__"):
            setattr(modeling_qwen3_5, name, wrapped.__wrapped__)
    return modeling_qwen3_5


def _checkpoint_tensors(directory: Path) -> dict[str, np.ndarray]:
    """Every shard's bf16 payload, widened to f32."""
    out: dict[str, np.ndarray] = {}
    for shard in sorted(directory.glob("*.safetensors")):
        raw = shard.read_bytes()
        length = int.from_bytes(raw[:8], "little")
        header = json.loads(raw[8:8 + length])
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            bits = np.frombuffer(raw, np.uint16, (end - start) // 2,
                                 8 + length + start)
            out[name] = ((bits.astype(np.uint32) << 16).view(np.float32)
                         .reshape(entry["shape"]).copy())
    return out


class Qwen35ForwardParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.modeling = _load_transformers()
        if cls.modeling is None:
            raise unittest.SkipTest("torch and transformers are needed for parity")
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls._directory.name) / "qwen35"
        fixture.build(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "_directory", None):
            cls._directory.cleanup()

    def reference_tokens(self) -> list[int]:
        import torch

        text = dict(fixture.CONFIG["text_config"])
        text.pop("model_type", None)
        config = self.modeling.Qwen3_5TextConfig(**text, layer_types=[
            "full_attention" if fixture.is_full_attention(layer) else "linear_attention"
            for layer in range(fixture.LAYERS)])
        config._attn_implementation = "eager"
        model = self.modeling.Qwen3_5ForCausalLM(config).to(torch.float32).eval()
        # The MTP block and the vision tower have no counterpart in the
        # reference decoder; everything else must land, so this is strict.
        model.load_state_dict(
            {name.replace("model.language_model.", "model."): torch.from_numpy(value)
             for name, value in _checkpoint_tensors(self.path).items()
             if not name.startswith(("model.visual.", "mtp."))}, strict=True)
        with torch.no_grad():
            logits = model(torch.tensor([TOKENS]), use_cache=False).logits[0]
        return logits.argmax(-1).tolist()

    def runtime_tokens(self) -> list[int]:
        previous = os.environ.get("COLIBRI_HF_QUANT")
        os.environ["COLIBRI_HF_QUANT"] = "F32"
        try:
            V2Model.select_backend("cpu")
            with V2Model(str(self.path)) as model:
                with model.native_qwen_runtime(context_limit=256) as runtime:
                    runtime.prepare()
                    return [runtime.decode(token) for token in TOKENS]
        finally:
            if previous is None:
                os.environ.pop("COLIBRI_HF_QUANT", None)
            else:
                os.environ["COLIBRI_HF_QUANT"] = previous

    def test_greedy_tokens_match_transformers(self) -> None:
        self.assertEqual(self.runtime_tokens(), self.reference_tokens())


if __name__ == "__main__":
    unittest.main()
