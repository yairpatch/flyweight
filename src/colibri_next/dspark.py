"""Native state for DeepSeek V4 Flash's DSpark draft sidecar."""

from __future__ import annotations

import ctypes
import numpy as np

from .v2 import V2Error, _library


class DsparkRuntime:
    def __init__(self, model, context_limit: int):
        self._library = _library()
        self._model = model
        self._width = int(model.config["kv_lora_rank"])
        handle = ctypes.c_void_p()
        self._check(self._library.colibri_v2_dspark_runtime_create(
            model._handle, context_limit, ctypes.byref(handle)
        ))
        self._handle = handle

    def _check(self, status: int) -> None:
        if status:
            raise V2Error((self._library.colibri_v2_last_error() or b"DSpark error").decode(errors="replace"))

    def inject(self, fused) -> None:
        values = np.ascontiguousarray(fused, dtype=np.float32).reshape(-1)
        self._check(self._library.colibri_v2_dspark_inject(
            self._handle, values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), values.size
        ))

    def cached(self, layer: int, position: int) -> np.ndarray:
        output = np.empty(self._width, dtype=np.float32)
        self._check(self._library.colibri_v2_dspark_cached(
            self._handle, layer, position,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output.size,
        ))
        return output

    def heads(self, base_logits, hidden, anchor_token: int):
        base = np.ascontiguousarray(base_logits, dtype=np.float32)
        states = np.ascontiguousarray(hidden, dtype=np.float32)
        if base.ndim != 2 or states.ndim != 2 or base.shape[0] != states.shape[0]:
            raise ValueError("base logits and hidden states must have matching rows")
        if base.shape[1] != int(self._model.config["vocabulary_size"]):
            raise ValueError("base logits have the wrong vocabulary width")
        if states.shape[1] != int(self._model.config["hidden_size"]):
            raise ValueError("hidden states have the wrong width")
        rows = base.shape[0]
        output = np.empty_like(base)
        confidence = np.empty(rows, dtype=np.float32)
        tokens = np.empty(rows, dtype=np.uint32)
        fp = ctypes.POINTER(ctypes.c_float)
        self._check(self._library.colibri_v2_dspark_heads(
            self._model._handle,
            base.ctypes.data_as(fp), states.ctypes.data_as(fp), rows, anchor_token,
            output.ctypes.data_as(fp), confidence.ctypes.data_as(fp),
            tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ))
        return output, confidence, tokens

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.colibri_v2_dspark_runtime_free(self._handle)
            self._handle = None
            self._model = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class DsparkSession:
    """Keep target capture, feature fusion, and draft KV injection in lockstep."""

    def __init__(self, target_model, sidecar_model, context_limit: int):
        from .deepseek4 import Deepseek4Runtime

        if sidecar_model.config["architecture"] != "dflash":
            raise ValueError("sidecar model must use the dflash architecture")
        target_layers = tuple(sidecar_model.target_layers)
        if not target_layers:
            raise ValueError("sidecar does not declare target layers")
        self.target = Deepseek4Runtime(target_model, context_limit)
        try:
            self.target.capture_layers(target_layers)
            self.draft = DsparkRuntime(sidecar_model, context_limit)
        except Exception:
            self.target.close()
            raise
        self.sidecar_model = sidecar_model

    def forward_target(self, token: int, *, logits: bool = True):
        result = self.target.forward(token, logits=logits)
        fused = self.sidecar_model.dspark_encode(self.target.captured)
        self.draft.inject(fused)
        return result

    def close(self) -> None:
        if getattr(self, "draft", None) is not None:
            self.draft.close()
            self.draft = None
        if getattr(self, "target", None) is not None:
            self.target.close()
            self.target = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
