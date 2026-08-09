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

    def attention(self, layer: int, queries, noise_kv):
        q = np.ascontiguousarray(queries, dtype=np.float32)
        kv = np.ascontiguousarray(noise_kv, dtype=np.float32)
        heads = int(self._model.config["attention_heads"])
        if q.ndim != 3 or q.shape[1:] != (heads, self._width):
            raise ValueError("queries must have shape [rows, heads, kv_width]")
        if kv.shape != (q.shape[0], self._width):
            raise ValueError("noise KV must have shape [rows, kv_width]")
        output = np.empty_like(q)
        fp = ctypes.POINTER(ctypes.c_float)
        self._check(self._library.colibri_v2_dspark_attention(
            self._handle, layer, q.ctypes.data_as(fp), kv.ctypes.data_as(fp),
            q.shape[0], output.ctypes.data_as(fp), output.size,
        ))
        return output

    def attention_stage(self, layer: int, streams):
        values = np.ascontiguousarray(streams, dtype=np.float32)
        expected = (int(self._model.config["hyper_connection_count"]),
                    int(self._model.config["hidden_size"]))
        if values.ndim != 3 or values.shape[1:] != expected:
            raise ValueError("streams must have shape [rows, hc, hidden]")
        output = np.empty_like(values); fp = ctypes.POINTER(ctypes.c_float)
        self._check(self._library.colibri_v2_dspark_attention_stage(
            self._handle, layer, values.ctypes.data_as(fp), values.shape[0],
            output.ctypes.data_as(fp), output.size,
        ))
        return output

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
