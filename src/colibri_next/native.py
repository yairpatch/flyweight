from __future__ import annotations

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .kernels import Q4SwiGLUExpert
    from .q4 import Q4BlockTensor


FEATURE_AVX2 = 1 << 0
FEATURE_AVX512 = 1 << 1


class NativeUnavailableError(RuntimeError):
    pass


class NativeBackend:
    def __init__(self, path: Path | str | None = None):
        library_path = Path(path) if path is not None else _library_path()
        if not library_path.is_file():
            raise NativeUnavailableError(
                f"native library not found: {library_path}; "
                "run python -m colibri_next.native_build"
            )
        try:
            library = ctypes.CDLL(str(library_path))
        except OSError as error:
            raise NativeUnavailableError(
                f"failed to load native library {library_path}: {error}"
            ) from error
        library.colibri_native_version.argtypes = []
        library.colibri_native_version.restype = ctypes.c_uint32
        library.colibri_cpu_features.argtypes = []
        library.colibri_cpu_features.restype = ctypes.c_uint32
        library.colibri_q4_matvec.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        library.colibri_q4_matvec.restype = ctypes.c_int
        self.path = library_path
        self.library = library
        self.version = int(library.colibri_native_version())
        self._fused_moe = self.version >= 2
        if self._fused_moe:
            library.colibri_q4_moe.argtypes = [
                ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
            ]
            library.colibri_q4_moe.restype = ctypes.c_int
        self.feature_mask = int(library.colibri_cpu_features())
        self._executor = ThreadPoolExecutor(
            max_workers=min(16, os.cpu_count() or 1),
            thread_name_prefix="colibri-native",
        )

    @property
    def features(self) -> tuple[str, ...]:
        output = []
        if self.feature_mask & FEATURE_AVX2:
            output.append("avx2")
        if self.feature_mask & FEATURE_AVX512:
            output.append("avx512")
        return tuple(output) or ("scalar",)

    def q4_matvec(
        self, tensor: Q4BlockTensor, vector: list[float]
    ) -> list[float]:
        return self._q4_matvec_array(tensor, vector).tolist()

    def q4_swiglu(
        self, expert: Q4SwiGLUExpert, vector: object
    ) -> object:
        np = _numpy()
        input_vector = np.ascontiguousarray(vector, dtype=np.float32)
        gate_up = self._q4_matvec_array(expert.gate_up, input_vector)
        intermediate = expert.intermediate_size
        gate = gate_up[:intermediate]
        activated = gate / (np.float32(1.0) + np.exp(-gate))
        activated *= gate_up[intermediate:]
        return self._q4_matvec_array(expert.down, activated)

    def q4_moe(
        self,
        experts: list[Q4SwiGLUExpert],
        routing_weights: list[float],
        shared_expert: Q4SwiGLUExpert,
        shared_weight: float,
        vector: object,
        *,
        as_array: bool = False,
        out: object = None,
    ) -> object:
        if len(experts) != len(routing_weights):
            raise ValueError("expert and routing-weight counts must match")
        np = _numpy()
        input_vector = np.ascontiguousarray(vector, dtype=np.float32)
        all_experts = [*experts, shared_expert]
        weights = np.asarray(
            [*routing_weights, shared_weight], dtype=np.float32
        )
        if self._fused_moe:
            output = self._fused_q4_moe(
                all_experts, weights, input_vector, out=out
            )
        else:
            futures = [
                self._executor.submit(self.q4_swiglu, expert, input_vector)
                for expert in all_experts
            ]
            outputs = np.stack([future.result() for future in futures])
            output = np.einsum("e,eh->h", weights, outputs, out=out)
        return output if as_array else output.tolist()

    def _fused_q4_moe(
        self,
        all_experts: list[Q4SwiGLUExpert],
        weights: object,
        input_vector: object,
        *,
        out: object = None,
    ) -> object:
        np = _numpy()
        count = len(all_experts)
        hidden_size = all_experts[0].hidden_size
        intermediate_size = all_experts[0].intermediate_size
        u8 = ctypes.POINTER(ctypes.c_uint8)
        u16 = ctypes.POINTER(ctypes.c_uint16)
        gate_up_packed = (u8 * count)()
        gate_up_scales = (u16 * count)()
        down_packed = (u8 * count)()
        down_scales = (u16 * count)()
        for index, expert in enumerate(all_experts):
            pointers = getattr(expert, "_native_pointers", None)
            if pointers is None:
                gate_up = expert.gate_up
                down = expert.down
                gp = np.frombuffer(gate_up.packed, dtype=np.uint8)
                gs = np.frombuffer(gate_up.scales, dtype="<u2")
                dp = np.frombuffer(down.packed, dtype=np.uint8)
                ds = np.frombuffer(down.scales, dtype="<u2")
                pointers = (
                    gp.ctypes.data_as(u8),
                    gs.ctypes.data_as(u16),
                    dp.ctypes.data_as(u8),
                    ds.ctypes.data_as(u16),
                    (gp, gs, dp, ds),
                )
                expert._native_pointers = pointers
            gate_up_packed[index] = pointers[0]
            gate_up_scales[index] = pointers[1]
            down_packed[index] = pointers[2]
            down_scales[index] = pointers[3]
        weight_array = np.ascontiguousarray(weights, dtype=np.float32)
        output = (
            out
            if out is not None
            else np.empty(hidden_size, dtype=np.float32)
        )
        status = self.library.colibri_q4_moe(
            gate_up_packed,
            gate_up_scales,
            down_packed,
            down_scales,
            weight_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            input_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            count,
            hidden_size,
            intermediate_size,
        )
        if status != 0:
            raise RuntimeError(f"native Q4 MoE failed with status {status}")
        return output

    def _q4_matvec_array(self, tensor: Q4BlockTensor, vector: object) -> object:
        np = _numpy()
        if len(tensor.shape) != 2:
            raise ValueError(f"matvec requires a rank-2 tensor, got {tensor.shape}")
        rows, columns = tensor.shape
        input_vector = np.ascontiguousarray(vector, dtype=np.float32)
        if input_vector.size != columns:
            raise ValueError(
                f"expected vector width {columns}, got {input_vector.size}"
            )
        packed = np.frombuffer(tensor.packed, dtype=np.uint8)
        scales = np.frombuffer(tensor.scales, dtype="<u2")
        output = np.empty(rows, dtype=np.float32)
        status = self.library.colibri_q4_matvec(
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            scales.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            input_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            columns,
        )
        if status != 0:
            raise RuntimeError(f"native Q4 matvec failed with status {status}")
        return output


def _numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise NativeUnavailableError(
            "native tensor dispatch requires NumPy"
        ) from error
    return np


_backend: NativeBackend | None = None
_attempted = False


def active_native() -> NativeBackend | None:
    global _attempted, _backend
    if os.environ.get("COLIBRI_DISABLE_NATIVE") == "1":
        return None
    if not _attempted:
        _attempted = True
        try:
            _backend = NativeBackend()
        except NativeUnavailableError:
            _backend = None
    return _backend


def reset_native() -> None:
    global _attempted, _backend
    _attempted = False
    _backend = None


def _library_path() -> Path:
    override = os.environ.get("COLIBRI_NATIVE_LIBRARY")
    if override:
        return Path(override)
    directory = Path(__file__).with_name("_native")
    if os.name == "nt":
        return directory / "colibri_native.dll"
    if os.uname().sysname == "Darwin":
        return directory / "colibri_native.dylib"
    return directory / "colibri_native.so"
