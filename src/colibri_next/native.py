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
        vector: list[float],
    ) -> list[float]:
        if len(experts) != len(routing_weights):
            raise ValueError("expert and routing-weight counts must match")
        np = _numpy()
        input_vector = np.ascontiguousarray(vector, dtype=np.float32)
        all_experts = [*experts, shared_expert]
        futures = [
            self._executor.submit(self.q4_swiglu, expert, input_vector)
            for expert in all_experts
        ]
        outputs = np.stack([future.result() for future in futures])
        weights = np.asarray(
            [*routing_weights, shared_weight], dtype=np.float32
        )
        return np.einsum("e,eh->h", weights, outputs).tolist()

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
