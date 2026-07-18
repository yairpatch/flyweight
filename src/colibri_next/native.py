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


class DeltaLayerStruct(ctypes.Structure):
    """Mirror of ColibriDeltaLayer in colibri_native.h."""

    _fields_ = [
        ("qz_packed", ctypes.c_uint64),
        ("qz_scales", ctypes.c_uint64),
        ("ba_weights", ctypes.c_uint64),
        ("out_proj_packed", ctypes.c_uint64),
        ("out_proj_scales", ctypes.c_uint64),
        ("input_norm", ctypes.c_uint64),
        ("conv_weights", ctypes.c_uint64),
        ("a_log", ctypes.c_uint64),
        ("dt_bias", ctypes.c_uint64),
        ("delta_norm", ctypes.c_uint64),
        ("conv_state", ctypes.c_uint64),
        ("recurrent_state", ctypes.c_uint64),
        ("router_gate", ctypes.c_uint64),
        ("post_attention_norm", ctypes.c_uint64),
        ("expert_gate_packed", ctypes.c_void_p),
        ("expert_gate_scales", ctypes.c_void_p),
        ("expert_down_packed", ctypes.c_void_p),
        ("expert_down_scales", ctypes.c_void_p),
        ("shared_gate_up_packed", ctypes.c_void_p),
        ("shared_gate_up_scales", ctypes.c_void_p),
        ("shared_down_packed", ctypes.c_void_p),
        ("shared_down_scales", ctypes.c_void_p),
    ]


class DeltaParamsStruct(ctypes.Structure):
    """Mirror of ColibriDeltaParams in colibri_native.h."""

    _fields_ = [
        ("hidden_size", ctypes.c_int32),
        ("conv_dim", ctypes.c_int32),
        ("conv_kernel", ctypes.c_int32),
        ("value_dim", ctypes.c_int32),
        ("num_key_heads", ctypes.c_int32),
        ("num_value_heads", ctypes.c_int32),
        ("key_head_dim", ctypes.c_int32),
        ("value_head_dim", ctypes.c_int32),
        ("qz_rows", ctypes.c_int32),
        ("ba_rows", ctypes.c_int32),
        ("num_experts", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("moe_intermediate", ctypes.c_int32),
        ("rms_norm_eps", ctypes.c_float),
        ("hidden", ctypes.c_uint64),
        ("normalized", ctypes.c_uint64),
        ("projected", ctypes.c_uint64),
        ("gates", ctypes.c_uint64),
        ("convolved", ctypes.c_uint64),
        ("cores", ctypes.c_uint64),
        ("mixed", ctypes.c_uint64),
        ("moe_normalized", ctypes.c_uint64),
        ("router_logits", ctypes.c_uint64),
        ("hidden_host", ctypes.c_void_p),
        ("normalized_host", ctypes.c_void_p),
        ("moe_host", ctypes.c_void_p),
        ("logits_host", ctypes.c_void_p),
    ]


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
        self._grouped_moe = self.version >= 3
        if self._grouped_moe:
            library.colibri_q4_moe_grouped.argtypes = [
                ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
            ]
            library.colibri_q4_moe_grouped.restype = ctypes.c_int
        self._gpu_driver = hasattr(library, "colibri_delta_moe_segment")
        if self._gpu_driver:
            library.colibri_gpu_available.restype = ctypes.c_int
            library.colibri_gpu_init.argtypes = [ctypes.c_int32]
            library.colibri_gpu_init.restype = ctypes.c_int
            library.colibri_gpu_compile.argtypes = [
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_char_p,
                ctypes.c_int32,
            ]
            library.colibri_gpu_compile.restype = ctypes.c_int
            library.colibri_delta_moe_segment.argtypes = [
                ctypes.POINTER(DeltaParamsStruct),
                ctypes.POINTER(DeltaLayerStruct),
                ctypes.c_int32,
            ]
            library.colibri_delta_moe_segment.restype = ctypes.c_int
        self._gpu_compiled = False

    def gpu_prepare(
        self, kernel_source: str, device: int, include_dirs: list[str]
    ) -> bool:
        """Initialize the CUDA driver and compile the kernel module once."""
        if self._gpu_compiled:
            return True
        if not self._gpu_driver or self.library.colibri_gpu_available() != 1:
            return False
        if self.library.colibri_gpu_init(device) != 0:
            return False
        options = [f"-I{directory}".encode() for directory in include_dirs]
        option_array = (ctypes.c_char_p * len(options))(*options)
        log = ctypes.create_string_buffer(16384)
        status = self.library.colibri_gpu_compile(
            kernel_source.encode(),
            option_array,
            len(options),
            device,
            log,
            len(log),
        )
        if status != 0:
            raise RuntimeError(
                f"native kernel compile failed ({status}): "
                f"{log.value.decode(errors='replace')[:2000]}"
            )
        self._gpu_compiled = True
        return True

    def delta_moe_segment(
        self, params: DeltaParamsStruct, layers: object, count: int
    ) -> None:
        status = self.library.colibri_delta_moe_segment(
            ctypes.byref(params), layers, count
        )
        if status != 0:
            raise RuntimeError(
                f"native delta segment failed with status {status}"
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
        hidden_size = all_experts[0].hidden_size
        intermediate_size = all_experts[0].intermediate_size
        pointer_arrays = self._expert_pointer_arrays(all_experts)
        weight_array = np.ascontiguousarray(weights, dtype=np.float32)
        output = (
            out
            if out is not None
            else np.empty(hidden_size, dtype=np.float32)
        )
        status = self.library.colibri_q4_moe(
            *pointer_arrays,
            weight_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            input_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(all_experts),
            hidden_size,
            intermediate_size,
        )
        if status != 0:
            raise RuntimeError(f"native Q4 MoE failed with status {status}")
        return output

    def q4_moe_grouped(
        self,
        experts: list[Q4SwiGLUExpert],
        assignment_expert: object,
        assignment_token: object,
        assignment_weight: object,
        inputs: object,
    ) -> object:
        """Expert-major MoE over a token batch; one native call per layer.

        ``experts`` are the unique experts referenced by ``assignment_expert``
        (which must arrive sorted by expert so the kernel streams each
        expert's weights once). Returns the (tokens, hidden) weighted expert
        sums without the residual.
        """
        if not self._grouped_moe:
            raise RuntimeError("grouped MoE requires native library v3+")
        np = _numpy()
        hidden_size = experts[0].hidden_size
        intermediate_size = experts[0].intermediate_size
        pointer_arrays = self._expert_pointer_arrays(experts)
        expert_ids = np.ascontiguousarray(assignment_expert, dtype=np.int32)
        token_ids = np.ascontiguousarray(assignment_token, dtype=np.int32)
        weight_values = np.ascontiguousarray(
            assignment_weight, dtype=np.float32
        )
        input_matrix = np.ascontiguousarray(inputs, dtype=np.float32)
        tokens = int(input_matrix.shape[0])
        outputs = np.empty((tokens, hidden_size), dtype=np.float32)
        status = self.library.colibri_q4_moe_grouped(
            *pointer_arrays,
            expert_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            token_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            weight_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            input_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(expert_ids.size),
            tokens,
            hidden_size,
            intermediate_size,
        )
        if status != 0:
            raise RuntimeError(
                f"native grouped Q4 MoE failed with status {status}"
            )
        return outputs

    def _expert_pointer_arrays(
        self, experts: list[Q4SwiGLUExpert]
    ) -> tuple[object, object, object, object]:
        np = _numpy()
        count = len(experts)
        u8 = ctypes.POINTER(ctypes.c_uint8)
        u16 = ctypes.POINTER(ctypes.c_uint16)
        gate_up_packed = (u8 * count)()
        gate_up_scales = (u16 * count)()
        down_packed = (u8 * count)()
        down_scales = (u16 * count)()
        for index, expert in enumerate(experts):
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
        return gate_up_packed, gate_up_scales, down_packed, down_scales

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
