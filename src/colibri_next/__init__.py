"""Hardware-adaptive expert residency and execution primitives."""

from .attention import AttentionKVCache, AttentionResult, QwenFullAttentionLayer
from .attention_converter import QwenAttentionConverter
from .bf16 import BF16Tensor
from .causal_lm import CausalLMResult, CausalLMState, QwenForCausalLM
from .cache import LayeredExpertCache
from .converter import QwenCheckpointConverter, QwenSafetensorCheckpoint
from .cuda import CudaAccelerator, active_cuda, configure_cuda, disable_cuda
from .decoder import (
    DecoderLayerResult,
    DecoderLayerState,
    DecoderResult,
    DecoderState,
    QwenDecoderLayer,
    QwenDecoderStack,
)
from .float_tensor import FloatTensor
from .gated_delta import GatedDeltaResult, GatedDeltaState, QwenGatedDeltaLayer
from .gated_delta_converter import QwenGatedDeltaConverter
from .generation import GenerationResult, GenerationStep, TextGenerator
from .hardware import HardwareTopology, probe_hardware
from .kernels import Q4SwiGLUExpert
from .model_io import QwenModelIO
from .model_io_converter import QwenModelIOConverter
from .models import MoEModelSpec, model_spec
from .moe import MoEResult, QwenMoELayer
from .moe_converter import QwenMoELayerConverter
from .planner import PlacementPlan, PlacementPlanner
from .predictor import TransitionPredictor
from .q4 import Q4BlockTensor
from .residency import ResidencyManager
from .runtime import ToyMoERuntime
from .sampling import LogitsSampler, SamplingConfig
from .server import APIError, InferenceService, serve
from .safetensors import SafeTensorFile
from .storage import ExpertStore
from .tokenizer import HuggingFaceTokenizer
from .tokenizer_converter import TokenizerAssetsConverter
from .tensor_container import ColiTensorFile

__all__ = [
    "AttentionKVCache",
    "AttentionResult",
    "APIError",
    "BF16Tensor",
    "CausalLMResult",
    "CausalLMState",
    "ColiTensorFile",
    "CudaAccelerator",
    "DecoderLayerResult",
    "DecoderLayerState",
    "DecoderResult",
    "DecoderState",
    "ExpertStore",
    "FloatTensor",
    "GatedDeltaResult",
    "GatedDeltaState",
    "GenerationResult",
    "GenerationStep",
    "HardwareTopology",
    "InferenceService",
    "HuggingFaceTokenizer",
    "LayeredExpertCache",
    "LogitsSampler",
    "MoEModelSpec",
    "MoEResult",
    "PlacementPlan",
    "PlacementPlanner",
    "Q4BlockTensor",
    "Q4SwiGLUExpert",
    "QwenAttentionConverter",
    "QwenCheckpointConverter",
    "QwenDecoderLayer",
    "QwenDecoderStack",
    "QwenFullAttentionLayer",
    "QwenGatedDeltaConverter",
    "QwenGatedDeltaLayer",
    "QwenForCausalLM",
    "QwenModelIO",
    "QwenModelIOConverter",
    "QwenMoELayer",
    "QwenMoELayerConverter",
    "QwenSafetensorCheckpoint",
    "ResidencyManager",
    "SafeTensorFile",
    "SamplingConfig",
    "TextGenerator",
    "TokenizerAssetsConverter",
    "ToyMoERuntime",
    "TransitionPredictor",
    "model_spec",
    "probe_hardware",
    "serve",
    "active_cuda",
    "configure_cuda",
    "disable_cuda",
]






