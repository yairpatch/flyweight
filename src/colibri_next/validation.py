from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .causal_lm import QwenForCausalLM


@dataclass(frozen=True, slots=True)
class LogitComparison:
    position: int
    input_token: int
    colibri_greedy_token: int
    reference_greedy_token: int
    greedy_match: bool
    max_absolute_error: float
    mean_absolute_error: float
    cosine_similarity: float
    top_k_overlap: float


@dataclass(frozen=True, slots=True)
class ValidationReport:
    input_tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    comparisons: tuple[LogitComparison, ...]
    top_k: int

    @property
    def greedy_matches(self) -> int:
        return sum(item.greedy_match for item in self.comparisons)

    @property
    def all_greedy_tokens_match(self) -> bool:
        return self.greedy_matches == len(self.comparisons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": list(self.input_tokens),
            "generated_tokens": list(self.generated_tokens),
            "top_k": self.top_k,
            "greedy_matches": self.greedy_matches,
            "comparisons_count": len(self.comparisons),
            "all_greedy_tokens_match": self.all_greedy_tokens_match,
            "comparisons": [asdict(item) for item in self.comparisons],
        }


@dataclass(frozen=True, slots=True)
class ActivationComparison:
    stage: str
    max_absolute_error: float
    mean_absolute_error: float
    cosine_similarity: float


def compare_activation_vectors(
    colibri_values: Sequence[float],
    reference_values: Sequence[float],
    *,
    stage: str,
) -> ActivationComparison:
    if len(colibri_values) != len(reference_values):
        raise ValueError(
            f"activation widths differ at {stage}: "
            f"{len(colibri_values)} != {len(reference_values)}"
        )
    colibri = [float(value) for value in colibri_values]
    reference = [float(value) for value in reference_values]
    if not colibri:
        raise ValueError(f"activation vectors must not be empty at {stage}")
    errors = [abs(left - right) for left, right in zip(colibri, reference)]
    dot = sum(left * right for left, right in zip(colibri, reference))
    left_norm = math.sqrt(sum(value * value for value in colibri))
    right_norm = math.sqrt(sum(value * value for value in reference))
    cosine = dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
    return ActivationComparison(
        stage=stage,
        max_absolute_error=max(errors),
        mean_absolute_error=sum(errors) / len(errors),
        cosine_similarity=cosine,
    )


def compare_logit_vectors(
    colibri_logits: Sequence[float],
    reference_logits: Sequence[float],
    *,
    position: int,
    input_token: int,
    top_k: int = 10,
) -> LogitComparison:
    if len(colibri_logits) != len(reference_logits):
        raise ValueError(
            "logit vocabulary sizes differ: "
            f"{len(colibri_logits)} != {len(reference_logits)}"
        )
    if len(colibri_logits) == 0:
        raise ValueError("logit vectors must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    colibri = [float(value) for value in colibri_logits]
    reference = [float(value) for value in reference_logits]
    errors = [abs(left - right) for left, right in zip(colibri, reference)]
    dot = sum(left * right for left, right in zip(colibri, reference))
    left_norm = math.sqrt(sum(value * value for value in colibri))
    right_norm = math.sqrt(sum(value * value for value in reference))
    cosine = dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
    count = min(top_k, len(colibri))
    colibri_top = set(sorted(range(len(colibri)), key=colibri.__getitem__, reverse=True)[:count])
    reference_top = set(
        sorted(range(len(reference)), key=reference.__getitem__, reverse=True)[:count]
    )
    colibri_greedy = max(range(len(colibri)), key=colibri.__getitem__)
    reference_greedy = max(range(len(reference)), key=reference.__getitem__)
    return LogitComparison(
        position=position,
        input_token=input_token,
        colibri_greedy_token=colibri_greedy,
        reference_greedy_token=reference_greedy,
        greedy_match=colibri_greedy == reference_greedy,
        max_absolute_error=max(errors),
        mean_absolute_error=sum(errors) / len(errors),
        cosine_similarity=cosine,
        top_k_overlap=len(colibri_top & reference_top) / count,
    )


class TransformersReference:
    """Lazy Hugging Face model wrapper used only by the validation command."""

    def __init__(
        self,
        source: Path | str,
        *,
        device: str = "cpu",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        offload_dir: Path | str | None = None,
        max_gpu_memory_mib: int | None = None,
        max_cpu_memory_mib: int | None = None,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as error:
            raise RuntimeError(
                "Transformers validation requires the 'validation' extra: "
                "pip install -e '.[validation]'"
            ) from error

        torch_dtype: Any = dtype
        if dtype != "auto":
            try:
                torch_dtype = getattr(torch, dtype)
            except AttributeError as error:
                raise ValueError(f"unsupported torch dtype: {dtype}") from error
        self._torch = torch
        load_options: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if device == "auto":
            load_options["device_map"] = "auto"
            max_memory: dict[Any, str] = {}
            if max_gpu_memory_mib is not None:
                max_memory[0] = f"{max_gpu_memory_mib}MiB"
            if max_cpu_memory_mib is not None:
                max_memory["cpu"] = f"{max_cpu_memory_mib}MiB"
            if max_memory:
                load_options["max_memory"] = max_memory
            if offload_dir is not None:
                directory = Path(offload_dir)
                directory.mkdir(parents=True, exist_ok=True)
                load_options["offload_folder"] = str(directory)
                load_options["offload_state_dict"] = True
            self.model = AutoModelForCausalLM.from_pretrained(
                str(source), **load_options
            )
            self.device = self.model.get_input_embeddings().weight.device
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(source), **load_options
            ).to(device)
            self.device = device
        self.model.eval()

    def logits(self, token_ids: Sequence[int]) -> list[float]:
        with self._torch.inference_mode():
            inputs = self._torch.tensor([list(token_ids)], device=self.device)
            logits = self.model(input_ids=inputs).logits[0, -1]
        return logits.float().cpu().tolist()

    def hidden_states(self, token_ids: Sequence[int]) -> list[list[float]]:
        with self._torch.inference_mode():
            inputs = self._torch.tensor([list(token_ids)], device=self.device)
            outputs = self.model(
                input_ids=inputs,
                output_hidden_states=True,
                use_cache=False,
            )
        if outputs.hidden_states is None:
            raise RuntimeError("Transformers did not return hidden states")
        return [
            hidden[0, -1].float().cpu().tolist()
            for hidden in outputs.hidden_states
        ]

    def layer_components(
        self, token_ids: Sequence[int], layer_index: int
    ) -> dict[str, list[float]]:
        base_model = self.model.model
        language_model = getattr(base_model, "language_model", base_model)
        layer = language_model.layers[layer_index]
        captured: dict[str, list[float]] = {}

        def save(name: str, *, use_input: bool = False):
            def hook(module: Any, inputs: Any, output: Any) -> None:
                value = inputs[0] if use_input else output
                if isinstance(value, tuple):
                    value = value[0]
                captured[name] = value[0, -1].float().cpu().tolist()

            return hook

        mixer = layer.self_attn if layer.block_type == "full_attention" else layer.linear_attn
        handles = [
            layer.register_forward_hook(save("input", use_input=True)),
            layer.input_layernorm.register_forward_hook(save("input-norm")),
            mixer.register_forward_hook(save("mixer")),
            layer.post_attention_layernorm.register_forward_hook(
                save("mixer-residual", use_input=True)
            ),
            layer.post_attention_layernorm.register_forward_hook(save("post-norm")),
            layer.mlp.register_forward_hook(save("mlp")),
            layer.register_forward_hook(save("output")),
        ]
        try:
            with self._torch.inference_mode():
                inputs = self._torch.tensor([list(token_ids)], device=self.device)
                self.model(input_ids=inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return captured


def diagnose_hidden_states(
    model: QwenForCausalLM,
    reference: TransformersReference,
    token_id: int,
) -> list[ActivationComparison]:
    """Compare one-token decoder activations without introducing cache drift."""
    state = model.new_state()
    embedding = model.model_io.embed(token_id)
    colibri_states: list[list[float]] = [embedding]

    if model.device_decode_available:
        accelerator = model._device_accelerator()
        hidden = accelerator.device_vector(embedding)
        for layer, layer_state in zip(
            model.decoder.layers, state.decoder_state.layer_states
        ):
            hidden = layer.forward_device(hidden, layer_state, accelerator)
            colibri_states.append(hidden.get().tolist())
    else:
        hidden = embedding
        for layer, layer_state in zip(
            model.decoder.layers, state.decoder_state.layer_states
        ):
            result = layer.forward(hidden, layer_state)
            hidden = result.output
            colibri_states.append(hidden)

    normalized = model.model_io.normalize(colibri_states[-1])
    reference_states = reference.hidden_states([token_id])
    if len(reference_states) == len(colibri_states):
        comparable = colibri_states[:-1] + [normalized]
        stages = ["embedding"] + [
            f"layer-{index:03d}" for index in range(len(colibri_states) - 2)
        ] + ["final-norm"]
    elif len(reference_states) == len(colibri_states) + 1:
        comparable = colibri_states + [normalized]
        stages = ["embedding"] + [
            f"layer-{index:03d}" for index in range(len(colibri_states) - 1)
        ] + ["final-norm"]
    else:
        raise ValueError(
            "unexpected Transformers hidden-state count: "
            f"{len(reference_states)} for {len(model.decoder.layers)} layers"
        )
    return [
        compare_activation_vectors(colibri, reference_values, stage=stage)
        for stage, colibri, reference_values in zip(
            stages, comparable, reference_states
        )
    ]


def diagnose_layer_components(
    model: QwenForCausalLM,
    reference: TransformersReference,
    token_id: int,
    layer_index: int,
) -> list[ActivationComparison]:
    if not 0 <= layer_index < len(model.decoder.layers):
        raise ValueError(f"layer index out of range: {layer_index}")
    state = model.new_state()
    embedding = model.model_io.embed(token_id)
    layer = model.decoder.layers[layer_index]

    if model.device_decode_available:
        accelerator = model._device_accelerator()
        hidden = accelerator.device_vector(embedding)
        for prior, prior_state in zip(
            model.decoder.layers[:layer_index],
            state.decoder_state.layer_states[:layer_index],
        ):
            hidden = prior.forward_device(hidden, prior_state, accelerator)
        layer_state = state.decoder_state.layer_states[layer_index]
        mixer_layer = layer.token_mixer
        components: dict[str, Any] = {"input": hidden}
        components["input-norm"] = accelerator.rms_norm_device(
            hidden,
            mixer_layer._input_norm_weights,
            mixer_layer.rms_norm_eps,
        )
        if layer.layer_type != "full_attention":
            raise ValueError("component diagnostic currently requires a full-attention layer")
        mixer, _ = accelerator.full_attention(
            mixer_layer,
            hidden,
            0,
            layer_state.token_mixer_state,
            residual=False,
            return_attention_weights=False,
            return_device=True,
        )
        components["mixer"] = mixer
        mixed = hidden + mixer
        components["mixer-residual"] = mixed
        components["post-norm"] = accelerator.rms_norm_device(
            mixed,
            layer.moe._post_attention_norm_weights,
            layer.moe.rms_norm_eps,
        )
        output = accelerator.moe_residual_device(layer.moe, mixed, layer_state)
        components["mlp"] = output - mixed
        components["output"] = output
        colibri = {
            name: value.get().tolist() for name, value in components.items()
        }
    else:
        raise ValueError("component diagnostic currently requires CUDA")

    reference_components = reference.layer_components([token_id], layer_index)
    return [
        compare_activation_vectors(
            colibri[name], reference_components[name], stage=name
        )
        for name in (
            "input",
            "input-norm",
            "mixer",
            "mixer-residual",
            "post-norm",
            "mlp",
            "output",
        )
    ]


def validate_against_reference(
    model: QwenForCausalLM,
    reference: Any,
    token_ids: Sequence[int],
    *,
    generate_tokens: int = 0,
    top_k: int = 10,
) -> ValidationReport:
    if not token_ids:
        raise ValueError("token_ids must not be empty")
    if generate_tokens < 0:
        raise ValueError("generate_tokens must not be negative")

    context = [int(token) for token in token_ids]
    state = model.new_state()
    device_decode = bool(getattr(model, "device_decode_available", False))
    result = (
        model.prefill_device(context, state)
        if device_decode
        else model.prefill(context, state)
    )
    comparisons: list[LogitComparison] = []
    generated: list[int] = []

    for step in range(generate_tokens + 1):
        reference_logits = reference.logits(context)
        comparisons.append(
            compare_logit_vectors(
                result.logits,
                reference_logits,
                position=len(context) - 1,
                input_token=context[-1],
                top_k=top_k,
            )
        )
        if step == generate_tokens:
            break
        next_token = comparisons[-1].reference_greedy_token
        generated.append(next_token)
        context.append(next_token)
        result = (
            model.forward_token_device(next_token, state)
            if device_decode
            else model.forward_token(next_token, state)
        )

    return ValidationReport(
        input_tokens=tuple(int(token) for token in token_ids),
        generated_tokens=tuple(generated),
        comparisons=tuple(comparisons),
        top_k=top_k,
    )
