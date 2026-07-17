from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .attention import AttentionKVCache, QwenFullAttentionLayer
from .expert import ExpertKey
from .gated_delta import GatedDeltaState, QwenGatedDeltaLayer
from .moe import QwenMoELayer
from .predictor import TransitionPredictor


TokenMixerState = AttentionKVCache | GatedDeltaState
TokenMixer = QwenFullAttentionLayer | QwenGatedDeltaLayer


@dataclass(frozen=True, slots=True)
class DecoderLayerResult:
    layer: int
    layer_type: str
    output: list[float]
    selected_experts: tuple[int, ...]
    routing_weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DecoderResult:
    output: list[float]
    layer_results: tuple[DecoderLayerResult, ...]


@dataclass(slots=True)
class DecoderLayerState:
    layer: int
    layer_type: str
    token_mixer_state: TokenMixerState
    last_selected_experts: tuple[int, ...] = ()
    sequence_selected_experts: tuple[tuple[int, ...], ...] = ()

    @property
    def tokens(self) -> int:
        if isinstance(self.token_mixer_state, AttentionKVCache):
            return self.token_mixer_state.length
        return self.token_mixer_state.tokens

    def clear(self) -> None:
        self.token_mixer_state.clear()


class DecoderState:
    def __init__(self, layer_states: list[DecoderLayerState]):
        self.layer_states = layer_states
        self.route_predictor = TransitionPredictor()

    def snapshot(self) -> list[tuple]:
        """Capture enough state to roll back speculative tokens.

        Attention KV caches append monotonically, so their snapshot is just
        the length; DeltaNet conv/recurrent tensors mutate in place and are
        copied (device-side when present, host-side otherwise).
        """
        snapshots: list[tuple] = []
        for state in self.layer_states:
            mixer = state.token_mixer_state
            routes = (
                state.last_selected_experts,
                state.sequence_selected_experts,
            )
            if isinstance(mixer, AttentionKVCache):
                snapshots.append(("attention", mixer.tokens, routes))
            else:
                if mixer.cuda_conv_state is not None:
                    conv = mixer.cuda_conv_state.copy()
                    recurrent = mixer.cuda_recurrent_state.copy()
                    device = True
                else:
                    conv = mixer.conv_state.copy()
                    recurrent = mixer.recurrent_state.copy()
                    device = False
                snapshots.append(
                    ("delta", mixer.tokens, routes, device, conv, recurrent)
                )
        return snapshots

    def restore(self, snapshots: list[tuple]) -> None:
        if len(snapshots) != len(self.layer_states):
            raise ValueError("snapshot layer count does not match state")
        for state, snapshot in zip(self.layer_states, snapshots):
            mixer = state.token_mixer_state
            kind, tokens, routes = snapshot[0], snapshot[1], snapshot[2]
            if kind == "attention":
                if not isinstance(mixer, AttentionKVCache):
                    raise ValueError("snapshot kind does not match state")
                mixer.tokens = tokens
                for head in mixer.keys:
                    del head[tokens:]
                for head in mixer.values:
                    del head[tokens:]
            else:
                if isinstance(mixer, AttentionKVCache):
                    raise ValueError("snapshot kind does not match state")
                device, conv, recurrent = snapshot[3], snapshot[4], snapshot[5]
                mixer.tokens = tokens
                if device:
                    mixer.cuda_conv_state[...] = conv
                    mixer.cuda_recurrent_state[...] = recurrent
                else:
                    mixer.conv_state[...] = conv
                    mixer.recurrent_state[...] = recurrent
                    mixer.cuda_conv_state = None
                    mixer.cuda_recurrent_state = None
            state.last_selected_experts = routes[0]
            state.sequence_selected_experts = routes[1]

    @property
    def tokens(self) -> int:
        if not self.layer_states:
            return 0
        positions = {state.tokens for state in self.layer_states}
        if len(positions) != 1:
            raise ValueError("decoder layer states have inconsistent token positions")
        return positions.pop()

    def clear(self) -> None:
        for state in self.layer_states:
            state.clear()
            state.last_selected_experts = ()
            state.sequence_selected_experts = ()
        self.route_predictor = TransitionPredictor()


class QwenDecoderLayer:
    """One complete Qwen decoder layer with token mixer and sparse MoE block."""

    def __init__(
        self,
        layer: int,
        layer_type: str,
        token_mixer: TokenMixer,
        moe: QwenMoELayer,
    ):
        if layer_type not in {"full_attention", "linear_attention"}:
            raise ValueError(f"unsupported decoder layer type: {layer_type}")
        self.layer = layer
        self.layer_type = layer_type
        self.token_mixer = token_mixer
        self.moe = moe
        self.hidden_size = moe.hidden_size
        self._validate()

    @classmethod
    def from_model_directory(
        cls, root: Path | str, layer: int, layer_type: str
    ) -> "QwenDecoderLayer":
        root_path = Path(root)
        if layer_type == "full_attention":
            token_mixer: TokenMixer = QwenFullAttentionLayer.from_model_directory(
                root_path, layer
            )
        elif layer_type == "linear_attention":
            token_mixer = QwenGatedDeltaLayer.from_model_directory(root_path, layer)
        else:
            raise ValueError(f"unsupported decoder layer type: {layer_type}")
        return cls(
            layer,
            layer_type,
            token_mixer,
            QwenMoELayer.from_model_directory(root_path, layer),
        )

    def new_state(self) -> DecoderLayerState:
        return DecoderLayerState(
            layer=self.layer,
            layer_type=self.layer_type,
            token_mixer_state=self.token_mixer.new_cache()
            if isinstance(self.token_mixer, QwenFullAttentionLayer)
            else self.token_mixer.new_state(),
        )

    def forward(
        self, hidden: list[float], state: DecoderLayerState
    ) -> DecoderLayerResult:
        if len(hidden) != self.hidden_size:
            raise ValueError(
                f"expected hidden width {self.hidden_size}, got {len(hidden)}"
            )
        self._validate_state(state)
        if isinstance(self.token_mixer, QwenFullAttentionLayer):
            assert isinstance(state.token_mixer_state, AttentionKVCache)
            mixed = self.token_mixer.forward_residual(
                hidden,
                state.token_mixer_state.length,
                state.token_mixer_state,
                return_attention_weights=False,
            ).output
        else:
            assert isinstance(state.token_mixer_state, GatedDeltaState)
            mixed = self.token_mixer.forward_residual(
                hidden, state.token_mixer_state
            ).output
        moe_result = self.moe.forward_residual(mixed)
        return DecoderLayerResult(
            layer=self.layer,
            layer_type=self.layer_type,
            output=moe_result.output,
            selected_experts=moe_result.selected_experts,
            routing_weights=moe_result.routing_weights,
        )

    def forward_device(
        self, hidden: Any, state: DecoderLayerState, accelerator: Any
    ) -> Any:
        if hidden.ndim != 1 or hidden.size != self.hidden_size:
            raise ValueError(
                f"expected hidden width {self.hidden_size}, got {hidden.size}"
            )
        self._validate_state(state)
        if isinstance(self.token_mixer, QwenFullAttentionLayer):
            assert isinstance(state.token_mixer_state, AttentionKVCache)
            self.token_mixer._validate_cache(
                state.token_mixer_state, state.token_mixer_state.length
            )
            mixed, _ = accelerator.full_attention(
                self.token_mixer,
                hidden,
                state.token_mixer_state.length,
                state.token_mixer_state,
                residual=True,
                return_attention_weights=False,
                return_device=True,
            )
        else:
            assert isinstance(state.token_mixer_state, GatedDeltaState)
            self.token_mixer._validate_state(state.token_mixer_state)
            mixed = accelerator.gated_delta(
                self.token_mixer,
                hidden,
                state.token_mixer_state,
                return_device=True,
            )
            state.token_mixer_state.tokens += 1
            mixed += hidden
        return accelerator.moe_residual_device(self.moe, mixed, state)

    def forward_sequence_device(
        self, hidden: Any, state: DecoderLayerState, accelerator: Any
    ) -> Any:
        if hidden.ndim != 2 or hidden.shape[1] != self.hidden_size:
            raise ValueError("decoder sequence has the wrong hidden width")
        self._validate_state(state)
        if isinstance(self.token_mixer, QwenFullAttentionLayer):
            assert isinstance(state.token_mixer_state, AttentionKVCache)
            mixed = accelerator.full_attention_sequence(
                self.token_mixer,
                hidden,
                state.token_mixer_state.length,
                state.token_mixer_state,
            )
        else:
            assert isinstance(state.token_mixer_state, GatedDeltaState)
            self.token_mixer._validate_state(state.token_mixer_state)
            mixed = accelerator.gated_delta_sequence(
                self.token_mixer, hidden, state.token_mixer_state
            )
        return accelerator.moe_sequence_device(self.moe, mixed, state)

    def _validate(self) -> None:
        expected_mixer = (
            QwenFullAttentionLayer
            if self.layer_type == "full_attention"
            else QwenGatedDeltaLayer
        )
        if not isinstance(self.token_mixer, expected_mixer):
            raise ValueError("token mixer implementation does not match layer type")
        if self.token_mixer.layer != self.layer or self.moe.layer != self.layer:
            raise ValueError("decoder component layer indices do not match")
        if self.token_mixer.hidden_size != self.hidden_size:
            raise ValueError("token mixer and MoE hidden sizes do not match")

    def _validate_state(self, state: DecoderLayerState) -> None:
        if state.layer != self.layer or state.layer_type != self.layer_type:
            raise ValueError("decoder state does not match layer")
        expected_state = (
            AttentionKVCache
            if self.layer_type == "full_attention"
            else GatedDeltaState
        )
        if not isinstance(state.token_mixer_state, expected_state):
            raise ValueError("decoder token-mixer state has the wrong type")


class QwenDecoderStack:
    """Mixed full-attention/Gated-DeltaNet decoder stack for token vectors."""

    def __init__(self, layers: list[QwenDecoderLayer]):
        if not layers:
            raise ValueError("decoder stack must contain at least one layer")
        self.layers = layers
        self.hidden_size = layers[0].hidden_size
        self._validate()

    @classmethod
    def from_model_directory(cls, root: Path | str) -> "QwenDecoderStack":
        root_path = Path(root)
        manifest_path = root_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = manifest["model"]
        layer_count = int(model["layers"])
        layer_types = model.get("layer_types")
        if layer_types is None:
            layer_types = cls._layer_types_from_storage(manifest, layer_count)
        if len(layer_types) != layer_count:
            raise ValueError("manifest layer_types length does not match layer count")
        return cls(
            [
                QwenDecoderLayer.from_model_directory(
                    root_path, layer, str(layer_types[layer])
                )
                for layer in range(layer_count)
            ]
        )

    @property
    def cpu_moe_layers(self) -> int:
        return sum(layer.moe.expert_device == "cpu" for layer in self.layers)

    def configure_moe_placement(self, cpu_layers: int) -> None:
        if cpu_layers < 0 or cpu_layers > len(self.layers):
            raise ValueError(
                f"cpu_layers must be between 0 and {len(self.layers)}"
            )
        for index, layer in enumerate(self.layers):
            layer.moe.set_expert_device(
                "cpu" if index < cpu_layers else "cuda"
            )

    @property
    def estimated_expert_storage_bytes(self) -> int:
        return sum(
            layer.moe.estimated_expert_storage_bytes for layer in self.layers
        )

    def preload_experts(
        self, progress: Callable[[int, int], None] | None = None
    ) -> int:
        total = sum(layer.moe.expert_count for layer in self.layers)
        completed = 0
        for layer in self.layers:
            completed += layer.moe.preload_experts()
            if progress is not None:
                progress(completed, total)
        return completed

    def new_state(self) -> DecoderState:
        return DecoderState([layer.new_state() for layer in self.layers])

    def forward_token(
        self, hidden: list[float], state: DecoderState
    ) -> DecoderResult:
        if len(hidden) != self.hidden_size:
            raise ValueError(
                f"expected hidden width {self.hidden_size}, got {len(hidden)}"
            )
        if len(state.layer_states) != len(self.layers):
            raise ValueError("decoder state layer count does not match stack")
        state.tokens
        output = hidden
        results: list[DecoderLayerResult] = []
        for layer, layer_state in zip(self.layers, state.layer_states):
            result = layer.forward(output, layer_state)
            results.append(result)
            output = result.output
        state.tokens
        return DecoderResult(output=output, layer_results=tuple(results))

    def forward_token_device(
        self, hidden: Any, state: DecoderState, accelerator: Any
    ) -> Any:
        if hidden.ndim != 1 or hidden.size != self.hidden_size:
            raise ValueError(
                f"expected hidden width {self.hidden_size}, got {hidden.size}"
            )
        if len(state.layer_states) != len(self.layers):
            raise ValueError("decoder state layer count does not match stack")
        state.tokens
        output = hidden
        previous_route: list[ExpertKey] | None = None
        last_index = len(self.layers) - 1
        for index, (layer, layer_state) in enumerate(
            zip(self.layers, state.layer_states)
        ):
            on_cuda = layer.moe.expert_device == "cuda"
            if on_cuda:
                predicted = layer_state.last_selected_experts
                if previous_route is not None:
                    prediction = state.route_predictor.predict(
                        previous_route,
                        layer.layer,
                        min(
                            layer.moe.top_k,
                            accelerator.expert_prefetch_budget,
                        ),
                    )
                    if prediction:
                        predicted = tuple(key.expert for key in prediction)
                accelerator.prefetch_moe(layer.moe, predicted)
            output = layer.forward_device(output, layer_state, accelerator)
            # Route bookkeeping only feeds GPU expert prefetch, so skip it
            # unless this layer or the next one keeps its experts on CUDA.
            if on_cuda or (
                index < last_index
                and self.layers[index + 1].moe.expert_device == "cuda"
            ):
                current_route = [
                    ExpertKey(layer.layer, expert_id)
                    for expert_id in layer_state.last_selected_experts
                ]
                if previous_route is not None:
                    state.route_predictor.observe(
                        previous_route, current_route
                    )
                previous_route = current_route
            else:
                previous_route = None
        state.tokens
        return output

    def prefill_device(
        self, hidden: Any, state: DecoderState, accelerator: Any
    ) -> Any:
        if hidden.ndim != 2 or hidden.shape[1] != self.hidden_size:
            raise ValueError("decoder sequence has the wrong hidden width")
        if len(state.layer_states) != len(self.layers):
            raise ValueError("decoder state layer count does not match stack")
        output = hidden
        for layer, layer_state in zip(self.layers, state.layer_states):
            output = layer.forward_sequence_device(
                output, layer_state, accelerator
            )
        state.tokens
        route_sequences = [
            layer_state.sequence_selected_experts
            for layer_state in state.layer_states
        ]
        tokens = int(hidden.shape[0])
        if all(len(routes) == tokens for routes in route_sequences):
            for token in range(tokens):
                for layer in range(1, len(route_sequences)):
                    current = [
                        ExpertKey(layer - 1, expert_id)
                        for expert_id in route_sequences[layer - 1][token]
                    ]
                    following = [
                        ExpertKey(layer, expert_id)
                        for expert_id in route_sequences[layer][token]
                    ]
                    state.route_predictor.observe(current, following)
        accelerator.batched_prefill_tokens += tokens
        return output

    def _validate(self) -> None:
        for index, layer in enumerate(self.layers):
            if layer.layer != index:
                raise ValueError("decoder layers must be contiguous and ordered")
            if layer.hidden_size != self.hidden_size:
                raise ValueError("decoder layers have inconsistent hidden sizes")

    @staticmethod
    def _layer_types_from_storage(
        manifest: dict[str, object], layer_count: int
    ) -> list[str]:
        attention_storage = manifest.get("attention_layer_storage", {})
        linear_storage = manifest.get("linear_layer_storage", {})
        if not isinstance(attention_storage, dict) or not isinstance(
            linear_storage, dict
        ):
            raise ValueError("manifest token-mixer storage is invalid")
        attention_layers = {int(layer) for layer in attention_storage.get("layers", [])}
        linear_layers = {int(layer) for layer in linear_storage.get("layers", [])}
        overlap = attention_layers & linear_layers
        if overlap:
            raise ValueError(f"layers have multiple token mixers: {sorted(overlap)}")
        missing = set(range(layer_count)) - attention_layers - linear_layers
        if missing:
            raise ValueError(f"layers have no converted token mixer: {sorted(missing)}")
        return [
            "full_attention" if layer in attention_layers else "linear_attention"
            for layer in range(layer_count)
        ]
