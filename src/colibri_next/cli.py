from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .attention import QwenFullAttentionLayer
from .attention_converter import QwenAttentionConverter
from .causal_lm import QwenForCausalLM
from .cache import LayeredExpertCache
from .converter import QwenCheckpointConverter, QwenSafetensorCheckpoint
from .cuda import active_cuda, configure_cuda, disable_cuda
from .decoder import QwenDecoderStack
from .gated_delta import QwenGatedDeltaLayer
from .gated_delta_converter import QwenGatedDeltaConverter
from .generation import TextGenerator
from .hardware import HardwareTopology, available_ram_bytes, probe_hardware
from .kernels import Q4SwiGLUExpert
from .model_io_converter import QwenModelIOConverter
from .models import model_spec
from .moe import QwenMoELayer
from .moe_converter import QwenMoELayerConverter
from .native import active_native
from .planner import PlacementPlanner
from .predictor import TransitionPredictor
from .sampling import SamplingConfig
from .server import InferenceService, serve as serve_http
from .residency import ResidencyManager
from .runtime import ToyMoERuntime
from .storage import ExpertStore
from .tokenizer import HuggingFaceTokenizer
from .tokenizer_converter import TokenizerAssetsConverter
from .validation import (
    TransformersReference,
    diagnose_hidden_states,
    diagnose_layer_components,
    validate_against_reference,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colibri-next",
        description="Hierarchical expert residency prototype for sparse MoE inference",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    inspect = subcommands.add_parser(
        "inspect-hardware", help="probe memory tiers and compute devices"
    )
    inspect.add_argument("--storage-path", type=Path, default=Path("."))
    inspect.add_argument("--save", type=Path)

    plan = subcommands.add_parser(
        "plan", help="create a hardware-adaptive model placement plan"
    )
    plan.add_argument("--model", default="qwen3.6-35b-a3b")
    plan.add_argument("--context", type=int, default=32_768)
    plan.add_argument("--expert-bits", type=int, choices=(4, 8, 16), default=4)
    plan.add_argument("--hardware", type=Path)
    plan.add_argument("--storage-path", type=Path, default=Path("."))
    plan.add_argument("--save", type=Path)

    inspect_model = subcommands.add_parser(
        "inspect-model", help="inspect a local sharded Qwen safetensors checkpoint"
    )
    inspect_model.add_argument("source", type=Path)

    convert = subcommands.add_parser(
        "convert-qwen", help="create a runtime manifest and optional expert files"
    )
    convert.add_argument("source", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--extract-experts", action="store_true")
    convert.add_argument("--overwrite", action="store_true")
    convert.add_argument("--quantization", choices=("q4", "bf16"), default="q4")

    benchmark = subcommands.add_parser(
        "benchmark-expert", help="execute and benchmark one converted Q4 expert"
    )
    benchmark.add_argument("path", type=Path)
    benchmark.add_argument("--iterations", type=int, default=5)

    convert_layers = subcommands.add_parser(
        "convert-moe-layers", help="convert routers, norms, and shared experts"
    )
    convert_layers.add_argument("source", type=Path)
    convert_layers.add_argument("output", type=Path)
    convert_layers.add_argument("--overwrite", action="store_true")

    benchmark_layer = subcommands.add_parser(
        "benchmark-moe-layer", help="execute one complete converted MoE block"
    )
    benchmark_layer.add_argument("root", type=Path)
    benchmark_layer.add_argument("--layer", type=int, default=0)
    benchmark_layer.add_argument("--iterations", type=int, default=3)
    benchmark_layer.add_argument("--residual", action="store_true")

    convert_attention = subcommands.add_parser(
        "convert-attention-layers", help="convert Qwen full-attention layers"
    )
    convert_attention.add_argument("source", type=Path)
    convert_attention.add_argument("output", type=Path)
    convert_attention.add_argument("--overwrite", action="store_true")
    convert_attention.add_argument("--quantization", choices=("bf16", "q4"), default="bf16")

    benchmark_attention = subcommands.add_parser(
        "benchmark-attention-layer", help="execute incremental full attention"
    )
    benchmark_attention.add_argument("root", type=Path)
    benchmark_attention.add_argument("--layer", type=int, default=3)
    benchmark_attention.add_argument("--tokens", type=int, default=8)

    convert_linear = subcommands.add_parser(
        "convert-linear-layers", help="convert Qwen Gated DeltaNet layers"
    )
    convert_linear.add_argument("source", type=Path)
    convert_linear.add_argument("output", type=Path)
    convert_linear.add_argument("--overwrite", action="store_true")
    convert_linear.add_argument("--quantization", choices=("bf16", "q4"), default="bf16")

    benchmark_linear = subcommands.add_parser(
        "benchmark-linear-layer", help="execute incremental Gated DeltaNet"
    )
    benchmark_linear.add_argument("root", type=Path)
    benchmark_linear.add_argument("--layer", type=int, default=0)
    benchmark_linear.add_argument("--tokens", type=int, default=8)

    benchmark_decoder = subcommands.add_parser(
        "benchmark-decoder", help="execute the complete converted decoder stack"
    )
    benchmark_decoder.add_argument("root", type=Path)
    benchmark_decoder.add_argument("--tokens", type=int, default=1)

    convert_model_io = subcommands.add_parser(
        "convert-model-io", help="convert embeddings, final norm, and LM head"
    )
    convert_model_io.add_argument("source", type=Path)
    convert_model_io.add_argument("output", type=Path)
    convert_model_io.add_argument("--overwrite", action="store_true")
    convert_model_io.add_argument("--quantization", choices=("bf16", "q4"), default="bf16")

    benchmark_logits = subcommands.add_parser(
        "benchmark-logits", help="execute token IDs through the complete model"
    )
    benchmark_logits.add_argument("root", type=Path)
    benchmark_logits.add_argument("--token-ids", default="0")
    benchmark_logits.add_argument("--rows-per-chunk", type=int, default=4096)
    _add_device_arguments(benchmark_logits)

    validate = subcommands.add_parser(
        "validate-transformers",
        help="compare converted-model logits with a Transformers checkpoint",
    )
    validate.add_argument("root", type=Path, help="converted Colibri model")
    validate.add_argument("source", type=Path, help="Hugging Face checkpoint")
    validate.add_argument("--token-ids", required=True)
    validate.add_argument("--generate-tokens", type=int, default=4)
    validate.add_argument("--top-k", type=int, default=10)
    validate.add_argument("--reference-device", default="cpu")
    validate.add_argument("--reference-dtype", default="auto")
    validate.add_argument("--reference-offload-dir", type=Path)
    validate.add_argument("--reference-gpu-memory-mib", type=int)
    validate.add_argument("--reference-cpu-memory-mib", type=int)
    validate.add_argument(
        "--layerwise",
        action="store_true",
        help="compare one-token hidden states instead of generation logits",
    )
    validate.add_argument(
        "--component-layer",
        type=int,
        help="compare internal stages of one full-attention layer",
    )
    validate.add_argument("--trust-remote-code", action="store_true")
    validate.add_argument("--rows-per-chunk", type=int, default=4096)
    _add_device_arguments(validate)

    convert_tokenizer = subcommands.add_parser(
        "convert-tokenizer", help="copy tokenizer and generation assets"
    )
    convert_tokenizer.add_argument("source", type=Path)
    convert_tokenizer.add_argument("output", type=Path)
    convert_tokenizer.add_argument("--overwrite", action="store_true")

    generate_text = subcommands.add_parser(
        "generate-text", help="generate decoded text from a prompt"
    )
    generate_text.add_argument("root", type=Path)
    generate_text.add_argument("--prompt", required=True)
    generate_text.add_argument("--system")
    generate_text.add_argument("--max-new-tokens", type=int, default=8)
    generate_text.add_argument("--temperature", type=float, default=0.0)
    generate_text.add_argument("--top-k", type=int, default=20)
    generate_text.add_argument("--top-p", type=float, default=0.95)
    generate_text.add_argument("--seed", type=int)
    generate_text.add_argument("--enable-thinking", action="store_true")
    generate_text.add_argument("--rows-per-chunk", type=int, default=4096)
    _add_device_arguments(generate_text)
    _add_expert_preload_argument(generate_text, default="none")
    _add_cpu_moe_argument(generate_text)

    serve_command = subcommands.add_parser(
        "serve", help="run a persistent OpenAI-compatible local server"
    )
    serve_command.add_argument("root", type=Path)
    serve_command.add_argument("--host", default="127.0.0.1")
    serve_command.add_argument("--port", type=int, default=8000)
    serve_command.add_argument("--model-name")
    serve_command.add_argument(
        "--api-key", default=os.environ.get("COLIBRI_API_KEY")
    )
    serve_command.add_argument("--cors-origin", default="*")
    serve_command.add_argument(
        "--context-window",
        type=int,
        default=4096,
        help="maximum combined prompt and generated tokens",
    )
    serve_command.add_argument(
        "--max-new-tokens",
        type=int,
        help="optional output ceiling below the context window",
    )
    serve_command.add_argument("--rows-per-chunk", type=int, default=4096)
    _add_device_arguments(serve_command)
    _add_expert_preload_argument(serve_command, default="auto")
    _add_cpu_moe_argument(serve_command)

    create = subcommands.add_parser("create-demo", help="create deterministic experts")
    create.add_argument("path", type=Path)
    create.add_argument("--layers", type=int, default=6)
    create.add_argument("--experts", type=int, default=12)
    create.add_argument("--width", type=int, default=16)

    run = subcommands.add_parser("run", help="run the toy MoE scheduler")
    run.add_argument("path", type=Path)
    run.add_argument("--tokens", type=int, default=32)
    run.add_argument("--cache-experts", type=int, default=4)
    run.add_argument("--top-k", type=int, default=2)
    run.add_argument("--prefetch", type=int, default=2)
    run.add_argument("--warmup", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect-hardware":
        topology = probe_hardware(args.storage_path)
        output = topology.to_json()
        if args.save:
            args.save.write_text(output, encoding="utf-8")
        print(output, end="")
        return 0

    if args.command == "plan":
        topology = (
            HardwareTopology.from_json_file(args.hardware)
            if args.hardware
            else probe_hardware(args.storage_path)
        )
        placement = PlacementPlanner(topology).plan(
            model_spec(args.model),
            context_length=args.context,
            expert_bits=args.expert_bits,
        )
        output = placement.to_json()
        if args.save:
            args.save.write_text(output, encoding="utf-8")
        print(output, end="")
        return 0 if placement.supported else 2

    if args.command == "inspect-model":
        print(json.dumps(QwenSafetensorCheckpoint(args.source).inspect(), indent=2))
        return 0

    if args.command == "convert-qwen":
        def progress(completed: int, total: int) -> None:
            if completed == total or completed % 64 == 0:
                print(
                    f"\rExtracted experts: {completed}/{total}",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )

        manifest = QwenCheckpointConverter(args.source).convert(
            args.output,
            extract_experts=args.extract_experts,
            quantization=args.quantization,
            overwrite=args.overwrite,
            progress=progress if args.extract_experts else None,
        )
        if args.extract_experts:
            print(file=sys.stderr)
        print(
            json.dumps(
                {
                    "manifest": str((args.output / "manifest.json").resolve()),
                    "architecture": manifest["architecture"],
                    "expert_storage": manifest["expert_storage"]["mode"],
                    "expected_expert_files": manifest["model"]["layers"]
                    * manifest["model"]["experts_per_layer"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "convert-moe-layers":
        def layer_progress(completed: int, total: int) -> None:
            print(
                f"\rConverted MoE layers: {completed}/{total}",
                end="",
                file=sys.stderr,
                flush=True,
            )

        storage = QwenMoELayerConverter(args.source).convert(
            args.output,
            overwrite=args.overwrite,
            progress=layer_progress,
        )
        print(file=sys.stderr)
        print(json.dumps(storage, indent=2))
        return 0

    if args.command == "benchmark-moe-layer":
        if args.iterations <= 0:
            raise ValueError("iterations must be positive")
        layer = QwenMoELayer.from_model_directory(args.root, args.layer)
        hidden = [math.sin(index * 0.17) for index in range(layer.hidden_size)]
        started = time.perf_counter()
        result = None
        for _ in range(args.iterations):
            result = (
                layer.forward_residual(hidden)
                if args.residual
                else layer.forward(hidden)
            )
        elapsed = time.perf_counter() - started
        assert result is not None
        print(
            json.dumps(
                {
                    "layer": args.layer,
                    "iterations": args.iterations,
                    "seconds": elapsed,
                    "layers_per_second": args.iterations / elapsed if elapsed else 0.0,
                    "selected_experts": result.selected_experts,
                    "routing_weights": result.routing_weights,
                    "output_checksum": round(sum(result.output), 6),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "convert-tokenizer":
        storage = TokenizerAssetsConverter(args.source).convert(
            args.output, overwrite=args.overwrite
        )
        print(json.dumps(storage, indent=2))
        return 0

    if args.command == "serve":
        _configure_execution(args)
        print(f"Loading model from {args.root}...", file=sys.stderr, flush=True)
        service = InferenceService.from_model_directory(
            args.root,
            model_name=args.model_name,
            rows_per_chunk=args.rows_per_chunk,
            max_new_tokens=args.max_new_tokens or args.context_window,
            context_window=args.context_window,
            api_key=args.api_key,
            cors_origin=args.cors_origin,
            expert_preload=args.expert_preload,
            cpu_moe_layers=args.cpu_moe_layers,
            expert_preload_progress=_expert_preload_progress
            if args.expert_preload != "none"
            else None,
        )
        if args.expert_preload != "none":
            print(file=sys.stderr)
        print(
            f"Serving {service.model_name} at http://{args.host}:{args.port}",
            file=sys.stderr,
            flush=True,
        )
        try:
            serve_http(service, host=args.host, port=args.port)
        except KeyboardInterrupt:
            print("Server stopped.", file=sys.stderr)
        return 0

    if args.command == "generate-text":
        _configure_execution(args)
        tokenizer = HuggingFaceTokenizer.from_model_directory(args.root)
        model = QwenForCausalLM.from_model_directory(
            args.root, rows_per_chunk=args.rows_per_chunk
        )
        model.configure_moe_placement(args.cpu_moe_layers)
        model.preload_experts(
            mode=args.expert_preload,
            available_ram_bytes=available_ram_bytes()
            if args.expert_preload == "auto"
            else None,
            progress=_expert_preload_progress
            if args.expert_preload != "none"
            else None,
        )
        if args.expert_preload != "none":
            print(file=sys.stderr)
        prompt_tokens = tokenizer.encode_chat(
            args.prompt,
            system=args.system,
            enable_thinking=args.enable_thinking,
        )
        print(
            f"Processing {len(prompt_tokens)} prompt tokens and up to "
            f"{args.max_new_tokens} new tokens...",
            file=sys.stderr,
            flush=True,
        )
        started = time.perf_counter()
        result = TextGenerator(model, tokenizer).generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            sampling=SamplingConfig(
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
            ),
            system=args.system,
            enable_thinking=args.enable_thinking,
        )
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "text": result.text,
                    "prompt_tokens": len(result.prompt_ids),
                    "generated_tokens": list(result.generated_ids),
                    "stopped_on_eos": result.stopped_on_eos,
                    "state_tokens": result.state_tokens,
                    "seconds": elapsed,
                    "execution": _execution_stats(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "convert-model-io":
        storage = QwenModelIOConverter(args.source).convert(
            args.output,
            overwrite=args.overwrite,
            quantization=args.quantization,
        )
        print(json.dumps(storage, indent=2))
        return 0

    if args.command == "benchmark-logits":
        _configure_execution(args)
        token_ids = [
            int(value.strip())
            for value in args.token_ids.split(",")
            if value.strip()
        ]
        if not token_ids:
            raise ValueError("token-ids must contain at least one integer")
        model = QwenForCausalLM.from_model_directory(
            args.root, rows_per_chunk=args.rows_per_chunk
        )
        state = model.new_state()
        started = time.perf_counter()
        results = model.forward_ids(token_ids, state)
        elapsed = time.perf_counter() - started
        result = results[-1]
        top_token = result.greedy_token
        print(
            json.dumps(
                {
                    "input_tokens": token_ids,
                    "tokens": len(token_ids),
                    "seconds": elapsed,
                    "tokens_per_second": len(token_ids) / elapsed if elapsed else 0.0,
                    "state_tokens": state.tokens,
                    "vocab_size": model.vocab_size,
                    "top_token": top_token,
                    "top_logit": result.logits[top_token],
                    "logits_checksum": round(sum(result.logits), 6),
                    "execution": _execution_stats(),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "validate-transformers":
        _configure_execution(args)
        token_ids = [
            int(value.strip())
            for value in args.token_ids.split(",")
            if value.strip()
        ]
        model = QwenForCausalLM.from_model_directory(
            args.root, rows_per_chunk=args.rows_per_chunk
        )
        reference = TransformersReference(
            args.source,
            device=args.reference_device,
            dtype=args.reference_dtype,
            trust_remote_code=args.trust_remote_code,
            offload_dir=args.reference_offload_dir,
            max_gpu_memory_mib=args.reference_gpu_memory_mib,
            max_cpu_memory_mib=args.reference_cpu_memory_mib,
        )
        if args.component_layer is not None:
            if len(token_ids) != 1:
                raise ValueError("component validation requires one token ID")
            comparisons = diagnose_layer_components(
                model, reference, token_ids[0], args.component_layer
            )
            print(
                json.dumps(
                    {
                        "input_token": token_ids[0],
                        "layer": args.component_layer,
                        "comparisons": [asdict(item) for item in comparisons],
                    },
                    indent=2,
                )
            )
        elif args.layerwise:
            if len(token_ids) != 1:
                raise ValueError("layerwise validation requires one token ID")
            comparisons = diagnose_hidden_states(model, reference, token_ids[0])
            print(
                json.dumps(
                    {
                        "input_token": token_ids[0],
                        "comparisons": [asdict(item) for item in comparisons],
                    },
                    indent=2,
                )
            )
        else:
            report = validate_against_reference(
                model,
                reference,
                token_ids,
                generate_tokens=args.generate_tokens,
                top_k=args.top_k,
            )
            print(json.dumps(report.to_dict(), indent=2))
        return 0

    if args.command == "benchmark-decoder":
        if args.tokens <= 0:
            raise ValueError("tokens must be positive")
        decoder = QwenDecoderStack.from_model_directory(args.root)
        state = decoder.new_state()
        started = time.perf_counter()
        result = None
        for position in range(args.tokens):
            hidden = [
                math.sin((position + 1) * (index + 1) * 0.017)
                for index in range(decoder.hidden_size)
            ]
            result = decoder.forward_token(hidden, state)
        elapsed = time.perf_counter() - started
        assert result is not None
        print(
            json.dumps(
                {
                    "layers": len(decoder.layers),
                    "tokens": args.tokens,
                    "seconds": elapsed,
                    "tokens_per_second": args.tokens / elapsed if elapsed else 0.0,
                    "state_tokens": state.tokens,
                    "selected_experts": [
                        list(layer.selected_experts)
                        for layer in result.layer_results
                    ],
                    "output_checksum": round(sum(result.output), 6),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "convert-linear-layers":
        def linear_progress(completed: int, total: int) -> None:
            print(
                f"\rConverted linear layers: {completed}/{total}",
                end="",
                file=sys.stderr,
                flush=True,
            )

        storage = QwenGatedDeltaConverter(args.source).convert(
            args.output,
            overwrite=args.overwrite,
            quantization=args.quantization,
            progress=linear_progress,
        )
        if storage["layer_count"]:
            print(file=sys.stderr)
        print(json.dumps(storage, indent=2))
        return 0

    if args.command == "benchmark-linear-layer":
        if args.tokens <= 0:
            raise ValueError("tokens must be positive")
        layer = QwenGatedDeltaLayer.from_model_directory(args.root, args.layer)
        state = layer.new_state()
        started = time.perf_counter()
        result = None
        for position in range(args.tokens):
            hidden = [
                math.sin((position + 1) * (index + 1) * 0.017)
                for index in range(layer.hidden_size)
            ]
            result = layer.forward_residual(hidden, state)
        elapsed = time.perf_counter() - started
        assert result is not None
        print(
            json.dumps(
                {
                    "layer": args.layer,
                    "tokens": args.tokens,
                    "seconds": elapsed,
                    "tokens_per_second": args.tokens / elapsed if elapsed else 0.0,
                    "state_tokens": state.tokens,
                    "output_checksum": round(sum(result.output), 6),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "convert-attention-layers":
        def attention_progress(completed: int, total: int) -> None:
            print(
                f"\rConverted attention layers: {completed}/{total}",
                end="",
                file=sys.stderr,
                flush=True,
            )

        storage = QwenAttentionConverter(args.source).convert(
            args.output,
            overwrite=args.overwrite,
            quantization=args.quantization,
            progress=attention_progress,
        )
        if storage["layer_count"]:
            print(file=sys.stderr)
        print(json.dumps(storage, indent=2))
        return 0

    if args.command == "benchmark-attention-layer":
        if args.tokens <= 0:
            raise ValueError("tokens must be positive")
        layer = QwenFullAttentionLayer.from_model_directory(args.root, args.layer)
        cache = layer.new_cache()
        started = time.perf_counter()
        result = None
        for position in range(args.tokens):
            hidden = [
                math.sin((position + 1) * (index + 1) * 0.017)
                for index in range(layer.hidden_size)
            ]
            result = layer.forward_residual(hidden, position, cache)
        elapsed = time.perf_counter() - started
        assert result is not None
        print(
            json.dumps(
                {
                    "layer": args.layer,
                    "tokens": args.tokens,
                    "seconds": elapsed,
                    "tokens_per_second": args.tokens / elapsed if elapsed else 0.0,
                    "cache_length": cache.length,
                    "output_checksum": round(sum(result.output), 6),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "benchmark-expert":
        if args.iterations <= 0:
            raise ValueError("iterations must be positive")
        expert = Q4SwiGLUExpert.from_file(args.path)
        hidden = [math.sin(index * 0.17) for index in range(expert.hidden_size)]
        started = time.perf_counter()
        output = []
        for _ in range(args.iterations):
            output = expert.forward(hidden)
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "path": str(args.path.resolve()),
                    "engine": expert.engine,
                    "hidden_size": expert.hidden_size,
                    "intermediate_size": expert.intermediate_size,
                    "iterations": args.iterations,
                    "seconds": elapsed,
                    "experts_per_second": args.iterations / elapsed if elapsed else 0.0,
                    "output_checksum": round(sum(output), 6),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "create-demo":
        store = ExpertStore.create_demo(
            args.path,
            layers=args.layers,
            experts_per_layer=args.experts,
            width=args.width,
        )
        print(
            json.dumps(
                {
                    "path": str(store.root),
                    "layers": store.layers,
                    "experts_per_layer": store.experts_per_layer,
                    "width": store.width,
                    "expert_bytes": store.expert_byte_size(),
                },
                indent=2,
            )
        )
        return 0

    store = ExpertStore(args.path)
    cache = LayeredExpertCache(store.expert_byte_size() * args.cache_experts)
    predictor = TransitionPredictor()
    with ResidencyManager(store, cache) as residency:
        runtime = ToyMoERuntime(
            residency,
            predictor,
            top_k=args.top_k,
            prefetch_budget=args.prefetch,
        )
        if args.warmup:
            runtime.run(list(range(args.warmup)))
        started = time.perf_counter()
        outputs = runtime.run(list(range(args.warmup, args.warmup + args.tokens)))
        elapsed = time.perf_counter() - started
        report = {
            "tokens": args.tokens,
            "seconds": elapsed,
            "tokens_per_second": args.tokens / elapsed if elapsed else 0.0,
            "output_checksum": round(sum(sum(output) for output in outputs), 6),
            **residency.stats(),
        }
        print(json.dumps(report, indent=2))
    return 0


def _add_expert_preload_argument(
    parser: argparse.ArgumentParser, *, default: str
) -> None:
    parser.add_argument(
        "--expert-preload",
        choices=("none", "auto", "all"),
        default=default,
        help="preload routed experts into RAM before inference",
    )


def _add_cpu_moe_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cpu-moe-layers",
        type=int,
        default=0,
        help="run the first N MoE blocks on the native CPU backend",
    )


def _expert_preload_progress(completed: int, total: int) -> None:
    print(
        f"\rPreloaded experts: {completed}/{total}",
        end="",
        file=sys.stderr,
        flush=True,
    )


def _add_device_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--gpu-cache-mib", type=int, default=8192)


def _configure_execution(args: argparse.Namespace) -> None:
    if args.device == "cuda":
        accelerator = configure_cuda(
            cache_mib=args.gpu_cache_mib, device_id=args.cuda_device
        )
        print(
            f"CUDA enabled: {accelerator.device_name} "
            f"({args.gpu_cache_mib} MiB weight cache)",
            file=sys.stderr,
            flush=True,
        )
        return
    disable_cuda()


def _execution_stats() -> dict[str, int | str]:
    accelerator = active_cuda()
    stats = accelerator.stats() if accelerator is not None else {"device": "cpu"}
    native = active_native()
    if native is not None:
        stats["native_cpu_features"] = ",".join(native.features)
    return stats


if __name__ == "__main__":
    raise SystemExit(main())
