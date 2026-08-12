"""Per-layer activation dump from the real BailingMoE3 checkpoint, on CPU.

The point of this is to make the assembled runtime debuggable one layer at a
time. Every component is already pinned against torch or a torch-pinned oracle
(router, MoE block, KDA recurrence, both MLA forms, RoPE, head gate), so when
the assembled model produces wrong output the fault is in the wiring -- and
diffing end-to-end logits tells you nothing about *which* layer.

Running the reference here needs work, because the checkpoint's own modeling
code reaches for Triton kernels that cannot run on this machine (the installed
PyTorch does not support the GPU's sm_120). So three pieces are replaced with
pure-PyTorch equivalents that compute the same function:

  * ShortConvolution -> a causal depthwise conv1d plus SiLU;
  * chunk_kda / fused_recurrent_kda -> fla's own `naive_recurrent_kda`,
    composed with `naive_kda_gate` and the q/k L2 normalization that
    `use_qk_l2norm_in_kernel=True` folds in;
  * FusedRMSNormGated -> RMS norm times sigmoid gate.

The first and third are transcriptions of small, unambiguous operations. The
second is the substitution that carries real risk, and it is exactly the one
already verified: kda_reference_check.py shows `naive_recurrent_kda` and
`naive_chunk_kda` agree with each other and with our oracle to ~1e-5.

REQUIRES transformers 4.57.x. This is not optional and not what config.json
says (it claims 4.45.0, which is stale). On a newer stack the model loads and
runs but returns DIFFERENT activations on every process, with roughly one run in
three producing NaN -- deterministic within a process, varying across them,
which is a read from uninitialized memory. It is not always visibly broken:
finite runs disagree with each other too, so a dump can look clean and be wrong.

    python -m venv ~/.venvs/ling-ref
    ~/.venvs/ling-ref/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    ~/.venvs/ling-ref/bin/pip install "transformers==4.57.1" flash-linear-attention \
        einops accelerate triton
    ~/.venvs/ling-ref/bin/python native/tools/bailing_reference.py <checkpoint>

`triton` is needed even on CPU because fla imports it at module load.

NOTE: the reference runs NON-CAUSAL attention. transformers hands the
checkpoint's `eager_attention_forward` no mask, and that function only masks
when given one. Comparisons must account for it: the last token is unaffected
(it attends to everything either way), and earlier tokens only match if the
implementation under test is also run non-causally.

    python native/tools/bailing_reference.py <checkpoint> [--layers 0,3] [--out dump.npz]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def install_transformers_shims() -> None:
    """Re-add symbols the checkpoint's modeling code imports but v5 removed.

    `modeling_bailing_moe_v3.py` was written against transformers 4.45 and is
    loaded verbatim through trust_remote_code, so it imports names that no
    longer exist. These are re-exported rather than the library downgraded --
    downgrading would silently change the reference this whole exercise is
    measured against.
    """
    import transformers.utils.import_utils as import_utils

    if not hasattr(import_utils, "is_torch_fx_available"):
        # Only ever consulted to decide whether to register fx tracing helpers,
        # which this harness does not use.
        import_utils.is_torch_fx_available = lambda: False
    import transformers.utils as utils
    if not hasattr(utils, "is_torch_fx_available"):
        utils.is_torch_fx_available = import_utils.is_torch_fx_available

    # v5 dropped the "default" (unscaled) entry from ROPE_INIT_FUNCTIONS, which
    # is the one this checkpoint asks for. Re-register it. The formula is the
    # plain unscaled definition, inv_freq = 1 / theta^(2i/d) with unit attention
    # scaling -- the same one partial_rope_norm was verified against, so the
    # harness and the implementation under test are not being fed different
    # rope definitions.
    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS:
        def _default_rope(config, device=None, seq_len=None, **kwargs):
            base = config.rope_theta
            partial = getattr(config, "partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim", None) or (
                config.hidden_size // config.num_attention_heads)
            dim = int(head_dim * partial)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float) / dim))
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = _default_rope


def install_cpu_substitutes() -> None:
    """Patch the Triton-only pieces before the modeling module imports them."""
    import torch
    import torch.nn.functional as F
    import fla.modules
    from fla.ops.kda.gate import naive_kda_gate
    from fla.ops.kda.naive import naive_recurrent_kda

    class CpuShortConvolution(torch.nn.Conv1d):
        """Causal depthwise conv1d + SiLU, matching ShortConvolution's contract.

        Returns (output, final_state); `cache` carries the trailing window so a
        step-at-a-time decode continues where a prefill left off.
        """

        def __init__(self, hidden_size, kernel_size, activation=None, bias=False,
                     **kwargs):
            # bias=False is not a default worth inheriting from Conv1d, which
            # has it True: the checkpoint carries only `*_conv1d.weight`, so a
            # bias here is silently random-initialized and poisons every KDA
            # layer. transformers reports it as MISSING, which is the only
            # visible sign.
            super().__init__(
                in_channels=hidden_size, out_channels=hidden_size,
                kernel_size=kernel_size, groups=hidden_size, padding=kernel_size - 1,
                bias=bias,
            )
            self.hidden_size = hidden_size
            self.activation = activation

        def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None, **kwargs):
            # x: [batch, tokens, channels] -> conv wants [batch, channels, tokens]
            shaped = x.transpose(1, 2)
            if cache is not None:
                shaped = torch.cat([cache, shaped], dim=-1)
            width = self.kernel_size[0]
            padded = shaped if cache is not None else F.pad(shaped, (width - 1, 0))
            out = F.conv1d(padded, self.weight, self.bias, groups=self.groups)
            out = out[..., -x.shape[1]:]
            if self.activation == "silu":
                out = F.silu(out)
            state = shaped[..., -(width - 1):] if output_final_state and width > 1 else None
            return out.transpose(1, 2), state

    class CpuFusedRMSNormGated(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-6, activation="sigmoid", **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.eps = eps
            self.activation = activation

        def forward(self, x, gate):
            dtype = x.dtype
            x = x.float()
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            x = x * self.weight.float()
            gate = torch.sigmoid(gate.float()) if self.activation == "sigmoid" \
                else F.silu(gate.float())
            return (x * gate).to(dtype)

    def cpu_kda(q, k, v, g, beta, A_log=None, dt_bias=None, initial_state=None,
                output_final_state=True, use_qk_l2norm_in_kernel=False,
                use_gate_in_kernel=False, **kwargs):
        # `**kwargs` also swallows safe_gate and lower_bound, exactly as fla
        # 0.4.1's own chunk_kda does -- see kda_reference.py for why that
        # matters and is deliberate rather than an oversight here.
        if use_gate_in_kernel:
            g = naive_kda_gate(g, A_log, dt_bias)
        if use_qk_l2norm_in_kernel:
            q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
            k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)
        # The recurrence runs in f32 regardless of the model's dtype -- the real
        # kernels do the same -- but the result has to come back in the caller's
        # dtype or the next Linear sees a mismatch.
        dtype = v.dtype
        out, state = naive_recurrent_kda(
            q=q.float(), k=k.float(), v=v.float(), g=g.float(), beta=beta.float(),
            initial_state=initial_state, output_final_state=output_final_state,
        )
        return out.to(dtype), state

    fla.modules.ShortConvolution = CpuShortConvolution
    fla.modules.FusedRMSNormGated = CpuFusedRMSNormGated
    import fla.ops.kda as kda_ops
    kda_ops.chunk_kda = cpu_kda
    kda_ops.fused_recurrent_kda = cpu_kda


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--layers", default="0,3",
                        help="comma-separated layer indices to dump")
    parser.add_argument("--out", default="")
    parser.add_argument("--dump-attention", type=int, default=-1,
                        help="also export this MLA layer's attention weights, "
                             "its input and its output, for C++ parity testing")
    arguments = parser.parse_args()

    install_transformers_shims()
    install_cpu_substitutes()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from transformers import AutoConfig

    path = Path(arguments.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    # config.json says `"rope_scaling": null`, but transformers v5 fills the
    # field with a default dict. The modeling code then takes its
    # `rope_scaling is not None` branch and looks for a "factor" that was never
    # there. Restoring None is what the checkpoint actually asks for -- and it
    # matters beyond loading, because that branch would also multiply the
    # attention scale by an mscale the model was not trained with.
    if getattr(config, "rope_scaling", None) is not None and \
            "factor" not in config.rope_scaling:
        config.rope_scaling = None

    print("loading on CPU (bf16 -> f32 is ~32 GB; this takes a while)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, config=config, trust_remote_code=True, dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    # Replace the rotary forward's matmul with a broadcast outer product.
    #
    # `BailingMoeV3RotaryEmbedding.forward` builds its angles as
    # `inv_freq[None,:,None] @ positions[:,None,:]` -- a 32x1 @ 1x5 matmul. On
    # this machine that intermittently returns NaN in a single row (observed at
    # frequency index 9, giving 2 NaNs per head per token in the rope span and
    # nowhere else), which then poisons every attention score. inv_freq itself
    # is clean and deterministic, so the fault is in the matmul, not its inputs.
    #
    # The replacement is the same quantity written as a broadcast product, which
    # avoids the BLAS path entirely. This is the one substitution here that is
    # not about CPU/Triton portability -- it is working around a numerically
    # unreliable kernel, and it is why the earlier NaN runs were intermittent.
    rotary = model.model.rotary_emb

    def stable_rotary(x, position_ids):
        inv_freq = rotary.inv_freq.to(torch.float32)
        positions = position_ids.to(torch.float32)
        angles = inv_freq[None, :, None] * positions[:, None, :]
        angles = angles.transpose(1, 2)
        emb = torch.cat((angles, angles), dim=-1)
        scaling = getattr(rotary, "attention_scaling", 1.0)
        return (emb.cos() * scaling).to(x.dtype), (emb.sin() * scaling).to(x.dtype)

    rotary.forward = stable_rotary

    # Force causal masking.
    #
    # transformers hands this checkpoint's `eager_attention_forward` no mask,
    # and that function only masks when given one -- so the MLA layers run
    # BIDIRECTIONAL by default. For a causal LM that is simply wrong: every
    # position below the last sees the future. It survives casual inspection
    # because the final position attends to the same set either way, so the
    # model still answers " Paris"; but the hidden states feeding the next layer
    # are polluted, and through 24 layers that reaches the final logits too.
    #
    # Patching the module global works because the MLA forward re-reads it on
    # every call rather than capturing it at import.
    layer_module = sys.modules[type(model.model.layers[0].attention).__module__]
    original_attention = layer_module.eager_attention_forward

    def causal_attention(module, query, key, value, attention_mask, scaling,
                         dropout=0.0, **kwargs):
        if attention_mask is None:
            tokens, keys = query.shape[-2], key.shape[-2]
            offset = keys - tokens
            positions = torch.arange(tokens, device=query.device)[:, None] + offset
            allowed = positions >= torch.arange(keys, device=query.device)[None, :]
            attention_mask = torch.zeros(1, 1, tokens, keys, dtype=query.dtype,
                                         device=query.device)
            attention_mask.masked_fill_(~allowed[None, None], float("-inf"))
        return original_attention(module, query, key, value, attention_mask,
                                  scaling, dropout, **kwargs)

    layer_module.eager_attention_forward = causal_attention

    wanted = {int(x) for x in arguments.layers.split(",") if x != ""}
    captured: dict[str, np.ndarray] = {}

    def capture(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().float().numpy()
        return hook

    handles = []
    layers = model.model.layers
    for index in sorted(wanted):
        block = layers[index]
        handles.append(block.register_forward_hook(capture(f"layer.{index}.out")))
        handles.append(block.attention.register_forward_hook(
            capture(f"layer.{index}.attention.out")))
        handles.append(block.mlp.register_forward_hook(capture(f"layer.{index}.mlp.out")))
        handles.append(block.input_layernorm.register_forward_hook(
            capture(f"layer.{index}.input_norm.out")))

    # Export one MLA layer's attention block so the C++ assembly can be diffed
    # against real weights rather than random ones. Random weights exercise the
    # arithmetic; only real ones catch a projection wired to the wrong tensor.
    if arguments.dump_attention >= 0:
        block = layers[arguments.dump_attention]
        attention = block.attention
        mla = hasattr(attention, "kv_b_proj")
        captured["kind"] = np.array([1 if mla else 0])
        if mla:
            names = ("q_a_proj", "q_a_layernorm", "q_b_proj",
                     "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj",
                     "g_proj", "dense")
        else:
            names = ("q_proj", "k_proj", "v_proj", "q_conv1d", "k_conv1d",
                     "v_conv1d", "f_proj", "b_proj", "g_proj", "o_norm",
                     "o_proj")
        for name in names:
            captured[f"w.{name}"] = \
                getattr(attention, name).weight.detach().float().numpy()
        if not mla:
            # A_log and dt_bias are bare Parameters, not modules with .weight.
            captured["w.A_log"] = attention.A_log.detach().float().numpy()
            captured["w.dt_bias"] = attention.dt_bias.detach().float().numpy()
        # The decoder layer calls attention with keyword arguments only, so a
        # plain pre-hook sees an empty positional tuple.
        def grab_input(_module, inputs, kwargs):
            tensor = kwargs.get("hidden_states")
            if tensor is None and inputs:
                tensor = inputs[0]
            captured["attention.in"] = tensor.detach().float().numpy()

        handles.append(attention.register_forward_pre_hook(
            grab_input, with_kwargs=True))
        handles.append(attention.register_forward_hook(
            capture("attention.out")))

    ids = tokenizer(arguments.prompt, return_tensors="pt")["input_ids"]
    print(f"prompt {arguments.prompt!r} -> {ids.shape[1]} tokens", flush=True)
    with torch.no_grad():
        out = model(ids, use_cache=False)
    for handle in handles:
        handle.remove()

    captured["input_ids"] = ids.numpy()
    captured["logits"] = out.logits.detach().float().numpy()
    for name in sorted(captured):
        array = captured[name]
        print(f"  {name:34s} {str(array.shape):20s} "
              f"mean {array.mean():+.6f}  std {array.std():.6f}")

    # Refuse to write a poisoned oracle.
    #
    # This is not hypothetical. While building this harness the MLA layers
    # produced NaN on two consecutive runs and then stopped, with no change to
    # the code in between; three runs since have been byte-identical. The cause
    # is not understood. A silently non-finite dump would be far worse than a
    # failed run, because every later parity check would be measured against it.
    poisoned = sorted(
        name for name, array in captured.items()
        if name != "input_ids" and not np.isfinite(array).all()
    )
    if poisoned:
        print("\nREFUSING TO WRITE: non-finite activations in " + ", ".join(poisoned),
              file=sys.stderr)
        return 1

    if arguments.out:
        np.savez(arguments.out, **captured)
        print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
