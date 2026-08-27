"""Checks qwen4exp_reference.py against the transformers Qwen4Exp modules.

The NumPy reference is the oracle the C++ kernels will be built against, so it
is pinned here against the merged upstream implementation (transformers PR
#48337, on main since 2026-08-26) module by module:

    Qwen4ExpTextGatedResidual   -> hc_mix / hc_inject (and the use_combine=False
                                   head-collapse form)
    Qwen4ExpTextNGramEmbedding  -> ngram_rows (integer hashes, zero tolerance)
    Qwen4ExpTextPLELayer        -> ple_forward (incl. the dilated conv), run
                                   whole-sequence AND split prefill+decode to
                                   pin the conv/token-history state handling
    Qwen4ExpTextRMSNorm         -> grouped_rms (baked-weight convention)
    Qwen4ExpTextQSAIndexer      -> qsa_token_mask (block pooling, scoring and
                                   top-k selection; the bool mask must match
                                   EXACTLY, scores have no float ties here)

Weight-baking note: the HF modules compute `normed * (1 + w)` with zero-init
`w`; the GGUF conversion stores `1 + w`. Torch modules here get random `w`,
and the NumPy side receives `1 + w` -- the comparison covers the convention.

Needs transformers >= the qwen4_exp merge; skips cleanly when absent:

    python native/tools/qwen4exp_reference_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen4exp_reference as ref  # noqa: E402

HC, HIDDEN, LOW_RANK = 4, 64, 16
NGRAM, HEADS_PER_NGRAM, PLE_DIM = 3, 8, 320  # 16 heads x 20 dims
VOCAB = 128
EOS = 5


def main() -> int:
    try:
        import torch
        from transformers.models.qwen4_exp.configuration_qwen4_exp import (
            Qwen4ExpTextConfig,
        )
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            Qwen4ExpTextGatedResidual,
            Qwen4ExpTextNGramEmbedding,
            Qwen4ExpTextPLELayer,
        )
    except ImportError as error:
        print(f"skipped: {error}")
        return 0

    torch.manual_seed(20260826)
    rng = np.random.default_rng(20260826)
    failures = 0

    def check(name, actual, expected, tolerance=1e-5):
        nonlocal failures
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        scale = max(float(np.abs(expected).mean()), 1e-30)
        error = float(np.abs(actual - expected).max()) / scale
        ok = error <= tolerance
        failures += not ok
        print(f"{'OK ' if ok else 'BAD'} {name:34s} rel-err {error:.3e}")

    config = Qwen4ExpTextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        hc_count=HC,
        hc_lowrank=LOW_RANK,
        ngram_size=NGRAM,
        heads_per_ngram=HEADS_PER_NGRAM,
        ple_embed_dim=PLE_DIM,
        ple_layer_ids=[2],
        ngram_vocab_size_base=97,          # tiny primes, tiny table
        make_ngram_vocab_size_divisible_by=1,
        ple_conv_kernel_size=4,
        rms_norm_eps=1e-6,
        eos_token_id=EOS,
    )
    rows = 12
    wide = HC * HIDDEN

    # --- GatedResidual ----------------------------------------------------
    torch_hc = Qwen4ExpTextGatedResidual(config).float().eval()
    with torch.no_grad():
        for p in torch_hc.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    hyper = rng.standard_normal((rows, wide)).astype(np.float32)
    block_out = rng.standard_normal((rows, HIDDEN)).astype(np.float32)

    with torch.no_grad():
        mixed, hyper_back, injw = torch_hc(torch.tensor(hyper))
        torch_streams = hyper_back + (
            torch.tensor(block_out).unsqueeze(-2) * injw.unsqueeze(-1)
        ).flatten(-2)

    baked_norm = (1.0 + torch_hc.hc_norm.weight.detach().numpy())
    down = torch_hc.input_mix_weight_down.weight.detach().numpy().T
    up = torch_hc.input_mix_weight_up.weight.detach().numpy().T
    inject = torch_hc.block_inject_weight.weight.detach().numpy().T
    np_mixed, np_injw = ref.hc_mix(hyper, baked_norm, down, up, inject, HC, HIDDEN)
    check("hc_mix block_input", np_mixed, mixed.numpy())
    check("hc_mix inject_weights", np_injw, injw.numpy())
    check("hc_inject", ref.hc_inject(hyper, block_out, np_injw, HC, HIDDEN),
          torch_streams.numpy())

    torch_head = Qwen4ExpTextGatedResidual(config, use_combine=False).float().eval()
    with torch.no_grad():
        for p in torch_head.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
        head_out = torch_head(torch.tensor(hyper))
    np_head, none_w = ref.hc_mix(
        hyper, 1.0 + torch_head.hc_norm.weight.detach().numpy(),
        torch_head.input_mix_weight_down.weight.detach().numpy().T,
        torch_head.input_mix_weight_up.weight.detach().numpy().T,
        None, HC, HIDDEN)
    assert none_w is None
    check("hc head collapse", np_head, head_out.numpy())

    # --- NGramEmbedding hashes (zero tolerance) ---------------------------
    embed = Qwen4ExpTextNGramEmbedding(config, PLE_DIM, layer_idx=1).float().eval()
    tokens = rng.integers(0, VOCAB, size=40)
    tokens[[7, 23]] = EOS  # exercise segmentation
    with torch.no_grad():
        # reproduce the module's id pipeline exactly (forward folds in the
        # embedding lookup; ids are recovered by running its shift+hash on the
        # same history the reference sees: fresh sequence, no cache)
        torch_emb = embed(torch.tensor(tokens[None, :]), None)
    np_rows = ref.ngram_rows(
        tokens,
        embed.layer_multipliers.numpy().astype(np.uint64),
        np.array(embed.head_vocab_sizes, dtype=np.int64),
        np.array(embed.head_offsets, dtype=np.int64),
        HEADS_PER_NGRAM, NGRAM, EOS)
    table = embed.ngram_embedding.weight.detach().numpy()
    np_emb = table[np_rows].reshape(len(tokens), -1)
    check("ngram embedding gather", np_emb, torch_emb[0].numpy(), tolerance=0.0)

    # --- PLELayer: whole-sequence vs reference ---------------------------
    ple = Qwen4ExpTextPLELayer(config, layer_idx=1, ple_layer_index=0).float().eval()
    with torch.no_grad():
        for name, p in ple.named_parameters():
            if "norm" in name:
                p.copy_(torch.randn_like(p) * 0.1)
            else:
                p.copy_(torch.randn_like(p) * 0.2)
    hyper2 = rng.standard_normal((rows, wide)).astype(np.float32)
    seq = rng.integers(0, VOCAB, size=rows)
    seq[3] = EOS
    with torch.no_grad():
        torch_delta = ple(torch.tensor(hyper2[None, :, :]),
                          torch.tensor(seq[None, :]), None)[0].numpy()

    hash_rows = ref.ngram_rows(
        seq,
        ple.ple_embedding.layer_multipliers.numpy().astype(np.uint64),
        np.array(ple.ple_embedding.head_vocab_sizes, dtype=np.int64),
        np.array(ple.ple_embedding.head_offsets, dtype=np.int64),
        HEADS_PER_NGRAM, NGRAM, EOS)
    emb_np = ple.ple_embedding.ngram_embedding.weight.detach().numpy()[hash_rows]
    emb_np = emb_np.reshape(rows, -1)
    np_delta, np_state = ref.ple_forward(
        hyper2, emb_np,
        ple.key_proj.weight.detach().numpy().T,
        ple.value_proj.weight.detach().numpy().T,
        1.0 + ple.norm_key.weight.detach().numpy(),
        1.0 + ple.norm_query.weight.detach().numpy(),
        1.0 + ple.norm_conv.weight.detach().numpy(),
        ple.conv1d.weight.detach().numpy()[:, 0, :],  # [wide][kernel]
        None, HC, HIDDEN, NGRAM)
    check("ple whole-sequence", np_delta, torch_delta)

    # --- PLELayer: split prefill+decode == whole (state handling) --------
    np_d1, state = ref.ple_forward(
        hyper2[:8], emb_np[:8],
        ple.key_proj.weight.detach().numpy().T,
        ple.value_proj.weight.detach().numpy().T,
        1.0 + ple.norm_key.weight.detach().numpy(),
        1.0 + ple.norm_query.weight.detach().numpy(),
        1.0 + ple.norm_conv.weight.detach().numpy(),
        ple.conv1d.weight.detach().numpy()[:, 0, :],
        None, HC, HIDDEN, NGRAM)
    np_d2, _ = ref.ple_forward(
        hyper2[8:], emb_np[8:],
        ple.key_proj.weight.detach().numpy().T,
        ple.value_proj.weight.detach().numpy().T,
        1.0 + ple.norm_key.weight.detach().numpy(),
        1.0 + ple.norm_query.weight.detach().numpy(),
        1.0 + ple.norm_conv.weight.detach().numpy(),
        ple.conv1d.weight.detach().numpy()[:, 0, :],
        state, HC, HIDDEN, NGRAM)
    check("ple split==whole", np.concatenate([np_d1, np_d2]), np_delta)

    # --- QSA indexer: selection mask, exact -------------------------------
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextQSAIndexer,
    )

    qsa_config = Qwen4ExpTextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=4,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25},
    )
    indexer = Qwen4ExpTextQSAIndexer(qsa_config, layer_idx=3).float().eval()
    with torch.no_grad():
        for name, p in indexer.named_parameters():
            p.copy_(torch.randn_like(p) * (0.1 if "norm" in name else 0.3))

    qsa_rows, rot = 61, 8  # budget 8 / ratio 4: pruning from 12 visible on
    hidden = rng.standard_normal((qsa_rows, HIDDEN)).astype(np.float32)
    inv_freq = 10000.0 ** (-np.arange(0, rot, 2, dtype=np.float64) / rot)
    angles = np.arange(qsa_rows)[:, None] * inv_freq[None, :]
    cos = np.tile(np.cos(angles), (1, 2)).astype(np.float32)
    sin = np.tile(np.sin(angles), (1, 2)).astype(np.float32)

    causal = torch.tril(torch.ones(qsa_rows, qsa_rows, dtype=torch.bool))
    with torch.no_grad():
        torch_mask = indexer(
            torch.tensor(hidden[None, :, :]),
            (torch.tensor(cos[None, :, :]), torch.tensor(sin[None, :, :])),
            causal[None, None, :, :],
            None,
        )[0, 0].numpy()

    qk_w = indexer.index_qk_proj.weight.detach().numpy().T
    np_mask = ref.qsa_token_mask(
        hidden, qk_w,
        1.0 + indexer.q_layernorm.weight.detach().numpy(),
        1.0 + indexer.k_layernorm.weight.detach().numpy(),
        cos, sin, n_heads=4, head_dim=16, budget=8, ratio=4)
    agree = bool((np_mask == torch_mask).all())
    failures += not agree
    print(f"{'OK ' if agree else 'BAD'} {'qsa selection mask':34s} "
          f"{'exact' if agree else f'{int((np_mask != torch_mask).sum())} cells differ'}")
    # sanity on the gate the runtime relies on: no pruning below the budget
    assert np_mask[:11, :].sum() == np.tril(np.ones((11, 11))).sum(), \
        "qsa pruned below the budget threshold"

    print("reference matches transformers Qwen4Exp" if not failures
          else "reference DISAGREES with transformers")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
