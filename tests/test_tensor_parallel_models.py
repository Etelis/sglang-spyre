"""Host numerical checks for the model-side Spyre TP boundaries."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang_spyre_backend.custom_ops.linear import (
    SpyreLMHeadMethod,
    shard_replicated_lm_head_weight,
)


@pytest.mark.parametrize("vocab", [16, 19])
def test_four_tied_head_shards_reconstruct_full_logits_without_changing_embedding(vocab):
    torch.manual_seed(12)
    weight = torch.randn(vocab, 64)
    hidden = torch.randn(4, 64)
    embedding_before = F.embedding(torch.tensor([0, vocab - 1, 3]), weight).clone()
    logits = []
    for rank in range(4):
        head = torch.nn.Module()
        head.weight = torch.nn.Parameter(
            shard_replicated_lm_head_weight(weight, rank, 4), requires_grad=False
        )
        method = SpyreLMHeadMethod()
        method.process_weights_after_loading(head)
        logits.append(method.apply(head, hidden))
    # Each rank must produce equal-width inputs for Gloo; only padding beyond
    # the real vocabulary is removed after gathering.
    assert len({part.shape[-1] for part in logits}) == 1
    gathered = torch.cat(logits, dim=-1)[..., :vocab]
    torch.testing.assert_close(gathered, F.linear(hidden, weight))
    torch.testing.assert_close(
        F.embedding(torch.tensor([0, vocab - 1, 3]), weight), embedding_before
    )


def test_single_rank_tied_head_keeps_the_full_table():
    weight = torch.randn(19, 64)
    assert shard_replicated_lm_head_weight(weight, 0, 1) is weight


@pytest.mark.skipif(
    importlib.util.find_spec("sglang") is None, reason="SGLang is not installed"
)
@pytest.mark.parametrize("rank", range(4))
def test_gemma_tp4_loader_shards_sliding_kv_and_replicates_full_k_eq_v(monkeypatch, rank):
    """Exercise the actual model loader with the checkpoint's head geometry.

    Full attention has two 512-wide KV heads, replicated across rank pairs;
    sliding attention has eight 256-wide KV heads, partitioned across ranks.
    A CPU-only scheduler must not pad the full KV count from two to four.
    """
    import sglang.srt.layers.linear as linear
    from sglang.srt.models.gemma4_causal import Gemma4ForCausalLM

    monkeypatch.setattr(linear, "_is_cpu", False)
    hidden = 16
    model = torch.nn.Module()
    model.config = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention"],
        attention_k_eq_v=True,
        tie_word_embeddings=True,
        num_experts=0,
    )
    model.pp_group = SimpleNamespace(world_size=1)
    model.model = torch.nn.Module()
    model.model.start_layer, model.model.end_layer = 0, 2
    model.model.layers = torch.nn.ModuleList()
    model._get_k_eq_v_layers = lambda: Gemma4ForCausalLM._get_k_eq_v_layers(model)
    weights, expected = [], []
    for layer_id, (head_dim, kv_heads) in enumerate(((256, 8), (512, 2))):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        qkv = linear.QKVParallelLinear(
            hidden, head_dim, 16, kv_heads,
            bias=False, params_dtype=torch.float32, tp_rank=rank, tp_size=4,
        )
        layer.self_attn.qkv_proj = qkv
        model.model.layers.append(layer)
        q = torch.arange(16 * head_dim * hidden, dtype=torch.float32).reshape(-1, hidden)
        k = torch.arange(kv_heads * head_dim * hidden, dtype=torch.float32).reshape(-1, hidden)
        v = k if layer_id == 1 else k + k.numel()
        prefix = f"model.language_model.layers.{layer_id}.self_attn"
        weights.extend(((f"{prefix}.q_proj.weight", q), (f"{prefix}.k_proj.weight", k)))
        if layer_id == 0:
            weights.append((f"{prefix}.v_proj.weight", v))
        replicas = max(1, 4 // kv_heads)
        local_kv = max(1, kv_heads // 4) * head_dim
        start = (rank // replicas) * local_kv
        expected.append(torch.cat((
            q.narrow(0, rank * 4 * head_dim, 4 * head_dim),
            k.narrow(0, start, local_kv),
            v.narrow(0, start, local_kv),
        )))
        assert qkv.num_kv_head_replicas == replicas

    loaded = Gemma4ForCausalLM.load_weights(model, weights)
    assert len(loaded) == 2
    for layer, weight in zip(model.model.layers, expected, strict=True):
        torch.testing.assert_close(layer.self_attn.qkv_proj.weight, weight)


@pytest.mark.skipif(
    importlib.util.find_spec("sglang") is None, reason="SGLang is not installed"
)
def test_gemma_tp4_dense_mlp_padding_preserves_rank_partials_and_full_output(monkeypatch):
    import sglang.srt.layers.linear as linear
    from sglang.srt.models.gemma4_causal import Gemma4MLP
    from sglang.srt.runtime_context import get_parallel

    from sglang_spyre_backend.custom_ops.weight_placement import pad_gated_mlp_shards

    monkeypatch.setattr(linear, "_is_cpu", False)
    torch.manual_seed(43)
    hidden, intermediate, local_width, padded_width = 64, 2112, 528, 576
    x = torch.randn(4, hidden)
    gate = torch.randn(intermediate, hidden) / hidden**0.5
    up = torch.randn_like(gate) / hidden**0.5
    down = torch.randn(hidden, intermediate) / intermediate**0.5
    expected = F.linear(F.gelu(F.linear(x, gate), approximate="tanh") * F.linear(x, up), down)
    partials = []
    for rank in range(4):
        with get_parallel().override(tp_rank=rank, tp_size=4):
            mlp = Gemma4MLP(hidden, intermediate, "gelu_pytorch_tanh")
        mlp.gate_up_proj.weight.weight_loader(mlp.gate_up_proj.weight, gate, 0)
        mlp.gate_up_proj.weight.weight_loader(mlp.gate_up_proj.weight, up, 1)
        mlp.down_proj.weight.weight_loader(mlp.down_proj.weight, down)

        def evaluate():
            g, u = F.linear(x, mlp.gate_up_proj.weight).chunk(2, dim=-1)
            return F.linear(F.gelu(g, approximate="tanh") * u, mlp.down_proj.weight)

        before = evaluate()
        assert pad_gated_mlp_shards(mlp) == 1
        assert mlp.gate_up_proj.output_partition_sizes == [padded_width] * 2
        assert mlp.gate_up_proj.output_size_per_partition == 2 * padded_width
        assert mlp.gate_up_proj.output_sizes == [padded_width * 4] * 2
        assert mlp.down_proj.input_size_per_partition == padded_width
        assert mlp.down_proj.input_size == padded_width * 4
        g, u = mlp.gate_up_proj.weight.chunk(2, dim=0)
        start = rank * local_width
        torch.testing.assert_close(g[:local_width], gate[start : start + local_width])
        torch.testing.assert_close(u[:local_width], up[start : start + local_width])
        assert torch.count_nonzero(g[local_width:]) == 0
        assert torch.count_nonzero(u[local_width:]) == 0
        assert torch.count_nonzero(mlp.down_proj.weight[:, local_width:]) == 0
        after = evaluate()
        torch.testing.assert_close(after, before)
        assert pad_gated_mlp_shards(mlp) == 0
        partials.append(after)
    torch.testing.assert_close(torch.stack(partials).sum(0), expected, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(
    importlib.util.find_spec("sglang") is None, reason="SGLang is not installed"
)
@pytest.mark.parametrize("tp_size,reduce_results", [(1, True), (4, True), (4, False)])
def test_gemma_moe_reduces_tensor_parallel_expert_partials(
    monkeypatch, tp_size, reduce_results
):
    import sglang.srt.distributed as distributed

    from sglang_spyre_backend.models import _gemma4_moe_forward

    torch.manual_seed(7)
    tokens, experts, hidden, intermediate, top_k = 4, 8, 16, 704, 2
    x = torch.randn(tokens, hidden)
    router = torch.randn(tokens, experts)
    gate = torch.randn(experts, intermediate, hidden) / hidden**0.5
    up = torch.randn_like(gate) / hidden**0.5
    down = torch.randn(experts, hidden, intermediate) / intermediate**0.5
    scale = torch.rand(experts) + 0.5
    down = down * scale[:, None, None]
    probs = torch.softmax(router, dim=-1)
    routing_weights, expert_ids = probs.topk(top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(-1, keepdim=True)

    def evaluate(g, u, d):
        act = F.gelu(torch.einsum("th,eih->tei", x, g), approximate="tanh")
        act = act * torch.einsum("th,eih->tei", x, u)
        all_experts = torch.einsum("tei,ehi->teh", act, d)
        selected = all_experts.gather(1, expert_ids[..., None].expand(-1, -1, hidden))
        return (routing_weights[..., None] * selected).sum(1)

    expected = evaluate(gate, up, down)
    partials = []
    width = intermediate // tp_size
    for rank in range(tp_size):
        start, end = rank * width, (rank + 1) * width
        # Gemma's TP4 shard is 176 lanes, which Spyre pads to 192. Zero lanes
        # in both projections must not alter the sum of the rank partials.
        padding = (-width) % 64
        partials.append(
            evaluate(
                F.pad(gate[:, start:end], (0, 0, 0, padding)),
                F.pad(up[:, start:end], (0, 0, 0, padding)),
                F.pad(down[:, :, start:end], (0, padding)),
            )
        )
    summed = torch.stack(partials).sum(0)
    torch.testing.assert_close(summed, expected, atol=2e-5, rtol=2e-5)

    opaque = ModuleType("sglang_spyre_backend.opaque")
    opaque.LayerName = lambda name: name
    monkeypatch.setitem(sys.modules, opaque.__name__, opaque)
    layer = SimpleNamespace(
        _spyre_moe=True,
        _spyre_moe_key="moe.0",
        experts=SimpleNamespace(reduce_results=reduce_results, moe_tp_size=tp_size),
    )
    calls = []

    def reduce(partial):
        calls.append(partial)
        torch.testing.assert_close(partial, partials[0])
        return summed

    monkeypatch.setattr(distributed, "tensor_model_parallel_all_reduce", reduce)
    monkeypatch.setattr(
        torch.ops.sglang_spyre, "moe_forward", lambda *args: partials[0], raising=False
    )
    got = _gemma4_moe_forward(layer, x, router)
    should_reduce = reduce_results and tp_size > 1
    assert len(calls) == int(should_reduce)
    torch.testing.assert_close(got, expected if should_reduce else partials[0])
