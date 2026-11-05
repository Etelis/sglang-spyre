"""Move model weights to Spyre and install the Spyre linear methods.

Called after weights load on CPU. Every unquantized ``LinearBase`` gets
spyre-inference's transposed-weight method (``custom_ops/linear.py``): merged
gate/up projections pad short row blocks (#651), every weight is transposed on
the host in blocks (#1327) before it moves to the device, and the LM head keeps
a separate row-padded ``Wᵀ`` (#633, #784). The LM head returns its logits on the
host. At TP > 1 SGLang gathers the vocabulary shards through its CPU group,
so logits post-processing and sampling stay on CPU.

Mirrors spyre-inference's ``move_model_to_spyre`` and LM-head placement at
``0ccdbf6ac2da7faacf98db93464b274fb8f40126``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang_spyre_backend.custom_ops.linear import (
    SpyreLMHeadMethod,
    SpyrePaddedRowsLinearMethod,
    SpyreUnquantizedLinearMethod,
    shard_replicated_lm_head_weight,
)
from sglang_spyre_backend.custom_ops.utils import convert

logger = logging.getLogger(__name__)

# Resolve only when placing/using weights, after worker backend registration.
# The host-only padding pass is also usable in card-free numerical tests.
_SPYRE = "spyre"
_DTYPE = torch.float16


def _lm_head_weight_shard(layer, weight: torch.Tensor) -> torch.Tensor:
    """Keep an already-sharded head, or shard a tied replicated embedding.

    Gemma-4 ties its head to a plain ``nn.Embedding``, whose input lookup must
    keep the complete host table. Its separate Spyre projection copy still
    needs one vocabulary shard per TP rank. ``VocabParallelEmbedding`` and
    ``ParallelLMHead`` already loaded their shard and must not be split twice.
    """
    if getattr(layer, "tp_size", None) is not None:
        return weight
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    shard = shard_replicated_lm_head_weight(weight, parallel.tp_rank, parallel.tp_size)
    if parallel.tp_size > 1:
        logger.info(
            "Spyre tied LM head: rank %d/%d projects %d of %d vocabulary rows; "
            "the input embedding stays replicated on CPU.",
            parallel.tp_rank,
            parallel.tp_size,
            shard.shape[0],
            weight.shape[0],
        )
    return shard


class _SpyreLinear:
    """Wrap a transposed-weight method with the host-side placement of its input."""

    def __init__(self, method):
        self.method = method

    def create_weights(self, *args, **kwargs):
        pass

    def process_weights_after_loading(self, layer):
        weight = layer.weight.data.to(device="cpu", dtype=_DTYPE)
        self.method.build_weight_t(layer, weight)
        layer.weight = nn.Parameter(layer.weight.data.to(_SPYRE), requires_grad=False)
        if getattr(layer, "bias", None) is not None:
            layer.bias = nn.Parameter(
                layer.bias.data.to(device=_SPYRE, dtype=_DTYPE), requires_grad=False
            )

    def apply(self, layer, x, bias=None):
        if x.device.type != "spyre" or x.dtype != _DTYPE:
            x = convert(x, device=_SPYRE, dtype=_DTYPE)
        return self.method.apply(layer, x, bias)


class SpyreLMHead:
    """LM-head method for SGLang's logits processor, tie-safe.

    For a tied head the module is also the input embedding, so ``embedding``
    stays the original method's host gather while ``apply`` projects through the
    row-padded ``Wᵀ`` on Spyre and returns host logits.
    """

    def __init__(self, original):
        self.original = original
        self.method = SpyreLMHeadMethod()

    def __getattr__(self, name):
        return getattr(self.original, name)

    def embedding(self, layer, input_):
        return self.original.embedding(layer, input_)

    def process_weights_after_loading(self, layer):
        weight = layer.weight.data.to(device="cpu", dtype=_DTYPE)
        weight = _lm_head_weight_shard(layer, weight)
        self.method.build_weight_t(layer, weight)
        layer.padded_weight_t = nn.Parameter(
            layer.padded_weight_t.data.to(_SPYRE), requires_grad=False
        )

    def apply(self, layer, x, bias=None):
        if x.device.type != "spyre" or x.dtype != _DTYPE:
            x = convert(x, device=_SPYRE, dtype=_DTYPE)
        logits = self.method.apply(layer, x, bias)
        return convert(logits, device="cpu")


def _is_merged_column(module) -> bool:
    from sglang.srt.layers.linear import MergedColumnParallelLinear

    return isinstance(module, MergedColumnParallelLinear)


def pad_gated_mlp_shards(model: nn.Module) -> int:
    """Stick-align already-loaded dense MLP shards without repartitioning them.

    Gemma-4's 2112-wide MLP leaves 528 lanes per TP4 rank. The fused activation
    cannot slice its up half at that unaligned offset. Pad each local gate/up
    half and the matching down-projection columns to 576. Added lanes are zero,
    so every rank's output and the subsequent TP sum retain their meaning.
    The checkpoint config is unchanged, and native TP loading has already run.
    """
    from sglang.srt.layers.activation import GeluAndMul, SiluAndMul
    from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    padded = 0
    for name, mlp in model.named_modules():
        gate_up = getattr(mlp, "gate_up_proj", None)
        down = getattr(mlp, "down_proj", None)
        if not (
            isinstance(gate_up, MergedColumnParallelLinear)
            and isinstance(down, RowParallelLinear)
            and isinstance(getattr(mlp, "act_fn", None), (GeluAndMul, SiluAndMul))
            and isinstance(gate_up.quant_method, UnquantizedLinearMethod)
            and isinstance(down.quant_method, UnquantizedLinearMethod)
        ):
            continue
        width = down.weight.shape[1]
        padding = (-width) % 64
        if not padding:
            continue
        if (
            gate_up.weight.ndim != 2
            or down.weight.ndim != 2
            or gate_up.weight.shape[0] != 2 * width
            or gate_up.gather_output
            or not down.input_is_parallel
            or gate_up.tp_size != down.tp_size
        ):
            raise ValueError(f"Unsupported Spyre gated MLP shard layout: {name}.")
        gate, up = gate_up.weight.data.chunk(2, dim=0)
        gate_up.weight.data = torch.cat((
            F.pad(gate, (0, 0, 0, padding)),
            F.pad(up, (0, 0, 0, padding)),
        ))
        if gate_up.bias is not None:
            gate_bias, up_bias = gate_up.bias.data.chunk(2, dim=0)
            gate_up.bias.data = torch.cat((
                F.pad(gate_bias, (0, padding)), F.pad(up_bias, (0, padding)),
            ))
        down.weight.data = F.pad(down.weight.data, (0, padding))
        aligned = width + padding
        gate_up.output_partition_sizes = [aligned, aligned]
        gate_up.output_size_per_partition = 2 * aligned
        gate_up.output_sizes = [aligned * gate_up.tp_size] * 2
        gate_up.output_size = sum(gate_up.output_sizes)
        down.input_size_per_partition = aligned
        down.input_size = aligned * down.tp_size
        mlp._spyre_original_partition_size = width
        padded += 1
        logger.info("Spyre gated MLP %s: local width %d -> %d.", name, width, aligned)
    return padded


def move_model_to_spyre(model: nn.Module) -> None:
    """Install the Spyre linear methods and move linear weights to the device."""
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    pad_gated_mlp_shards(model)
    n_patched = 0
    n_skipped_quant = 0
    for _name, module in model.named_modules():
        if isinstance(module, LinearBase):
            qm = getattr(module, "quant_method", None)
            if isinstance(qm, UnquantizedLinearMethod):
                method = (
                    SpyrePaddedRowsLinearMethod()
                    if _is_merged_column(module)
                    else SpyreUnquantizedLinearMethod()
                )
                module.quant_method = _SpyreLinear(method)
                module.quant_method.process_weights_after_loading(module)
                n_patched += 1
            elif qm is not None:
                n_skipped_quant += 1

    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "weight"):
        head_method = SpyreLMHead(getattr(lm_head, "quant_method", None))
        head_method.process_weights_after_loading(lm_head)
        lm_head.quant_method = head_method

    logger.info(
        "patched %d LinearBase layers, skipped %d quantized", n_patched, n_skipped_quant
    )


__all__ = ["SpyreLMHead", "move_model_to_spyre", "pad_gated_mlp_shards"]
