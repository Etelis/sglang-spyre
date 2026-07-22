"""Move model weights to Spyre and patch linear quant_method.

Called after model weights are loaded onto CPU. For each linear layer:
  1. Move weight to Spyre device
  2. Replace quant_method with SpyreLinearMethod so F.linear runs on Spyre

For non-linear modules (embedding, norm):
  - Embeddings stay on CPU (integer indexing not supported on Spyre)
  - RMSNorm weights stay on CPU (our register_oot_forward handles conversion)

Mirrors spyre-inference's TorchSpyreModelRunner._move_spyre_native_layers().
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_SPYRE = torch.device("spyre")
_DTYPE = torch.bfloat16


class SpyreLinearMethod:
    """Drop-in replacement for UnquantizedLinearMethod.

    Converts input to Spyre BF16 and calls F.linear with the weight
    already on Spyre. Output is returned on Spyre so MLP residuals
    and attention projections stay on-device.

    RowParallelLinear (output projection, o_proj) converts the result
    back to CPU — that's handled per-layer in move_model_to_spyre().
    """

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        # Delegate to the original method (already created weights); this
        # is only called at model init before we swap the method.
        pass

    def process_weights_after_loading(self, layer):
        # Move weight to Spyre
        if hasattr(layer, "weight") and layer.weight is not None:
            layer.weight.data = layer.weight.data.to(
                device=_SPYRE, dtype=_DTYPE
            )

    def apply(self, layer, x, bias=None):
        # Convert input to Spyre BF16 if not already there
        if x.device.type != "spyre":
            x = x.to(device=_SPYRE, dtype=_DTYPE)
        elif x.dtype != _DTYPE:
            x = x.to(dtype=_DTYPE)
        bias_s = bias.to(device=_SPYRE, dtype=_DTYPE) if bias is not None else None
        return F.linear(x, layer.weight.data, bias_s)


def move_model_to_spyre(model: nn.Module) -> None:
    """Walk the model and move linear weights to Spyre.

    Modules patched:
      - Any LinearBase subclass (ColumnParallelLinear, RowParallelLinear,
        MergedColumnParallelLinear, QKVParallelLinear, ReplicatedLinear):
        swap quant_method to SpyreLinearMethod.

    Modules left on CPU intentionally:
      - Embedding layers (integer index lookup not on Spyre)
      - RMSNorm (weights transferred per-call by our register_oot_forward)
      - RotaryEmbedding (CPU fallback)
    """
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    n_patched = 0
    n_skipped_quant = 0
    for name, module in model.named_modules():
        if isinstance(module, LinearBase):
            qm = getattr(module, "quant_method", None)
            if isinstance(qm, UnquantizedLinearMethod):
                module.quant_method = SpyreLinearMethod()
                # Replace the parameter wholesale — SGLang uses
                # ModelWeightParameter which rejects .data assignment of
                # a different device/dtype.
                if hasattr(module, "weight") and module.weight is not None:
                    new_weight = module.weight.data.to(device=_SPYRE, dtype=_DTYPE)
                    module.weight = nn.Parameter(new_weight, requires_grad=False)
                n_patched += 1
            elif qm is not None:
                n_skipped_quant += 1

    logger.info(
        "patched %d LinearBase layers, skipped %d quantized",
        n_patched, n_skipped_quant,
    )

    # Wrap model.forward so hidden_states comes back to CPU at the model
    # boundary — the logits processor uses advanced indexing which Spyre
    # doesn't support, and downstream sampling lives on CPU.
    _wrap_model_forward_for_cpu_output(model)


def _wrap_model_forward_for_cpu_output(model: nn.Module) -> None:
    """Monkeypatch the inner LM model so its forward() return is on CPU.

    Granite (and most decoder-only models) return `hidden_states` from
    `self.model(...)` as the last hidden state. Our optimized RMSNorm
    keeps that tensor on Spyre. The LogitsProcessor then runs
    `hidden_states[index]` which Spyre doesn't support. Bring it back
    here at the model boundary."""
    # The LM model is at `model.model` for granite; find it by attribute.
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "forward"):
        logger.warning("no inner .model attribute — skipping CPU output wrap")
        return

    original_forward = inner.forward

    def wrapped_forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        # Move hidden_states back to CPU at the boundary
        if isinstance(out, torch.Tensor) and out.device.type == "spyre":
            out = out.to("cpu")
        elif isinstance(out, tuple):
            out = tuple(
                t.to("cpu") if isinstance(t, torch.Tensor) and t.device.type == "spyre" else t
                for t in out
            )
        return out

    inner.forward = wrapped_forward
    logger.info("wrapped inner model.forward to return CPU hidden_states")
