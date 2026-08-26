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

logger = logging.getLogger(__name__)

_SPYRE = torch.device("spyre")
_DTYPE = torch.float16


class SpyreLinearMethod:
    """Unquantized linear method using spyre-inference's transposed fast path."""

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        # Delegate to the original method (already created weights); this
        # is only called at model init before we swap the method.
        pass

    def process_weights_after_loading(self, layer):
        # Store W^T physically as [in, out]. This is the current
        # spyre-inference fast path and avoids F.linear's x @ A^T layout.
        if hasattr(layer, "weight") and layer.weight is not None:
            weight_t = layer.weight.data.to(dtype=_DTYPE).t().contiguous().to(_SPYRE)
            layer.weight = nn.Parameter(weight_t, requires_grad=False)
            layer.spyre_weight_transposed = True

    def apply(self, layer, x, bias=None):
        # Convert input to the pinned Spyre dtype if not already there.
        if x.device.type != "spyre":
            x = x.to(device=_SPYRE, dtype=_DTYPE)
        elif x.dtype != _DTYPE:
            x = x.to(dtype=_DTYPE)
        bias_s = bias.to(device=_SPYRE, dtype=_DTYPE) if bias is not None else None
        out = torch.matmul(x, layer.weight)
        if bias_s is not None:
            out = out + bias_s
        return out


class SpyreLMHeadMethod:
    """LM-head variant of the transposed fast path.

    The original weight remains on CPU for tied embeddings; an independently
    stored, row-padded transpose is used for the projection on Spyre.
    """

    def process_weights_after_loading(self, layer):
        weight = layer.weight.data.to(dtype=_DTYPE)
        padding = (-weight.shape[0]) % (64 * 32)
        if padding:
            weight = torch.nn.functional.pad(weight, (0, 0, 0, padding))
        layer.spyre_row_padding = padding
        layer.padded_weight_t = nn.Parameter(
            weight.t().contiguous().to(_SPYRE), requires_grad=False
        )

    def apply(self, layer, x, bias=None):
        if x.device.type != "spyre" or x.dtype != _DTYPE:
            x = x.to(device=_SPYRE, dtype=_DTYPE)
        out = torch.matmul(x, layer.padded_weight_t)
        if bias is not None:
            out = out + bias.to(device=_SPYRE, dtype=_DTYPE)
        padding = layer.spyre_row_padding
        return out[:, :-padding] if padding else out


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
                module.quant_method.process_weights_after_loading(module)
                n_patched += 1
            elif qm is not None:
                n_skipped_quant += 1

    # Port spyre-inference's dedicated tied-weight-safe LM-head path.
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "weight"):
        lm_method = SpyreLMHeadMethod()
        lm_method.process_weights_after_loading(lm_head)
        lm_head.quant_method = lm_method

    logger.info(
        "patched %d LinearBase layers, skipped %d quantized",
        n_patched,
        n_skipped_quant,
    )

    # Keep indexing/sampling on CPU while running the large LM-head projection
    # on Spyre, matching spyre-inference's model-wrapper boundary.
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
                t.to("cpu")
                if isinstance(t, torch.Tensor) and t.device.type == "spyre"
                else t
                for t in out
            )
        return out

    inner.forward = wrapped_forward
    if hasattr(model, "compute_logits"):
        original_compute_logits = model.compute_logits

        def wrapped_compute_logits(hidden_states, *args, **kwargs):
            if isinstance(hidden_states, torch.Tensor):
                hidden_states = hidden_states.to(device=_SPYRE, dtype=_DTYPE)
            result = original_compute_logits(hidden_states, *args, **kwargs)

            def to_cpu(value):
                if isinstance(value, torch.Tensor) and value.device.type == "spyre":
                    return value.to("cpu")
                return value

            if isinstance(result, tuple):
                return tuple(to_cpu(value) for value in result)
            if hasattr(result, "__dict__"):
                for key, value in vars(result).items():
                    setattr(result, key, to_cpu(value))
                return result
            return to_cpu(result)

        model.compute_logits = wrapped_compute_logits
    logger.info("wrapped inner model.forward to return CPU hidden_states")
