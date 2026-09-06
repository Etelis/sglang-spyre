"""Register Spyre on-device ops ported from current spyre-inference.

The point of this module: when SGLang runs with ``device="spyre"`` the whole
model body (RMSNorm, activations, RoPE, linear) is materialised on the Spyre
device, and the residual stream stays resident across the forward pass — so the
CPU<->Spyre boundary is crossed once per forward, not once per attention layer.
That is the only regime in which Spyre beats CPU (see the perf analysis: the
attention-only prototype was ~2x slower than CPU because of the per-layer
ping-pong).

Reuse strategy (maximal): SGLang's ops are ``MultiPlatformOp`` subclasses whose
``dispatch_forward`` consults ``_oot_forward_registry[get_dispatch_key_name()]``
*first* (layers/utils/multi_platform.py:109-119). So we register a
``forward_spyre`` per op via ``MultiPlatformOp.register_oot_forward(op_cls, fn,
"spyre")``. The compute bodies are lifted from spyre-inference's custom_ops
(the canonical Spyre kernels) and adapted from its static-method / forward_oot
shape to SGLang's bound-method signature.

Provenance: github.com/torch-spyre/spyre-inference (Apache-2.0) @ 6e5ff996.
Each op below carries a ``Source:`` link to the exact file/lines it was adapted
from. spyre-inference itself adapts the upstream vLLM/torch ops.

These are deliberately thin: on a real ``device="spyre"`` tensor most of this is
plain torch that ``torch_spyre`` lowers via its eager kernels; the value here is
(a) matching SGLang's op signatures and (b) carrying spyre-inference's device
quirk workarounds (e.g. slicing Spyre tensors corrupts memory -> slice on CPU).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from sglang.srt.layers.utils import MultiPlatformOp

SPYRE_KEY = "spyre"


# ---------------------------------------------------------------------------
# RMSNorm — reuses spyre-inference/custom_ops/rms_norm.py:forward_spyre math.
# SGLang RMSNorm carries self.variance_epsilon, self.hidden_size, self.weight.
# Source: torch-spyre/spyre-inference @ 29faeb2 · custom_ops/rms_norm.py
#   forward_spyre() — the pow(2).mean / rsqrt math at L153-159:
#   https://github.com/torch-spyre/spyre-inference/blob/29faeb240801e7793c397ffa80422e0629e80aa9/spyre_inference/custom_ops/rms_norm.py#L130-L166
# ---------------------------------------------------------------------------
def _rmsnorm_forward_spyre(self, x, residual=None, post_residual_addition=None):
    # On-device RMSNorm. VALIDATED on real Spyre hardware (eager + compiled, in a
    # realistic residual+linear layer): the variance reduction runs on the device
    # in the tensor's native (bf16/fp16) dtype. The ONLY thing that fails is the
    # FP32 promotion that SGLang's forward_native does (`x.to(float32)`) — that
    # raises "Unsupported: mean on DataFormats.IEEE_FP32". So we deliberately do
    # NOT promote: keep everything in x.dtype. This keeps the residual stream
    # resident on the device (no CPU round-trip), which is the whole point.
    #
    # Identical in spirit to spyre-inference's SpyreRMSNorm.forward_spyre
    # (custom_ops/rms_norm.py): "No dtype promotion to torch.float32
    # (torch-spyre limitation)". Accepts a small accuracy cost vs fp32.
    if post_residual_addition is not None:
        x = x + post_residual_addition

    if residual is not None:
        x = x + residual
        residual_out = x
    else:
        residual_out = None

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + self.variance_epsilon)
    if getattr(self, "has_weight", True):
        x = x * self.weight.to(x.dtype)

    if residual_out is None:
        return x
    return x, residual_out


# ---------------------------------------------------------------------------
# SiluAndMul / GeluAndMul stay device-resident. NewGELU carries the current
# spyre-inference multiplication workaround for torch-spyre#4009.
# ---------------------------------------------------------------------------
def _silu_and_mul_forward_spyre(self, x):
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def _gelu_and_mul_forward_spyre(self, x):
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d]) * x[..., d:]


def _new_gelu_forward_spyre(self, x):
    """Port of spyre-inference #636; torch.pow(x, 3) is incorrect on Spyre."""
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x * x * x)))


# ---------------------------------------------------------------------------
# RotaryEmbedding uses spyre-inference's device-resident 2x2 formulation. The
# CPU cache is converted once and subsequent position gathers run on Spyre.
# ---------------------------------------------------------------------------
def _rotate_neox_2x2(x, rot, head_size):
    num_tokens = x.shape[0]
    inner = head_size // 2
    x_pairs = x.view(num_tokens, -1, 2, inner)
    out = (rot.unsqueeze(1) * x_pairs.unsqueeze(-3)).sum(dim=-2)
    return out.flatten(-2).view(x.shape)


def _rope_forward_spyre(
    self,
    positions,
    query,
    key=None,
    offsets=None,
    fused_set_kv_buffer_arg=None,
):
    """Device-resident 2x2 RoPE from spyre-inference at 6e5ff996."""
    assert fused_set_kv_buffer_arg is None, "Spyre RoPE: fused KV write unsupported"
    if offsets is not None:
        positions = positions + offsets
    if not (self.is_neox_style and self.rotary_dim == self.head_size):
        raise NotImplementedError("Spyre supports neox-style full rotary only")

    cache = getattr(self, "_spyre_rotation_cache", None)
    if cache is None:
        cos, sin = self.cos_sin_cache.to("cpu").chunk(2, dim=-1)
        cache_cpu = torch.stack([cos, -sin, sin, cos], dim=1).flatten(1).contiguous()
        cache = cache_cpu.to(device=query.device, dtype=query.dtype)
        self._spyre_rotation_cache = cache
    pos = positions.flatten()
    if pos.device.type != query.device.type:
        pos = pos.to(query.device)
    rot = cache.index_select(0, pos).view(-1, 2, 2, self.head_size // 2)
    out_query = _rotate_neox_2x2(query, rot, self.head_size)
    out_key = _rotate_neox_2x2(key, rot, self.head_size) if key is not None else None
    return out_query, out_key


# ---------------------------------------------------------------------------
# Registration. Import is the side effect (called from register.py).
# Each registration is guarded so a class that doesn't exist in this SGLang
# version is skipped, not fatal.
# ---------------------------------------------------------------------------
def register_spyre_ops() -> list[str]:
    registered: list[str] = []

    def _reg(module_path, cls_name, fn):
        try:
            mod = __import__(module_path, fromlist=[cls_name])
            op_cls = getattr(mod, cls_name)
            MultiPlatformOp.register_oot_forward(op_cls, fn, SPYRE_KEY)
            registered.append(cls_name)
        except Exception:
            pass

    # RMSNorm family (on-device, no fp32 promotion)
    _reg("sglang.srt.layers.layernorm", "RMSNorm", _rmsnorm_forward_spyre)
    _reg("sglang.srt.layers.layernorm", "GemmaRMSNorm", _rmsnorm_forward_spyre)
    # Activations (on-device)
    _reg("sglang.srt.layers.activation", "SiluAndMul", _silu_and_mul_forward_spyre)
    _reg("sglang.srt.layers.activation", "GeluAndMul", _gelu_and_mul_forward_spyre)
    _reg("sglang.srt.layers.activation", "NewGELU", _new_gelu_forward_spyre)
    # MultiPlatformOp dispatch keys on the concrete class name.
    _reg(
        "sglang.srt.layers.rotary_embedding.base",
        "RotaryEmbedding",
        _rope_forward_spyre,
    )
    _reg(
        "sglang.srt.layers.rotary_embedding.rope_variant",
        "Llama3RotaryEmbedding",
        _rope_forward_spyre,
    )
    _reg(
        "sglang.srt.layers.rotary_embedding.yarn",
        "YaRNScalingRotaryEmbedding",
        _rope_forward_spyre,
    )

    return registered


__all__ = ["register_spyre_ops", "SPYRE_KEY"]
