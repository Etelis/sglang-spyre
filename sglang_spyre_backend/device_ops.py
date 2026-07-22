"""Register Spyre on-device op implementations onto SGLang's MultiPlatformOp surface.

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

Provenance: github.com/torch-spyre/spyre-inference (Apache-2.0) @ 29faeb2.
Each op below carries a ``Source:`` link to the exact file/lines it was adapted
from. spyre-inference itself adapts the upstream vLLM/torch ops.

These are deliberately thin: on a real ``device="spyre"`` tensor most of this is
plain torch that ``torch_spyre`` lowers via its eager kernels; the value here is
(a) matching SGLang's op signatures and (b) carrying spyre-inference's device
quirk workarounds (e.g. slicing Spyre tensors corrupts memory -> slice on CPU).
"""

from __future__ import annotations

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

    # Spyre cannot reduce fp32 (the hardware rejects mean/sum on IEEE_FP32, and
    # torch_spyre's compiled `mean` lowering may upcast). Granite keeps an fp32
    # residual/hidden, so on-device we compute the variance in bf16.
    if x.device.type == "spyre" and x.dtype == torch.float32:
        x = x.to(torch.bfloat16)
    eps = torch.full(x.shape, self.variance_epsilon, dtype=x.dtype, device=x.device)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    if getattr(self, "has_weight", True):
        x = x * self.weight.to(x.dtype)

    if residual_out is None:
        return x
    return x, residual_out


# ---------------------------------------------------------------------------
# SiluAndMul / GeluAndMul — reuses spyre-inference/custom_ops/silu_and_mul.py.
# Carry its workaround: slicing a Spyre tensor corrupts memory, so slice on CPU
# then move the contiguous halves back to the device.
# Source: torch-spyre/spyre-inference @ 29faeb2 · custom_ops/silu_and_mul.py
#   forward_oot() — slice-on-CPU then F.silu(x1)*x2 at L87-94:
#   https://github.com/torch-spyre/spyre-inference/blob/29faeb240801e7793c397ffa80422e0629e80aa9/spyre_inference/custom_ops/silu_and_mul.py#L64-L94
# (GeluAndMul has no spyre-inference original — same workaround, F.gelu swapped in.)
# ---------------------------------------------------------------------------
def _silu_and_mul_forward_spyre(self, x):
    dev = x.device
    if dev.type == "spyre":
        x = x.to("cpu")
    d = x.shape[-1] // 2
    x1 = x[..., :d].contiguous().to(dev)
    x2 = x[..., d:].contiguous().to(dev)
    return F.silu(x1) * x2


def _gelu_and_mul_forward_spyre(self, x):
    dev = x.device
    if dev.type == "spyre":
        x = x.to("cpu")
    d = x.shape[-1] // 2
    x1 = x[..., :d].contiguous().to(dev)
    x2 = x[..., d:].contiguous().to(dev)
    return F.gelu(x1) * x2


# ---------------------------------------------------------------------------
# RotaryEmbedding — CPU-assisted (reuses spyre-inference/custom_ops/rotary_embedding.py).
# RoPE's inv_freq / cos-sin cache uses a division that crashes torch_spyre's
# dxp_standalone compiler when built on-device, so RoPE runs CPU-side: move
# positions/q/k to CPU, call SGLang's own forward_native, move q/k back. RoPE is
# a small op; the round-trip is cheap relative to the linear/MLP that stay
# on-device. The cos_sin_cache is forced to CPU by the OOT patch in
# rotary_embedding/base.py (init_device="cpu" for OOT).
# Source: torch-spyre/spyre-inference @ 29faeb2 · custom_ops/rotary_embedding.py
#   forward() — to-CPU -> forward_native -> back at L46-73:
#   https://github.com/torch-spyre/spyre-inference/blob/29faeb240801e7793c397ffa80422e0629e80aa9/spyre_inference/custom_ops/rotary_embedding.py#L46-L73
# ---------------------------------------------------------------------------
def _rope_forward_spyre(self, positions, query, key=None, offsets=None,
                        fused_set_kv_buffer_arg=None):
    # fused KV-buffer path is a CUDA-only optimization; must be absent on Spyre.
    assert fused_set_kv_buffer_arg is None, "Spyre RoPE: fused_set_kv_buffer unsupported"
    qdev, qdt = query.device, query.dtype
    pos_c = positions.to("cpu")
    q_c = query.to("cpu")
    k_c = key.to("cpu") if key is not None else None
    if offsets is not None:
        out = self.forward_native(pos_c, q_c, k_c, offsets=offsets.to("cpu"))
    else:
        out = self.forward_native(pos_c, q_c, k_c)
    if isinstance(out, tuple):
        rq, rk = out
        rq = rq.to(device=qdev, dtype=qdt)
        rk = rk.to(device=qdev, dtype=qdt) if rk is not None else None
        return rq, rk
    return out.to(device=qdev, dtype=qdt)


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
    # Activations (on-device, slice-on-cpu workaround)
    _reg("sglang.srt.layers.activation", "SiluAndMul", _silu_and_mul_forward_spyre)
    _reg("sglang.srt.layers.activation", "GeluAndMul", _gelu_and_mul_forward_spyre)
    # RoPE (CPU-assisted). Only the base RotaryEmbedding is registered.
    # MultiPlatformOp keys on exact type, so any scaling subclass (linear/
    # dynamic/YaRN) needs its own _reg entry or it silently falls back to
    # forward_native on a Spyre tensor. Add them here as models require.
    _reg("sglang.srt.layers.rotary_embedding.base", "RotaryEmbedding", _rope_forward_spyre)

    return registered


__all__ = ["register_spyre_ops", "SPYRE_KEY"]
