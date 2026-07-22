"""Body-on-device wrapper for Spyre — the reuse-aligned alternative to device="spyre".

Mirrors spyre-inference's _SpyreModelWrapper. SGLang keeps device="cpu", so the KV
pool, RoPE cos_sin_cache, page tables, sampling indices and all scheduling stay on
CPU (where they work — no uint64/stick/Parameter issues). We move ONLY the model
*body* to Spyre and cross the CPU<->Spyre boundary once per forward:

  input_ids (CPU) -> embed_tokens (CPU) --hook--> hidden on Spyre
                  -> [decoder layers on Spyre: RMSNorm/linear/MLP fire forward_spyre,
                      residual stream stays resident; attention backend handles its
                      own KV transfer; RoPE is CPU-assisted]
                  -> final norm (Spyre) --hook--> hidden on CPU -> lm_head (CPU)

Kept on CPU deliberately: embed_tokens (int index lookup), lm_head + sampling,
and every rotary-embedding module (its cache/arange can't run on Spyre).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

SPYRE = torch.device("spyre")
CPU = torch.device("cpu")


def _move(obj, dev):
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    if isinstance(obj, (tuple, list)):
        return type(obj)(_move(x, dev) for x in obj)
    return obj


def _find(model, *paths):
    for p in paths:
        cur, ok = model, True
        for part in p.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and cur is not model:
            return cur
    return None


def bodify_model_for_spyre(model):
    """Move the model body onto Spyre, keep boundaries on CPU, install hooks."""
    # 1. Everything to Spyre, then pull back what must stay on CPU.
    model.to(SPYRE)
    for _name, m in model.named_modules():
        if "rotary" in type(m).__name__.lower():
            m.to(CPU)  # RoPE cache/arange can't run on Spyre

    embed = _find(model, "model.embed_tokens", "transformer.embed_tokens", "embed_tokens")
    lm_head = _find(model, "lm_head")
    if embed is not None:
        embed.to(CPU)
    if lm_head is not None:
        lm_head.to(CPU)

    # 2. Re-dispatch MultiPlatformOps. dispatch_forward() is resolved at op
    # __init__ (model-build time); ops built before register_spyre_ops() populated
    # the OOT registry cached forward_native (which upcasts to fp32 — fatal on
    # Spyre). Re-resolve now so RMSNorm/SiLU/RoPE use their forward_spyre bodies.
    try:
        import sglang_spyre_backend.register  # noqa: F401  (runs register_spyre_ops)
    except Exception:
        pass
    redispatched = 0
    failed = 0
    for m in model.modules():
        if hasattr(m, "dispatch_forward") and hasattr(m, "_forward_method"):
            try:
                m._forward_method = m.dispatch_forward()
                redispatched += 1
            except Exception:
                failed += 1
    logger.info(
        "bodify: re-dispatched %d MultiPlatformOps (%d failed)",
        redispatched, failed,
    )

    # 3. Boundary hooks: embed output -> Spyre, body output -> CPU.
    if embed is not None:
        embed.register_forward_hook(lambda mod, inp, out: _move(out, SPYRE))
    body = _find(model, "model", "transformer")
    if body is not None:
        body.register_forward_hook(lambda mod, inp, out: _move(out, CPU))

    return model
