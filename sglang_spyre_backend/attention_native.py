"""SpyreAttnBackend — Spyre attention as a thin subclass of SGLang's
TorchNativeAttnBackend.

Reuse strategy (the whole point of this module):
    SGLang's ``TorchNativeAttnBackend`` already implements the per-sequence
    KV gather, the save-into-pool, the GQA/causal/cross-attention plumbing,
    and the ``forward_extend`` / ``forward_decode`` entry points
    (sglang/srt/layers/attention/torch_native_backend.py). The ONLY thing
    that is Spyre-specific is the inner attention math: instead of
    ``torch.nn.functional.scaled_dot_product_attention`` running on CPU, we
    bucket the shapes to what Spyre's Inductor lowering accepts, build the
    additive 4D mask, move q/k/v to ``torch.device("spyre")`` and run the
    compiled ``_attn_4d`` kernel.

    So we inherit everything and override only the single per-request compute
    seam. The previous ``SimpleSpyreAttentionBackend`` re-implemented the KV
    gather, the pool save, the seq-len bookkeeping and the 2D<->3D reshapes
    that the base class already does — all of that is deleted here in favour
    of the base implementation.

The ``_attn_4d`` kernel and the bucketing constants are imported from
spyre-inference (single canonical source). Until that import path is
published (Phase B), we fall back to a local copy guarded by ImportError.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

# ---------------------------------------------------------------------------
# Spyre kernel + bucketing constants.
# Phase B will replace this block with a direct import of the vLLM-free
# ``spyre_inference.kernels`` module so the kernel lives in exactly one place.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only where spyre-inference is installed
    from spyre_inference.kernels import (  # type: ignore
        _attn_4d,
        KV_LENGTH_ALIGNMENT,
        QUERY_CHUNK_SIZE,
    )
except Exception:  # spyre-inference not importable (no vLLM-free module yet)
    # Matches spyre-inference's spyre_attn.py exactly.
    KV_LENGTH_ALIGNMENT = 256
    QUERY_CHUNK_SIZE = 32

    def _attn_4d(q, k, v, scale, mask):
        """The compiled Spyre kernel (q@kᵀ, scale, +mask, softmax, p@v).

        Identical to spyre-inference's ``_attn_4d``. torch_spyre applies its
        per-op compile internally, so this stays eager here.
        """
        scores = q @ k.transpose(-2, -1)
        scores = scores * scale
        scores = scores + mask
        p = scores.softmax(dim=-1)
        return p @ v


def _ceil_to(x: int, mult: int) -> int:
    return (x + mult - 1) // mult * mult


class SpyreAttnBackend(TorchNativeAttnBackend):
    """Spyre attention: reuse TorchNativeAttnBackend, swap only the SDPA seam.

    Registered as ``"spyre"`` via ``SpyreSRTPlatform.get_default_attention_backend``.

    Constraints (same as spyre-inference): BF16 compute, single sequence
    (``num_seqs == 1``), static shapes via bucketing. KV cache stays in
    SGLang's pool (CPU for now); Phase C moves it onto the device.
    """

    def __init__(self, model_runner):
        super().__init__(model_runner)
        self._target_device = torch.device("spyre")
        self._target_dtype = torch.bfloat16

    # The KV pool lives on CPU (device="cpu"); with the body-on-device wrapper k/v
    # arrive on Spyre, so move them to CPU before the base saves them into the pool
    # (and the gather reads CPU then moves to Spyre). No-op when k/v are already CPU.
    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if k is not None:
            k = k.to("cpu")
        if v is not None:
            v = v.to("cpu")
        return super().forward_extend(q, k, v, layer, forward_batch, save_kv_cache)

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if k is not None:
            k = k.to("cpu")
        if v is not None:
            v = v.to("cpu")
        return super().forward_decode(q, k, v, layer, forward_batch, save_kv_cache)

    # ------------------------------------------------------------------
    # The only Spyre-specific seam. The base class calls these with the
    # already-gathered per-request query and the KV cache buffers; we run
    # the bucketed _attn_4d on Spyre instead of CPU SDPA, writing results
    # back into ``output`` in place (same contract as the base methods).
    # ------------------------------------------------------------------
    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        encoder_lens=None,
        scaling=None,
        enable_gqa=False,
        causal=False,
        is_cross_attn=False,
        sliding_window_size=None,
    ):
        return self._spyre_sdpa(
            query=query,
            output=output,
            k_cache=k_cache,
            v_cache=v_cache,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            scaling=scaling,
            causal=causal,
        )

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens=None,
        scaling=None,
        enable_gqa=False,
        causal=False,
        is_cross_attn=False,
        sliding_window_size=None,
    ):
        # Decode: one query token per sequence; whole history is "prefix".
        extend_seq_lens = torch.ones_like(seq_lens)
        extend_prefix_lens = seq_lens - extend_seq_lens
        return self._spyre_sdpa(
            query=query,
            output=output,
            k_cache=k_cache,
            v_cache=v_cache,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            scaling=scaling,
            causal=False,
        )

    # ------------------------------------------------------------------
    def _spyre_sdpa(
        self,
        *,
        query,            # [num_tokens, num_q_heads, head_dim]
        output,           # [num_tokens, num_q_heads, head_dim] (written in place)
        k_cache,          # [N, num_kv_heads, head_dim]
        v_cache,          # [N, num_kv_heads, head_dim]
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_prefix_lens,
        extend_seq_lens,
        scaling,
        causal,
    ):
        """Bucketed Spyre attention for a single sequence.

        Mirrors spyre-inference's _compute_attention shape conventions, but
        sources the gathered KV from SGLang's pool via req_to_token (the same
        indices the base TorchNativeAttnBackend uses).
        """
        num_seqs = seq_lens.shape[0]
        assert num_seqs == 1, (
            f"SpyreAttnBackend supports num_seqs == 1 only (got {num_seqs}); "
            "set --max-running-requests 1."
        )

        num_q_heads, head_dim = query.shape[1], query.shape[2]
        num_kv_heads = k_cache.shape[1]
        num_queries_per_kv = num_q_heads // num_kv_heads

        seq_len = int(seq_lens[0].item())
        q_len = int(extend_seq_lens[0].item())
        req_idx = int(req_pool_indices[0].item())

        # Gather this sequence's KV slice from the pool (CPU indices).
        kv_slots = req_to_token[req_idx, :seq_len].to(torch.long)
        k_seq = k_cache[kv_slots]  # [seq_len, num_kv_heads, head_dim]
        v_seq = v_cache[kv_slots]

        # Bucket KV length up to KV_LENGTH_ALIGNMENT, pad with zeros.
        aligned_kv = _ceil_to(seq_len, KV_LENGTH_ALIGNMENT)
        if aligned_kv > seq_len:
            pad = aligned_kv - seq_len
            k_seq = torch.nn.functional.pad(k_seq, (0, 0, 0, 0, 0, pad))
            v_seq = torch.nn.functional.pad(v_seq, (0, 0, 0, 0, 0, pad))

        # [aligned_kv, H_kv, D] -> [H_kv, aligned_kv, D] -> [H_kv, 1, aligned_kv, D]
        k_c = k_seq.transpose(0, 1).contiguous().reshape(num_kv_heads, 1, aligned_kv, head_dim)
        v_c = v_seq.transpose(0, 1).contiguous().reshape(num_kv_heads, 1, aligned_kv, head_dim)

        # Query for this request, padded to QUERY_CHUNK_SIZE.
        q_seq = query[:q_len]  # [q_len, num_q_heads, head_dim]
        padded_q = _ceil_to(q_len, QUERY_CHUNK_SIZE)
        # [q_len, H_q, D] -> [H_q, q_len, D] -> [H_kv, qpk, q_len, D]
        q_h = q_seq.transpose(0, 1).contiguous()
        if padded_q > q_len:
            q_h = torch.nn.functional.pad(q_h, (0, 0, 0, padded_q - q_len))
        q_b = q_h.reshape(num_kv_heads, num_queries_per_kv, padded_q, head_dim)

        mask = self._build_mask(
            seq_len=seq_len,
            q_len=q_len,
            padded_q=padded_q,
            aligned_kv=aligned_kv,
            num_kv_heads=num_kv_heads,
            apply_causal=causal and q_len > 1,
        )

        q_dev = q_b.to(self._target_dtype).to(self._target_device).contiguous()
        k_dev = k_c.to(self._target_dtype).to(self._target_device).contiguous()
        v_dev = v_c.to(self._target_dtype).to(self._target_device).contiguous()

        out_dev = _attn_4d(q_dev, k_dev, v_dev, scaling, mask)
        # Write back on the SAME device as `output` (= query's device). With the
        # device="cpu" model that's CPU; with the body-on-device wrapper the query
        # arrives on Spyre, so the result stays resident on the card.
        out = out_dev.to(device=output.device, dtype=query.dtype)
        out = out.reshape(num_q_heads, padded_q, head_dim)[:, :q_len, :]
        output[:q_len] = out.transpose(0, 1).contiguous()
        return output

    def _build_mask(
        self, *, seq_len, q_len, padded_q, aligned_kv, num_kv_heads, apply_causal
    ):
        """Additive 4D mask, shape [num_kv_heads, 1, padded_q, aligned_kv].

        Mirrors spyre-inference._build_attention_mask: pad/causal positions
        get -inf, valid positions get 0.
        """
        q_pos = torch.arange(padded_q)
        kv_pos = torch.arange(aligned_kv)
        q_valid = q_pos.unsqueeze(1) < q_len           # real query rows
        kv_valid = kv_pos.unsqueeze(0) < seq_len       # real kv cols
        attend = q_valid & kv_valid                    # [padded_q, aligned_kv]

        if apply_causal:
            context = seq_len - q_len
            causal_limit = context + q_pos.unsqueeze(1)  # [padded_q, 1]
            attend = attend & (kv_pos.unsqueeze(0) <= causal_limit)

        neg_inf = torch.finfo(self._target_dtype).min
        additive = torch.where(
            attend,
            torch.tensor(0.0, dtype=self._target_dtype),
            torch.tensor(neg_inf, dtype=self._target_dtype),
        )
        mask = additive.reshape(1, 1, padded_q, aligned_kv).expand(
            num_kv_heads, 1, padded_q, aligned_kv
        )
        return mask.contiguous().to(self._target_device)
