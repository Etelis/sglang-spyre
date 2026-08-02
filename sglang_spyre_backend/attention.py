"""SpyreAttentionBackend — BMM-based paged attention for Spyre AIU.

Ported from spyre_attn_exp.py in spyre-inference, adapted to SGLang's
AttentionBackend interface.

Key design decisions (unchanged from spyre-inference):
- No advanced tensor indexing on Spyre: scatter/gather done via BMM
  against one-hot selection masks, built on CPU and transferred.
- KV cache lives on Spyre device as a flat [num_kv_heads,
  aligned_num_physical_blocks, block_width] tensor per layer.
- Attention computation runs compiled on Spyre (_attn_4d).
- max_num_seqs=1 constraint: single-sequence indexing only.

References:
  - spyre-inference/spyre_inference/v1/attention/backends/spyre_attn_exp.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Alignment constants (must match spyre-inference)
KV_LENGTH_ALIGNMENT = 256
QUERY_CHUNK_SIZE = 32
BLOCK_ALIGN = 64  # Physical blocks must be multiple of 64 for stick alignment


def _attn_4d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Batched 4D attention with additive mask."""
    scores = q @ k.transpose(-2, -1)
    scores = scores * scale
    scores = scores + mask
    p = scores.softmax(dim=-1)
    return p @ v


def _scatter_cache(cache, mask, values):
    """Elementwise scatter: update cache positions indicated by mask."""
    return cache * (1.0 - mask) + values


class SpyreAttentionState:
    """Per-forward-pass metadata for a single layer."""

    def __init__(self):
        self.attention_mask: Optional[torch.Tensor] = None
        self.gather_sel_mask_dev: Optional[torch.Tensor] = None
        self.aligned_max_seq_len: int = 0
        self.scatter_mask_dev: Optional[torch.Tensor] = None
        self.scatter_row_sel_dev: Optional[torch.Tensor] = None
        self.scatter_col_sel_dev: Optional[torch.Tensor] = None
        self.num_actual_tokens: int = 0
        self.apply_causal_mask: bool = False


class SpyreAttentionBackend(AttentionBackend):
    """SGLang attention backend for Spyre AIU using BMM-based paged KV cache.

    Registered as "spyre" via SpyreSRTPlatform.get_default_attention_backend().
    """

    # No CUDA graph support
    needs_cpu_seq_lens: bool = True

    def __init__(self, model_runner):
        self.model_runner = model_runner
        self._target_device = torch.device("spyre")
        self._target_dtype = torch.bfloat16

        # Per-layer on-device KV caches, keyed by layer_id
        self._k_cache: dict[int, torch.Tensor] = {}
        self._v_cache: dict[int, torch.Tensor] = {}
        self._cache_initialized: set[int] = set()

        # Current forward metadata (built once per step, shared across layers)
        self._state = SpyreAttentionState()

        # Attention kernel. torch.compile is intentionally NOT applied yet:
        # Spyre compilation of _attn_4d isn't stable, so we run it eagerly.
        # To re-enable once stable: self._attn_4d = torch.compile(_attn_4d, dynamic=False)
        self._attn_4d = _attn_4d

    # ------------------------------------------------------------------
    # Forward metadata (called once per forward pass, shared across layers)
    # ------------------------------------------------------------------

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Build all per-step metadata on CPU, transfer masks to Spyre."""
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        state = self._state

        seq_lens = forward_batch.seq_lens_cpu  # CPU tensor
        max_seq_len = int(seq_lens.max().item()) if len(seq_lens) > 0 else 0
        max_query_len = 1 if forward_batch.forward_mode.is_decode() else max_seq_len

        # num_actual_tokens = tokens actually carried by q/k/v this step.
        # Decode: 1 per sequence. Extend: only the *newly extended* tokens —
        # not seq_lens.sum(), which is the full sequence length. The two are
        # equal on a cold prefill but diverge on any RadixCache prefix hit,
        # where the cached prefix is already in the pool and q holds just the
        # uncached tail.
        if forward_batch.forward_mode.is_decode():
            state.num_actual_tokens = forward_batch.batch_size
        elif forward_batch.extend_num_tokens is not None:
            state.num_actual_tokens = int(forward_batch.extend_num_tokens)
        else:
            state.num_actual_tokens = int(seq_lens.sum().item())

        state.apply_causal_mask = (
            forward_batch.forward_mode.is_extend() and max_query_len > 1
        )

        # Aligned max sequence length (buckets recompilation)
        state.aligned_max_seq_len = (
            (max_seq_len + KV_LENGTH_ALIGNMENT - 1)
            // KV_LENGTH_ALIGNMENT
            * KV_LENGTH_ALIGNMENT
        )

        # We only support batch_size=1 for now
        if forward_batch.batch_size > 1:
            raise ValueError(
                "SpyreAttentionBackend: max_num_seqs=1 constraint. "
                f"Got batch_size={forward_batch.batch_size}."
            )

        # Build attention mask
        query_start_loc = torch.tensor([0, max_query_len], dtype=torch.int32)
        state.attention_mask = self._build_attention_mask(
            seq_lens,
            query_start_loc,
            state.apply_causal_mask,
            max_query_len,
            state.aligned_max_seq_len,
        )

        # Build gather selection mask. SGLang's req_to_token is a token->slot map
        # of shape [num_reqs+1, max_context_len] that lives on req_to_token_pool
        # (NOT on forward_batch). Take this request's per-token slot row; the mask
        # builder derives the logical-page->physical-page table from it.
        num_kv_heads = self._get_num_kv_heads()
        req_idx = int(forward_batch.req_pool_indices[0].item())
        seq_len0 = int(seq_lens[0].item())
        token_slots = self.model_runner.req_to_token_pool.req_to_token[
            req_idx, :seq_len0
        ].to("cpu")
        if num_kv_heads > 0:
            state.gather_sel_mask_dev = self._build_gather_sel_mask(
                token_slots,
                seq_len0,
                state.aligned_max_seq_len,
                num_kv_heads,
            )

        # Build scatter masks from slot mapping
        slot_mapping = forward_batch.out_cache_loc  # [num_actual_tokens]
        if slot_mapping is not None and num_kv_heads > 0:
            (
                state.scatter_mask_dev,
                state.scatter_row_sel_dev,
                state.scatter_col_sel_dev,
            ) = self._build_scatter_masks(slot_mapping, num_kv_heads)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # No graph capture on Spyre
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        return self._forward_impl(q, k, v, layer, forward_batch, save_kv_cache)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        return self._forward_impl(q, k, v, layer, forward_batch, save_kv_cache)

    def _forward_impl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
    ) -> torch.Tensor:
        layer_id = layer.layer_id
        num_kv_heads = layer.tp_k_head_num
        head_size = layer.head_dim
        num_heads = layer.tp_q_head_num
        scale = layer.scaling

        self._ensure_cache(layer_id, num_kv_heads, head_size)

        state = self._state
        num_tokens = state.num_actual_tokens

        # q/k/v arrive FLATTENED as [num_tokens, n_heads*head_size] (SGLang passes
        # the raw projection output; the dense path's base forward_extend does this
        # view itself). View to 3D [num_tokens, n_heads, head_size] for the bmm
        # scatter and the per-head reshape below.
        q_cpu = q[:num_tokens].contiguous().view(num_tokens, num_heads, head_size)
        k_cpu = k[:num_tokens].contiguous().view(num_tokens, num_kv_heads, head_size)
        v_cpu = v[:num_tokens].contiguous().view(num_tokens, num_kv_heads, head_size)

        # Step 1: Scatter new KV tokens into on-device cache
        if save_kv_cache and state.scatter_mask_dev is not None:
            self._scatter_to_device_cache(
                layer_id,
                k_cpu,
                v_cpu,
                state.scatter_mask_dev,
                state.scatter_row_sel_dev,
                state.scatter_col_sel_dev,
            )

        # Step 2: Gather dense KV from on-device cache
        compact_k, compact_v = self._gather_from_device_cache(
            layer_id,
            state.gather_sel_mask_dev,
            state.aligned_max_seq_len,
            num_kv_heads,
            head_size,
        )

        # Step 3: Reshape query and run attention
        # query arrives as [num_tokens, num_heads, head_size] on CPU.
        # Following spyre-inference convention, treat batch=1 (max_num_seqs=1).
        # Reshape to [num_seqs, max_query_len, num_heads, head_size] = [1, num_tokens, ...]
        num_queries_per_kv = num_heads // num_kv_heads
        max_query_len = q_cpu.shape[0]
        padded_query_len = (
            (max_query_len + QUERY_CHUNK_SIZE - 1) // QUERY_CHUNK_SIZE * QUERY_CHUNK_SIZE
        )

        # [1, max_query_len, num_heads, head_size]
        query = q_cpu.unsqueeze(0).contiguous()
        if padded_query_len > max_query_len:
            query = torch.nn.functional.pad(
                query, (0, 0, 0, 0, 0, padded_query_len - max_query_len),
                mode="constant", value=0.0,
            )

        # [1, num_heads, padded_q, head_size] -> [1*num_kv_heads, num_queries_per_kv, padded_q, head_size]
        q = query.transpose(1, 2).contiguous()
        q = q.reshape(1 * num_kv_heads, num_queries_per_kv, padded_query_len, head_size)

        # compact_k/v arrive as [1, num_kv_heads, aligned_max_seq_len, head_size]
        # reshape to [1*num_kv_heads, 1, aligned_max_seq_len, head_size] for broadcast
        kv_len = compact_k.shape[2]
        k_4d = compact_k.reshape(1 * num_kv_heads, 1, kv_len, head_size)
        v_4d = compact_v.reshape(1 * num_kv_heads, 1, kv_len, head_size)

        # Move query to Spyre
        q_dev = q.to(self._target_dtype).to(self._target_device)

        mask = state.attention_mask  # on Spyre already, shape [1*num_kv_heads, 1, padded_q, kv_len]

        attn_out_dev = self._attn_4d(q_dev, k_4d, v_4d, scale, mask)
        # attn_out_dev: [1*num_kv_heads, num_queries_per_kv, padded_q, head_size]

        # Reshape back to [1, num_heads, padded_q, head_size] and to CPU
        attn_out = (
            attn_out_dev
            .reshape(1, num_heads, padded_query_len, head_size)
            .transpose(1, 2)         # [1, padded_q, num_heads, head_size]
            .contiguous()
            .to(device="cpu", dtype=q_cpu.dtype)
        )
        attn_out = attn_out[0, :num_tokens]  # [num_tokens, num_heads, head_size]
        return attn_out.reshape(num_tokens, num_heads * head_size)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _ensure_cache(self, layer_id: int, num_kv_heads: int, head_size: int):
        if layer_id in self._cache_initialized:
            return
        block_size = self.model_runner.token_to_kv_pool.page_size
        # Physical axis is PAGES, not tokens (pool.size is a token count).
        num_pages = self.model_runner.token_to_kv_pool.size // block_size
        aligned_blocks = ((num_pages + BLOCK_ALIGN - 1) // BLOCK_ALIGN) * BLOCK_ALIGN
        block_width = block_size * head_size

        self._k_cache[layer_id] = torch.zeros(
            num_kv_heads, aligned_blocks, block_width,
            dtype=self._target_dtype,
            device=self._target_device,
        )
        self._v_cache[layer_id] = torch.zeros(
            num_kv_heads, aligned_blocks, block_width,
            dtype=self._target_dtype,
            device=self._target_device,
        )
        self._cache_initialized.add(layer_id)

    def _scatter_to_device_cache(
        self,
        layer_id: int,
        k_cpu: torch.Tensor,
        v_cpu: torch.Tensor,
        scatter_mask_dev: torch.Tensor,
        row_sel_dev: torch.Tensor,
        col_sel_dev: torch.Tensor,
    ):
        """Two-BMM scatter: place K/V into the on-device cache."""
        k_dev = k_cpu.to(self._target_dtype).to(self._target_device).contiguous()
        v_dev = v_cpu.to(self._target_dtype).to(self._target_device).contiguous()

        # BMM 1: spread each token's head_size-vector into block_width slot
        # [num_tokens, num_kv_heads, head_size] @ [num_tokens, head_size, block_width]
        # -> [num_tokens, num_kv_heads, block_width]
        k_spread = torch.bmm(k_dev, col_sel_dev)
        v_spread = torch.bmm(v_dev, col_sel_dev)

        # Permute to [num_kv_heads, num_tokens, block_width]
        k_spread = k_spread.transpose(0, 1).contiguous()
        v_spread = v_spread.transpose(0, 1).contiguous()

        # BMM 2: route each token row to its physical block
        # [num_kv_heads, aligned_blocks, num_tokens] @ [num_kv_heads, num_tokens, block_width]
        # -> [num_kv_heads, aligned_blocks, block_width]
        k_vals = torch.bmm(row_sel_dev, k_spread)
        v_vals = torch.bmm(row_sel_dev, v_spread)

        self._k_cache[layer_id] = _scatter_cache(
            self._k_cache[layer_id], scatter_mask_dev, k_vals
        )
        self._v_cache[layer_id] = _scatter_cache(
            self._v_cache[layer_id], scatter_mask_dev, v_vals
        )

    def _gather_from_device_cache(
        self,
        layer_id: int,
        sel_mask_dev: torch.Tensor,
        aligned_max_seq_len: int,
        num_kv_heads: int,
        head_size: int,
    ):
        """BMM gather: assemble dense KV from paged cache."""
        # sel_mask_dev: [num_kv_heads, aligned_logical_blocks, aligned_physical_blocks]
        # cache:        [num_kv_heads, aligned_physical_blocks, block_width]
        # result:       [num_kv_heads, aligned_logical_blocks, block_width]
        gathered_k = torch.bmm(sel_mask_dev, self._k_cache[layer_id])
        gathered_v = torch.bmm(sel_mask_dev, self._v_cache[layer_id])

        # Reshape: [num_kv_heads, seq_len, head_size]
        gathered_k = gathered_k.reshape(num_kv_heads, aligned_max_seq_len, head_size)
        gathered_v = gathered_v.reshape(num_kv_heads, aligned_max_seq_len, head_size)

        # Add batch dim for attention: [1, num_kv_heads, seq_len, head_size]
        return gathered_k.unsqueeze(0), gathered_v.unsqueeze(0)

    # ------------------------------------------------------------------
    # Mask builders (CPU → Spyre)
    # ------------------------------------------------------------------

    def _build_attention_mask(
        self,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        apply_causal_mask: bool,
        max_query_len: int,
        aligned_max_seq_len: int,
    ) -> torch.Tensor:
        num_kv_heads = self._get_num_kv_heads()
        num_seqs = len(seq_lens)
        query_lens = query_start_loc[1:] - query_start_loc[:-1]

        padded_query_len = (
            (max_query_len + QUERY_CHUNK_SIZE - 1) // QUERY_CHUNK_SIZE * QUERY_CHUNK_SIZE
        )

        q_pos = torch.arange(max_query_len)
        kv_pos = torch.arange(aligned_max_seq_len)

        q_valid = q_pos.unsqueeze(0) < query_lens.unsqueeze(1)   # [S, Qmax]
        kv_valid = kv_pos.unsqueeze(0) < seq_lens.unsqueeze(1)   # [S, KVmax]
        attend = q_valid.unsqueeze(2) & kv_valid.unsqueeze(1)    # [S, Qmax, KVmax]

        if apply_causal_mask:
            context_lens = seq_lens - query_lens
            causal_limit = (
                (context_lens.unsqueeze(1) + q_pos.unsqueeze(0)).unsqueeze(2)
            )
            attend = attend & (kv_pos.view(1, 1, -1) <= causal_limit)

        mask_bool = ~attend

        if padded_query_len > max_query_len:
            pad = torch.ones(
                num_seqs, padded_query_len - max_query_len, aligned_max_seq_len,
                dtype=torch.bool,
            )
            mask_bool = torch.cat([mask_bool, pad], dim=1)

        neg_inf = torch.finfo(self._target_dtype).min
        mask = torch.where(
            mask_bool,
            torch.tensor(neg_inf, dtype=self._target_dtype),
            torch.tensor(0.0, dtype=self._target_dtype),
        )

        # [num_seqs * num_kv_heads, 1, padded_q, aligned_kv]
        num_queries_per_kv = self._get_num_heads() // num_kv_heads if num_kv_heads > 0 else 1
        mask_4d = (
            mask.unsqueeze(1)
            .expand(-1, num_kv_heads, -1, -1)
            .reshape(num_seqs * num_kv_heads, 1, padded_query_len, aligned_max_seq_len)
            .contiguous()
        )
        return mask_4d.to(device=self._target_device)

    def _build_gather_sel_mask(
        self,
        token_slots: torch.Tensor,   # [seq_len] per-token KV-pool slot ids (CPU)
        seq_len: int,
        aligned_max_seq_len: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        block_size = self.model_runner.token_to_kv_pool.page_size
        # Physical axis is PAGES, not tokens.
        num_pages = self.model_runner.token_to_kv_pool.size // block_size
        aligned_blocks = ((num_pages + BLOCK_ALIGN - 1) // BLOCK_ALIGN) * BLOCK_ALIGN
        aligned_logical = aligned_max_seq_len // block_size

        num_kv_blocks = (seq_len + block_size - 1) // block_size
        sel_2d = torch.zeros(aligned_logical, aligned_blocks, dtype=self._target_dtype)

        # Derive the vLLM-style block table from SGLang's token->slot row. The
        # PagedTokenToKVPoolAllocator lays out each logical page's tokens
        # page-aligned and contiguous (slot = page*block_size + offset), so the
        # first token of logical page L gives physical page = slot // block_size.
        logical = torch.arange(num_kv_blocks)
        physical = (token_slots[::block_size][:num_kv_blocks] // block_size).long()
        sel_2d[logical, physical] = 1.0

        sel_3d = sel_2d.unsqueeze(0).expand(num_kv_heads, -1, -1).contiguous()
        return sel_3d.to(device=self._target_device)

    def _build_scatter_masks(
        self,
        slot_mapping: torch.Tensor,
        num_kv_heads: int,
    ):
        block_size = self.model_runner.token_to_kv_pool.page_size
        head_size = self._get_head_size()
        # Physical axis is PAGES, not tokens.
        num_pages = self.model_runner.token_to_kv_pool.size // block_size
        aligned_blocks = ((num_pages + BLOCK_ALIGN - 1) // BLOCK_ALIGN) * BLOCK_ALIGN
        block_width = block_size * head_size
        num_tokens = len(slot_mapping)

        slot_cpu = slot_mapping.cpu()
        block_indices = slot_cpu // block_size
        block_offsets = slot_cpu % block_size

        d_range = torch.arange(head_size)
        row_idx = block_indices.unsqueeze(1).expand(num_tokens, head_size).reshape(-1)
        col_idx = (
            (block_offsets * head_size).unsqueeze(1) + d_range.unsqueeze(0)
        ).reshape(-1)

        mask = torch.zeros(num_kv_heads, aligned_blocks, block_width, dtype=self._target_dtype)
        mask[:, row_idx, col_idx] = 1.0
        mask_dev = mask.to(device=self._target_device)

        # row_sel: [num_kv_heads, aligned_blocks, num_tokens]
        row_sel_2d = torch.zeros(aligned_blocks, num_tokens, dtype=self._target_dtype)
        row_sel_2d[block_indices.long(), torch.arange(num_tokens)] = 1.0
        row_sel_dev = (
            row_sel_2d.unsqueeze(0).expand(num_kv_heads, -1, -1).contiguous()
            .to(device=self._target_device)
        )

        # col_sel: [num_tokens, head_size, block_width]
        col_sel = torch.zeros(num_tokens, head_size, block_width, dtype=self._target_dtype)
        t_idx = torch.arange(num_tokens).unsqueeze(1).expand(num_tokens, head_size)
        d_idx = d_range.unsqueeze(0).expand(num_tokens, head_size)
        c_idx = block_offsets.long().unsqueeze(1) * head_size + d_idx
        col_sel[t_idx.reshape(-1), d_idx.reshape(-1), c_idx.reshape(-1)] = 1.0
        col_sel_dev = col_sel.to(device=self._target_device)

        return mask_dev, row_sel_dev, col_sel_dev

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_num_kv_heads(self) -> int:
        try:
            return self.model_runner.model_config.get_num_kv_heads(
                self.model_runner.server_args.tp_size
            )
        except Exception:
            return 0

    def _get_num_heads(self) -> int:
        try:
            return self.model_runner.model_config.get_num_attention_heads(
                self.model_runner.server_args.tp_size
            )
        except Exception:
            return 0

    def _get_head_size(self) -> int:
        try:
            return self.model_runner.model_config.head_dim
        except Exception:
            return 128
