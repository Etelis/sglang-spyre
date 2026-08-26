# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SGLang adapter for spyre-inference's dense paged-attention implementation.

The device kernels are ported verbatim in :mod:`spyre_attention_kernel`; this
module only translates SGLang's ``ForwardBatch`` metadata into the tensors that
those kernels consume.  Baseline: spyre-inference 6e5ff996 (2026-08-26).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

from sglang_spyre_backend.spyre_attention_kernel import (
    create_compilable_page_attn,
    maybe_compile,
    reshape_and_cache_kernel,
    slot_major_kv_layout,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


KV_LENGTH_ALIGNMENT = 256
QUERY_CHUNK_SIZE = 32
INT32_ELEMS_PER_STICK = 32


def _ceil_to(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _convert(tensor: torch.Tensor, device: torch.device, dtype=None) -> torch.Tensor:
    """Match spyre-inference's dtype-on-CPU then device conversion contract."""
    target_dtype = tensor.dtype if dtype is None else dtype
    if tensor.device.type == "spyre" and tensor.dtype != target_dtype:
        tensor = tensor.to("cpu")
    if tensor.dtype != target_dtype:
        tensor = tensor.to(dtype=target_dtype)
    if tensor.device.type != device.type:
        tensor = tensor.to(device)
    return tensor


@dataclass
class SpyreAttentionState:
    num_actual_tokens: int = 0
    query_len: int = 0
    seq_len: int = 0
    block_size: int = 0
    aligned_query_len: int = 0
    page_index_table_cpu: torch.Tensor | None = None
    page_index_table_device: torch.Tensor | None = None
    slot_mapping_cpu: torch.Tensor | None = None
    slot_mapping_device: torch.Tensor | None = None
    mask_tiles_cpu: list[torch.Tensor] | None = None
    mask_tiles_device: list[torch.Tensor] | None = None


class SpyreAttentionBackend(AttentionBackend):
    """Latest spyre-inference paged attention behind SGLang's backend API.

    The supported path intentionally has the same current limits as this
    prototype's original paged path: decoder self-attention, one sequence,
    and TP=1.  KV pages remain resident on Spyre and are read through the
    same compiled indirect ``index_select`` loop used by spyre-inference.
    """

    needs_cpu_seq_lens: bool = True

    def __init__(self, model_runner):
        super().__init__()
        self.model_runner = model_runner
        self._target_device = torch.device("spyre")
        self._target_dtype = torch.float16
        self._state = SpyreAttentionState()
        self._k_cache: dict[int, torch.Tensor] = {}
        self._v_cache: dict[int, torch.Tensor] = {}
        self._compile_attn = os.environ.get("SGLANG_SPYRE_COMPILE", "1") != "0"
        # Latest spyre-inference always compiles index_copy_: eager int32
        # indices otherwise fall back to CPU.
        self._reshape_fn = torch.compile(reshape_and_cache_kernel, dynamic=False)
        self._attn_fns: dict[tuple[int, int, str, int], object] = {}

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.batch_size != 1:
            raise ValueError(
                "The SGLang Spyre paged backend currently supports one sequence; "
                f"got batch_size={forward_batch.batch_size}."
            )

        seq_lens = forward_batch.seq_lens_cpu
        if seq_lens is None:
            seq_lens = forward_batch.seq_lens.to("cpu")
        seq_len = int(seq_lens[0].item())

        if forward_batch.forward_mode.is_decode():
            query_len = 1
        else:
            extend_lens = forward_batch.extend_seq_lens_cpu
            if extend_lens is not None:
                query_len = int(extend_lens[0])
            elif forward_batch.extend_seq_lens is not None:
                query_len = int(forward_batch.extend_seq_lens[0].item())
            else:
                query_len = int(forward_batch.extend_num_tokens or seq_len)

        block_size = int(self.model_runner.token_to_kv_pool.page_size)
        if block_size % 64:
            raise ValueError(
                "Spyre paged attention requires page_size to be a multiple of 64; "
                f"got {block_size}."
            )

        state = SpyreAttentionState(
            num_actual_tokens=query_len,
            query_len=query_len,
            seq_len=seq_len,
            block_size=block_size,
            aligned_query_len=(
                1 if query_len == 1 else _ceil_to(query_len, QUERY_CHUNK_SIZE)
            ),
            slot_mapping_cpu=forward_batch.out_cache_loc[:query_len]
            .to(device="cpu", dtype=torch.int32)
            .contiguous(),
        )

        req_idx = int(forward_batch.req_pool_indices[0].item())
        token_slots = self.model_runner.req_to_token_pool.req_to_token[
            req_idx, :seq_len
        ].to("cpu")
        num_blocks = _ceil_to(seq_len, block_size) // block_size
        physical_pages = (token_slots[::block_size][:num_blocks] // block_size).to(
            torch.int32
        )
        page_table = torch.zeros(num_blocks, INT32_ELEMS_PER_STICK, dtype=torch.int32)
        page_table[:, 0] = physical_pages
        state.page_index_table_cpu = page_table
        state.mask_tiles_cpu = self._build_mask_tiles(
            seq_len=seq_len,
            query_len=query_len,
            block_size=block_size,
            aligned_query_len=state.aligned_query_len,
        )
        self._state = state

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

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
        return self._forward_impl(q, k, v, layer, save_kv_cache)

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
        return self._forward_impl(q, k, v, layer, save_kv_cache)

    def _forward_impl(self, q, k, v, layer, save_kv_cache: bool) -> torch.Tensor:
        state = self._state
        num_heads = layer.tp_q_head_num
        num_kv_heads = layer.tp_k_head_num
        head_size = layer.qk_head_dim
        if head_size != layer.v_head_dim:
            raise NotImplementedError(
                "Spyre currently requires equal QK and V head dimensions"
            )
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self._ensure_cache(layer.layer_id, num_kv_heads, head_size, state.block_size)
        k_pages = self._k_cache[layer.layer_id]
        v_pages = self._v_cache[layer.layer_id]
        original_device = q.device

        q_dev = _convert(
            q[: state.num_actual_tokens]
            .contiguous()
            .view(state.num_actual_tokens, num_heads, head_size),
            self._target_device,
            self._target_dtype,
        )
        k_dev = _convert(
            k[: state.num_actual_tokens]
            .contiguous()
            .view(state.num_actual_tokens, num_kv_heads, head_size),
            self._target_device,
            self._target_dtype,
        )
        v_dev = _convert(
            v[: state.num_actual_tokens]
            .contiguous()
            .view(state.num_actual_tokens, num_kv_heads, head_size),
            self._target_device,
            self._target_dtype,
        )

        if state.slot_mapping_device is None:
            assert state.slot_mapping_cpu is not None
            state.slot_mapping_device = _convert(
                state.slot_mapping_cpu, self._target_device
            )
        if state.page_index_table_device is None:
            assert state.page_index_table_cpu is not None
            state.page_index_table_device = _convert(
                state.page_index_table_cpu, self._target_device
            )
        if state.mask_tiles_device is None:
            assert state.mask_tiles_cpu is not None
            state.mask_tiles_device = [
                _convert(tile, self._target_device) for tile in state.mask_tiles_cpu
            ]

        if save_kv_cache:
            slots = (-1, num_kv_heads, head_size)
            self._reshape_fn(
                k_dev,
                v_dev,
                k_pages.view(slots),
                v_pages.view(slots),
                state.slot_mapping_device,
            )

        if state.query_len == 1:
            query = q_dev.unbind(dim=0)[0].reshape(
                num_kv_heads, num_heads // num_kv_heads, 1, head_size
            )
            # Preserve the canonical layout required by torch-spyre on decode.
            query = torch.ops.spyre.opaque_copy_(
                query,
                torch.zeros(
                    query.shape,
                    dtype=query.dtype,
                    device=query.device,
                ),
            )
        else:
            if state.aligned_query_len > state.query_len:
                q_dev = torch.nn.functional.pad(
                    q_dev,
                    (0, 0, 0, 0, 0, state.aligned_query_len - state.query_len),
                )
            query = (
                q_dev.unsqueeze(0)
                .transpose(1, 2)
                .contiguous()
                .reshape(
                    num_kv_heads,
                    num_heads // num_kv_heads,
                    state.aligned_query_len,
                    head_size,
                )
            )

        num_blocks = len(state.mask_tiles_device)
        store_mode = "copy" if self._compile_attn else "none"
        fn = self._get_attn_fn(
            num_blocks,
            state.aligned_query_len,
            num_heads,
            head_size,
            store_mode,
            state.query_len,
        )
        output = torch.empty(
            state.query_len,
            num_heads,
            head_size,
            dtype=self._target_dtype,
            device=self._target_device,
        )
        result = fn(
            query,
            k_pages,
            v_pages,
            state.page_index_table_device,
            state.mask_tiles_device,
            float(layer.scaling),
            out=output if store_mode == "copy" else None,
        )
        if store_mode == "none":
            output.copy_(result[: state.query_len])

        flat = output.reshape(state.query_len, num_heads * head_size)
        if original_device.type != "spyre":
            flat = flat.to(device=original_device, dtype=q.dtype)
        return flat

    def _get_attn_fn(
        self,
        num_blocks: int,
        query_len: int,
        num_heads: int,
        head_size: int,
        store_mode: str,
        store_len: int,
    ):
        key = (num_blocks, query_len, store_mode, store_len)
        if key not in self._attn_fns:
            self._attn_fns[key] = maybe_compile(
                create_compilable_page_attn(
                    num_blocks,
                    query_len,
                    num_heads,
                    head_size,
                    store_mode=store_mode,
                    store_len=store_len,
                ),
                self._compile_attn,
            )
        return self._attn_fns[key]

    def _ensure_cache(
        self, layer_id: int, num_kv_heads: int, head_size: int, block_size: int
    ) -> None:
        if layer_id in self._k_cache:
            return
        num_slots = int(self.model_runner.token_to_kv_pool.size)
        num_pages = _ceil_to(num_slots, block_size) // block_size
        layout = slot_major_kv_layout(
            num_pages * block_size,
            num_kv_heads,
            head_size,
            self._target_dtype,
        )
        shape = (num_pages, block_size, num_kv_heads, head_size)
        self._k_cache[layer_id] = torch.zeros(shape, dtype=self._target_dtype).to(
            self._target_device, device_layout=layout
        )
        self._v_cache[layer_id] = torch.zeros(shape, dtype=self._target_dtype).to(
            self._target_device, device_layout=layout
        )

    def _build_mask_tiles(
        self,
        *,
        seq_len: int,
        query_len: int,
        block_size: int,
        aligned_query_len: int,
    ) -> list[torch.Tensor]:
        context_len = seq_len - query_len
        q_pos = torch.arange(aligned_query_len)
        kv_pos = torch.arange(_ceil_to(seq_len, block_size))
        q_valid = q_pos < query_len
        kv_valid = kv_pos < seq_len
        causal = kv_pos.unsqueeze(0) <= (context_len + q_pos).unsqueeze(1)
        attend = q_valid.unsqueeze(1) & kv_valid.unsqueeze(0) & causal
        mask = torch.where(
            attend,
            torch.tensor(0.0, dtype=self._target_dtype),
            torch.tensor(torch.finfo(self._target_dtype).min, dtype=self._target_dtype),
        )
        return [tile.contiguous() for tile in mask.split(block_size, dim=1)]


__all__ = ["SpyreAttentionBackend"]
