# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared Spyre paged-attention kernels.

This is a direct port of the framework-neutral kernel section in
``spyre_inference/v1/attention/backends/spyre_attn.py`` at
``6e5ff996178255ff63acec480cf8fb43ce1e43de``.  Keep this module free of
SGLang and vLLM types so both integrations execute the same kernel design.
"""

from __future__ import annotations

import torch


def slot_major_kv_layout(
    num_slots: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
):
    """Return the slot-outermost physical layout required by indirect access."""
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype, get_elem_in_stick

    elems_per_stick = get_elem_in_stick(dtype)
    sticks = (head_size + elems_per_stick - 1) // elems_per_stick
    return SpyreTensorLayout(
        device_size=[num_slots, num_kv_heads, sticks, elems_per_stick],
        stride_map=[
            num_kv_heads * sticks * elems_per_stick,
            sticks * elems_per_stick,
            elems_per_stick,
            1,
        ],
        device_dtype=get_device_dtype(dtype),
    )


def maybe_compile(fn, compile_enabled: bool):
    if compile_enabled:
        return torch.compile(fn, dynamic=False)
    return fn


def reshape_and_cache_kernel(key, value, k_slots, v_slots, slot_mapping):
    """Scatter the current K/V rows into the dense slot-major cache."""
    k_slots.index_copy_(0, slot_mapping, key)
    v_slots.index_copy_(0, slot_mapping, value)


def create_compilable_page_attn(
    num_blocks: int,
    padded_query_len: int,
    num_heads: int,
    head_size: int,
    has_alibi: bool = False,
    logits_soft_cap: float = 0.0,
    store_mode: str = "none",
    store_len: int = 0,
):
    """Create the latest Spyre online-softmax kernel for a fixed page count."""

    def specialized_paged_attn_kernel(
        q,
        k_pages,
        v_pages,
        page_index_table,
        mask_tiles,
        scale,
        alibi_bias_tiles=None,
        out=None,
        out_rows=None,
    ):
        tile_max = None
        tile_sum = None
        tile_output = None

        for i in range(num_blocks):
            page_idx = page_index_table[i, 0:1]
            k_page = k_pages.index_select(0, page_idx)
            v_page = v_pages.index_select(0, page_idx)
            k_page_4d = k_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)
            v_page_4d = v_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)
            mask_tile = mask_tiles[i]

            scores = torch.matmul(q, k_page_4d.transpose(-2, -1)) * scale
            if logits_soft_cap > 0.0:
                scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
            if has_alibi:
                assert alibi_bias_tiles is not None
                scores = scores + alibi_bias_tiles[i]
            scores = scores + mask_tile
            scores_max = torch.amax(scores, dim=-1, keepdim=True)

            if i == 0:
                tile_max = scores_max
                tile_probs = torch.exp(scores - tile_max)
                tile_output = torch.matmul(tile_probs, v_page_4d)
                tile_sum = tile_probs.sum(dim=-1, keepdim=True)
            else:
                assert tile_max is not None
                assert tile_sum is not None
                assert tile_output is not None
                new_max = torch.maximum(tile_max, scores_max)
                rescale = torch.exp(tile_max - new_max)
                tile_output = tile_output * rescale
                tile_sum = tile_sum * rescale
                tile_probs = torch.exp(scores - new_max)
                tile_output += torch.matmul(tile_probs, v_page_4d)
                tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
                tile_max = new_max

        assert tile_output is not None and tile_sum is not None
        attn = tile_output / tile_sum
        attn = attn.reshape(1, num_heads, padded_query_len, head_size).transpose(1, 2)
        attn = attn.reshape(padded_query_len, num_heads, head_size)
        if store_mode == "copy":
            assert out is not None
            out.copy_(attn[:store_len])
            return out
        if store_mode == "index":
            assert out is not None and out_rows is not None
            out.index_copy_(0, out_rows, attn[:store_len])
            return out
        return attn

    return specialized_paged_attn_kernel


__all__ = [
    "create_compilable_page_attn",
    "maybe_compile",
    "reshape_and_cache_kernel",
    "slot_major_kv_layout",
]
