"""Card-free checks of SGLang's request/page boundary at batch four."""

from types import SimpleNamespace

import pytest
import torch

from sglang_spyre_backend.attention import SpyreAttentionBackend


@pytest.mark.parametrize("decode", [False, True])
def test_four_out_of_order_requests_keep_their_pages_and_query_rows(decode):
    block_size = 64
    req_indices = [5, 1, 3, 0]
    lengths = [65, 128, 150, 80]
    pages = [[7, 2], [12, 3], [16, 5, 11], [9, 4]]
    queries = [1, 1, 1, 1] if decode else [1, 1, 7, 9]
    req_to_token = torch.full((6, 512), -1, dtype=torch.int64)
    cache_locations = []
    for req, length, physical_pages, query_len in zip(req_indices, lengths, pages, queries):
        locations = torch.tensor(
            [physical_pages[t // block_size] * block_size + t % block_size for t in range(length)]
        )
        req_to_token[req, :length] = locations
        cache_locations.append(locations[-query_len:])
    forward_batch = SimpleNamespace(
        batch_size=4,
        seq_lens_cpu=torch.tensor(lengths),
        req_pool_indices=torch.tensor(req_indices),
        forward_mode=SimpleNamespace(is_decode=lambda: decode),
        extend_seq_lens_cpu=queries,
        out_cache_loc=torch.cat(cache_locations),
    )
    # Constructor setup allocates a Spyre device; metadata conversion itself is
    # host-only and can be exercised without a card or torch-spyre activation.
    backend = object.__new__(SpyreAttentionBackend)
    backend.block_size = block_size
    backend._max_model_len = 512
    backend.model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token)
    )
    backend._metadata = {"old-step": object()}
    backend.init_forward_metadata(forward_batch)
    batch = backend._batch
    expected_starts = torch.tensor([0] + list(torch.tensor(queries).cumsum(0).tolist()))
    torch.testing.assert_close(batch.query_start_loc.long(), expected_starts)
    assert batch.seq_lens.tolist() == lengths
    assert batch.slot_mapping.tolist() == torch.cat(cache_locations).tolist()
    for index, physical_pages in enumerate(pages):
        assert batch.block_table[index, : len(physical_pages)].tolist() == physical_pages
        assert not batch.block_table[index, len(physical_pages) :].any()
    assert backend._metadata == {}


def test_tp4_attention_groups_use_local_heads_and_replicated_kv_geometry():
    backend = object.__new__(SpyreAttentionBackend)
    backend._layer_groups = {}
    backend._groups = {}
    backend.block_size = 128
    backend._max_model_len = 2048
    backend.max_num_batched_tokens = 512
    backend.max_num_seqs = 4
    backend.model_dtype = torch.float16
    full = SimpleNamespace(
        layer_id=5, tp_q_head_num=4, tp_k_head_num=1, qk_head_dim=512,
        v_head_dim=512, sliding_window_size=-1,
    )
    sliding = SimpleNamespace(
        layer_id=0, tp_q_head_num=4, tp_k_head_num=2, qk_head_dim=256,
        v_head_dim=256, sliding_window_size=127,
    )
    full_spec = backend._group_of(full)
    sliding_spec = backend._group_of(sliding)
    assert (full_spec.num_heads, full_spec.num_kv_heads, full_spec.head_size) == (4, 1, 512)
    assert (sliding_spec.num_heads, sliding_spec.num_kv_heads, sliding_spec.head_size) == (4, 2, 256)
    assert full_spec.sliding_window is None
    assert sliding_spec.sliding_window == 128
    assert len(backend._groups) == 2
    assert backend._groups[full_spec].num_kv_heads == 1
