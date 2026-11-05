"""Card-free: the ported metadata builder and head-major kernels against dense attention.

Drives ``SpyreAttentionMetadataBuilder`` and ``SpyreHeadMajorAttention`` end to end on
CPU, through the Python page walk, for the cases the SGLang port adds to the previous
baseline: sliding-window masks with active-block skipping, attention logit soft-capping,
grouped-query heads, prefill vs decode dispatch, and the batched decode kernel that a
one-sequence decode step now takes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from sglang_spyre_backend.ops import tile_loop
from sglang_spyre_backend.ops import batched_decode_head_major as bdhm
from sglang_spyre_backend import spyre_attention as sa

BLOCK = 64


@dataclass(frozen=True)
class _Cfg:
    block_size: int = BLOCK
    max_model_len: int = 1024
    max_num_batched_tokens: int = 512
    max_num_seqs: int = 1
    min_num_kv_heads: int = 2


@pytest.fixture(autouse=True)
def _cpu_kernels(monkeypatch):
    # The Python page walk (carry=None on the first tile) runs on CPU; for_each_tile
    # is a Spyre lowering. Kernels are called eagerly, not through torch.compile.
    monkeypatch.setattr(tile_loop, "USE_FOR_EACH_TILE", False)
    monkeypatch.setattr(bdhm, "USE_FOR_EACH_TILE", False)
    monkeypatch.setattr(sa, "USE_FOR_EACH_TILE", False)
    monkeypatch.setattr(sa, "_page_attn_prefill_compiled", sa.page_attn_head_major_prefill_kernel)
    monkeypatch.setattr(sa, "_page_attn_decode_compiled", sa.page_attn_head_major_decode_kernel)
    monkeypatch.setattr(sa, "_batched_decode_compiled", sa.batched_decode_head_major_kernel)
    monkeypatch.setattr(sa, "_reshape_compiled", sa.reshape_and_cache_head_major_kernel)
    monkeypatch.setattr(sa, "_run_with_capped_cores", lambda units, fn, *args: fn(*args))
    # Host stand-in for the device transfer: CPU index_copy_ takes int64 rows where
    # Spyre takes int32, so integer metadata is widened on the way "to the device".
    monkeypatch.setattr(
        sa,
        "convert",
        lambda t, device=None, dtype=None, device_layout=None: (
            t.long() if t.dtype == torch.int32 else t
        ),
    )


def _dense_reference(q, k, v, kv_len, query_len, scale, window, cap):
    """Causal (optionally windowed, capped) attention over one sequence, fp32."""
    heads, kv_heads = q.shape[1], k.shape[1]
    group = heads // kv_heads
    out = torch.empty(query_len, heads, q.shape[2])
    context = kv_len - query_len
    for i in range(query_len):
        pos = context + i
        lo = 0 if window is None else max(0, pos - window + 1)
        for h in range(heads):
            kh = k[lo : pos + 1, h // group].float()
            vh = v[lo : pos + 1, h // group].float()
            sc = (q[i, h].float() @ kh.T) * scale
            if cap:
                sc = torch.tanh(sc / cap) * cap
            out[i, h] = torch.softmax(sc, dim=-1) @ vh
    return out


def _run(kv_len, query_len, *, heads=4, kv_heads=2, head=64, window=None, cap=0.0, pages=None):
    torch.manual_seed(0)
    dtype = torch.float32
    num_blocks = (kv_len + BLOCK - 1) // BLOCK
    pages = pages or list(range(3, 3 + num_blocks))
    num_pages = max(pages) + 2
    spec = sa.GroupSpec(heads, kv_heads, head, window)
    builder = sa.SpyreAttentionMetadataBuilder(spec, _Cfg(min_num_kv_heads=kv_heads), dtype)
    impl = sa.SpyreHeadMajorAttention(
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_size=head,
        scale=head**-0.5,
        block_size=BLOCK,
        staging_rows=513,
        logits_soft_cap=cap,
        model_dtype=dtype,
    )
    k = torch.randn(kv_len, kv_heads, head)
    v = torch.randn(kv_len, kv_heads, head)
    q = torch.randn(query_len, heads, head)
    kv_cache = sa.SpyrePagedKVCache(
        torch.zeros(num_pages, kv_heads, BLOCK, head), torch.zeros(num_pages, kv_heads, BLOCK, head)
    )
    # Write the whole sequence through the ported head-major store index.
    slots = torch.tensor([pages[t // BLOCK] * BLOCK + t % BLOCK for t in range(kv_len)])
    k_rows, v_rows = impl.kv_slot_views(kv_cache)
    for h, idx in enumerate(impl.kv_write_index(slots, torch.device("cpu"))):
        k_rows.index_copy_(0, idx, k[:, h])
        v_rows.index_copy_(0, idx, v[:, h])

    # As wide as the engine's table, ceil(max_model_len / block_size); padding is page 0.
    block_table = torch.zeros(1, _Cfg.max_model_len // BLOCK, dtype=torch.int32)
    block_table[0, : len(pages)] = torch.tensor(pages, dtype=torch.int32)
    md = builder.build(
        sa.SeqBatch(
            query_start_loc=torch.tensor([0, query_len], dtype=torch.int32),
            seq_lens=torch.tensor([kv_len], dtype=torch.int32),
            block_table=block_table,
            slot_mapping=slots[kv_len - query_len :],
        )
    )
    q_staging = torch.zeros(513, heads, head, dtype=dtype)
    out_staging = torch.zeros(513, heads, head, dtype=dtype)
    q_staging[:query_len] = q
    impl.forward(q_staging, out_staging, kv_cache, md)
    expected = _dense_reference(q, k, v, kv_len, query_len, head**-0.5, window, cap)
    return out_staging[:query_len], expected, md


@pytest.mark.parametrize("kv_len,query_len", [(40, 40), (200, 200), (300, 45)])
def test_prefill_matches_dense(kv_len, query_len):
    got, expected, md = _run(kv_len, query_len)
    assert md.aligned_query_lens == [512]
    torch.testing.assert_close(got, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("kv_len", [1, 63, 64, 65, 300])
def test_decode_takes_batched_kernel_and_matches_dense(kv_len):
    got, expected, md = _run(kv_len, 1)
    # Since spyre-inference #1068 a single decoding sequence runs the batched kernel.
    assert md.padded_num_seqs == 1
    torch.testing.assert_close(got, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("kv_len,query_len", [(300, 300), (300, 1), (500, 40)])
def test_sliding_window_matches_dense(kv_len, query_len):
    got, expected, md = _run(kv_len, query_len, window=128)
    assert md.active_block_indices is not None
    torch.testing.assert_close(got, expected, atol=2e-4, rtol=2e-4)


def test_sliding_window_skips_blocks_outside_every_window():
    _, _, md = _run(500, 1, window=64)
    # Window 64 at position 499 touches blocks 6 and 7 only; padding repeats the last.
    active = md.active_block_indices[0]
    assert set(active) <= {6, 7}
    assert md.attention_mask_stacks[0].shape[0] == len(active)


@pytest.mark.parametrize("query_len", [1, 37])
def test_logit_soft_cap_matches_dense(query_len):
    got, expected, _ = _run(200, query_len, cap=3.0)
    torch.testing.assert_close(got, expected, atol=2e-4, rtol=2e-4)


def test_scattered_pages_are_gathered_by_the_page_table():
    got, expected, _ = _run(200, 7, pages=[9, 2, 5, 7])
    torch.testing.assert_close(got, expected, atol=2e-4, rtol=2e-4)


def test_head_major_write_index_layout():
    impl = sa.SpyreHeadMajorAttention(
        num_heads=4, num_kv_heads=2, head_size=64, scale=1.0, block_size=BLOCK, staging_rows=8
    )
    idx = impl.kv_write_index(torch.tensor([0, 1, 64, 130]), torch.device("cpu"))
    # Row (block * KV + h) * block_size + offset.
    assert idx[0].tolist() == [0, 1, 128, 258]
    assert idx[1].tolist() == [64, 65, 192, 322]


def test_builder_rejects_unaligned_block_size():
    with pytest.raises(ValueError, match="multiple of 64"):
        sa.SpyreAttentionMetadataBuilder(
            sa.GroupSpec(4, 2, 64, None), _Cfg(block_size=96), torch.float32
        )


@pytest.mark.parametrize(
    "heads,kv_heads,head,window",
    [
        pytest.param(8, 2, 128, None, id="granite-tp4"),
        pytest.param(4, 2, 256, 128, id="gemma-sliding-tp4"),
        # Gemma has two global KV heads across four ranks: KV heads replicate,
        # while each rank still receives four distinct query heads.
        pytest.param(4, 1, 512, None, id="gemma-full-tp4-kv-replicated"),
    ],
)
@pytest.mark.parametrize("query_lens", [[1, 1, 1, 1], [1, 1, 7, 9], [7, 11, 2, 1]])
def test_four_requests_with_tp4_head_geometry_match_dense(
    monkeypatch, heads, kv_heads, head, window, query_lens
):
    """Different histories and physical pages must not mix requests' KV or rows."""
    torch.manual_seed(19)
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")
    kv_lens = [63, 65, 137, 200]
    num_blocks = [(n + BLOCK - 1) // BLOCK for n in kv_lens]
    # Every history owns different, nonconsecutive pages, including a boundary
    # on request 1. Reordering physical pages catches a sequence/page mix-up.
    allocated = torch.randperm(sum(num_blocks)) + 1
    pages, start = [], 0
    for count in num_blocks:
        pages.append(allocated[start : start + count].tolist())
        start += count
    # The device gather requires a strict subset of its source allocation.
    # Leave enough unused pages for the batched dispatch, rather than making
    # this accidentally test only the per-sequence fallback.
    num_pages = max(len(allocated) + 2, 33)
    spec = sa.GroupSpec(heads, kv_heads, head, window)
    builder = sa.SpyreAttentionMetadataBuilder(
        spec, _Cfg(max_num_seqs=4, min_num_kv_heads=kv_heads), torch.float32
    )
    impl = sa.SpyreHeadMajorAttention(
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_size=head,
        scale=head**-0.5,
        block_size=BLOCK,
        staging_rows=513,
        model_dtype=torch.float32,
    )
    kv_cache = sa.SpyrePagedKVCache(
        torch.zeros(num_pages, kv_heads, BLOCK, head),
        torch.zeros(num_pages, kv_heads, BLOCK, head),
    )
    k_rows, v_rows = impl.kv_slot_views(kv_cache)
    block_table = torch.zeros(4, _Cfg.max_model_len // BLOCK, dtype=torch.int32)
    queries, slots, references, new_keys, new_values = [], [], [], [], []
    starts = [0]
    for index, (kv_len, query_len) in enumerate(zip(kv_lens, query_lens)):
        k = torch.randn(kv_len, kv_heads, head)
        v = torch.randn(kv_len, kv_heads, head)
        q = torch.randn(query_len, heads, head)
        sequence_slots = torch.tensor(
            [pages[index][t // BLOCK] * BLOCK + t % BLOCK for t in range(kv_len)]
        )
        context = kv_len - query_len
        for h, indices in enumerate(
            impl.kv_write_index(sequence_slots[:context], torch.device("cpu"))
        ):
            k_rows.index_copy_(0, indices, k[:context, h])
            v_rows.index_copy_(0, indices, v[:context, h])
        block_table[index, : num_blocks[index]] = torch.tensor(pages[index])
        queries.append(q)
        slots.append(sequence_slots[-query_len:])
        new_keys.append(k[-query_len:])
        new_values.append(v[-query_len:])
        references.append(
            _dense_reference(q, k, v, kv_len, query_len, head**-0.5, window, 0.0)
        )
        starts.append(starts[-1] + query_len)
    md = builder.build(
        sa.SeqBatch(
            query_start_loc=torch.tensor(starts, dtype=torch.int32),
            seq_lens=torch.tensor(kv_lens, dtype=torch.int32),
            block_table=block_table,
            slot_mapping=torch.cat(slots),
        )
    )
    q_staging = torch.zeros(513, heads, head)
    out_staging = torch.full_like(q_staging, float("nan"))
    q_staging[: starts[-1]] = torch.cat(queries)
    impl.do_kv_cache_update(
        torch.cat(new_keys), torch.cat(new_values), kv_cache,
        impl.kv_write_index(torch.cat(slots), torch.device("cpu")),
    )
    impl.forward(q_staging, out_staging, kv_cache, md)
    if query_lens == [1, 1, 1, 1]:
        assert md.padded_num_seqs == 4
        assert impl._batched_decode_preconditions_met(md, num_pages)
    elif query_lens[:2] == [1, 1]:
        assert md.num_decode_seqs == 2 and md.padded_num_seqs == 2
        assert impl._batched_decode_preconditions_met(md, num_pages)
    else:
        assert md.num_decode_seqs == 0
    torch.testing.assert_close(
        out_staging[: starts[-1]], torch.cat(references), atol=2e-4, rtol=2e-4
    )
