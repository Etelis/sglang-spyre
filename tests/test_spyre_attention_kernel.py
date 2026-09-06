import torch

from sglang_spyre_backend.spyre_attention_kernel import create_compilable_page_attn


def test_online_softmax_matches_dense_attention_on_cpu():
    torch.manual_seed(7)
    num_blocks, block_size = 3, 4
    num_heads, num_kv_heads, head_size = 4, 2, 8
    q_len = 2
    q = torch.randn(num_kv_heads, num_heads // num_kv_heads, q_len, head_size)
    k_pages = torch.randn(num_blocks, block_size, num_kv_heads, head_size)
    v_pages = torch.randn_like(k_pages)
    page_table = torch.zeros(num_blocks, 32, dtype=torch.int64)
    page_table[:, 0] = torch.tensor([2, 0, 1])
    masks = [torch.zeros(q_len, block_size) for _ in range(num_blocks)]

    actual = create_compilable_page_attn(num_blocks, q_len, num_heads, head_size)(
        q, k_pages, v_pages, page_table, masks, 1.0 / head_size**0.5
    )

    order = page_table[:, 0]
    keys = k_pages.index_select(0, order).flatten(0, 1).permute(1, 0, 2).unsqueeze(1)
    values = v_pages.index_select(0, order).flatten(0, 1).permute(1, 0, 2).unsqueeze(1)
    scores = torch.matmul(q, keys.transpose(-2, -1)) / head_size**0.5
    expected = torch.matmul(scores.softmax(dim=-1), values)
    expected = (
        expected.reshape(1, num_heads, q_len, head_size)
        .transpose(1, 2)
        .reshape(q_len, num_heads, head_size)
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
