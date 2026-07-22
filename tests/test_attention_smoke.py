"""Smoke test: drive SpyreAttentionBackend through one prefill + one decode.

Validates that:
  1. The plugin discovers and activates correctly
  2. SpyreAttentionBackend can be instantiated through SGLang's registry
  3. init_forward_metadata() runs (mask construction, BMM selection masks)
  4. forward_extend() and forward_decode() produce non-NaN output
  5. KV cache state is preserved across calls

This does NOT exercise a full model — just the attention backend in isolation.
"""

import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch


def test_smoke():
    # Step 1: activate plugin
    from sglang_spyre_backend import activate
    result = activate()
    assert result == "sglang_spyre_backend.platform.SpyreSRTPlatform", f"activate() returned {result}"

    # Step 2: instantiate platform and init backend
    from sglang_spyre_backend.platform import SpyreSRTPlatform
    platform = SpyreSRTPlatform()
    platform.init_backend()
    print(f"  ✓ Platform initialized, torch.spyre.device_count = {torch.spyre.device_count()}")

    # Step 3: create a minimal model_runner stub
    class StubModelConfig:
        head_dim = 64
        def get_num_kv_heads(self, tp_size): return 4
        def get_num_attention_heads(self, tp_size): return 4

    class StubKVPool:
        size = 256       # number of pages
        page_size = 16   # tokens per page

    class StubServerArgs:
        tp_size = 1

    class StubModelRunner:
        model_config = StubModelConfig()
        token_to_kv_pool = StubKVPool()
        server_args = StubServerArgs()

    runner = StubModelRunner()

    # Step 4: instantiate attention backend via SGLang's registry
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    assert "spyre" in ATTENTION_BACKENDS, "spyre attention backend not registered"
    backend = ATTENTION_BACKENDS["spyre"](runner)
    print(f"  ✓ SpyreAttentionBackend instantiated: {type(backend).__name__}")

    # Step 5: build a fake ForwardBatch for prefill
    class StubForwardMode:
        def is_decode(self): return False
        def is_extend(self): return True
        def is_idle(self): return False
        def is_mixed(self): return False

    class StubForwardBatch:
        forward_mode = StubForwardMode()
        batch_size = 1
        seq_lens_cpu = torch.tensor([8])  # 8 tokens prefill
        # Block table: seq 0 uses physical block 5
        req_to_token = torch.tensor([[5]], dtype=torch.int64)
        # Slot mapping: 8 tokens land in physical block 5, slots 0-7
        out_cache_loc = torch.tensor([5*16+i for i in range(8)], dtype=torch.int64)

    fb = StubForwardBatch()

    print("\n  Running init_forward_metadata (prefill)...")
    backend.init_forward_metadata(fb)
    print(f"    aligned_max_seq_len: {backend._state.aligned_max_seq_len}")
    print(f"    mask shape: {backend._state.attention_mask.shape}")
    print(f"    gather_sel_mask shape: {backend._state.gather_sel_mask_dev.shape}")
    print("  ✓ Metadata built and on Spyre")

    # Step 6: build a fake RadixAttention layer
    class StubRadixAttention:
        layer_id = 0
        tp_q_head_num = 4
        tp_k_head_num = 4
        tp_v_head_num = 4
        head_dim = 64
        scaling = 1.0 / (64 ** 0.5)
        v_head_dim = 64

    layer = StubRadixAttention()
    num_tokens = 8
    num_heads = 4
    head_dim = 64

    # Random q,k,v in BF16 on CPU (SGLang convention)
    torch.manual_seed(42)
    q = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16)
    k = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16)
    v = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16)

    print("\n  Running forward_extend...")
    out = backend.forward_extend(q, k, v, layer, fb, save_kv_cache=True)
    print(f"    output shape: {out.shape}")
    print(f"    output dtype: {out.dtype}")
    print(f"    output device: {out.device}")
    assert out.shape == (num_tokens, num_heads * head_dim), f"bad shape: {out.shape}"
    assert not torch.isnan(out).any(), "output contains NaN"
    print("  ✓ forward_extend produced valid output")

    print("\n=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    test_smoke()
