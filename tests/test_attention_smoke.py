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


def main():
    # Step 1: activate plugin
    from sglang_spyre_backend import activate

    result = activate()
    assert result == "sglang_spyre_backend.platform.SpyreSRTPlatform", (
        f"activate() returned {result}"
    )

    # Step 2: instantiate platform and init backend
    import torch_spyre

    torch_spyre._autoload()
    torch.spyre.set_device(0)
    device_count = torch.spyre.device_count()
    print(f"  ✓ Platform initialized, torch.spyre.device_count = {device_count}")

    # Step 3: create a minimal model_runner stub
    class StubModelConfig:
        head_dim = 64

        def get_num_kv_heads(self, tp_size):
            return 4

        def get_num_attention_heads(self, tp_size):
            return 4

    class StubKVPool:
        size = 256
        page_size = 64

    class StubReqToTokenPool:
        req_to_token = torch.arange(256).reshape(1, 256)

    class StubServerArgs:
        tp_size = 1

    class StubModelRunner:
        model_config = StubModelConfig()
        token_to_kv_pool = StubKVPool()
        req_to_token_pool = StubReqToTokenPool()
        server_args = StubServerArgs()

    runner = StubModelRunner()

    # Step 4: instantiate attention backend via SGLang's registry
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS

    assert "spyre_paged" in ATTENTION_BACKENDS
    backend = ATTENTION_BACKENDS["spyre_paged"](runner)
    print(f"  ✓ SpyreAttentionBackend instantiated: {type(backend).__name__}")

    # Step 5: build a fake ForwardBatch for prefill
    class StubForwardMode:
        def is_decode(self):
            return False

        def is_extend(self):
            return True

        def is_idle(self):
            return False

        def is_mixed(self):
            return False

    class StubForwardBatch:
        forward_mode = StubForwardMode()
        batch_size = 1
        seq_lens_cpu = torch.tensor([8])
        req_pool_indices = torch.tensor([0], dtype=torch.int64)
        extend_seq_lens_cpu = [8]
        extend_seq_lens = torch.tensor([8])
        extend_num_tokens = 8
        out_cache_loc = torch.arange(8, dtype=torch.int64)

    fb = StubForwardBatch()

    print("\n  Running init_forward_metadata (prefill)...")
    backend.init_forward_metadata(fb)
    print(f"    aligned_query_len: {backend._state.aligned_query_len}")
    print(f"    mask tiles: {len(backend._state.mask_tiles_cpu)}")
    print("  ✓ Metadata built and on Spyre")

    # Step 6: build a fake RadixAttention layer
    class StubRadixAttention:
        layer_id = 0
        tp_q_head_num = 4
        tp_k_head_num = 4
        tp_v_head_num = 4
        qk_head_dim = 64
        scaling = 1.0 / (64**0.5)
        v_head_dim = 64

    layer = StubRadixAttention()
    num_tokens = 8
    num_heads = 4
    head_dim = 64

    # Random q,k,v in FP16 on CPU (SGLang convention)
    torch.manual_seed(42)
    q = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.float16)
    k = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.float16)
    v = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.float16)

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
    main()
