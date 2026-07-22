"""Run SGLang Engine with attention_backend='spyre' on Granite.

Goal: verify Spyre compute actually runs for attention. Single short prompt;
expect first request to be slow (compile) and second fast.
"""

import os
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")


def main():
    import sglang as sgl

    MODEL = os.environ.get(
        "SGLANG_TEST_MODEL", "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
    )

    from sglang.srt.platforms import current_platform
    print(f"Platform: {type(current_platform).__name__}")
    print(f"Model: {MODEL}")
    print(f"Attention backend: spyre (Simple — KV-on-CPU, attn-on-Spyre)\n")

    print("Loading model...")
    t0 = time.time()
    engine = sgl.Engine(
        model_path=MODEL,
        device="cpu",
        attention_backend="spyre",
        dtype="bfloat16",
        mem_fraction_static=0.5,
        disable_cuda_graph=True,
        log_level="warning",
        # Spyre attention only supports single-sequence right now
        max_running_requests=1,
    )
    print(f"Loaded in {time.time() - t0:.1f}s\n")

    # First call: includes Spyre compile time
    print("Request 1 (cold — Spyre kernels compile)...")
    t0 = time.time()
    out1 = engine.generate(
        "The capital of France is",
        sampling_params={"max_new_tokens": 8, "temperature": 0.0},
    )
    print(f"  text: {out1['text'][:80]}")
    print(f"  time: {time.time()-t0:.1f}s")
    print(f"  meta: completion_tokens={out1['meta_info']['completion_tokens']}, "
          f"cached={out1['meta_info'].get('cached_tokens', 0)}")

    # Second call: same prefix → should be much faster
    print("\nRequest 2 (warm — kernels cached)...")
    t0 = time.time()
    out2 = engine.generate(
        "The capital of France is",
        sampling_params={"max_new_tokens": 8, "temperature": 0.0},
    )
    print(f"  text: {out2['text'][:80]}")
    print(f"  time: {time.time()-t0:.1f}s")
    print(f"  meta: completion_tokens={out2['meta_info']['completion_tokens']}, "
          f"cached={out2['meta_info'].get('cached_tokens', 0)}")

    engine.shutdown()
    print("\n✓ DONE")


if __name__ == "__main__":
    main()
