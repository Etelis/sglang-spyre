"""Radix attention demo on SGLang with Spyre platform active.

Demonstrates SGLang's prefix-sharing RadixCache:
  - 4 prompts sharing a long common prefix
  - First prompt: cold cache, full prefill
  - Subsequent prompts: prefix tokens come from RadixCache
  - meta_info["cached_tokens"] visibly grows on shared prefixes

Spyre platform plugin is active (visible via current_platform). KV cache
operations route through SpyreMHATokenToKVPool. Attention compute itself
runs on CPU (torch_native backend) — Spyre attention requires upstream
fixes to torch_spyre that are out of scope here.
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

    # Surface platform info
    from sglang.srt.platforms import current_platform
    print(f"\n{'='*60}")
    print("Spyre platform on SGLang — radix demo")
    print(f"{'='*60}")
    print(f"  Platform:       {type(current_platform).__name__}")
    print(f"  Model:          {MODEL}")
    print(f"  Out-of-tree:    {current_platform.is_out_of_tree()}")
    print(f"  KV pool class:  {current_platform.get_mha_kv_pool_cls().__name__}")
    print(f"  Default attn:   {current_platform.get_default_attention_backend()}")
    print(f"{'='*60}\n")

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
        max_running_requests=1,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # Long shared prefix to make radix benefits visible
    SHARED_PREFIX = (
        "You are a knowledgeable assistant. The user is asking about "
        "various topics related to artificial intelligence, machine "
        "learning, and the IBM Spyre AIU accelerator. Please provide "
        "concise, accurate answers based on the context.\n\nQuestion: "
    )

    SUFFIXES = [
        "What does Spyre stand for?",
        "What dtype does Spyre use?",
        "How many cores does Spyre have?",
        "What is the head_size constraint on Spyre?",
    ]

    sampling = {"max_new_tokens": 32, "temperature": 0.0}

    print(f"Shared prefix: {len(SHARED_PREFIX)} chars\n")
    print(f"{'#':>2}  {'cached':>6}  {'prompt_tok':>10}  {'gen_tok':>7}  {'time(s)':>7}  preview")
    print("─" * 90)

    cache_history = []
    for i, suffix in enumerate(SUFFIXES, 1):
        prompt = SHARED_PREFIX + suffix
        t_start = time.time()
        out = engine.generate(prompt, sampling_params=sampling)
        t_end = time.time()

        meta = out.get("meta_info", {})
        cached = meta.get("cached_tokens", 0)
        prompt_tok = meta.get("prompt_tokens", 0)
        gen_tok = meta.get("completion_tokens", 0)
        elapsed = t_end - t_start
        text_preview = out["text"].strip().replace("\n", " ")[:50]

        cache_history.append(cached)
        print(
            f"{i:>2}  {cached:>6}  {prompt_tok:>10}  {gen_tok:>7}  {elapsed:>7.2f}  "
            f"{text_preview}..."
        )

    print("─" * 90)
    print()

    # Verification
    if all(c > 0 for c in cache_history[1:]):
        print(f"✓ All requests after the first hit cache: cached_tokens = {cache_history}")
    else:
        print(f"⚠ Unexpected cache miss pattern: {cache_history}")

    # Show ratio of cached tokens to prompt tokens for last request
    if cache_history[-1] > 0 and prompt_tok > 0:
        pct = 100 * cache_history[-1] / prompt_tok
        print(f"  Last request: {cache_history[-1]}/{prompt_tok} prompt tokens cached ({pct:.0f}%)")

    engine.shutdown()
    print("\n✓ Radix demo complete")


if __name__ == "__main__":
    main()
