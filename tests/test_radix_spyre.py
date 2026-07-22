"""Radix attention demo with KV cache on Spyre.

Uses attention_backend="spyre_paged":
  - KV cache lives on Spyre permanently ([H_kv, blocks, block_width])
  - Scatter: overwrite_f per token (no BMM, direct device write)
  - Gather:  BMM against one-hot selection mask
  - Attend:  full _attn_4d on Spyre

SGLang's RadixCache reuses block table entries for shared prefixes —
req 2+ point at the same Spyre blocks as req 1's prefix, so no
re-scatter of those tokens happens.
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
    print(f"\n{'='*60}")
    print("Spyre KV-on-device radix demo")
    print(f"{'='*60}")
    print(f"  Platform:       {type(current_platform).__name__}")
    print(f"  Model:          {MODEL}")
    print(f"  Attention:      spyre_paged (KV on Spyre, overwrite_f scatter)")
    print(f"{'='*60}\n")

    print("Loading model...")
    t0 = time.time()
    engine = sgl.Engine(
        model_path=MODEL,
        device="cpu",
        attention_backend="spyre_paged",
        dtype="bfloat16",
        # Small KV pool: 64 blocks × 16 tok/block = 1024 max tokens.
        # Keeps Spyre KV cache well within device memory.
        mem_fraction_static=0.01,
        max_total_tokens=1024,
        disable_cuda_graph=True,
        log_level="warning",
        max_running_requests=1,
    )
    print(f"Loaded in {time.time() - t0:.1f}s\n")

    SHARED_PREFIX = (
        "You are a helpful assistant. The user is asking about "
        "IBM's Spyre AIU accelerator, a specialized chip for AI inference. "
        "Question: "
    )

    SUFFIXES = [
        "What is Spyre?",
        "How many cores does Spyre have?",
        "What dtype does Spyre use?",
        "What is head_size constraint on Spyre?",
    ]

    sampling = {"max_new_tokens": 24, "temperature": 0.0}

    print(f"Shared prefix: {len(SHARED_PREFIX)} chars")
    print(f"{'#':>2}  {'cached':>6}  {'prompt':>6}  {'gen':>5}  {'time(s)':>7}  preview")
    print("─" * 80)

    cache_hits = []
    for i, suffix in enumerate(SUFFIXES, 1):
        prompt = SHARED_PREFIX + suffix
        t_start = time.time()
        out = engine.generate(prompt, sampling_params=sampling)
        elapsed = time.time() - t_start

        meta = out.get("meta_info", {})
        cached = meta.get("cached_tokens", 0)
        prompt_tok = meta.get("prompt_tokens", 0)
        gen_tok = meta.get("completion_tokens", 0)
        preview = out["text"].strip().replace("\n", " ")[:45]
        cache_hits.append(cached)

        print(f"{i:>2}  {cached:>6}  {prompt_tok:>6}  {gen_tok:>5}  {elapsed:>7.2f}  {preview}...")

    print("─" * 80)

    if all(c > 0 for c in cache_hits[1:]):
        print(f"\n✓ Radix cache hits on all warm requests: {cache_hits}")
    else:
        print(f"\n? Cache hit pattern: {cache_hits}")

    last_prompt = len(SHARED_PREFIX.split()) + len(SUFFIXES[-1].split())
    if cache_hits[-1] > 0:
        print(f"  Last request: {cache_hits[-1]} tokens served from Spyre KV cache")

    engine.shutdown()
    print("\n✓ Radix-on-Spyre demo complete")


if __name__ == "__main__":
    main()
