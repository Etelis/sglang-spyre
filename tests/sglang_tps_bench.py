"""TPS benchmark for SGLang on Spyre pod (CPU attention + Spyre platform).

Mirrors spyre_tps_bench.py (vLLM) structure for apples-to-apples
comparison once Spyre attention lands.

Current state (Stage 2): SGLang's attention runs on CPU via torch_native
backend. Our Spyre platform plugin is auto-discovered and on the critical
path (KV pool factory, graph runner) but compute is CPU.
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

    MODEL         = os.environ.get("SGLANG_TEST_MODEL", "ibm-ai-platform/micro-g3.3-8b-instruct-1b")
    DECODE_TOKENS = 200
    WARMUP_TOKENS = 100
    PROMPT        = "What are IBMs main businesses?"
    NUM_RUNS      = int(os.environ.get("SGLANG_BENCH_RUNS","1"))

    # Surface which platform SGLang picked
    from sglang.srt.platforms import current_platform
    print(f"\n{'='*60}")
    print(f"Model:           {MODEL}")
    print(f"Decode tokens:   {DECODE_TOKENS}")
    print(f"Platform:        {type(current_platform).__name__}")
    print(f"  device_type:   {current_platform.device_type}")
    print(f"  default attn:  {current_platform.get_default_attention_backend()}")
    print(f"{'='*60}\n")

    print("Loading model...")
    t0 = time.time()
    backend_name = os.environ.get("SGLANG_ATTN_BACKEND", "spyre")
    engine_kwargs = dict(
        model_path=MODEL,
        device="cpu",
        attention_backend=backend_name,
        dtype="bfloat16",
        mem_fraction_static=0.5,
        disable_cuda_graph=True,
        log_level="warning",
        max_running_requests=1,
    )
    # spyre_paged allocates a flat KV buffer sized to max_total_tokens; with
    # the model default (full context), one allocation exceeds the device
    # FlexAllocator. Cap it so the buffer fits.
    if backend_name == "spyre_paged":
        engine_kwargs["max_total_tokens"] = int(
            os.environ.get("SGLANG_MAX_TOTAL_TOKENS", "2048")
        )
    engine = sgl.Engine(**engine_kwargs)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s\n")

    # ── Warmup ────────────────────────────────────────────────────────────
    print(f"Warming up ({WARMUP_TOKENS} tokens) — excluded from TPS...")
    t_warm = time.time()
    _ = engine.generate(
        PROMPT,
        sampling_params={"max_new_tokens": WARMUP_TOKENS, "temperature": 0.0},
    )
    warmup_time = time.time() - t_warm
    print(f"Warmup done in {warmup_time:.1f}s\n")

    # ── Benchmark ─────────────────────────────────────────────────────────
    print(f"Running {NUM_RUNS} timed requests...\n")

    results = []
    sample_text = ""
    for i in range(NUM_RUNS):
        t_start = time.time()
        out = engine.generate(
            PROMPT,
            sampling_params={"max_new_tokens": DECODE_TOKENS, "temperature": 0.0},
        )
        t_end = time.time()

        meta = out.get("meta_info", {})
        prompt_tokens = meta.get("prompt_tokens", 0)
        output_tokens = meta.get("completion_tokens", 0)
        total_time    = t_end - t_start
        tps           = output_tokens / total_time if total_time > 0 else 0.0

        results.append({
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_time_s":  total_time,
            "tps":           tps,
        })

        print(f"  Run {i+1}: {prompt_tokens} prompt tokens -> {output_tokens} output tokens "
              f"in {total_time:.2f}s  ({tps:.1f} tok/s)")
        sample_text = out["text"]

    # ── Summary ───────────────────────────────────────────────────────────
    avg_tps  = sum(r["tps"]          for r in results) / NUM_RUNS
    avg_time = sum(r["total_time_s"] for r in results) / NUM_RUNS
    out_tok  = results[0]["output_tokens"] or 1
    avg_tbt  = avg_time / out_tok * 1000

    print(f"\n{'='*60}")
    print(f"RESULTS (avg over {NUM_RUNS} runs)")
    print(f"{'='*60}")
    print(f"  Throughput (output tok/s):  {avg_tps:.2f}")
    print(f"  Avg time per request (s):   {avg_time:.2f}")
    print(f"  Approx TBT (ms/token):      {avg_tbt:.1f}")
    print(f"  Warmup time (s):            {warmup_time:.1f}")
    print(f"  Model load time (s):        {load_time:.1f}")
    print(f"{'='*60}\n")

    print("Sample output (run 1):")
    print("-" * 40)
    print(sample_text[:300])
    print("-" * 40)

    engine.shutdown()


if __name__ == "__main__":
    main()
