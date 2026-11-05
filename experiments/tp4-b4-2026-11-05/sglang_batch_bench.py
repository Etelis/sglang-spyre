"""SGLang/Spyre counterpart to the pinned TP4, batch-four vLLM benchmark.

Four copies of each MMLU case are submitted together. The one-token request
measures batch completion latency as a TTFT proxy, matching the reference leg.
No vision inputs or vision tower are used for Gemma-4.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

# Worker bootstrap supplies the actual per-rank environment. Setting WORLD_SIZE
# or RANK to the TP1 defaults here would be inherited by all four workers.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ.setdefault("SGLANG_FORCE_DEVICE_SPYRE", "1")
# SGLang may import vLLM utility modules; its platform plugins must not load a
# second native backend before SGLang's scheduler has selected its TP rank.
os.environ["VLLM_PLUGINS"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from mmlu_sglang_vs_vllm_bench import (
    MMLU_REVISION,
    first_letter,
    load_cases,
    package_version,
)
# This imports only the shared Arrow data loader, not the vLLM runtime.
from vllm_batch_bench import cached_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--log-level", choices=["info", "warning", "error"], default="warning")
    args = parser.parse_args()
    if min(args.tp_size, args.batch_size, args.context_length, args.iterations) < 1:
        parser.error("TP size, batch size, context length and iterations must be positive")
    if args.max_new_tokens < 2 or args.warmups < 1:
        parser.error("At least two generated tokens and one warmup are required")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cases = (
        cached_cases(tokenizer, args.dataset_cache)
        if args.dataset_cache
        else load_cases(tokenizer)
    )
    text_override = (
        {"architectures": ["Gemma4ForCausalLM"]}
        if "gemma-4" in args.model.lower()
        else {}
    )
    engine_args = dict(
        model_path=args.model,
        device="cpu",
        attention_backend="spyre_paged",
        dtype="float16",
        tp_size=args.tp_size,
        dp_size=1,
        page_size=128,
        context_length=args.context_length,
        max_total_tokens=max(2, args.batch_size) * args.context_length,
        max_running_requests=args.batch_size,
        chunked_prefill_size=512,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        random_seed=0,
        log_level=args.log_level,
        json_model_override_args=json.dumps(text_override),
    )
    record = dict(
        model=args.model,
        framework="sglang",
        tensor_parallel_size=args.tp_size,
        data_parallel_size=1,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        dataset="cais/mmlu",
        dataset_revision=MMLU_REVISION,
        max_num_batched_tokens=512,
        prefix_caching=False,
        batch_composition=f"{args.batch_size} copies of each case",
        warmups=args.warmups,
        iterations=args.iterations,
        model_override=text_override,
        timing_method="batch completion time; one-token request is a TTFT proxy",
        engine_args=engine_args,
        software={
            p: package_version(p)
            for p in ("torch", "torch-spyre", "sglang", "transformers")
        },
        examples=[],
        complete=False,
    )

    def checkpoint():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    checkpoint()
    engine = None
    try:
        import sglang as sgl

        engine = sgl.Engine(**engine_args)

        def generate(ids, length):
            output = engine.generate(
                input_ids=[list(ids) for _ in range(args.batch_size)],
                sampling_params={
                    "max_new_tokens": length,
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
            )
            assert isinstance(output, list) and len(output) == args.batch_size, (
                f"Expected {args.batch_size} responses, received {output!r}"
            )
            assert all(item["meta_info"]["completion_tokens"] == length for item in output)
            return [item["text"] for item in output]

        for case in cases:
            ids = case["input_ids"]
            for _ in range(args.warmups):
                generate(ids, 1)
                generate(ids, args.max_new_tokens)
            ttft, latency, texts = [], [], []
            for _ in range(args.iterations):
                start = time.perf_counter()
                generate(ids, 1)
                ttft.append(time.perf_counter() - start)
                start = time.perf_counter()
                texts.append(generate(ids, args.max_new_tokens))
                latency.append(time.perf_counter() - start)
            decode = statistics.mean(latency) - statistics.mean(ttft)
            row = {key: value for key, value in case.items() if key != "input_ids"}
            row.update(
                input_tokens=len(ids),
                batch_ttft_seconds=ttft,
                batch_latency_seconds=latency,
                mean_batch_ttft_seconds=statistics.mean(ttft),
                mean_batch_latency_seconds=statistics.mean(latency),
                aggregate_decode_tokens_per_second=(
                    args.batch_size * (args.max_new_tokens - 1) / decode
                    if decode > 0 else None
                ),
                generated_texts=texts,
                predicted=[[first_letter(text) for text in batch] for batch in texts],
            )
            record["examples"].append(row)
            checkpoint()
            print(json.dumps(row), flush=True)
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        checkpoint()
        raise
    finally:
        if engine is not None:
            engine.shutdown()
    record["complete"] = True
    checkpoint()


if __name__ == "__main__":
    main()
