"""Compare warmed SGLang/Spyre and bare PyTorch on three pinned MMLU prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
os.environ.setdefault("SGLANG_FORCE_DEVICE_SPYRE", "1")

MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
CASES = (
    ("abstract_algebra", 0),
    ("anatomy", 0),
    ("high_school_physics", 0),
)
ANSWER_LABELS = "ABCD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_prompt(question: str, choices: list[str]) -> str:
    choices_text = "\n".join(
        f"{label}. {choice}" for label, choice in zip(ANSWER_LABELS, choices)
    )
    return (
        "Answer the following multiple-choice question. Reply with only A, B, C, "
        f"or D.\n\n{question}\n{choices_text}\nAnswer:"
    )


def load_cases(tokenizer: Any) -> list[dict[str, Any]]:
    from datasets import load_dataset

    cases = []
    for subject, index in CASES:
        row = load_dataset(
            "cais/mmlu",
            subject,
            split="test",
            revision=MMLU_REVISION,
        )[index]
        prompt = build_prompt(row["question"], row["choices"])
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        cases.append(
            {
                "subject": subject,
                "test_index": index,
                "question": row["question"],
                "choices": row["choices"],
                "answer": ANSWER_LABELS[row["answer"]],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "input_ids": input_ids,
            }
        )
    return cases


def summarize(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_seconds": samples,
        "mean_seconds": statistics.mean(samples),
        "median_seconds": statistics.median(samples),
    }


def package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def benchmark_spyre(
    model: str,
    cases: list[dict[str, Any]],
    warmups: int,
    iterations: int,
) -> list[dict[str, Any]]:
    import sglang as sgl

    engine = sgl.Engine(
        model_path=model,
        device="cpu",
        attention_backend="spyre_paged",
        dtype="float16",
        page_size=64,
        max_total_tokens=512,
        max_running_requests=1,
        chunked_prefill_size=-1,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        log_level="warning",
    )
    sampling_params = {
        "max_new_tokens": 1,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    results = []
    try:
        for case in cases:
            result = None
            for _ in range(warmups):
                result = engine.generate(
                    input_ids=case["input_ids"], sampling_params=sampling_params
                )
            samples = []
            for _ in range(iterations):
                started = time.perf_counter()
                result = engine.generate(
                    input_ids=case["input_ids"], sampling_params=sampling_params
                )
                samples.append(time.perf_counter() - started)
            summary = summarize(samples)
            summary["generated_text"] = result["text"] if result else ""
            results.append(summary)
    finally:
        engine.shutdown()
    return results


def benchmark_pytorch(
    model_name: str,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    warmups: int,
    iterations: int,
) -> tuple[list[dict[str, Any]], int]:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    model.eval()
    results = []
    with torch.inference_mode():
        for case in cases:
            input_ids = torch.tensor([case["input_ids"]], dtype=torch.long)
            generated = None
            for _ in range(warmups):
                generated = model.generate(
                    input_ids,
                    do_sample=False,
                    max_new_tokens=1,
                    min_new_tokens=1,
                    use_cache=True,
                )
            samples = []
            for _ in range(iterations):
                started = time.perf_counter()
                generated = model.generate(
                    input_ids,
                    do_sample=False,
                    max_new_tokens=1,
                    min_new_tokens=1,
                    use_cache=True,
                )
                samples.append(time.perf_counter() - started)
            summary = summarize(samples)
            summary["generated_token_id"] = int(generated[0, -1])
            summary["generated_text"] = tokenizer.decode(generated[0, -1:])
            results.append(summary)
    return results, torch.get_num_threads()


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    if args.warmups < 1 or args.iterations < 1:
        raise ValueError("--warmups and --iterations must both be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cases = load_cases(tokenizer)
    spyre_results = benchmark_spyre(args.model, cases, args.warmups, args.iterations)
    pytorch_results, torch_threads = benchmark_pytorch(
        args.model, tokenizer, cases, args.warmups, args.iterations
    )

    examples = []
    speedups = []
    for case, spyre, pytorch in zip(cases, spyre_results, pytorch_results):
        speedup = pytorch["mean_seconds"] / spyre["mean_seconds"]
        speedups.append(speedup)
        examples.append(
            {key: value for key, value in case.items() if key != "input_ids"}
            | {
                "input_tokens": len(case["input_ids"]),
                "sglang_spyre": spyre,
                "pytorch_cpu": pytorch,
                "speedup_over_pytorch_cpu": speedup,
            }
        )

    payload = {
        "model": args.model,
        "dataset": "cais/mmlu",
        "dataset_revision": MMLU_REVISION,
        "split": "test",
        "warmup_iterations_per_example": args.warmups,
        "measured_iterations_per_example": args.iterations,
        "output_tokens": 1,
        "batch_size": 1,
        "pytorch_cpu_threads": torch_threads,
        "software": {
            "sglang": package_version("sglang"),
            "torch": package_version("torch"),
            "torch_spyre": package_version("torch-spyre"),
            "transformers": package_version("transformers"),
        },
        "timing_scope": "end-to-end generate call after per-shape warmup",
        "examples": examples,
        "aggregate": {
            "geometric_mean_speedup": math.prod(speedups) ** (1 / len(speedups)),
            "total_mean_latency_speedup": sum(
                result["mean_seconds"] for result in pytorch_results
            )
            / sum(result["mean_seconds"] for result in spyre_results),
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print("MMLU_SPEEDUP_RESULT=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
