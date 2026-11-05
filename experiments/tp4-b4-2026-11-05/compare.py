"""Validate the matched SGLang/vLLM records and produce the PR results table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


CONFIG_KEYS = (
    "model", "tensor_parallel_size", "data_parallel_size", "batch_size",
    "max_new_tokens", "context_length", "dataset_revision", "iterations",
    "warmups", "max_num_batched_tokens", "prefix_caching",
    "model_override",
)
CASE_KEYS = ("subject", "test_index", "answer", "prompt_sha256", "input_tokens")


def summarize(record):
    assert record.get("complete") and not record.get("error"), "Run did not complete"
    assert len(record["examples"]) == 5
    for row in record["examples"]:
        for key in ("batch_ttft_seconds", "batch_latency_seconds", "generated_texts"):
            assert len(row[key]) == record["iterations"]
        assert all(len(batch) == record["batch_size"] for batch in row["generated_texts"])
        ttft = statistics.mean(row["batch_ttft_seconds"])
        latency = statistics.mean(row["batch_latency_seconds"])
        throughput = record["batch_size"] * (record["max_new_tokens"] - 1) / (latency - ttft)
        for key, value in (
            ("mean_batch_ttft_seconds", ttft),
            ("mean_batch_latency_seconds", latency),
            ("aggregate_decode_tokens_per_second", throughput),
        ):
            assert abs(row[key] - value) < 1e-9, f"Incorrect derived metric: {key}"
    result = {
        key: statistics.mean(row[key] for row in record["examples"])
        for key in (
            "mean_batch_ttft_seconds", "mean_batch_latency_seconds",
            "aggregate_decode_tokens_per_second",
        )
    }
    result["answer_variation"] = [
        {"subject": row["subject"], "predicted": row["predicted"]}
        for row in record["examples"]
        if len({answer for batch in row["predicted"] for answer in batch}) > 1
    ]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    result = {
        "timing_method": "Warmed batch completion latency; the one-token batch is a TTFT proxy. "
        "Aggregate decode throughput is 4*15/(mean 16-token latency - mean 1-token latency) "
        "per case, then averaged across five pinned MMLU cases.",
        "models": {},
    }
    table = [
        "| Framework | Model | TP / DP | Batch size | Mean batch TTFT proxy | Mean batch latency (16 tokens/request) | Aggregate decode tok/s (estimate) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for name, label in (("granite", "Granite 3.3 8B"), ("gemma", "Gemma 4 26B-A4B (text)")):
        sg = json.loads((args.results / f"{name}-sglang-tp4-b4.json").read_text())
        vl = json.loads((args.results / f"{name}-vllm-tp4-b4-matched.json").read_text())
        assert all(sg[key] == vl[key] for key in CONFIG_KEYS)
        assert all(
            sg["software"][key] == vl["software"][key]
            for key in ("torch", "torch-spyre", "transformers")
        ), "Shared software version mismatch"
        assert all(
            all(a[key] == b[key] for key in CASE_KEYS)
            for a, b in zip(sg["examples"], vl["examples"], strict=True)
        ), "Prompt or case mismatch"
        summary = {"sglang": summarize(sg), "vllm": summarize(vl)}
        summary["configuration"] = {key: sg[key] for key in CONFIG_KEYS}
        summary["sglang_over_vllm_latency"] = (
            summary["sglang"]["mean_batch_latency_seconds"]
            / summary["vllm"]["mean_batch_latency_seconds"]
        )
        summary["output_comparison"] = [
            {
                "subject": a["subject"], "gold": a["answer"],
                "sglang_predicted": a["predicted"], "vllm_predicted": b["predicted"],
                "identical_text_slots": sum(
                    x == y for ab, bb in zip(a["generated_texts"], b["generated_texts"], strict=True)
                    for x, y in zip(ab, bb, strict=True)
                ),
            }
            for a, b in zip(sg["examples"], vl["examples"], strict=True)
        ]
        result["models"][name] = summary
        for framework, display in (("sglang", "SGLang / Spyre"), ("vllm", "vLLM / spyre-inference")):
            stats = summary[framework]
            table.append(
                f"| {display} | {label} | 4 / 1 | 4 | "
                f"{stats['mean_batch_ttft_seconds']:.3f} s | "
                f"{stats['mean_batch_latency_seconds']:.3f} s | "
                f"{stats['aggregate_decode_tokens_per_second']:.2f} |"
            )
    (args.results / "matched-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.results / "matched-table.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))


if __name__ == "__main__":
    main()
