"""Repeat PR #3's pinned MMLU cases with TP4 and four simultaneous requests.

Each batch repeats one case four times, preserving the original per-case shapes.
Batch completion time is reported separately from aggregate output throughput.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from mmlu_sglang_vs_vllm_bench import (
    load_cases, build_prompt, CASES, ANSWER_LABELS, first_letter, package_version, MMLU_REVISION,
)


def cached_cases(tokenizer, cache):
    import pyarrow as pa
    result = []
    for subject, index in CASES:
        path = cache / subject / "0.0.0" / MMLU_REVISION
        files = list(path.glob("*test*.arrow"))
        assert len(files) == 1, f"Expected exactly one pinned test shard in {path}: {files}"
        with pa.memory_map(str(files[0]), "r") as source:
            row = pa.ipc.open_stream(source).read_all().slice(index, 1).to_pylist()[0]
        prompt = build_prompt(row["question"], row["choices"])
        result.append(dict(subject=subject, test_index=index, answer=ANSWER_LABELS[row["answer"]],
                           prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                           input_ids=tokenizer.encode(prompt, add_special_tokens=True)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument(
        "--text-only", action="store_true",
        help="Load Gemma-4's text backbone without its vision tower.",
    )
    args = parser.parse_args()
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cases = cached_cases(tokenizer, args.dataset_cache) if args.dataset_cache else load_cases(tokenizer)
    model_override = {}
    if args.text_only:
        config = AutoConfig.from_pretrained(args.model)
        if config.model_type not in {"gemma4", "gemma4_text"}:
            parser.error("--text-only applies to Gemma-4 checkpoints")
        from spyre_inference.models.gemma4 import GEMMA4_TEXT_BACKBONE_OVERRIDE

        model_override = dict(GEMMA4_TEXT_BACKBONE_OVERRIDE)
    llm = LLM(model=args.model, tensor_parallel_size=4, dtype="float16",
              max_model_len=2048, max_num_seqs=4, max_num_batched_tokens=512,
              enable_prefix_caching=False,
              **({"hf_overrides": model_override} if model_override else {}))
    record = dict(model=args.model, framework="vllm", tensor_parallel_size=4,
                  data_parallel_size=1, batch_size=4, max_new_tokens=16,
                  context_length=2048, dataset="cais/mmlu", dataset_revision=MMLU_REVISION,
                  max_num_batched_tokens=512, prefix_caching=False,
                  model_override=model_override,
                  batch_composition="four copies of each case", warmups=1, iterations=3,
                  software={p: package_version(p) for p in
                            ("torch", "torch-spyre", "vllm", "spyre-inference", "transformers")},
                  examples=[])

    def generate(prompts, length):
        out = llm.generate(prompts, SamplingParams(max_tokens=length, temperature=0,
                           ignore_eos=True), use_tqdm=False)
        assert len(out) == 4, f"Expected four requests, received {len(out)}"
        assert all(len(r.outputs[0].token_ids) == length for r in out)
        return [r.outputs[0].text for r in out]

    def checkpoint():
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    checkpoint()
    for case in cases:
        prompts = [TokensPrompt(prompt_token_ids=case["input_ids"]) for _ in range(4)]
        generate(prompts, 1)
        generate(prompts, 16)
        ttft, latency, texts = [], [], []
        for _ in range(3):
            start = time.perf_counter()
            generate(prompts, 1)
            ttft.append(time.perf_counter() - start)
            start = time.perf_counter()
            texts.append(generate(prompts, 16))
            latency.append(time.perf_counter() - start)
        decode = statistics.mean(latency) - statistics.mean(ttft)
        row = {k: v for k, v in case.items() if k != "input_ids"}
        row.update(input_tokens=len(case["input_ids"]), batch_ttft_seconds=ttft,
                   batch_latency_seconds=latency, mean_batch_ttft_seconds=statistics.mean(ttft),
                   mean_batch_latency_seconds=statistics.mean(latency),
                   aggregate_decode_tokens_per_second=4 * 15 / decode if decode > 0 else None,
                   generated_texts=texts,
                   predicted=[[first_letter(s) for s in batch] for batch in texts])
        record["examples"].append(row)
        checkpoint()
        print(json.dumps(row), flush=True)
    llm.llm_engine.engine_core.shutdown(timeout=60)
    record["complete"] = True
    checkpoint()


if __name__ == "__main__":
    main()
