"""Reproduce the two-token FP32 CPU reference for the TP4/batch-four canary.

Run in the pinned Transformers/Torch environment with the model already cached:

    HF_HOME=/path/to/huggingface python micro_cpu_reference.py \
      --model ibm-ai-platform/micro-g3.3-8b-instruct-1b \
      --dataset-cache /path/to/cais___mmlu \
      --sglang-result results/micro-sglang-tp4-b4.json \
      --output micro-hf-fp32-cpu-reference.json

Each pinned prompt is generated once on CPU; its two-token output is compared
with every saved SGLang response. SGLang's canary did not save raw token IDs,
so re-tokenizing its decoded text is a secondary check, not raw-token parity.
No SGLang engine, vLLM runtime or accelerator backend is imported.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# These must precede Torch/Transformers imports. Match the original CPU probe's
# thread budget and prevent installed device entry points from loading Spyre.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--sglang-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.sglang_result.resolve():
        parser.error("--output must differ from --sglang-result")

    # cached_cases imports only Arrow and the shared prompt helpers. Importing
    # this module does not execute its vLLM benchmark main().
    from vllm_batch_bench import cached_cases

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sglang = json.loads(args.sglang_result.read_text())
    if not sglang.get("complete") or sglang.get("model") != args.model:
        parser.error("SGLang result must be complete and use --model")
    if sglang.get("max_new_tokens") != 2:
        parser.error("The reference requires a two-token SGLang canary")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    cases = cached_cases(tokenizer, args.dataset_cache)
    saved = {
        (row["subject"], row["test_index"]): row for row in sglang["examples"]
    }
    keys = {(case["subject"], case["test_index"]) for case in cases}
    if set(saved) != keys or len(saved) != len(sglang["examples"]):
        parser.error("SGLang result must contain exactly the five pinned cases")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).to("cpu").eval()
    record = dict(
        model=args.model,
        framework="huggingface-transformers",
        device="cpu",
        dtype="float32",
        attention_implementation="eager",
        torch=torch.__version__,
        max_new_tokens=2,
        min_new_tokens=2,
        eos_token_id=None,
        do_sample=False,
        tensor_parallel_size=1,
        comparison_source=args.sglang_result.name,
        comparison_note=(
            "SGLang raw token ids were not saved; compare exact decoded text "
            "and its retokenized ids."
        ),
        examples=[],
        complete=False,
    )
    total_responses = 0
    matching_texts = 0
    matching_ids = 0
    with torch.inference_mode():
        for case in cases:
            source = saved[(case["subject"], case["test_index"])]
            ids = case["input_ids"]
            if (source["prompt_sha256"] != case["prompt_sha256"]
                    or source["input_tokens"] != len(ids)):
                raise ValueError(f"Prompt mismatch for {case['subject']}")
            batches = source["generated_texts"]
            if (len(batches) != sglang["iterations"]
                    or any(len(batch) != sglang["batch_size"] for batch in batches)):
                raise ValueError(f"Incomplete saved responses for {case['subject']}")
            texts = [text for batch in batches for text in batch]
            if not texts or not all(isinstance(text, str) for text in texts):
                raise ValueError(f"Invalid saved responses for {case['subject']}")

            output = model.generate(
                input_ids=torch.tensor([ids], dtype=torch.long, device="cpu"),
                max_new_tokens=2,
                min_new_tokens=2,
                do_sample=False,
                num_beams=1,
                eos_token_id=None,
                forced_eos_token_id=None,
                return_dict_in_generate=True,
                output_scores=True,
            )
            generated_ids = output.sequences[0, len(ids):].tolist()
            if len(generated_ids) != 2 or len(output.scores) != 2:
                raise ValueError("CPU reference did not generate exactly two tokens")
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            steps = []
            for scores in output.scores:
                values, indices = scores[0].topk(3)
                steps.append(dict(
                    top3=[
                        dict(token_id=int(token), text=tokenizer.decode([int(token)]),
                             logit=float(value))
                        for token, value in zip(indices.tolist(), values.tolist())
                    ],
                    top1_margin=float(values[0] - values[1]),
                ))
            text_matches = [candidate == text for candidate in texts]
            id_matches = [
                tokenizer.encode(candidate, add_special_tokens=False) == generated_ids
                for candidate in texts
            ]
            row = {key: value for key, value in case.items() if key != "input_ids"}
            row.update(
                input_tokens=len(ids),
                generated_token_ids=generated_ids,
                generated_text=text,
                steps=steps,
                sglang_unique_texts=sorted(set(texts)),
                all_sglang_texts_match=all(text_matches),
                all_retokenized_sglang_ids_match=all(id_matches),
            )
            record["examples"].append(row)
            total_responses += len(texts)
            matching_texts += sum(text_matches)
            matching_ids += sum(id_matches)

    record.update(
        complete=True,
        all_texts_match=matching_texts == total_responses,
        all_retokenized_ids_match=matching_ids == total_responses,
        compared_responses=total_responses,
        matching_texts=matching_texts,
        matching_retokenized_ids=matching_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "examples"}))
    return 0 if record["all_texts_match"] and record["all_retokenized_ids_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
