# TP4, DP1, four-request comparison

These harnesses repeat the PR's five pinned MMLU prompts with four copies of
each prompt submitted together. Both frameworks use FP16, context 2048,
16 greedy output tokens, a 512-token prefill budget, no prefix caching,
one warmup and three measurements per length and case. Gemma uses the
`Gemma4ForCausalLM` text backbone in both frameworks.

The one-token batch completion time is a TTFT proxy. The decode throughput
estimate is `4 * 15 / (mean_16_token_latency - mean_1_token_latency)` for each
case, averaged across five cases. It includes scheduler effects and is not
a direct measurement of streamed token intervals. Four is the submitted
batch size and concurrency cap; prefill chunking can produce smaller device
batches.

Run each framework separately in its pinned environment on the same four
cards, with `TORCH_DEVICE_BACKEND_AUTOLOAD=0` before any imports:

```bash
python sglang_batch_bench.py --model ibm-granite/granite-3.3-8b-instruct \
  --output results/granite-sglang-tp4-b4.json
python vllm_batch_bench.py --model ibm-granite/granite-3.3-8b-instruct \
  --output results/granite-vllm-tp4-b4-matched.json
python sglang_batch_bench.py --model google/gemma-4-26B-A4B-it \
  --output results/gemma-sglang-tp4-b4.json
python vllm_batch_bench.py --model google/gemma-4-26B-A4B-it --text-only \
  --output results/gemma-vllm-tp4-b4-matched.json
python compare.py results
```

Use `--dataset-cache /path/to/cais___mmlu` for the cached Arrow files at the
pinned revision. `run-sglang.sh` and `run-tp4-validation.sh` archive the exact
paths and environment settings used on the allocated test pod. The latter
accepts `vllm` to run only the references, or `sglang gemma` to retry Gemma.
Its default runs the TP4 and TP1 micro checks, both SGLang models, then both
vLLM references. Accelerator jobs must run sequentially.

`results/micro-*.json` contains the two-token canaries and independent eager
FP32 Transformers CPU reference. Reproduce the reference with
`micro_cpu_reference.py`; it compares decoded text and retokenized IDs because
the SGLang canary did not save raw token IDs.

The earlier `*-vllm-tp4-b4.json`, `summary.json` and `environment.json` record
the vLLM-only experiment before TP4 support was implemented. That Gemma run
loaded the default multimodal architecture with text inputs. Keep it separate
from the matched text-only comparison, whose vLLM records use the
`*-matched.json` suffix. Logs are retained locally and excluded from Git.
`results/attempts/` preserves incomplete Gemma runs that exposed CPU-shape
padding and dense MLP alignment bugs; they are excluded from the comparison.

All new runs use worker49 (Xeon Platinum 8474C). The PR's original TP1/B1
baseline used worker80 (Xeon 6736P), so these records do not establish TP
scaling relative to that baseline.
