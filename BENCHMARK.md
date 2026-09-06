# Spyre parity benchmark

The reproducible result record is
[`benchmarks/2026-08-26-spyre-dev.json`](benchmarks/2026-08-26-spyre-dev.json).
Both framework runs used one refreshed Spyre pod, the same Granite micro 1B
model, FP16, batch size 1, 64 input tokens, 16 output tokens, one warmup, and
three measured requests. The pulled runtime image resolved to digest
`sha256:daa809d6c0487b019bc992e988570be5e358aa9c2b451911074654bbeec2dacd`.

| Path | Mean latency | Output throughput |
|---|---:|---:|
| latest spyre-inference | 427.74 ms | 37.41 tok/s |
| SGLang with this backend | 424.71 ms | 37.67 tok/s |

SGLang is 0.7% faster in this small run (1.007x), satisfying the ±15% parity
gate. The directly ported attention kernel also matches the reference output
exactly and is within 1.3% of its median latency.

For the requested native PyTorch comparison, one decode-attention forward with
the supported model's geometry (32 query heads, 8 KV heads, head size 128, 256
cached tokens) took 1.270 ms on CPU and 0.979 ms on Spyre, including the output
copy: a 1.30x speedup. This is an operator-level forward comparison, not a claim
about end-to-end server speedup. A separate model-level check recorded a
14.57 ms cached-token bare PyTorch CPU forward. A warmed 64-to-16 generation
request took 334.19 ms in bare PyTorch and 424.71 ms through SGLang/Spyre, or
0.79x. In other words, this tiny batch-1 model reaches parity with the reference
Spyre stack but does not beat a 96-thread CPU end to end; the measured 1.30x
speedup is specifically the synchronized attention forward.

## Three-example MMLU comparison

[`tests/mmlu_speedup_bench.py`](tests/mmlu_speedup_bench.py) compares an
end-to-end, one-token greedy `generate` call through SGLang/Spyre with bare
PyTorch CPU. It uses the same model and exact tokenized prompt on both paths,
batch size 1, one warmup per prompt shape, and three measured iterations. The
three cases are test index 0 from the pinned
[`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu/tree/c30699e8356da336a370243923dbaf21066bb9fe)
revision `c30699e8356da336a370243923dbaf21066bb9fe`.

| MMLU subject | Input tokens | PyTorch CPU mean | SGLang/Spyre mean | Speedup vs CPU |
|---|---:|---:|---:|---:|
| abstract algebra | 71 | 101.31 ms | 135.78 ms | 0.746x |
| anatomy | 125 | 125.27 ms | 167.85 ms | 0.746x |
| high-school physics | 100 | 106.76 ms | 147.09 ms | 0.726x |

The total-latency ratio is **0.740x** (geometric mean 0.739x), meaning that
SGLang/Spyre was 35.2% slower than the 96-thread PyTorch CPU baseline for this
small-request workload. SGLang's timing includes scheduler IPC and sampling;
the bare PyTorch timing calls `model.generate` directly. This is a latency-only
comparison: the single generated token matched on the physics case but differed
on the other two, so the result does not establish output equivalence.

Run it on a configured Spyre host with:

```bash
python tests/mmlu_speedup_bench.py \
  --output benchmarks/2026-08-26-mmlu-speedup.json
```

The checked-in machine-readable result is
[`benchmarks/2026-08-26-mmlu-speedup.json`](benchmarks/2026-08-26-mmlu-speedup.json).
