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
