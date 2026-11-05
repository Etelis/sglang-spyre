| Framework | Model | TP / DP | Batch size | Mean batch TTFT proxy | Mean batch latency (16 tokens/request) | Aggregate decode tok/s (estimate) |
|---|---|---|---:|---:|---:|---:|
| SGLang / Spyre | Granite 3.3 8B | 4 / 1 | 4 | 0.305 s | 1.479 s | 56.13 |
| vLLM / spyre-inference | Granite 3.3 8B | 4 / 1 | 4 | 0.681 s | 1.484 s | 74.81 |
| SGLang / Spyre | Gemma 4 26B-A4B (text) | 4 / 1 | 4 | 0.590 s | 2.156 s | 40.28 |
| vLLM / spyre-inference | Gemma 4 26B-A4B (text) | 4 / 1 | 4 | 1.502 s | 2.648 s | 52.40 |
