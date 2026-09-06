"""Small on-hardware A/B benchmark against the pinned spyre-inference kernel."""

from __future__ import annotations

import json
import os
import statistics
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")

import torch
import torch_spyre
from spyre_inference.v1.attention.backends.spyre_attn import (
    _create_compilable_page_attn as create_reference_page_attn,
)

from sglang_spyre_backend.spyre_attention_kernel import create_compilable_page_attn


def _sync() -> None:
    synchronize = getattr(torch.spyre, "synchronize", None)
    if synchronize is not None:
        synchronize()


def _measure(fn, args, *, samples: int, iterations: int) -> list[float]:
    times = []
    for _ in range(samples):
        _sync()
        started = time.perf_counter()
        for _ in range(iterations):
            fn(*args)
        _sync()
        times.append((time.perf_counter() - started) * 1000 / iterations)
    return times


def main() -> None:
    torch_spyre._autoload()
    torch.spyre.set_device(0)
    torch.manual_seed(7)

    num_blocks = 4
    block_size = 64
    # ibm-ai-platform/micro-g3.3-8b-instruct-1b attention geometry.
    num_heads = 32
    num_kv_heads = 8
    queries_per_kv = num_heads // num_kv_heads
    query_len = 1
    head_size = 128
    dtype = torch.float16
    device = torch.device("spyre")

    q = torch.randn(num_kv_heads, queries_per_kv, query_len, head_size, dtype=dtype).to(
        device
    )
    k = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype).to(
        device
    )
    v = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype).to(
        device
    )
    indices = torch.zeros(num_blocks, 32, dtype=torch.int32)
    indices[:, 0] = torch.arange(num_blocks, dtype=torch.int32)
    indices = indices.to(device)
    masks = [
        torch.zeros(query_len, block_size, dtype=dtype).to(device)
        for _ in range(num_blocks)
    ]
    args = (q, k, v, indices, masks, 1.0 / head_size**0.5)

    candidate = torch.compile(
        create_compilable_page_attn(num_blocks, query_len, num_heads, head_size),
        dynamic=False,
    )
    reference = torch.compile(
        create_reference_page_attn(num_blocks, query_len, num_heads, head_size),
        dynamic=False,
    )

    candidate_out = candidate(*args)
    reference_out = reference(*args)
    _sync()
    torch.testing.assert_close(
        candidate_out.to("cpu"), reference_out.to("cpu"), rtol=0, atol=0
    )

    # Compile/cache warmup is deliberately excluded.
    for _ in range(3):
        candidate(*args)
        reference(*args)
    _sync()

    samples = int(os.environ.get("SPYRE_AB_SAMPLES", "5"))
    iterations = int(os.environ.get("SPYRE_AB_ITERATIONS", "20"))
    candidate_ms = _measure(candidate, args, samples=samples, iterations=iterations)
    reference_ms = _measure(reference, args, samples=samples, iterations=iterations)
    candidate_median = statistics.median(candidate_ms)
    reference_median = statistics.median(reference_ms)
    ratio = candidate_median / reference_median
    result = {
        "candidate_ms": candidate_ms,
        "reference_ms": reference_ms,
        "candidate_median_ms": candidate_median,
        "reference_median_ms": reference_median,
        "candidate_over_reference": ratio,
        "within_15_percent": 0.85 <= ratio <= 1.15,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["within_15_percent"]:
        raise SystemExit("ported kernel differs from spyre-inference by more than 15%")


if __name__ == "__main__":
    main()
