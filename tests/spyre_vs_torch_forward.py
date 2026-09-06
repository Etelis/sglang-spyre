"""Compare one Granite-shaped attention forward on Spyre and native PyTorch CPU."""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")

import torch
import torch_spyre

from sglang_spyre_backend.spyre_attention_kernel import create_compilable_page_attn


def _elapsed_once(fn, args, *, copy_to_cpu: bool) -> tuple[float, torch.Tensor]:
    started = time.perf_counter()
    output = fn(*args)
    if copy_to_cpu:
        output = output.to("cpu")
    elapsed_ms = (time.perf_counter() - started) * 1000
    return elapsed_ms, output


def main() -> None:
    torch_spyre._autoload()
    torch.spyre.set_device(0)
    torch.manual_seed(11)

    # micro-g3.3-8b-instruct-1b: 32 query heads, 8 KV heads, head size 128.
    num_blocks, block_size = 4, 64
    num_heads, num_kv_heads, head_size = 32, 8, 128
    query_len = 1
    scale = 1.0 / head_size**0.5

    q_cpu = torch.randn(
        num_kv_heads,
        num_heads // num_kv_heads,
        query_len,
        head_size,
        dtype=torch.float16,
    )
    k_cpu = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.float16,
    )
    v_cpu = torch.randn_like(k_cpu)
    indices_cpu = torch.zeros(num_blocks, 32, dtype=torch.int64)
    indices_cpu[:, 0] = torch.arange(num_blocks)
    masks_cpu = [
        torch.zeros(query_len, block_size, dtype=torch.float16)
        for _ in range(num_blocks)
    ]

    native = create_compilable_page_attn(num_blocks, query_len, num_heads, head_size)
    compiled_spyre = torch.compile(
        create_compilable_page_attn(num_blocks, query_len, num_heads, head_size),
        dynamic=False,
    )
    indices_spyre = indices_cpu.to(torch.int32).to("spyre")
    spyre_args = (
        q_cpu.to("spyre"),
        k_cpu.to("spyre"),
        v_cpu.to("spyre"),
        indices_spyre,
        [mask.to("spyre") for mask in masks_cpu],
        scale,
    )
    cpu_args = (q_cpu, k_cpu, v_cpu, indices_cpu, masks_cpu, scale)

    # Compile and warm both paths outside the reported single-pass timing.
    spyre_warm = compiled_spyre(*spyre_args).to("cpu")
    cpu_warm = native(*cpu_args)
    torch.testing.assert_close(spyre_warm, cpu_warm, rtol=2e-2, atol=2e-2)

    cpu_ms, cpu_output = _elapsed_once(native, cpu_args, copy_to_cpu=False)
    spyre_ms, spyre_output = _elapsed_once(compiled_spyre, spyre_args, copy_to_cpu=True)
    torch.testing.assert_close(spyre_output, cpu_output, rtol=2e-2, atol=2e-2)

    print(
        json.dumps(
            {
                "model_geometry": "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
                "operation": "one decode attention forward over 256 cached tokens",
                "pytorch_cpu_ms": cpu_ms,
                "spyre_ms_including_output_copy": spyre_ms,
                "speedup_over_pytorch_cpu": cpu_ms / spyre_ms,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
