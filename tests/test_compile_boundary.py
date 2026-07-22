"""Probe: does torch.compile(fn, dynamic=False) on a Spyre tensor function
actually work in our env? This is the foundation for Stage 3 / Stage 4.
"""

import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch_spyre

torch_spyre._autoload()
torch.spyre.set_device(0)

# Apply the addm decomposition workaround (torch-spyre issue #1420)
import torch._inductor.decomposition
from torch_spyre._inductor.decompositions import spyre_decompositions
for op, impl in spyre_decompositions.items():
    if "addm" in op.name():
        torch._inductor.decomposition.decompositions[op] = impl
print(f"Patched torch._inductor.decomposition with {sum(1 for op in spyre_decompositions if 'addm' in op.name())} addm decompositions")


def lm_head_kernel(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The compute we want to run on Spyre: F.linear(x, weight) = x @ weight.T."""
    return torch.nn.functional.linear(x, weight)


def main():
    print("=== Direct eager bmm on Spyre (expected to crash) ===")
    try:
        a = torch.zeros(8, 4, 64, dtype=torch.bfloat16, device="spyre")
        b = torch.zeros(8, 64, 1024, dtype=torch.bfloat16, device="spyre")
        c = torch.bmm(a, b)
        print(f"  ✓ bmm worked: {c.shape}")
    except Exception as e:
        print(f"  ✗ bmm crashed: {type(e).__name__}")

    print("\n=== Compiled F.linear on Spyre ===")
    compiled = torch.compile(lm_head_kernel, dynamic=False)

    # CPU input → Spyre input via .to()
    hidden_size = 2048
    vocab_size = 49152  # Granite tokenizer

    x_cpu = torch.randn(1, hidden_size, dtype=torch.bfloat16)
    w_cpu = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16)

    x_spyre = x_cpu.to(device="spyre")
    w_spyre = w_cpu.to(device="spyre")

    print(f"  x: {x_spyre.shape} on {x_spyre.device}")
    print(f"  w: {w_spyre.shape} on {w_spyre.device}")

    try:
        out_spyre = compiled(x_spyre, w_spyre)
        print(f"  ✓ compiled F.linear OK: {out_spyre.shape} on {out_spyre.device}")

        # Move back to CPU and compare
        out_cpu_from_spyre = out_spyre.to("cpu")
        out_cpu_baseline = torch.nn.functional.linear(x_cpu, w_cpu)

        print(f"  CPU baseline shape: {out_cpu_baseline.shape}")
        print(f"  Spyre output shape: {out_cpu_from_spyre.shape}")

        # bf16 tolerance is generous
        diff = (out_cpu_from_spyre.float() - out_cpu_baseline.float()).abs()
        rel = diff / out_cpu_baseline.float().abs().clamp(min=1e-3)
        print(f"  max abs diff: {diff.max().item():.4f}")
        print(f"  max rel diff: {rel.max().item():.4f}")
        print(f"  mean abs diff: {diff.mean().item():.4f}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n  ✗ compiled F.linear crashed: {type(e).__name__}")
        return

    print("\n✓ Compile boundary works — Stage 3 is viable")


if __name__ == "__main__":
    main()
