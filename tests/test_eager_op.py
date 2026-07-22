"""Probe: does direct eager dispatch to torch.ops.spyre.batched_matmul work?

If yes, we sidestep the Inductor compile crash entirely and can do
all our compute via eager Spyre ops.
"""

import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch_spyre

torch_spyre._autoload()
torch.spyre.set_device(0)


def main():
    print("=== List of torch.ops.spyre.* ops ===")
    spyre_ns = torch.ops.spyre
    op_names = [n for n in dir(spyre_ns) if not n.startswith("_")]
    print(f"  found {len(op_names)} ops")
    for n in sorted(op_names):
        print(f"    torch.ops.spyre.{n}")

    print("\n=== Direct eager batched_matmul (no torch.bmm) ===")
    a = torch.randn(2, 4, 8, dtype=torch.bfloat16, device="spyre")
    b = torch.randn(2, 8, 16, dtype=torch.bfloat16, device="spyre")
    print(f"  a: {a.shape} on {a.device}")
    print(f"  b: {b.shape} on {b.device}")

    try:
        c = torch.ops.spyre.batched_matmul(a, b)
        print(f"  ✓ batched_matmul OK: {c.shape} on {c.device}")
        print(f"  sample: {c[0,0,0].item()}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ✗ failed: {type(e).__name__}")
        return

    print("\n=== Compare against CPU bmm ===")
    a_cpu = a.to("cpu").float()
    b_cpu = b.to("cpu").float()
    c_baseline = torch.bmm(a_cpu, b_cpu)
    c_spyre_cpu = c.to("cpu").float()

    diff = (c_spyre_cpu - c_baseline).abs()
    print(f"  max abs diff:  {diff.max().item():.4f}")
    print(f"  mean abs diff: {diff.mean().item():.4f}")
    print("  (BF16 → FP32 baseline diff expected to be small but nonzero)")

    print("\n✓ Eager Spyre ops work — Stage 3 viable via this path")


if __name__ == "__main__":
    main()
