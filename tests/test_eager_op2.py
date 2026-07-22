"""Probe simpler ops — does basic device-resident compute work?"""

import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch_spyre
torch_spyre._autoload()
torch.spyre.set_device(0)


def trial(name, fn):
    try:
        result = fn()
        if result is None:
            print(f"  ✗ {name}: returned None")
            return None
        if isinstance(result, torch.Tensor):
            print(f"  ✓ {name}: {result.shape} on {result.device}")
            return result
        print(f"  ? {name}: {type(result).__name__} = {result}")
        return result
    except Exception as e:
        print(f"  ✗ {name}: {type(e).__name__}: {e}")
        return None


def main():
    print("=== Simple constructions ===")
    trial("zeros on spyre", lambda: torch.zeros(4, 4, dtype=torch.bfloat16, device="spyre"))
    trial("randn on cpu, .to(spyre)",
          lambda: torch.randn(4, 4, dtype=torch.bfloat16).to("spyre"))

    print("\n=== Elementwise ops ===")
    a = torch.randn(4, 4, dtype=torch.bfloat16).to("spyre")
    b = torch.randn(4, 4, dtype=torch.bfloat16).to("spyre")
    trial("a + b", lambda: a + b)
    trial("a * b", lambda: a * b)
    trial("a.abs()", lambda: a.abs())
    trial("torch.relu(a)", lambda: torch.relu(a))

    print("\n=== Matmul ===")
    x = torch.randn(8, 16, dtype=torch.bfloat16).to("spyre")
    y = torch.randn(16, 32, dtype=torch.bfloat16).to("spyre")
    trial("x @ y (mm)", lambda: x @ y)
    trial("torch.matmul(x, y)", lambda: torch.matmul(x, y))
    trial("torch.mm(x, y)", lambda: torch.mm(x, y))

    print("\n=== Linear ===")
    weight = torch.randn(64, 16, dtype=torch.bfloat16).to("spyre")
    trial("F.linear(x, weight)", lambda: torch.nn.functional.linear(x, weight))


if __name__ == "__main__":
    main()
