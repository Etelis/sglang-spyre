"""Stage 1 smoke test: SGLang serves a small model on CPU.

No Spyre involvement. Just proves the SGLang Engine works end-to-end
in our spyre-inference venv on the pod, AND that radix prefix-cache
hits are observable.

Run inside the pod:
    source /etc/bashrc
    source $HOME/dt-inductor/spyre-inference/.venv/bin/activate
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    python3 tests/test_cpu_smoke.py
"""

import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import json


def main():
    import sglang as sgl

    model_path = os.environ.get(
        "SGLANG_TEST_MODEL", "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
    )
    print(f"Loading model: {model_path}")

    engine = sgl.Engine(
        model_path=model_path,
        device="cpu",
        attention_backend="torch_native",
        dtype="bfloat16",
        mem_fraction_static=0.5,
        disable_cuda_graph=True,
        log_level="warning",
    )
    print("Engine ready.\n")

    # Shared prefix to exercise radix cache
    prefix = "The capital of France is"

    print("=== Request 1 (cold cache) ===")
    out1 = engine.generate(
        prefix + " called",
        sampling_params={"max_new_tokens": 16, "temperature": 0.0},
    )
    print("text:", out1["text"])
    print("meta:", json.dumps(out1.get("meta_info", {}), default=str, indent=2))

    print("\n=== Request 2 (warm cache, shared prefix) ===")
    out2 = engine.generate(
        prefix + " known as",
        sampling_params={"max_new_tokens": 16, "temperature": 0.0},
    )
    print("text:", out2["text"])
    print("meta:", json.dumps(out2.get("meta_info", {}), default=str, indent=2))

    cached = out2.get("meta_info", {}).get("cached_tokens", 0)
    print(f"\ncached_tokens on req 2: {cached}")
    assert cached > 0, "Expected radix cache hit on second request with shared prefix"

    engine.shutdown()
    print("\n✓ Stage 1 PASSED")


if __name__ == "__main__":
    main()
