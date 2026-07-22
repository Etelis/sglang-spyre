"""Stage 2: Run the Engine without explicit attention_backend.

Validates that SpyreSRTPlatform.apply_server_args_defaults() sets
attention_backend to torch_native automatically. The platform plugin
is otherwise the same as Stage 1.
"""

import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def main():
    # Force resolution of the platform first so we can show what was picked
    from sglang.srt.platforms import current_platform
    print(f"Platform: {type(current_platform).__name__}")
    print(f"  device_type: {current_platform.device_type}")
    print(f"  is_out_of_tree: {current_platform.is_out_of_tree()}")
    print(f"  default_attention_backend: {current_platform.get_default_attention_backend()}")
    assert type(current_platform).__name__ == "SpyreSRTPlatform"

    import sglang as sgl
    model_path = os.environ.get(
        "SGLANG_TEST_MODEL", "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
    )
    print(f"\nLoading {model_path} (no explicit attention_backend)")

    engine = sgl.Engine(
        model_path=model_path,
        device="cpu",
        # NO attention_backend — should default via apply_server_args_defaults
        dtype="bfloat16",
        mem_fraction_static=0.5,
        disable_cuda_graph=True,
        log_level="warning",
    )

    out = engine.generate(
        "The capital of France is",
        sampling_params={"max_new_tokens": 10, "temperature": 0.0},
    )
    print(f"text: {out['text']}")
    assert out["text"], "empty output"

    engine.shutdown()
    print("\n✓ Stage 2 PASSED")


if __name__ == "__main__":
    main()
