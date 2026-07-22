#!/usr/bin/env python3
"""Apply the four SGLang OOT-compatibility patches needed for the Spyre backend.

These teach SGLang core that an out-of-tree (OOT) device like Spyre is not CUDA
and not Triton-capable, so it skips CUDA `sgl_kernel`, the vLLM RoPE kernel, and
the Triton req->token writer, and accepts "spyre" as a supported device.
Idempotent: already-patched files are skipped.

Run inside the venv where sglang is installed:
    python sglang_oot_patches/apply.py
"""
from __future__ import annotations

import os
import sys


def _sglang_srt_dir() -> str:
    # sglang.srt may be a namespace package whose __file__ is None; resolve via
    # a concrete submodule that always has a real path. We want the srt/ dir
    # (patch paths are relative to srt/, not the outer sglang/).
    import sglang.srt.configs.device_config as _probe

    # _probe.__file__ -> .../sglang/srt/configs/device_config.py
    # we want         -> .../sglang/srt/
    return os.path.dirname(os.path.dirname(_probe.__file__))


PATCHES = [
    # (relative path, old, new)
    (
        "mem_cache/memory_pool_host.py",
        "if not (_is_npu or _is_xpu or _is_mps):",
        "if not (_is_npu or _is_xpu or _is_mps) and _is_cuda:  # OOT/Spyre: skip CUDA sgl_kernel",
    ),
    (
        "layers/rotary_embedding/base.py",
        (
            "            else:\n"
            "                from vllm._custom_ops import rotary_embedding\n"
            "\n"
            "            self.use_fallback_kernel = True\n"
            "            self.fallback_rotary_embedding = rotary_embedding"
        ),
        (
            "            else:\n"
            "                try:\n"
            "                    from vllm._custom_ops import rotary_embedding\n"
            "                except ImportError:\n"
            "                    # OOT/Spyre: no vllm kernels — use the torch-native RoPE.\n"
            "                    rotary_embedding = None\n"
            "\n"
            "            if rotary_embedding is None:\n"
            "                self.use_fallback_kernel = False\n"
            "            else:\n"
            "                self.use_fallback_kernel = True\n"
            "                self.fallback_rotary_embedding = rotary_embedding"
        ),
    ),
    (
        "utils/common.py",
        'return backend not in ["torch_native", "intel_amx"]',
        'return backend not in ["torch_native", "intel_amx", "spyre", "spyre_paged"]',
    ),
    # 4. configs/device_config.py — the device whitelist rejects any device not
    #    in SUPPORTED_DEVICES, so device="spyre" raises at the first load_model
    #    step. Allow OOT devices so the model body can be materialised on Spyre.
    (
        "configs/device_config.py",
        'SUPPORTED_DEVICES = ["cuda", "xpu", "hpu", "cpu", "npu", "musa", "mps"]',
        'SUPPORTED_DEVICES = ["cuda", "xpu", "hpu", "cpu", "npu", "musa", "mps", "spyre"]  # OOT/Spyre',
    ),
]


def main() -> int:
    root = _sglang_srt_dir()
    print(f"sglang.srt at: {root}")
    for rel, old, new in PATCHES:
        path = os.path.join(root, rel)
        with open(path) as f:
            src = f.read()
        if new in src:
            print(f"  [skip] {rel} already patched")
            continue
        if old not in src:
            print(f"  [WARN] {rel}: anchor not found (sglang version drift?) — review manually")
            continue
        with open(path, "w") as f:
            f.write(src.replace(old, new, 1))
        print(f"  [ok]   {rel} patched")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
