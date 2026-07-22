# SGLang out-of-tree (OOT) compatibility patches for Spyre

Running the Spyre backend end-to-end through SGLang's `Engine` surfaced four
places where SGLang core assumes a non-CUDA device must be NPU / XPU / MPS and
overlooks out-of-tree (OOT) platforms like Spyre. Each is a small, upstreamable
fix. They are applied to the **installed SGLang** (not this plugin); `apply.py`
patches them in place against `sglang==0.5.12.post1`.

Validated on the dev.spyre pod: with these four patches plus this plugin,
`tests/test_radix_demo.py` generates on the AIU and RadixCache prefix-sharing
works (`cached_tokens = [0, 51, 50, 51]`, 82% prompt cache hit on Granite-1B).

## The four gaps

1. **`mem_cache/memory_pool_host.py:46`** — imports CUDA `sgl_kernel.kvcacheio`
   for every non-`npu`/`xpu`/`mps` device. OOT/Spyre has no CUDA `sgl_kernel`
   (needs `libnvrtc.so.13`, `libcudart`, …). Fix: also require `_is_cuda`.

   ```diff
   - if not (_is_npu or _is_xpu or _is_mps):
   + if not (_is_npu or _is_xpu or _is_mps) and _is_cuda:  # OOT/Spyre: skip CUDA sgl_kernel
   ```

2. **`layers/rotary_embedding/base.py:~90`** — the non-cuda/non-hip fallback path
   does `from vllm._custom_ops import rotary_embedding`, hard-requiring vLLM.
   OOT/Spyre has no vLLM. Fix: make the import optional and fall back to the
   torch-native RoPE (what cpu/npu already do).

   ```diff
   -            else:
   -                from vllm._custom_ops import rotary_embedding
   -
   -            self.use_fallback_kernel = True
   -            self.fallback_rotary_embedding = rotary_embedding
   +            else:
   +                try:
   +                    from vllm._custom_ops import rotary_embedding
   +                except ImportError:
   +                    # OOT/Spyre: no vllm kernels — use the torch-native RoPE.
   +                    rotary_embedding = None
   +
   +            if rotary_embedding is None:
   +                self.use_fallback_kernel = False
   +            else:
   +                self.use_fallback_kernel = True
   +                self.fallback_rotary_embedding = rotary_embedding
   ```

3. **`utils/common.py:379` `support_triton()`** — returns True for any backend
   except `torch_native`/`intel_amx`, so `"spyre"` is treated as Triton-capable
   and `write_cache_indices` (mem_cache/common.py) dispatches the
   `write_req_to_token_pool_triton` kernel → `RuntimeError: 0 active drivers`
   (no Triton on Spyre). Fix: exclude the Spyre backends.

   ```diff
   - return backend not in ["torch_native", "intel_amx"]
   + return backend not in ["torch_native", "intel_amx", "spyre", "spyre_paged"]
   ```

4. **`configs/device_config.py`** — `SUPPORTED_DEVICES` is a whitelist that
   rejects any device not in the list, so `device="spyre"` raises at the first
   `load_model` step. Fix: add `"spyre"` so the model body can be materialised
   on the AIU (the max-on-Spyre path).

   ```diff
   - SUPPORTED_DEVICES = ["cuda", "xpu", "hpu", "cpu", "npu", "musa", "mps"]
   + SUPPORTED_DEVICES = ["cuda", "xpu", "hpu", "cpu", "npu", "musa", "mps", "spyre"]  # OOT/Spyre
   ```

## Apply

```sh
python sglang_oot_patches/apply.py        # patches the installed sglang in-place
```

Idempotent: re-running detects already-patched files and skips them.
