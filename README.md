# sglang-spyre-backend

SGLang backend plugin for the IBM Spyre AIU accelerator. Runs SGLang's
RadixCache scheduling on Spyre, demonstrating that radix-style prefix
sharing and Spyre compute coexist end-to-end.

This is a working prototype, not a production backend. It validates the
plugin architecture, the `MultiPlatformOp` dispatch path, and the
runtime gotchas needed to make Spyre kernels reachable through SGLang.

## Three runtime modes

Pick one via `attention_backend=...` on `sgl.Engine`:

| Mode | `attention_backend` | Attention math | KV cache | Use case |
|---|---|---|---|---|
| **1. Radix on Spyre** | `"spyre_paged"` | `_attn_4d` on Spyre | **on-Spyre** (overwrite_f scatter, BMM gather) | Real radix-on-device demo. Prefix hits avoid recompute AND re-transfer. |
| **2. Spyre attention only** | `"spyre"` | `_attn_4d` on Spyre | CPU (SGLang default) | Subclass of `TorchNativeAttnBackend`. KV stays in SGLang's pool; attention runs on AIU. |
| **3. CPU radix** | `"torch_native"` | torch SDPA on CPU | CPU | Baseline / debugging. No Spyre compute. |

All three exercise the same SGLang RadixCache scheduler. Mode 1 is what
"radix on Spyre" actually means; mode 2 is the simplest end-to-end Spyre
path; mode 3 is the apples-to-apples baseline.

## What works today

- **Plugin auto-discovery**: SGLang's `sglang.srt.platforms` entry point
  picks up our `SpyreSRTPlatform` automatically. `current_platform`
  becomes our class with `device_type="spyre"`.
- **Full Spyre attention through SGLang**: every attention layer runs
  the complete `_attn_4d` kernel (q@k.T, softmax, p@v) on Spyre — same
  shape conventions as spyre-inference. Validated with Granite-1B (BF16)
  on the dev.spyre pod.
- **On-device model body** via `device_ops.py`: RMSNorm / SiluAndMul /
  RoPE registered as Spyre-aware OOT forwards through SGLang's
  `MultiPlatformOp` surface. The residual stream stays on the device
  across the forward pass — CPU↔Spyre boundary crossed once per request,
  not once per attention layer.
- **OOT compatibility patches**: `sglang_oot_patches/apply.py` applies
  three idempotent patches to SGLang core (memory_pool_host CUDA-only
  guard, optional vllm._custom_ops in rotary_embedding, support_triton
  exclusion). Replaces hand-edits in the SGLang clone.
- **RadixCache prefix sharing**: 4 prompts sharing a 51-token prefix
  show `cached_tokens=51` on requests 2-4 (~82% prompt cache hit).
- **TPS-equivalent bench**: matches the structure of the vLLM
  `spyre_tps_bench.py` for apples-to-apples timing.

## Reuse over bespoke code

The backend leans on existing SGLang modules instead of reimplementing them:

- **Attention** (`attention_native.py`): `SpyreAttnBackend` now *subclasses*
  SGLang's `TorchNativeAttnBackend` and overrides only the inner SDPA seam
  (`_run_sdpa_forward_extend`/`_decode`) with the bucketed `_attn_4d`. The KV
  gather, pool save, GQA/causal handling and `forward_extend`/`forward_decode`
  entry points are inherited, not re-implemented. Replaces the old
  `attention_simple.py`.
- **Graph runner**: deleted. SGLang core sets `graph_runner=None` itself when
  `support_cuda_graph()` is `False`, so the hand-written `SpyreEagerGraphRunner`
  was redundant.
- **KV pool** (`memory_pool.py`): real `MHATokenToKVPool` subclass (was an
  identity re-export). The base already allocates on `self.device`; Phase C
  overrides `_create_buffers` to place buffers on Spyre.
- **`_attn_4d` / bucketing**: imported from `spyre_inference.kernels` when
  available (single canonical source), with a local fallback until that
  vLLM-free module is published (Phase B).

### Demo result (Granite-1B, 8 output tokens)

```
Request 1 (cold — Spyre kernels compile)...
  text:  Paris, and the capital of France
  time: 22.3s

Request 2 (warm — kernels cached)...
  text:  Paris, and the capital of France
  time: 0.8s
  cached: 4
```

**~28× wall-clock speedup** from compile cache + RadixCache hit.

## What does NOT work yet

- **KV cache on Spyre device**: currently lives on CPU and gets
  transferred per-call. This means RadixCache prefix hits avoid
  *recomputation* (the scheduler sees the prefix as already-attended)
  but not *re-transfer* — every forward still uploads the prefix KV.
  True radix-on-Spyre needs the BMM scatter/gather backend
  (`attention.py`, currently behind `"spyre_paged"`) which fails on
  torch_spyre's stick-layout constraints for some shapes.
- **Custom layer ops**: RMSNorm, RotaryEmbedding, ParallelLMHead all
  run on CPU. spyre-inference replaces these via vLLM's
  `register_oot()`; we'd do the same via SGLang's
  `MultiPlatformOp.register_oot_forward`.
- **TP > 1**, **batch > 1**, **dynamic shapes** — none supported.

## Architecture

```
SGLang Engine (CPU compute except attention)
   ↓
SpyreSRTPlatform (sglang.srt.platforms entry point)
   ↓ get_default_attention_backend() → "spyre"
SimpleSpyreAttentionBackend
   ├── q,k,v 2D from SGLang's RadixAttention (CPU)
   ├── view q as 3D, gather KV slice from CPU pool
   ├── pad query to QUERY_CHUNK_SIZE=32, KV to KV_LENGTH_ALIGNMENT=256
   ├── reshape to spyre-inference's _attn_4d convention:
   │     q: [num_seqs * num_kv_heads, num_queries_per_kv, padded_q, head_size]
   │     k: [num_seqs * num_kv_heads, 1, aligned_kv, head_size]
   │     mask: [num_seqs * num_kv_heads, 1, padded_q, aligned_kv]
   ├── move to Spyre, run _attn_4d (q@k.T → softmax → p@v) ← all on AIU
   ├── move output back to CPU, unpad query dim
   └── return as [num_actual_tokens, num_q_heads * head_dim]
```

## Critical setup gotchas

These are non-obvious and cost real time to discover. Captured here so
future readers don't repeat them.

### spyreccl env vars MUST be set before torch_spyre import

`libspyre_comms.so` reads `RANK`, `WORLD_SIZE`, `LOCAL_RANK`,
`LOCAL_WORLD_SIZE` at dlopen time. Setting them late silently corrupts
the device runtime in ways that surface as `meta_tensor` no_dispatch
AssertionError hundreds of frames deep in Dynamo.

We set them in both `activate()` and `init_backend()` via
`os.environ.setdefault(...)` before any torch_spyre import.

### addmm decomposition workaround (torch-spyre issue #1420)

```python
import torch._inductor.decomposition
from torch_spyre._inductor.decompositions import spyre_decompositions
for op, impl in spyre_decompositions.items():
    if "addm" in op.name():
        torch._inductor.decomposition.decompositions[op] = impl
```

Without this, the first FX trace of any Spyre op crashes. spyre-inference
applies the same workaround in its worker.

### SGLang's `support_triton` allowlist

`sglang/srt/utils/common.py::support_triton(backend)` returns True for
backends not in `["torch_native", "intel_amx"]`. Our `"spyre"` and
`"spyre_paged"` backends are NOT triton-capable but get treated as such
— SGLang's allocator then calls a Triton kernel that needs CUDA driver.

Fix: patch the SGLang clone:
```bash
sed -i 's|return backend not in \["torch_native", "intel_amx"\]|return backend not in ["torch_native", "intel_amx", "spyre", "spyre_paged"]|' \
    python/sglang/srt/utils/common.py
```

### flashinfer crashes at import even when device=cpu

`flashinfer.comm.cuda_ipc` runs `cudart = CudaRTLibrary()` at module
load with an `assert` for libcudart. SGLang's `try/except ImportError`
doesn't catch the AssertionError. Fix: uninstall flashinfer.

## Running the demos

On the Spyre pod (after `source /etc/bashrc`):

```bash
source $HOME/dt-inductor/spyre-inference/.venv/bin/activate
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PATH=/opt/ibm/spyre/deeptools/bin:/opt/ibm/spyre/runtime/bin:/opt/ibm/spyre/senlib/bin:/opt/ibm/spyre/tvm/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

# One-time: apply OOT patches to the SGLang clone
python3 -m sglang_oot_patches.apply
```

### Mode 3 — CPU radix baseline

```bash
python3 tests/test_cpu_smoke.py        # smoke + cached_tokens > 0
```

### Mode 2 — Spyre attention, KV on CPU

```bash
python3 tests/test_radix_demo.py       # 4 prompts shared prefix, cached_tokens=[0,51,50,51]
```

### Mode 1 — Radix on Spyre (KV on device)

```bash
python3 tests/test_radix_spyre.py      # same demo but with attention_backend="spyre_paged"
```

## Repository layout

```
sglang_spyre_backend/
├── __init__.py            ← activate() entry point + env-var setup + device stubs
├── platform.py            ← SpyreSRTPlatform: factory methods, init_backend
├── device.py              ← SpyreDeviceMixin: device identity + memory query
├── register.py            ← Registers "spyre" + "spyre_paged" backends with sglang
├── attention_native.py    ← "spyre": thin subclass of TorchNativeAttnBackend,
│                             overrides the SDPA seam with _attn_4d (KV on CPU)
├── attention.py           ← "spyre_paged": full BMM scatter/gather, paged KV on Spyre
├── memory_pool.py         ← SpyreMHATokenToKVPool: real MHATokenToKVPool subclass
├── paged_allocator.py     ← Torch-native paged KV allocator (no Triton)
├── device_ops.py          ← RMSNorm/SiluAndMul/GeluAndMul/RoPE Spyre forwards
│                             via MultiPlatformOp OOT registry
├── model_wrapper.py       ← bodify_model_for_spyre: move model body onto device
└── custom_ops/            ← max-on-Spyre wiring
    ├── __init__.py            ← install(): OOT op forwards + post-load hook
    └── weight_placement.py    ← move Linear weights to Spyre, patch quant_method

tests/
# Automated (pytest-collectable, real assertions):
├── test_reuse_structure.py  ← subclass/reuse invariants (no device needed)
├── test_attention_smoke.py  ← SpyreAttentionBackend in isolation (BMM path)
# Smoke / stage checks (run by hand on the pod):
├── test_cpu_smoke.py        ← Stage 1: SGLang CPU baseline
├── test_stage2_platform.py  ← Stage 2: platform plugin discovery
├── test_max_on_spyre.py     ← Mode 4: Linear + RMSNorm/SiluAndMul/RoPE on Spyre
# Demos (print-based, run by hand):
├── test_spyre_attention.py  ← End-to-end: Engine + Spyre attention
├── test_radix_demo.py       ← Mode 2 demo: 4 prompts, shared prefix
├── test_radix_spyre.py      ← Mode 1 demo: same, attention_backend="spyre_paged"
# Hardware probes (torch-spyre spikes, not package tests):
├── test_compile_boundary.py ← Probe: torch.compile boundary on Spyre
├── test_eager_op.py         ← Probe: torch.ops.spyre.* eager dispatch
├── test_eager_op2.py        ← Probe: basic eager ops
└── sglang_tps_bench.py      ← TPS bench, matches vLLM/spyre-inference shape
```

## Comparison to spyre-inference

| Aspect | spyre-inference (vLLM) | This (SGLang) |
|---|---|---|
| Spyre compute | full attention math + LM head + RMSNorm | full attention math (`_attn_4d`) |
| Scheduler | vLLM | SGLang (RadixCache) |
| Prefix sharing | block-level | radix tree, arbitrary length |
| TP | 1 | 1 |
| KV cache | on-Spyre paged | on-CPU paged (re-transferred per call) |
| Maturity | production | prototype |

Both run the same `_attn_4d` kernel on Spyre with the same shape
conventions (KV bucketed to 256, query bucketed to 32). What
spyre-inference does that we don't:
- KV cache lives on Spyre persistently; scatter/gather via BMM avoids
  per-call re-transfer of the prefix.
- RMSNorm / RotaryEmbedding / LMHead replaced with Spyre-native ops.

Next milestone: KV cache on Spyre device so RadixCache prefix hits also
skip the per-call KV transfer.
