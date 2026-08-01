# Proposal: make Triton capability a platform predicate

**Status:** draft, not yet filed upstream.
**Applies to:** `sglang/srt/utils/common.py`, `sglang/srt/platforms/interface.py`.

## The problem

`support_triton()` is a denylist:

```python
def support_triton(backend: str) -> bool:
    return backend not in ["torch_native", "intel_amx"]
```

Every backend not named in that list is assumed Triton-capable. An out-of-tree
platform with no Triton backend therefore answers `True` by default, and the
three call sites go on to dispatch Triton kernels the device cannot run:

- `mem_cache/common.py` — `write_req_to_token_pool_triton` in the KV-cache
  allocator
- `model_executor/forward_batch_info.py:1417`
- `layers/rotary_embedding/mrope.py:220`

On the Spyre AIU this surfaces as `RuntimeError: 0 active drivers`, raised from
inside the allocator, some distance from anything the backend author wrote. The
failure is not obviously about Triton, and is not obviously about the fact that
the platform never claimed to support it.

Adding each new device name to the denylist does not scale, and asks every
out-of-tree backend to discover this the same way.

## Why now

The three sibling out-of-tree gaps we hit have all since been fixed upstream,
each by asking the platform what it is rather than pattern-matching a name:

| Site | Fix |
|---|---|
| `mem_cache/memory_pool_host.py` | guard on `_is_cuda or _is_hip` |
| `layers/rotary_embedding/base.py` | `and not (current_platform.is_out_of_tree())` |
| `configs/device_config.py` | `if device in SUPPORTED_DEVICES or current_platform.is_out_of_tree()` |

`support_triton()` is the remaining instance of the older pattern. This proposal
follows the same direction, and mirrors the existing `support_cuda_graph()`
predicate rather than inventing a new mechanism.

## Proposed change

### 1. `platforms/interface.py` — add the predicate

Alongside `support_cuda_graph()` on `SRTPlatform`:

```python
def support_triton(self) -> bool:
    """Whether Triton kernels can be dispatched on this platform.

    Gates Triton kernel selection in the KV-cache allocator, mrope, and the
    forward-batch path. Out-of-tree platforms default to False, since a
    platform that has not been built against Triton cannot run those kernels;
    an OOT platform that does have Triton should override this to True.
    """
    return not self.is_out_of_tree()
```

`SRTPlatform` already extends `DeviceMixin`, which provides `is_out_of_tree()`,
so no new plumbing is required.

### 2. `utils/common.py` — consult the platform first

```diff
 def support_triton(backend: str) -> bool:
-    return backend not in ["torch_native", "intel_amx"]
+    if not current_platform.support_triton():
+        return False
+    return backend not in ["torch_native", "intel_amx"]
```

## Behaviour

- **In-tree platforms** (cuda, rocm, npu, xpu, cpu, …): `is_out_of_tree()` is
  False, so the predicate returns True and the denylist decides exactly as
  before. No behaviour change.
- **Out-of-tree platforms**: default to False and stop dispatching kernels they
  cannot execute. A backend that does have Triton opts back in with a one-line
  override — the same shape as opting into `support_cuda_graph()`.

The default is deliberately the conservative direction. A platform wrongly
answering False loses a fast path; a platform wrongly answering True crashes
inside an allocator, which is the situation today.

## Alternative considered

Adding `"spyre"` and `"spyre_paged"` to the denylist. That is what this repo
currently patches in, and it is what we would like to stop doing: it fixes one
device rather than the category, and the next out-of-tree backend rediscovers
the same crash.
