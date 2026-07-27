# A Growing Stack: Bringing SGLang to Spyre

Four prompts sharing a 51-token system prompt. The first one pays for the
prefix; the next three don't. On the IBM Spyre AIU, running through SGLang's
scheduler, that looks like this:

```
cached_tokens = [0, 51, 50, 51]
```

An 82% prompt cache hit on Granite-1B, with every attention layer's math
executing on the accelerator. That's SGLang's RadixCache scheduler and Spyre
compute coexisting end-to-end.

This post is about what it took, what the framework gave us for free, what it
didn't, and — since this is a prototype and not a product — what still doesn't
work.

## What actually runs

The Spyre stack already had one way to serve models: the `spyre-inference`
plugin for vLLM, which is the production path. We wanted a second one, for
reasons we'll get to. The result is an out-of-tree SGLang platform plugin that
SGLang discovers through its own `sglang.srt.platforms` entry point — no fork,
no vendored copy of the framework.

It runs in a few configurations, selected by `attention_backend` on
`sgl.Engine`:

- **`torch_native`** — the CPU baseline. SGLang's RadixCache scheduler, no
  Spyre compute. This is the control.
- **`spyre`** — every attention layer runs the full `_attn_4d` kernel
  (q@kᵀ, softmax, p@v) on the AIU, using the same shape conventions as
  `spyre-inference`: KV bucketed to 256, query bucketed to 32. The KV cache
  stays in SGLang's CPU pool.
- **`spyre` with the model body on device** — RMSNorm, SiluAndMul, RoPE and
  the Linear layers also execute on Spyre, with the residual stream staying
  resident across the forward pass so the CPU↔Spyre boundary is crossed once
  per forward instead of once per attention layer.
- **`spyre_paged`** — the KV cache itself living on the device. This is the
  one that isn't finished; more on that below.

All of them run the same SGLang scheduler, which is the point. The prefix
sharing above isn't something the backend participates in. SGLang's RadixCache
lives in the scheduler proper, resolves the shared prefix to KV-pool slots
before the forward begins, and hands the attention backend a `req_to_token`
table that already encodes the hit. From the device's perspective, a cache hit
is simply work that never arrives.

## Why it took a third less code

Counted the same way on both sides — non-blank lines of Python, plugin code
only, no tests — `spyre-inference` is 2,797 lines and the SGLang plugin is
1,674. Both target the same hardware, call the same Spyre kernels, and work
around the same toolchain constraints. The ratio holds under raw line counting
too (3,416 against 2,021), so it isn't an artifact of how we counted.

Where that difference sits is more interesting than its size.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/plugin-size-by-component-dark.svg">
  <img src="figures/plugin-size-by-component-light.svg" width="700"
       alt="Component-by-component comparison of the two plugins. The Worker and
       ModelRunner category is 483 lines in the vLLM plugin and absent entirely
       from the SGLang plugin; attention is 1,142 against 696.">
</picture>

| Component | spyre-inference (vLLM) | sglang-spyre (SGLang) |
|---|---:|---:|
| Attention | 1,142 | 696 |
| Worker + ModelRunner | 483 | 0 |
| Custom ops (norm, activation, RoPE, linear) | 516 | 421 |
| Platform + registration | 187 | 268 |
| KV pool + allocator | inside ModelRunner | 111 |
| Init, env, boundary hooks | 163 | 178 |
| **Comparable subtotal** | **2,491** | **1,674** |
| Tensor-parallel communicator | 184 | 0 |
| Parallel LM head | 122 | 0 |
| **Total** | **2,797** | **1,674** |

One deduction to make before drawing any conclusion from that. The bottom two
rows are things the SGLang plugin doesn't do at all: there's no tensor-parallel
communicator because we don't support TP > 1, and no on-device LM head. That's
306 lines, roughly a quarter of the gap, and it's a missing feature rather than
a framework saving. On comparable surface the comparison is 1,674 against
2,491 — about a third less code, which is the number we'd defend.

The largest single contributor isn't attention. It's the 483-line Worker and
ModelRunner block, which has no counterpart at all. vLLM asks you to subclass
`Worker` and `ModelRunner`, own the per-process model, KV pool and sampler, and
implement `execute_model()` as the per-step entry point. SGLang asks for a
platform object with factory methods and an `init_backend()` lifecycle hook.
There is no worker to write, so there is no worker in the table.

Attention is second, at 446 lines, and the reason is one design decision.
vLLM's `AttentionBackend` is an interface: every plugin reimplements KV gather,
block-table interpretation, extend-versus-decode dispatch, and GQA and causal
handling. SGLang ships `TorchNativeAttnBackend`, a real base class with working
defaults in pure torch. Our Spyre backend subclasses it and overrides exactly
one seam — the inner SDPA call — swapping `F.scaled_dot_product_attention` for
the bucketed `_attn_4d`. The gather, the pool save, the dispatch bookkeeping,
the causal masking: all inherited, none rewritten.

**No dual-buffer dance.** vLLM V1's sampler runs CPU-side and expects integer
inputs — token IDs, positions — to arrive on CPU while model tensors live on
the device. `spyre-inference` carries a `_SpyreModelWrapper` that does that
conversion at the model boundary on every single forward. In SGLang's
attention-only mode, no staging is required at all. In the on-device-body mode,
about ten lines of boundary hooks cover the same ground.

Two rows run the other way, and they're the honest cost of the trade. Our KV
pool and allocator are 111 lines that vLLM appears to spend nothing on — but
only because vLLM's equivalent is buried inside that 483-line ModelRunner
rather than being free. Having it as a separate, subclassable
`MHATokenToKVPool` is what makes device-resident KV a contained change instead
of a rewrite. And our platform layer is genuinely bigger, 268 lines against
187, because SGLang front-loads into the platform object what vLLM spreads
across a Worker: factories for the KV pool, the allocator and the attention
backend, plus device identity and backend registration. That's the shape of the
trade, and at this ratio we'd take it every time, but it isn't free.

We also deleted code we'd written. An early version had a hand-rolled
`SpyreEagerGraphRunner`, until we found that SGLang core sets
`graph_runner = None` on its own when `support_cuda_graph()` returns `False`.
The right amount of code for that problem was zero.

One more structural detail that matters on this hardware: SGLang addresses the
KV cache per token (`req_to_token[req, slot] → token_idx`) where vLLM addresses
it per block (`block_table[seq, page] → block_id`). Page size never has to
thread through the Spyre kernels, `page_size=1` works without special cases,
and the 256-token KV bucketing stays a decision local to the kernel dispatcher
rather than a property of the storage layout. It's also a closer fit for the
indexed-gather lowering Spyre is moving toward.

## What the framework didn't do for us

A good extension surface lowers the cost of integration. It does not eliminate
it, and claiming otherwise would be easy to disprove by cloning the repo.

We ship four patches against installed SGLang. They're small, idempotent, and
each one is a place where SGLang core assumes that a non-CUDA device must be
NPU, XPU or MPS, and doesn't yet consider that an out-of-tree platform might
exist:

1. `memory_pool_host.py` imports a CUDA `sgl_kernel` for every device that
   isn't NPU/XPU/MPS. Spyre has no `libcudart`. Fix: also require `_is_cuda`.
2. The rotary embedding base falls back to `from vllm._custom_ops import
   rotary_embedding` on non-CUDA paths, hard-requiring vLLM. Fix: make the
   import optional and fall back to torch-native RoPE, which is what CPU and
   NPU already do.
3. `support_triton()` returns `True` for any backend not explicitly excluded,
   so `"spyre"` gets treated as Triton-capable and the allocator dispatches a
   Triton kernel into `RuntimeError: 0 active drivers`. Fix: exclude the Spyre
   backends.
4. `SUPPORTED_DEVICES` is a whitelist that rejects anything unlisted, so
   `device="spyre"` raises at the first `load_model`. Fix: add it.

That's four one-line changes, all upstreamable, none of them Spyre-specific in
substance — they're SGLang out-of-tree readiness gaps that the next new backend
would hit too. We'd rather report that accurately than claim a zero-friction
integration, because "we needed four one-line fixes and all four belong
upstream" is a stronger statement about API quality than "we needed nothing"
would be.

Beneath the framework layer, the device had its own opinions. `libspyre_comms.so`
reads `RANK`, `WORLD_SIZE`, `LOCAL_RANK` and `LOCAL_WORLD_SIZE` at dlopen time;
set them late and the runtime corrupts quietly, surfacing hundreds of frames
deep in Dynamo as a `meta_tensor` no-dispatch assertion that tells you nothing
about the actual cause. The first FX trace of any Spyre op crashes without an
`addmm` decomposition override. `flashinfer` raises an `AssertionError` at
import when it can't find libcudart, which SGLang's `try/except ImportError`
correctly declines to catch. None of these are framework problems, and all of
them cost real time.

## The numbers, honestly

Two things are worth separating here: whether the kernels compute the right
answer, and how fast they do it. The first result is much better than the
second.

**Correctness.** Greedy decoding is deterministic, so a correct attention kernel
has to reproduce the CPU baseline token for token. Across four prompts with
unambiguous continuations — the capital of France, the freezing point of water,
the first three primes, and a counting sequence — the Spyre path is byte-identical
to CPU SDPA for every one of 24 generated tokens. The single prompt that diverges
is open-ended, and it diverges only after 78 identical characters, at the exact
point where the model falls into a repetition loop and the top logits are
effectively tied. That is bf16 rounding changing an argmax where the model has no
preference, not a masking or bucketing error. Running the same check with the
model body also on the device gives byte-identical output to the attention-only
mode, which localises any remaining numerical difference to `_attn_4d` and
clears the on-device RMSNorm, SiluAndMul, RoPE and Linear paths.

**Throughput.** All modes measured in one sitting on one machine — Granite-1B
(`micro-g3.3-8b-instruct-1b`, 4 layers, 32 query heads over 8 KV heads), 200
output tokens, batch of one, one Spyre PF card:

| Mode | TPS | ms/token | What runs on Spyre |
|---|---:|---:|---|
| `torch_native` (CPU baseline) | 37.64 | 26.6 | nothing |
| `spyre` | 6.13 | 163.2 | `_attn_4d`; KV on CPU |
| `spyre` + model body on device | 6.56 | 152.4 | Linear, RMSNorm, SiluAndMul, RoPE, attention |
| `spyre_paged` | (12.39) | (80.7) | KV on device — **fails parity, see below** |

The baseline needs its terms stated or the table misleads: it is an
unconstrained 144-core Xeon 6736P with 1.5 TB of RAM, and SGLang used one of its
two NUMA nodes. So this is seventy-odd server cores against a single accelerator
at batch size one — not a hardware comparison anyone should draw a conclusion
from, and we publish it because the shape of the gap is informative even though
its magnitude isn't a verdict.

Two things in that table are worth pointing at. Moving the model body onto the
device *helps* — 6.56 against 6.13 — which is the direction the design predicts,
since the residual stream stays resident and the CPU↔Spyre boundary is crossed
once per forward instead of once per attention layer. But the gain is 7%, far
short of what that reasoning implies, and it costs 50% longer model load
(143.8 s against 96.6 s) to place the weights.

The `spyre_paged` row is parenthesised because it is not a result. That mode
keeps the KV cache on the device, and it does run at 12.39 TPS — twice either
working mode, which is what the design predicts, since a prefix hit then avoids
re-transfer as well as recompute. But it fails the parity check: on "The first
three prime numbers are" it diverges from CPU at the very first generated token
and answers "1, 2, 3, 4, 5…". Both correct modes get that prompt right. So the
number is real and the answer is wrong, and a wrong answer arriving twice as
fast is not a speedup. We are reporting it only to be clear about what is and
isn't established.

That path taught us something about our own documentation, too. It had been
recorded as blocked on the Spyre toolchain's stick-layout constraints. It
wasn't: it was crashing on a token-count bug of ours — extend-mode metadata used
the full sequence length where `q` only carries the newly-extended tokens, which
are equal on a cold prefill and diverge on exactly a RadixCache prefix hit. The
on-device-radix backend was broken by the one thing it exists to do. Fixing that
got it running, and running is what exposed the second, still-open defect in the
gather path. Two bugs wearing one trench coat, and the outer one had been
misattributed to the vendor toolchain.

The reasons for the overall gap are understood and mostly not about SGLang. Every
distinct query and KV bucket compiles its own graph, so a warm run is still
paying dispatch overhead a fused GPU kernel wouldn't. In the two correct modes
the prefix KV lives on CPU, so a RadixCache hit saves recomputation but not
re-transfer — the scheduler correctly skips the prefill, then every forward
uploads the prefix KV anyway. And `num_seqs == 1` is hard-coded in both plugins'
attention backends, so continuous batching, the thing that makes serving
throughput interesting, isn't in play at all.

A comparison against vLLM on the same hardware would be the useful next
measurement, and we don't have one taken under these conditions. We'd rather
leave that blank than quote a number from a different machine — which is exactly
the mistake an earlier draft of this table made.

There is one genuinely large number, and it's about compile caching rather than
compute. On a short 8-token generation, the first request takes 22.3 seconds
while the Spyre kernels compile; the second takes 0.8. That ~28× is a real
property of the deployment model — it says warm pods matter — but it isn't a
throughput result and shouldn't be read as one.

## What changes next

Most of the gap above traces to a single constraint: Spyre's toolchain is
currently static-shape. Every distinct query and KV bucket is a separate
compiled graph. Runtime tensor indexing isn't expressible, so the on-device KV
path has to fake it — build an `[N_blocks, block_size, num_blocks]` one-hot
tensor on CPU, transfer it, and use a batched matmul as a runtime-indexable
gather. That's what `spyre_paged` does. Interestingly, the measurement above
suggests the one-hot BMM is *not* the bottleneck we assumed — that path is the
fastest of the three once it runs — but it is also the one that currently
computes the wrong answer, so its cost profile shouldn't be trusted until the
gather is correct.

The dynamic-shapes refactor underway in the Spyre toolchain addresses exactly
this. Variable-length attention without bucketing, `num_seqs > 1`, and — the
one we care about most — KTIR `construct_indirect_access_tile` lowering, which
replaces the one-hot BMM with real `cache[block_table[i]]` indirection.

That's the point where the architecture stops being theoretical. RadixCache's
advantage over block-level prefix sharing is that it shares at any token
boundary, where a block-paged cache has to round down to the last whole block
and re-prefill the remainder.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/prefix-sharing-granularity-dark.svg">
  <img src="figures/prefix-sharing-granularity-light.svg" width="700"
       alt="Block-aligned sharing rounds down to the last whole 16-token block,
       sharing 192 of 200 tokens and 48 of 51. Radix sharing ends where the
       prefix does, sharing 200 of 200 and 51 of 51.">
</picture>

On its own, three tokens is nothing. The point is that it isn't on its own: the
same tail is given up by every request that shares the prefix, and the shortfall
grows with how badly the prefix length misses a block boundary. For RAG with
shared templates, multi-turn chat, and agent flows with long tool prompts —
workloads where prefix lengths are set by application code and have no reason to
land on multiples of sixteen — that's the lever. It just needs the KV cache to
actually live on the device for the saving to be real.

## The part that outlasts the numbers

Here's the argument for doing this now, while the numbers are still
unimpressive.

The Spyre stack is moving. The lowering is being rewritten, the shape regime is
changing, kernels are being tuned. For anything sitting on top of that, the
useful question isn't "does it run today" but "does it help or hurt the next
time something underneath changes."

A full serving framework turns out to be a good regression harness, and not
incidentally. Several of the sharpest bugs we hit were only visible through the
complete path: the `support_triton` allowlist silently routing the allocator
into a CUDA-driver-dependent kernel is not something a unit test on the
attention backend would ever surface, because the allocator isn't part of the
attention backend. Neither is the memory-pool host import, or the RoPE fallback
reaching for vLLM. Those are integration failures, and they need an integration
to find them. Now that SGLang runs end-to-end on the AIU, every change to the
Spyre stack has a realistic serving workload to be checked against — with real
scheduling, real cache reuse, and a real request loop — rather than a
purpose-built script that only exercises what we already thought to test.

And because the integration rides on entry points and subclassable base
classes, with four upstreamable patches rather than a fork to reconcile, keeping
it current is cheap. That was the actual reason to be careful about *how* we
integrated, more than the line count.

## Where this leaves things

Spyre now runs SGLang. It's a prototype: single tenant, batch of one, no tensor
parallelism, on-device KV still in progress, and no configuration that beats CPU
on a 1B model yet. All of that is written down in the repo alongside the parts
that work.

What's genuinely established is the shape of the thing. RadixCache scheduling
and Spyre compute coexist end-to-end. The plugin surface holds up: a platform
registered through a documented entry point, an attention backend that inherits
almost everything, a KV pool one method away from device residency. And on
comparable functionality the integration cost is about a third less than the
same job took against vLLM, for structural reasons that won't change when the
toolchain does.

Two frameworks on one accelerator isn't a competition to resolve. They're good
at different things — vLLM's continuous batching scheduler has three more years
of tuning behind it and a broader serving-feature matrix, and SGLang's
any-length prefix sharing is the better fit for shared-prompt workloads. The
useful outcome is having both, and knowing concretely which one to reach for.

The code is public, the failure modes are documented, and the gotchas that cost
us days are written down so they don't cost anyone else the same. If you work on
inference frameworks, accelerators, or the glue between them, we'd welcome the
company.
