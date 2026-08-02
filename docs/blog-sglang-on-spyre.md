# SGLang Comes to the IBM Spyre AIU

SGLang now runs on the IBM Spyre AIU, IBM's purpose-built accelerator for
enterprise inference. Developers building on Spyre can serve models through
SGLang, and the SGLang community gains support for another class of hardware.

The integration ships as an out-of-tree platform plugin. SGLang discovers Spyre
at startup through its platform entry point and treats the AIU as a device it
knows about — no fork, no vendored copy of the framework. The scheduler drives
the accelerator, attention executes on the device, and the model body and KV
cache can both live there too.

Below is what the enablement actually took.

## Prefix caching came along for free

The result we were most curious about was RadixCache, and it turned out to need
nothing device-specific at all. That is worth explaining, because it says
something about where SGLang draws its abstraction line.

RadixCache lives in the scheduler, not in the attention backend. Incoming
requests are matched against a radix tree of token IDs, the longest already-cached
prefix is found, and those tokens are mapped to slots that already hold their keys
and values. All of this happens *before* the forward pass is assembled. What the
attention backend receives is a table mapping each request's positions to physical
KV slots, with the cache hit already baked into it — so from the device's point of
view, a prefix hit is simply work that never arrives. The backend is never asked
to make a caching decision, and never has to know one happened.

The practical consequence for a new backend is that prefix sharing is not a
feature you port. Four requests sharing a system prompt reuse it on Spyre for the
same reason they do anywhere else: the scheduler resolved it upstream.

One detail of that interface suits non-GPU silicon nicely. SGLang addresses the
KV cache per token rather than in fixed-size blocks, so there is no page-size
constant threading down into device kernels, and a backend is free to choose its
own memory layout without fighting an assumption baked in above it.

## Attention was one seam, not a rewrite

Spyre's attention kernel expects static shapes: query and key/value lengths
bucketed to fixed sizes, with padding to the next bucket. We expected fitting that
into a serving framework to be the hard part.

It wasn't, because SGLang's torch-native attention backend is a real, inheritable
base class rather than an interface to reimplement. We subclassed it and replaced
exactly one thing — the innermost attention computation — with the Spyre kernel
and its bucketing. Everything around that came for free: gathering keys and values
out of the pool, writing new ones back, the split between prefill and decode, and
grouped-query and causal handling. The device-specific code ends up being the
kernel and the shape bookkeeping it needs, which is where hardware code belongs.

## The rest, and what we deleted

Layer operations — normalisation, activations, rotary embeddings — register
against the device through SGLang's multi-platform dispatch surface. With those in
place the model body runs on the AIU with the residual stream staying resident
across the forward pass, so the boundary between host and device is crossed once
per forward rather than once per layer. The KV cache pool is device-agnostic in
its base class, which makes moving the cache onto the accelerator a contained
change.

Some of the enablement was removing our own code. An early version carried a
hand-written graph runner until we found core already handles devices without
graph capture — SGLang has an explicit path for out-of-tree platforms and does the
right thing unprompted. We also needed a handful of one-line compatibility fixes
where core assumed a CUDA-like device; none were Spyre-specific, and the project
has since addressed most of them upstream.

Work continues: larger batches and continuous batching, tensor parallelism,
dynamic shapes as the Spyre toolchain evolves, and broader model coverage. If you
work on inference frameworks, accelerators, or the glue between them, we would
welcome the company.
