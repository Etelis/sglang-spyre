# SGLang Comes to the IBM Spyre AIU

IBM & the SGLang community

August 2, 2026

SGLang now runs on the IBM Spyre AIU, giving developers another hardware choice
for serving models with the same API and the same features they already use.

Spyre is IBM's accelerator for enterprise inference, shipping in IBM Z, LinuxONE
and Power systems. Support for it lands as an out-of-tree SGLang platform plugin:
SGLang discovers it at startup through the standard platform entry point and
treats the AIU as a device it knows how to target, with no fork of SGLang and no
vendored copy of the framework.

Developers can serve models on Spyre today through the usual `sgl.Engine` API,
with SGLang's scheduler, request handling and sampling running unchanged.
Attention executes on the accelerator in BF16 — the query-key product, the softmax
and the value product. The model body can run there too: RMSNorm, SiLU and GeLU,
rotary embeddings and the linear layers all have Spyre implementations registered
through SGLang's multi-platform op dispatch, with the residual stream staying
resident on the device across the forward pass rather than returning to the host
between layers. The KV cache can live on the accelerator as well, in a paged pool
backed by a torch-native allocator that pulls in no Triton dependency. Two
attention backends are selectable: `spyre` keeps the cache in SGLang's pool, and
`spyre_paged` moves it onto the device.

RadixCache prefix sharing works on Spyre, and it needed no device-specific code —
which says something about where SGLang draws its abstraction line. Prefix
matching happens in the scheduler, not the backend: a request's tokens are looked
up in a radix tree, the longest cached prefix is resolved to the KV slots already
holding its keys and values, and all of that finishes before the forward pass is
assembled. What reaches the attention backend is a table of positions to physical
slots with the hit already encoded, so a prefix hit is simply work that never
arrives at the device. SGLang also indexes that cache per token rather than in
fixed-size blocks, which means no page size leaks down into device kernels — a
useful property on hardware like Spyre, whose attention kernels want statically
shaped, bucketed inputs.

Work continues on continuous batching and larger batch sizes, tensor parallelism,
dynamic shapes as the Spyre compiler adds support for them, the low-precision
formats the AIU handles natively, and broader model coverage.

> *[Quote from IBM Spyre leadership — needs real attribution before publication]*

> *[Quote from the SGLang team — needs real attribution before publication]*

The plugin is developed in the open and available today at
[github.com/torch-spyre/sglang-spyre](https://github.com/torch-spyre/sglang-spyre).

The SGL family always welcomes new contributors to help build an open,
high-performance serving engine together!
