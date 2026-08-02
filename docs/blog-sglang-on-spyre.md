# A Growing Stack: Bringing SGLang to Spyre

**SGLang now runs on the IBM Spyre AIU.**

The Spyre software stack has been growing steadily, and it has just taken a
meaningful step forward. SGLang — one of the most widely adopted open-source
inference and serving frameworks — now runs on Spyre, giving developers a second
first-class way to serve models on the accelerator.

## What runs today

The integration is an out-of-tree platform plugin. SGLang discovers Spyre
automatically through its own platform entry point, and from that moment the
engine treats the AIU as a device it knows about. There is no fork of SGLang and
no vendored copy of the framework — the plugin lives alongside it.

From there, the full serving path works end to end:

- **SGLang's scheduler drives the accelerator.** Requests arrive through the
  standard engine, are scheduled by SGLang, and dispatch real work to Spyre.
- **Attention runs on the AIU**, using the same kernels and shape conventions the
  Spyre stack already uses elsewhere.
- **RadixCache prefix sharing works on Spyre.** Requests sharing a system prompt
  reuse the cached prefix, exactly as on any other backend.
- **The model body can live on the device** — normalisation, activations, rotary
  embeddings and the linear layers — with the residual stream staying resident
  across the forward pass.
- **The KV cache can live on the device too**, keeping cached prefixes where the
  compute is.

## How we enabled it

The work came down to filling in seams SGLang already provides, rather than
opening up its internals.

**Registering the platform.** SGLang looks for platform plugins at startup, so
declaring one is enough for the engine to pick up Spyre, adopt its device type,
and route through it. The platform object is where a backend answers the
questions the framework asks — which device this is, how to initialise it per
process, and which classes to use for the pieces that are device-specific. That
last part matters: SGLang asks for factories rather than assuming, so supplying a
KV cache pool, a paged allocator and an attention backend is a matter of naming
them.

**Attention.** This is the part we expected to be hardest and wasn't. SGLang
ships a working torch-native attention backend as a real, inheritable base class.
We subclassed it and replaced only the innermost step — the attention computation
itself — with the Spyre kernel. Everything around that step came for free: the KV
gather, writing new keys and values back into the pool, the split between prefill
and decode, and the grouped-query and causal handling. The device-specific
surface ends up being the kernel and the shape bookkeeping it needs, which is
exactly where a hardware backend's code should live.

**The rest of the model.** SGLang routes its layer operations through a
multi-platform dispatch surface, so giving normalisation, activations and rotary
embeddings Spyre implementations is a matter of registering them against the
device. The KV cache pool is device-agnostic in its base class, which makes
putting the cache on the accelerator a contained change rather than a rewrite.

**What we removed.** Some of the enablement was deleting our own code. An early
version carried a hand-written graph runner before we found that SGLang already
handles that case for devices without graph capture — it has an explicit path for
out-of-tree platforms and does the right thing without being asked.

We did need a few small compatibility fixes where core code assumed a CUDA-like
device. They were one-line changes, none of them specific to Spyre, and the
project has since addressed most of them upstream on its own — out-of-tree
support is something SGLang is actively growing.

---

The result is a compact plugin that rides on supported interfaces, which means it
stays current as both SGLang and the Spyre stack keep moving. We are doing this
work in the open, and we would welcome collaboration from anyone working on
inference frameworks, accelerators, or the glue between them.
