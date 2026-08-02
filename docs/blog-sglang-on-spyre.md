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
no vendored copy of the framework.

From there, the full serving path works end to end:

- **SGLang's scheduler drives the accelerator.** Requests arrive through the
  standard engine, are scheduled by SGLang, and dispatch real work to Spyre.
- **Attention runs on the AIU.** Every attention layer executes on the device,
  using the same kernels and shape conventions the Spyre stack already uses.
- **RadixCache prefix sharing works on Spyre.** Requests that share a system
  prompt reuse the cached prefix, exactly as they would on any other backend.
  The scheduler resolves the cache hit before the forward begins, so from the
  device's point of view a shared prefix is simply work that never arrives.
- **The model body can live on the device.** Normalisation, activations, rotary
  embeddings and the linear layers all have Spyre implementations, with the
  residual stream staying resident across the forward pass.
- **The KV cache can live on the device too**, keeping cached prefixes where the
  compute is.

## Why it went smoothly

The part worth dwelling on is how naturally this came together — and that is a
credit to SGLang's design rather than to us.

SGLang treats a new backend as a first-class idea. Bringing up a device means
registering a platform and filling in well-defined seams, not carving open the
framework's internals. The attention layer is a real base class with working
defaults, so a new backend swaps in its own kernel and inherits the rest of the
machinery. The KV cache pool is device-agnostic by design, so putting the cache
on the accelerator is a small, contained change. There is no worker class to
reimplement and no scheduling logic to duplicate.

Best of all, out-of-tree platforms are a concept SGLang already understands.
Core code asks what kind of platform it is talking to and adapts, and that
support has visibly continued to improve while we were building on it. Several
of the small rough edges we met early on have since been smoothed by the project
itself.

The result is a compact component that lives alongside SGLang and rides on
supported interfaces — which means it stays current as both SGLang and the Spyre
stack keep moving.

## Part of a wider shift

Spyre is not the only accelerator walking through this door. Shortly before we
wrote this, RadixArk and Google
[announced](https://www.lmsys.org/blog/2026-07-30-sglang-google-tpu)
`SGL-torchtpu`, bringing SGLang to Google's TPUs.

That is a good thing to see. Independent teams, working on very different
silicon, reaching for the same framework says something encouraging about where
open inference infrastructure is heading: toward frameworks designed so new
hardware can join through clean, supported interfaces, and toward a serving
ecosystem that is not tied to a single kind of device.

It also means the small improvements each of us contributes upstream compound.
Every backend that follows has an easier path than the one before it.

## What's next

This is a step, not a finish line. The Spyre stack will keep growing, and having
SGLang in place makes that easier — a real serving framework to bring new
capabilities up against, and a clean set of seams to extend as the hardware and
its toolchain evolve.

We are doing this work in the open, and we would welcome collaboration. If you
work on inference frameworks, accelerators, or the glue between them, come and
build with us — the stack is growing, and there is plenty of room.
