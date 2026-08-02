# SGLang Comes to the IBM Spyre AIU

*August 2, 2026*

We are excited to announce that SGLang now runs on the IBM Spyre AIU, IBM's
purpose-built accelerator for enterprise AI inference. Developers building on
Spyre can now serve models through SGLang, and the SGLang community gains support
for another class of hardware.

The integration ships as an out-of-tree platform plugin. SGLang discovers Spyre
automatically at startup and treats the AIU as a device it knows about, with no
fork of SGLang and no vendored copy of the framework. The full serving path works
end to end: SGLang's scheduler drives the accelerator, attention executes on the
AIU, and RadixCache prefix sharing works on Spyre, so requests sharing a system
prompt reuse the cached prefix exactly as they would on any other backend. The
model body and the KV cache can both live on the device, keeping cached prefixes
where the compute is.

Enabling it was largely a matter of filling in the seams SGLang already provides.
Registering the platform is enough for the engine to adopt the device and ask for
the pieces that are hardware-specific — a KV cache pool, a paged allocator, an
attention backend. The attention layer proved the smoothest part: SGLang ships a
working torch-native backend as a real, inheritable base class, so we replaced
only the innermost computation with the Spyre kernel and inherited the KV gather,
the pool write, the prefill and decode split, and the grouped-query and causal
handling. Layer operations register through SGLang's multi-platform dispatch
surface, and the KV pool is device-agnostic in its base class. Some of the work
was deleting our own code once we found core already handled the case — SGLang
has an explicit path for out-of-tree platforms and does the right thing without
being asked.

Work continues. On the roadmap are larger batch sizes and continuous batching,
tensor parallelism, dynamic shapes as the Spyre toolchain evolves, and broader
model coverage — along with upstreaming the small compatibility fixes that help
any out-of-tree backend, not just this one.

> *[Quote from IBM Spyre leadership — placeholder]*

> *[Quote from the SGLang team — placeholder]*

Thanks to everyone across the SGLang community and the IBM Spyre team who helped
get this running. The SGL family always welcomes new contributors — if you work
on inference frameworks, accelerators, or the glue between them, come and build
with us.
