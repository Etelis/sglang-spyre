"""SpyreMHATokenToKVPool — KV cache pool for Spyre.

Reuse strategy: SGLang's base ``MHATokenToKVPool`` is already device-agnostic
— ``_create_buffers`` allocates ``k_buffer``/``v_buffer`` on ``self.device``
with no hard-coded ``cuda`` (sglang/srt/mem_cache/memory_pool.py). So the
Spyre pool is a *real subclass* of the base rather than the identity
re-export it used to be.

Phase A (this commit): a proper subclass that behaves exactly like the base
(KV in the standard pool, on whatever ``device`` SGLang passes — CPU today).
This is the seam Phase C overrides: ``_create_buffers`` will place the K/V
buffers on ``torch.device("spyre")`` so decode transfers only the newly
generated token's KV instead of re-shipping the whole history every step.
"""

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


class SpyreMHATokenToKVPool(MHATokenToKVPool):
    """KV pool for Spyre AIU.

    Inherits the device-agnostic buffer allocation, scatter (``set_kv_buffer``)
    and gather (``get_key_buffer``/``get_value_buffer``) from the base. Phase C
    overrides ``_create_buffers`` to put the buffers on the Spyre device.
    """

    # Phase C will add:
    #   def _create_buffers(self):
    #       # allocate k_buffer / v_buffer on torch.device("spyre")
    #       ...
    pass


__all__ = ["SpyreMHATokenToKVPool"]
