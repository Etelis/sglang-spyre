"""Structural tests for the reuse-refactor (no Spyre card required).

These assert the Phase-A reuse decisions hold:
  - SpyreAttnBackend is a *subclass* of SGLang's TorchNativeAttnBackend
    (we inherit the gather/save/forward plumbing, override only the SDPA seam).
  - SpyreMHATokenToKVPool is a real *subclass* of MHATokenToKVPool, not an
    identity re-export.
  - The platform no longer ships a bespoke eager graph runner (core runs eager
    with graph_runner=None when support_cuda_graph() is False), and there is no
    graph_runner module left to import.

Requires SGLang importable (torch + sglang). Skips cleanly if not present, so
it is safe to collect anywhere; run it on the dev pod for real coverage.
"""

import importlib
import importlib.util

import pytest

sglang_available = importlib.util.find_spec("sglang") is not None
pytestmark = pytest.mark.skipif(
    not sglang_available, reason="sglang not importable in this environment"
)


def test_attention_subclasses_torch_native():
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

    from sglang_spyre_backend.attention_native import SpyreAttnBackend

    assert issubclass(SpyreAttnBackend, TorchNativeAttnBackend), (
        "SpyreAttnBackend must reuse TorchNativeAttnBackend's gather/save/"
        "forward plumbing and override only the SDPA seam."
    )
    # The Spyre-specific override must exist...
    assert "_run_sdpa_forward_extend" in SpyreAttnBackend.__dict__
    assert "_run_sdpa_forward_decode" in SpyreAttnBackend.__dict__
    # ...and the inherited entry points must NOT be overridden (proves reuse).
    assert "forward_extend" not in SpyreAttnBackend.__dict__
    assert "forward_decode" not in SpyreAttnBackend.__dict__


def test_kv_pool_is_real_subclass():
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    from sglang_spyre_backend.memory_pool import SpyreMHATokenToKVPool

    assert issubclass(SpyreMHATokenToKVPool, MHATokenToKVPool)
    # Must be a distinct subclass, not the identity re-export it used to be.
    assert SpyreMHATokenToKVPool is not MHATokenToKVPool


def test_no_bespoke_graph_runner():
    # The eager graph runner module was deleted; nothing should import it.
    assert importlib.util.find_spec("sglang_spyre_backend.graph_runner") is None

    from sglang_spyre_backend.platform import SpyreSRTPlatform

    plat = SpyreSRTPlatform.__new__(SpyreSRTPlatform)
    assert plat.support_cuda_graph() is False
    # get_graph_runner_cls is dead code under eager execution and must refuse.
    with pytest.raises(NotImplementedError):
        plat.get_graph_runner_cls()


def test_attention_registry_points_at_subclass():
    # Importing register wires "spyre" → SpyreAttnBackend factory.
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS

    import sglang_spyre_backend.register  # noqa: F401  (registers as side effect)

    assert "spyre" in ATTENTION_BACKENDS
