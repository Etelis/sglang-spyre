"""SGLang backend plugin for IBM Spyre AIU accelerator."""

import contextlib
import os

# Native comms reads placement when its library loads. Only the scheduler
# bootstrap has the actual TP rank, so platform discovery must stay card-free.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def _install_device_module_stubs(spyre_mod) -> None:
    """Give torch.spyre the CUDA-style Stream/Event/stream API SGLang expects.

    CRITICAL: torch_spyre's module ``__getattr__`` lazily opens the (single-tenant)
    Spyre VFIO device on access to *any unknown* attribute. So we must NOT use
    ``getattr``/``hasattr`` here — that would open the card inside activate(), in
    every process (parent Engine + scheduler subprocess), and the second one to
    open it dies with "Device or resource busy". We touch only ``vars(spyre_mod)``
    (the module __dict__), which never triggers __getattr__, so this stays
    completely card-free.

    Self-contained (no sglang import); safe from activate() mid-import. Spyre is
    eager/synchronous, so a stream is a no-op context and an event is always
    'done'. Idempotent.
    """
    d = vars(spyre_mod)
    if d.get("_sglang_stubs_installed"):
        return

    class _NoOpStream:
        def __init__(self, *a, **k): pass
        def synchronize(self): pass
        def wait_stream(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _NoOpEvent:
        def __init__(self, *a, **k): pass
        def record(self, *a, **k): pass
        def wait(self, *a, **k): pass
        def synchronize(self): pass
        def query(self): return True
        def elapsed_time(self, other): return 0.0

    # Direct __dict__ writes only (no getattr/hasattr -> no card open).
    d.setdefault("Stream", _NoOpStream)
    d.setdefault("Event", _NoOpEvent)
    d.setdefault("stream", lambda *a, **k: contextlib.nullcontext())
    d.setdefault("current_stream", lambda *a, **k: _NoOpStream())
    d.setdefault("synchronize", lambda *a, **k: None)
    d["_sglang_stubs_installed"] = True


def _install_when_imported(target: str, module: str, func: str) -> None:
    """Call ``module.func()`` once ``target`` has finished importing.

    activate() runs while ``sglang.srt.*`` is mid-import, so it must not import
    SGLang itself; a meta-path finder wraps the target's loader instead.
    """
    import importlib
    import importlib.abc
    import sys

    def run():
        getattr(importlib.import_module(module), func)()

    if target in sys.modules and getattr(sys.modules[target], "__spec__", None) is not None:
        mod = sys.modules[target]
        if getattr(mod, "_spyre_hooked", False):
            return
        if hasattr(mod, "HfModelConfigParser"):
            mod._spyre_hooked = True
            run()
            return

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target_=None):
            if fullname != target:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return spec
            original = spec.loader.exec_module

            def exec_module(m):
                original(m)
                m._spyre_hooked = True
                run()

            spec.loader.exec_module = exec_module
            return spec

    import importlib.util  # noqa: F401

    sys.meta_path.insert(0, _Finder())


def activate():
    """Discover Spyre without loading the native runtime in the Engine parent."""
    import importlib.util

    if importlib.util.find_spec("torch_spyre") is None:
        return None
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    # Model-config normalization runs in the parent; native device and comms
    # registration wait until the scheduler has published its rank placement.
    _install_when_imported(
        "sglang.srt.utils.hf_transformers.config",
        "sglang_spyre_backend.models",
        "install_config_hooks",
    )
    return "sglang_spyre_backend.platform.SpyreSRTPlatform"
