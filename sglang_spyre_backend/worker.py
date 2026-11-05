"""Ranked Spyre worker initialization and card-free host stream facade."""

from __future__ import annotations

import contextlib
import os
import sys
from types import SimpleNamespace


class _HostStream:
    def __init__(self, *args, **kwargs):
        pass

    def synchronize(self):
        pass

    def wait_stream(self, other):
        pass

    def wait_event(self, event):
        pass

    def record_event(self, event=None):
        event = event or _HostEvent()
        event.record(self)
        return event

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _HostEvent:
    def __init__(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        pass

    def wait(self, *args, **kwargs):
        pass

    def synchronize(self):
        pass

    def query(self):
        return True

    def elapsed_time(self, other):
        return 0.0


HOST_DEVICE_MODULE = SimpleNamespace(
    __name__="cpu",
    Stream=_HostStream,
    Event=_HostEvent,
    StreamContext=lambda stream: contextlib.nullcontext(),
    stream=lambda *args, **kwargs: contextlib.nullcontext(),
    device=lambda *args, **kwargs: contextlib.nullcontext(),
    is_current_stream_capturing=lambda: False,
    current_stream=lambda *args, **kwargs: _HostStream(),
    current_device=lambda: 0,
    set_device=lambda *args, **kwargs: None,
    synchronize=lambda *args, **kwargs: None,
)


@contextlib.contextmanager
def keep_sigterm_handler():
    """senlib must not replace SGLang's scheduler shutdown handler."""
    import signal

    previous = signal.getsignal(signal.SIGTERM)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def configure_rank_environment(*, rank: int, world_size: int, local_rank: int) -> None:
    if not 0 <= rank < world_size or not 0 <= local_rank < world_size:
        raise ValueError("Invalid local Spyre TP rank placement.")
    # SGLang spawns workers itself. Launcher/default values inherited from the
    # Engine parent must never give every rank the same native comms identity.
    placement = (
        ("RANK", rank), ("WORLD_SIZE", world_size),
        ("LOCAL_RANK", local_rank), ("LOCAL_WORLD_SIZE", world_size),
    )
    if "torch_spyre._C" in sys.modules and any(
        os.environ.get(name) != str(value) for name, value in placement
    ):
        raise RuntimeError(
            "Spyre native library loaded before scheduler rank selection; "
            "start a fresh process with TORCH_DEVICE_BACKEND_AUTOLOAD=0."
        )
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    for name, value in placement:
        os.environ[name] = str(value)
