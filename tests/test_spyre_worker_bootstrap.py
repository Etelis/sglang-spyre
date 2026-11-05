"""Bootstrap must stay card-free until each scheduler knows its TP rank."""

from __future__ import annotations

import os
import sys
import signal
from types import ModuleType
from unittest.mock import patch
import unittest

from sglang_spyre_backend import activate
from sglang_spyre_backend.worker import (
    HOST_DEVICE_MODULE, configure_rank_environment, keep_sigterm_handler,
)


class WorkerBootstrapTests(unittest.TestCase):
    def test_scheduler_placement_overrides_inherited_tp1_defaults(self):
        with patch.dict(os.environ, {name: "0" for name in (
            "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"
        )}):
            configure_rank_environment(rank=3, world_size=4, local_rank=3)
            self.assertEqual(
                [os.environ[name] for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")],
                ["3", "4", "3", "4"],
            )

    def test_invalid_placement_fails_before_touching_environment(self):
        with patch.dict(os.environ, {"RANK": "sentinel"}):
            with self.assertRaises(ValueError):
                configure_rank_environment(rank=4, world_size=4, local_rank=0)
            self.assertEqual(os.environ["RANK"], "sentinel")

    def test_discovery_does_not_import_native_runtime(self):
        with patch("importlib.util.find_spec", return_value=object()), patch(
            "sglang_spyre_backend._install_when_imported"
        ), patch.dict(os.environ, {"RANK": "sentinel"}):
            self.assertEqual(activate(), "sglang_spyre_backend.platform.SpyreSRTPlatform")
            self.assertEqual(os.environ["RANK"], "sentinel")

    def test_already_loaded_native_library_cannot_be_reassigned_to_another_rank(self):
        with patch.dict(sys.modules, {"torch_spyre._C": ModuleType("torch_spyre._C")}), patch.dict(
            os.environ, {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1"}
        ):
            with self.assertRaisesRegex(RuntimeError, "loaded before scheduler rank"):
                configure_rank_environment(rank=3, world_size=4, local_rank=3)
            self.assertEqual(os.environ["WORLD_SIZE"], "1")

    def test_host_streams_and_events_work_without_torch_spyre(self):
        with HOST_DEVICE_MODULE.stream(HOST_DEVICE_MODULE.Stream()):
            event = HOST_DEVICE_MODULE.Event()
            event.record()
            event.wait()
            self.assertTrue(event.query())
        with HOST_DEVICE_MODULE.StreamContext(HOST_DEVICE_MODULE.current_stream()):
            event.record(stream=HOST_DEVICE_MODULE.current_stream())
            event.wait(stream=HOST_DEVICE_MODULE.current_stream())

    def test_native_import_signal_changes_are_restored_even_on_failure(self):
        previous = signal.getsignal(signal.SIGTERM)
        with self.assertRaisesRegex(RuntimeError, "native import failed"):
            with keep_sigterm_handler():
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                raise RuntimeError("native import failed")
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)


if __name__ == "__main__":
    unittest.main()
