"""Card-free collective routing and world-topology safety checks."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from datetime import timedelta
import sys

import pytest
import torch

from sglang_spyre_backend import distributed as spyre_dist


@pytest.fixture
def patched_groups(monkeypatch):
    class Group:
        def __init__(
            self, group_ranks, local_rank, torch_distributed_backend,
            use_pynccl=False, use_mscclpp=False, use_custom_allreduce=False,
            use_torch_symm_mem_all_reduce=False, use_hpu_communicator=False,
            use_xpu_communicator=False, use_npu_communicator=False,
        ):
            self.backend = torch_distributed_backend
            self.world_size = len(group_ranks[0])
            self.device_group = "device-group"
            self.cpu_group = "host-gloo-group"
            self.custom_allreduce = use_custom_allreduce

        def all_reduce(self, tensor):
            raise AssertionError("Spyre routing unexpectedly entered the CUDA/SHM reducer")

        def all_gather(self, tensor, dim=-1, output_tensor_list=None):
            raise AssertionError("Spyre routing unexpectedly entered the CUDA/SHM gather")

        def all_gather_into_tensor(self, output, tensor):
            raise AssertionError("Spyre routing unexpectedly entered the CUDA gather op")

    module = ModuleType("sglang.srt.distributed")
    module.parallel_state = SimpleNamespace(GroupCoordinator=Group)
    module.communication_op = SimpleNamespace()
    module.bootstrap = SimpleNamespace(
        _resolve_backend=lambda **kwargs: "original",
        _bind_threads_if_cpu=lambda **kwargs: "original",
        _init_cpu_threads_env=lambda **kwargs: "original",
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", module)
    monkeypatch.setattr(spyre_dist, "_is_spyre_platform", lambda: True)
    monkeypatch.setenv("WORLD_SIZE", "4")
    import torch.distributed._functional_collectives as functional

    monkeypatch.setattr(functional, "_resolve_group_name", lambda group: "functional-group")
    spyre_dist.install_distributed_hooks()
    return Group, module.bootstrap


def test_singleton_groups_never_construct_spyre_backend(patched_groups):
    Group, _ = patched_groups
    group = Group([[0], [1], [2], [3]], 2, "cpu:gloo,spyre:spyreccl")
    assert group.backend == "gloo"


def test_full_world_group_uses_mixed_backend_and_no_cuda_collective(patched_groups):
    Group, _ = patched_groups
    group = Group([[0, 1, 2, 3]], 2, "gloo", use_custom_allreduce=True)
    assert group.backend == "cpu:gloo,spyre:spyreccl"
    assert not group.custom_allreduce
    assert group._spyre_group_name == "functional-group"


def test_native_collectives_reject_nontrivial_subgroups(patched_groups):
    Group, _ = patched_groups
    with pytest.raises(ValueError, match="whole native comms world"):
        Group([[0, 1], [2, 3]], 0, "gloo")


def test_host_embedding_reduce_and_logits_gather_use_gloo(patched_groups, monkeypatch):
    Group, _ = patched_groups
    group = Group([[0, 1, 2, 3]], 0, "gloo")
    used = []

    def reduce(tensor, *, group):
        used.append(group)
        tensor.mul_(4)

    def gather(outputs, tensor, *, group):
        used.append(group)
        for rank, output in enumerate(outputs):
            output.copy_(tensor + rank)

    monkeypatch.setattr(spyre_dist.dist, "all_reduce", reduce)
    monkeypatch.setattr(spyre_dist.dist, "all_gather", gather)
    value = torch.tensor([[1.0, 2.0]])
    assert group.all_reduce(value) is value
    torch.testing.assert_close(value, torch.tensor([[4.0, 8.0]]))
    got = group.all_gather(torch.tensor([[10.0, 20.0]]), dim=-1)
    torch.testing.assert_close(got, torch.tensor([[10., 20., 11., 21., 12., 22., 13., 23.]]))
    into = torch.empty(4, 1, 2)
    group.all_gather_into_tensor(into, torch.tensor([[10.0, 20.0]]))
    torch.testing.assert_close(into[:, 0], torch.tensor([[10., 20.], [11., 21.], [12., 22.], [13., 23.]]))
    assert used == ["host-gloo-group"] * 3


def test_spyre_reduction_is_out_of_place_and_restores_shape(monkeypatch):
    group = SimpleNamespace(world_size=4, _spyre_group_name="test")
    calls = []

    def reduce(tensor, reduction, group_name):
        calls.append((tensor.shape, reduction, group_name))
        return tensor * 4

    monkeypatch.setattr(torch.ops._c10d_functional, "all_reduce", reduce)
    monkeypatch.setattr(torch.ops._c10d_functional, "wait_tensor", lambda value: value)
    value = torch.arange(8, dtype=torch.float16).reshape(2, 4)
    original = value.clone()
    got = spyre_dist.spyre_all_reduce(group, value)
    torch.testing.assert_close(value, original)
    torch.testing.assert_close(got, original * 4)
    assert calls == [(torch.Size([8]), "sum", "test")]


def test_compiled_gather_rejects_unsafe_partial_stick(monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    group = SimpleNamespace(world_size=4, _spyre_group_name="test")
    with pytest.raises(ValueError, match="64-aligned"):
        spyre_dist.spyre_all_gather(group, torch.zeros(1, 65, dtype=torch.float16))


def _gloo_compiled_worker(rank, rendezvous):
    """Exercise the actual functional ops, using Gloo rather than a Spyre card."""
    import torch.distributed as dist
    from torch.distributed._functional_collectives import _resolve_group_name
    from sglang.srt.distributed import communication_op
    from sglang.srt.runtime_context import get_parallel

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        group = SimpleNamespace(
            world_size=2, _spyre_group_name=_resolve_group_name(dist.group.WORLD)
        )
        value = torch.full((2, 4), rank + 1, dtype=torch.float16)
        reduce = torch.compile(
            lambda x: spyre_dist.spyre_all_reduce(group, x), backend="eager", fullgraph=True
        )
        torch.testing.assert_close(reduce(value), torch.full_like(value, 3))
        torch.testing.assert_close(value, torch.full_like(value, rank + 1))

        class LiveGroup:
            def __init__(self, world_size):
                self.world_size = world_size
                self._spyre_group_name = group._spyre_group_name

            def all_reduce(self, tensor):
                return spyre_dist.spyre_all_reduce(self, tensor)

        spyre_dist._install_live_group_getters(communication_op)
        live_group = LiveGroup(2)
        singleton = LiveGroup(1)
        helper = torch.compile(
            communication_op.tensor_model_parallel_all_reduce,
            backend="eager", fullgraph=True,
        )
        # Exercise the actual SGLang wrapper and live runtime context, then
        # redirect its group as draft/context scopes and engine rebuilds do.
        with get_parallel().override(tp_group=live_group):
            assert communication_op.get_tp_group() is live_group
            torch.testing.assert_close(helper(value), torch.full_like(value, 3))
            with get_parallel().override(tp_group=singleton):
                assert communication_op.get_tp_group() is singleton
                torch.testing.assert_close(helper(value), value)
            assert communication_op.get_tp_group() is live_group
            torch.testing.assert_close(helper(value), torch.full_like(value, 3))

        value = torch.arange(128, dtype=torch.float16).reshape(2, 64) + rank * 1000
        gather = torch.compile(
            lambda x: spyre_dist.spyre_all_gather(group, x, dim=1),
            backend="eager", fullgraph=True,
        )
        base = torch.arange(128, dtype=torch.float16).reshape(2, 64)
        torch.testing.assert_close(gather(value), torch.cat([base, base + 1000], dim=1))
    finally:
        dist.destroy_process_group()


def test_functional_collectives_trace_and_execute_across_two_cpu_ranks(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    torch.multiprocessing.spawn(
        _gloo_compiled_worker, args=(str(tmp_path / "gloo-rendezvous"),), nprocs=2, join=True
    )
