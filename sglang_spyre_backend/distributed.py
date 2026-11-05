"""SGLang TP collectives over Spyre, with Gloo for the host boundary.

The device paths follow spyre-inference's SpyreCommunicator at 0ccdbf6.
Spyre's lowered collectives span the native comms world, so only a single
local TP world is supported. Singleton bookkeeping groups stay Gloo-only.
"""

from __future__ import annotations

from functools import wraps
import inspect
import os

import torch
import torch.distributed as dist


def _is_spyre_platform() -> bool:
    from sglang.srt.platforms import current_platform

    return current_platform.device_type == "spyre"


def spyre_all_reduce(group, input_: torch.Tensor) -> torch.Tensor:
    """Out-of-place reduction that remains in the compiled decoder graph."""
    if group.world_size == 1:
        return input_
    if input_.dtype != torch.float16:
        raise ValueError("Spyre TP all-reduce requires float16 tensors.")
    shape = input_.shape
    # The eager reduction's scheduler cannot split many rank-2 shapes. A flat
    # view is supported by both eager spyreccl and the compiled lowering.
    value = input_.reshape(-1) if input_.dim() > 1 else input_
    value = torch.ops._c10d_functional.all_reduce(value, "sum", group._spyre_group_name)
    value = torch.ops._c10d_functional.wait_tensor(value)
    return value.reshape(shape)


def spyre_all_gather(group, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Gather rank shards without unaligned Spyre transfers or host logits."""
    if group.world_size == 1:
        return input_
    if input_.dtype != torch.float16:
        raise ValueError("Spyre TP all-gather requires float16 tensors.")
    if not -input_.dim() <= dim < input_.dim():
        raise ValueError(f"Invalid gather dim {dim} for shape {input_.shape}.")
    dim %= input_.dim()
    if torch.compiler.is_compiling():
        if input_.numel() % 64:
            raise ValueError("Compiled Spyre all-gather requires 64-aligned shard numel.")
        shape = input_.shape
        output = torch.ops._c10d_functional.all_gather_into_tensor(
            input_, group.world_size, group._spyre_group_name
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        output = output.reshape((group.world_size, *shape)).movedim(0, dim)
        return output.reshape((*shape[:dim], shape[dim] * group.world_size, *shape[dim + 1 :]))

    width = input_.shape[dim]
    padding = (-width) % 64
    if padding:
        # Padding and slicing at a partial stick must happen on the host.
        from sglang_spyre_backend.custom_ops.utils import convert

        pads = [0, 0] * (input_.dim() - dim - 1) + [0, padding]
        value = convert(
            torch.nn.functional.pad(convert(input_, device="cpu"), pads),
            device=input_.device,
        )
    else:
        value = input_.contiguous()
    outputs = [torch.empty_like(value) for _ in range(group.world_size)]
    dist.all_gather(outputs, value, group=group.device_group)
    if padding:
        outputs = [convert(value, device="cpu").narrow(dim, 0, width) for value in outputs]
        return convert(torch.cat(outputs, dim=dim), device=input_.device)
    return torch.cat(outputs, dim=dim)


def _cpu_all_gather(group, input_, dim=-1, output_tensor_list=None):
    if output_tensor_list is not None:
        dist.all_gather(output_tensor_list, input_.contiguous(), group=group.cpu_group)
        return None
    outputs = [torch.empty_like(input_) for _ in range(group.world_size)]
    dist.all_gather(outputs, input_.contiguous(), group=group.cpu_group)
    return torch.cat(outputs, dim=dim)


def _install_live_group_getters(communication_op) -> None:
    """Keep deprecated-getter warning introspection out of model graphs.

    SGLang 0.5.21's warning wrapper calls sys._getframe even for its exempt
    internal communication_op callers. The underlying getter still resolves
    get_parallel() on every call, preserving scope overrides and group rebuilds.
    """
    for name in (
        "get_tp_group", "get_attn_tp_group", "get_attn_cp_group",
        "get_moe_tp_group", "get_moe_ep_group",
    ):
        getter = getattr(communication_op, name, None)
        undecorated = getattr(getter, "__wrapped__", None)
        if undecorated is not None:
            setattr(communication_op, name, undecorated)


def install_distributed_hooks() -> None:
    """Install only after worker rank selection and torch-spyre registration."""
    from sglang.srt.distributed import bootstrap, communication_op, parallel_state

    _install_live_group_getters(communication_op)
    cls = parallel_state.GroupCoordinator
    if getattr(cls, "_spyre_collectives_installed", False):
        return
    original_init = cls.__init__
    signature = inspect.signature(original_init)

    @wraps(original_init)
    def init(self, *args, **kwargs):
        if not _is_spyre_platform():
            return original_init(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        ranks = bound.arguments["group_ranks"]
        world = int(os.environ["WORLD_SIZE"])
        if all(len(group) == 1 for group in ranks):
            backend = "gloo"
        elif any(group != list(range(world)) for group in ranks):
            raise ValueError("Spyre TP groups must span the whole native comms world.")
        else:
            backend = "cpu:gloo,spyre:spyreccl"
        bound.arguments["torch_distributed_backend"] = backend
        for flag in (
            "use_pynccl", "use_mscclpp", "use_custom_allreduce",
            "use_torch_symm_mem_all_reduce", "use_hpu_communicator",
            "use_xpu_communicator", "use_npu_communicator",
        ):
            bound.arguments[flag] = False
        original_init(*bound.args, **bound.kwargs)
        self._spyre_group = True
        if self.world_size > 1:
            from torch.distributed._functional_collectives import _resolve_group_name

            self._spyre_group_name = _resolve_group_name(self.device_group)

    original_reduce = cls.all_reduce

    @wraps(original_reduce)
    def all_reduce(self, input_):
        if not getattr(self, "_spyre_group", False):
            return original_reduce(self, input_)
        if self.world_size == 1:
            return input_
        if input_.device.type == "spyre":
            return spyre_all_reduce(self, input_)
        dist.all_reduce(input_, group=self.cpu_group)
        return input_

    original_gather = cls.all_gather

    @wraps(original_gather)
    def all_gather(self, input_, dim=-1, output_tensor_list=None):
        if not getattr(self, "_spyre_group", False):
            return original_gather(self, input_, dim, output_tensor_list)
        if self.world_size == 1:
            if output_tensor_list is not None:
                output_tensor_list[0].copy_(input_)
                return None
            return input_
        if input_.device.type == "spyre":
            if output_tensor_list is not None:
                raise NotImplementedError("Spyre TP does not support list-output device gathers.")
            return spyre_all_gather(self, input_, dim)
        return _cpu_all_gather(self, input_, dim, output_tensor_list)

    original_gather_into = cls.all_gather_into_tensor

    @wraps(original_gather_into)
    def all_gather_into_tensor(self, output, input_):
        if not getattr(self, "_spyre_group", False):
            return original_gather_into(self, output, input_)
        if self.world_size == 1:
            output.copy_(input_.reshape(output.shape))
        elif input_.device.type == "spyre":
            output.copy_(spyre_all_gather(self, input_, dim=0).reshape(output.shape))
        else:
            # LogitsProcessor allocates [world, rows, shard_vocab], while
            # all_gather_into_tensor's low-level API also accepts [world*rows, vocab].
            outputs = list(output.reshape(self.world_size, *input_.shape).unbind(0))
            dist.all_gather(outputs, input_.contiguous(), group=self.cpu_group)

    cls.__init__ = init
    cls.all_reduce = all_reduce
    cls.all_gather = all_gather
    cls.all_gather_into_tensor = all_gather_into_tensor
    cls._spyre_collectives_installed = True

    original_backend = bootstrap._resolve_backend

    @wraps(original_backend)
    def resolve_backend(*, device):
        if _is_spyre_platform():
            return "cpu:gloo,spyre:spyreccl"
        return original_backend(device=device)

    # device="cpu" describes SGLang's scheduler, not an AMX model. Avoid
    # initializing the C++ CPU SHM communicator and its collective fast paths.
    original_bind = bootstrap._bind_threads_if_cpu
    original_threads = bootstrap._init_cpu_threads_env

    @wraps(original_bind)
    def bind_threads(*, device):
        return None if _is_spyre_platform() else original_bind(device=device)

    @wraps(original_threads)
    def init_threads(**kwargs):
        if not _is_spyre_platform():
            return original_threads(**kwargs)

    bootstrap._resolve_backend = resolve_backend
    bootstrap._bind_threads_if_cpu = bind_threads
    bootstrap._init_cpu_threads_env = init_threads
