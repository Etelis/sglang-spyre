"""SpyreSRTPlatform — SGLang SRT backend for IBM Spyre AIU."""

from sglang.srt.platforms.interface import SRTPlatform

from sglang_spyre_backend.device import SpyreDeviceMixin


class SpyreSRTPlatform(SRTPlatform, SpyreDeviceMixin):
    """SGLang SRT platform for IBM Spyre AIU accelerator.

    Execution model:
    - No CUDA graph capture/replay
    - FP16 compute, matching the current spyre-inference compatibility contract
    - KV cache lives on Spyre in a dense slot-major layout and uses indirect access
    - Single-node tensor parallelism over native spyreccl collectives
    - Static shapes only (Spyre Inductor cannot handle SymInt)
    """

    # ------------------------------------------------------------------
    # Configuration lifecycle
    # ------------------------------------------------------------------

    def apply_server_args_defaults(self, server_args) -> None:
        """Apply Spyre-specific defaults after ServerArgs parsing."""
        # Disable CUDA graph — Spyre uses eager execution
        if not server_args.disable_cuda_graph:
            server_args.disable_cuda_graph = True

        # Match spyre-inference: float16 is the supported model/cache dtype.
        if getattr(server_args, "dtype", "auto") == "auto":
            server_args.dtype = "float16"

        # The synchronized paged backend is the production-comparable path.
        if getattr(server_args, "attention_backend", None) is None:
            server_args.attention_backend = "spyre_paged"

        # spyre-inference's default KV block size (platform._DEFAULT_BLOCK_SIZE, #746):
        # a power of two and a multiple of the 64-element stick.
        if getattr(server_args, "page_size", None) is None:
            server_args.page_size = 128

        # spyre-inference keeps every layer's KV in one paged cache and applies a
        # layer's sliding window in its attention mask (#1317, #752); this backend
        # owns the pages the same way, so SGLang's separate SWA pool is not used.
        if hasattr(server_args, "disable_hybrid_swa_memory"):
            server_args.disable_hybrid_swa_memory = True

        # Spyre runs each step synchronously, so SGLang's overlap scheduler has no
        # device time to hide host work behind; it only launches one speculative
        # decode step past a request's last token. vLLM's spyre-inference engine
        # steps synchronously too.
        if hasattr(server_args, "disable_overlap_schedule"):
            server_args.disable_overlap_schedule = True

        # The first request traces and compiles every decoder block; raise the
        # watchdog so that warmup does not count as a hung forward.
        if getattr(server_args, "watchdog_timeout", 300) < 3600:
            server_args.watchdog_timeout = 3600

        tp_size = getattr(server_args, "tp_size", 1)
        if tp_size < 1:
            raise ValueError("Spyre tensor parallel size must be positive.")
        for name in ("dp_size", "pp_size", "nnodes", "ep_size", "moe_dp_size", "attn_cp_size", "dcp_size"):
            if getattr(server_args, name, 1) not in (None, 1):
                raise ValueError(f"Spyre supports one local TP world; {name} must be 1.")
        if getattr(server_args, "enable_dp_attention", False):
            raise ValueError("Spyre does not support DP attention.")
        if tp_size > 1 and server_args.dtype not in ("float16", "half"):
            raise ValueError("Spyre tensor parallelism requires float16.")
        if tp_size > 1 and getattr(server_args, "device", "cpu") != "cpu":
            raise ValueError("Spyre TP uses device=cpu with its model body on Spyre.")
        for name, value in (
            ("disable_custom_all_reduce", True), ("enable_mscclpp", False),
            ("enable_torch_symm_mem", False), ("enable_symm_mem", False),
            ("pre_warm_nccl", False),
        ):
            if hasattr(server_args, name):
                setattr(server_args, name, value)
        from sglang_spyre_backend.threading_config import configure_threading

        configure_threading(worker_count=tp_size)

    # ------------------------------------------------------------------
    # Subsystem factory methods
    # ------------------------------------------------------------------

    def get_default_attention_backend(self) -> str:
        return "spyre_paged"

    def get_graph_runner_cls(self) -> type:
        # Spyre runs eager: support_cuda_graph() returns False, so SGLang core
        # sets model_runner.graph_runner = None itself (model_runner.py, the
        # is_out_of_tree() branch) and never calls this. We therefore need no
        # bespoke graph runner — the previous SpyreEagerGraphRunner just
        # re-implemented the eager forward path the core already runs.
        raise NotImplementedError(
            "SpyreSRTPlatform runs eager (support_cuda_graph() is False); "
            "core uses graph_runner=None and never requests a graph runner class."
        )

    def get_mha_kv_pool_cls(self) -> type:
        from sglang_spyre_backend.memory_pool import SpyreMHATokenToKVPool

        return SpyreMHATokenToKVPool

    def get_mla_kv_pool_cls(self) -> type:
        # MLA (DeepSeek) not yet supported on Spyre; fall back to MHA
        from sglang_spyre_backend.memory_pool import SpyreMHATokenToKVPool

        return SpyreMHATokenToKVPool

    def get_dsa_kv_pool_cls(self) -> type:
        # DSA (DeepSeek V3.2) not yet supported
        from sglang_spyre_backend.memory_pool import SpyreMHATokenToKVPool

        return SpyreMHATokenToKVPool

    def get_paged_allocator_cls(self) -> type:
        # SGLang's PagedTokenToKVPoolAllocator.alloc_extend/alloc_decode dispatch
        # Triton kernels, which crash on Spyre ("0 active drivers" — no Triton
        # backend). SpyrePagedAllocator overrides just those two with torch-native
        # equivalents that keep the exact page-aligned slot layout. Only used at
        # page_size>1 (the "spyre_paged" path); page_size=1 uses the non-paged
        # torch allocator, so the dense "spyre" backend is unaffected.
        from sglang_spyre_backend.paged_allocator import SpyrePagedAllocator

        return SpyrePagedAllocator

    def get_piecewise_backend_cls(self) -> type:
        # Piecewise torch.compile not used on Spyre
        raise NotImplementedError("SpyreSRTPlatform does not support piecewise backend")

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    def support_cuda_graph(self) -> bool:
        return False

    def support_piecewise_cuda_graph(self) -> bool:
        return False

    def supports_fp8(self) -> bool:
        # Spyre 1.5 supports FP8 at the hardware level, but the current
        # spyre-inference stack does not yet expose FP8 quantization.
        return False

    def is_pin_memory_available(self, device=None) -> bool:
        # Spyre uses its own allocator; pin_memory is a no-op.
        return False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_backend(self) -> None:
        """Load native libraries only after the scheduler publishes placement.

        SGLang also calls this at model_runner import. That call is deliberately
        card-free; the OOT bootstrap hook calls again immediately before WORLD
        initialization, when every worker has its real rank and local card.
        """
        from sglang.srt.runtime_context import publish_role

        if publish_role() != "scheduler":
            return
        if getattr(self, "_spyre_backend_inited", False) or getattr(
            self, "_spyre_backend_initializing", False
        ):
            return
        # Installing the post-load hook imports model_runner, whose module body
        # calls init_backend again. Native setup must run once per worker.
        self._spyre_backend_initializing = True
        try:
            self._init_worker_backend()
            self._spyre_backend_inited = True
        finally:
            self._spyre_backend_initializing = False

    def _init_worker_backend(self) -> None:
        from sglang.srt.runtime_context import get_device, get_parallel

        parallel = get_parallel()
        if parallel.dp_size != 1 or parallel.pp_size != 1:
            raise ValueError("Spyre native collectives require a single local TP world.")
        from sglang_spyre_backend.worker import configure_rank_environment, keep_sigterm_handler

        configure_rank_environment(
            rank=parallel.tp_rank,
            world_size=parallel.tp_size,
            local_rank=get_device().gpu_id,
        )
        import logging
        import os

        import torch

        local_rank = get_device().gpu_id
        devices = os.environ.get("SPYRE_DEVICES", "").split(",")
        address = (
            os.environ.get(f"AIU_WORLD_RANK_{devices[local_rank].strip()}", "unknown")
            if local_rank < len(devices)
            else "unknown"
        )
        logging.getLogger(__name__).info(
            "Spyre worker rank=%d/%d local_rank=%d pcie_address=%s",
            parallel.tp_rank, parallel.tp_size, local_rank, address,
        )
        with keep_sigterm_handler():
            import torch_spyre

            torch_spyre._autoload()
            torch.spyre.set_device(get_device().gpu_id)
            from sglang_spyre_backend import _install_device_module_stubs
            from sglang_spyre_backend.distributed import install_distributed_hooks

            _install_device_module_stubs(torch.spyre)
            install_distributed_hooks()
            # Materialize on the selected card before constructing spyreccl groups.
            # Do not suppress an acquisition failure: continuing would strand peers.
            torch.empty(64, device="spyre", dtype=torch.float16)
        import sglang_spyre_backend.register  # noqa: F401
        from sglang_spyre_backend.custom_ops import install

        install()
        # Retain the pinned stack's addmm decomposition repair (torch-spyre #1420).
        import torch._inductor.decomposition
        from torch_spyre._inductor.decompositions import spyre_decompositions

        for op, implementation in spyre_decompositions.items():
            if "addm" in op.name():
                torch._inductor.decomposition.decompositions[op] = implementation

    # ------------------------------------------------------------------
    # MultiPlatformOp dispatch
    # ------------------------------------------------------------------

    def get_dispatch_key_name(self) -> str:
        return "spyre"
