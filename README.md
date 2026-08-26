# SGLang Spyre backend

An experimental [SGLang](https://github.com/sgl-project/sglang) backend for
[IBM Spyre Accelerator](https://www.ibm.com/products/spyre-accelerator), built
on [torch-spyre](https://github.com/torch-spyre/torch-spyre). The device kernel
path is synchronized from
[spyre-inference](https://github.com/torch-spyre/spyre-inference), while SGLang
continues to own scheduling and its RadixAttention cache semantics.

> [!IMPORTANT]
> This repository is a working prototype. It is not yet a production-supported
> backend, but its dependency set and benchmark baseline are intentionally
> reproducible.

## Compatibility baseline

The exact machine-readable baseline is in [`compatibility.toml`](compatibility.toml).

| Component | Pin |
|---|---|
| spyre-inference source | `6e5ff996178255ff63acec480cf8fb43ce1e43de` |
| torch-spyre | `a31289852145a59099edccc3e506cf5336e8e2e0` |
| PyTorch | `2.13.0` |
| Transformers | `5.14.1` |
| SGLang | `0.5.18` (`71de97b264b04dcd514cf904003028aefe9775c8`) |
| Spyre system RPMs | [`spyre-rpms.lock`](spyre-rpms.lock) |

These pins match the current `spyre-inference` dependency set. In particular,
the old `torch~=2.11.0` constraint has been removed; it cannot resolve against
the current torch-spyre stack.

## What is synchronized

- Dense slot-major KV pages on Spyre.
- Compiled `index_copy_` KV writes.
- Indirect `index_select` page reads.
- Page-at-a-time online softmax, avoiding a dense KV gather.
- Decode canonical-layout copy, per-step mask transfer, and in-kernel output
  stores from the latest spyre-inference attention path.
- Float16 runtime defaults, 64-token pages, transposed linear weights,
  device-resident RoPE, and the `gelu_new` correctness workaround.

The imported kernel is kept framework-neutral in
`sglang_spyre_backend/spyre_attention_kernel.py`; `attention.py` is the thin
SGLang metadata adapter. This keeps the port reviewable against the pinned
spyre-inference source instead of maintaining an independently rewritten
kernel.

## Current scope

- Decoder-only text generation.
- One active sequence and tensor parallel size 1.
- FP16 model and KV-cache tensors.
- SGLang RadixCache with a 64-token page size.
- No CUDA graphs; torch.compile is used for Spyre attention kernels.

## Installation

Use a Spyre runtime image containing the RPM versions in `spyre-rpms.lock`.
The normal development flow uses `uv` so the torch-spyre source revision and
its build-time Torch version remain locked:

```bash
git clone https://github.com/torch-spyre/sglang-spyre.git
cd sglang-spyre
uv sync --frozen
source .venv/bin/activate
python -m sglang_oot_patches.apply
```

`torch-spyre` must be built on a supported Spyre host with the IBM runtime and
compiler available. Set the rank variables before importing torch-spyre:

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
```

## Run

```python
import sglang as sgl

engine = sgl.Engine(
    model_path="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
    device="cpu",
    attention_backend="spyre_paged",
    dtype="float16",
    page_size=64,
    max_running_requests=1,
    disable_cuda_graph=True,
)

result = engine.generate(
    "What are IBM's main businesses?",
    sampling_params={"max_new_tokens": 32, "temperature": 0.0},
)
print(result["text"])
engine.shutdown()
```

Set `SGLANG_FORCE_DEVICE_SPYRE=1` to place supported model-body operations on
Spyre while SGLang keeps host-side scheduling tensors on CPU.

## Verification and benchmark

CPU correctness test:

```bash
pytest -q tests/test_spyre_attention_kernel.py
```

On a Spyre pod with both checkouts installed, the A/B test compares the exact
ported kernel with the pinned spyre-inference implementation using identical
inputs and excludes compilation warm-up:

```bash
python tests/spyre_kernel_ab.py
```

The existing end-to-end smoke and RadixCache examples remain under `tests/`.
See [`BENCHMARK.md`](BENCHMARK.md) for the latest measured parity result and
the machine-readable benchmark record.

## License

Licensed under the [Apache License 2.0](LICENSE). Code ported from
spyre-inference retains its upstream copyright and license notice.
