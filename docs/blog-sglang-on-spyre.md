# SGLang Comes to the IBM Spyre AIU

IBM & the SGLang community

August 2, 2026

SGLang now runs on the IBM Spyre AIU, giving developers another hardware choice
for serving models with the same API and the same features they already use.

Spyre is IBM's accelerator for enterprise inference, shipping in IBM Z, LinuxONE
and Power systems. It is a dataflow architecture rather than a GPU: kernels are
compiled ahead of time for statically known shapes, there is no Triton and no
CUDA-style runtime beneath it, and the device is driven through PyTorch by IBM's
`torch-spyre` backend. Support lands as an out-of-tree SGLang platform plugin,
discovered at startup through the standard platform entry point, with no fork of
SGLang and no vendored copy of the framework.

Models are served through the usual `sgl.Engine` API, with the scheduler, request
handling and sampling running unchanged. RadixCache prefix sharing works, and
needed no Spyre-specific code — the scheduler resolves the hit before the forward
is assembled, so it never becomes the backend's problem. Attention executes on the
accelerator in BF16 against IBM's kernels, which take four-dimensional inputs with
query and key/value lengths padded up to fixed bucket sizes; the plugin's job at
that seam is bucketing and shape bookkeeping, sitting inside SGLang's torch-native
attention backend. The model body can run on the device as well — RMSNorm, SiLU
and GeLU, rotary embeddings and the linear layers, registered through the
multi-platform op dispatch — with the residual stream staying resident across the
forward pass rather than returning to the host between layers. The KV cache can
live on the accelerator too, in a paged pool with a torch-native allocator, since
the Triton path the stock allocator reaches for has nothing to run on here.

Because shapes are static, each distinct bucket compiles its own graph, so a warm
process matters more on Spyre than on a GPU. That constraint is also what the next
round of work is aimed at: the Spyre compiler is gaining dynamic shapes, which
brings variable-length attention and real indexed gathers into reach, and with
them continuous batching and batch sizes above one. Tensor parallelism, the
low-precision formats the AIU handles natively, and broader model coverage follow.

The plugin is developed in the open and available today at
[github.com/torch-spyre/sglang-spyre](https://github.com/torch-spyre/sglang-spyre).
