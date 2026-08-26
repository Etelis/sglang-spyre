# SGLang 0.5.18 compatibility patch

SGLang 0.5.18 supports out-of-tree devices, but three small compatibility gaps
remain for a text-only Spyre environment: `support_triton()` treats every
custom attention backend as Triton-capable, the common utilities import
`torchvision` unconditionally, and Transformers 5.14 already owns the Qwen3-ASR
configuration names that SGLang registers.

Run this once in the environment where the pinned SGLang is installed:

```bash
python -m sglang_oot_patches.apply
```

The idempotent patch excludes `spyre` and `spyre_paged` from Triton dispatch,
makes the unused image decoder optional, and makes the Qwen3-ASR registration
idempotent. It fails loudly if the pinned SGLang source has drifted instead of
silently producing a partially patched environment.
