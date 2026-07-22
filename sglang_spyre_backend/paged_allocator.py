"""Torch-native paged KV allocator for Spyre (no Triton).

SGLang's ``PagedTokenToKVPoolAllocator.alloc_extend`` / ``alloc_decode`` dispatch
the Triton kernels ``alloc_extend_kernel`` / ``alloc_decode_kernel``
(mem_cache/allocator.py). Triton has no Spyre backend, so on Spyre they crash
with ``RuntimeError: 0 active drivers``. (The non-paged page_size=1 allocator
avoids this — its ``alloc`` is pure torch — which is why the dense ``spyre``
backend never hit it; only ``page_size>1`` pulls in the Triton allocator.)

This subclass overrides only those two methods with pure-torch equivalents that
reproduce the *exact* page-fill layout of the kernels, so each request's
``req_to_token`` slots stay page-aligned and contiguous (``slot = page*page_size
+ offset``) — the property SpyreAttentionBackend's gather relies on.

bs == 1 only, matching SpyreAttentionBackend's single-sequence constraint
(max_running_requests=1). Everything else (alloc, free, clear, free_pages
accounting, need_sort) is inherited unchanged.
"""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator


class SpyrePagedAllocator(PagedTokenToKVPoolAllocator):
    def alloc_extend(
        self,
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
        num_new_pages=None,
    ):
        bs = len(prefix_lens)
        assert bs == 1, "SpyrePagedAllocator supports bs==1 only"
        ps = self.page_size
        pre = int(prefix_lens_cpu[0].item())
        seq = int(seq_lens_cpu[0].item())
        last = int(last_loc[0].item())

        def ceil_to_page(a):
            return (a + ps - 1) // ps * ps

        # pages newly required by this sequence (matches get_num_new_pages, bs=1)
        n_new = (seq + ps - 1) // ps - (pre + ps - 1) // ps
        if self.need_sort and n_new > len(self.free_pages):
            self.merge_and_sort_free()
        if n_new > len(self.free_pages):
            return None
        new_pages = self.free_pages[:n_new]  # physical page ids for this seq

        out = torch.empty((extend_num_tokens,), dtype=torch.int64, device=self.device)
        pos = 0
        # Part 1: fill the tail of the prefix's (partial) page, from last_loc+1.
        num_part1 = min(seq, ceil_to_page(pre)) - pre
        if num_part1 > 0:
            out[pos : pos + num_part1] = last + 1 + torch.arange(num_part1, device=self.device)
            pos += num_part1
        # Part 2: fill whole new pages.
        num_part2 = (seq // ps) * ps - ceil_to_page(pre)
        if num_part2 > 0:
            off = torch.arange(num_part2, device=self.device)
            page = new_pages[off // ps]
            out[pos : pos + num_part2] = page * ps + (off % ps)
            pos += num_part2
        # Part 3: fill the final new (partial) page.
        num_part3 = seq - (seq // ps) * ps
        if num_part3 > 0:
            start = int(new_pages[n_new - 1].item())
            out[pos : pos + num_part3] = start * ps + torch.arange(num_part3, device=self.device)
            pos += num_part3

        self.free_pages = self.free_pages[n_new:]
        return out

    def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
        bs = len(seq_lens)
        assert bs == 1, "SpyrePagedAllocator supports bs==1 only"
        ps = self.page_size
        seq = int(seq_lens_cpu[0].item())
        pre = seq - 1
        last = int(last_loc[0].item())

        n_new = (seq + ps - 1) // ps - (pre + ps - 1) // ps  # 0 or 1
        if self.need_sort and bs > len(self.free_pages):
            self.merge_and_sort_free()

        out = torch.empty((bs,), dtype=torch.int64, device=self.device)
        if n_new == 0:
            out[0] = last + 1  # same page as the prefix's last token
        else:
            if n_new > len(self.free_pages):
                return None
            page = int(self.free_pages[0].item())
            out[0] = page * ps  # first slot of a fresh page
            self.free_pages = self.free_pages[n_new:]
        return out
