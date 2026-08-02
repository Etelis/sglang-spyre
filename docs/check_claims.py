#!/usr/bin/env python3
"""Check the measured claims in docs/sglang-vs-vllm.md against their source data.

The blog post is a narrative enablement piece and deliberately carries no
figures. All measurements live in the comparison doc, so that is what this
guards — plus a check that the blog stays free of numbers.

Every defect below was one we actually shipped into a draft and caught later. The
failure mode is always the same: a sentence that was true when written and quietly
became false when the numbers underneath it moved. Nothing about such a sentence
looks wrong in isolation, so re-reading does not find it.

Run:  python3 docs/check_claims.py
Exits non-zero if any check fails, so it can gate a commit.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CMP = ROOT / "docs" / "sglang-vs-vllm.md"
BLOG = ROOT / "docs" / "blog-sglang-on-spyre.md"
FIGURES = ROOT / "docs" / "figures"

# ---------------------------------------------------------------- source data
# Measured 2026-07-26, one machine, Granite-1B, 200 output tokens, batch 1.
TPS = {"cpu": 37.64, "spyre": 6.13, "body": 6.56, "paged": 12.39}
LOAD = {"spyre": 96.6, "body": 143.8}

# Non-blank lines of Python, plugin code only, no tests.
SI = {"attn": 1142, "worker": 483, "ops": 516, "plat": 187, "other": 163,
      "tp": 184, "lm": 122}
SG = {"attn": 696, "worker": 0, "ops": 421, "plat": 268, "other": 178, "kv": 111}
NOT_IMPLEMENTED = ("tp", "lm")           # features absent here, not framework wins

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def approx(name: str, value: float, lo: float, hi: float) -> None:
    ok = lo <= value <= hi
    check(name, ok, f"{value:.2f}" if ok else f"{value:.2f} not in [{lo}, {hi}]")


cmp_doc = CMP.read_text()
blog = BLOG.read_text()

# ------------------------------------------------------- 1. derived arithmetic
print("\n[1] derived figures vs measured data")
approx("CPU is ~6x the attention-only mode", TPS["cpu"] / TPS["spyre"], 5.5, 6.5)
approx("paged is ~2x the attention-only mode", TPS["paged"] / TPS["spyre"], 1.8, 2.2)
approx("on-device body gains ~7%", (TPS["body"] / TPS["spyre"] - 1) * 100, 6.0, 8.0)
approx("model load ~50% longer", (LOAD["body"] / LOAD["spyre"] - 1) * 100, 45, 55)

comparable_si = sum(v for k, v in SI.items() if k not in NOT_IMPLEMENTED)
approx("comparable surface ~a third less code",
       (1 - sum(SG.values()) / comparable_si) * 100, 30, 36)
approx("not-implemented rows ~a quarter of the raw gap",
       sum(SI[k] for k in NOT_IMPLEMENTED) / (sum(SI.values()) - sum(SG.values())) * 100,
       23, 28)

# ------------------------------------- 2. totals quoted in the comparison doc
print("\n[2] line-count totals are correct and present in the comparison doc")
for label, want in (("2,797", sum(SI.values())),
                    ("1,674", sum(SG.values())),
                    ("2,491", comparable_si)):
    check(f"{label} is arithmetically correct", int(label.replace(",", "")) == want,
          f"data gives {want}")
    check(f"{label} present in comparison doc", label in cmp_doc)

# ------------------------------------------- 3. superlatives are scoped to Spyre
# paged (12.39) beats the other Spyre modes but loses to CPU (37.64) by ~3x, so
# any unqualified "fastest" contradicts the table it sits beside.
#
# Two lessons baked in, both learned by shipping the bug: normalise whitespace
# before testing (hard wrapping splits qualifiers across newlines), and do not
# require "paged" and "fastest" in one sentence — the real defect put the subject
# in the previous sentence as "This one runs, and it's the fastest of the four".
print("\n[3] 'fastest' claims are scoped to Spyre modes")
unscoped = [
    flat for flat in (
        " ".join(s.split()) for s in re.split(r"(?<=[.;])\s", cmp_doc)
        if "fastest" in s.lower()
    )
    if "Spyre mode" not in flat
]
check("no unqualified 'fastest' claims", not unscoped,
      f"{len(unscoped)} found: {unscoped[:1]}")

# ------------------------------- 4. the unreconcilable cache percentage stays out
# cached_tokens = [0,51,50,51] is 152 tokens; four prompts each containing the
# 51-token prefix total >= 204, so at most 74.5%. The repo's 82% only follows if
# the cold first request is excluded. Keep it out until the denominator is stated.
print("\n[4] the unreconciled 82% cache figure stays out")
check("82% absent from the blog", "82%" not in blog)
for svg in sorted(FIGURES.glob("*.svg")):
    check(f"82% absent from {svg.name}", "82%" not in svg.read_text())

# ------------------------------------------------ 5. the blog stays a narrative
# The post is an enablement piece; measurements belong in the comparison doc,
# where the checks above guard them. The lmsys URL contains a date, so skip it.
print("\n[5] the blog carries no performance figures")
digits = re.findall(r"\b\d+(?:[.,]\d+)?\b",
                    "\n".join(l for l in blog.splitlines() if "lmsys.org" not in l))
check("blog prose is free of numeric claims", not digits, f"found {digits[:5]}")

# --------------------------------------------------------------------- verdict
print()
if failures:
    print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
