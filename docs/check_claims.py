#!/usr/bin/env python3
"""Check the blog post's factual claims against the measured data.

Every defect this guards against was one we actually shipped into a draft and
caught later. The failure mode is always the same: a sentence that was true when
written and quietly became false when the numbers underneath it moved. Nothing
about such a sentence looks wrong in isolation, so re-reading does not find it.

Run:  python3 docs/check_claims.py
Exits non-zero if any check fails, so it can gate a commit.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BLOG = ROOT / "docs" / "blog-sglang-on-spyre.md"
CMP = ROOT / "docs" / "sglang-vs-vllm.md"

# ---------------------------------------------------------------- source data
# Measured 2026-07-26, one machine, Granite-1B, 200 output tokens, batch 1.
TPS = {"cpu": 37.64, "spyre": 6.13, "body": 6.56, "paged": 12.39}
LOAD = {"spyre": 96.6, "body": 143.8}
COLD_WARM = (22.3, 0.8)

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
    check(name, lo <= value <= hi, f"{value:.2f} not in [{lo}, {hi}]" if not (lo <= value <= hi) else f"{value:.2f}")


blog = BLOG.read_text()
cmp_doc = CMP.read_text()

# ------------------------------------------------------- 1. derived arithmetic
print("\n[1] derived figures vs measured data")
approx("CPU is ~6x the attention-only mode", TPS["cpu"] / TPS["spyre"], 5.5, 6.5)
approx("paged is ~2x the attention-only mode", TPS["paged"] / TPS["spyre"], 1.8, 2.2)
approx("on-device body gains ~7%", (TPS["body"] / TPS["spyre"] - 1) * 100, 6.0, 8.0)
approx("model load ~50% longer", (LOAD["body"] / LOAD["spyre"] - 1) * 100, 45, 55)
approx("cold/warm ~28x", COLD_WARM[0] / COLD_WARM[1], 27, 29)

comparable_si = sum(v for k, v in SI.items() if k not in NOT_IMPLEMENTED)
approx("comparable surface ~a third less code",
       (1 - sum(SG.values()) / comparable_si) * 100, 30, 36)
approx("not-implemented rows ~a quarter of the raw gap",
       sum(SI[k] for k in NOT_IMPLEMENTED) / (sum(SI.values()) - sum(SG.values())) * 100, 23, 28)

# --------------------------------------------------- 2. totals quoted in prose
print("\n[2] line-count totals appear correctly in both documents")
for label, want in (("2,797", sum(SI.values())),
                    ("1,674", sum(SG.values())),
                    ("2,491", comparable_si)):
    check(f"{label} is arithmetically correct", int(label.replace(",", "")) == want,
          f"prose says {label}, data gives {want}")
    check(f"{label} present in blog", label in blog)
    check(f"{label} present in comparison doc", label in cmp_doc)

# ------------------------------------------- 3. superlatives are scoped to Spyre
# paged (12.39) beats the other Spyre modes but loses to CPU (37.64) by ~3x, so
# any unqualified "fastest" contradicts the post's own table.
print("\n[3] 'fastest' claims about spyre_paged are scoped to Spyre modes")
# Two lessons are baked in here, both learned by shipping the bug.
#
# 1. Normalise whitespace *before* testing. The document is hard-wrapped, so a
#    qualifying phrase like "Spyre modes" is routinely split across a newline;
#    testing the raw match reports a false failure on correctly-scoped prose.
# 2. Do not require "paged" and "fastest" in the same sentence. The real defect
#    read "...the KV cache itself living on the device. This one runs, and it's
#    the fastest of the four" — the subject is a pronoun in the next sentence,
#    so a same-sentence regex silently passes it. Every "fastest" in this post
#    refers to spyre_paged, so require all of them to be scoped.
unscoped = [
    flat for flat in (
        " ".join(s.split()) for s in re.split(r"(?<=[.;])\s", blog) if "fastest" in s.lower()
    )
    if "Spyre mode" not in flat
]
check("no unqualified 'fastest' claims", not unscoped,
      f"{len(unscoped)} found: {unscoped[:1]}")

# ------------------------------------------ 4. each section states its own result
print("\n[4] major sections state their own headline result")
# Needles must be specific to the claim being guarded. An early version looked
# for bare "byte-identical", which also occurs in the mode-4-vs-mode-2 sentence,
# so deleting the CPU-parity result left the check satisfied.
for title, needles in {
    "Why it took a third less code": ["1,674", "2,491"],
    "The numbers, honestly": ["byte-identical to CPU SDPA", "37.64", "12.39"],
    "What the framework didn't do for us": ["upstream"],
}.items():
    section = next((s for s in re.split(r"\n## ", blog) if s.startswith(title)), "")
    missing = [n for n in needles if n not in section]
    check(f"{title!r} is self-contained", section and not missing, f"missing {missing}")

# ------------------------------- 5. the unreconcilable cache percentage stays out
# cached_tokens = [0,51,50,51] is 152 tokens; four prompts each containing the
# 51-token prefix total >= 204, so at most 74.5%. The repo's 82% only follows if
# the cold first request is excluded. Keep it out of prose and figures until the
# denominator is stated.
print("\n[5] the unreconciled 82% cache figure stays out of prose and figures")
prose = re.sub(r"<!--.*?-->", "", blog, flags=re.S)     # review comments may discuss it
check("82% absent from blog prose", "82%" not in prose)
for svg in sorted((ROOT / "docs" / "figures").glob("*.svg")):
    check(f"82% absent from {svg.name}", "82%" not in svg.read_text())

# ------------------------------------------------------------- 6. link targets
print("\n[6] repo-relative links resolve")
for target in re.findall(r"\]\((\.\./[^)]+)\)", blog):
    check(f"link {target}", (BLOG.parent / target).resolve().exists())

# --------------------------------------------------------------------- verdict
print()
if failures:
    print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
