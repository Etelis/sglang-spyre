#!/usr/bin/env python3
"""Generate the blog figures as fixed-theme SVGs.

Two files per figure, ``-light`` and ``-dark``, each with its colours baked in.
Deliberately *not* one file with a ``prefers-color-scheme`` query: an SVG
embedded via ``<img>`` resolves that against the viewer's OS setting rather than
the page it sits on, so a self-switching figure renders dark on a light article
whenever the reader's machine is in dark mode. The page picks, not the OS —
see the ``<picture>`` blocks in ../blog-sglang-on-spyre.md.

Run:  python3 docs/figures/generate.py
"""

from pathlib import Path

OUT = Path(__file__).parent

# Palette slots. Both columns are selected for their own surface, not flipped.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", track="#d6d5cd",
                  s1="#2a78d6", s2="#eb6834"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835", track="#3a3a37",
                  s1="#3987e5", s2="#d95926"),
}

SANS = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


def css(t):
    """Shared type scale. Sizes are set for a ~700px blog column at 1:1."""
    return f"""
  .bg   {{ fill: {t['surface']}; }}
  .t1   {{ fill: {t['ink']};   font: 600 17px {SANS}; }}
  .t2   {{ fill: {t['ink2']};  font: 400 13px {SANS}; }}
  .cat  {{ fill: {t['ink']};   font: 400 13.5px {SANS}; }}
  .sub  {{ fill: {t['muted']}; font: 400 12px {SANS}; }}
  .grp  {{ fill: {t['ink2']};  font: 600 11.5px {SANS};
           letter-spacing: .07em; text-transform: uppercase; }}
  .val  {{ fill: {t['ink2']};  font: 400 12.5px {SANS};
           font-variant-numeric: tabular-nums; }}
  .zero {{ fill: {t['muted']}; font: 400 12.5px {SANS}; }}
  .tick {{ fill: {t['muted']}; font: 400 12px {SANS};
           font-variant-numeric: tabular-nums; text-anchor: middle; }}
  .note {{ fill: {t['muted']}; font: 400 12px {SANS}; }}
  .gl   {{ stroke: {t['grid']}; stroke-width: 1; }}
  .ax   {{ stroke: {t['axis']}; stroke-width: 1; }}
  .tk   {{ stroke: {t['muted']}; stroke-width: 1; }}
  .gap  {{ fill: {t['surface']}; }}
  .trk  {{ fill: {t['track']}; }}
  .a    {{ fill: {t['s1']}; }}
  .b    {{ fill: {t['s2']}; }}
"""


def bar(cls, x, y, w, h, r=4):
    """Horizontal bar: square at the baseline, 4px rounded data-end."""
    if w <= r:
        return f'<rect class="{cls}" x="{x}" y="{y}" width="{max(w,1):.1f}" height="{h}"/>'
    return (f'<path class="{cls}" d="M{x},{y} H{x+w-r:.1f} a{r},{r} 0 0 1 {r},{r} '
            f'V{y+h-r} a{r},{r} 0 0 1 -{r},{r} H{x} Z"/>')


# ---------------------------------------------------------------------------
# Figure 1 — plugin size by component
# ---------------------------------------------------------------------------

# (label, sub-label, vLLM lines, SGLang lines, zero-note for vLLM, zero-note for SGLang)
ROWS_A = [
    ("Attention",                 None,                            1142, 696, None, None),
    ("Worker + ModelRunner",      None,                             483,   0, None, "0 — no such class in SGLang"),
    ("Custom ops",                "norm, activation, RoPE, linear",  516, 421, None, None),
    ("Platform + registration",   None,                             187, 268, None, None),
    ("KV pool + allocator",       None,                               0, 111, "0 — inside vLLM's ModelRunner", None),
    ("Init, env, boundary hooks", None,                             163, 178, None, None),
]
ROWS_B = [
    ("Tensor-parallel comms", None, 184, 0, None, "0"),
    ("Parallel LM head",      None, 122, 0, None, "0"),
]

X0, PLOTW, VMAX = 228, 380, 1200
SCALE = PLOTW / VMAX
BARH, BARGAP, PITCH = 13, 2, 42
W1, H1 = 700, 572


def figure1(theme):
    t = THEMES[theme]
    o = []
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W1} {H1}" '
             f'width="{W1}" height="{H1}" role="img" aria-label="Plugin size by '
             f'component. spyre-inference on vLLM totals 2797 non-blank lines; '
             f'sglang-spyre on SGLang totals 1674. The largest single difference is '
             f'the Worker and ModelRunner category, 483 lines in vLLM and absent '
             f'entirely from the SGLang plugin.">')
    o.append(f"<style>{css(t)}</style>")
    o.append(f'<rect class="bg" x="0" y="0" width="{W1}" height="{H1}"/>')

    o.append('<text class="t1" x="20" y="30">Where the plugin code goes</text>')
    o.append('<text class="t2" x="20" y="52">Non-blank lines of Python, counted the '
             'same way on both plugins. Totals: 2,797 vs 1,674.</text>')

    # legend
    o.append('<rect class="a" x="20" y="65" width="11" height="11" rx="2"/>')
    o.append('<text class="t2" x="37" y="75">spyre-inference (vLLM)</text>')
    o.append('<rect class="b" x="192" y="65" width="11" height="11" rx="2"/>')
    o.append('<text class="t2" x="209" y="75">sglang-spyre (SGLang)</text>')

    top_a, top_b = 114, 412
    bottom = top_b + len(ROWS_B) * PITCH - (PITCH - 2 * BARH - BARGAP)

    # gridlines + axis
    for v in (400, 800, 1200):
        gx = X0 + v * SCALE
        o.append(f'<line class="gl" x1="{gx:.1f}" y1="106" x2="{gx:.1f}" y2="{bottom}"/>')
        o.append(f'<text class="tick" x="{gx:.1f}" y="{bottom+22}">{v:,}</text>')
    o.append(f'<line class="ax" x1="{X0}" y1="106" x2="{X0}" y2="{bottom}"/>')
    o.append(f'<text class="tick" x="{X0}" y="{bottom+22}">0</text>')

    def emit_rows(rows, top):
        for i, (label, sub, va, vb, za, zb) in enumerate(rows):
            y = top + i * PITCH
            ya, yb = y, y + BARH + BARGAP
            o.append(f'<text class="cat" x="20" y="{y + 18}">{label}</text>')
            if sub:
                o.append(f'<text class="sub" x="20" y="{y + 33}">{sub}</text>')
            for v, z, yy, cls in ((va, za, ya, "a"), (vb, zb, yb, "b")):
                if v:
                    w = v * SCALE
                    o.append(bar(cls, X0, yy, w, BARH))
                    o.append(f'<text class="val" x="{X0 + w + 12:.1f}" '
                             f'y="{yy + 11}">{v:,}</text>')
                elif z:
                    o.append(f'<text class="zero" x="{X0 + 6}" y="{yy + 11}">{z}</text>')

    o.append('<text class="grp" x="20" y="100">Comparable surface</text>')
    emit_rows(ROWS_A, top_a)

    o.append(f'<line class="gl" x1="20" y1="376" x2="{W1-20}" y2="376"/>')
    o.append('<text class="grp" x="20" y="398">Not implemented here — not a '
             'framework saving</text>')
    emit_rows(ROWS_B, top_b)

    o.append(f'<text class="note" x="20" y="{bottom+56}">On comparable surface the '
             'SGLang plugin is 1,674 lines against 2,491 — about a third less code. '
             'The 306 lines</text>')
    o.append(f'<text class="note" x="20" y="{bottom+74}">below the rule are features '
             'it does not have yet, and are roughly a quarter of the raw '
             'difference.</text>')
    o.append("</svg>")
    return "\n".join(o)


# ---------------------------------------------------------------------------
# Figure 2 — prefix-sharing granularity
# ---------------------------------------------------------------------------

SX, STRIPW = 180, 357          # both strips share a left edge and total width
BLOCK = 16                     # vLLM block size, in tokens
W2, H2 = 700, 438


def figure2(theme):
    t = THEMES[theme]
    o = []
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W2} {H2}" '
             f'width="{W2}" height="{H2}" role="img" aria-label="Block-aligned '
             f'prefix sharing rounds down to the last whole 16-token block, sharing '
             f'192 of 200 tokens and 48 of 51. Radix sharing ends where the prefix '
             f'does, sharing 200 of 200 and 51 of 51.">')
    o.append(f"<style>{css(t)}</style>")
    o.append(f'<rect class="bg" x="0" y="0" width="{W2}" height="{H2}"/>')

    o.append('<text class="t1" x="20" y="30">Where prefix sharing stops</text>')
    o.append('<text class="t2" x="20" y="53">Block-aligned sharing reuses whole '
             'blocks only, so it gives up the tail.</text>')
    o.append('<text class="t2" x="20" y="71">Radix sharing ends where the prefix '
             'does.</text>')

    # legend
    o.append('<rect class="a" x="20" y="86" width="11" height="11" rx="2"/>')
    o.append('<text class="t2" x="37" y="96">shared — vLLM, 16-token blocks</text>')
    o.append('<rect class="b" x="242" y="86" width="11" height="11" rx="2"/>')
    o.append('<text class="t2" x="259" y="96">shared — SGLang RadixCache</text>')
    o.append('<rect class="trk" x="462" y="86" width="11" height="11" rx="2"/>')
    o.append('<text class="t2" x="479" y="96">recomputed</text>')

    LBLX = 549   # both value labels align here

    # ---- panel 1: 200 tokens, continuous strip ----
    n1, h1 = 200, 18
    per1 = STRIPW / n1
    shared1 = (n1 // BLOCK) * BLOCK          # 192
    o.append('<text class="grp" x="20" y="125">A 200-token system prompt</text>')

    o.append('<text class="cat" x="20" y="150">vLLM</text>')
    o.append(f'<rect class="a" x="{SX}" y="137" width="{shared1*per1:.1f}" '
             f'height="{h1}" rx="2"/>')
    o.append(f'<rect class="trk" x="{SX+shared1*per1:.1f}" y="137" '
             f'width="{(n1-shared1)*per1:.1f}" height="{h1}" rx="2"/>')
    for k in range(1, n1 // BLOCK + 1):      # block gaps; surface does the dividing
        gx = SX + k * BLOCK * per1
        w = 2 if k * BLOCK == shared1 else 1.5
        o.append(f'<rect class="gap" x="{gx-w/2:.1f}" y="137" width="{w}" height="{h1}"/>')
    o.append(f'<text class="val" x="{LBLX}" y="150">192 of 200</text>')

    o.append('<text class="cat" x="20" y="190">SGLang</text>')
    o.append(f'<rect class="b" x="{SX}" y="177" width="{STRIPW}" height="{h1}" rx="2"/>')
    o.append(f'<text class="val" x="{LBLX}" y="190">200 of 200</text>')

    o.append(f'<text class="note" x="{SX}" y="218">The partial tail block does not '
             'count — 8 tokens re-prefilled every request.</text>')

    o.append(f'<line class="gl" x1="20" y1="242" x2="{W2-20}" y2="242"/>')

    # ---- panel 2: 51 tokens, one cell each ----
    n2, h2, pitch = 51, 16, 7
    shared2 = (n2 // BLOCK) * BLOCK          # 48
    o.append('<text class="grp" x="20" y="266">The measured case — 51-token shared '
             'prefix, one cell per token</text>')

    def cells(y, n_shared, cls, n_total):
        # One fill, then surface gaps cut it into per-token cells.
        w_shared = n_shared * pitch - 1
        out = [f'<rect class="{cls}" x="{SX}" y="{y}" width="{w_shared}" '
               f'height="{h2}" rx="2"/>']
        if n_total > n_shared:
            gx = SX + n_shared * pitch
            out.append(f'<rect class="trk" x="{gx}" y="{y}" '
                       f'width="{(n_total-n_shared)*pitch-1}" height="{h2}" rx="2"/>')
        for i in range(n_total - 1):
            # wider gap on a 16-token block boundary, so the structure is visible
            w = 2 if (i + 1) % BLOCK == 0 else 1
            out.append(f'<rect class="gap" x="{SX+i*pitch+pitch-1:.1f}" y="{y}" '
                       f'width="{w}" height="{h2}"/>')
        return out

    o.append('<text class="cat" x="20" y="292">vLLM</text>')
    o.extend(cells(279, shared2, "a", n2))
    o.append(f'<text class="val" x="{LBLX}" y="292">48 of 51</text>')
    for k in range(1, n2 // BLOCK + 1):
        tx = SX + k * BLOCK * pitch - 0.5
        o.append(f'<line class="tk" x1="{tx:.1f}" y1="299" x2="{tx:.1f}" y2="305"/>')
    o.append(f'<text class="note" x="{SX}" y="319">16-token block boundaries — '
             'sharing can only stop on one, so 3 tokens are lost.</text>')

    o.append('<text class="cat" x="20" y="349">SGLang</text>')
    o.extend(cells(336, n2, "b", n2))
    o.append(f'<text class="val" x="{LBLX}" y="349">51 of 51</text>')
    o.append(f'<text class="note" x="{SX}" y="376">No block structure to round '
             'down to.</text>')

    o.append('<text class="note" x="20" y="408">Measured on Granite-1B: four prompts '
             'sharing a 51-token prefix reported</text>')
    o.append('<text class="note" x="20" y="426">cached_tokens = [0, 51, 50, 51] — an '
             '82% prompt cache hit.</text>')
    o.append("</svg>")
    return "\n".join(o)


if __name__ == "__main__":
    for theme in THEMES:
        for name, fn in (("plugin-size-by-component", figure1),
                         ("prefix-sharing-granularity", figure2)):
            path = OUT / f"{name}-{theme}.svg"
            path.write_text(fn(theme) + "\n")
            print(f"wrote {path.relative_to(OUT.parent.parent)}")
