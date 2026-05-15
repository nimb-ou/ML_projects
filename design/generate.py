"""Generate the design/* assets for Digit Memory v2.

Philosophy: Computational Quiet — Swiss grid, monumental numerals, single
accent. See DESIGN_PHILOSOPHY.md.

Outputs:
    design/architecture.png          16x10 architecture diagram
    design/digit_memory_poster.pdf   tabloid 11x17 poster
    design/digit_memory_case_study.pdf  letter multi-page case study
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager
from sklearn.datasets import load_digits

# ---------- fonts ----------
FONT_DIR = Path(
    "/Users/nimitjain/Library/Application Support/Claude/"
    "local-agent-mode-sessions/skills-plugin/"
    "464deac6-5c08-4842-b426-8d44c09a4920/"
    "21aee2b7-abfa-431a-9896-6a1013e0fe6b/skills/canvas-design/canvas-fonts"
)
for name in (
    "BigShoulders-Bold.ttf",
    "BigShoulders-Regular.ttf",
    "InstrumentSans-Regular.ttf",
    "InstrumentSans-Bold.ttf",
    "InstrumentSerif-Italic.ttf",
    "InstrumentSerif-Regular.ttf",
    "GeistMono-Regular.ttf",
    "GeistMono-Bold.ttf",
    "IBMPlexMono-Regular.ttf",
):
    fp = FONT_DIR / name
    if fp.exists():
        fontManager.addfont(str(fp))

DISPLAY = FontProperties(family="Big Shoulders", weight="bold")
DISPLAY_REG = FontProperties(family="Big Shoulders")
BODY = FontProperties(family="Instrument Sans")
BODY_BOLD = FontProperties(family="Instrument Sans", weight="bold")
SERIF = FontProperties(family="Instrument Serif", style="italic")
SERIF_REG = FontProperties(family="Instrument Serif")
MONO = FontProperties(family="Geist Mono")
MONO_BOLD = FontProperties(family="Geist Mono", weight="bold")

# ---------- palette ----------
BG = "#F2EFE6"
INK = "#131312"
INK_SOFT = "#3A3A37"
INK_MUTE = "#8A8A85"
RULE = "#1A1A19"
ACCENT = "#B8331F"

HERE = Path(__file__).parent
OUT = HERE


def base_axes(fig):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def hairline(ax, x0, y0, x1, y1, color=RULE, lw=0.4):
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
            solid_capstyle="butt")


def label(ax, x, y, text, fp, size, color=INK, ha="left", va="baseline", **kw):
    ax.text(x, y, text, fontproperties=fp, fontsize=size, color=color,
            ha=ha, va=va, **kw)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=0.7):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>,head_length=0.18,head_width=0.09",
                                color=color, lw=lw, shrinkA=0, shrinkB=0))


# ---------- architecture.png ----------
def build_architecture():
    fig = plt.figure(figsize=(16, 10), dpi=200)
    ax = base_axes(fig)
    M = 0.06

    # frame
    hairline(ax, M, 1 - M, 1 - M, 1 - M)
    hairline(ax, M, M, 1 - M, M)

    # header
    label(ax, M, 1 - M + 0.012, "FIG.01  ·  TWO-TIER MEMORY",
          MONO_BOLD, 11, color=INK)
    label(ax, 1 - M, 1 - M + 0.012,
          "DIGIT  MEMORY  /  ARCHITECTURE",
          MONO, 11, color=INK_MUTE, ha="right")

    # title block
    label(ax, M, 0.85, "QUERY", DISPLAY, 64, color=INK)
    label(ax, M, 0.79, "PATH",  DISPLAY, 64, color=INK)
    label(ax, M, 0.74,
          "Hash first. Tree on miss. Return whichever answer arrives.",
          SERIF, 18, color=INK_SOFT)

    def box(x, y, w, h, top, big, sub=None, accent=False):
        ec = ACCENT if accent else INK
        lw = 1.0 if accent else 0.55
        ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False,
                                        edgecolor=ec, linewidth=lw))
        label(ax, x + 0.012, y + h - 0.028, top,
              MONO, 10, color=INK_MUTE)
        label(ax, x + 0.012, y + 0.058, big, DISPLAY, 44,
              color=ACCENT if accent else INK)
        if sub:
            label(ax, x + 0.012, y + 0.025, sub, MONO, 10, color=INK_SOFT)

    # layout
    qx, qy, qw, qh = 0.08, 0.32, 0.18, 0.18
    hx, hy, hw, hh = 0.38, 0.44, 0.20, 0.18
    tx, ty, tw, th = 0.38, 0.18, 0.20, 0.18
    ox, oy, ow, oh = 0.72, 0.32, 0.18, 0.18

    box(qx, qy, qw, qh, "STAGE 01",  "QUERY",  "64 floats")
    box(hx, hy, hw, hh, "STAGE 02a", "HASH",   "O(1) bytes key", accent=True)
    box(tx, ty, tw, th, "STAGE 02b", "TREE",   "O(log n)  BallTree")
    box(ox, oy, ow, oh, "STAGE 03",  "LABEL",  "int  0..9")

    # arrows
    arrow(ax, qx + qw, qy + qh / 2, hx, hy + hh / 2)
    arrow(ax, hx + hw, hy + hh / 2, ox, oy + oh / 2 + 0.02,
          color=ACCENT, lw=1.1)
    arrow(ax, hx + hw / 2, hy, tx + tw / 2, ty + th)
    arrow(ax, tx + tw, ty + th / 2, ox, oy + oh / 2 - 0.02)

    label(ax, hx + hw + 0.008, hy + hh / 2 + 0.022, "HIT",
          MONO_BOLD, 10, color=ACCENT)
    label(ax, hx + hw / 2 + 0.008, hy - 0.025, "MISS",
          MONO_BOLD, 10, color=INK_MUTE)

    # footer
    hairline(ax, M, 0.10, 1 - M, 0.10)
    label(ax, M + 0.005, 0.075, "EXACT MATCH FIRST",
          MONO_BOLD, 10, color=INK)
    label(ax, M + 0.005, 0.05,
          "When the input is byte-identical to a stored sample, the hash returns the label immediately.",
          MONO, 9, color=INK_SOFT)
    label(ax, 1 - M - 0.005, 0.075, "NN FALLBACK",
          MONO_BOLD, 10, color=INK, ha="right")
    label(ax, 1 - M - 0.005, 0.05,
          "Otherwise, the tree finds the closest stored vector and we return its label.",
          MONO, 9, color=INK_SOFT, ha="right")

    fig.savefig(OUT / "architecture.png", dpi=200, facecolor=BG,
                bbox_inches=None)
    plt.close(fig)


# ---------- poster ----------
def build_poster():
    fig = plt.figure(figsize=(11, 17), dpi=200)
    ax = base_axes(fig)
    M = 0.07

    # frame rules
    hairline(ax, M, 0.965, 1 - M, 0.965)
    hairline(ax, M, 0.035, 1 - M, 0.035)

    # header strip
    label(ax, M, 0.975, "DIGIT  MEMORY  /  v2",
          MONO, 9, color=INK_MUTE)
    label(ax, 1 - M, 0.975,
          "EDITION 01  /  RETRIEVAL AS A BASELINE",
          MONO, 9, color=INK_MUTE, ha="right")

    # title — generous line spacing (line-height ~1.3x cap)
    label(ax, M, 0.925, "DIGIT", DISPLAY, 76, color=INK)
    label(ax, M, 0.870, "MEMORY", DISPLAY, 76, color=INK)
    label(ax, M, 0.838,
          "A study in retrieval as classification.",
          SERIF, 16, color=INK_SOFT)
    hairline(ax, M, 0.815, 1 - M, 0.815)

    # primary metric
    label(ax, M, 0.715, "98.4", DISPLAY, 130, color=ACCENT)
    label(ax, M + 0.28, 0.730, "%", DISPLAY_REG, 50, color=ACCENT)
    label(ax, M, 0.695, "HELD-OUT ACCURACY",
          MONO_BOLD, 11, color=INK)
    label(ax, M, 0.678,
          "1-NN on a stratified 75/25 split, sklearn digits.",
          MONO, 10, color=INK_SOFT)

    hairline(ax, M, 0.655, 1 - M, 0.655)

    # supporting numbers — three columns
    col_w = (1 - 2 * M) / 3
    cols = [
        ("STAGE 02A · HASH",  "3",  "us",
         "EXACT LOOKUP, O(1)",
         "bytes-keyed dict."),
        ("STAGE 02B · TREE",  "50", "us",
         "NEAREST NEIGHBOR, O(log n)",
         "BallTree / KDTree."),
        ("STAGE 03 · LABEL",  "10", "classes",
         "DIGITS  0 .. 9",
         "1797 samples."),
    ]
    y_top = 0.640
    for i, (top, big, unit, mid, sub) in enumerate(cols):
        cx = M + i * col_w
        label(ax, cx, y_top, top, MONO, 9, color=INK_MUTE)
        # numeral
        label(ax, cx, y_top - 0.050, big, DISPLAY, 78, color=INK)
        # unit beside number
        big_w = 0.085 if len(big) >= 2 else 0.045
        label(ax, cx + big_w, y_top - 0.038, unit,
              DISPLAY_REG, 28, color=INK)
        label(ax, cx, y_top - 0.078, mid, MONO_BOLD, 10, color=INK)
        label(ax, cx, y_top - 0.094, sub, MONO, 9, color=INK_SOFT)

    hairline(ax, M, 0.520, 1 - M, 0.520)

    # architecture mini diagram
    label(ax, M, 0.505, "FIG.01  THE TWO-TIER LOOKUP",
          MONO_BOLD, 10, color=INK)

    # five-box layout
    bw, bh = 0.13, 0.045
    row1_y = 0.440
    row2_y = 0.380

    def mb(x, y, w, h, t, accent=False):
        ec = ACCENT if accent else INK
        ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False,
                                        edgecolor=ec,
                                        linewidth=0.9 if accent else 0.55))
        label(ax, x + w / 2, y + h / 2 - 0.004, t, MONO_BOLD, 9,
              color=ACCENT if accent else INK, ha="center", va="center")

    px = M
    mb(px,                 row1_y, bw, bh, "QUERY")
    mb(px + 0.21,          row1_y, bw, bh, "HASH", accent=True)
    mb(px + 0.42,          row1_y, bw, bh, "LABEL")
    mb(px + 0.21,          row2_y, bw, bh, "TREE")
    mb(px + 0.42,          row2_y, bw, bh, "LABEL")

    arrow(ax, px + bw,              row1_y + bh/2,
              px + 0.21,             row1_y + bh/2)
    arrow(ax, px + 0.21 + bw,       row1_y + bh/2,
              px + 0.42,             row1_y + bh/2, color=ACCENT, lw=1.0)
    arrow(ax, px + 0.21 + bw/2,     row1_y,
              px + 0.21 + bw/2,     row2_y + bh)
    arrow(ax, px + 0.21 + bw,       row2_y + bh/2,
              px + 0.42,            row2_y + bh/2)

    label(ax, px + 0.21 + bw + 0.005, row1_y + bh + 0.005, "HIT",
          MONO_BOLD, 9, color=ACCENT)
    label(ax, px + 0.21 + bw/2 + 0.005, row1_y - 0.020, "MISS",
          MONO_BOLD, 9, color=INK_MUTE)

    hairline(ax, M, 0.345, 1 - M, 0.345)

    # digit samples row
    label(ax, M, 0.330, "FIG.02  ONE SAMPLE PER CLASS",
          MONO_BOLD, 10, color=INK)
    digits = load_digits()
    sample_row_y = 0.225
    cell_w = (1 - 2 * M) / 10
    cell_h = cell_w
    for cls in range(10):
        sub = digits.images[digits.target == cls][0]
        ix0 = M + cls * cell_w + cell_w * 0.10
        iy0 = sample_row_y
        iw, ih = cell_w * 0.80, cell_h * 0.80
        inset = fig.add_axes([ix0, iy0, iw, ih])
        inset.imshow(sub, cmap="gray_r", interpolation="nearest")
        inset.set_xticks([])
        inset.set_yticks([])
        for s in inset.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.4)
        label(ax, ix0 + iw / 2, iy0 - 0.020, str(cls),
              MONO_BOLD, 10, color=INK_MUTE, ha="center")

    hairline(ax, M, 0.165, 1 - M, 0.165)

    # bottom block — robustness summary
    label(ax, M, 0.150,
          "ROBUSTNESS",
          MONO_BOLD, 10, color=INK)
    label(ax, M, 0.125,
          "Smooth degradation under injected Gaussian noise.",
          SERIF, 14, color=INK_SOFT)
    label(ax, M, 0.090,
          "sigma = 0  ->  98.4%      sigma = 5  ->  93.8%      sigma = 10  ->  67.1%",
          MONO_BOLD, 12, color=INK)
    label(ax, M, 0.068,
          "no cliff, no catastrophic forgetting, no retraining",
          MONO, 10, color=INK_SOFT)

    # footer
    label(ax, M, 0.020,
          "NIMIT JAIN  /  ML_PROJECTS  /  github.com/nimb-ou/ML_projects",
          MONO, 9, color=INK_MUTE)
    label(ax, 1 - M, 0.020, "EDITION 01",
          MONO, 9, color=INK_MUTE, ha="right")

    fig.savefig(OUT / "digit_memory_poster.pdf", facecolor=BG)
    plt.close(fig)


# ---------- case study ----------
def page_frame(fig, ax, M, num, section, total=8):
    hairline(ax, M, 0.945, 1 - M, 0.945)
    label(ax, M, 0.957, f"§ {num}  ·  {section.upper()}",
          MONO, 9, color=INK_MUTE)
    label(ax, 1 - M, 0.957, f"{num} / {total:02d}",
          MONO, 9, color=INK_MUTE, ha="right")
    hairline(ax, M, 0.055, 1 - M, 0.055)
    label(ax, M, 0.038, "DIGIT MEMORY · CASE STUDY",
          MONO, 9, color=INK_MUTE)
    label(ax, 1 - M, 0.038, f"PAGE {num}",
          MONO, 9, color=INK_MUTE, ha="right")


def build_case_study():
    pp = PdfPages(OUT / "digit_memory_case_study.pdf")
    M = 0.085

    # PAGE 1 — cover
    fig = plt.figure(figsize=(8.5, 11), dpi=200)
    ax = base_axes(fig)
    hairline(ax, M, 0.945, 1 - M, 0.945)
    label(ax, M, 0.957, "CASE STUDY  ·  N. JAIN  ·  ML_PROJECTS",
          MONO, 9, color=INK_MUTE)
    label(ax, 1 - M, 0.957, "01 / 08", MONO, 9, color=INK_MUTE, ha="right")

    label(ax, M, 0.89, "DIGIT",  DISPLAY, 64, color=INK)
    label(ax, M, 0.82, "MEMORY", DISPLAY, 64, color=INK)
    label(ax, M, 0.780,
          "A study in retrieval as classification.",
          SERIF, 15, color=INK_SOFT)
    label(ax, M, 0.760,
          "Built three ways. No model trained.",
          SERIF, 15, color=INK_SOFT)

    hairline(ax, M, 0.72, 1 - M, 0.72)

    label(ax, M, 0.58, "98.4", DISPLAY, 110, color=ACCENT)
    label(ax, M + 0.28, 0.595, "%", DISPLAY_REG, 42, color=ACCENT)
    label(ax, M, 0.540, "HELD-OUT ACCURACY",
          MONO_BOLD, 11, color=INK)
    label(ax, M, 0.522,
          "1-NN on a stratified 75/25 split of the sklearn digits dataset.",
          MONO, 9, color=INK_SOFT)

    hairline(ax, M, 0.485, 1 - M, 0.485)

    label(ax, M, 0.450, "CONTENTS", MONO_BOLD, 10, color=INK)
    contents = [
        ("01", "Cover"),
        ("02", "The hypothesis"),
        ("03", "Architecture"),
        ("04", "Numbers"),
        ("05", "Dataset"),
        ("06", "Robustness"),
        ("07", "Three implementations"),
        ("08", "Colophon"),
    ]
    for i, (num, name) in enumerate(contents):
        y = 0.420 - i * 0.034
        label(ax, M, y, num, MONO, 10, color=INK_MUTE)
        label(ax, M + 0.06, y, name, BODY, 12, color=INK)
        hairline(ax, M + 0.06, y - 0.008, 1 - M, y - 0.008, lw=0.25)

    hairline(ax, M, 0.055, 1 - M, 0.055)
    label(ax, M, 0.038, "github.com/nimb-ou/ML_projects",
          MONO, 9, color=INK_MUTE)
    label(ax, 1 - M, 0.038, "EDITION 01",
          MONO, 9, color=INK_MUTE, ha="right")

    pp.savefig(fig, facecolor=BG); plt.close(fig)

    # ---- PAGE 2: hypothesis ----
    fig = plt.figure(figsize=(8.5, 11), dpi=200)
    ax = base_axes(fig)
    page_frame(fig, ax, M, "02", "The hypothesis")

    label(ax, M, 0.87, "Hypothesis", DISPLAY, 72, color=INK)
    hairline(ax, M, 0.825, 1 - M, 0.825)
    label(ax, M, 0.79, "01", MONO_BOLD, 10, color=ACCENT)

    lede = [
        "The fastest model is the one you",
        "don't need. Most ML projects start",
        "with what model to use. This one",
        "started with a different question.",
    ]
    y = 0.755
    for line in lede:
        label(ax, M, y, line, SERIF_REG, 22, color=INK)
        y -= 0.035

    hairline(ax, M, y - 0.005, 1 - M, y - 0.005)

    body = [
        "What if you just stored the data, organized it well, and looked",
        "things up? For a dataset of 1797 small samples in 64 dimensions,",
        "the answer turns out to be: that's enough.",
        "",
        "Two query patterns drive the design. An exact match — the input is",
        "byte-identical to something seen — wants a hash table. O(1) lookup.",
        "There is no smarter answer.",
        "",
        "An approximate match — the input is close but not identical — wants",
        "a spatial index. Hashing breaks: change one bit and the hash changes",
        "completely. We need a structure that knows about geometric closeness.",
        "",
        "The combined system tries the cheap lookup first and falls back to",
        "the expensive one only when needed. That is the entire idea.",
    ]
    y -= 0.040
    for line in body:
        label(ax, M, y, line, BODY, 11, color=INK)
        y -= 0.023

    pp.savefig(fig, facecolor=BG); plt.close(fig)

    # ---- PAGE 3: architecture ----
    fig = plt.figure(figsize=(8.5, 11), dpi=200)
    ax = base_axes(fig)
    page_frame(fig, ax, M, "03", "Architecture")

    label(ax, M, 0.87, "Two-tier lookup", DISPLAY, 60, color=INK)
    hairline(ax, M, 0.825, 1 - M, 0.825)
    label(ax, M, 0.79, "FIG.01  HASH FIRST · TREE ON MISS",
          MONO_BOLD, 10, color=ACCENT)

    def cs_box(x, y, w, h, top, big, sub, accent=False):
        ec = ACCENT if accent else INK
        ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False,
                                        edgecolor=ec,
                                        linewidth=0.85 if accent else 0.55))
        label(ax, x + 0.010, y + h - 0.018, top, MONO, 9, color=INK_MUTE)
        label(ax, x + 0.010, y + 0.035, big, DISPLAY, 38,
              color=ACCENT if accent else INK)
        label(ax, x + 0.010, y + 0.014, sub, MONO, 9, color=INK_SOFT)

    bw, bh = 0.20, 0.12
    qx, qy = M,         0.59
    hx, hy = M + 0.27,  0.65
    tx, ty = M + 0.27,  0.45
    ox, oy = M + 0.54,  0.59
    cs_box(qx, qy, bw, bh, "STAGE 01",  "QUERY", "64 floats")
    cs_box(hx, hy, bw, bh, "STAGE 02a", "HASH",  "O(1)", accent=True)
    cs_box(tx, ty, bw, bh, "STAGE 02b", "TREE",  "O(log n)")
    cs_box(ox, oy, bw, bh, "STAGE 03",  "LABEL", "0..9")

    arrow(ax, qx + bw,         qy + bh / 2,
              hx,                hy + bh / 2)
    arrow(ax, hx + bw,         hy + bh / 2,
              ox,                oy + bh / 2 + 0.020,
              color=ACCENT, lw=1.0)
    arrow(ax, hx + bw / 2,     hy,
              tx + bw / 2,       ty + bh)
    arrow(ax, tx + bw,         ty + bh / 2,
              ox,                oy + bh / 2 - 0.020)

    label(ax, hx + bw + 0.006, hy + bh / 2 + 0.025, "HIT",
          MONO_BOLD, 9, color=ACCENT)
    label(ax, hx + bw / 2 + 0.006, hy - 0.020, "MISS",
          MONO_BOLD, 9, color=INK_MUTE)

    hairline(ax, M, 0.36, 1 - M, 0.36)
    label(ax, M, 0.325,
          "Most production lookup systems quietly do this — caches in front",
          BODY, 12, color=INK)
    label(ax, M, 0.300,
          "of expensive indexes, content-addressed stores in front of full-text",
          BODY, 12, color=INK)
    label(ax, M, 0.275,
          "search. The same pattern applies cleanly to classification when",
          BODY, 12, color=INK)
    label(ax, M, 0.250,
          "the data fits in memory.",
          BODY, 12, color=INK)

    pp.savefig(fig, facecolor=BG); plt.close(fig)

    # ---- PAGE 4: numbers ----
    fig = plt.figure(figsize=(8.5, 11), dpi=200)
    ax = base_axes(fig)
    page_frame(fig, ax, M, "04", "Numbers")

    label(ax, M, 0.87, "Numbers", DISPLAY, 72, color=INK)
    hairline(ax, M, 0.825, 1 - M, 0.825)

    # (top, big_numeral, unit, unit_x_offset_from_M, sub, accent)
    items = [
        ("HELD-OUT ACCURACY", "98.4",  "%",  0.33,
         "stratified 75/25 split · 1-NN · no training", True),
        ("EXACT LOOKUP",       "3",    "us", 0.10,
         "Python dict, bytes(features) -> label",      False),
        ("NEAREST NEIGHBOR",   "50",   "us", 0.18,
         "BallTree query, k = 1, 1797 samples",        False),
    ]
    y = 0.79
    for top, big, unit, ux, sub, accent in items:
        col = ACCENT if accent else INK
        label(ax, M, y, top, MONO_BOLD, 10, color=col)
        label(ax, M, y - 0.150, big, DISPLAY, 120, color=col)
        label(ax, M + ux, y - 0.120, unit, DISPLAY_REG, 44, color=col)
        label(ax, M, y - 0.180, sub, MONO, 9, color=INK_SOFT)
        hairline(ax, M, y - 0.200, 1 - M, y - 0.200)
        y -= 0.225

    pp.savefig(fig, facecolor=BG); plt.close(fig)

    # ---- PAGE 5: dataset gallery ----
    fig = plt.figure(figsize=(8.5, 11), dpi=200)
    ax = base_axes(fig)
    page_frame(fig, ax, M, "05", "Dataset")

    label(ax, M, 0.87, "Dataset", DISPLAY, 72, color=INK)
    hairline(ax, M, 0.825, 1 - M, 0.825)
    label(ax, M, 0.79,
          "Ten samples per class. The variation across rows is the noise",
          SERIF, 14, color=INK_SOFT)
    label(ax, M, 0.770,
          "the memory has to tolerate.",
          SERIF, 14, color=INK_SOFT)

    digits = load_digits()
    n_per = 10
    gutter_x = 0.005
    gutter_y = 0.004
    grid_w = 1 - 2 * M
    cell_w = (grid_w - (n_per - 1) * gutter_x) / n_per
    cell_h = 0.0565
    y_top = 0.745
    for cls in range(10):
        row_y_top = y_top - cls * (cell_h + gutter_y)
        label(ax, M - 0.015, row_y_top - cell_h / 2, str(cls),
              MONO_BOLD, 11, color=INK_MUTE, ha="right", va="center")
        samples = digits.images[digits.target == cls][:n_per]
        for j in range(n_per):
            ix = M + j * (cell_w + gutter_x)
            iy = row_y_top - cell_h
            inset = fig.add_axes([ix, iy, cell_w, cell_h])
            inset.imshow(samples[j], cmap="gray_r", interpolation="nearest")
            inset.set_xticks([])
            inset.set_yticks([])
            for s in inset.spines.values():
                s.set_color(INK)
                s.set_linewidth(0.25)

    bottom_y = 0.115
    hairline(ax, M, bottom_y, 1 - M, bottom_y)
    label(ax, M, bottom_y - 0.025,
          "1,797 SAMPLES  ·  8x8 GRAYSCALE  ·  16 INTENSITY STEPS",
          MONO_BOLD, 10, color=INK)
    label(ax, M, bottom_y - 0.045,
          "Classes balanced: ~180 examples per digit.",
          MONO, 9, color=INK_SOFT)

    pp.savefig(fig, facecolor=BG); plt.close(fig)

    # ---- PAGE 6: robustness ----
    fig = plt.figure(figsize=(8.5, 11), dpi=200)
    ax = base_axes(fig)
    page_frame(fig, ax, M, "06", "Robustness")

    label(ax, M, 0.87, "Robustness", DISPLAY, 72, color=INK)
    hairline(ax, M, 0.825, 1 - M, 0.825)
    label(ax, M, 0.79,
          "Accuracy on the held-out test set with injected Gaussian noise.",
          SERIF, 14, color=INK_SOFT)

    plot_ax = fig.add_axes([M, 0.34, 1 - 2 * M, 0.40])
    plot_ax.set_facecolor(BG)
    sigmas = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
    acc    = np.array([0.984, 0.982, 0.982, 0.984, 0.973, 0.964,
                       0.938, 0.847, 0.671])
    plot_ax.plot(sigmas, acc, color=INK, linewidth=0.9, marker="o",
                 markersize=6, markeredgecolor=INK, markerfacecolor=BG,
                 markeredgewidth=0.9)
    plot_ax.plot([0], [0.984], marker="o", markersize=8,
                 markerfacecolor=ACCENT, markeredgecolor=ACCENT)
    plot_ax.set_xlim(-0.4, 10.6)
    plot_ax.set_ylim(0.55, 1.02)
    plot_ax.set_xticks([0, 2, 4, 6, 8, 10])
    plot_ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    for s in plot_ax.spines.values():
        s.set_color(INK_MUTE)
        s.set_linewidth(0.5)
    plot_ax.spines["top"].set_visible(False)
    plot_ax.spines["right"].set_visible(False)
    plot_ax.tick_params(axis="both", colors=INK_SOFT, labelsize=9, length=2)
    for tl in plot_ax.get_xticklabels() + plot_ax.get_yticklabels():
        tl.set_fontproperties(MONO)
    plot_ax.set_xlabel("NOISE SIGMA  (PIXEL UNITS)",
                       fontproperties=MONO, fontsize=9, color=INK_SOFT,
                       labelpad=8)
    plot_ax.set_ylabel("ACCURACY",
                       fontproperties=MONO, fontsize=9, color=INK_SOFT,
                       labelpad=8)
    plot_ax.grid(True, color=INK_MUTE, linewidth=0.2, alpha=0.4)

    hairline(ax, M, 0.30, 1 - M, 0.30)
    label(ax, M, 0.265,
          "sigma = 0  ->  98.4%      sigma = 5  ->  93.8%      sigma = 10  ->  67.1%",
          MONO_BOLD, 12, color=INK)
    label(ax, M, 0.240,
          "Smooth degradation, no cliff. That's what robust retrieval looks like.",
          BODY, 12, color=INK_SOFT)

    pp.savefig(fig, facecolor=BG); plt.close(fig)

    # ---- PAGE 7: three implementations ----
    fig = plt.figure(figsize=(8.5, 11), dpi=200)
    ax = base_axes(fig)
    page_frame(fig, ax, M, "07", "Implementations")

    label(ax, M, 0.87, "Three ways", DISPLAY, 72, color=INK)
    hairline(ax, M, 0.825, 1 - M, 0.825)
    label(ax, M, 0.79,
          "Writing the same system three times is the closest I have found",
          SERIF, 14, color=INK_SOFT)
    label(ax, M, 0.770,
          "to actually understanding it.",
          SERIF, 14, color=INK_SOFT)

    cards = [
        ("01", "PYTHON", "digits_memory.py",
         "Production-style. SQLite persistence. sklearn KDTree.",
         "import sqlite3, numpy, sklearn.neighbors",
         "Use when:  you want the cleanest summary."),
        ("02", "JUPYTER", "digits_study_guide.ipynb",
         "43 cells. Pandas DataFrames. Three NN backends benchmarked.",
         "%timeit  ·  pd.DataFrame  ·  plt.imshow",
         "Use when:  you want to learn the project hands-on."),
        ("03", "C", "digits_memory.c",
         "Zero dependencies. FNV-1a open-addressed hash. Brute-force NN.",
         "gcc -O2 -o digits_memory digits_memory.c -lm",
         "Use when:  you want to understand every byte."),
    ]
    y = 0.72
    for num, title, file, body, code, when in cards:
        label(ax, M, y, num, MONO, 10, color=INK_MUTE)
        label(ax, M + 0.05, y - 0.005, title, DISPLAY, 38, color=INK)
        label(ax, 1 - M, y, file, MONO_BOLD, 10, color=ACCENT, ha="right")
        label(ax, M, y - 0.040, body, BODY, 11, color=INK)
        label(ax, M, y - 0.062, code, MONO, 9, color=INK_SOFT)
        label(ax, M, y - 0.082, when, SERIF, 11, color=INK_SOFT)
        hairline(ax, M, y - 0.100, 1 - M, y - 0.100)
        y -= 0.155

    pp.savefig(fig, facecolor=BG); plt.close(fig)

    # ---- PAGE 8: colophon ----
    fig = plt.figure(figsize=(8.5, 11), dpi=200)
    ax = base_axes(fig)
    page_frame(fig, ax, M, "08", "Colophon")

    label(ax, M, 0.87, "Colophon", DISPLAY, 72, color=INK)
    hairline(ax, M, 0.825, 1 - M, 0.825)

    label(ax, M, 0.78, "DESIGN", MONO_BOLD, 10, color=INK)
    rows = [
        ("PHILOSOPHY",   "Computational Quiet"),
        ("DISPLAY",      "Big Shoulders Bold"),
        ("BODY",         "Instrument Sans"),
        ("SERIF ACCENT", "Instrument Serif Italic"),
        ("MONO",         "Geist Mono"),
        ("INK",          "#131312"),
        ("PAPER",        "#F2EFE6"),
        ("ACCENT",       "#B8331F"),
    ]
    y = 0.75
    for k, v in rows:
        label(ax, M, y, k, MONO, 10, color=INK_MUTE)
        label(ax, M + 0.30, y, v, BODY, 11, color=INK)
        hairline(ax, M, y - 0.007, 1 - M, y - 0.007, lw=0.2)
        y -= 0.026

    label(ax, M, y - 0.020, "ENGINEERING", MONO_BOLD, 10, color=INK)
    y -= 0.050
    rows2 = [
        ("DATASET",     "sklearn.datasets.load_digits  ·  1797 x 64"),
        ("EXACT MATCH", "Python dict, bytes(features) key"),
        ("NN PYTHON",   "sklearn BallTree / KDTree"),
        ("NN C",        "FNV-1a hash + brute force"),
        ("PERSISTENCE", "SQLite (.db) + CSV export"),
        ("ACCURACY",    "98.4% held-out, 1-NN"),
    ]
    for k, v in rows2:
        label(ax, M, y, k, MONO, 10, color=INK_MUTE)
        label(ax, M + 0.30, y, v, BODY, 11, color=INK)
        hairline(ax, M, y - 0.007, 1 - M, y - 0.007, lw=0.2)
        y -= 0.026

    hairline(ax, M, y - 0.020, 1 - M, y - 0.020)
    label(ax, M, y - 0.055, "ALL CODE OPEN SOURCE  ·  MIT",
          MONO_BOLD, 11, color=ACCENT)
    label(ax, M, y - 0.080, "github.com/nimb-ou/ML_projects",
          MONO, 11, color=INK)
    label(ax, M, y - 0.105, "Nimit Jain  ·  2026",
          SERIF, 13, color=INK_SOFT)

    pp.savefig(fig, facecolor=BG); plt.close(fig)
    pp.close()


if __name__ == "__main__":
    print("[1/3] architecture.png ...")
    build_architecture()
    print("[2/3] digit_memory_poster.pdf ...")
    build_poster()
    print("[3/3] digit_memory_case_study.pdf ...")
    build_case_study()
    print("Done.")
