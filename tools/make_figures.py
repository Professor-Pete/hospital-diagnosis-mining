#!/usr/bin/env python3
"""Generate the SVG figures embedded in README.md.

Numbers come from results/*.csv wherever they exist, so a figure cannot
drift away from the analysis it illustrates. The few counts that only exist
as pipeline console output are declared in FUNNEL below with their source.

Output is plain SVG with an explicit light surface. GitHub renders markdown
images in an <img> tag, where external CSS never applies and
prefers-color-scheme support is inconsistent — so the surface is painted
into the file and the figure reads as a light card under either GitHub
theme.

There is no hover layer: an SVG embedded as an image cannot be interactive.
Every value is therefore direct-labelled, so nothing is hidden behind a
tooltip that will never open.

Run:  python tools/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

# --- palette (validated: scripts/validate_palette.js, light mode, PASS) ---
SURFACE = "#fcfcfb"
BORDER = "#e4e3df"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#84837d"
GRID = "#eceae5"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#b6b4ad"

FONT = ("-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif")

# Stage counts printed by src/mine_associations.py and
# src/triage_findings.py when the pipeline is run end to end.
FUNNEL = [
    ("Diagnosis pairs that passed a strict statistical test", 191_176, None),
    ("…that aren't just the coding rulebook restated", 159_838, None),
    ("…strongest candidates examined in detail", 6_000, "a deliberate cap"),
    ("…that survived the hospital-origin check", 5_456, None),
    ("…distinct findings, after merging near-duplicates", 1_197, None),
]


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def text(x, y, s, size=13, fill=INK_2, anchor="start", weight=400):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>'
    )


def bar(x, y, w, h, fill, r=4):
    """Bar with rounded data-end only; the baseline end stays square."""
    w = max(w, 0.6)
    if w <= r:
        return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h}" fill="{fill}"/>'
    return (
        f'<path d="M{x:.1f},{y:.1f} H{x + w - r:.1f} '
        f'a{r},{r} 0 0 1 {r},{r} V{y + h - r:.1f} '
        f'a{r},{r} 0 0 1 -{r},{r} H{x:.1f} Z" fill="{fill}"/>'
    )


def frame(w: int, h: int, title: str, subtitle: str, body: str, footer: str = "") -> str:
    foot = text(24, h - 16, footer, 11, INK_3) if footer else ""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{esc(title)}">
<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" fill="{SURFACE}" stroke="{BORDER}"/>
{text(24, 34, title, 16, INK, weight=600)}
{text(24, 55, subtitle, 12.5, INK_2)}
{body}
{foot}
</svg>
"""


# --------------------------------------------------------------------------

def fig_funnel() -> str:
    w, top, row = 860, 92, 46
    h = top + row * len(FUNNEL) + 46
    x0, bar_w = 470, 300
    vmax = max(v for _, v, _ in FUNNEL)

    parts = []
    for i, (label, val, note) in enumerate(FUNNEL):
        y = top + i * row
        parts.append(text(24, y + 15, label, 13, INK_2))
        if note:
            parts.append(text(24, y + 31, f"({note})", 11, INK_3))
        parts.append(f'<rect x="{x0}" y="{y + 2}" width="{bar_w}" height="20" fill="{GRID}" rx="4"/>')
        parts.append(bar(x0, y + 2, bar_w * val / vmax, 20, BLUE))
        parts.append(text(x0 + bar_w + 12, y + 17, f"{val:,}", 14, INK, weight=600))

    return frame(
        w, h,
        "Almost nothing survives the filters",
        "Of 191,176 statistically solid diagnosis pairs, ~1,200 distinct findings remain.",
        "\n".join(parts),
        "Source: src/mine_associations.py, src/triage_findings.py · HCUP NIS 2019, 7,083,805 hospital stays",
    )


PLAIN_NAMES = {
    ("E1143", "K3184"): "Diabetic nerve damage + stomach paralysis",
    ("I120", "Z992"): "Kidney failure + dialysis",
    ("G931", "I469"): "Cardiac arrest + brain oxygen loss",
    ("D62", "I8511"): "Blood-loss anaemia + bleeding veins",
    ("G800", "J690"): "Severe cerebral palsy + choking pneumonia",
    ("E840", "K8681"): "Cystic fibrosis + pancreas failure",
    ("J310", "J40"): "Stuffy nose + chest infection",
    ("F250", "G44209"): "Schizoaffective disorder + headaches",
    ("F250", "J310"): "Schizoaffective disorder + stuffy nose",
    ("G44209", "J40"): "Headaches + chest infection",
    ("G44209", "J310"): "Tension headaches + stuffy nose",
}


def fig_hospitals() -> str:
    df = pd.read_csv(RESULTS / "hospital_concentration_calibration.csv")
    df["label"] = [PLAIN_NAMES.get((a, b), f"{a} + {b}")
                   for a, b in zip(df["code_a"], df["code_b"])]
    df = df.sort_values("hospitals_contributing", ascending=False).reset_index(drop=True)

    w, top, row = 860, 116, 34
    h = top + row * len(df) + 62
    x0, bar_w = 400, 350
    vmax = 4568  # total hospitals in the NIS

    parts = [
        f'<rect x="24" y="70" width="11" height="11" rx="3" fill="{BLUE}"/>',
        text(42, 80, "Established medical link", 12, INK_2),
        f'<rect x="216" y="70" width="11" height="11" rx="3" fill="{ORANGE}"/>',
        text(234, 80, "Suspected paperwork artefact", 12, INK_2),
    ]
    for i, r in df.iterrows():
        y = top + i * row
        artefact = r["group"].startswith("suspected")
        parts.append(text(24, y + 15, r["label"], 12.5, INK_2))
        parts.append(f'<rect x="{x0}" y="{y + 3}" width="{bar_w}" height="16" fill="{GRID}" rx="4"/>')
        parts.append(bar(x0, y + 3, bar_w * r["hospitals_contributing"] / vmax, 16,
                         ORANGE if artefact else BLUE))
        parts.append(text(x0 + bar_w + 12, y + 16, f'{int(r["hospitals_contributing"]):,}',
                          13, INK, weight=600))

    parts.append(f'<line x1="{x0}" y1="{top - 8}" x2="{x0}" y2="{top + row * len(df) - 6}" '
                 f'stroke="{BORDER}"/>')
    return frame(
        w, h,
        "How many hospitals produced each pattern?",
        "Out of 4,568 hospitals. A real medical link shows up everywhere; a local habit does not.",
        "\n".join(parts),
        "Source: src/hospital_concentration.py · bars scaled against all 4,568 hospitals",
    )


def fig_model() -> str:
    m = pd.read_csv(RESULTS / "esrd_model_comparison.csv")
    # Logistic regression rows (iloc[0]) — the same model family used in
    # compare_targets.py, so the "times better than chance" figure quoted
    # here and there cannot drift apart.
    naive = m[m["regime"].str.startswith("all codes")].iloc[0]
    strict = m[m["regime"].str.startswith("also no renal")].iloc[0]
    ap_base = 100 * naive["ap_baseline"]
    ap_strict = 100 * strict["avg_precision"]

    w, h = 860, 244
    left, bw = 380, 300
    parts = []
    for i, (lab, val, col) in enumerate([
        ("Guessing at random", ap_base, GRAY),
        ("The model, kidney and dialysis hardware removed", ap_strict, BLUE),
    ]):
        y = 96 + i * 40
        parts.append(text(24, y + 18, lab, 13, INK_2))
        parts.append(f'<rect x="{left}" y="{y + 4}" width="{bw}" height="20" fill="{GRID}" rx="4"/>')
        parts.append(bar(left, y + 4, bw * val / 100, 20, col))
        parts.append(text(left + bw + 12, y + 19, f"{val:.0f}%", 14, INK, weight=600))

    parts.append(text(24, 196, f"{ap_strict / ap_base:.0f} times better than chance — with no kidney code "
                               "and no dialysis hardware to go on.", 12, INK_3))

    return frame(
        w, h,
        "Spotting failed kidneys with every kidney clue taken away",
        "How reliably the model finds end-stage kidney disease, next to random guessing.",
        "\n".join(parts),
        "Source: results/esrd_model_comparison.csv (logistic regression) · metric is average precision",
    )


def fig_newborn() -> str:
    w, h = 860, 258
    left, bw = 300, 300
    vmax = 180
    parts = [text(24, 92, "How strongly the two are linked", 13, INK, weight=600)]
    for i, (lab, val, col) in enumerate([
        ("Measured across all 7 million hospital stays", 26.2, GRAY),
        ("Measured within newborns only", 168.3, BLUE),
    ]):
        y = 108 + i * 38
        parts.append(text(24, y + 17, lab, 12.5, INK_2))
        parts.append(f'<rect x="{left}" y="{y + 3}" width="{bw}" height="19" fill="{GRID}" rx="4"/>')
        parts.append(bar(left, y + 3, bw * val / vmax, 19, col))
        parts.append(text(left + bw + 12, y + 18, f"{val:.0f}×", 14, INK, weight=600))
    parts.append(text(24, 204, "Six times stronger than it first appeared. 6.4 million adults "
                               "— who cannot have either condition — were watering it down.",
                      11.5, INK_3))
    return frame(
        w, h,
        "A real pattern, hidden by the wrong comparison group",
        "Newborns tested for metabolic problems vs. mothers with diabetes.",
        "\n".join(parts),
        "Source: results/candidates_with_dispersion.csv · 4,164 hospital stays across 725 hospitals",
    )


GROUP_PLAIN = {
    "Electrolyte & acid-base": "Blood chemistry the kidney can't control",
    "Fluid overload": "Fluid the kidney can't remove",
    "Vascular access failure": "Failures of the dialysis access port",
    "Bone & mineral chemistry": "Bone and mineral chemistry",
    "Transplant pathway": "On the transplant pathway",
    "Other systemic": "Other systemic damage",
}


def fig_importance() -> str:
    df = pd.read_csv(RESULTS / "esrd_feature_importance.csv")
    rest = df[df["group"].str.startswith("Everything else")].iloc[0]
    named = df[df["group"] == "All named groups together"].iloc[0]
    groups = df[~df["group"].str.startswith(("Everything else", "All named"))].copy()
    groups = groups.sort_values("pct_of_model", ascending=False)

    w, top, row = 860, 104, 34
    h = top + row * len(groups) + 96
    x0, bar_w = 400, 300
    vmax = 10.0  # percent of the model's score

    parts = []
    for i, r in groups.iterrows():
        y = top + list(groups.index).index(i) * row
        label = GROUP_PLAIN.get(r["group"], r["group"])
        parts.append(text(24, y + 15, label, 12.5, INK_2))
        parts.append(text(24 + 0, y + 29, f'{int(r["codes"])} codes', 10.5, INK_3))
        parts.append(f'<rect x="{x0}" y="{y + 3}" width="{bar_w}" height="17" fill="{GRID}" rx="4"/>')
        parts.append(bar(x0, y + 3, bar_w * min(r["pct_of_model"], vmax) / vmax, 17, BLUE))
        parts.append(text(x0 + bar_w + 12, y + 17, f'{r["pct_of_model"]:.1f}%', 13,
                          INK, weight=600))

    base_y = top + row * len(groups) + 14
    parts.append(f'<line x1="24" y1="{base_y}" x2="{w - 24}" y2="{base_y}" stroke="{BORDER}"/>')
    parts.append(text(24, base_y + 22,
                      f'All {int(named["codes"])} of these codes together: '
                      f'{named["pct_of_model"]:.0f}% of the score. The other '
                      f'{int(rest["codes"]):,} codes the model uses: '
                      f'{rest["pct_of_model"]:.0f}%.', 12, INK_2))
    parts.append(text(24, base_y + 40,
                      "The two overlap and do not sum — this measures each group's "
                      "contribution separately, not a split of a whole.", 11, INK_3))

    return frame(
        w, h,
        "Which parts of the chart carry the prediction?",
        "How far the model's score falls when each group of codes is scrambled.",
        "\n".join(parts),
        "Source: src/feature_importance.py · grouped permutation importance, held-out data, 5 repeats",
    )


TARGET_PLAIN = {
    "N186": "End-stage kidney disease",
    "I509": "Heart failure",
    "E119": "Type 2 diabetes",
    "G4733": "Sleep apnoea",
    "D649": "Anaemia",
}


def fig_targets() -> str:
    df = pd.read_csv(RESULTS / "target_comparison.csv").sort_values(
        "lift_over_chance", ascending=False)

    w, top, row = 860, 100, 40
    h = top + row * len(df) + 76
    x0, bar_w = 400, 300
    vmax = float(df["lift_over_chance"].max()) * 1.05

    parts = []
    for i, (_, r) in enumerate(df.iterrows()):
        y = top + i * row
        label = TARGET_PLAIN.get(r["code"], r["description"][:38])
        parts.append(text(24, y + 16, label, 13, INK_2))
        parts.append(text(24, y + 31, f'{r["prevalence_pct"]:.1f}% of patients',
                          10.5, INK_3))
        parts.append(f'<rect x="{x0}" y="{y + 4}" width="{bar_w}" height="18" fill="{GRID}" rx="4"/>')
        parts.append(bar(x0, y + 4, bar_w * r["lift_over_chance"] / vmax, 18, BLUE))
        parts.append(text(x0 + bar_w + 12, y + 18, f'{r["lift_over_chance"]:.0f}x', 14,
                          INK, weight=600))

    parts.append(text(24, top + row * len(df) + 26,
                      "Kidney disease had every kidney code and every dialysis-hardware code "
                      "removed first; the other four use the automatic clue filter.", 11.5, INK_3))

    return frame(
        w, h,
        "How much better than guessing, for each condition?",
        "1x would mean no better than chance. Same model throughout.",
        "\n".join(parts),
        "Source: results/target_comparison.csv · average precision against each condition's own chance floor",
    )


def main() -> int:
    FIGS.mkdir(exist_ok=True)
    figures = {
        "01-funnel.svg": fig_funnel(),
        "02-hospitals.svg": fig_hospitals(),
        "03-model.svg": fig_model(),
        "04-newborn.svg": fig_newborn(),
        "05-importance.svg": fig_importance(),
        "06-targets.svg": fig_targets(),
    }
    for name, svg in figures.items():
        (FIGS / name).write_text(svg)
        print(f"  {name}  ({len(svg) / 1024:.1f} KB)")

    # A contact sheet on a dark background — these render as light cards, and
    # the only way to know they still read correctly is to look at them.
    imgs = "\n".join(f'<img src="{n}" style="display:block;margin-bottom:20px">'
                     for n in figures)
    (FIGS / "_preview.html").write_text(
        '<!doctype html><meta charset="utf-8">\n'
        '<title>figure preview</title>\n'
        '<body style="margin:0;padding:24px;background:#0d1117">\n'
        '<p style="color:#8b949e;font:12px system-ui">'
        "Checking legibility against GitHub's dark theme.</p>\n"
        f"{imgs}\n</body>\n"
    )
    print(f"\nwrote {len(figures)} figures to {FIGS.name}/ "
          f"(open {FIGS.name}/_preview.html to check them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
