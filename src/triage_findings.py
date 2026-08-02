"""Collapse the mined pairs into a shortlist a clinician could actually read.

`cross_category_pairs.csv` has thousands of rows, and most of them are the
same story told at different code specificity — E11.22 with N18.3, E11.22 with
N18.4, E11.65 with N18.3, and so on are one finding, not nine. Grouping by
AHRQ CCSR *category* pair collapses that, keeping the commonest code pair in
each group as the exemplar.

What survives is then ranked on the criteria that actually separate a finding
from an artefact:

* **adjusted OR** — after Mantel-Haenszel stratification on age band and sex,
  so "both are common in the very old" has already been removed;
* **confounding ratio** near 1 — age and sex explained little of the crude
  association, which makes it more likely to be about the conditions;
* **enough discharges** to have a tight interval;
* **different ICD chapters** — a cross-chapter link is more likely to be
  interesting than another circulatory-with-circulatory pair.

Run after `mine_associations.py`:  python src/triage_findings.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CACHE, RESULTS

MIN_N_BOTH = 500       # discharges carrying both codes
MAX_CONFOUNDING = 2.0  # crude OR no more than 2x the age/sex-adjusted OR


def collapse_to_ccsr(df: pd.DataFrame) -> pd.DataFrame:
    """One row per CCSR category pair, represented by its commonest code pair."""
    key = df.apply(
        lambda r: " | ".join(sorted([str(r["ccsr_desc_a"]), str(r["ccsr_desc_b"])])),
        axis=1,
    )
    df = df.assign(_key=key)
    grouped = (
        df.sort_values("n_both", ascending=False)
        .groupby("_key", as_index=False)
        .agg(
            code_a=("code_a", "first"), desc_a=("desc_a", "first"),
            code_b=("code_b", "first"), desc_b=("desc_b", "first"),
            n_both=("n_both", "first"), lift=("lift", "first"),
            odds_ratio=("odds_ratio", "first"),
            or_adjusted=("or_adjusted", "first"),
            confounding_ratio=("confounding_ratio", "first"),
            chapter_a=("chapter_a", "first"), chapter_b=("chapter_b", "first"),
            ccsr_desc_a=("ccsr_desc_a", "first"), ccsr_desc_b=("ccsr_desc_b", "first"),
            code_pairs_in_group=("code_a", "size"),
            max_lift_in_group=("lift", "max"),
        )
    )
    return grouped.drop(columns=[])


def main() -> None:
    # Prefer the dispersion-scored file when it exists: a pair produced by a
    # handful of hospitals is documentation practice, not clinical signal,
    # and nothing in the pair statistics can tell the difference.
    scored = RESULTS / "candidates_with_dispersion.csv"
    src = scored if scored.exists() else CACHE / "cross_category_pairs.csv"
    df = pd.read_csv(src)
    print(f"{len(df):,} cross-category pairs from {src.name}")

    adjusted = df.dropna(subset=["or_adjusted"]).copy()
    print(f"{len(adjusted):,} of them have an age/sex-adjusted OR")

    if "hospital_dispersed" in adjusted.columns:
        n_before = len(adjusted)
        adjusted = adjusted[adjusted["hospital_dispersed"]]
        print(f"{n_before - len(adjusted):,} dropped as hospital-concentrated "
              f"(see hospital_concentration.py)")

    strong = adjusted[
        (adjusted["n_both"] >= MIN_N_BOTH)
        & (adjusted["confounding_ratio"].between(1 / MAX_CONFOUNDING, MAX_CONFOUNDING))
    ]
    print(f"{len(strong):,} clear n>={MIN_N_BOTH} and are not explained by age/sex")

    shortlist = collapse_to_ccsr(strong)
    print(f"{len(shortlist):,} distinct CCSR category pairs after collapsing")

    shortlist = shortlist.sort_values("or_adjusted", ascending=False)
    cross_chapter = shortlist[shortlist["chapter_a"] != shortlist["chapter_b"]]

    shortlist.to_csv(RESULTS / "shortlist_all.csv", index=False)
    cross_chapter.to_csv(RESULTS / "shortlist_cross_chapter.csv", index=False)

    cols = ["code_a", "desc_a", "code_b", "desc_b", "n_both",
            "or_adjusted", "confounding_ratio", "code_pairs_in_group"]
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 46)
    print("\nTop 40 cross-chapter candidates by age/sex-adjusted odds ratio:\n")
    print(cross_chapter.head(40)[cols].to_string(index=False,
                                                 float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    main()
