"""Is a co-occurrence clinical, or is it one hospital's coding habit?

A real disease relationship shows up everywhere. An artefact of documentation
practice — a facility that imports the whole outpatient problem list onto the
inpatient record, say — shows up in a handful of hospitals and nowhere else,
and produces exactly the same large odds ratio.

Nothing in the pair statistics distinguishes those two cases, which is why
this test exists. For a given code pair it asks: of the ~4,500 hospitals in
the NIS, how many contribute any of the co-occurrences, and how much of the
total comes from the ten biggest contributors relative to their share of
discharges?

`concentration` is the ratio of those two shares. Around 1 means the pair is
spread across hospitals in proportion to their size — what a clinical
relationship looks like. Much above 1 means a few facilities are producing
most of it.

Only aggregate statistics are reported. No per-hospital figure is written
anywhere, and pairs below the HCUP cell threshold are dropped.

Run:  python src/hospital_concentration.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyreadstat
import scipy.sparse as sp

from config import CACHE, MIN_CELL, RESULTS, SAV

TOP_K = 10


def hospital_ids() -> np.ndarray:
    """HOSP_NIS for every discharge, cached as a compact integer code."""
    cached = CACHE / "hosp_id.npy"
    if cached.exists():
        return np.load(cached)

    parts = []
    offset = 0
    while True:
        df, _ = pyreadstat.read_sav(
            str(SAV), usecols=["HOSP_NIS"], row_offset=offset, row_limit=1_000_000
        )
        if len(df) == 0:
            break
        parts.append(df["HOSP_NIS"].to_numpy())
        offset += len(df)
        if len(df) < 1_000_000:
            break

    raw = np.concatenate(parts)
    _, ids = np.unique(raw, return_inverse=True)
    ids = ids.astype(np.int32)
    np.save(cached, ids)
    return ids


def concentration(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    X = sp.load_npz(CACHE / "transactions.npz").tocsc()
    codes = np.load(CACHE / "code_index.npy", allow_pickle=True)
    lookup = pd.read_csv(CACHE / "code_lookup.csv", index_col=0, dtype=str).fillna("")
    col_of = {c: i for i, c in enumerate(codes)}

    hosp = hospital_ids()
    n_hosp = int(hosp.max()) + 1
    disch_per_hosp = np.bincount(hosp, minlength=n_hosp).astype(np.float64)
    total_disch = disch_per_hosp.sum()

    rows_of = {}
    for code in {c for p in pairs for c in p}:
        col = col_of[code]
        rows_of[code] = X.indices[X.indptr[col]:X.indptr[col + 1]]

    out = []
    for a, b in pairs:
        both = np.intersect1d(rows_of[a], rows_of[b], assume_unique=False)
        if len(both) < MIN_CELL:
            continue
        per_hosp = np.bincount(hosp[both], minlength=n_hosp).astype(np.float64)
        contributing = int((per_hosp > 0).sum())

        top = np.argsort(per_hosp)[::-1][:TOP_K]
        share_of_pairs = per_hosp[top].sum() / per_hosp.sum()
        share_of_discharges = disch_per_hosp[top].sum() / total_disch

        out.append({
            "code_a": a, "desc_a": lookup["description"].get(a, "?"),
            "code_b": b, "desc_b": lookup["description"].get(b, "?"),
            "n_both": int(len(both)),
            "hospitals_contributing": contributing,
            "pct_of_all_hospitals": 100 * contributing / n_hosp,
            f"top{TOP_K}_share_of_pairs": share_of_pairs,
            f"top{TOP_K}_share_of_discharges": share_of_discharges,
            "concentration": share_of_pairs / share_of_discharges,
        })
    return pd.DataFrame(out)


# Pairs whose clinical basis is not in question — cystic fibrosis causing
# pancreatic insufficiency, diabetic autonomic neuropathy causing
# gastroparesis, ESRD, aspiration in severe cerebral palsy. If the test is
# calibrated, these come out near 1.
CONTROLS = [
    ("E840", "K8681"),
    ("E1143", "K3184"),
    ("I120", "Z992"),
    ("G800", "J690"),
    ("D62", "I8511"),
    ("G931", "I469"),
]

# The cluster that prompted the test: chronic, low-acuity "problem list"
# codes co-occurring at odds ratios of 20-260 with nothing clinical linking
# them.
SUSPECTS = [
    ("G44209", "J310"),
    ("G44209", "J40"),
    ("J310", "J40"),
    ("F250", "J310"),
    ("F250", "G44209"),
]


# A pair spread over fewer than this many hospitals, or this much more
# concentrated than hospital size alone predicts, is treated as documentation
# practice rather than clinical signal. Thresholds sit between the two groups
# below, which are separated by an order of magnitude.
MIN_HOSPITALS = 150
MAX_CONCENTRATION = 25.0


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 40)

    n_hosp = len(np.unique(hospital_ids()))
    print(f"{n_hosp:,} hospitals in the 2019 NIS\n")

    ctrl = concentration(CONTROLS).assign(group="established clinical link")
    susp = concentration(SUSPECTS).assign(group="suspected coding artefact")
    calib = pd.concat([ctrl, susp], ignore_index=True)
    calib.to_csv(RESULTS / "hospital_concentration_calibration.csv", index=False)

    cols = ["group", "code_a", "code_b", "desc_a", "desc_b", "n_both",
            "hospitals_contributing", "pct_of_all_hospitals", "concentration"]
    print("Calibration — six relationships nobody disputes, five suspected artefacts:\n")
    print(calib[cols].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # Now score every candidate that survived the structural filters.
    src = CACHE / "cross_category_pairs.csv"
    cand = pd.read_csv(src).dropna(subset=["or_adjusted"])
    cand = cand[cand["n_both"] >= 500]
    print(f"\n\nscoring all {len(cand):,} adjusted candidates from {src.name}...")

    scores = concentration(list(zip(cand["code_a"], cand["code_b"])))
    merged = cand.merge(
        scores.drop(columns=["desc_a", "desc_b", "n_both"]),
        on=["code_a", "code_b"], how="inner",
    )
    merged["hospital_dispersed"] = (
        (merged["hospitals_contributing"] >= MIN_HOSPITALS)
        & (merged["concentration"] <= MAX_CONCENTRATION)
    )

    n_bad = int((~merged["hospital_dispersed"]).sum())
    print(f"{n_bad:,} of {len(merged):,} ({100 * n_bad / len(merged):.1f}%) fail the "
          f"dispersion test — concentrated in <{MIN_HOSPITALS} hospitals "
          f"or >{MAX_CONCENTRATION:g}x their share of discharges")

    top100 = merged.nlargest(100, "lift")
    print(f"among the 100 highest-lift candidates, "
          f"{int((~top100['hospital_dispersed']).sum())} fail it")

    merged.sort_values("or_adjusted", ascending=False).to_csv(
        RESULTS / "candidates_with_dispersion.csv", index=False)
    print(f"\nwrote {(RESULTS / 'candidates_with_dispersion.csv').name}")


if __name__ == "__main__":
    main()
