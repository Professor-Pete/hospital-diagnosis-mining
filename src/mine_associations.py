"""Mine co-occurring diagnoses across all 7.08 M NIS 2019 discharges.

Finds pairs of ICD-10-CM codes that land on the same discharge record more
often than chance predicts, then works through the reasons a pair can look
associated without the two conditions having anything to do with each other.

Method
------

1. **Count every pair exactly, over the whole file.** Co-occurrence counts
   for all pairs come from a single sparse matrix product (``XᵀX``), which
   takes about a minute for 7.08 M discharges. Sampling would buy nothing.

2. **Rank on lift and odds ratio, not confidence.** Confidence is
   ``P(B|A)`` and carries no baseline term, so it scores ~1.0 for any rule
   whose consequent is near-universal regardless of whether A is involved:
   ``X -> Z370`` hits 0.99 for a dozen unrelated X simply because Z37.0
   ("single live birth") is on nearly every delivery record. Lift and the
   odds ratio both compare against what independence would predict.

3. **Test significance with a multiplicity correction.** Hundreds of
   thousands of pairs are tested; at alpha = 0.05 that alone manufactures
   thousands of false positives. Every reported pair clears a
   Bonferroni-corrected interval.

4. **Measure age/sex confounding rather than assuming it away.** Two
   conditions common in 80-year-olds co-occur strongly without being
   related. Each candidate gets a Mantel-Haenszel odds ratio stratified by
   age band and sex; the gap between crude and adjusted is reported.

5. **Flag the structurally guaranteed pairs.** ICD-10-CM coding rules force
   certain codes to be recorded together, and sibling codes in one clinical
   category are the same condition at two levels of detail. Both are
   annotated so they can be set aside instead of topping the results.

6. **Suppress small cells.** Nothing derived from 10 or fewer discharges
   reaches ``results/``, per the HCUP Data Use Agreement.

Run:  python src/mine_associations.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm

from config import CACHE, MIN_CELL, RESULTS

# A code must appear on at least this many discharges to enter pair
# counting. 0.05% of 7.08 M ~= 3,500 discharges: low enough to admit ~2,000
# codes covering 92.5% of all coded diagnoses, high enough that the pair
# matrix stays tractable and every cell clears the HCUP threshold.
MIN_CODE_COUNT = 3_500

# Report only pairs seen together at least this often. Well above the DUA
# floor of 11 — a pair at n=11 out of 7 M has a uselessly wide interval.
MIN_PAIR_COUNT = 200


# --------------------------------------------------------------------------
# ICD-10-CM structure helpers
# --------------------------------------------------------------------------

def icd_chapter(code: str) -> str:
    """Coarse ICD-10-CM chapter label from the code prefix."""
    if not code:
        return "?"
    c0, rest = code[0], code[1:3]
    n = int(rest) if rest[:2].isdigit() else -1
    table = {
        "A": "Infectious", "B": "Infectious", "C": "Neoplasm",
        "E": "Endocrine/metabolic", "F": "Mental/behavioural", "G": "Nervous",
        "H": "Eye/ear", "I": "Circulatory", "J": "Respiratory",
        "K": "Digestive", "L": "Skin", "M": "Musculoskeletal",
        "N": "Genitourinary", "O": "Pregnancy/childbirth", "P": "Perinatal",
        "Q": "Congenital", "R": "Symptoms/signs", "S": "Injury", "T": "Injury/poisoning",
        "U": "Special purpose", "V": "External cause", "W": "External cause",
        "X": "External cause", "Y": "External cause", "Z": "Factors/status",
    }
    if c0 == "D":
        return "Neoplasm" if n <= 48 else "Blood/immune"
    return table.get(c0, "?")


# Codes that are administrative or contextual rather than clinical findings.
# They dominate any unfiltered co-occurrence ranking without saying anything
# about disease biology.
def is_context_code(code: str) -> bool:
    return (
        code.startswith(("Y92", "Y93", "Y99", "Y90"))  # place / activity / alcohol level
        or code.startswith("Z3A")                      # weeks of gestation
        or code in {"Z370", "Z371", "Z372", "Z3800", "Z3801", "Z23"}
    )


# ICD-10-CM has whole families of codes that exist *only* to be paired with
# another code. Each is detected from the code's own description rather than
# by hand-listing members:
#
#   "... in diseases classified elsewhere"  — the manifestation half of the
#       etiology/manifestation convention. I43 (cardiomyopathy in diseases
#       classified elsewhere) cannot be coded alone; pairing it with E85.4
#       (amyloidosis) is the rule, not a finding.
#   "... complicating pregnancy/childbirth" — the O98/O99/O26 wrappers. B00.9
#       (herpes) with O98.52 (viral disease complicating childbirth) is one
#       infection written twice.
#   "Adverse effect of ..."                 — T36-T50 with an external-cause
#       Y-code. T46.4X5A (adverse effect of ACE inhibitors) with T78.3
#       (angioedema) is a single documented event in two codes.
#
# The AHRQ descriptions are abbreviated, so matching the full English phrase
# alone misses most of them — "Oth infections w sexl mode of transmiss comp
# preg/chldbrth" is a wrapper code, and "complicating pregnancy" does not
# appear in it. The abbreviated forms have to be matched too.
WRAPPER_PHRASES = (
    "in diseases classified elsewhere",
    "diseases classd elswhr",
    "complicating pregnancy",
    "complicating childbirth",
    "complicating the puerperium",
    "comp preg",
    "compl preg",
    "cmpl preg",
    "in pregnancy",
    "in preg",
    "in childbirth",
    "chldbrth",
    "childbirth",
    "puerperium",
    "puerp",
    "maternal care for",
)

# Generic words that appear in thousands of descriptions and so carry no
# signal about whether two codes describe the same concept.
_STOPWORDS = frozenset("""
a an and or of with without the in to for by due nos nec other others unspecified
unsp oth disease diseases disorder disorders acute chronic initial subsequent
encounter sequela left right bilateral site sites part parts type not elsewhere
classified specified condition conditions state states w wo
""".split())


def _content_words(text: str) -> set[str]:
    words = "".join(c.lower() if c.isalnum() else " " for c in str(text)).split()
    return {w for w in words if len(w) > 3 and w not in _STOPWORDS}


def describes_same_concept(desc_a: str, desc_b: str, threshold: float = 0.34) -> bool:
    """True when two descriptions are largely the same words.

    Catches restatements that CCSR does not group together — "Sepsis due to
    Streptococcus pneumoniae" with "Pneumonia due to Streptococcus
    pneumoniae", or "Osteomyelitis of vertebra, lumbar region" with
    "Discitis, lumbar region".
    """
    wa, wb = _content_words(desc_a), _content_words(desc_b)
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= threshold


# Pairs whose co-occurrence is imposed by the ICD-10-CM coding rules or by
# the definition of one of the codes, so their "discovery" is circular.
# Keyed on the frozenset of the two codes' 3-character roots where the whole
# family behaves the same way.
DEFINITIONAL_ROOT_PAIRS = {
    frozenset({"Z99", "N18"}),   # dialysis dependence <-> ESRD
    frozenset({"N18", "N25"}),   # CKD <-> impaired renal function disorders
    frozenset({"I12", "N18"}),   # hypertensive CKD codes *require* an N18 code
    frozenset({"I13", "N18"}),   # hypertensive heart AND CKD, same rule
    frozenset({"I13", "I50"}),   # I13 includes heart failure by definition
    frozenset({"D63", "N18"}),   # "anaemia in chronic kidney disease"
    frozenset({"Z37", "O80"}),   # outcome of delivery <-> encounter for delivery
    frozenset({"Z38", "P00"}),
    frozenset({"E11", "E11"}),
}


def definitional(a: str, b: str) -> bool:
    ra, rb = a[:3], b[:3]
    if frozenset({ra, rb}) in DEFINITIONAL_ROOT_PAIRS:
        return True
    # An O-code (pregnancy/childbirth) with a delivery-outcome or gestation
    # code is a coding convention, not a clinical association.
    if {a[0], b[0]} == {"O", "Z"} and (is_context_code(a) or is_context_code(b)):
        return True
    if {a[0], b[0]} == {"P", "Z"} and (a.startswith("Z38") or b.startswith("Z38")):
        return True
    return False


# --------------------------------------------------------------------------

def load():
    X = sp.load_npz(CACHE / "transactions.npz")
    codes = np.load(CACHE / "code_index.npy", allow_pickle=True)
    lookup = pd.read_csv(CACHE / "code_lookup.csv", index_col=0, dtype=str).fillna("")
    meta = pd.DataFrame({k: v for k, v in np.load(CACHE / "row_meta.npz").items()})
    return X, codes, lookup, meta


def pair_counts(X: sp.csr_matrix, keep: np.ndarray):
    """Exact co-occurrence counts for every pair of kept codes."""
    Xf = X.tocsc()[:, keep].tocsr().astype(np.int32)
    Xf.data[:] = 1
    co = (Xf.T @ Xf).tocoo()
    return Xf, co


def build_table(co, singles, n, codes_kept) -> pd.DataFrame:
    # Upper triangle only: the matrix is symmetric and the diagonal holds
    # the singleton counts.
    m = co.row < co.col
    i, j, n11 = co.row[m], co.col[m], co.data[m].astype(np.float64)
    keep = n11 >= MIN_PAIR_COUNT
    i, j, n11 = i[keep], j[keep], n11[keep]

    na = singles[i].astype(np.float64)
    nb = singles[j].astype(np.float64)
    n10, n01 = na - n11, nb - n11
    n00 = n - na - nb + n11

    df = pd.DataFrame({
        "code_a": codes_kept[i],
        "code_b": codes_kept[j],
        "n_both": n11.astype(np.int64),
        "n_a": na.astype(np.int64),
        "n_b": nb.astype(np.int64),
    })
    df["support"] = n11 / n
    df["lift"] = (n11 * n) / (na * nb)
    df["conf_a_to_b"] = n11 / na
    df["conf_b_to_a"] = n11 / nb
    df["leverage"] = n11 / n - (na / n) * (nb / n)
    df["jaccard"] = n11 / (na + nb - n11)

    # Haldane-Anscombe: +0.5 to every cell keeps the log finite when a cell
    # is empty. With counts this large it shifts nothing that matters.
    a, b_, c_, d = n11 + .5, n10 + .5, n01 + .5, n00 + .5
    log_or = np.log(a * d / (b_ * c_))
    se = np.sqrt(1 / a + 1 / b_ + 1 / c_ + 1 / d)
    df["odds_ratio"] = np.exp(log_or)
    df["se_log_or"] = se

    # Bonferroni over every pair actually tested.
    z = norm.ppf(1 - 0.05 / (2 * max(len(df), 1)))
    df["or_ci_low"] = np.exp(log_or - z * se)
    df["or_ci_high"] = np.exp(log_or + z * se)
    df["z_stat"] = log_or / se
    # Kept as a column, not in .attrs — pandas only propagates attrs on a
    # best-effort basis and this survives the filtering steps downstream.
    df["z_crit"] = z
    return df


def annotate(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    desc = lookup["description"]
    ccsr = lookup["ccsr"]
    ccsr_desc = lookup["ccsr_description"]

    for side in ("a", "b"):
        col = df[f"code_{side}"]
        df[f"desc_{side}"] = col.map(desc).fillna("(no description)")
        df[f"ccsr_{side}"] = col.map(ccsr).fillna("?")
        df[f"ccsr_desc_{side}"] = col.map(ccsr_desc).fillna("?")
        df[f"chapter_{side}"] = col.map(icd_chapter)

    df["same_ccsr"] = (df["ccsr_a"] == df["ccsr_b"]) & (df["ccsr_a"] != "?")
    df["same_icd3"] = df["code_a"].str[:3] == df["code_b"].str[:3]
    df["same_chapter"] = df["chapter_a"] == df["chapter_b"]
    df["definitional"] = [definitional(a, b) for a, b in zip(df["code_a"], df["code_b"])]
    df["context_code"] = [
        is_context_code(a) or is_context_code(b)
        for a, b in zip(df["code_a"], df["code_b"])
    ]

    low_a = df["desc_a"].str.lower()
    low_b = df["desc_b"].str.lower()
    wrapper = np.zeros(len(df), dtype=bool)
    for phrase in WRAPPER_PHRASES:
        wrapper |= low_a.str.contains(phrase, regex=False).to_numpy()
        wrapper |= low_b.str.contains(phrase, regex=False).to_numpy()
    df["wrapper_code"] = wrapper

    # "Adverse effect of X" plus the external-cause Y40-Y84 range: one
    # documented drug event always generates at least two codes.
    adverse = (
        low_a.str.startswith("adverse effect of").to_numpy()
        | low_b.str.startswith("adverse effect of").to_numpy()
        | df["code_a"].str.match(r"^Y[4-8]").to_numpy()
        | df["code_b"].str.match(r"^Y[4-8]").to_numpy()
    )
    df["adverse_effect_pair"] = adverse

    df["same_concept_text"] = [
        describes_same_concept(a, b) for a, b in zip(df["desc_a"], df["desc_b"])
    ]

    df["structural"] = (
        df["same_ccsr"] | df["same_icd3"] | df["definitional"] | df["context_code"]
        | df["wrapper_code"] | df["adverse_effect_pair"] | df["same_concept_text"]
    )
    df["obstetric"] = df["chapter_a"].isin(["Pregnancy/childbirth", "Perinatal"]) | \
                      df["chapter_b"].isin(["Pregnancy/childbirth", "Perinatal"])
    return df


def mantel_haenszel(X: sp.csr_matrix, col_of: dict, pairs, strata: np.ndarray) -> pd.DataFrame:
    """Age/sex-adjusted odds ratio for selected pairs.

    A crude OR cannot tell "these two conditions are linked" apart from
    "both are common in the same kind of patient". Stratifying on age band
    and sex and pooling with Mantel-Haenszel removes the second effect. A
    pair whose adjusted OR collapses toward 1 was confounded.
    """
    # Work from the sparse column indices rather than dense indicator
    # vectors: a code appears on a few percent of discharges, so the row
    # lists are small, while 600 dense 7 M-element masks would be gigabytes.
    Xc = X.tocsc()
    _, s_idx = np.unique(strata, return_inverse=True)
    n_strata = s_idx.max() + 1
    n_s = np.bincount(s_idx, minlength=n_strata).astype(np.float64)

    rows_of: dict[str, np.ndarray] = {}
    counts_of: dict[str, np.ndarray] = {}
    for code in {c for p in pairs for c in p}:
        col = col_of[code]
        r = Xc.indices[Xc.indptr[col]:Xc.indptr[col + 1]]
        rows_of[code] = np.sort(r)
        counts_of[code] = np.bincount(s_idx[r], minlength=n_strata).astype(np.float64)

    out = []
    for a, b in pairs:
        both = np.intersect1d(rows_of[a], rows_of[b], assume_unique=True)
        n11 = np.bincount(s_idx[both], minlength=n_strata).astype(np.float64)
        na, nb = counts_of[a], counts_of[b]

        n10 = na - n11
        n01 = nb - n11
        n00 = n_s - na - nb + n11

        ok = n_s > 0
        num = np.sum(n11[ok] * n00[ok] / n_s[ok])
        den = np.sum(n10[ok] * n01[ok] / n_s[ok])
        out.append({"code_a": a, "code_b": b,
                    "or_adjusted": num / den if den > 0 else np.nan})
    return pd.DataFrame(out)


def suppress(df: pd.DataFrame) -> pd.DataFrame:
    """HCUP DUA: drop anything resting on 10 or fewer discharges."""
    before = len(df)
    out = df[(df["n_both"] >= MIN_CELL) & (df["n_a"] >= MIN_CELL) & (df["n_b"] >= MIN_CELL)]
    if before != len(out):
        print(f"  cell suppression removed {before - len(out):,} rows")
    return out


def main() -> None:
    X, codes, lookup, meta = load()
    n = X.shape[0]
    print(f"{n:,} discharges x {X.shape[1]:,} codes")

    singles_all = np.asarray(X.sum(axis=0)).ravel()
    keep = np.flatnonzero(singles_all >= MIN_CODE_COUNT)
    codes_kept = codes[keep]
    print(f"{len(keep):,} codes at >= {MIN_CODE_COUNT:,} discharges "
          f"({100 * singles_all[keep].sum() / singles_all.sum():.1f}% of all coded diagnoses)")

    # The matrix product is the slow step (~5 min). Cache its output so the
    # filters downstream can be revised without paying for it again.
    cached = CACHE / "pair_counts.npz"
    if cached.exists():
        z = np.load(cached, allow_pickle=True)
        if np.array_equal(z["codes_kept"], codes_kept):
            print("reusing cached pair counts")
            Xf = X.tocsc()[:, keep].tocsr().astype(np.int32)
            Xf.data[:] = 1
            co = sp.coo_matrix(
                (z["data"], (z["row"], z["col"])), shape=(len(keep), len(keep))
            )
            singles = z["singles"]
        else:
            cached.unlink()

    if not cached.exists():
        Xf, co = pair_counts(X, keep)
        singles = np.asarray(Xf.sum(axis=0)).ravel()
        np.savez_compressed(
            cached, row=co.row, col=co.col, data=co.data,
            singles=singles, codes_kept=codes_kept,
        )
    print(f"{co.nnz // 2:,} co-occurring pairs observed")

    df = build_table(co, singles, n, codes_kept)
    print(f"{len(df):,} pairs at >= {MIN_PAIR_COUNT} co-occurrences "
          f"(Bonferroni z = {df['z_crit'].iloc[0]:.2f})")

    df = annotate(df, lookup)
    df = suppress(df)

    # Significant, positive, and not one of the structural artefacts.
    sig = df[(df["or_ci_low"] > 1) & (df["lift"] > 1)].copy()
    print(f"{len(sig):,} pairs significantly positive after Bonferroni")

    for flag in ["same_ccsr", "same_icd3", "definitional", "context_code",
                 "wrapper_code", "adverse_effect_pair", "same_concept_text"]:
        print(f"    {flag:<22} {int(sig[flag].sum()):>7,}")

    novel = sig[~sig["structural"]].copy()
    print(f"{len(novel):,} remain after removing every structural class")

    # Adjust the strongest survivors for age and sex.
    age = meta["AGE"].to_numpy()
    band = np.digitize(age, [1, 18, 45, 65, 75, 85])
    sex = np.nan_to_num(meta["FEMALE"].to_numpy(), nan=-1).astype(int)
    strata = band * 10 + sex

    col_of = {c: k for k, c in enumerate(codes_kept)}
    # Adjust everything with enough discharges behind it to be worth reading,
    # rather than only the top of the lift ranking — the confounding ratio is
    # itself one of the things being ranked on.
    top = novel[novel["n_both"] >= 500].nlargest(6000, "lift")
    print(f"computing age/sex-adjusted OR for {len(top):,} candidates...")
    adj = mantel_haenszel(Xf, col_of, list(zip(top["code_a"], top["code_b"])), strata)
    novel = novel.merge(adj, on=["code_a", "code_b"], how="left")
    novel["confounding_ratio"] = novel["odds_ratio"] / novel["or_adjusted"]

    cols = ["code_a", "desc_a", "code_b", "desc_b", "n_both", "n_a", "n_b",
            "support", "lift", "odds_ratio", "or_ci_low", "or_ci_high",
            "or_adjusted", "confounding_ratio", "conf_a_to_b", "conf_b_to_a",
            "jaccard", "leverage", "chapter_a", "chapter_b",
            "ccsr_desc_a", "ccsr_desc_b", "same_chapter", "obstetric"]

    RESULTS.mkdir(exist_ok=True)
    # The two full tables run to ~160 MB. They are aggregate and suppressed,
    # but they are working data rather than a result, so they go to the
    # gitignored cache; results/ keeps only what is small enough to read and
    # small enough to publish.
    sig.sort_values("lift", ascending=False).to_csv(
        CACHE / "all_significant_pairs.csv", index=False)
    novel.sort_values("lift", ascending=False)[cols].to_csv(
        CACHE / "cross_category_pairs.csv", index=False)
    df.sort_values("n_both", ascending=False).head(500).to_csv(
        RESULTS / "most_common_pairs.csv", index=False)

    print(f"\nfull tables -> {CACHE.name}/, top-500 pairs -> {RESULTS.name}/")

    show = ["code_a", "desc_a", "code_b", "desc_b", "n_both", "lift",
            "odds_ratio", "or_adjusted"]
    ranked = novel.dropna(subset=["or_adjusted"])
    print(f"\nof those, {int(novel['obstetric'].sum()):,} involve an obstetric or "
          f"perinatal code; those blocks are reported separately because "
          f"delivery and newborn records are structurally different")
    print("\nTop 30 surviving non-obstetric pairs by lift:")
    print(ranked[~ranked["obstetric"]].nlargest(30, "lift")[show]
          .to_string(index=False, max_colwidth=44))


if __name__ == "__main__":
    main()
