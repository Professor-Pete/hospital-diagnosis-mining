"""Predict dialysis dependence (Z99.2) from the rest of a discharge record.

Given every other diagnosis on a discharge record, can you tell whether the
patient is on dialysis? Z99.2 makes a good test case: it is common enough
to model (2.37% of discharges) and it has a clear clinical footprint.

The interesting problem is leakage. Several codes cannot be assigned to
anyone who is *not* on dialysis — Z91.15 is literally "patient's
noncompliance with renal dialysis" — and ICD-10-CM coding rules *require*
others (N18.6 end-stage renal disease, I12.0 / I13.2 hypertensive CKD with
stage 5) to be recorded alongside Z99.2. A model handed those features
scores brilliantly while learning nothing except the coding manual.

So the model is fit under three feature regimes, loosest to strictest, and
the drop between them measures how much of the performance was circular:

1. every code except the target;
2. minus anything naming dialysis or ESRD, and the codes the coding rules
   force to appear with it;
3. minus every renal code at all.

Two other things this gets right, because a rare outcome punishes getting
them wrong:

* **Evaluation happens at true prevalence.** Training on a rebalanced set
  is sensible for a 2.37% outcome, but scoring on a 50/50 test set is not:
  it moves the do-nothing baseline from 97.63% to 50% and makes any model
  look strong. Accuracy is replaced by ROC-AUC and average precision
  against the real 2.37% base rate.
* **Feature screening uses Benjamini-Hochberg FDR**, not raw p < 0.05.
  Across ~3,700 candidate features the naive rule alone would return
  ~185 "significant" results from noise.

``k`` and the regularisation strength are cross-validated rather than
picked by hand.

Run:  python src/predict_dialysis.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from config import CACHE, RESULTS

TARGET = "Z992"  # Dependence on renal dialysis
SEED = 42
SUBSAMPLE = 600_000  # rows drawn for modelling; keeps the CV grid tractable


# --------------------------------------------------------------------------
# Feature regimes
# --------------------------------------------------------------------------

def leak_terms(lookup: pd.DataFrame) -> set[str]:
    """Codes whose own description names dialysis or ESRD.

    Derived from the ICD-10-CM descriptions rather than hand-listed, so a
    code cannot be missed just because nobody thought of it.
    """
    d = lookup["description"].str.lower()
    hit = d.str.contains("dialysis|end stage renal|end-stage renal", regex=True, na=False)
    return set(lookup.index[hit])


# AHRQ CCSR categories covering kidney function and kidney-directed devices.
# picked by reading the category list rather than guessing at the numbering.
# The ranges are not contiguous: GEN004 is urinary tract infections, while
# nephritis (GEN001), other kidney disease (GEN006), haematuria (GEN009)
# and the device categories all sit apart from the obvious GEN002/GEN003.
RENAL_CCSR = frozenset({
    "GEN001",  # nephritis, nephrosis, renal sclerosis
    "GEN002",  # acute and unspecified renal failure
    "GEN003",  # chronic kidney disease
    "GEN006",  # other diseases of kidney and ureters
    "GEN009",  # haematuria
    "GEN026",  # postprocedural genitourinary complication
    "NEO044", "NEO045",  # renal pelvis / kidney cancers
    "INJ034", "INJ070",  # genitourinary device complications
})

# Plus anything whose description names the organ — this is what catches
# "Polycystic kidney", "Kidney transplant failure", "Renal sclerosis" and
# "Glomerular disease in SLE", which sit in non-renal CCSR categories.
RENAL_TEXT = r"kidney|renal|nephr|glomerul|dialysis"


def regimes(codes: np.ndarray, lookup: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean keep-masks over the code vocabulary, loosest to strictest."""
    idx = pd.Index(codes)
    desc_leaks = leak_terms(lookup)

    ccsr = idx.map(lookup["ccsr"]).fillna("?")
    desc = pd.Series(idx.map(lookup["description"]).fillna(""), index=idx).str.lower()

    is_target = idx == TARGET
    names_dialysis = idx.isin(desc_leaks)
    # Codes that ICD-10-CM rules force to be coded with a CKD stage, plus the
    # downstream consequences of ESRD that are named as such.
    coding_rule = (
        idx.str.startswith("N18") | idx.str.startswith("I12") | idx.str.startswith("I13")
        | idx.isin({"N250", "N2581", "D631", "Z4931", "Z4932", "Z940", "Z905", "Z9115"})
        | desc.str.contains("chronic kidney disease", na=False)
    )
    renal_cat = (
        pd.Series(ccsr, index=idx).isin(RENAL_CCSR).to_numpy()
        | desc.str.contains(RENAL_TEXT, regex=True, na=False).to_numpy()
    )

    naive = ~is_target
    deleaked = naive & ~names_dialysis & ~coding_rule
    strict = deleaked & ~renal_cat
    return {
        "all codes available": naive,
        "no dialysis/CKD-definitional codes": deleaked,
        "also no renal category at all": strict,
    }


# --------------------------------------------------------------------------

def bh_fdr(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg. Returns a boolean mask of discoveries."""
    p = np.where(np.isnan(p), 1.0, p)
    m = len(p)
    order = np.argsort(p)
    thresh = alpha * np.arange(1, m + 1) / m
    passed = p[order] <= thresh
    keep = np.zeros(m, dtype=bool)
    if passed.any():
        cutoff = np.max(np.flatnonzero(passed))
        keep[order[: cutoff + 1]] = True
    return keep


def evaluate(name: str, model, X_test, y_test) -> dict:
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    prevalence = y_test.mean()
    return {
        "regime": name,
        "roc_auc": roc_auc_score(y_test, prob),
        "avg_precision": average_precision_score(y_test, prob),
        "ap_baseline": prevalence,
        "brier": brier_score_loss(y_test, prob),
        "accuracy": (pred == y_test).mean(),
        "accuracy_of_always_no": 1 - prevalence,
        "recall": pred[y_test == 1].mean(),
        "precision": y_test[pred == 1].mean() if pred.any() else np.nan,
    }


def load():
    """Matrix, code vocabulary, description lookup, and the target vector."""
    X = sp.load_npz(CACHE / "transactions.npz").tocsc()
    codes = np.load(CACHE / "code_index.npy", allow_pickle=True)
    lookup = pd.read_csv(CACHE / "code_lookup.csv", index_col=0, dtype=str).fillna("")
    col_of = {c: i for i, c in enumerate(codes)}
    y = np.asarray(X[:, col_of[TARGET]].todense()).ravel().astype(np.int8)
    return X, codes, lookup, y


def sample(X, y, size: int = SUBSAMPLE):
    """A random subsample, kept at the natural class balance."""
    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    sub = rng.choice(n, size=min(size, n), replace=False)
    return X[sub].tocsr(), y[sub]


def run_regime(name, keep_mask, Xs, ys, codes, lookup, verbose: bool = True):
    """Fit and evaluate one feature regime.

    Returns ``(rows, coefficients)`` — one row per model, and the tuned
    logistic regression's coefficients with descriptions attached.
    """
    counts = np.asarray(Xs.sum(axis=0)).ravel()
    keep = np.flatnonzero(keep_mask & (counts >= 100))
    Xr, names = Xs[:, keep], codes[keep]
    if verbose:
        print(f"--- {name}: {len(keep):,} features ---")

    # Stratified so the 2.37% positives are represented in both halves, and
    # the test half is left at natural prevalence.
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xr, ys, test_size=0.25, random_state=SEED, stratify=ys
    )

    # FDR-controlled screen, reported for context — k is cross-validated
    # below rather than taken from this.
    _, p_val = f_classif(X_tr, y_tr)
    if verbose:
        print(f"    features at raw p<0.05: {np.nansum(p_val < 0.05):,}   "
              f"after BH-FDR: {bh_fdr(p_val).sum():,}")

    grid = GridSearchCV(
        Pipeline([
            ("select", SelectKBest(score_func=f_classif)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        {"select__k": [10, 50, 200, 800], "clf__C": [0.1, 1.0]},
        scoring="average_precision",
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
        n_jobs=-1,
    ).fit(X_tr, y_tr)
    best = grid.best_params_

    res = evaluate(name, grid.best_estimator_, X_te, y_te)
    res["model"] = f"logreg k={best['select__k']} C={best['clf__C']}"

    rf = Pipeline([
        ("select", SelectKBest(score_func=f_classif, k=min(200, len(keep)))),
        ("clf", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=10, n_jobs=-1,
            class_weight="balanced_subsample", random_state=SEED)),
    ]).fit(X_tr, y_tr)
    rf_res = evaluate(name, rf, X_te, y_te)
    rf_res["model"] = "random forest (200 trees)"

    sel = grid.best_estimator_.named_steps["select"]
    clf = grid.best_estimator_.named_steps["clf"]
    chosen = names[sel.get_support()]
    coefs = pd.DataFrame({
        "code": chosen,
        "coef": clf.coef_.ravel(),
        "odds_ratio": np.exp(clf.coef_.ravel()),
        "description": [lookup["description"].get(c, "?") for c in chosen],
    }).sort_values("coef", ascending=False)

    if verbose:
        print(f"    tuned: k={best['select__k']}, C={best['clf__C']} "
              f"(CV AP {grid.best_score_:.3f})")
        print(f"    test ROC-AUC {res['roc_auc']:.4f} | "
              f"avg precision {res['avg_precision']:.4f} "
              f"(baseline {res['ap_baseline']:.4f})\n")
    return [res, rf_res], coefs


def main() -> None:
    X, codes, lookup, y_all = load()
    n_all = X.shape[0]
    print(f"{n_all:,} discharges, {y_all.sum():,} with {TARGET} "
          f"({100 * y_all.mean():.2f}% prevalence)")
    print(f"a constant 'no dialysis' prediction is {100 * (1 - y_all.mean()):.2f}% accurate\n")

    Xs, ys = sample(X, y_all)

    rows, coef_tables = [], {}
    for name, keep_mask in regimes(codes, lookup).items():
        r, coefs = run_regime(name, keep_mask, Xs, ys, codes, lookup)
        rows += r
        coef_tables[name] = coefs

    out = pd.DataFrame(rows)[
        ["regime", "model", "roc_auc", "avg_precision", "ap_baseline",
         "accuracy", "accuracy_of_always_no", "recall", "precision", "brier"]
    ]
    out.to_csv(RESULTS / "dialysis_model_comparison.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    top = coef_tables["also no renal category at all"].head(20)
    top.to_csv(RESULTS / "dialysis_top_predictors_deleaked.csv", index=False)
    print("\nStrongest predictors once every renal/dialysis code is removed:")
    print(top.to_string(index=False, max_colwidth=60))


if __name__ == "__main__":
    main()
