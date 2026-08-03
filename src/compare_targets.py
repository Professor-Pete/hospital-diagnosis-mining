"""Which diagnosis is easiest to predict from the rest of the chart?

N18.6 (end-stage renal disease) was the first target tried. This runs the
same model against a spread of other conditions to see whether any is both
predicted more accurately and easier to explain.

Two things make the comparison fair:

**One automatic leakage filter, applied identically to every target.** No
code gets hand-tuned exclusions that another does not. For target T the
filter drops T itself, everything sharing T's 3-character ICD root,
everything in T's AHRQ CCSR clinical category, and every code whose
description reuses T's distinctive words. That last rule is what removes
"end stage renal disease" when the target is "dependence on renal dialysis",
without anyone having to think of it.

**One fixed model configuration.** Cross-validating hyperparameters per
target would let an easy target look good partly because it got a better
search. Every target gets the same 800-feature selection and the same
regularisation — the same settings ``predict_esrd.py`` lands on, so the
N18.6 number here matches the one reported there.

Scores are reported two ways because prevalence differs enormously across
these codes: ROC-AUC is prevalence-independent and comparable directly,
while average precision is divided by its own chance floor to give a "times
better than guessing" figure.

Run:  python src/compare_targets.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import RESULTS
from mine_associations import _content_words
from predict_esrd import SEED, load, sample

# Matched to predict_esrd.py exactly. If these drifted, N18.6 would score
# differently here than in its own figure, and a reader would have no way to
# reconcile the two numbers.
SUBSAMPLE = 600_000
K_FEATURES = 800

# Conditions a patient *has*. AHRQ gives each of these a real clinical CCSR
# category, meaning it can stand alone as a reason for admission.
TARGETS = [
    "I509",    # heart failure, unspecified
    "E119",    # type 2 diabetes without complications
    "G4733",   # obstructive sleep apnea
    "D649",    # anaemia, unspecified
]

# Run separately, under the strict renal exclusion (every kidney-related CCSR
# category, plus any code whose description names the organ) rather than the
# uniform automatic filter. The generic rule leaves too much behind for this
# one: it would keep "dependence on renal dialysis", which gives the answer
# away outright. Using the conservative filter keeps one number for this
# disease across the whole project.
RENAL_TARGETS = {
    "N186": "End stage renal disease",
}


def leak_filter(target: str, codes: np.ndarray, lookup: pd.DataFrame) -> np.ndarray:
    """Codes to keep as features for `target`. Identical rules for every target."""
    idx = pd.Index(codes)
    desc = pd.Series(idx.map(lookup["description"]).fillna(""), index=idx)
    ccsr = pd.Series(idx.map(lookup["ccsr"]).fillna("?"), index=idx)

    target_words = _content_words(lookup["description"].get(target, ""))
    target_ccsr = lookup["ccsr"].get(target, "?")

    shares_words = np.array([
        bool(target_words & _content_words(d)) for d in desc
    ])
    same_ccsr = (ccsr.to_numpy() == target_ccsr) & (target_ccsr != "?")
    return (
        np.asarray(idx != target)
        & np.asarray(idx.str[:3] != target[:3])
        & ~same_ccsr
        & ~shares_words
    )


def main() -> None:
    X, codes, lookup, _ = load()
    col_of = {c: i for i, c in enumerate(codes)}
    Xs, _ = sample(X, np.zeros(X.shape[0], dtype=np.int8), size=SUBSAMPLE)
    counts = np.asarray(Xs.sum(axis=0)).ravel()
    common = counts >= 100

    rows, tops = [], {}
    for target in TARGETS:
        if target not in col_of:
            continue
        y = np.asarray(Xs[:, col_of[target]].todense()).ravel().astype(np.int8)
        if y.sum() < 500:
            print(f"skipping {target}: only {y.sum()} positives in the subsample")
            continue

        keep = np.flatnonzero(leak_filter(target, codes, lookup) & common)
        Xr, names = Xs[:, keep], codes[keep]

        X_tr, X_te, y_tr, y_te = train_test_split(
            Xr, y, test_size=0.25, random_state=SEED, stratify=y
        )
        model = Pipeline([
            ("select", SelectKBest(score_func=f_classif, k=min(K_FEATURES, len(keep)))),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        ]).fit(X_tr, y_tr)

        prob = model.predict_proba(X_te)[:, 1]
        prevalence = y_te.mean()
        ap = average_precision_score(y_te, prob)
        rows.append({
            "code": target,
            "description": lookup["description"].get(target, "?"),
            "prevalence_pct": 100 * prevalence,
            "roc_auc": roc_auc_score(y_te, prob),
            "avg_precision": ap,
            "ap_baseline": prevalence,
            "lift_over_chance": ap / prevalence,
            "features_dropped_as_leaky": int(len(codes) - len(keep)),
        })

        sel = model.named_steps["select"]
        clf = model.named_steps["clf"]
        chosen = names[sel.get_support()]
        tops[target] = (
            pd.DataFrame({
                "target": target,
                "code": chosen,
                "odds_ratio": np.exp(clf.coef_.ravel()),
                "description": [lookup["description"].get(c, "?") for c in chosen],
            })
            .nlargest(8, "odds_ratio")
        )
        print(f"  {target:<7} {lookup['description'].get(target,'?')[:44]:<46} "
              f"AUC {rows[-1]['roc_auc']:.3f}  AP {ap:.3f}  "
              f"({rows[-1]['lift_over_chance']:.0f}x chance)")

    from predict_esrd import regimes as esrd_regimes
    strict = esrd_regimes(codes, lookup)["also no renal category at all"]

    for target, label in RENAL_TARGETS.items():
        y = np.asarray(Xs[:, col_of[target]].todense()).ravel().astype(np.int8)
        # `strict` already excludes every renal code, but the target itself
        # is renal, so make certain it is out.
        keep = np.flatnonzero(strict & common & (codes != target))
        Xr, names = Xs[:, keep], codes[keep]
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xr, y, test_size=0.25, random_state=SEED, stratify=y
        )
        model = Pipeline([
            ("select", SelectKBest(score_func=f_classif, k=min(K_FEATURES, len(keep)))),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        ]).fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        prevalence = y_te.mean()
        ap = average_precision_score(y_te, prob)
        rows.append({
            "code": target, "description": label,
            "prevalence_pct": 100 * prevalence,
            "roc_auc": roc_auc_score(y_te, prob), "avg_precision": ap,
            "ap_baseline": prevalence, "lift_over_chance": ap / prevalence,
            "features_dropped_as_leaky": int(len(codes) - len(keep)),
        })

        sel, clf = model.named_steps["select"], model.named_steps["clf"]
        chosen = names[sel.get_support()]
        tops[target] = pd.DataFrame({
            "target": target, "code": chosen,
            "odds_ratio": np.exp(clf.coef_.ravel()),
            "description": [lookup["description"].get(c, "?") for c in chosen],
        }).nlargest(8, "odds_ratio")
        print(f"  {target:<7} {label[:44]:<46} AUC {rows[-1]['roc_auc']:.3f}  "
              f"AP {ap:.3f}  ({rows[-1]['lift_over_chance']:.0f}x chance)")

    df = pd.DataFrame(rows).sort_values("lift_over_chance", ascending=False)
    df["filter"] = np.where(df["code"].isin(RENAL_TARGETS),
                            "strict renal exclusion", "uniform automatic")
    df.to_csv(RESULTS / "target_comparison.csv", index=False)
    pd.concat(tops.values()).to_csv(RESULTS / "target_top_predictors.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 46)
    print("\nRanked by how far above chance the model gets:\n")
    print(df[["code", "description", "prevalence_pct", "roc_auc",
              "avg_precision", "lift_over_chance"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote target_comparison.csv and target_top_predictors.csv")


if __name__ == "__main__":
    main()
