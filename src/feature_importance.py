"""How much predictive power does each part of the kidney-failure signature carry?

The de-leaked model identifies end-stage renal disease without ever seeing a
kidney code. Its surviving predictors fall into a few clinical groups, and the
obvious question is which of them is doing the work.

Regression coefficients answer a different question: they give the effect of
one feature holding the others fixed, which is misleading when features move
together — five vascular-access complication codes are highly correlated, so
each one's individual coefficient understates what the group contributes.

So this measures **grouped permutation importance** on the held-out test set.
For each group, the values of all its columns are shuffled across patients
together, destroying that group's relationship with the outcome while leaving
everything else intact, and the drop in average precision is recorded. That
is a direct answer to "how much predictive power did this carry", and it
handles the correlation correctly because the whole group moves at once.

Run after predict_esrd.py:  python src/feature_importance.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import RESULTS
from predict_esrd import SEED, TARGET, load, regimes, sample

N_REPEATS = 5
STRICT = "also no renal codes or dialysis hardware"

# The clinical story each surviving predictor belongs to. Grouping is by what
# the condition *is*, not by ICD chapter — the vascular-access codes sit in
# the injury/complication chapter but they are a dialysis phenomenon.
GROUPS = {
    # The classic consequences of kidneys that can no longer filter blood:
    # potassium and acid the body cannot clear.
    "Electrolyte & acid-base": ["E875", "E872", "E871", "E8749"],
    "Bone & mineral chemistry": [
        "E839", "E8359", "E213", "E211", "E892", "M898X9", "M899", "E8351",
    ],
    # No vascular-access group: every T82 device-complication code, and the
    # external-cause codes for the same events, are now excluded as leakage
    # before the model is fit. Nothing is left here to measure.
    "Fluid overload": ["E8770", "E8779", "E877", "J810"],
    "Transplant pathway": ["Z7682", "Z9483", "Y830", "Z940", "Z948"],
    "Other systemic": ["E8889", "I776", "E859"],
}


def with_permuted_columns(X: sp.csr_matrix, cols: np.ndarray, rng) -> sp.csr_matrix:
    """Copy of X with `cols` shuffled across rows *together*.

    Shuffling the group as a block preserves the correlations within it, so
    what gets destroyed is the group's link to the outcome and nothing else.
    """
    n, m = X.shape
    if len(cols) == 0:
        return X
    perm = rng.permutation(n)
    keep = np.ones(m, dtype=np.int8)
    keep[cols] = 0
    zeroed = X.multiply(sp.csr_matrix(keep.reshape(1, -1)))
    scatter = sp.csr_matrix(
        (np.ones(len(cols)), (np.arange(len(cols)), cols)), shape=(len(cols), m)
    )
    return (zeroed + X[perm][:, cols] @ scatter).tocsr()


def main() -> None:
    X, codes, lookup, y_all = load()
    Xs, ys = sample(X, y_all)

    keep_mask = regimes(codes, lookup)[STRICT]
    counts = np.asarray(Xs.sum(axis=0)).ravel()
    keep = np.flatnonzero(keep_mask & (counts >= 100))
    Xr, names = Xs[:, keep], codes[keep]

    X_tr, X_te, y_tr, y_te = train_test_split(
        Xr, ys, test_size=0.25, random_state=SEED, stratify=ys
    )

    model = Pipeline([
        ("select", SelectKBest(score_func=f_classif, k=800)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
    ]).fit(X_tr, y_tr)

    prob = model.predict_proba(X_te)[:, 1]
    base = average_precision_score(y_te, prob)
    print(f"{TARGET}: baseline average precision on held-out data = {base:.4f}\n")

    # Average precision is hard to read as a number. Precision-at-k says the
    # same thing concretely: take the k patients the model ranks highest, and
    # count how many really have the disease, against what picking at random
    # would have given.
    order = np.argsort(prob)[::-1]
    prevalence = y_te.mean()
    pak = pd.DataFrame([
        {"ranked_by_model": k,
         "actually_have_esrd": int(y_te[order[:k]].sum()),
         "hit_rate_pct": 100 * y_te[order[:k]].mean(),
         "expected_if_picked_at_random": round(k * prevalence),
         "test_set_size": len(y_te)}
        for k in (100, 500, 1000, 5000)
    ])
    pak.to_csv(RESULTS / "esrd_precision_at_k.csv", index=False)
    print(pak.to_string(index=False), "\n")

    pos = {c: i for i, c in enumerate(names)}
    rng = np.random.default_rng(SEED)

    rows = []
    assigned: set[str] = set()
    for group, members in GROUPS.items():
        cols = np.array([pos[c] for c in members if c in pos], dtype=int)
        present = [c for c in members if c in pos]
        assigned.update(present)
        if len(cols) == 0:
            continue
        drops = []
        for _ in range(N_REPEATS):
            Xp = with_permuted_columns(X_te, cols, rng)
            drops.append(base - average_precision_score(y_te, model.predict_proba(Xp)[:, 1]))
        rows.append({
            "group": group,
            "codes": len(cols),
            "ap_drop": float(np.mean(drops)),
            "ap_drop_sd": float(np.std(drops)),
            "pct_of_model": 100 * float(np.mean(drops)) / base,
            "example_codes": ", ".join(present[:4]),
        })
        print(f"  {group:<28} {len(cols):>2} codes   AP drop {np.mean(drops):.4f}")

    # All the named groups permuted together — how much of the model do the
    # recognisable clinical signatures actually account for?
    named = np.array([pos[c] for c in assigned], dtype=int)
    drops = []
    for _ in range(N_REPEATS):
        Xp = with_permuted_columns(X_te, named, rng)
        drops.append(base - average_precision_score(y_te, model.predict_proba(Xp)[:, 1]))
    named_drop = float(np.mean(drops))

    # And everything the groups did not claim, so the table accounts for the
    # whole model rather than only its highlights.
    rest = np.array([i for c, i in pos.items() if c not in assigned], dtype=int)
    drops = []
    for _ in range(N_REPEATS):
        Xp = with_permuted_columns(X_te, rest, rng)
        drops.append(base - average_precision_score(y_te, model.predict_proba(Xp)[:, 1]))
    rows.append({
        "group": "All named groups together", "codes": len(named),
        "ap_drop": named_drop, "ap_drop_sd": 0.0,
        "pct_of_model": 100 * named_drop / base, "example_codes": "",
    })
    rows.append({
        "group": "Everything else (all remaining codes)", "codes": len(rest),
        "ap_drop": float(np.mean(drops)), "ap_drop_sd": float(np.std(drops)),
        "pct_of_model": 100 * float(np.mean(drops)) / base, "example_codes": "",
    })
    print(f"\n  {'All named groups together':<38} {len(named):>4} codes   "
          f"AP drop {named_drop:.4f}  ({100 * named_drop / base:.0f}% of the model)")
    print(f"  {'Everything else':<38} {len(rest):>4} codes   "
          f"AP drop {np.mean(drops):.4f}  ({100 * np.mean(drops) / base:.0f}%)")

    df = pd.DataFrame(rows).sort_values("ap_drop", ascending=False)
    df["baseline_ap"] = base
    df.to_csv(RESULTS / "esrd_feature_importance.csv", index=False)
    print(f"\nwrote {(RESULTS / 'esrd_feature_importance.csv').name}")
    print(df[["group", "codes", "ap_drop", "pct_of_model"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
