"""Loading helpers shared by the notebooks and the analysis scripts.

Everything here reads from ``cache/``, which is produced by
``build_transactions.py`` and is gitignored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from config import CACHE, MIN_CELL


class Dataset:
    """The NIS discharges as a sparse discharge x ICD-10-code matrix."""

    def __init__(self) -> None:
        self.X: sp.csr_matrix = sp.load_npz(CACHE / "transactions.npz")
        self.codes: np.ndarray = np.load(CACHE / "code_index.npy", allow_pickle=True)
        self.lookup: pd.DataFrame = pd.read_csv(
            CACHE / "code_lookup.csv", index_col=0, dtype=str
        ).fillna("")
        self.meta: pd.DataFrame = pd.DataFrame(
            {k: v for k, v in np.load(CACHE / "row_meta.npz").items()}
        )
        self._col = {c: i for i, c in enumerate(self.codes)}
        self._counts: np.ndarray | None = None

    # -- vocabulary ------------------------------------------------------
    def __len__(self) -> int:
        return self.X.shape[0]

    def column(self, code: str) -> int:
        if code not in self._col:
            raise KeyError(f"{code!r} does not appear in the 2019 NIS")
        return self._col[code]

    def describe(self, code: str) -> str:
        return self.lookup["description"].get(code, "(not in the AHRQ lookup)")

    @property
    def counts(self) -> np.ndarray:
        """Number of discharges carrying each code."""
        if self._counts is None:
            self._counts = np.asarray(self.X.sum(axis=0)).ravel()
        return self._counts

    # -- cohorts ---------------------------------------------------------
    def has_code(self, code: str) -> np.ndarray:
        """Boolean mask over discharges carrying `code` in *any* DX slot.

        One column of the discharge x code matrix is already the union over
        all 40 diagnosis positions, so this cannot double-count a discharge
        that lists the same code twice — which per-column filtering stitched
        back together with ``concat`` would.
        """
        return np.asarray(self.X[:, self.column(code)].todense()).ravel().astype(bool)

    def cooccurring(self, code: str, min_count: int = MIN_CELL) -> pd.DataFrame:
        """Codes appearing alongside `code`, with lift against base rate.

        Counts below the HCUP cell-size threshold are dropped, so the result
        is safe to publish.
        """
        target = self.column(code)
        mask = self.has_code(code)
        n, n_cohort = len(self), int(mask.sum())

        within = np.asarray(self.X[mask].sum(axis=0)).ravel()
        overall = self.counts

        df = pd.DataFrame({
            "code": self.codes,
            "n_in_cohort": within,
            "n_overall": overall,
        })
        df = df[(df["code"] != code) & (df["n_in_cohort"] >= min_count)].copy()
        df["pct_of_cohort"] = df["n_in_cohort"] / n_cohort
        df["pct_overall"] = df["n_overall"] / n
        df["lift"] = df["pct_of_cohort"] / df["pct_overall"]
        df["description"] = df["code"].map(self.lookup["description"])
        df["ccsr_description"] = df["code"].map(self.lookup["ccsr_description"])
        assert target not in df.index
        return df.sort_values("n_in_cohort", ascending=False)

    def profile(self, mask: np.ndarray) -> pd.Series:
        """Aggregate description of a cohort. No record-level values."""
        m = self.meta[mask]
        n = len(m)
        if n < MIN_CELL:
            raise ValueError(
                f"cohort of {n} discharges is below the HCUP cell-size "
                f"threshold of {MIN_CELL}; nothing may be reported"
            )
        return pd.Series({
            "discharges": n,
            "pct_of_all_discharges": 100 * n / len(self),
            "weighted_national_estimate": float(m["DISCWT"].sum()),
            "mean_age": m["AGE"].mean(),
            "pct_female": 100 * m["FEMALE"].mean(),
            "mean_length_of_stay": m["LOS"].mean(),
            "in_hospital_mortality_pct": 100 * m["DIED"].mean(),
            "mean_diagnoses_coded": m["I10_NDX"].mean(),
        })
