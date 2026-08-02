"""Shared paths and constants.

Every path that points at restricted data is resolved here so there is
exactly one place to check when auditing what the code touches.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Restricted inputs (gitignored, never leave this machine) ---
SAV = ROOT / "NIS_2019_Core.SAV"
CODE_LOOKUP = ROOT / "Diagnoses Codes.xlsx"
CODE_LOOKUP_SHEET = "T.2_By_DX_Code"

# --- Derived intermediates (gitignored) ---
CACHE = ROOT / "cache"

# --- Publishable aggregate output ---
RESULTS = ROOT / "results"

# --- NIS structure ---
DX_COLS = [f"I10_DX{i}" for i in range(1, 41)]  # all 40 — DX22 included
N_DX = 40

# HCUP Data Use Agreement: never publish a statistic derived from a cell
# containing 10 or fewer discharges. Enforced in mine_associations.py.
MIN_CELL = 11

for _d in (CACHE, RESULTS):
    _d.mkdir(exist_ok=True)
