#!/usr/bin/env python3
"""Strip all outputs and execution counts from the project notebooks.

The NIS notebooks embed record-level HCUP data in their saved outputs
(KEY_NIS, HOSP_NIS, NIS_STRATUM, AGE, RACE, ZIPINC_QRTL and the full
40-code diagnosis vector for individual discharges). Committing an
.ipynb with those outputs redistributes restricted data even though the
.SAV itself is gitignored.

Run this before every commit:  python tools/strip_notebook_outputs.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKUP = ROOT / "_originals_do_not_commit"


def strip(path: Path, backup: bool = True) -> tuple[int, int]:
    nb = json.loads(path.read_text())
    before = len(json.dumps(nb))

    touched = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            touched += 1
        cell["outputs"] = []
        cell["execution_count"] = None
        # Papermill/VSCode stash scratch data here too.
        cell.get("metadata", {}).pop("execution", None)

    nb.get("metadata", {}).pop("widgets", None)

    if backup:
        BACKUP.mkdir(exist_ok=True)
        target = BACKUP / path.name
        if not target.exists():
            shutil.copy2(path, target)

    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    after = len(json.dumps(nb))
    return touched, before - after


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv[1:]] or sorted(ROOT.glob("*.ipynb"))
    if not paths:
        print("no notebooks found")
        return 0
    for p in paths:
        touched, saved = strip(p)
        print(f"{p.name}: cleared {touched} cells, {saved / 1e6:.2f} MB removed")
    print(f"\nOriginals (with outputs) kept in {BACKUP.name}/ — that folder is gitignored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
