# Data handling — read before you push anything

## What this data actually is

`NIS_2019_Core.SAV` is the **HCUP National Inpatient Sample, 2019 Core file**
(AHRQ). 7,083,805 discharge records — a 20% stratified sample of US community
hospital discharges, which is why every record carries a `DISCWT` of ~5.

Two things worth being precise about, because they change what the risk is:

- **It is not HIPAA PHI.** The NIS ships de-identified: no names, no MRNs, no
  dates of birth, no admission dates, ages top-coded at 90. So a leak is not a
  HIPAA breach and you do not need to say it is.
- **It is restricted-use data under a signed Data Use Agreement.** That is the
  live risk. The HCUP DUA forbids redistributing record-level data to anyone
  who has not signed their own DUA. Pushing the `.SAV` — or a notebook with
  record-level rows in its saved output — to a public *or private* GitHub repo
  is redistribution. That is a DUA violation regardless of who actually looks
  at it, and it is the kind of thing that gets data access revoked.

The DUA also carries a **cell-size rule**: do not publish any statistic derived
from a cell containing **10 or fewer discharges**. This applies to your
portfolio writeup, not just to raw files. Every number in `results/` is
suppressed against this threshold automatically — see
`src/mine_associations.py`.

`Diagnoses Codes.xlsx` and `HCUP-NIS2016-2020-DXandPRfreqs.xlsx` are AHRQ
publications. They are aggregate, but they are AHRQ's to distribute, not yours,
so they stay gitignored too.

## Where the leak risk actually is

Not the `.SAV`. That one is obvious and a `.gitignore` handles it.

The risk is **notebooks**. Running a cell that prints a dataframe saves that
output *inside the `.ipynb`* — so a file that looks like source code can carry
hundreds of patient records in it. A single `df.head()` during exploration is
enough. Left unattended in this project it reached 1.95 MB of record-level
rows across three notebooks, each printed row carrying `KEY_NIS`, `HOSP_NIS`,
`NIS_STRATUM`, `AGE`, `RACE`, `ZIPINC_QRTL` and a full 40-code diagnosis
vector.

That combination is the problem. Hospital + stratum + age + sex + a 40-code
diagnosis vector is close to unique for an individual patient, and no rule
about `*.SAV` catches any of it.

Outputs are stripped before every commit, and the pre-commit hook refuses any
notebook that still has them.

## The three layers now in place

1. **`.gitignore`** — raw data, reference workbooks, derived caches, notebook
   checkpoints, `_originals_do_not_commit/`. Allows `results/*.csv` back in.
2. **`tools/pre-commit`** — blocks the commit if a restricted extension is
   staged (even via `git add -f`), if a notebook has saved outputs, if any file
   is >5 MB, or if `KEY_NIS` values appear in the diff. Install it:
   ```bash
   bash tools/install_git_hooks.sh
   ```
   Run this immediately after `git init` — the hook lives in `.git/hooks/`,
   which is not itself version-controlled, so cloning the repo elsewhere means
   installing it again.
3. **`tools/strip_notebook_outputs.py`** — run before any commit that touches a
   notebook. The hook will tell you if you forget.

## Publishing the portfolio version

Safe to publish: everything in `src/`, `tools/`, the stripped notebooks, and
`results/` (aggregate, ≥11 discharges per cell, no record-level rows).

Not safe, ever: the `.SAV`, anything derived from it at record level, the two
AHRQ workbooks, and `_originals_do_not_commit/`.

If a recruiter asks to run it: they can read the code and the results, but they
cannot reproduce it without their own HCUP DUA. Say that in the README — it
reads as rigor, not as an excuse.

## If something does get pushed

Deleting the file in a later commit does **not** remove it — it stays in git
history and in every clone and fork. You would need to purge history
(`git filter-repo`), force-push, delete any forks, and rotate the repo. On
GitHub, also contact support to expire cached views. Assume anything that
reached a public remote is permanently disclosed and notify HCUP.
