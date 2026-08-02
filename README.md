# What co-occurs with what: 7 million hospital records

No medical or statistics background needed to read this.

## The data

Every time someone is admitted to a US hospital and later discharged, the
hospital records what was wrong with them — not in sentences, but as codes.
"Type 2 diabetes with kidney complications" is `E11.22`. "Dependence on renal
dialysis" is `Z99.2`. About 73,000 of these codes exist, and a typical hospital
stay is tagged with around twelve of them.

This project uses the **HCUP National Inpatient Sample 2019**: **7,083,805
discharge records**, about one in five hospital stays in the country that year,
carrying **86 million individual diagnosis codes**.

**The data file is not in this repository and never will be.** It is
restricted-use federal data, released only under a signed Data Use Agreement
that forbids redistributing patient-level records to anyone who has not signed
their own. Publishing it — to a public repo *or* a private one — would breach
that agreement. It is de-identified, so this is not a HIPAA matter, but it is a
legal one.

Everything in `results/` is aggregate and passes the agreement's cell-size rule
(no figure may rest on 10 or fewer patients), so the findings are readable
without the data. Reproducing them requires your own DUA with AHRQ. Details in
[`DATA_HANDLING.md`](DATA_HANDLING.md).

## What the project was trying to do

A simple question: **which diagnoses show up together more often than you'd
expect by chance?**

This is the same question Amazon asks about products. If people who buy a tent
also tend to buy a sleeping bag, that's worth knowing. The technique is called
association mining, and it works the same way whether the basket holds camping
gear or diagnoses.

With 7 million hospital stays, the hope was that the data might reveal
connections between conditions that people hadn't noticed.

## Filtering out the false leads

Finding pairs that occur together is easy. Almost all of them are junk, and the
work is in removing the junk without also removing the real findings.

![Almost nothing survives the filters: 191,176 statistically significant diagnosis pairs reduce to roughly 1,200 distinct findings.](figures/01-funnel.svg)

Reading that chart from top to bottom:

- **191,176 pairs passed a strict statistical test.** Co-occurrence counts for
  every possible pair of codes come from a single large matrix multiplication
  over all 7 million records. A pair only counts if it appears together far
  more often than chance predicts *and* clears a significance threshold
  tightened to account for the sheer number of pairs being tested at once —
  test enough combinations and thousands will look remarkable by luck alone.

- **159,838 aren't just the coding rulebook restated.** Many code pairs are
  guaranteed to appear together because the rules say so. "Dialysis dependence"
  and "end-stage kidney disease" is not a discovery — dialysis is the treatment
  *for* kidney failure, and coders are instructed to record both. Same for
  hamburger-buns-and-hamburgers pairs like a birth-related tear alongside a code
  that literally means "single live birth". These get flagged automatically by
  reading each code's own description, and set aside.

- **6,000 strongest candidates examined in detail.** A deliberate cap — the
  remaining checks are slower, so they run on the strongest candidates rather
  than all 159,838.

- **5,456 survived the hospital-origin check.** Explained below.

- **1,197 distinct findings.** Many surviving pairs are the same medical story
  told at slightly different levels of detail — diabetes-with-complication-A
  alongside kidney-stage-3, then stage-4, then stage-5. Merging those leaves
  about 1,200 genuinely separate findings.

## Finding 1: nearly everything real was already known — and the exceptions were fake

Once the artefacts were stripped out, the strongest surviving connections were
things like cystic fibrosis with pancreatic failure, diabetic nerve damage with
a stomach-emptying disorder, and severe bedsores with bone infection at the same
site. All well documented. No surprises.

**That is a good result, not a disappointing one.** A measuring instrument that
gets the verifiable things right is one you can trust elsewhere. If the method
had instead produced a pile of shocking new medical claims, the correct reaction
would be that the method is broken.

Which leaves the handful of pairs that *weren't* medically recognised. The
strongest was **tension headaches and chronic stuffy nose** — a link nothing
obvious explains, which survived every quality check. By the numbers, the single
most interesting result in the dataset.

So one more question: which hospitals is it coming from?

![How many hospitals produced each pattern. Established medical links appear at 271 to 2,758 hospitals; the suspected artefacts at 11 to 55.](figures/02-hospitals.svg)

The dataset covers **4,568 hospitals**. That pattern came from **eleven**.

Some hospitals copy a patient's entire history of minor ongoing complaints onto
the record even when the admission is for something unrelated. At those
hospitals, a set of unconnected background problems gets written down together
over and over, and ends up looking statistically inseparable.

The top six bars are relationships nobody disputes, used to check the test was
calibrated; the bottom five are the suspicious ones. You don't need statistics
to see the difference. Applied to every candidate, **9.1% failed** — and the
failure rate was *higher* among the top-ranked results, because a quirk
concentrated in a few hospitals is exactly what produces an extreme-looking
number.

None of the standard statistical tools can catch this. A real medical link and a
local paperwork habit produce identical-looking output. The only way to tell
them apart is to ask where the records came from.

## Bonus finding: some real patterns are hidden by the wrong comparison group

Newborns get tested for metabolic problems when their mother has diabetes. That
link looks moderately strong in the raw numbers — and about six times stronger
once you compare newborns only against other newborns.

![The same link measured two ways: 26 times more likely than chance across all hospital stays, but 168 times when measured within newborns only.](figures/04-newborn.svg)

Roughly 600,000 records here are newborns; 6.4 million are adults. A pattern
that can only exist among babies gets averaged against millions of adults who
couldn't possibly have it, and comes out looking weaker than it is.

This is worth flagging because it runs backwards from what people expect. The
usual worry is that a statistic looks *stronger* than reality, so analysts
discount big numbers. Here the raw figure was too *small*. Twenty-two other
pairs in this dataset had the same thing happen, and there are almost certainly
more — any condition confined to one age group, one sex, or one narrow patient
population will be understated the same way whenever it's measured against
everybody. It's likely a more common problem than it gets credit for.

## Building a model to predict a diagnosis

The second half of the project: pick one diagnosis code, and see whether it can
be predicted from the *other* codes on the same record.

The code chosen was **`Z99.2` — dependence on renal dialysis**. It appears on
2.37% of hospital stays.

That number matters immediately. A "model" that answers *no* to everyone, every
time, is right 97.6% of the time and has learned nothing. So accuracy is useless
here, and the model is scored on how well it picks dialysis patients out of the
crowd, against a random-guessing floor.

The first attempt scored extremely well — and it was cheating. Its most powerful
clue was a code meaning *"patient's noncompliance with renal dialysis."* You
cannot skip a dialysis appointment if you aren't on dialysis. Several other top
predictors were codes the rulebook *requires* to be recorded alongside `Z99.2`.
The model wasn't diagnosing anyone; it was noticing that the chart already said
"dialysis" a few columns over — like predicting who owns a car by checking
whether they have car insurance. This is called **target leakage**, and it is
the most common way a medical prediction model produces a number that falls
apart in the real world.

So every kidney- and dialysis-related code was removed, and the model was fit
again from scratch.

![Spotting dialysis patients with every kidney clue taken away: 14 times better than random guessing.](figures/03-model.svg)

**This is the result worth reporting.** With no kidney reference of any kind
available, the model still identifies dialysis patients roughly 14 times better
than chance. It does it by recognising the body-wide damage that kidney failure
causes:

- **Disrupted bone and mineral chemistry.** Failing kidneys stop regulating
  calcium and phosphate, so the skeleton pays for it — disorders of mineral
  metabolism, calcium metabolism, and overactive parathyroid glands all rank
  near the top.
- **Fluid overload.** Kidneys that can no longer excrete water.
- **Repeated failures of the surgically implanted access port.** Dialysis
  requires a permanent connection into a blood vessel, and those connections
  bleed, clot, narrow, and get infected. The model picks up all four.

None of those codes mentions a kidney. The model reconstructed the shape of the
disease from its consequences — which is a genuinely interesting result, and one
that was completely hidden while the leaky codes were still in play.

## What this cannot tell you

- **These are hospital stays, not people.** There's no patient identifier, so
  someone admitted three times in 2019 appears as three unrelated records.
- **Two things on the same chart doesn't mean one caused the other.** It can be
  a shared underlying cause, a billing incentive to record both, or — as Finding
  1 shows — one hospital's habit.
- **Only age and sex were adjusted for.** Not income, insurance, or region.
- **The data flags whether a condition was present on arrival, and I didn't use
  it.** That would separate "these go together in patients" from "one happened
  *during* the stay" — a different claim, and the biggest available improvement
  to this work.

## Running it

```bash
python src/build_transactions.py     # .SAV -> sparse matrix, ~3 min
python src/mine_associations.py      # pair mining over all 7.08 M discharges
python src/hospital_concentration.py # is it clinical, or one hospital's habit?
python src/triage_findings.py        # collapse to a reviewable shortlist
python src/predict_dialysis.py       # the model, under three leakage regimes
python tools/make_figures.py         # regenerate the charts above
```

Dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `pyreadstat`.

```
src/          the pipeline
tools/        figure generator, notebook output stripper, pre-commit leak guard
figures/      the charts above, regenerated from results/
results/      aggregate, cell-suppressed findings (safe to publish)
*.ipynb       the same analyses as annotated notebooks, in reading order
```

The full ranked tables behind every figure are in `results/`.
