# PathSnap improvement run

This is an engineering experiment record, not the team's submission report.
See [the audit](EXPERIMENT_AUDIT.md) for the starting pipeline and weaknesses.

## Results

CV is on the same three grouped folds of the 80 training trips. The remaining
columns are means across the ten student-test trips. All point metrics use the
same frozen nearest-reference proxy, not organizer labels. The student set was
scored after configuration selection in this run; it was already known from the
original project and is not a pristine external holdout.

| Stage | CV macro F1 | Student macro F1 | Student point agreement | Student continuity | Student sequence similarity | Scoring ms/trip |
|---|---:|---:|---:|---:|---:|---:|
| Original classifier | .5393 | .5592 | 76.14% | 86.02% | .6058 | 0.36 |
| Original full pipeline | .5967 | .6278 | 78.65% | 99.60% | .6754 | 4.72 |
| Best features: heading | .6510 | .6815 | 81.09% | 90.77% | .7206 | 0.43 |
| Best candidates: sampled 16 / 35 m | .6930 | .7143 | 84.34% | 91.64% | .7321 | 0.92 |
| Best classifier: logistic | .6930 | .7143 | 84.34% | 91.64% | .7321 | 0.81 |
| Best hyperparameters: logistic C = .011825 | .6932 | .7145 | 84.29% | 91.64% | .7327 | 0.83 |
| Updated classifier + original decoder settings | .7227 | .7421 | 84.68% | 99.76% | .7829 | 6.23 |
| Best decoder / final combined pipeline | .7537 | .7575 | 85.70% | 99.55% | .8007 | 11.60 |

## What changed the outcome

- **Features:** The strongest single ablation adds forward and smoothed
  heading. Temporal candidate support, directed projection/progress, and
  the combined features all trailed heading alone on these folds.
- **Candidates:** The chosen index supplements segment midpoints with
  interior samples. The 35 m radius removes distant candidates, except a
  nearest fallback, with a cap of 16. It averaged 7.05 active candidates
  versus 8, with CV recall of 97.29% versus 96.19%. Search averaged
  45.82 ms/trip versus 17.56 ms/trip, excluding file loading. Unfiltered
  sampled 16 retained more proxy arcs but scored worse on CV macro F1.
- **Classifier:** Logistic regression led on CV macro F1. Class weighting
  reduced CV macro F1 to .6848. Random forest and the extended and weighted
  histogram boosting variants did not beat logistic. XGBoost could not import
  because its native OpenMP runtime was unavailable. TPE varied C over
  [.01, 30] for eight trials, with practically no change in classifier CV F1.
- **Decoder:** The selected values were 6 s gap, jump penalty 4, emission
  weight 1, motion weight 1.5. The motion term penalizes differences between
  GPS displacement and projected directed along-road distance for same-arc
  or direct-neighbour moves. It does not treat disconnected arcs as connected.
  Against the updated scorer with original decoder settings, this gained
  .0311 CV macro F1 and 1.02 percentage points of student agreement, for
  a 0.20-point continuity cost.
- **Paired check:** Agreement improved on 10/10 student trips and macro F1
  on 9/10. Mean paired agreement gain was 7.04 percentage points; the
  descriptive 95% trip-bootstrap interval was 4.49 to 9.57 points. The
  analogous interval for continuity change spans zero. Neither interval
  establishes performance against independently verified labels.

## Artifacts and remaining work

The [comparison CSV](results_improved/comparison.csv) includes weighted/micro
F1, candidate recall, and inference timing. [Experiment logs](results_improved/experiments.json),
[candidate timings](results_improved/candidate_benchmark.csv),
[TPE trials](results_improved/tuning_trials.csv), and per-trip CSVs record the
40 comparisons. The same directory contains plots, the Folium comparison map,
model weights and inference config. See [the README](README.md) for reproduction
and opt-in deployment. No original artifacts were replaced.

The development CV figures are selection scores, not nested-CV estimates.
Research next: compare ordered reference alignment under a separate scoring
protocol; inspect difficult trips and long/disconnected arcs; and verify
official server-side scoring and the two-laptop run. The point-level scorer
remains approximate; the decoder cannot reconstruct unobserved intermediate
arcs and the heading model assumes a complete trajectory.
