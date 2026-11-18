# PathSnap final validation audit

**Scope:** Original and improved saved pipelines, training/selection source,
and recorded experiments. Recomputed student-test results without fitting a
model. No implementation, model, configuration, experiment log, or existing
result was changed. No reserved evaluator data was used.

**Verdict:** The saved improved model reproduces the reported development
scores. Ten checks passed at the scopes below. Two remain **NOT VERIFIED**:
complete historical byte provenance for the baseline and a fresh replay of
all 40 training experiments. No failed metric checks were found.

## Verification register

| # | Verification | Status | Evidence |
|---|---|---|---|
| 1 | Original model, configuration, and results remain unchanged | **NOT VERIFIED** for complete historical byte identity | The original model matches the SHA-256 recorded before the improvement handoff and is byte-identical to `results_recheck/model.joblib`. Original config and inventory also match `results_recheck/` byte-for-byte. All non-timing metrics in the two original summaries agree, and the final baseline metrics were independently reproduced. There is no Git history or authenticated pre-improvement snapshot for every original result/config file. All 90 files across `results/`, `results_recheck/`, and `results_improved/` had identical SHA-256 hashes before and after this audit. |
| 2 | Improved artifact-to-results correspondence | **PASS** | Loaded `results_improved/model.joblib` and `results_improved/config.json` through the API. The config matches the selected `decoder_motion_gap_6` experiment; the estimator expects six features and its fitted logistic setting is `C = 0.011825176269334062`, matching trial 6. Every non-timing per-trip metric reproduced. `experiments.py:308–328` scores the fitted model before exporting that same object and config. This establishes operational correspondence, not cryptographic historical provenance. The final full-data artifact is not the set of fold-specific estimators that produced the CV scores. |
| 3 | Identical student-test trips and evaluation | **PASS** | Both model bundles returned exactly the same ordered ten trip IDs as the original student-test CSV, all eight saved student comparison CSVs, and the local student split. The audit independently reconstructed one proxy-label array per trip and scored both models against it. Both used identical arc metrics, sequence collapse, 15 s continuity cutoff, and equal-trip aggregation. Candidate log loss is over each model's eligible candidate set and should not be compared directly across candidate configurations. |
| 4 | All 40 experiments reproducible from saved configurations and logs | **NOT VERIFIED** for full end-to-end replay | **Log replay passed:** checked 40 valid feature/model/candidate/decoder configurations, all 3,200 CV trip rows, every logged fold and aggregate metric, eight tuning objectives and parameter values, five CV-only stage selections, and all nine comparison-table rows. The saved Python and five package versions match the training interpreter. The 360 permitted raw files are byte-identical to the supplied ZIP. **Fresh training replay not run:** the logs contain aggregate scores and configurations, not every fold estimator, raw prediction, or a complete immutable environment/source snapshot. |
| 5 | No student-test or reserved labels used for selection | **PASS** for the recorded automated workflow | In `experiments.py:156–190` and `:213–297`, feature, candidate, model, TPE, and decoder selection use out-of-fold predictions from `train` only. The first student-test dataset access in the runner is at `:299`, after selection is saved. Independently recovered every stage winner from CV logs alone. No reserved split is loaded by the runner. This does not establish what a human may have known before designing experiments; earlier student-test results were already available. |
| 6 | Trip-grouped CV and leakage-free fold preprocessing | **PASS** | Independently rebuilt the exact saved GroupKFold split: 27/27/26 validation trips, each training trip present exactly once in validation, no student ID overlaps, and identical validation membership in all 40 logs. The fold fit receives only its training trips; `StandardScaler` is inside the per-fold pipeline (`experiments.py:49–68,183–190`). Cached candidate/feature extraction uses only its own trip's GPS and road graph; reference labels are calculated separately. Forward features use *within-trip* lookahead, appropriate for batch matching, not streaming. |
| 7 | Prediction does not access references | **PASS** | Guarded file opens of `.route` files during live API tests and exercised all ten trips with and without scoring references for each bundle. No reference was opened by feature extraction, probability scoring, or decoding. Predicted arc IDs were identical without references; accuracy was then null. The endpoint **does** open the reference afterward when available, solely to return aggregate agreement (`api.py:50–62`). |
| 8 | Recompute all ten baseline-vs-improved predictions and scores | **PASS** | Fresh artifact inference, no retraining. Every deterministic metric in the two saved final per-trip CSVs matched the independent calculation at relative tolerance `1e-12`, absolute `1e-14`. Arc IDs were in bounds and returned geometry matched their road endpoints. Paired deltas and bootstrap intervals also matched. See the table below. |
| 9 | Metric formulas | **PASS** | Used independent proxy geometry and scikit-learn for macro precision/recall and macro, weighted, and micro F1. Independently counted exact arc matches and same-arc/directed-adjacency transitions at GPS intervals ≤15 s. Independently collapsed consecutive duplicates and ran `SequenceMatcher(autojunk=False)` against the collapsed point-proxy sequence. Compared each trip and the trip averages with the logged output. Micro F1 equals point agreement here because each point has one predicted and one reference arc. |
| 10 | Full-pipeline CV macro F1 consistency | **PASS** | Rebuilt it from the two full-pipeline per-trip fold logs: mean macro F1 within each fold, then unweighted mean across the three folds, consistent with `experiments.py:192–204`. The baseline fold values were 0.628700 / 0.557756 / 0.603501 → **0.596652**; improved 0.804908 / 0.712674 / 0.743663 → **0.753748**. Classifier-only scores 0.539330 and 0.693191 are separate. CV predictions themselves were not regenerated. |
| 11 | Five unit tests and both API modes | **PASS** | `python3 -m unittest -v test_pathsnap`: five passed. FastAPI TestClient checked `/trips`, 10 valid `/predict/{trip_id}` responses per mode, response schema and geometry, unknown-trip 404, CORS preflight, and no-reference prediction in both modes. It did not exercise real two-laptop networking. |
| 12 | Improved model is opt-in; default UI behavior | **PASS** | With artifact overrides unset, the API resolved to `results/`; setting `PATHSNAP_ARTIFACTS=results_improved` selected the improved bundle. Both use the same routes and payload schema. `ui/index.html:12–44` still defaults to `http://127.0.0.1:8000` and selects no model itself. JavaScript syntax and simulated-DOM smoke tests passed for loading a trip, requesting a prediction at the default address, and rendering the returned route. Browser rendering on a separate laptop was not tested. |

## Independent student-test recomputation

Metrics are means over the ten trips, not pooled across GPS rows. These are
proxy-label metrics, not official evaluator scores.

| Metric | Original | Improved |
|---|---:|---:|
| Macro F1 | 0.627834 | 0.757531 |
| Macro precision | 0.672654 | 0.785959 |
| Macro recall | 0.615193 | 0.750978 |
| Point agreement | 78.6536% | 85.6953% |
| Route continuity | 99.5992% | 99.5525% |
| Sequence similarity | 0.675355 | 0.800669 |
| Weighted F1 | 0.820583 | 0.876343 |
| Candidate recall | 98.6003% | 99.8160% |

Each trip cell is **original → improved**. Agreement and continuity are
percentages; the remaining metrics are on a 0–1 scale.

| Trip | Agreement % | Macro F1 | Macro precision | Macro recall | Continuity % | Sequence similarity |
|---|---:|---:|---:|---:|---:|---:|
| 00000000 | 88.57 → 94.97 | .6535 → .8370 | .6890 → .8614 | .6423 → .8260 | 99.72 → 99.84 | .7038 → .8607 |
| 00000007 | 89.36 → 89.51 | .8908 → .9628 | .9270 → .9748 | .8726 → .9565 | 100.00 → 99.93 | .6352 → .9490 |
| 00000009 | 76.55 → 84.75 | .6179 → .7351 | .6653 → .7676 | .6070 → .7247 | 99.14 → 99.05 | .8085 → .8719 |
| 00000040 | 84.17 → 93.66 | .7051 → .8974 | .7487 → .9140 | .6909 → .8925 | 99.97 → 100.00 | .5714 → .8467 |
| 00000052 | 88.55 → 90.08 | .7403 → .6782 | .7660 → .6954 | .7416 → .6773 | 99.62 → 99.23 | .9302 → .8757 |
| 00000056 | 71.73 → 76.00 | .5560 → .6085 | .5997 → .6464 | .5535 → .6069 | 99.42 → 99.29 | .5288 → .7143 |
| 00000065 | 72.14 → 78.65 | .6017 → .6970 | .6511 → .7341 | .5955 → .6988 | 99.35 → 99.87 | .7764 → .6915 |
| 00000076 | 76.45 → 89.23 | .4791 → .7231 | .5441 → .7607 | .4536 → .7106 | 99.66 → 99.39 | .6809 → .8601 |
| 00000077 | 58.81 → 72.36 | .4177 → .6968 | .4835 → .7367 | .3869 → .6880 | 99.92 → 99.62 | .4083 → .5080 |
| 00000091 | 80.21 → 87.76 | .6163 → .7394 | .6522 → .7685 | .6082 → .7284 | 99.20 → 99.31 | .7101 → .8287 |

## Inconsistencies and remaining risks

- **No numeric score discrepancy found.** Original and rerun wall-clock
  fields differ normally. Decoder benchmark times are not deterministic, and
  the experiment inference time excludes loading/candidate extraction.
- **The metrics do not all move together.** Improved point agreement and
  macro F1 rose, but mean continuity fell by 0.047 percentage points.
  Trip `00000052` gained agreement but lost macro F1; sequence similarity
  fell on two trips. The higher mean should not be described as an improvement
  in every metric or on every trip.
- **Continuity and sequence similarity are limited proxies.** Continuity
  counts staying on the same arc as connected and does not prove the vehicle
  moved plausibly. Sequence similarity uses the collapsed **point-derived
  proxy**, not the raw ordered `.route` sequence. Nearest-reference labels
  can be ambiguous at loops, directed overlaps, and intersections.
- **Selection uncertainty remains.** CV scores are non-nested selection
  scores. The ten student trips are known development data, not a pristine
  blind holdout; paired intervals describe those trips under this proxy.
  No claim is made about reserved trips or official scoring.
- **Deployment still needs a live rehearsal.** Verify the evaluator's
  scoring contract, private reference directory, second laptop connectivity,
  and browser rendering. Saved joblib files are trusted local artifacts; do
  not load user-supplied replacements.

## Evidence files

- Baseline: [config](results/config.json), [original metrics](results/metrics.json),
  and [independent rerun metrics](results_recheck/metrics.json).
- Improved: [inference config](results_improved/config.json),
  [selection](results_improved/selection.json),
  [environment](results_improved/environment.json),
  [folds](results_improved/folds.json),
  [experiment registry](results_improved/experiments.json),
  [TPE trials](results_improved/tuning_trials.csv), and
  [comparison table](results_improved/comparison.csv).
- Per-trip input to reconciliation:
  [original pipeline](results_improved/original_pipeline_student_trips.csv),
  [improved pipeline](results_improved/decoder_motion_gap_6_student_trips.csv),
  and [paired deltas](results_improved/paired_student_deltas.json).
- Audit-time SHA-256, original `model.joblib`:
  `bf4548911498e3f6f9d70aa35bcf75e0a098d08755cf537695785646d28701bc`.
  Improved `model.joblib`:
  `663d395334affa327ff557e79597e2ef0bb5ad4b560978d76525e12c40a4d6a4`.
  Improved `config.json`:
  `7fa5bdc19fd6e822e5423ab713d4117deab6066784a7230b1182cbd04a856e99`.
