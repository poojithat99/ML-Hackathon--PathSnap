# PathSnap — local starter

Kaggle is not required. Training and the evaluator UI run without internet after
dependencies are installed. A two-laptop demo requires a shared LAN or hotspot,
not an internet connection. The Folium report map normally loads browser assets
and tiles from the internet; the generated route PNG and demo UI do not.

## Run on this Mac

From the workspace root:

```sh
python3 train.py
PYTHONPATH=.deps python3 visualize_results.py
PYTHONPATH=.deps python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

This workspace already has the training dependencies and uses `.deps` for
FastAPI and Folium. On a fresh laptop that will train or host the API, create a
virtual environment and run `python3 -m pip install -r requirements.txt`
once, then omit `PYTHONPATH=.deps`. Use the same scikit-learn version as the
training run (1.7.2) when loading the saved model.

On laptop 2, copy the `ui/` directory and run:

```sh
python3 -m http.server 8080 --directory ui
```

Open `http://localhost:8080`. Enter `http://<laptop-1-LAN-IP>:8000`,
click **Load trips**, choose a trip, then click **Match route**. For a one-laptop
rehearsal, use the default API address. The API terminal logs requests.

## Data and scoring

- The default demo uses the 10 supplied student-test trips. Training uses
  80 training trips, with three trip-grouped folds and a fixed seed.
- Each observation uses up to 8 nearby directed segments. The spatial index
  shortlists 48 segment midpoints; exact point-to-segment distances then rank
  the shortlist. This may miss long segments. Distance, heading agreement,
  segment length and GPS speed are features.
- The provided reference contains a route sequence, not pointwise labels.
  This starter derives a proxy label by choosing the closest reference segment
  for each GPS point. Parallel roads, overlapping directions and loops can
  produce ambiguous labels because this alignment does not enforce route order.
- Viterbi uses log classifier probabilities and a penalty for arc pairs that
  neither stay on the same arc nor connect end-to-start. Gaps over 15 seconds
  restart decoding. The decoder does not insert intermediate routing arcs.
- Scores average across trips. Top-1 and macro F1 use proxy arc IDs; log loss
  is binary over candidate rows. Sequence similarity compares sequences after
  collapsing consecutive repeated arcs. Continuity excludes large GPS gaps.

For the final demo, get the reserved data and scoring convention from the
organizers. The API expects trip folders with matching numeric filenames.
Set `PATHSNAP_TRIPS` to the reserved trip directory and
`PATHSNAP_REFERENCES` to the private directory containing
`<trip-id>/<trip-id>.route` files. The backend reads those files only after
prediction; it returns aggregate reference agreement, never the labels. If
the organizers supply a different scoring format, adapt the scoring separately.
Do not include raw data or evaluator references in the public repository.

## Controlled improvement experiments

The baseline in `results/` and the original training command remain available.
See `EXPERIMENT_AUDIT.md` for the original pipeline analysis. To reproduce the
new experiments without overwriting the saved runs, use a fresh output folder:

```sh
python3 -u experiments.py --output results_repeat
python3 experiments.py --output results_repeat --benchmark-candidates
PYTHONPATH=.deps python3 visualize_results.py --improved results_repeat
python3 -m unittest -v test_pathsnap
```

The runner compares feature ablations, candidate counts/indices/radii, the
five original classifiers plus weighted logistic, random forest, and two
boosting variants. It saves TPE trials, decoder sweeps, experiment configs,
fold assignments, dependency versions, and per-trip metrics. All selection
uses the same three training folds. Decoder selection prefers configurations
with at least 98% CV continuity, then maximizes macro F1. Student testing
runs after selection and includes the original pipeline and stage winners;
it is a development comparison, not official evaluator scoring.

The candidate radius is a filter after the spatial shortlist. If no arc is
within the radius, the closest shortlisted arc remains available. Sampled
search augments midpoint indexing with interior segment samples. Whole-trip
heading features may use following observations, so the model is for batch
matching, not real-time streaming. Candidate log loss is binary over eligible
candidate rows and should not be compared directly across changed candidate sets.
Continuity always uses the same 15 s evaluation cutoff, regardless of decoder
gap settings. Inference time in the experiment logs covers scoring/decoding,
not data loading; the separate preparation time includes filesystem reads.

Use the improved artifact explicitly; the default API still serves the
original model. Restart the backend to switch artifacts:

```sh
PATHSNAP_ARTIFACTS=results_improved PYTHONPATH=.deps python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

See `results_improved/comparison.csv` for the baseline-to-final comparison,
`paired_student_deltas.json` for exploratory paired trip-bootstrap intervals,
and `development_map.html` in that folder for baseline/selected route layers.
The ten-trip bootstrap intervals are descriptive and do not account for all
model-selection uncertainty or the approximate labels.

## Report and submission checklist

- Use `results/dataset_inventory.csv` and add your own EDA observations and
  data-quality analysis.
- Use `results/metrics.json`, per-trip CSVs and `results/model_comparison.png`
  for the five model comparisons. Explain what the actual results show.
- Use `results/tuning_trials.csv` and the tuned/untuned and Viterbi comparisons.
- Open `results/development_map.html` for the required Folium screenshot.
  `results/development_route.png` is an offline preview, not a Folium screenshot.
- Capture the working UI and receiving laptop's API terminal. Test the real
  laptop-to-laptop connection before evaluation.
- Submit one public repository including source, logs, config and model, plus
  one consolidated PDF written in your own words. Disclose assistance as
  required by the event; report the proxy-label limitation explicitly.
