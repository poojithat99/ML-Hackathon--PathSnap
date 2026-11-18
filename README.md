# PathSnap

PathSnap reconstructs a vehicle’s path on a directed road network from noisy, timestamped GPS fixes. Around intersections, a single fix can lie equally close to several roads or both directions of the same road. PathSnap ranks nearby arcs and uses the surrounding trajectory to resolve ambiguity.

## Results

Each student-test number averages ten trips equally. Point agreement is the fraction of fixes assigned the same directed arc as an approximate reference-derived point label. These are development metrics. Official evaluator performance is unknown.

| Pipeline | Grouped CV Macro F1 | Student agreement | Student Macro F1 | Student continuity |
|---|---:|---:|---:|---:|
| Original logistic + Viterbi | .5967 | 78.65% | .6278 | 99.60% |
| Improved logistic + motion-aware Viterbi | .7537 | 85.70% | .7575 | 99.55% |
| Random forest + endpoint penalty | .8222 | 88.84% | .8335 | 98.41% |
| Logistic + predicted-route refinement | .9070 | 94.28% | .9263 | 94.46% |

Choose the forest for the strongest agreement among configurations meeting the grouped CV continuity floor of 98%. Choose the refined route for point-proxy agreement. Its second pass can create disconnected transitions. [Full comparison and limitations](reports/final_results.md)

## Pipeline

1. Validate the timestamps and directed graph. Project longitude and latitude into local metres around the first observation.
2. Retrieve nearby road midpoints and interior samples using spatial indexes. Rerank the shortlist by distance to the finite segment. The improved and both research models keep up to 16 eligible candidates within 35 m and retain the closest when the radius is empty.
3. Score each candidate from distance, directed heading alignment, arc length and speed. The improved logistic model adds forward and smoothed alignment. The forest also uses transformed and relative distance, endpoint clearance and neighbouring-fix support.
4. Decode a sequence using log classifier probabilities, graph adjacency and motion consistency. The research decoder also penalizes disconnected transitions by endpoint separation. The agreement model then reassigns each fix to the nearest eligible arc visited on its own decoded path. All inference inputs come from the track and road graph.
5. FastAPI returns directed arc IDs, segment coordinates and per-trip metrics. The Leaflet dashboard displays GPS and both prediction modes. Development reference metrics are computed after prediction.

## Install and run

Use Python 3.14 and the commands below from the repository root. Use editable installation so the application can locate its bundled models and the checkout dashboard.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

Place the supplied student data under `Student folder/train/<id>/` and `Student folder/student_test/<id>/`, or export `PATHSNAP_DATA` pointing at the parent of both splits. Preserve zero-padded IDs. Each folder contains matching `.track` (lon, lat, seconds), `.nodes` (lon, lat) and `.arcs` (directed node indices) files. Training and student scoring also use the ordered `.route` arc indices. Supply only permitted data locally. To serve unlabeled inputs use `PATHSNAP_TRIPS`.

Start each server in a separate terminal in the checkout.

```sh
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

```sh
python -m http.server 8080 --bind 127.0.0.1 --directory ui
```

Open http://localhost:8080 and load trips. Bare API requests use Original unless `PATHSNAP_ARTIFACTS` points elsewhere. The dashboard initially selects Improved. The model selector also offers both research pipelines. Mode choices are classifier, Viterbi and a side-by-side metrics comparison. The refined model's Viterbi mode includes its second pass. For LAN origins, remote deployments and browser recovery see [demo instructions](DEMO_INSTRUCTIONS.md).

```sh
curl -fsS 'http://localhost:8000/trips'
curl -fsS 'http://localhost:8000/predict/00000007?model=agreement&mode=viterbi'
```

## Reproduce and inspect

```sh
python -m unittest discover -v
PATHSNAP_TEST_STUDENT=1 python -m unittest discover -v
python benchmark.py --seed 42 --output results_repeat
python analyze_errors.py --output reports
```

The benchmark trains, selects, exports both winners and evaluates them on student-test. Use a fresh output directory. For the historical runs use `python train.py --output results_recheck` and `python experiments.py --output historical_repeat` with fresh directories. Candidate preprocessing is deterministic per trip. Scalers and classifiers fit on fold-training trips only. See [reproduction notes](reports/final_results.md) for objectives, timing scope and lookahead.

| Location | Contents |
|---|---|
| `pathsnap_core/`, `pathsnap.py` | Geometry, candidate search, observations, sequential decoding, separate proxy scoring, compatibility imports |
| `train.py`, `experiments.py`, `benchmark.py` | Original training, historical staged search and final grouped comparisons |
| `results/`, `results_improved/` | Frozen original and improved bundles and evidence |
| `results_final/`, `results_final/agreement/` | Continuity-constrained and agreement-selected bundles, configurations and measurements |
| `api.py`, `ui/` | FastAPI service and web map |
| `tests/`, `test_pathsnap.py` | API, artifact, inference and ML regression checks |
| `reports/`, `analyze_errors.py`, `visualize_results.py` | Comparison, failure analysis and visualization |

Team PT-047 comprises Piyush S, Piyush G Nadgir, Poojitha T and Poorvi Rangaswamy. Contributor attribution is recorded in Git. Dataset redistribution rights and a project license have not been supplied.
