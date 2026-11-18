"""Trip-grouped development benchmark. A new output directory is required."""
import argparse
import csv
import json
import platform
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from experiments import (SEED, STRIDE, fit_model, make_model, means, predict_records,
                         score, trip_dirs)
from pathsnap import CandidateConfig, DecoderConfig, decode, load_trip, reference_labels

ROOT = Path(__file__).resolve().parent
C = 0.011825176269334062
CURRENT = DecoderConfig(gap_seconds=6, motion_weight=1.5)
DECODERS = {"classifier": None, "motion": CURRENT,
            "endpoint": DecoderConfig(gap_seconds=6, motion_weight=1.5, endpoint_scale_m=15)}
DECODERS["route_refined"] = replace(DECODERS["endpoint"], refine_predicted_route=True)
# Set before validation. Keep the existing pipeline among the candidates.
RUNS = [
    ("baseline", 16, 35, "logistic", {"C": C}),
    ("heading", 16, 35, "logistic", {"C": C}),
    ("geometry", 16, 35, "logistic", {"C": C}),
    ("geometry", 16, 35, "logistic", {"C": 1}),
    ("geometry", 16, 35, "random_forest", {}),
    ("geometry", 16, 35, "extra_trees", {}),
    ("geometry", 16, 35, "hist_boost", {}),
    ("geometry", 20, 45, "logistic", {"C": C}),
]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def table(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, rows[0])
        writer.writeheader()
        writer.writerows(rows)


def records(folders, features, candidates):
    result = []
    for folder in folders:
        trip = load_trip(folder, candidates, features)
        result.append((folder.name, trip, reference_labels(folder, trip)))
    return result


def error_counts(record, decoder):
    _, trip, labels, values, _ = record
    predicted = decode(trip, values, decoder)
    wrong = predicted != labels
    true_arc = trip.arcs[labels]
    chosen = trip.arcs[predicted]
    present = ((trip.candidates == labels[:, None]) & trip.eligible).any(axis=1)
    reverse = (true_arc == chosen[:, ::-1]).all(axis=1)
    shared = (true_arc[:, :, None] == chosen[:, None, :]).any(axis=(1, 2))
    return {key: int((wrong & flag).sum()) for key, flag in {
        "missing_candidate": ~present, "reverse_arc": present & reverse,
        "shared_node": present & ~reverse & shared,
        "other_geometry": present & ~reverse & ~shared}.items()}


def run(out, seed=SEED):
    out.mkdir(parents=True, exist_ok=False)
    folders = trip_dirs("train")
    folds = list(GroupKFold(3).split(folders, groups=[f.name for f in folders]))
    dump(out / "folds.json", [[folders[i].name for i in val] for _, val in folds])
    dump(out / "environment.json", {"python": platform.python_version(), "stride": STRIDE,
         "seed": seed, "packages": {k: version(k) for k in
         ("numpy", "scipy", "scikit-learn", "joblib", "optuna", "threadpoolctl")}})
    results, configs, trip_rows, errors = [], {}, [], []
    last = None
    for index, (features, count, radius, model_name, params) in enumerate(RUNS):
        candidates = CandidateConfig(count, radius, "sampled")
        if last != (features, candidates):
            train = records(folders, features, candidates)
            last = features, candidates
        oof, fit_seconds, training_agreement = [], 0, []
        for fit, val in folds:
            estimator = make_model(model_name, params, seed)
            fit_records = [train[i] for i in fit]
            fit_seconds += fit_model(estimator, fit_records)
            training_agreement.extend(score(predict_records(estimator, fit_records)))
            oof.append(list(predict_records(estimator, [train[i] for i in val])))
        for name, decoder in DECODERS.items():
            ident = f"run_{index}_{name}"
            fold_metrics = []
            for fold, batch in enumerate(oof):
                scored_fold = score(batch, decoder)
                fold_metrics.append(means(scored_fold))
                for row in scored_fold:
                    trip_rows.append(dict(experiment=ident, fold=fold, **row))
            config = dict(model=model_name, parameters=estimator.get_params(deep=False)
                          if not model_name.startswith("logistic") else params,
                          overrides=params, features=features, candidates=asdict(candidates),
                          decoder=asdict(decoder) if decoder else None)
            configs[ident] = config
            results.append(dict(experiment=ident, model=model_name, features=features,
                                candidates=json.dumps(asdict(candidates)), decoder=json.dumps(config["decoder"]),
                                parameters=json.dumps(config["parameters"]), seed=seed, validation="3-fold GroupKFold by trip",
                                train_classifier_top1=means(training_agreement)["top1"],
                                train_seconds=fit_seconds, **means(fold_metrics)))
            if decoder and index == 1 and name == "motion":
                errors = [error_counts(r, decoder) for batch in oof for r in batch]
        print(f"Completed grouped comparison {index + 1}/{len(RUNS)}", flush=True)
        table(out / "experiments.csv", results)
        table(out / "cv_trips.csv", trip_rows)
    dump(out / "cv_errors.json", {key: sum(r[key] for r in errors) for key in errors[0]})
    # Selection uses equal-fold CV scores. Preserve Macro F1 at the incumbent level.
    floor = next(r for r in results if r["experiment"] == "run_1_motion")["macro_f1"]
    best = max((r for r in results if r["macro_f1"] >= floor and r["continuity"] >= 0.98),
               key=lambda r: (r["top1"], r["macro_f1"]))
    export_winner(out, folders, best, configs[best["experiment"]])
    alternate = max((r for r in results if r["macro_f1"] >= floor),
                    key=lambda r: (r["top1"], r["macro_f1"]))
    secondary = out / "agreement"
    secondary.mkdir()
    export_winner(secondary, folders, alternate, configs[alternate["experiment"]])


def export_winner(out, folders, selection, config):
    dump(out / "selection.json", selection)
    dump(out / "config.json", config)
    candidates = CandidateConfig(**config["candidates"])
    train = records(folders, config["features"], candidates)
    model = make_model(config["model"], config["overrides"], selection["seed"])
    final_seconds = fit_model(model, train)
    joblib.dump(model, out / "model.joblib", compress=3)
    student = records(trip_dirs("student_test"), config["features"], candidates)
    decoder = DecoderConfig(**config["decoder"]) if config["decoder"] else None
    scored = score(predict_records(model, student), decoder)
    table(out / "student_trips.csv", scored)
    dump(out / "student_summary.json", dict(final_fit_seconds=final_seconds, **means(scored)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.seed)
