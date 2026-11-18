import argparse
import csv
import json
import platform
from dataclasses import asdict, replace
from difflib import SequenceMatcher
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from pathsnap import (
    DATA, GAP_SECONDS, CandidateConfig, DecoderConfig, connected, decode,
    load_trip, probabilities, reference_labels,
)


SEED = 42
STRIDE = 5
DEFAULT_CANDIDATES = CandidateConfig()
DEFAULT_DECODER = DecoderConfig()
BASELINE_MODEL = {"C": 7.114476009343418}
PARAMETERS = {
    "logistic": {"C": 1},
    "naive_bayes": {},
    "tree": {"max_depth": 8},
    "extra_trees": {"n_estimators": 24, "max_depth": 10, "n_jobs": 2},
    "hist_boost": {"max_iter": 50, "max_leaf_nodes": 15, "early_stopping": False},
    "logistic_weighted": {"C": 1, "class_weight": "balanced"},
    "random_forest": {"n_estimators": 36, "max_depth": 13, "min_samples_leaf": 5, "n_jobs": 2},
    "hist_boost_extended": {"max_iter": 90, "max_leaf_nodes": 31, "early_stopping": False},
    "hist_boost_weighted": {"max_iter": 50, "max_leaf_nodes": 15, "early_stopping": False,
                            "class_weight": "balanced"},
}


def make_model(name, overrides=None):
    params = PARAMETERS[name] | (overrides or {})
    if name.startswith("logistic"):
        return make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=150, random_state=SEED, **params))
    if name == "naive_bayes":
        return GaussianNB(**params)
    estimator = {"tree": DecisionTreeClassifier, "extra_trees": ExtraTreesClassifier,
                 "random_forest": RandomForestClassifier}.get(name, HistGradientBoostingClassifier)
    return estimator(random_state=SEED, **params)


def fit_model(model, records):
    features = np.concatenate([trip.features[::STRIDE][trip.eligible[::STRIDE]]
                               for _, trip, _ in records])
    labels = np.concatenate([(trip.candidates[::STRIDE] == target[::STRIDE, None])[
        trip.eligible[::STRIDE]] for _, trip, target in records])
    started = perf_counter()
    model.fit(features, labels)
    return perf_counter() - started


def arc_metrics(reference, prediction):
    classes, inverse = np.unique(np.r_[reference, prediction], return_inverse=True)
    actual, predicted = np.split(inverse, 2)
    support = np.bincount(actual, minlength=len(classes))
    proposed = np.bincount(predicted, minlength=len(classes))
    correct = np.bincount(actual[actual == predicted], minlength=len(classes))
    f1 = 2 * correct / (support + proposed)
    accuracy = correct.sum() / len(reference)
    return dict(macro_f1=f1.mean(), weighted_f1=np.average(f1, weights=support),
                micro_f1=accuracy, top1=accuracy,
                macro_precision=(correct / np.maximum(proposed, 1)).mean(),
                macro_recall=(correct / np.maximum(support, 1)).mean())


def collapse(route):
    return route[np.r_[True, route[1:] != route[:-1]]]


def score(records, decoder=None):
    rows = []
    for trip_id, trip, reference, values, prediction_seconds in records:
        started = perf_counter()
        predicted = (decode(trip, values, decoder) if decoder is not None
                     else trip.candidates[np.arange(len(values)), values.argmax(axis=1)])
        elapsed = prediction_seconds + perf_counter() - started
        positive = (trip.candidates == reference[:, None]) & trip.eligible
        links = connected(trip.arcs, predicted[:-1], predicted[1:])
        links = links[np.diff(trip.track[:, 2]) <= GAP_SECONDS]
        active = values[trip.eligible]
        log_loss = -np.log(np.where(positive[trip.eligible], active, 1 - active)).mean()
        rows.append(dict(trip_id=trip_id, **arc_metrics(reference, predicted),
                         candidate_recall=positive.any(axis=1).mean(),
                         mean_candidates=trip.eligible.sum(axis=1).mean(),
                         candidate_log_loss=log_loss,
                         continuity=links.mean() if links.size else None,
                         sequence_similarity=SequenceMatcher(
                             None, collapse(reference), collapse(predicted), autojunk=False).ratio(),
                         inference_seconds=elapsed))
    return rows


def predict_records(model, records):
    for trip_id, trip, reference in records:
        started = perf_counter()
        values = probabilities(model, trip)
        yield trip_id, trip, reference, values, perf_counter() - started


def means(rows):
    return {key: float(np.mean(values)) for key in rows[0] if key not in {"trip_id", "fold"}
            if (values := [row[key] for row in rows if row[key] is not None])}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trip_dirs(split):
    return sorted(folder for folder in (DATA / split).iterdir()
                  if folder.is_dir() and folder.name.isdecimal())


def benchmark_candidates(output):
    experiments = json.loads((output / "experiments.json").read_text())
    selection = json.loads((output / "selection.json").read_text())["stages"]
    folder_list = trip_dirs("train")
    selected = [selection["best_features"]] + [
        identifier for identifier, row in experiments.items() if row["stage"] == "candidate"]
    rows = []
    for identifier in selected:
        experiment = experiments[identifier]
        config = CandidateConfig(**experiment["candidates"])
        runtimes = [load_trip(folder, config, experiment["features"]).candidate_seconds
                    for folder in folder_list]
        rows.append(dict(experiment=identifier, **asdict(config),
                         mean_search_seconds=float(np.mean(runtimes)),
                         mean_candidates=experiment["cv"]["mean_candidates"],
                         candidate_recall=experiment["cv"]["candidate_recall"],
                         cv_macro_f1=experiment["cv"]["macro_f1"]))
    write_csv(output / "candidate_benchmark.csv", rows)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    folders = trip_dirs("train")
    groups = [folder.name for folder in folders]
    folds = list(GroupKFold(n_splits=3).split(folders, groups=groups))
    (output / "folds.json").write_text(json.dumps(
        [[groups[index] for index in val] for _, val in folds], indent=2))
    (output / "environment.json").write_text(json.dumps(
        dict(python=platform.python_version(), seed=SEED, stride=STRIDE,
             packages={name: version(name) for name in
                       ("numpy", "scipy", "scikit-learn", "optuna", "joblib")}), indent=2))
    labels = {}
    results = {}

    @lru_cache(maxsize=2)
    def dataset(features, candidates):
        records = []
        durations = []
        for folder in folders:
            started = perf_counter()
            trip = load_trip(folder, candidates, features)
            durations.append(perf_counter() - started)
            if folder not in labels:
                labels[folder] = reference_labels(folder, trip)
            records.append((folder.name, trip, labels[folder]))
        return records, float(np.mean(durations))

    def cross_validate(features, candidates, name, overrides):
        records, preparation = dataset(features, candidates)
        predictions, duration = [], 0.0
        for fit, val in folds:
            model = make_model(name, overrides)
            duration += fit_model(model, [records[index] for index in fit])
            predictions.append(list(predict_records(model, [records[index] for index in val])))
        return predictions, duration, preparation

    def record(identifier, stage, name, features, candidates, overrides=None, decoder=None,
               cached=None):
        predictions, duration, preparation = (cached if cached is not None else
                                               cross_validate(features, candidates, name, overrides))
        by_fold = [score(fold, decoder) for fold in predictions]
        fold_means = [means(rows) for rows in by_fold]
        cv = {key: float(np.mean([item[key] for item in fold_means])) for key in fold_means[0]}
        write_csv(output / f"{identifier}_cv_trips.csv",
                  [dict(fold=fold, **row) for fold, rows in enumerate(by_fold) for row in rows])
        results[identifier] = dict(stage=stage, model=name, features=features,
                                   candidates=asdict(candidates), overrides=overrides or {},
                                   parameters=PARAMETERS[name] | (overrides or {}),
                                   decoder=asdict(decoder) if decoder is not None else None,
                                   cv=cv, folds=fold_means, cv_fit_seconds=duration,
                                   mean_load_and_features_seconds=preparation)
        (output / "experiments.json").write_text(json.dumps(results, indent=2))
        print(f"{identifier:28} F1 {cv['macro_f1']:.4f}  agreement {cv['top1']:.4f}"
              f"  continuity {cv['continuity']:.4f}  recall {cv['candidate_recall']:.4f}",
              flush=True)
        return identifier

    def best(ids):
        return max(ids, key=lambda identifier: tuple(results[identifier]["cv"][metric]
                                                    for metric in ("macro_f1", "top1", "continuity")))

    original = cross_validate("baseline", DEFAULT_CANDIDATES, "logistic", BASELINE_MODEL)
    record("original_classifier", "baseline", "logistic", "baseline", DEFAULT_CANDIDATES,
           BASELINE_MODEL, cached=original)
    record("original_pipeline", "baseline", "logistic", "baseline", DEFAULT_CANDIDATES,
           BASELINE_MODEL, DEFAULT_DECODER, original)
    del original
    feature_ids = ["original_classifier"] + [
        record(f"feature_{features}", "feature", "logistic", features, DEFAULT_CANDIDATES,
               BASELINE_MODEL) for features in ("heading", "temporal", "directional", "combined")]
    feature_best = best(feature_ids)
    features = results[feature_best]["features"]
    candidate_ids = [feature_best]
    for identifier, config in (
        ("midpoint_4", CandidateConfig(count=4)),
        ("midpoint_16", CandidateConfig(count=16)),
        ("sampled_8", CandidateConfig(index="sampled")),
        ("sampled_16", CandidateConfig(count=16, index="sampled")),
        ("sampled_16_35m", CandidateConfig(16, 35, "sampled")),
        ("sampled_16_75m", CandidateConfig(16, 75, "sampled")),
    ):
        candidate_ids.append(record(identifier, "candidate", "logistic", features,
                                    config, BASELINE_MODEL))
    candidate_best = best(candidate_ids)
    candidates = CandidateConfig(**results[candidate_best]["candidates"])
    model_ids = [candidate_best] + [
        record(f"model_{name}", "model", name, features, candidates) for name in PARAMETERS]
    model_best = best(model_ids)
    name = results[model_best]["model"]
    starting = results[model_best]["overrides"]

    def objective(trial):
        if name.startswith("logistic"):
            params = {"C": trial.suggest_float("C", 0.01, 30, log=True)}
        elif name.startswith("hist_boost"):
            params = {"max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 12, 48),
                      "l2_regularization": trial.suggest_float("l2_regularization", 0.01, 10, log=True)}
        elif name == "naive_bayes":
            params = {"var_smoothing": trial.suggest_float("var_smoothing", 1e-10, 1e-2, log=True)}
        else:
            params = {"max_depth": trial.suggest_int("max_depth", 5, 18),
                      "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 25)}
        identifier = record(f"trial_{trial.number}", "tuning", name, features, candidates,
                            starting | params)
        return results[identifier]["cv"]["macro_f1"]

    search_started = perf_counter()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
        seed=SEED, n_startup_trials=2))
    study.optimize(objective, n_trials=8)
    study.trials_dataframe().to_csv(output / "tuning_trials.csv", index=False)
    tuned = best([model_best] + [f"trial_{trial.number}" for trial in study.trials])
    overrides = results[tuned]["overrides"]
    search_seconds = perf_counter() - search_started
    cached = cross_validate(features, candidates, name, overrides)
    decoder_ids = []
    for identifier, config in (
        ("default", DEFAULT_DECODER),
        ("jump_2", replace(DEFAULT_DECODER, jump_penalty=2)),
        ("jump_6", replace(DEFAULT_DECODER, jump_penalty=6)),
        ("gap_6", replace(DEFAULT_DECODER, gap_seconds=6)),
        ("gap_30", replace(DEFAULT_DECODER, gap_seconds=30)),
        ("emission_075", replace(DEFAULT_DECODER, emission_weight=0.75)),
        ("motion_05", replace(DEFAULT_DECODER, motion_weight=0.5)),
        ("motion_15", replace(DEFAULT_DECODER, motion_weight=1.5)),
        ("motion_3", replace(DEFAULT_DECODER, motion_weight=3)),
        ("motion_jump_6", replace(DEFAULT_DECODER, motion_weight=1.5, jump_penalty=6)),
        ("motion_gap_6", replace(DEFAULT_DECODER, motion_weight=1.5, gap_seconds=6)),
    ):
        decoder_ids.append(record(f"decoder_{identifier}", "decoder", name, features,
                                  candidates, overrides, config, cached))
    eligible = [key for key in decoder_ids if results[key]["cv"]["continuity"] >= 0.98]
    decoder_best = best(eligible or decoder_ids)
    del cached
    stages = {"original_classifier": "original_classifier", "original_pipeline": "original_pipeline",
              "best_features": feature_best, "best_candidates": candidate_best,
              "best_classifier": model_best, "best_hyperparameters": tuned,
              "default_decoder": "decoder_default", "best_decoder": decoder_best,
              "final": decoder_best}
    selection = dict(stages=stages, decoder_cv_floor_met=bool(eligible),
                     tuning_seconds=search_seconds, selection="training CV only")
    (output / "selection.json").write_text(json.dumps(selection, indent=2))

    test_folders = trip_dirs("student_test")
    test_labels = {}
    fitted = {}
    for identifier in dict.fromkeys(stages.values()):
        experiment = results[identifier]
        name, features = experiment["model"], experiment["features"]
        candidates = CandidateConfig(**experiment["candidates"])
        overrides = experiment["overrides"]
        key = (name, features, candidates, json.dumps(overrides, sort_keys=True))
        if key not in fitted:
            model = make_model(name, overrides)
            seconds = fit_model(model, dataset(features, candidates)[0])
            test_records = []
            for folder in test_folders:
                trip = load_trip(folder, candidates, features)
                if folder not in test_labels:
                    test_labels[folder] = reference_labels(folder, trip)
                test_records.append((folder.name, trip, test_labels[folder]))
            fitted[key] = model, seconds, list(predict_records(model, test_records))
        model, seconds, predictions = fitted[key]
        decoder = DecoderConfig(**experiment["decoder"]) if experiment["decoder"] else None
        rows = score(predictions, decoder)
        experiment["student_test"] = means(rows)
        experiment["full_fit_seconds"] = seconds
        write_csv(output / f"{identifier}_student_trips.csv", rows)
        if identifier == decoder_best:
            joblib.dump(model, output / "model.joblib")
            (output / "config.json").write_text(json.dumps(
                {key: experiment[key] for key in ("model", "features", "candidates", "decoder")},
                indent=2))

    (output / "experiments.json").write_text(json.dumps(results, indent=2))
    comparison = [dict(milestone=stage, experiment=identifier,
                       cv_macro_f1=results[identifier]["cv"]["macro_f1"],
                       **{f"student_{key}": value for key, value in
                          results[identifier]["student_test"].items()})
                  for stage, identifier in stages.items()]
    write_csv(output / "comparison.csv", comparison)
    baseline_rows = list(csv.DictReader((output / "original_pipeline_student_trips.csv").open()))
    final_rows = list(csv.DictReader((output / f"{decoder_best}_student_trips.csv").open()))
    assert [row["trip_id"] for row in baseline_rows] == [row["trip_id"] for row in final_rows]
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(final_rows), size=(10000, len(final_rows)))
    paired = {}
    for metric in ("macro_f1", "top1", "continuity", "sequence_similarity"):
        delta = np.array([float(final[metric]) - float(base[metric])
                          for base, final in zip(baseline_rows, final_rows)])
        paired[metric] = dict(delta=float(delta.mean()), trips_improved=int((delta > 0).sum()),
                              bootstrap_95=np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist())
    (output / "paired_student_deltas.json").write_text(json.dumps(paired, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results_improved"))
    parser.add_argument("--benchmark-candidates", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.benchmark_candidates:
            benchmark_candidates(args.output)
        else:
            run(args.output)
