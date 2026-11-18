import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from pathsnap import DATA, FEATURES, GAP_SECONDS, JUMP_PENALTY, K
from pathsnap import connected, decode, load_trip, probabilities, reference_labels


def load_split(name):
    trips = []
    for folder in sorted((DATA / name).iterdir()):
        if folder.is_dir() and folder.name.isdecimal():
            trip = load_trip(folder)
            labels = reference_labels(folder, trip)
            trips.append((folder.name, trip, labels))
    print(f"Loaded {len(trips)} {name} trips", flush=True)
    return trips


def fit_data(trips, stride):
    features = np.concatenate([trip.features[::stride].reshape(-1, len(FEATURES))
                               for _, trip, _ in trips])
    labels = np.concatenate([(trip.candidates[::stride] == reference[::stride, None]).ravel()
                             for _, trip, reference in trips])
    return features, labels


def evaluate(model, trips, use_decoder=False):
    rows = []
    for trip_id, trip, reference in trips:
        start = perf_counter()
        values = probabilities(model, trip)
        predicted = (decode(trip, values) if use_decoder
                     else trip.candidates[np.arange(len(values)), values.argmax(axis=1)])
        elapsed = perf_counter() - start
        positive = trip.candidates == reference[:, None]
        valid_steps = np.diff(trip.track[:, 2]) <= GAP_SECONDS
        links = connected(trip.arcs, predicted[:-1], predicted[1:])[valid_steps]
        collapse = lambda route: route[np.r_[True, route[1:] != route[:-1]]]
        predicted_seq, reference_seq = collapse(predicted), collapse(reference)
        from difflib import SequenceMatcher

        rows.append(dict(
            trip_id=trip_id,
            top1=float(np.mean(predicted == reference)),
            macro_f1=f1_score(reference, predicted, average="macro", zero_division=0),
            candidate_log_loss=log_loss(positive.ravel(), values.ravel(), labels=[0, 1]),
            candidate_miss=float(np.mean(~positive.any(axis=1))),
            continuity=float(links.mean()) if links.size else None,
            sequence_similarity=SequenceMatcher(
                None, reference_seq, predicted_seq, autojunk=False
            ).ratio(),
            inference_seconds=elapsed,
        ))
    return rows


def averages(rows):
    return {key: float(np.mean(valid)) for key in rows[0] if key != "trip_id"
            if (valid := [row[key] for row in rows if row[key] is not None])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.trials < 1 or args.stride < 1:
        parser.error("--trials and --stride must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    train = load_split("train")
    test = load_split("student_test")
    with (args.output / "dataset_inventory.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "trip", "observations", "nodes", "arcs", "gaps_over_15s"))
        for split, trips in (("train", train), ("student_test", test)):
            for trip_id, trip, _ in trips:
                writer.writerow((split, trip_id, len(trip.track), len(trip.nodes), len(trip.arcs),
                                 np.sum(np.diff(trip.track[:, 2]) > GAP_SECONDS)))
    folds = [(fit_data([train[index] for index in fit], args.stride),
              [train[index] for index in valid])
             for fit, valid in GroupKFold(n_splits=3).split(train, groups=[row[0] for row in train])]
    full = fit_data(train, args.stride)
    factories = {
        "logistic": lambda strength=1: make_pipeline(
            StandardScaler(), LogisticRegression(C=strength, max_iter=150, random_state=42)),
        "naive_bayes": lambda smoothing=1e-9: GaussianNB(var_smoothing=smoothing),
        "tree": lambda depth=8: DecisionTreeClassifier(max_depth=depth, random_state=42),
        "extra_trees": lambda depth=10: ExtraTreesClassifier(
            n_estimators=24, max_depth=depth, n_jobs=2, random_state=42),
        "hist_boost": lambda leaves=15: HistGradientBoostingClassifier(
            max_iter=50, max_leaf_nodes=leaves, early_stopping=False, random_state=42),
    }
    results = {}

    def record(name, factory):
        started = perf_counter()
        fold_scores = cv(factory)
        model = factory().fit(*full)
        fitting_seconds = perf_counter() - started
        rows = evaluate(model, test)
        with (args.output / f"{name}_per_trip.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        results[name] = dict(cv_macro_f1=float(np.mean(fold_scores)), cv_folds=fold_scores,
                             cv_and_fit_seconds=fitting_seconds, student_test=averages(rows),
                             config=str(model))
        save_results()
        print(name, results[name]["cv_macro_f1"], results[name]["student_test"], flush=True)
        return model

    def cv(factory):
        return [averages(evaluate(factory().fit(*fit), valid))["macro_f1"] for fit, valid in folds]

    def save_results():
        (args.output / "metrics.json").write_text(json.dumps(results, indent=2))

    with threadpool_limits(limits=2):
        for name, factory in factories.items():
            record(name, factory)
        winner = max(factories, key=lambda name: results[name]["cv_macro_f1"])
        factory = factories[winner]

        def objective(trial):
            if winner == "logistic":
                value = trial.suggest_float("strength", 0.01, 10, log=True)
            elif winner == "naive_bayes":
                value = trial.suggest_float("smoothing", 1e-10, 1e-2, log=True)
            elif winner == "hist_boost":
                value = trial.suggest_int("leaves", 7, 35)
            else:
                value = trial.suggest_int("depth", 4, 18)
            return float(np.mean(cv(lambda: factory(value))))

        started = perf_counter()
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=2))
        study.optimize(objective, n_trials=args.trials)
        study.trials_dataframe().to_csv(args.output / "tuning_trials.csv", index=False)
        model = record("tuned", lambda: factory(**study.best_params))
        results["tuning"] = dict(model=winner, best_params=study.best_params,
                                 seconds=perf_counter() - started, seed=42, stride=args.stride)
        results["tuned_viterbi"] = averages(evaluate(model, test, use_decoder=True))
        save_results()
        joblib.dump(model, args.output / "model.joblib")
        (args.output / "config.json").write_text(json.dumps(
            dict(features=FEATURES, candidates=K, gap_seconds=GAP_SECONDS,
                 jump_penalty=JUMP_PENALTY), indent=2))


if __name__ == "__main__":
    main()
