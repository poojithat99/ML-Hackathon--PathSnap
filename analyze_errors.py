"""Summarize incumbent errors on held-out training trips using observable geometry."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from benchmark import C, CURRENT, records, table
from experiments import fit_model, make_model, predict_records, trip_dirs
from pathsnap import CandidateConfig, decode, project


def summarize(trip, labels, values):
    prediction = decode(trip, values, CURRENT)
    wrong = prediction != labels
    points = project(trip.track[:, :2], trip.track[0, :2])
    steps = np.diff(points, axis=0, prepend=points[:1])
    lengths = np.linalg.norm(steps, axis=1)
    angle = np.ones(len(points))
    angle[1:] = ((steps[:-1] * steps[1:]).sum(axis=1)
                 / np.maximum(lengths[1:] * lengths[:-1], 1e-9))
    moving = (lengths >= 3) & (np.r_[0, lengths[:-1]] >= 3)
    arcs = trip.arcs
    eligible = trip.eligible
    # Degree uses unique undirected edges so reverse representations count once.
    edges = np.unique(np.sort(arcs, axis=1), axis=0)
    degree = np.bincount(edges.ravel(), minlength=len(trip.nodes))
    network = project(trip.nodes, trip.track[0, :2])
    vectors = network[arcs[:, 1]] - network[arcs[:, 0]]
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    alternatives = (trip.candidates != labels[:, None]) & eligible
    direction = (vectors[trip.candidates] * vectors[labels, None]).sum(axis=-1)
    masks = {
        "all": np.ones_like(wrong),
        "missing proxy candidate": ~((trip.candidates == labels[:, None]) & eligible).any(axis=1),
        "reverse directed arc chosen": (arcs[labels] == arcs[prediction, ::-1]).all(axis=1),
        "junction arc (degree >=3)": (degree[arcs[labels]] >= 3).any(axis=1),
        "competing parallel arc within 10m": (alternatives & (direction > .95)
                                              & (trip.features[:, :, 0] <= 10)).any(axis=1),
        "turn above 60 degrees": moving & (angle < .5),
        "turn above 150 degrees": moving & (angle < -.866),
        "GPS interval over 6s": np.r_[False, np.diff(trip.track[:, 2]) > 6],
        "nearest candidate over 20m": trip.features[:, 0, 0] > 20,
    }
    return {k: np.array([v.sum(), (wrong & v).sum()]) for k, v in masks.items()}


def run(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folders = trip_dirs("train")
    train = records(folders, "heading", CandidateConfig(16, 35, "sampled"))
    totals = {}
    for fit, val in GroupKFold(3).split(folders, groups=[p.name for p in folders]):
        model = make_model("logistic", {"C": C})
        fit_model(model, [train[i] for i in fit])
        for _, trip, labels, values, _ in predict_records(model, [train[i] for i in val]):
            for key, counts in summarize(trip, labels, values).items():
                totals[key] = totals.get(key, 0) + counts
    rows = [dict(condition=k, fixes=n, errors=e, error_rate=e / n if n else 0)
            for k, (n, e) in totals.items()]
    table(output / "error_conditions.csv", rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([r["condition"] for r in rows], [100*r["error_rate"] for r in rows])
    ax.set_xlabel("Proxy disagreement within each group (%)")
    ax.set_title("Incumbent on out-of-fold GPS observations")
    fig.tight_layout()
    fig.savefig(output / "error_conditions.png", dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    with threadpool_limits(limits=1):
        run(parser.parse_args().output)
