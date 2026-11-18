import argparse
import csv
import json
from pathlib import Path

import folium
import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from pathsnap import DATA, CandidateConfig, DecoderConfig, decode, load_trip, probabilities


def comparison_chart(results):
    metrics = json.loads((results / "metrics.json").read_text())
    names = [name for name, row in metrics.items() if "cv_macro_f1" in row]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    x = np.arange(len(names))
    for offset, label, values in (
        (-.25, "CV macro F1", [metrics[name]["cv_macro_f1"] for name in names]),
        (0, "Student-test macro F1", [metrics[name]["student_test"]["macro_f1"] for name in names]),
        (.25, "Student-test top-1", [metrics[name]["student_test"]["top1"] for name in names]),
    ):
        axes[0].bar(x + offset, values, .24, label=label)
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1)
    axes[1].bar(x, [metrics[name]["student_test"]["candidate_log_loss"] for name in names])
    axes[1].set_ylabel("Student-test binary candidate log loss")
    for axis in axes:
        axis.set_xticks(x, names, rotation=30, ha="right")
    figure.savefig(results / "model_comparison.png", dpi=150)
    plt.close(figure)


def development_map(results, trip_id, improved=None):
    folder = DATA / "student_test" / trip_id
    trip = load_trip(folder)
    model = joblib.load(results / "model.joblib")
    predicted = decode(trip, probabilities(model, trip))
    reference = np.loadtxt(folder / f"{trip_id}.route", dtype=int, ndmin=1)
    gps = trip.track[:, :2]
    map_view = folium.Map(location=gps[:, ::-1].mean(axis=0).tolist())
    figure, axis = plt.subplots(figsize=(9, 6), layout="constrained")
    layers = [
        ("Nearby roads", "#999999", np.unique(trip.candidates), 1),
        ("Reference route", "#cc5c13", np.unique(reference), 5),
        ("Baseline predicted route" if improved else "Predicted route",
         "#845ef7", np.unique(predicted), 3),
    ]
    if improved:
        config = json.loads((improved / "config.json").read_text())
        revised_trip = load_trip(folder, CandidateConfig(**config["candidates"]), config["features"])
        revised_model = joblib.load(improved / "model.joblib")
        revised = decode(revised_trip, probabilities(revised_model, revised_trip),
                         DecoderConfig(**config["decoder"]))
        layers.append(("Improved predicted route", "#087f5b", np.unique(revised), 2))
    for name, color, arc_ids, weight in layers:
        segments = trip.nodes[trip.arcs[arc_ids]]
        layer = folium.FeatureGroup(name=name).add_to(map_view)
        folium.PolyLine(segments[:, :, ::-1].tolist(), color=color, weight=weight).add_to(layer)
        from matplotlib.collections import LineCollection

        axis.add_collection(LineCollection(segments, colors=color, linewidths=weight,
                                           label=name, alpha=.65))
    gps_layer = folium.FeatureGroup(name="GPS track").add_to(map_view)
    folium.PolyLine(gps[:, ::-1].tolist(), color="#4263eb", weight=2).add_to(gps_layer)
    axis.scatter(*gps.T, s=5, color="#4263eb", label="GPS observations")
    axis.set(xlabel="Longitude", ylabel="Latitude", title=f"Student-test trip {trip_id}")
    axis.set_aspect(1 / np.cos(np.deg2rad(gps[:, 1].mean())))
    axis.autoscale()
    axis.legend()
    folium.LayerControl().add_to(map_view)
    map_view.fit_bounds([gps[:, ::-1].min(axis=0).tolist(), gps[:, ::-1].max(axis=0).tolist()])
    destination = improved or results
    map_view.save(str(destination / "development_map.html"))
    figure.savefig(destination / "development_route.png", dpi=150)
    plt.close(figure)


def experiment_charts(output):
    rows = json.loads((output / "experiments.json").read_text())
    selection = json.loads((output / "selection.json").read_text())["stages"]
    stages = {
        "feature_ablation": (["original_classifier"] + [key for key, row in rows.items()
                            if row["stage"] == "feature"], ("macro_f1", "top1")),
        "candidate_comparison": ([selection["best_features"]] + [key for key, row in rows.items()
                                if row["stage"] == "candidate"], ("candidate_recall", "macro_f1", "top1")),
        "classifier_comparison": ([key for key, row in rows.items() if row["stage"] == "model"],
                                  ("macro_f1", "top1", "continuity")),
        "decoder_sensitivity": ([key for key, row in rows.items() if row["stage"] == "decoder"],
                                ("macro_f1", "top1", "continuity")),
    }
    for title, (keys, metrics) in stages.items():
        figure, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5), layout="constrained")
        for axis, metric in zip(axes, metrics):
            axis.barh(keys[::-1], [rows[key]["cv"][metric] for key in reversed(keys)], color="#147d75")
            axis.set(xlim=(0, 1), xlabel=f"CV {metric.replace('_', ' ')}")
        figure.savefig(output / f"{title}.png", dpi=150)
        plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(12, 5), layout="constrained")
    data = {}
    for identifier in ("original_pipeline", selection["final"]):
        with (output / f"{identifier}_student_trips.csv").open() as handle:
            data[identifier] = list(csv.DictReader(handle))
    for axis, metric in zip(axes, ("macro_f1", "top1", "continuity")):
        for identifier, trips in data.items():
            axis.scatter([float(trip[metric]) for trip in trips],
                         [trip["trip_id"] for trip in trips],
                         label="Baseline" if identifier == "original_pipeline" else "Selected",
                         alpha=.8)
        axis.set(xlim=(0, 1.02), xlabel=f"Student-test {metric.replace('_', ' ')}")
    axes[0].legend()
    figure.savefig(output / "per_trip_comparison.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip", default="00000007", choices=sorted(
        folder.name for folder in (DATA / "student_test").iterdir() if folder.is_dir()))
    parser.add_argument("--improved", type=Path)
    args = parser.parse_args()
    results = Path("results")
    if args.improved:
        experiment_charts(args.improved)
    else:
        comparison_chart(results)
    development_map(results, args.trip, args.improved)
