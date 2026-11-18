"""Write one comparison table from the historical and research experiment logs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / "results_final/experiments.csv").open() as source:
        rows = list(csv.DictReader(source))
    historical = json.loads((ROOT / "results_improved/experiments.json").read_text())
    for name, run in historical.items():
        rows.append(dict(experiment=name, model=run["model"], features=run["features"],
                         candidates=json.dumps(run["candidates"]), decoder=json.dumps(run["decoder"]),
                         parameters=json.dumps(run["parameters"]), seed=42,
                         validation="3-fold GroupKFold by trip", train_seconds=run["cv_fit_seconds"],
                         **run["cv"]))
    with (ROOT / "reports/all_experiments.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
