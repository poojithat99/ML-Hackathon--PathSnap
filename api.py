import json
import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pathsnap import (
    DATA, GAP_SECONDS, CandidateConfig, DecoderConfig, connected, decode, load_trip,
    probabilities, reference_labels,
)


ROOT = Path(__file__).resolve().parent
TRIPS = Path(os.environ.get("PATHSNAP_TRIPS", DATA / "student_test"))
ARTIFACTS = Path(os.environ.get("PATHSNAP_ARTIFACTS", ROOT / "results"))
BUNDLES = {"original": ROOT / "results", "improved": ROOT / "results_improved",
           "best": ROOT / "results_final", "agreement": ROOT / "results_final" / "agreement"}
# Only the known student development split may be scored after inference.
STUDENT_SCORING = TRIPS.resolve() == (DATA / "student_test").resolve()
app = FastAPI(title="PathSnap")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("PATHSNAP_CORS_ORIGINS", "").split(",")
                   if origin.strip()],
    allow_origin_regex=r"http://(?:localhost|127\.0\.0\.1|\[::1\])(?::[0-9]+)?",
    allow_methods=["GET"],
)


@lru_cache(maxsize=2)
def get_model(artifacts=ARTIFACTS):
    return joblib.load(artifacts / "model.joblib")


@lru_cache(maxsize=2)
def inference_config(artifacts=ARTIFACTS):
    config = json.loads((artifacts / "config.json").read_text())
    if isinstance(config["candidates"], dict):
        return (CandidateConfig(**config["candidates"]), config["features"],
                DecoderConfig(**config["decoder"]))
    return (CandidateConfig(count=config["candidates"]), "baseline",
            DecoderConfig(config["gap_seconds"], config["jump_penalty"]))


@app.get("/trips")
def list_trips():
    if not TRIPS.is_dir():
        raise HTTPException(status_code=503, detail="Trip directory is unavailable")
    return sorted(folder.name for folder in TRIPS.iterdir()
                  if folder.is_dir() and folder.name.isascii() and folder.name.isdecimal())


@app.get("/predict/{trip_id}")
def predict(
    trip_id: str,
    model: Literal["original", "improved", "best", "agreement"] | None = None,
    mode: Literal["classifier", "viterbi"] = "viterbi",
):
    if trip_id not in list_trips():
        raise HTTPException(status_code=404, detail="Unknown trip")
    started = perf_counter()
    artifacts = BUNDLES[model] if model else ARTIFACTS
    if not all((artifacts / name).is_file() for name in ("config.json", "model.joblib")):
        raise HTTPException(status_code=503, detail="Selected model bundle is not installed")
    candidates, features, decoder = inference_config(artifacts)
    try:
        trip = load_trip(TRIPS / trip_id, candidates, features)
    except (ValueError, IndexError, OSError) as error:
        raise HTTPException(status_code=422, detail="Invalid trip inputs") from error
    values = probabilities(get_model(artifacts), trip)
    arcs = (decode(trip, values, decoder) if mode == "viterbi"
            else trip.candidates[np.arange(len(values)), values.argmax(axis=1)])
    elapsed = perf_counter() - started
    accuracy = macro_f1 = None
    if STUDENT_SCORING:
        reference_dir = TRIPS / trip_id
        if (reference_dir / f"{trip_id}.route").is_file():
            labels = reference_labels(reference_dir, trip)
            accuracy = float(np.mean(arcs == labels))
            _, inverse = np.unique(np.concatenate((labels, arcs)), return_inverse=True)
            observed, predicted = np.split(inverse, 2)
            support = np.bincount(inverse)
            correct = np.bincount(observed[observed == predicted], minlength=len(support))
            macro_f1 = float(np.mean(2 * correct / support))
    links = connected(trip.arcs, arcs[:-1], arcs[1:])
    links = links[np.diff(trip.track[:, 2]) <= GAP_SECONDS]
    bundle_name = next((name for name, path in BUNDLES.items()
                        if path.resolve() == artifacts.resolve()), "custom")
    return dict(trip_id=trip_id, gps=trip.track[:, :2].tolist(),
                predicted_arc_ids=arcs.tolist(),
                segments=trip.nodes[trip.arcs[arcs]].tolist(),
                accuracy=accuracy, inference_seconds=elapsed,
                model=bundle_name, mode=mode, macro_f1=macro_f1,
                route_continuity=float(links.mean()) if links.size else None,
                gps_count=len(trip.track),
                candidate_arc_count=int(np.unique(trip.candidates[trip.eligible]).size),
                matched_arc_count=int(np.unique(arcs).size))
