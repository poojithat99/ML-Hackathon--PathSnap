import json
import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pathsnap import (
    DATA, CandidateConfig, DecoderConfig, decode, load_trip, probabilities, reference_labels,
)


ROOT = Path(__file__).resolve().parent
TRIPS = Path(os.environ.get("PATHSNAP_TRIPS", DATA / "student_test"))
REFERENCES = Path(os.environ.get("PATHSNAP_REFERENCES", TRIPS))
ARTIFACTS = Path(os.environ.get("PATHSNAP_ARTIFACTS", ROOT / "results"))
app = FastAPI(title="PathSnap")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])


@lru_cache(maxsize=1)
def get_model():
    return joblib.load(ARTIFACTS / "model.joblib")


@lru_cache(maxsize=1)
def inference_config():
    config = json.loads((ARTIFACTS / "config.json").read_text())
    if isinstance(config["candidates"], dict):
        return (CandidateConfig(**config["candidates"]), config["features"],
                DecoderConfig(**config["decoder"]))
    return (CandidateConfig(count=config["candidates"]), "baseline",
            DecoderConfig(config["gap_seconds"], config["jump_penalty"]))


@app.get("/trips")
def list_trips():
    return sorted(folder.name for folder in TRIPS.iterdir()
                  if folder.is_dir() and folder.name.isascii() and folder.name.isdecimal())


@app.get("/predict/{trip_id}")
def predict(trip_id: str):
    if trip_id not in list_trips():
        raise HTTPException(status_code=404, detail="Unknown trip")
    started = perf_counter()
    candidates, features, decoder = inference_config()
    trip = load_trip(TRIPS / trip_id, candidates, features)
    arcs = decode(trip, probabilities(get_model(), trip), decoder)
    elapsed = perf_counter() - started
    reference_dir = REFERENCES / trip_id
    reference = reference_dir / f"{trip_id}.route"
    accuracy = (float(np.mean(arcs == reference_labels(reference_dir, trip)))
                if reference.is_file() else None)
    return dict(trip_id=trip_id, gps=trip.track[:, :2].tolist(),
                predicted_arc_ids=arcs.tolist(),
                segments=trip.nodes[trip.arcs[arcs]].tolist(),
                accuracy=accuracy, inference_seconds=elapsed)
