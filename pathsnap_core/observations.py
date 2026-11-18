"""Load and validate a GPS trip and calculate inference-time features."""
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .candidates import CandidateConfig, candidate_search
from .geometry import connected, project, segment_distance

DATA = Path(os.environ.get("PATHSNAP_DATA", Path(__file__).resolve().parent.parent / "Student folder")).resolve()
FEATURES = ("distance_m", "heading_agreement", "segment_length_m", "gps_speed_mps")
GAP_SECONDS = 15
FEATURE_GROUPS = {
    "baseline": FEATURES,
    "heading": FEATURES + ("forward_heading", "smoothed_heading"),
    "temporal": FEATURES + ("previous_support", "next_support"),
    "directional": FEATURES + ("directed_fraction", "forward_progress"),
}
FEATURE_GROUPS["combined"] = tuple(dict.fromkeys(
    feature for group in FEATURE_GROUPS.values() for feature in group))


FEATURE_GROUPS["geometry"] = FEATURE_GROUPS["heading"] + (
    "log_distance", "distance_above_nearest", "endpoint_clearance", "local_support",
)


@dataclass
class Trip:
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    candidates: np.ndarray
    features: np.ndarray
    eligible: np.ndarray | None = None
    candidate_seconds: float = 0.0


def load_trip(folder, candidate_config=CandidateConfig(), feature_group="baseline"):
    folder = Path(folder)
    prefix = folder / folder.name
    track = np.loadtxt(prefix.with_suffix(".track"), ndmin=2)
    nodes = np.loadtxt(prefix.with_suffix(".nodes"), ndmin=2)
    arcs = np.loadtxt(prefix.with_suffix(".arcs"), dtype=int, ndmin=2)
    if (track.shape[1] < 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2
            or not len(track) or not len(nodes) or not len(arcs)):
        raise ValueError(f"{folder.name}: invalid track or graph dimensions")
    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError(f"{folder.name}: non-finite coordinates or timestamps")
    if np.any(np.diff(track[:, 2]) < 0) or arcs.min() < 0 or arcs.max() >= len(nodes):
        raise ValueError(f"{folder.name}: invalid timestamp order or node index")
    origin = track[0, :2]
    points = project(track[:, :2], origin)
    network = project(nodes, origin)
    starts, ends = network[arcs].transpose(1, 0, 2)
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    started = perf_counter()
    candidates, distances, eligible = candidate_search(
        points, starts, ends, lengths, candidate_config)
    candidate_seconds = perf_counter() - started
    steps = np.diff(points, axis=0, prepend=points[:1])
    elapsed = np.diff(track[:, 2], prepend=track[0, 2])
    step_lengths = np.linalg.norm(steps, axis=1)
    agreement = (steps[:, None] * vectors[candidates]).sum(axis=-1)
    agreement /= np.maximum(step_lengths[:, None] * lengths[candidates], 1e-9)
    agreement[(step_lengths < 3) | (elapsed > GAP_SECONDS)] = 0
    speed = step_lengths / np.maximum(elapsed, 1)
    speed[elapsed > GAP_SECONDS] = 0
    features = dict(zip(FEATURES, (distances, agreement, lengths[candidates],
                                   np.broadcast_to(speed[:, None], candidates.shape))))
    if feature_group in {"heading", "combined", "geometry"}:
        good = (elapsed > 0) & (elapsed <= GAP_SECONDS)
        clean = np.where(good[:, None], steps, 0)
        forward = np.vstack((clean[1:], np.zeros((1, 2))))
        backward = np.vstack((np.zeros((1, 2)), clean[:-1]))
        next_forward = np.vstack((forward[1:], np.zeros((1, 2))))
        smooth = clean + forward + backward * good[:, None]
        smooth += next_forward * np.r_[good[1:], False][:, None]
        for name, movement in (("forward_heading", forward), ("smoothed_heading", smooth)):
            magnitude = np.linalg.norm(movement, axis=1)
            alignment = (movement[:, None] * vectors[candidates]).sum(axis=-1)
            alignment /= np.maximum(magnitude[:, None] * lengths[candidates], 1e-9)
            alignment[magnitude < 3] = 0
            features[name] = alignment
    if feature_group in {"temporal", "combined"}:
        links = connected(arcs, candidates[:-1, :, None], candidates[1:, None, :])
        links &= eligible[:-1, :, None] & eligible[1:, None, :]
        links &= (elapsed[1:] <= GAP_SECONDS)[:, None, None]
        support = np.exp(-distances / 20)
        zero = np.zeros_like(support[:1])
        features["previous_support"] = np.concatenate(
            (zero, (links * support[:-1, :, None]).max(axis=1)))
        features["next_support"] = np.concatenate(
            ((links * support[1:, None, :]).max(axis=2), zero))
    if feature_group in {"directional", "combined"}:
        projection = ((points[:, None] - starts[candidates]) * vectors[candidates]).sum(axis=-1)
        features["directed_fraction"] = np.clip(
            projection / np.maximum(lengths[candidates] ** 2, 1e-9), 0, 1)
        features["forward_progress"] = agreement * np.minimum(step_lengths[:, None], 50)
    if feature_group == "geometry":
        features["log_distance"] = np.log1p(distances)
        features["distance_above_nearest"] = distances - distances[:, :1]
        arc_vectors = vectors[candidates]
        along = ((points[:, None] - starts[candidates]) * arc_vectors).sum(axis=-1)
        along /= np.maximum(lengths[candidates], 1e-9)
        features["endpoint_clearance"] = np.clip(
            np.minimum(along, lengths[candidates] - along), -50, 50)
        # Neighbouring fixes support arcs that also explain the local trajectory.
        support = np.zeros_like(distances)
        count = np.zeros(len(points))
        for shift in (-2, -1, 1, 2):
            neighbours = np.clip(np.arange(len(points)) + shift, 0, len(points) - 1)
            valid = (neighbours != np.arange(len(points))) & (
                np.abs(track[neighbours, 2] - track[:, 2]) <= 6)
            nearby = segment_distance(points[neighbours], starts[candidates], arc_vectors)
            support += valid[:, None] * np.exp(-nearby / 12)
            count += valid
        features["local_support"] = support / np.maximum(count[:, None], 1)
    matrix = np.stack([features[name] for name in FEATURE_GROUPS[feature_group]], axis=-1)
    return Trip(track, nodes, arcs, candidates, matrix.astype(np.float32), eligible, candidate_seconds)
