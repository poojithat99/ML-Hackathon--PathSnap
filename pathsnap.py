from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree


DATA = Path(__file__).resolve().parent / "Student folder"
FEATURES = ("distance_m", "heading_agreement", "segment_length_m", "gps_speed_mps")
K = 8
GAP_SECONDS = 15
JUMP_PENALTY = 4.0
FEATURE_GROUPS = {
    "baseline": FEATURES,
    "heading": FEATURES + ("forward_heading", "smoothed_heading"),
    "temporal": FEATURES + ("previous_support", "next_support"),
    "directional": FEATURES + ("directed_fraction", "forward_progress"),
}
FEATURE_GROUPS["combined"] = tuple(dict.fromkeys(
    feature for group in FEATURE_GROUPS.values() for feature in group))


@dataclass(frozen=True)
class CandidateConfig:
    count: int = K
    radius_m: float | None = None
    index: str = "midpoint"

    def __post_init__(self):
        if self.count < 1 or self.index not in {"midpoint", "sampled"}:
            raise ValueError("Invalid candidate count or spatial index")
        if self.radius_m is not None and not 0 < self.radius_m < np.inf:
            raise ValueError("Candidate radius must be positive and finite")


@dataclass(frozen=True)
class DecoderConfig:
    gap_seconds: float = GAP_SECONDS
    jump_penalty: float = JUMP_PENALTY
    emission_weight: float = 1.0
    motion_weight: float = 0.0


def project(coords, origin):
    return (coords - origin) * [111320 * np.cos(np.deg2rad(origin[1])), 110574]


def segment_distance(points, starts, vectors):
    offset = points[:, None] - starts
    fraction = np.clip(
        (offset * vectors).sum(axis=-1) / np.maximum((vectors**2).sum(axis=-1), 1e-9),
        0, 1,
    )
    return np.linalg.norm(offset - fraction[..., None] * vectors, axis=-1)


@dataclass
class Trip:
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    candidates: np.ndarray
    features: np.ndarray
    eligible: np.ndarray | None = None
    candidate_seconds: float = 0.0


def candidate_search(points, starts, ends, lengths, config):
    vectors = ends - starts
    _, shortlist = cKDTree((starts + ends) / 2).query(
        points, k=min(max(48, config.count), len(starts)))
    shortlist = shortlist.reshape(len(points), -1)
    if config.index == "sampled":
        subdivisions = np.maximum(np.ceil(lengths / 35).astype(int), 1)
        owners = np.repeat(np.arange(len(starts)), subdivisions)
        first = np.repeat(np.cumsum(subdivisions) - subdivisions, subdivisions)
        position = (np.arange(len(owners)) - first + 0.5) / subdivisions[owners]
        samples = starts[owners] + position[:, None] * vectors[owners]
        _, neighbours = cKDTree(samples).query(points, k=min(48, len(samples)))
        shortlist = np.sort(np.column_stack((shortlist, owners[neighbours])), axis=1)
    distances = segment_distance(points, starts[shortlist], vectors[shortlist])
    if config.index == "sampled":
        distances[:, 1:][np.diff(shortlist, axis=1) == 0] = np.inf
    order = np.argsort(distances, axis=1)[:, :config.count]
    candidates = np.take_along_axis(shortlist, order, axis=1)
    distances = np.take_along_axis(distances, order, axis=1)
    eligible = np.isfinite(distances)
    if config.radius_m is not None:
        eligible &= distances <= config.radius_m
    eligible[:, 0] = True
    return candidates, np.where(eligible, distances, 0), eligible


def load_trip(folder, candidate_config=CandidateConfig(), feature_group="baseline"):
    folder = Path(folder)
    prefix = folder / folder.name
    track = np.loadtxt(prefix.with_suffix(".track"), ndmin=2)
    nodes = np.loadtxt(prefix.with_suffix(".nodes"), ndmin=2)
    arcs = np.loadtxt(prefix.with_suffix(".arcs"), dtype=int, ndmin=2)
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
    if feature_group in {"heading", "combined"}:
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
    matrix = np.stack([features[name] for name in FEATURE_GROUPS[feature_group]], axis=-1)
    return Trip(track, nodes, arcs, candidates, matrix.astype(np.float32), eligible, candidate_seconds)


def reference_labels(folder, trip):
    folder = Path(folder)
    route = np.loadtxt(folder / f"{folder.name}.route", dtype=int, ndmin=1)
    if not len(route) or route.min() < 0 or route.max() >= len(trip.arcs):
        raise ValueError(f"{folder.name}: invalid reference route")
    origin = trip.track[0, :2]
    points = project(trip.track[:, :2], origin)
    starts, ends = project(trip.nodes[trip.arcs[route]], origin).transpose(1, 0, 2)
    vectors = ends - starts
    return np.concatenate([
        route[segment_distance(points[offset:offset + 128], starts, vectors).argmin(axis=1)]
        for offset in range(0, len(points), 128)
    ])


def probabilities(model, trip):
    values = model.predict_proba(trip.features.reshape(-1, trip.features.shape[-1]))[:, 1]
    values = np.clip(values.reshape(trip.candidates.shape), 1e-7, 1 - 1e-7)
    return values if trip.eligible is None else np.where(trip.eligible, values, 0)


def connected(arcs, before, after):
    return (before == after) | (arcs[before, 1] == arcs[after, 0])


def decode(trip, values, config=DecoderConfig()):
    with np.errstate(divide="ignore"):
        emission = config.emission_weight * np.log(values)
    before = trip.candidates[:-1, :, None]
    after = trip.candidates[1:, None, :]
    links = connected(trip.arcs, before, after)
    transitions = np.where(links, 0.0, -config.jump_penalty)
    if config.motion_weight:
        points = project(trip.track[:, :2], trip.track[0, :2])
        endpoints = project(trip.nodes[trip.arcs[trip.candidates]], trip.track[0, :2])
        starts, ends = endpoints[:, :, 0], endpoints[:, :, 1]
        vectors = ends - starts
        lengths = np.linalg.norm(vectors, axis=-1)
        along = np.clip(((points[:, None] - starts) * vectors).sum(axis=-1)
                        / np.maximum(lengths, 1e-9), 0, lengths)
        displacement = np.linalg.norm(np.diff(points, axis=0), axis=1)
        route_step = np.where(
            before == after, along[1:, None, :] - along[:-1, :, None],
            (lengths[:-1] - along[:-1])[:, :, None] + along[1:, None, :])
        mismatch = np.minimum(np.abs(route_step - displacement[:, None, None]) / 25, 3)
        transitions -= config.motion_weight * mismatch * links
    scores = emission[0].copy()
    back = np.full(values.shape, -1, dtype=int)
    states = np.empty(len(values), dtype=int)
    for index in range(1, len(values)):
        if trip.track[index, 2] - trip.track[index - 1, 2] > config.gap_seconds:
            states[index - 1] = scores.argmax()
            scores = emission[index].copy()
            continue
        totals = scores[:, None] + transitions[index - 1]
        back[index] = totals.argmax(axis=0)
        scores = emission[index] + totals.max(axis=0)
    states[-1] = scores.argmax()
    for index in range(len(states) - 1, 0, -1):
        previous = back[index, states[index]]
        if previous >= 0:
            states[index - 1] = previous
    return trip.candidates[np.arange(len(states)), states]
