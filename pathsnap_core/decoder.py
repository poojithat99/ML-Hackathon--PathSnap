"""Classifier emissions and optional batch sequence decoding."""
from dataclasses import dataclass

import numpy as np

from .geometry import connected, project, segment_distance
from .observations import GAP_SECONDS

JUMP_PENALTY = 4.0


@dataclass(frozen=True)
class DecoderConfig:
    gap_seconds: float = GAP_SECONDS
    jump_penalty: float = JUMP_PENALTY
    emission_weight: float = 1.0
    motion_weight: float = 0.0
    endpoint_scale_m: float = 0.0
    refine_predicted_route: bool = False


def probabilities(model, trip):
    values = model.predict_proba(trip.features.reshape(-1, trip.features.shape[-1]))[:, 1]
    values = np.clip(values.reshape(trip.candidates.shape), 1e-7, 1 - 1e-7)
    return values if trip.eligible is None else np.where(trip.eligible, values, 0)


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
    if config.endpoint_scale_m:
        network = project(trip.nodes, trip.track[0, :2])
        exit_points = network[trip.arcs[before, 1]]
        entry_points = network[trip.arcs[after, 0]]
        separation = np.linalg.norm(exit_points - entry_points, axis=-1)
        transitions -= (~links) * np.minimum(separation / config.endpoint_scale_m, 4)
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
    decoded = trip.candidates[np.arange(len(states)), states]
    if not config.refine_predicted_route:
        return decoded
    # Offline second pass over the predicted route. Ties follow first encounter.
    order = np.full(len(trip.arcs), len(decoded), dtype=int)
    np.minimum.at(order, decoded, np.arange(len(decoded)))
    ranked = np.argsort(order[trip.candidates], axis=1, kind="stable")
    candidates = np.take_along_axis(trip.candidates, ranked, axis=1)
    allowed = order[candidates] < len(decoded)
    if trip.eligible is not None:
        allowed &= np.take_along_axis(trip.eligible, ranked, axis=1)
    origin = trip.track[0, :2]
    network = project(trip.nodes, origin)
    start = network[trip.arcs[candidates, 0]]
    vector = network[trip.arcs[candidates, 1]] - start
    distances = segment_distance(project(trip.track[:, :2], origin), start, vector)
    nearest = np.where(allowed, distances, np.inf).argmin(axis=1)
    return candidates[np.arange(len(decoded)), nearest]
