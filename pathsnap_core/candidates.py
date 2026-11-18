"""Spatial shortlist and exact candidate reranking."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .geometry import segment_distance

K = 8


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
        shortlist = np.sort(np.column_stack((shortlist, owners[neighbours.reshape(len(points), -1)])), axis=1)
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
