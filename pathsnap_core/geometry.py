"""Local projection and directed road adjacency."""
import numpy as np


def project(coords, origin):
    return (coords - origin) * [111320 * np.cos(np.deg2rad(origin[1])), 110574]


def segment_distance(points, starts, vectors):
    offset = points[:, None] - starts
    fraction = np.clip(
        (offset * vectors).sum(axis=-1) / np.maximum((vectors**2).sum(axis=-1), 1e-9),
        0, 1,
    )
    return np.linalg.norm(offset - fraction[..., None] * vectors, axis=-1)


def connected(arcs, before, after):
    return (before == after) | (arcs[before, 1] == arcs[after, 0])
