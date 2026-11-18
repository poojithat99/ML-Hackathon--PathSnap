"""Student reference to nearest-route point proxies for offline evaluation."""
from pathlib import Path

import numpy as np

from .geometry import project, segment_distance


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
