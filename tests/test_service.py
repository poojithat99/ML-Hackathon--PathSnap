import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import api


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.folder = root / "01"
        self.folder.mkdir()
        prefix = self.folder / "01"
        prefix.with_suffix(".nodes").write_text("0 0\n0.001 0\n")
        prefix.with_suffix(".arcs").write_text("0 1\n")
        prefix.with_suffix(".track").write_text("0 0 0\n0.0001 0 1\n")
        # Fail if any inference request attempts to parse a route.
        prefix.with_suffix(".route").write_text("forbidden")
        for guard in (patch.object(api, "TRIPS", root),
                      patch.object(api, "STUDENT_SCORING", False),
                      patch.object(api, "reference_labels", side_effect=AssertionError)):
            guard.start()
            self.addCleanup(guard.stop)

    def test_modes_and_bundles_without_reference(self):
        self.assertEqual(self.client.get("/trips").json(), ["01"])
        for model in api.BUNDLES:
            for mode in ("classifier", "viterbi"):
                with self.subTest(model=model, mode=mode):
                    result = self.client.get("/predict/01", params=dict(model=model, mode=mode))
                    self.assertEqual(result.status_code, 200, result.text)
                    self.assertEqual(result.json()["predicted_arc_ids"], [0, 0])
                    self.assertIsNone(result.json()["accuracy"])

    def test_bad_requests(self):
        self.assertEqual(self.client.get("/predict/02").status_code, 404)
        self.assertEqual(self.client.get("/predict/01?mode=unknown").status_code, 422)
        (self.folder / "01.track").write_text("bad")
        self.assertEqual(self.client.get("/predict/01").status_code, 422)

    def test_origin(self):
        result = self.client.options("/trips", headers={"Origin": "http://localhost:8080",
                                     "Access-Control-Request-Method": "GET"})
        self.assertEqual(result.headers["access-control-allow-origin"], "http://localhost:8080")


@unittest.skipUnless(os.environ.get("PATHSNAP_TEST_STUDENT") == "1", "requires student split")
class StudentRegression(unittest.TestCase):
    def test_saved_bundles_on_all_student_trips(self):
        client = TestClient(api.app)
        files = {"original": "results_improved/original_pipeline_student_trips.csv",
                 "improved": "results_improved/decoder_motion_gap_6_student_trips.csv",
                 "best": "results_final/student_trips.csv",
                 "agreement": "results_final/agreement/student_trips.csv"}
        for model, file in files.items():
            with (api.ROOT / file).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 10)
            for row in rows:
                with self.subTest(model=model, trip=row["trip_id"]):
                    response = client.get("/predict/" + row["trip_id"], params={"model": model})
                    self.assertEqual(response.status_code, 200)
                    self.assertAlmostEqual(response.json()["accuracy"], float(row["top1"]))
                    self.assertAlmostEqual(response.json()["macro_f1"], float(row["macro_f1"]))
