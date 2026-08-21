"""Tests for the ladder breakdowns.

These read artefacts already on disk. The property that matters most: the
breakdowns must not be presentable as significance claims, because each band
holds a fraction of the cohort and the frozen MDE was computed on all of it.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sailor.stage4 import ladder_analysis as LA  # noqa: E402


def project(rung="C0res_v2", rows=None):
    root = Path(tempfile.mkdtemp())
    ck = root / "11_CHECKPOINTS"
    ck.mkdir(parents=True)
    rows = rows or [
        {"subject": "sub-01", "model_log_ratio": 0.30, "pers_log_ratio": 0.40,
         "delta_days": 14.0},
        {"subject": "sub-02", "model_log_ratio": 0.60, "pers_log_ratio": 0.40,
         "delta_days": 60.0},
        {"subject": "sub-03", "model_log_ratio": 0.50, "pers_log_ratio": 0.50,
         "delta_days": 200.0},
    ]
    (ck / f"{rung}_r0f0_eval.json").write_text(json.dumps({"per_pair": rows}))
    return root


class TestBands(unittest.TestCase):

    def test_band_edges_match_amd_002(self):
        self.assertEqual(LA._band(14), "<=21d")
        self.assertEqual(LA._band(21), "<=21d")
        self.assertEqual(LA._band(22), "22-90d")
        self.assertEqual(LA._band(90), "22-90d")
        self.assertEqual(LA._band(91), ">90d")
        self.assertEqual(LA._band(None), "unknown")

    def test_bands_are_declared_frozen(self):
        r = LA.by_delta_band(project(), rungs=("C0res_v2",))
        self.assertIn("AMD-002", r["frozen_by"])

    def test_power_caveat_is_carried(self):
        r = LA.by_delta_band(project(), rungs=("C0res_v2",))
        self.assertIn("does NOT apply within a band", r["power_caveat"])
        self.assertIn("should be read as significant", r["power_caveat"])

    def test_each_pair_lands_in_exactly_one_band(self):
        r = LA.by_delta_band(project(), rungs=("C0res_v2",))
        total = sum(b["n_pairs"] for b in r["per_rung"]["C0res_v2"].values())
        self.assertEqual(total, 3)


class TestPerPatient(unittest.TestCase):

    def test_computes_model_minus_persistence(self):
        r = LA.per_patient(project(), rungs=("C0res_v2",))
        by = {s["subject"]: s["mean_across_rungs"] for s in r["per_patient"]}
        self.assertAlmostEqual(by["sub-01"], -0.10, places=6)   # model better
        self.assertAlmostEqual(by["sub-02"], +0.20, places=6)   # model worse
        self.assertAlmostEqual(by["sub-03"], 0.0, places=6)

    def test_counts_patients_the_model_helped(self):
        r = LA.per_patient(project(), rungs=("C0res_v2",))
        self.assertEqual(r["n_patients_where_model_beat_persistence"], 1)
        self.assertEqual(r["patients_helped"], ["sub-01"])

    def test_distinguishes_uniform_null_from_opposing_effects(self):
        r = LA.per_patient(project(), rungs=("C0res_v2",))
        self.assertIn("AVERAGE over opposing effects", r["reading"])

    def test_refuses_to_frame_itself_as_a_subgroup_search(self):
        r = LA.per_patient(project(), rungs=("C0res_v2",))
        self.assertIn("HYPOTHESIS for a future cohort", r["not_a_subgroup_search"])
        self.assertIn("does not transfer", r["not_a_subgroup_search"])

    def test_undefined_rows_are_skipped_not_zeroed(self):
        rows = [{"subject": "sub-01", "model_log_ratio": None,
                 "pers_log_ratio": 0.4, "delta_days": 14.0},
                {"subject": "sub-01", "model_log_ratio": 0.3,
                 "pers_log_ratio": 0.4, "delta_days": 14.0}]
        r = LA.per_patient(project(rows=rows), rungs=("C0res_v2",))
        self.assertAlmostEqual(r["per_patient"][0]["mean_across_rungs"],
                               -0.10, places=6)


class TestMissingRung(unittest.TestCase):

    def test_absent_rung_raises_rather_than_returning_empty(self):
        with self.assertRaises(FileNotFoundError):
            LA.load_rung(tempfile.mkdtemp(), "C9_v2")
