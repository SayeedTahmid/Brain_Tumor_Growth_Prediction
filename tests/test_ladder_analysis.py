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


class TestDeltaDaysJoin(unittest.TestCase):
    """Rungs completed before v0.39 carry no Δt on the eval row. The join must
    report its own ambiguity rather than guessing."""

    def test_reports_collision_rate_and_refuses_ambiguous_keys(self):
        import numpy as np
        from sailor.stage4 import mask_cache as MC
        root = Path(tempfile.mkdtemp())
        ad = root / "01_DATA_FOUNDATION" / "v2_arrays"; ad.mkdir(parents=True)
        # Two pairs of one patient with IDENTICAL volumes -> ambiguous key.
        for ses, n in (("ses-01", 5), ("ses-02", 5), ("ses-03", 5)):
            a = np.zeros((60, 60, 60), dtype=np.float32)
            a[20:20 + n, 20:20 + n, 20:20 + n] = 1
            np.savez_compressed(
                ad / f"sub-01__{ses}__ContrastEnhancedMask-CL.npz", array=a)
        MC.build(root, verify=False, min_extent=32)
        split = {"pairs": {"pairs": [
            {"subject": "sub-01", "input_session": "ses-01",
             "target_session": "ses-02", "delta_days": 14.0},
            {"subject": "sub-01", "input_session": "ses-02",
             "target_session": "ses-03", "delta_days": 200.0}]}}
        rows = [{"subject": "sub-01", "n_input": 125, "n_target": 125}]
        info = LA.attach_delta_days(root, rows, split)
        self.assertGreaterEqual(info["n_ambiguous_keys"], 1)
        self.assertIsNone(rows[0]["delta_days"])     # not guessed
        self.assertIn("rather than being guessed", info["note"])

    def test_existing_delta_days_are_not_overwritten(self):
        import numpy as np
        from sailor.stage4 import mask_cache as MC
        root = Path(tempfile.mkdtemp())
        ad = root / "01_DATA_FOUNDATION" / "v2_arrays"; ad.mkdir(parents=True)
        for ses in ("ses-01", "ses-02"):
            a = np.zeros((60, 60, 60), dtype=np.float32); a[20:25, 20:25, 20:25] = 1
            np.savez_compressed(
                ad / f"sub-01__{ses}__ContrastEnhancedMask-CL.npz", array=a)
        MC.build(root, verify=False, min_extent=32)
        split = {"pairs": {"pairs": [{"subject": "sub-01",
                                      "input_session": "ses-01",
                                      "target_session": "ses-02",
                                      "delta_days": 14.0}]}}
        rows = [{"subject": "sub-01", "n_input": 125, "n_target": 125,
                 "delta_days": 99.0}]
        LA.attach_delta_days(root, rows, split)
        self.assertEqual(rows[0]["delta_days"], 99.0)
