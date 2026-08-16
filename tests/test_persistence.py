"""Tests for the persistence baseline and the BINDING empty-target rule.

The rule exists because 5 of 208 pairs are empty->empty. The dangerous failure
is not a crash: it is a mean that silently describes 203 pairs while claiming
208.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from sailor.stage3 import persistence as P  # noqa: E402


def m(*coords, shape=(8, 8, 8)):
    a = np.zeros(shape, dtype=bool)
    for c in coords:
        a[c] = True
    return a


EMPTY = np.zeros((8, 8, 8), dtype=bool)


class TestEmptyTargetRule(unittest.TestCase):

    def test_dice_is_undefined_not_one_on_empty_empty(self):
        self.assertIsNone(P.dice(EMPTY, EMPTY))

    def test_dice_is_zero_when_only_one_side_empty(self):
        self.assertEqual(P.dice(EMPTY, m((1, 1, 1))), 0.0)
        self.assertEqual(P.dice(m((1, 1, 1)), EMPTY), 0.0)

    def test_volume_change_error_is_defined_at_zero(self):
        # empty -> empty: true change 0, predicted change 0, error 0.
        self.assertEqual(P.volume_change_error(EMPTY, EMPTY, EMPTY), 0)

    def test_change_region_dice_undefined_when_nothing_changed(self):
        a = m((1, 1, 1), (1, 1, 2))
        self.assertIsNone(P.change_region_dice(a, a, a))

    def test_undefined_pairs_are_counted_not_dropped(self):
        rows = [{"subject": "sub-01", "dice": 0.5},
                {"subject": "sub-01", "dice": None},
                {"subject": "sub-02", "dice": 0.7}]
        agg = P._aggregate(rows, "dice")
        self.assertEqual(agg["n_pairs_total"], 3)
        self.assertEqual(agg["n_pairs_defined"], 2)
        self.assertEqual(agg["n_pairs_undefined"], 1)
        self.assertIn("describes 2 pairs, not 3", agg["undefined_note"])

    def test_note_absent_when_nothing_undefined(self):
        rows = [{"subject": "sub-01", "dice": 0.5}, {"subject": "sub-02", "dice": 0.7}]
        self.assertIsNone(P._aggregate(rows, "dice")["undefined_note"])


class TestMetrics(unittest.TestCase):

    def test_dice_identical_is_one(self):
        a = m((1, 1, 1), (1, 1, 2))
        self.assertAlmostEqual(P.dice(a, a), 1.0)

    def test_dice_disjoint_is_zero(self):
        self.assertEqual(P.dice(m((1, 1, 1)), m((5, 5, 5))), 0.0)

    def test_volume_change_error_perfect_persistence(self):
        prev = m((1, 1, 1), (1, 1, 2))
        self.assertEqual(P.volume_change_error(prev, prev, prev), 0)

    def test_volume_change_error_counts_missed_growth(self):
        prev = m((1, 1, 1))
        ref = m((1, 1, 1), (1, 1, 2), (1, 1, 3))   # grew by 2
        self.assertEqual(P.volume_change_error(prev, ref, prev), 2)

    def test_change_region_dice_penalises_copy_forward_on_real_change(self):
        prev = m((1, 1, 1))
        ref = m((1, 1, 1), (1, 1, 2))
        # persistence predicts no change; a real change occurred -> 0, not ~1
        self.assertEqual(P.change_region_dice(prev, ref, prev), 0.0)

    def test_whole_mask_dice_is_near_ceiling_where_change_dice_is_zero(self):
        # AMD-005's motivation, as a test rather than an assertion in prose.
        prev = m(*[(1, 1, k) for k in range(6)])
        ref = m(*[(1, 1, k) for k in range(7)])
        self.assertGreater(P.dice(prev, ref), 0.9)
        self.assertEqual(P.change_region_dice(prev, ref, prev), 0.0)


class TestPatientLevelBootstrap(unittest.TestCase):

    def test_resamples_patients_not_pairs(self):
        # 2 patients, 50 pairs each. Pair-level CIs would be far too narrow.
        by = {"sub-01": [0.2] * 50, "sub-02": [0.8] * 50}
        r = P._patient_bootstrap(by, n=2000)
        self.assertEqual(r["n_patients"], 2)
        self.assertAlmostEqual(r["mean"], 0.5, places=6)
        self.assertLess(r["ci_low"], 0.35)     # wide, because n=2 patients
        self.assertGreater(r["ci_high"], 0.65)

    def test_deterministic_under_seed(self):
        by = {f"sub-{i:02d}": [i / 10.0] for i in range(1, 11)}
        a = P._patient_bootstrap(by, seed=1337, n=1000)
        b = P._patient_bootstrap(by, seed=1337, n=1000)
        self.assertEqual(a["ci_low"], b["ci_low"])


class TestBands(unittest.TestCase):

    def test_frozen_band_edges(self):
        self.assertEqual(P._band(14), "<=21d")
        self.assertEqual(P._band(21), "<=21d")
        self.assertEqual(P._band(22), "22-90d")
        self.assertEqual(P._band(90), "22-90d")
        self.assertEqual(P._band(91), ">90d")
        self.assertEqual(P._band(None), "unknown")


class TestAbsentIsNotEmpty(unittest.TestCase):

    def test_missing_file_returns_none_not_empty_array(self):
        self.assertIsNone(P.load_mask(tempfile.mkdtemp(), "sub-25", "ses-09"))

    def test_present_empty_file_returns_empty_array_not_none(self):
        d = Path(tempfile.mkdtemp())
        np.savez(d / "sub-25__ses-05__ContrastEnhancedMask-CL.npz",
                 array=np.zeros((4, 4, 4), dtype=np.float32))
        a = P.load_mask(d, "sub-25", "ses-05")
        self.assertIsNotNone(a)
        self.assertEqual(int(a.sum()), 0)

    def test_pair_with_missing_end_reports_error_not_a_score(self):
        d = Path(tempfile.mkdtemp())
        np.savez(d / "sub-25__ses-08__ContrastEnhancedMask-CL.npz",
                 array=np.zeros((4, 4, 4), dtype=np.float32))
        r = P.score_pair(d, {"subject": "sub-25", "input_session": "ses-08",
                             "target_session": "ses-09"})
        self.assertIn("error", r)
        self.assertNotIn("dice", r)


class TestMDE(unittest.TestCase):
    """MDE defaults to volume_change_error, NOT change-region Dice: persistence
    predicts no change, so its change-region Dice is structurally 0.0 or
    undefined and has no between-patient variance to compute an MDE from."""

    def _res(self, n_patients, sd=0.2):
        return {"overall": {"primary_volume_change_error": {
            "metric": "volume_change_error", "per_patient_sd": sd,
            "n_patients": n_patients}}}

    def test_default_metric_is_volume_change_error(self):
        self.assertEqual(P.minimum_detectable_effect(self._res(26))["metric"],
                         "volume_change_error")

    def test_mde_shrinks_with_more_patients(self):
        self.assertGreater(P.minimum_detectable_effect(self._res(10))["mde"],
                           P.minimum_detectable_effect(self._res(100))["mde"])

    def test_mde_is_labelled_an_approximation(self):
        self.assertIn("approximation_note",
                      P.minimum_detectable_effect(self._res(26)))

    def test_degenerate_baseline_yields_no_mde(self):
        # A constant-zero baseline has zero between-patient sd -> no MDE.
        r = P.minimum_detectable_effect(
            {"overall": {"primary_change_region_dice": {
                "metric": "change_region_dice", "per_patient_sd": None,
                "n_patients": 26}}}, metric="change_region_dice")
        self.assertIsNone(r["mde"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestScaleFreeHeadroomMetrics(unittest.TestCase):
    """Added after the baseline was seen but BEFORE any model exists. The reason
    is structural — absolute error is scale-dependent — not result-driven."""

    def big(self, n, shape=(40, 40, 40)):
        a = np.zeros(shape, dtype=bool)
        a.reshape(-1)[:n] = True
        return a

    def test_relative_error_is_scale_free(self):
        # Two patients, same PROPORTIONAL change, 100x different tumour size.
        small_prev, small_ref = self.big(100), self.big(200)
        big_prev, big_ref = self.big(10000), self.big(20000)
        rs = P.relative_volume_change_error(small_prev, small_ref, small_prev)
        rb = P.relative_volume_change_error(big_prev, big_ref, big_prev)
        self.assertAlmostEqual(rs, rb, places=6)
        # while the ABSOLUTE error differs by 100x — the problem being fixed
        self.assertEqual(P.volume_change_error(big_prev, big_ref, big_prev),
                         100 * P.volume_change_error(small_prev, small_ref, small_prev))

    def test_relative_error_bounded_at_one_for_persistence(self):
        prev, ref = self.big(500), EMPTY                     # total disappearance
        self.assertAlmostEqual(P.relative_volume_change_error(prev, ref, prev), 1.0)

    def test_relative_error_undefined_on_empty_empty(self):
        self.assertIsNone(P.relative_volume_change_error(EMPTY, EMPTY, EMPTY))

    def test_log_ratio_defined_on_empty_empty(self):
        self.assertEqual(P.log_volume_ratio_error(EMPTY, EMPTY, EMPTY), 0.0)

    def test_log_ratio_symmetric_in_growth_and_shrinkage(self):
        a, b = self.big(1000), self.big(2000)
        grow = P.log_volume_ratio_error(a, b, a)     # 1000 -> 2000
        shrink = P.log_volume_ratio_error(b, a, b)   # 2000 -> 1000
        self.assertAlmostEqual(grow, shrink, places=6)

    def test_log_ratio_zero_when_volume_preserved(self):
        a = self.big(750)
        self.assertAlmostEqual(P.log_volume_ratio_error(a, a, a), 0.0)


class TestGate3Preregistration(unittest.TestCase):
    from sailor.stage3 import headroom as H

    def _cmp(self):
        from sailor.stage3 import headroom as H
        res = {"n_patients": 26, "overall": {
            "primary_volume_change_error": {
                "metric": "volume_change_error", "mean": 4463.7,
                "ci_low": 3042.7, "ci_high": 6053.1,
                "per_patient_sd": 3994.8, "n_patients": 26,
                "n_pairs_defined": 208, "n_pairs_undefined": 0},
            "headroom_log_volume_ratio_error": {
                "metric": "log_volume_ratio_error", "mean": 0.8,
                "ci_low": 0.6, "ci_high": 1.0,
                "per_patient_sd": 0.5, "n_patients": 26,
                "n_pairs_defined": 208, "n_pairs_undefined": 0}}}
        return H.compare(res)

    def test_comparison_selects_nothing(self):
        c = self._cmp()
        self.assertIn("no_metric_selected", c)
        self.assertGreaterEqual(len(c["candidates"]), 2)

    def test_change_region_dice_is_excluded_with_a_reason(self):
        from sailor.stage3 import headroom as H
        self.assertIn("change_region_dice", H.EXCLUDED)
        self.assertNotIn("change_region_dice", H.CANDIDATES)

    def test_unknown_metric_rejected(self):
        from sailor.stage3 import headroom as H
        with self.assertRaises(ValueError):
            H.preregister(tempfile.mkdtemp(), "made_up", "x" * 50, self._cmp())

    def test_thin_justification_rejected(self):
        from sailor.stage3 import headroom as H
        with self.assertRaises(ValueError):
            H.preregister(tempfile.mkdtemp(), "volume_change_error", "because",
                          self._cmp())

    def test_second_registration_refused(self):
        from sailor.stage3 import headroom as H
        root = Path(tempfile.mkdtemp())
        j = "Chosen on structural grounds before any model exists; " * 2
        H.preregister(root, "volume_change_error", j, self._cmp())
        with self.assertRaises(RuntimeError):
            H.preregister(root, "log_volume_ratio_error", j, self._cmp())

    def test_deliberate_overwrite_preserves_the_prior_choice(self):
        from sailor.stage3 import headroom as H
        root = Path(tempfile.mkdtemp())
        j = "Chosen on structural grounds before any model exists; " * 2
        H.preregister(root, "volume_change_error", j, self._cmp())
        rec = H.preregister(root, "log_volume_ratio_error", j, self._cmp(),
                            overwrite=True)
        self.assertEqual(rec["superseded"]["primary_metric"], "volume_change_error")
