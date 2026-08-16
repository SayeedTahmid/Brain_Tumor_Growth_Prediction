"""Tests for the GATE-3 amendment and verdict.

The gate's literal criterion is unsatisfiable: the comparable literature reports
no persistence baseline. These tests pin that the replacement does not quietly
weaken the gate.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sailor.experiments import gate3 as G3  # noqa: E402
from sailor.experiments import gates  # noqa: E402


class TestLiteratureFinding(unittest.TestCase):

    def test_records_that_no_persistence_baseline_exists(self):
        self.assertFalse(G3.LITERATURE_FINDING["persistence_baseline_reported"])

    def test_records_why_the_numbers_are_not_comparable(self):
        reasons = G3.LITERATURE_FINDING["not_comparable_because"]
        self.assertGreaterEqual(len(reasons), 4)
        joined = " ".join(reasons).lower()
        for token in ("2d", "slice", "edema", "z-score"):
            self.assertIn(token, joined)

    def test_does_not_claim_tadiff_fails_to_beat_persistence(self):
        # The absence of a check is not evidence of failure.
        self.assertIn("nobody checked",
                      G3.LITERATURE_FINDING["what_it_does_not_license"])

    def test_carries_a_verification_caveat(self):
        self.assertIn("VERIFY", G3.LITERATURE_FINDING["verification_caveat"])


class TestHeadroomFraction(unittest.TestCase):

    def test_fraction_is_mde_over_baseline_for_a_distance_metric(self):
        h = G3.headroom_fraction(baseline_mean=0.4928, mde=0.1585)
        self.assertAlmostEqual(h["fraction_of_headroom_required"], 0.3216, places=3)
        self.assertEqual(h["total_available_headroom"], 0.4928)

    def test_smaller_mde_needs_a_smaller_share(self):
        a = G3.headroom_fraction(0.5, 0.25)["fraction_of_headroom_required"]
        b = G3.headroom_fraction(0.5, 0.05)["fraction_of_headroom_required"]
        self.assertGreater(a, b)

    def test_missing_inputs_do_not_fabricate_a_fraction(self):
        self.assertIsNone(G3.headroom_fraction(None, 0.1)["fraction"])
        self.assertIsNone(G3.headroom_fraction(0.5, None)["fraction"])

    def test_upper_bound_caveat_is_carried(self):
        self.assertIn("UPPER BOUND", G3.headroom_fraction(0.5, 0.1)["caveat"])


class TestAmendmentDoesNotWeakenTheGate(unittest.TestCase):

    def test_amd_008_registered_in_the_protocol(self):
        ids = [a["id"] for a in gates.AMENDMENTS]
        self.assertIn("AMD-008", ids)

    def test_no_numerical_threshold_invented(self):
        # A threshold set after the fraction is known would be post-hoc.
        self.assertIn("NO numerical pass threshold",
                      G3.AMENDMENT["no_threshold_invented"])

    def test_no_model_had_run_when_amended(self):
        self.assertIn("NO model", G3.AMENDMENT["results_seen_before_amendment"])


class TestVerdictRecording(unittest.TestCase):

    JUST = ("Recorded after seeing the baseline and MDE but before any model, "
            "C-rung or conditioning comparison exists. " * 2)

    def _assessment(self):
        return {"gate": "GATE-3", "status": "ASSESSED_NO_VERDICT",
                "preregistered_metric": "log_volume_ratio_error",
                "baseline": {"mean": 0.4928}, "mde": {"mde": 0.1585},
                "headroom": G3.headroom_fraction(0.4928, 0.1585)}

    def test_rejects_an_invalid_verdict(self):
        with self.assertRaises(ValueError):
            G3.record_verdict(tempfile.mkdtemp(), "MAYBE", self.JUST,
                              self._assessment())

    def test_rejects_a_thin_justification(self):
        with self.assertRaises(ValueError):
            G3.record_verdict(tempfile.mkdtemp(), "GO", "looks fine",
                              self._assessment())

    def test_second_verdict_refused(self):
        root = Path(tempfile.mkdtemp())
        G3.record_verdict(root, "GO", self.JUST, self._assessment())
        with self.assertRaises(RuntimeError):
            G3.record_verdict(root, "NO_GO", self.JUST, self._assessment())

    def test_deliberate_overwrite_preserves_the_prior_verdict(self):
        root = Path(tempfile.mkdtemp())
        G3.record_verdict(root, "GO", self.JUST, self._assessment())
        rec = G3.record_verdict(root, "NO_GO", self.JUST, self._assessment(),
                                overwrite=True)
        self.assertEqual(rec["superseded"]["verdict"], "GO")

    def test_no_go_forbids_rescuing_before_reporting(self):
        root = Path(tempfile.mkdtemp())
        rec = G3.record_verdict(root, "NO_GO", self.JUST, self._assessment())
        self.assertIn("Do not", rec["consequence"])
        self.assertIn("reported as-is", rec["consequence"])

    def test_go_requires_paired_mde_recomputation(self):
        root = Path(tempfile.mkdtemp())
        rec = G3.record_verdict(root, "GO", self.JUST, self._assessment())
        self.assertIn("paired difference SD", rec["consequence"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
