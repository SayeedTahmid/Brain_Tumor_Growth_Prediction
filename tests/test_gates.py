"""Tests for the pre-Phase-5 decision gates."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sailor.experiments import gates as G  # noqa: E402


class TestClaims(unittest.TestCase):
    def test_three_claims_are_separate(self):
        self.assertEqual(set(G.CLAIMS),
                         {"C_REALISM", "C_PREDICTION", "C_TREATMENT"})

    def test_realism_and_prediction_do_not_establish_treatment(self):
        for k in ("C_REALISM", "C_PREDICTION"):
            joined = " ".join(G.CLAIMS[k]["does_not_establish"])
            self.assertIn("treatment", joined)

    def test_every_claim_names_its_trap(self):
        for k, v in G.CLAIMS.items():
            self.assertTrue(v["trap"], k)


class TestGates(unittest.TestCase):
    def test_every_gate_has_a_go_and_no_go(self):
        for g in G.GATES:
            self.assertIn("GO", g["decision"], g["id"])
            self.assertIn("NO_GO", g["decision"], g["id"])

    def test_gate_order_matches_gate_ids(self):
        self.assertEqual(G.GATE_ORDER, [g["id"] for g in G.GATES])

    def test_dose_gate_precedes_any_c3_work(self):
        g1 = next(g for g in G.GATES if g["id"] == "GATE-1")
        self.assertIn("C3", g1["runs_before"])

    def test_p2_alone_is_declared_insufficient(self):
        g2 = next(g for g in G.GATES if g["id"] == "GATE-2")
        self.assertTrue(any("INSUFFICIENT" in c for c in g2["criteria"]))

    def test_mde_gate_requires_patient_level_resampling(self):
        g3 = next(g for g in G.GATES if g["id"] == "GATE-3")
        self.assertTrue(any("PATIENT level" in c for c in g3["criteria"]))

    def test_no_go_is_declared_a_result(self):
        self.assertIn("publishable", G.NO_GO_IS_A_RESULT)


class TestLadder(unittest.TestCase):
    def test_eight_scientific_rungs_including_entanglement_and_dose_control(self):
        ids = [r["id"] for r in G.SCIENTIFIC_RUNGS]
        for need in ("C-1", "C0", "C1", "C2", "C2+Δt", "C3-G", "C3-R", "FULL"):
            self.assertIn(need, ids)

    def test_c0_is_the_primary_treatment_free_reference(self):
        c0 = next(r for r in G.SCIENTIFIC_RUNGS if r["id"] == "C0")
        self.assertIn("PRIMARY", c0["role"])
        c1 = next(r for r in G.SCIENTIFIC_RUNGS if r["id"] == "C1")
        self.assertIn("NOT treatment-free", c1["role"])

    def test_families_are_separated(self):
        self.assertIn("never interleaved", G.ABLATION_SEPARATION)

    def test_one_primary_comparison_is_named_in_advance(self):
        self.assertEqual(G.PRIMARY_COMPARISON["comparison"], "C3-R vs C3-G")
        self.assertIn("both", G.PRIMARY_COMPARISON["decision_rule"].lower())

    def test_largest_gap_selection_is_prohibited(self):
        self.assertIn("largest observed gap",
                      G.SECONDARY_COMPARISONS["prohibited"])

    def test_full_model_is_built_regardless_of_gate4(self):
        g4 = next(g for g in G.GATES if g["id"] == "GATE-4")
        self.assertIn("NOT the full model", g4["runs_before"])
        self.assertIn("regardless", g4["note"])

    def test_framing_rejects_the_overclaim(self):
        self.assertIn("proves treatment-aware", G.FRAMING["not_this"])


class TestAmendments(unittest.TestCase):
    def test_every_amendment_records_what_was_seen_first(self):
        for a in G.AMENDMENTS:
            self.assertIn("results_seen_before_amendment", a)
            self.assertTrue(a["prompted_by"])

    def test_no_amendment_followed_a_c_rung_result(self):
        """Amending a design after seeing a rung result is the forking path."""
        for a in G.AMENDMENTS:
            self.assertIn("none", a["results_seen_before_amendment"].lower(), a["id"])

    def test_protocol_serialises(self):
        out = Path(tempfile.mkdtemp())
        res = G.write(out)
        data = json.loads(Path(res["path"]).read_text())
        self.assertEqual(len(data["gates"]), 5)
        # Amendment count is expected to grow; assert the log is present and
        # that each entry is complete rather than pinning a number.
        self.assertGreaterEqual(len(data["amendments"]), 7)
        for a in data["amendments"]:
            self.assertTrue(a["id"] and a["change"] and a["prompted_by"])
        self.assertIn("PRE-REGISTERED", data["status"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
