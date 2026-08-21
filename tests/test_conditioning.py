"""Tests for rung conditioning and its leakage guard.

The property that matters: standardisation statistics come from TRAINING pairs
only. Fitting on all pairs would let the held-out Δt distribution shape the
inputs seen during training — invisible in results, indefensible once noticed.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from sailor.stage4 import conditioning as CD  # noqa: E402

TRAIN = [{"delta_days": d, "input_treatment": t} for d, t in
         [(14, "CRT"), (14, "CRT"), (15, "CRT"), (91, "TMZ"),
          (91, "TMZ"), (371, "TMZ"), (70, "no"), (28, None)]]


class TestCondDim(unittest.TestCase):

    def test_dims_match_the_ladder_spec(self):
        self.assertEqual(CD.cond_dim("C0"), 0)
        self.assertEqual(CD.cond_dim("C1"), 1)
        self.assertEqual(CD.cond_dim("C2"), 5)     # Δt + 3 classes + observed
        self.assertEqual(CD.cond_dim("C4"), 5)     # + dose when n_dose given
        self.assertEqual(CD.cond_dim("C4", n_dose=3), 8)

    def test_c0_has_no_conditioning_function(self):
        self.assertIsNone(CD.make_cond_fn("C0"))

    def test_c1_requires_a_standardiser(self):
        with self.assertRaises(ValueError):
            CD.make_cond_fn("C1", None)


class TestFoldStandardiserIsLeakageSafe(unittest.TestCase):

    def test_statistics_come_only_from_the_pairs_given(self):
        train = TRAIN[:4]
        held = [{"delta_days": 9999, "input_treatment": "TMZ"}]
        s_train = CD.FoldStandardiser(train)
        s_all = CD.FoldStandardiser(train + held)
        self.assertNotAlmostEqual(s_train.mean, s_all.mean, places=3)

    def test_held_out_value_does_not_change_the_transform(self):
        s = CD.FoldStandardiser(TRAIN)
        before = s.transform(14)
        _ = s.transform(9999)                      # scoring an extreme pair
        self.assertEqual(s.transform(14), before)  # transform is stateless

    def test_records_that_it_was_fitted_on_training_only(self):
        self.assertIn("TRAINING pairs only",
                      CD.FoldStandardiser(TRAIN).to_dict()["fitted_on"])

    def test_degenerate_spread_does_not_produce_infinities(self):
        s = CD.FoldStandardiser([{"delta_days": 14}] * 5)
        self.assertTrue(s.degenerate_sd)
        self.assertTrue(np.isfinite(s.transform(14)))
        self.assertTrue(np.isfinite(s.transform(400)))

    def test_empty_input_raises_rather_than_defaulting(self):
        with self.assertRaises(ValueError):
            CD.FoldStandardiser([{"delta_days": None}])

    def test_log_compresses_the_range(self):
        s = CD.FoldStandardiser(TRAIN)
        raw_ratio = 371 / 7
        log_ratio = abs(s.transform(371) - s.transform(7))
        self.assertGreater(raw_ratio, 50)
        self.assertLess(log_ratio, 6)


class TestTreatmentEncoding(unittest.TestCase):

    def test_missing_status_is_flagged_not_imputed(self):
        f = CD.make_cond_fn("C2", CD.FoldStandardiser(TRAIN))
        v = f({"delta_days": 28, "input_treatment": None})
        self.assertEqual(list(v[1:]), [0.0, 0.0, 0.0, 0.0])   # no class, not observed

    def test_known_status_sets_one_class_and_the_flag(self):
        f = CD.make_cond_fn("C2", CD.FoldStandardiser(TRAIN))
        v = f({"delta_days": 14, "input_treatment": "CRT"})
        self.assertEqual(sum(v[1:4]), 1.0)
        self.assertEqual(v[4], 1.0)

    def test_unrecognised_token_is_treated_as_unobserved(self):
        f = CD.make_cond_fn("C2", CD.FoldStandardiser(TRAIN))
        v = f({"delta_days": 14, "input_treatment": "unknown"})
        self.assertEqual(v[4], 0.0)

    def test_uses_input_side_treatment_not_target(self):
        # The target session's status is information from the future.
        f = CD.make_cond_fn("C2", CD.FoldStandardiser(TRAIN))
        a = f({"delta_days": 14, "input_treatment": "CRT",
               "target_treatment": "TMZ"})
        b = f({"delta_days": 14, "input_treatment": "CRT",
               "target_treatment": "no"})
        np.testing.assert_array_equal(a, b)


class TestAmd001Recorded(unittest.TestCase):
    """C1 is not a treatment-free reference; it is easy to forget."""

    def test_c1_description_carries_the_warning(self):
        d = CD.describe("C1")
        self.assertIn("NOT a treatment-free reference", d["amd_001_note"])
        self.assertIn("understates", d["amd_001_note"])

    def test_treatment_side_is_documented(self):
        self.assertIn("never the target", CD.describe("C2")["treatment_side"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestControlRungShapes(unittest.TestCase):
    """Permutation controls carry a C-rung's information under another name.

    Name-prefix dispatch sent P2 down the Δt-only branch, so the model expected
    5 conditioning values and received 1 — surfacing as an opaque matmul error
    inside FiLM rather than at the cause.
    """

    def test_controls_inherit_the_shape_they_control(self):
        # ROS §11.1: P1 shuffles TREATMENT (so it mirrors C2), P2 shuffles DOSE.
        # These assertions encoded the inverted v0.39 mapping and are corrected.
        self.assertEqual(CD.cond_dim("P1"), CD.cond_dim("C2"))
        self.assertEqual(CD.cond_dim("P3"), CD.cond_dim("C1"))

    def test_versioned_control_names_resolve(self):
        self.assertEqual(CD.cond_dim("P1_v2"), 5)
        self.assertEqual(CD.cond_dim("C2_v2"), 5)

    def test_p1_actually_carries_treatment(self):
        f = CD.make_cond_fn("P1_v2", CD.FoldStandardiser(TRAIN))
        v = f({"delta_days": 14, "input_treatment": "CRT"})
        self.assertEqual(len(v), 5)
        self.assertEqual(v[4], 1.0)          # observed flag

    def test_describe_records_which_rung_it_mirrors(self):
        self.assertEqual(CD.describe("P1_v2")["conditioning_shape_of"], "C2")
        self.assertIsNone(CD.describe("C2")["conditioning_shape_of"])

    def test_unknown_control_name_does_not_silently_become_c0(self):
        # A name that matches nothing resolves to itself; C-prefix rules apply.
        self.assertEqual(CD.cond_dim("QX"), 1)   # not 0 — it is not a C0 name


class TestRosIdsAreAuthoritative(unittest.TestCase):
    """ROS §11.1: P1 is the TREATMENT shuffle, P2 the DOSE shuffle. v0.39 had
    these inverted because the mapping was written from memory of rung ordering
    rather than read from the constitution."""

    def test_p1_is_the_treatment_shuffle(self):
        self.assertEqual(CD.CONTROL_SHAPES["P1"], "C2")
        self.assertEqual(CD.cond_dim("P1"), CD.cond_dim("C2"))

    def test_p2_is_the_dose_shuffle(self):
        self.assertEqual(CD.CONTROL_SHAPES["P2"], "C3")

    def test_p3_is_the_time_only_reference(self):
        self.assertEqual(CD.CONTROL_SHAPES["P3"], "C1")

    def test_versioned_control_names_still_resolve(self):
        self.assertEqual(CD.cond_dim("P1_v2"), 5)


class TestRosAblationIds(unittest.TestCase):

    def test_a3_is_the_residual_formulation(self):
        from sailor.experiments.gates import ARCHITECTURAL_ABLATIONS as A
        a3 = next(a for a in A if a["id"] == "A3")
        self.assertIn("residual", a3["configuration"])

    def test_table_is_additive_and_starts_at_a0(self):
        from sailor.experiments.gates import ARCHITECTURAL_ABLATIONS as A
        self.assertEqual(A[0]["id"], "A0")
        self.assertEqual(len(A), 8)          # A0..A7, per ROS §11.2
        self.assertTrue(all("removes" not in a for a in A))

    def test_correction_records_the_effect_on_completed_work(self):
        from sailor.experiments.gates import ABLATION_ID_CORRECTION as C
        self.assertIn("No completed result changes", C["effect_on_completed_work"])
        self.assertIn("A0 = C0res_v2", C["procedural_note"])

    def test_p2_is_recorded_as_unrunnable(self):
        from sailor.experiments.gates import PERMUTATION_CONTROLS as P
        p2 = next(x for x in P if x["id"] == "P2")
        self.assertIn("UNRUNNABLE", p2["status"])
        self.assertIn("GATE-1", p2["status"])
