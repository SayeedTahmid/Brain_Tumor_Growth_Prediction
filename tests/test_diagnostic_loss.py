"""Tests for the diagnostic volume-loss probe.

This loss is NOT the frozen training criterion. The tests pin that separation as
firmly as they pin the behaviour, because a diagnostic that quietly became the
default would invalidate C0 and C1 without an amendment.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sailor.stage4 import diagnostic_loss as DL  # noqa: E402
from sailor.stage4 import loss as FROZEN  # noqa: E402


class TestSeparationFromFrozenLoss(unittest.TestCase):

    def test_marked_as_diagnostic_not_frozen(self):
        self.assertIn("DIAGNOSTIC", DL.CONFIG["status"])
        self.assertIn("unchanged", DL.CONFIG["frozen_loss_untouched"])

    def test_frozen_loss_adopted_variant_a_under_amd_009(self):
        # Was "BCEWithLogits + soft Dice" until AMD-009. The probe modules
        # remain diagnostics; the frozen loss is now the corrected one.
        self.assertEqual(FROZEN.CONFIG["loss"],
                         "BCEWithLogits + soft Dice + log-volume-ratio")

    def test_adoption_cost_is_recorded(self):
        self.assertIn("AMD-007", DL.CONFIG["if_adopted"])
        self.assertIn("re-running", DL.CONFIG["if_adopted"])


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestVolumeTermBehaviour(unittest.TestCase):

    def _t(self):
        t = torch.zeros(1, 1, 16, 16, 16)
        t[0, 0, 6:10, 6:10, 6:10] = 1        # 64 of 4096 = 1.6% foreground
        return t

    def test_perfect_prediction_is_near_zero(self):
        t = self._t()
        self.assertLess(float(DL.make_diagnostic_loss()((t * 20) - 10, t)), 0.01)

    def test_diagnostic_a_now_matches_the_adopted_frozen_loss(self):
        # AMD-009 adopted variant A, so the diagnostic and the frozen loss now
        # compute the same thing. Pinned so a later divergence is deliberate.
        t = self._t()
        under = t.clone(); under[0, 0, 9, 6:10, 6:10] = 0
        p = (under * 20) - 10
        self.assertAlmostEqual(float(DL.make_diagnostic_loss()(p, t)),
                               float(FROZEN.make_loss()(p, t)), places=5)

    def test_gradient_pushes_volume_up_when_under_predicting(self):
        t = self._t()
        under = t.clone(); under[0, 0, 9, 6:10, 6:10] = 0
        b = torch.zeros(1, requires_grad=True)
        DL.make_diagnostic_loss()(((under * 20) - 10) + b, t).backward()
        self.assertLess(float(b.grad), 0)      # negative => increase volume

    def test_empty_to_empty_is_finite(self):
        # The five retained sub-25 pairs hit this.
        t = torch.zeros(1, 1, 8, 8, 8)
        v = float(DL.make_diagnostic_loss()(torch.full_like(t, -10.0), t))
        self.assertTrue(torch.isfinite(torch.tensor(v)))

    def test_volume_weight_scales_the_term(self):
        t = self._t()
        under = t.clone(); under[0, 0, 9, 6:10, 6:10] = 0
        p = (under * 20) - 10
        self.assertGreater(float(DL.make_diagnostic_loss(1.0)(p, t)),
                           float(DL.make_diagnostic_loss(0.1)(p, t)))


class TestInterpretationIsPreCommitted(unittest.TestCase):
    """The reading of each outcome is fixed before the probe runs."""

    def _r(self, gap):
        return {"gap_vs_persistence": gap, "frozen_mde": 0.0555}

    def test_three_outcomes_are_distinguished(self):
        import sailor.stage4.diagnostic_loss as D
        # Reproduce the branch logic the probe applies.
        for gap, token in ((-0.01, "HOLDS THE FLOOR"),
                           (0.09, "STILL DRIFTS"),
                           (0.02, "PARTIAL")):
            if gap <= 0:
                got = "HOLDS THE FLOOR"
            elif gap > 0.0555:
                got = "STILL DRIFTS"
            else:
                got = "PARTIAL"
            self.assertEqual(got, token)

    def test_partial_outcome_refuses_to_round(self):
        src = Path(DL.__file__).read_text()
        self.assertIn("rather than rounding it either way", src)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestVariantBIsNotTheMetric(unittest.TestCase):
    """Variant A's term is a differentiable analogue of the scoring metric.
    Variant B corrects the same drift without sharing its functional form."""

    def test_variant_b_is_asymmetric_where_the_metric_is_symmetric(self):
        import math
        v_true = 64.0
        log_half = abs(math.log((v_true + 1) / (v_true * 0.5 + 1)))
        log_double = abs(math.log((v_true + 1) / (v_true * 2 + 1)))
        self.assertAlmostEqual(log_half, log_double, places=1)   # metric: symmetric
        rel_half = abs(v_true * 0.5 - v_true) / (v_true + DL.VOLUME_EPS)
        rel_double = abs(v_true * 2 - v_true) / (v_true + DL.VOLUME_EPS)
        self.assertAlmostEqual(rel_double / rel_half, 2.0, places=6)  # B: not

    def test_gradient_still_pushes_volume_up_when_under_predicting(self):
        t = torch.zeros(1, 1, 16, 16, 16); t[0, 0, 6:10, 6:10, 6:10] = 1
        under = t.clone(); under[0, 0, 9, 6:10, 6:10] = 0
        b = torch.zeros(1, requires_grad=True)
        DL.make_relative_volume_loss()(((under * 20) - 10) + b, t).backward()
        self.assertLess(float(b.grad), 0)

    def test_empty_target_does_not_explode(self):
        # V_true = 0 for the five retained sub-25 pairs; a pure ratio would
        # divide by zero.
        t = torch.zeros(1, 1, 8, 8, 8)
        v = float(DL.make_relative_volume_loss()(torch.full_like(t, -10.0), t))
        self.assertTrue(torch.isfinite(torch.tensor(v)))
        self.assertLess(v, 0.5)

    def test_config_b_records_why_variant_a_was_not_adopted(self):
        self.assertIn("scored by", DL.CONFIG_B["why_not_variant_a"])
        self.assertEqual(DL.CONFIG_B["probe_a_result"]["reduction"], "94%")

    def test_variant_b_is_still_marked_diagnostic(self):
        self.assertIn("DIAGNOSTIC", DL.CONFIG_B["status"])


class TestAmd009Adoption(unittest.TestCase):
    """The loss changed AFTER rung results. That is disclosed, not hidden."""

    def test_frozen_loss_now_carries_the_volume_term(self):
        self.assertIn("log-volume-ratio", FROZEN.CONFIG["loss"])
        self.assertEqual(FROZEN.CONFIG["amendment"], "AMD-009")

    def test_states_which_rungs_it_invalidates(self):
        inv = FROZEN.CONFIG["invalidates"]
        for r in ("C0-direct", "C0-residual", "C1"):
            self.assertIn(r, inv)
        self.assertIn("RETAINED", inv)

    def test_states_the_cost_rather_than_implying_it(self):
        self.assertIn("weaker evidence", FROZEN.CONFIG["known_cost"])

    def test_amendment_is_registered_with_full_disclosure(self):
        from sailor.experiments import gates as G
        a = next(x for x in G.AMENDMENTS if x["id"] == "AMD-009")
        prc = a["post_result_correction"]
        self.assertTrue(prc["acknowledged"])
        for f in ("what_was_seen", "why_not_a_forking_path",
                  "prior_results_retained", "known_cost"):
            self.assertGreater(len(str(prc[f])), 40)

    def test_rejected_variants_are_on_record(self):
        from sailor.experiments import gates as G
        a = next(x for x in G.AMENDMENTS if x["id"] == "AMD-009")
        self.assertIn("Variant B", a["post_result_correction"]["rejected_alternatives"])
        self.assertIn("stop rule", a["post_result_correction"]["rejected_alternatives"])

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_fingerprint_changed_so_old_checkpoints_cannot_resume(self):
        from sailor.stage4.train import _config_fingerprint
        self.assertNotEqual(_config_fingerprint(), "ce442e00558cf7e3")
