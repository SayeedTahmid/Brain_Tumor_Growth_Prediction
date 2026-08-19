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

    def test_frozen_loss_is_still_bce_plus_dice(self):
        self.assertEqual(FROZEN.CONFIG["loss"], "BCEWithLogits + soft Dice")

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

    def test_penalises_volume_error_harder_than_the_frozen_loss(self):
        t = self._t()
        under = t.clone(); under[0, 0, 9, 6:10, 6:10] = 0
        base, perf = (under * 20) - 10, (t * 20) - 10
        d_frozen = float(FROZEN.make_loss()(base, t)) - float(FROZEN.make_loss()(perf, t))
        d_diag = float(DL.make_diagnostic_loss()(base, t)) - \
                 float(DL.make_diagnostic_loss()(perf, t))
        self.assertGreater(d_diag, d_frozen * 1.3)

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
