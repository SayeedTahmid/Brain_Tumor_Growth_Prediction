"""Tests for training, resume, volume inference and the plateau rule.

The property that matters most: a resumed run must be BIT-EXACT with the run it
claims to continue. A resume that restores weights but not optimizer moments,
scaler state or RNG position gives numbers that look fine and are not
reproducible — worse than an obvious failure.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

try:
    import torch
    from sailor.stage4.model import ResidualUNet3D
    from sailor.stage4 import train as TR
    from sailor.stage4 import inference as INF
    from sailor.stage4 import convergence as CV
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sailor.stage4 import mask_cache as MC  # noqa: E402
from sailor.stage4.patches import CachedPairPatchSampler  # noqa: E402


def tiny_project(n=3, shape=(120, 120, 120)):
    root = Path(tempfile.mkdtemp())
    ad = root / "01_DATA_FOUNDATION" / "v2_arrays"
    ad.mkdir(parents=True)
    pairs = []
    for s in range(1, n + 1):
        for ses, g in (("ses-01", 6), ("ses-02", 9)):
            a = np.zeros(shape, dtype=np.float32)
            a[50:50 + g, 55:55 + g, 58:58 + g] = 1
            np.savez_compressed(
                ad / f"sub-{s:02d}__{ses}__ContrastEnhancedMask-CL.npz", array=a)
        pairs.append({"subject": f"sub-{s:02d}", "input_session": "ses-01",
                      "target_session": "ses-02"})
    MC.build(root, verify=False, min_extent=32)
    return root, MC.CachedMasks(root), pairs


class TestPlateauRule(unittest.TestCase):
    """The rule is fixed BEFORE the curve is seen, so a budget is not eyeballed."""

    def _h(self, vals):
        return [{"step": (i + 1) * 250, "model": {"log_ratio": {"mean": v}}}
                for i, v in enumerate(vals)]

    # Plateau detection is covered by TestPlateauRuleV2 below; the v0.27
    # assertions here referenced `still_improving_at_end`, a key the corrected
    # rule replaced because it could not distinguish descent from oscillation.

    def test_empty_history_does_not_fabricate_a_budget(self):
        self.assertIsNone(TR.find_plateau([])["plateau_step"])

    def test_undefined_metric_is_reported_not_skipped(self):
        r = TR.find_plateau([{"step": 250, "model": {"log_ratio": {"mean": None}}}])
        self.assertIsNone(r["plateau_step"])
        self.assertIn("never defined", r["note"])


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestResumeIsBitExact(unittest.TestCase):

    def test_resume_reproduces_uninterrupted_training_exactly(self):
        _, cache, pairs = tiny_project()
        s = CachedPairPatchSampler(cache, pairs, patch=32)
        r = TR.verify_resume(lambda: ResidualUNet3D(), s, steps=8,
                             batch_size=1, device="cpu", amp=False)
        self.assertTrue(r["bit_exact"], r["worst"])
        self.assertEqual(r["n_params_differing"], 0)

    def test_checkpoint_roundtrip_restores_step(self):
        _, cache, pairs = tiny_project()
        s = CachedPairPatchSampler(cache, pairs, patch=32)
        ck = Path(tempfile.mkdtemp()) / "c.pt"
        m = ResidualUNet3D()
        TR.train_fold(m, s, steps=4, batch_size=1, device="cpu", amp=False,
                      checkpoint_path=ck, checkpoint_every=4, resume=False,
                      log_every=0)
        m2 = ResidualUNet3D()
        opt = torch.optim.AdamW(m2.parameters())
        sc = torch.amp.GradScaler("cuda", enabled=False)
        step, meta = TR.load_checkpoint(ck, m2, opt, sc)
        self.assertEqual(step, 4)
        self.assertTrue(meta["complete"])

    def test_missing_checkpoint_starts_from_zero(self):
        m = ResidualUNet3D()
        opt = torch.optim.AdamW(m.parameters())
        sc = torch.amp.GradScaler("cuda", enabled=False)
        step, meta = TR.load_checkpoint(Path(tempfile.mkdtemp()) / "none.pt",
                                        m, opt, sc)
        self.assertEqual(step, 0)
        self.assertIsNone(meta)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestVolumeInference(unittest.TestCase):
    """Validation must measure the TASK, not patch loss."""

    def test_prediction_covers_the_whole_volume(self):
        _, cache, pairs = tiny_project()
        vol = cache.get("sub-01", "ses-01")
        p = INF.predict_mask(ResidualUNet3D(), vol, patch=32, batch_size=2,
                             device="cpu", amp=False)
        self.assertEqual(p.shape, vol.shape)

    def test_threshold_is_fixed_not_tuned(self):
        self.assertEqual(INF.THRESHOLD, 0.5)

    def test_evaluation_reports_persistence_on_the_same_pairs(self):
        _, cache, pairs = tiny_project()
        ev = INF.evaluate_fold(ResidualUNet3D(), cache, pairs, patch=32,
                               batch_size=2, device="cpu", amp=False)
        self.assertIn("persistence", ev)
        self.assertIsNotNone(ev["persistence"]["log_ratio"]["mean"])
        self.assertEqual(ev["n_pairs"], len(pairs))

    def test_tiling_covers_the_far_edge(self):
        starts = INF._starts(120, 32, 16)
        self.assertEqual(starts[-1], 120 - 32)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestFoldSplitting(unittest.TestCase):

    def test_train_and_test_patients_never_overlap(self):
        split = {"folds": {"repeats": [{"folds": [
            {"train_patients": ["sub-01", "sub-02"],
             "test_patients": ["sub-03"]}]}]},
            "pairs": {"pairs": [
                {"subject": "sub-01"}, {"subject": "sub-02"},
                {"subject": "sub-03"}]}}
        tr, te = CV.fold_pairs(split)
        self.assertEqual({p["subject"] for p in tr}, {"sub-01", "sub-02"})
        self.assertEqual({p["subject"] for p in te}, {"sub-03"})
        self.assertFalse({p["subject"] for p in tr} & {p["subject"] for p in te})

    def test_probe_fold_is_pre_specified(self):
        self.assertEqual((CV.PROBE_REPEAT, CV.PROBE_FOLD), (0, 0))

    def test_probe_validates_beyond_the_placeholder_budget(self):
        self.assertGreater(CV.MAX_STEPS, 2000)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestPlateauRuleV2(unittest.TestCase):
    """v0.28. The v0.27 rule reported plateau=500 on a curve whose own best was
    at step 4000 — two contradictory lines from one dataset. It tracked
    best-so-far and could not tell oscillation from convergence."""

    def _h(self, vals):
        return [{"step": (i + 1) * 250, "model": {"log_ratio": {"mean": v}}}
                for i, v in enumerate(vals)]

    REAL = [0.6796, 0.4556, 0.4849, 0.8664, 0.5742, 0.6946, 0.7126, 0.7027,
            0.5070, 0.7974, 0.4597, 0.5433, 0.5135, 0.4772, 0.4904, 0.4480]

    def test_real_probe_curve_is_not_converged(self):
        r = TR.find_plateau(self._h(self.REAL))
        self.assertFalse(r["converged"])
        self.assertIsNone(r["plateau_step"])

    def test_no_contradiction_between_plateau_and_best(self):
        # The v0.27 defect: plateau_step earlier than best_step.
        r = TR.find_plateau(self._h(self.REAL))
        if r["plateau_step"] is not None:
            self.assertGreaterEqual(r["plateau_step"], r["best_step"])

    def test_true_convergence_still_detected(self):
        r = TR.find_plateau(self._h([0.9, 0.6, 0.45, 0.44, 0.438, 0.437,
                                     0.437, 0.4368]))
        self.assertTrue(r["converged"])
        self.assertEqual(r["plateau_step"], 1000)

    def test_monotone_descent_is_still_improving_not_oscillating(self):
        r = TR.find_plateau(self._h([0.9, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08]))
        self.assertFalse(r["converged"])
        self.assertIn("STILL IMPROVING", r["note"])

    def test_pure_oscillation_is_named_as_such(self):
        r = TR.find_plateau(self._h([0.5, 0.8, 0.45, 0.85, 0.5, 0.79, 0.47, 0.83]))
        self.assertFalse(r["converged"])
        self.assertIn("OSCILLATING", r["note"])
        self.assertIn("Do NOT freeze", r["note"])


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestC0Diagnosis(unittest.TestCase):
    """Distinguish a degenerate copy from a genuine absence of signal."""

    def test_thresholds_are_declared_in_source(self):
        from sailor.stage4 import diagnose_c0 as DG
        _, cache, pairs = tiny_project()

        class Copy(torch.nn.Module):
            """Returns its input as large positive logits — a perfect copier."""
            def forward(self, x, cond=None):
                return (x * 20.0) - 10.0

        d = DG.diagnose(Copy(), cache, pairs, patch=32, batch_size=2,
                        device="cpu", amp=False)
        self.assertEqual(d["verdict"], "A_DEGENERATE_COPY")
        self.assertGreater(d["mean_dice_pred_vs_input"], 0.95)
        self.assertIn("NO conclusion about signal", d["detail"])

    def test_non_copying_model_is_not_flagged(self):
        from sailor.stage4 import diagnose_c0 as DG
        _, cache, pairs = tiny_project()

        class Empty(torch.nn.Module):
            def forward(self, x, cond=None):
                return torch.full_like(x, -10.0)

        d = DG.diagnose(Empty(), cache, pairs, patch=32, batch_size=2,
                        device="cpu", amp=False)
        self.assertEqual(d["verdict"], "NOT_A_COPY")

    def test_reports_what_it_cannot_establish(self):
        from sailor.stage4 import diagnose_c0 as DG
        _, cache, pairs = tiny_project()

        class Empty(torch.nn.Module):
            def forward(self, x, cond=None):
                return torch.full_like(x, -10.0)

        d = DG.diagnose(Empty(), cache, pairs, patch=32, batch_size=2,
                        device="cpu", amp=False)
        self.assertIn("what_this_does_not_establish", d)
        self.assertIn("Nothing about whether C0 has signal",
                      d["what_this_does_not_establish"])


class TestFrozenLoss(unittest.TestCase):
    """The loss is fixed once and identical across rungs (AMD-007)."""

    def test_config_records_the_change_and_its_timing(self):
        from sailor.stage4.loss import CONFIG
        self.assertIn("AMD-007", CONFIG["fixed_by"])
        self.assertEqual(CONFIG["changed_from"], "BCEWithLogits alone")
        self.assertIn("NO official", CONFIG["results_seen_when_fixed"])

    def test_states_it_is_not_training_on_the_metric(self):
        from sailor.stage4.loss import CONFIG
        self.assertIn("volume ratio", CONFIG["not_training_on_the_metric"])

    def test_weights_are_unweighted_one_to_one(self):
        from sailor.stage4 import loss as LS
        self.assertEqual(LS.BCE_WEIGHT, LS.DICE_WEIGHT)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestLossRemovesShrinkageIncentive(unittest.TestCase):
    """The measured mechanism: BCE's background gradient mass is the class ratio."""

    def _target(self):
        t = torch.zeros(1, 1, 16, 16, 16)
        t[0, 0, 6:10, 6:10, 6:10] = 1          # 64 of 4096 = 1.6% foreground
        return t

    def _grad_ratio(self, fn):
        t = self._target()
        z = torch.zeros_like(t, requires_grad=True)
        fn(z, t).backward()
        g = z.grad
        return abs(g[t == 0].sum().item()) / abs(g[t > 0].sum().item())

    def test_compound_loss_reduces_background_gradient_dominance(self):
        from sailor.stage4.loss import make_loss
        bce_ratio = self._grad_ratio(torch.nn.BCEWithLogitsLoss())
        cmp_ratio = self._grad_ratio(make_loss())
        self.assertGreater(bce_ratio, 50)      # ~63x, the class ratio
        self.assertLess(cmp_ratio, bce_ratio * 0.5)

    def test_bce_alone_is_symmetric_between_under_and_over(self):
        # The refuted explanation, pinned so it is not re-adopted.
        t = self._target()
        under = t.clone(); under[0, 0, 9, 6:10, 6:10] = 0
        over = t.clone(); over[0, 0, 10, 6:10, 6:10] = 1
        bce = torch.nn.BCEWithLogitsLoss()
        lu = float(bce((under * 20) - 10, t))
        lo = float(bce((over * 20) - 10, t))
        self.assertAlmostEqual(lu, lo, places=6)

    def test_empty_to_empty_pair_gives_finite_loss(self):
        # The five retained sub-25 pairs hit this.
        from sailor.stage4.loss import make_loss
        t = torch.zeros(1, 1, 8, 8, 8)
        v = float(make_loss()(torch.full_like(t, -10.0), t))
        self.assertTrue(np.isfinite(v))
        self.assertLess(v, 0.1)
