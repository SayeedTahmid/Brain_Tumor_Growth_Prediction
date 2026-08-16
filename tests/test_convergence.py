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

    def test_detects_a_clear_plateau(self):
        r = TR.find_plateau(self._h([0.9, 0.6, 0.45, 0.44, 0.438, 0.437, 0.437]))
        self.assertEqual(r["plateau_step"], 1000)
        self.assertFalse(r["still_improving_at_end"])

    def test_reports_when_still_improving(self):
        r = TR.find_plateau(self._h([0.9, 0.7, 0.5, 0.35, 0.25, 0.18]))
        self.assertIsNone(r["plateau_step"])
        self.assertTrue(r["still_improving_at_end"])
        self.assertIn("must be extended", r["note"])

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
