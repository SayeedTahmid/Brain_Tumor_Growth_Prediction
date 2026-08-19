"""Tests for patch sampling, the shared architecture, and the budget calculator.

These pin the properties that make the ladder comparable, not just runnable.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from sailor.stage4 import patches as PA  # noqa: E402
from sailor.stage4 import benchmark as BM  # noqa: E402

try:
    import torch
    from sailor.stage4.model import ResidualUNet3D, param_count
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def make_arrays(d, sub, sessions, lesion=True):
    for ses in sessions:
        a = np.zeros((60, 70, 60), dtype=np.float32)
        if lesion:
            a[28:34, 32:40, 28:34] = 1
        np.savez(Path(d) / f"{sub}__{ses}__ContrastEnhancedMask-CL.npz", array=a)


class TestPatchConfigIsFixed(unittest.TestCase):
    """AMD-007: these change PREDICTIONS, not just speed."""

    def test_config_records_why_it_is_fixed(self):
        self.assertIn("AMD-007", PA.CONFIG["fixed_by"])
        self.assertIn("two ways at once", PA.CONFIG["why_fixed"])

    def test_constants_are_module_level_not_defaults_only(self):
        for name in ("PATCH", "FOREGROUND_RATIO", "JITTER", "SAMPLING_SEED"):
            self.assertTrue(hasattr(PA, name), name)


class TestPatchSampling(unittest.TestCase):

    def test_shapes_are_the_fixed_patch_size(self):
        d = tempfile.mkdtemp()
        make_arrays(d, "sub-01", ["ses-01", "ses-02"])
        s = PA.PairPatchSampler(d, [{"subject": "sub-01",
                                     "input_session": "ses-01",
                                     "target_session": "ses-02"}], patch=32)
        x, y = s.batch(3)
        self.assertEqual(x.shape, (3, 1, 32, 32, 32))
        self.assertEqual(y.shape, (3, 1, 32, 32, 32))

    def test_foreground_is_actually_reached(self):
        d = tempfile.mkdtemp()
        make_arrays(d, "sub-01", ["ses-01", "ses-02"])
        s = PA.PairPatchSampler(d, [{"subject": "sub-01",
                                     "input_session": "ses-01",
                                     "target_session": "ses-02"}], patch=16)
        _, y = s.batch(40)
        hits = int((y.reshape(40, -1).sum(1) > 0).sum())
        self.assertGreater(hits, 5, "foreground centring never fired")

    def test_empty_target_pair_still_samples(self):
        # The 5 empty->empty pairs must contribute background statistics, not
        # be silently skipped.
        d = tempfile.mkdtemp()
        make_arrays(d, "sub-25", ["ses-05", "ses-06"], lesion=False)
        s = PA.PairPatchSampler(d, [{"subject": "sub-25",
                                     "input_session": "ses-05",
                                     "target_session": "ses-06"}], patch=16)
        x, y = s.batch(4)
        self.assertEqual(x.shape[0], 4)
        self.assertEqual(int(y.sum()), 0)

    def test_absent_mask_yields_none_not_zeros(self):
        d = tempfile.mkdtemp()
        make_arrays(d, "sub-25", ["ses-08"])
        self.assertIsNone(PA.load_pair_arrays(
            d, {"subject": "sub-25", "input_session": "ses-08",
                "target_session": "ses-09"}))

    def test_sampling_is_reproducible_under_seed(self):
        d = tempfile.mkdtemp()
        make_arrays(d, "sub-01", ["ses-01", "ses-02"])
        p = [{"subject": "sub-01", "input_session": "ses-01",
              "target_session": "ses-02"}]
        a = PA.PairPatchSampler(d, p, patch=16).batch(4, epoch=2)[0]
        b = PA.PairPatchSampler(d, p, patch=16).batch(4, epoch=2)[0]
        np.testing.assert_array_equal(a, b)

    def test_different_epochs_give_different_patches(self):
        d = tempfile.mkdtemp()
        make_arrays(d, "sub-01", ["ses-01", "ses-02"])
        p = [{"subject": "sub-01", "input_session": "ses-01",
              "target_session": "ses-02"}]
        s = PA.PairPatchSampler(d, p, patch=16)
        self.assertFalse(np.array_equal(s.batch(4, epoch=0)[0],
                                        s.batch(4, epoch=1)[0]))


class TestBudget(unittest.TestCase):
    """The ladder is 200 fold-fits. That number drives everything."""

    def test_total_fits_is_two_hundred(self):
        self.assertEqual(BM.TOTAL_FITS, 200)
        self.assertEqual(BM.FITS_PER_RUNG, 25)

    def test_rung_fitting_in_a_job_is_the_binding_test(self):
        bench = {"sec_per_step_total": 1.0}
        # 2000 steps/fit -> 0.56 h/fit -> 13.9 h/rung -> fits
        d = BM.budget(bench, steps_per_fit=2000)
        self.assertTrue(d["rung_fits_in_one_job"])
        # 5000 steps/fit -> 1.39 h/fit -> 34.7 h/rung -> does not
        d2 = BM.budget(bench, steps_per_fit=5000)
        self.assertFalse(d2["rung_fits_in_one_job"])
        self.assertEqual(d2["verdict"], "DOES_NOT_FIT")

    def test_tight_is_distinguished_from_fits(self):
        # ~20 h/rung: inside the limit but with no restart headroom
        bench = {"sec_per_step_total": 1.0}
        d = BM.budget(bench, steps_per_fit=2900)
        self.assertEqual(d["verdict"], "TIGHT")

    def test_refuses_to_offer_patch_changes_as_a_saving(self):
        d = BM.budget({"sec_per_step_total": 1.0}, steps_per_fit=9000)
        refused = [o for o in d["if_it_does_not_fit"] if o.startswith("REFUSED")]
        self.assertTrue(refused)
        self.assertIn("patch size", refused[0])

    def test_cv_reduction_is_marked_as_a_lock_change(self):
        d = BM.budget({"sec_per_step_total": 1.0}, steps_per_fit=9000)
        joined = " ".join(d["if_it_does_not_fit"])
        self.assertIn("CHANGES A LOCK", joined)

    def test_aggregate_is_reported_not_hidden(self):
        d = BM.budget({"sec_per_step_total": 1.0}, steps_per_fit=2000)
        self.assertGreater(d["aggregate_gpu_hours"], 0)
        self.assertIn("whole experiment", d["note"])


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestSharedArchitecture(unittest.TestCase):
    """AMD-007: rungs must differ in CONDITIONING, never in capacity."""

    def test_output_matches_input_shape(self):
        m = ResidualUNet3D()
        self.assertEqual(tuple(m(torch.zeros(2, 1, 32, 32, 32)).shape),
                         (2, 1, 32, 32, 32))

    def test_conditioned_rung_adds_only_film_parameters(self):
        c0, c4 = ResidualUNet3D(), ResidualUNet3D(cond_dim=4)
        delta = param_count(c4) - param_count(c0)
        self.assertLess(delta, 0.01 * param_count(c0),
                        "conditioning changed capacity by >1% — rung gaps would "
                        "measure architecture, not conditioning")

    def test_conditioning_reaches_the_bottleneck(self):
        """v0.32: conditioning has NO effect at initialisation by design — the
        residual head is zero-initialised so the untrained model is exactly
        persistence. This checks FiLM reaches the bottleneck, which is what makes
        conditioning learnable; that it becomes EFFECTIVE after training is
        covered by TestResidualFormulation.test_conditioning_effective_after_training.
        """
        torch.manual_seed(0)
        m = ResidualUNet3D(cond_dim=4)
        x = torch.randn(1, 1, 32, 32, 32)
        # Bypass the zero-init head to observe the pre-head feature map.
        feats = {}
        m.head.register_forward_hook(lambda mod, i, o: feats.setdefault("in", i[0]))
        m(x, torch.zeros(1, 4)); a = feats.pop("in").clone()
        m(x, torch.ones(1, 4) * 5); b = feats.pop("in").clone()
        self.assertFalse(torch.allclose(a, b),
                         "FiLM did not alter the features reaching the head")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestMaskCache(unittest.TestCase):
    """Correctness first: cropping changes the coordinate frame, and a silent
    offset would shift every patch while looking entirely healthy."""

    def _project(self, n_subjects=4, lesion_at=(95, 110, 118)):
        from sailor.stage4 import mask_cache as MC
        root = Path(tempfile.mkdtemp())
        ad = root / "01_DATA_FOUNDATION" / "v2_arrays"
        ad.mkdir(parents=True)
        for s in range(1, n_subjects + 1):
            for ses in ("ses-01", "ses-02"):
                a = np.zeros((193, 229, 193), dtype=np.float32)
                z, y, x = lesion_at
                a[z + s:z + 10 + s, y:y + 16, x:x + 12] = 1
                np.savez_compressed(
                    ad / f"sub-{s:02d}__{ses}__ContrastEnhancedMask-CL.npz", array=a)
        return root, MC

    def test_verification_confirms_exact_equality(self):
        root, MC = self._project()
        res = MC.build(root)
        self.assertEqual(res["verification"]["mismatches"], 0)
        self.assertGreaterEqual(res["verification"]["volumes_checked"], 8)

    def test_crop_is_never_smaller_than_the_patch(self):
        # A crop below patch size would make every patch mostly zero padding.
        root, MC = self._project()
        res = MC.build(root, verify=False)
        for d in res["crop_shape"]:
            self.assertGreaterEqual(d, PA.PATCH)

    def test_cache_preserves_every_voxel(self):
        root, MC = self._project()
        MC.build(root, verify=False)
        c = MC.CachedMasks(root)
        ad = root / "01_DATA_FOUNDATION" / "v2_arrays"
        for key in c.keys:
            sub, ses = key.split("/")
            with np.load(ad / f"{sub}__{ses}__ContrastEnhancedMask-CL.npz") as z:
                original = int((np.asarray(z["array"]) > 0).sum())
            self.assertEqual(int(c.get(sub, ses).sum()), original, key)

    def test_dtype_is_uint8_and_small(self):
        root, MC = self._project()
        MC.build(root, verify=False)
        c = MC.CachedMasks(root)
        self.assertEqual(c.volumes.dtype, np.uint8)

    def test_foreground_coordinates_are_precomputed(self):
        root, MC = self._project()
        MC.build(root, verify=False)
        c = MC.CachedMasks(root)
        fg = c.foreground("sub-01", "ses-01")
        self.assertEqual(fg.shape[1], 3)
        self.assertEqual(len(fg), int(c.get("sub-01", "ses-01").sum()))

    def test_missing_cache_raises_clearly(self):
        from sailor.stage4 import mask_cache as MC
        with self.assertRaises(FileNotFoundError):
            MC.CachedMasks(tempfile.mkdtemp())


class TestCachedSampler(unittest.TestCase):

    def _setup(self):
        from sailor.stage4 import mask_cache as MC
        root = Path(tempfile.mkdtemp())
        ad = root / "01_DATA_FOUNDATION" / "v2_arrays"
        ad.mkdir(parents=True)
        pairs = []
        for s in range(1, 4):
            for ses in ("ses-01", "ses-02"):
                a = np.zeros((193, 229, 193), dtype=np.float32)
                a[95:105, 110:126, 118:130] = 1
                np.savez_compressed(
                    ad / f"sub-{s:02d}__{ses}__ContrastEnhancedMask-CL.npz", array=a)
            pairs.append({"subject": f"sub-{s:02d}", "input_session": "ses-01",
                          "target_session": "ses-02"})
        MC.build(root, verify=False)
        return MC.CachedMasks(root), pairs

    def test_returns_uint8_for_gpu_side_casting(self):
        c, pairs = self._setup()
        x, y = PA.CachedPairPatchSampler(c, pairs).batch(4)
        self.assertEqual(x.dtype, np.uint8)
        self.assertEqual(x.shape, (4, 1, PA.PATCH, PA.PATCH, PA.PATCH))

    def test_pairs_missing_from_cache_are_dropped_with_a_count(self):
        c, pairs = self._setup()
        pairs = pairs + [{"subject": "sub-99", "input_session": "ses-01",
                          "target_session": "ses-02"}]
        s = PA.CachedPairPatchSampler(c, pairs)
        self.assertEqual(len(s.dropped), 1)
        self.assertIn("sub-99", s.dropped[0])

    def test_reproducible_under_seed(self):
        c, pairs = self._setup()
        a = PA.CachedPairPatchSampler(c, pairs).batch(4, epoch=3)[0]
        b = PA.CachedPairPatchSampler(c, pairs).batch(4, epoch=3)[0]
        np.testing.assert_array_equal(a, b)


class TestCropFastPath(unittest.TestCase):

    def test_in_bounds_matches_manual_slice(self):
        v = np.arange(8 * 8 * 8, dtype=np.uint8).reshape(8, 8, 8)
        np.testing.assert_array_equal(PA.crop(v, (2, 2, 2), 4),
                                      v[2:6, 2:6, 2:6])

    def test_out_of_bounds_zero_pads(self):
        v = np.ones((8, 8, 8), dtype=np.uint8)
        out = PA.crop(v, (6, 6, 6), 4)
        self.assertEqual(out.shape, (4, 4, 4))
        self.assertEqual(int(out.sum()), 8)   # 2x2x2 real, rest padding


class TestConfigFingerprint(unittest.TestCase):
    """v0.30. Resume was keyed on the checkpoint PATH alone, so a loss change
    let fold 0 silently continue a model trained under the OLD objective while
    folds 1-4 trained under the new one."""

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_resume_refuses_when_config_changed(self):
        import torch
        from sailor.stage4 import train as TR
        from sailor.stage4.model import ResidualUNet3D
        ck = Path(tempfile.mkdtemp()) / "c.pt"
        m = ResidualUNet3D()
        opt = torch.optim.AdamW(m.parameters())
        sc = torch.amp.GradScaler("cuda", enabled=False)
        TR.save_checkpoint(ck, m, opt, sc, 100, {"steps_planned": 100})

        real = TR._config_fingerprint
        TR._config_fingerprint = lambda: "DIFFERENT_CONFIG"
        try:
            with self.assertRaises(RuntimeError) as cm:
                TR.load_checkpoint(ck, ResidualUNet3D(),
                                   torch.optim.AdamW(ResidualUNet3D().parameters()),
                                   torch.amp.GradScaler("cuda", enabled=False))
            self.assertIn("REFUSING TO RESUME", str(cm.exception))
            self.assertIn("mix two objectives", str(cm.exception))
        finally:
            TR._config_fingerprint = real

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_resume_proceeds_when_config_matches(self):
        import torch
        from sailor.stage4 import train as TR
        from sailor.stage4.model import ResidualUNet3D
        ck = Path(tempfile.mkdtemp()) / "c.pt"
        m = ResidualUNet3D()
        opt = torch.optim.AdamW(m.parameters())
        sc = torch.amp.GradScaler("cuda", enabled=False)
        TR.save_checkpoint(ck, m, opt, sc, 100, {})
        step, meta = TR.load_checkpoint(ck, ResidualUNet3D(),
                                        torch.optim.AdamW(m.parameters()), sc)
        self.assertEqual(step, 100)

    def test_fingerprint_changes_with_the_loss(self):
        from sailor.stage4 import train as TR, loss as LS
        a = TR._config_fingerprint()
        old = LS.DICE_WEIGHT
        LS.DICE_WEIGHT = 0.5
        LS.CONFIG["dice_weight"] = 0.5
        try:
            self.assertNotEqual(a, TR._config_fingerprint())
        finally:
            LS.DICE_WEIGHT = old
            LS.CONFIG["dice_weight"] = old


class TestRungAggregation(unittest.TestCase):

    def test_bootstrap_resamples_patients_not_pairs(self):
        from sailor.stage4 import rung as RG
        by = {"sub-01": [0.2] * 40, "sub-02": [0.8] * 40}
        r = RG._patient_bootstrap(by, n=2000)
        self.assertEqual(r["n_patients"], 2)
        self.assertAlmostEqual(r["mean"], 0.5, places=6)
        self.assertLess(r["ci_low"], 0.35)      # wide: n=2 patients, not 80 pairs

    def test_empty_input_does_not_fabricate_a_mean(self):
        from sailor.stage4 import rung as RG
        self.assertIsNone(RG._patient_bootstrap({})["mean"])

    def test_deterministic_under_seed(self):
        from sailor.stage4 import rung as RG
        by = {f"sub-{i:02d}": [i / 10] for i in range(1, 11)}
        self.assertEqual(RG._patient_bootstrap(by, n=1000)["ci_low"],
                         RG._patient_bootstrap(by, n=1000)["ci_low"])


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestResidualFormulation(unittest.TestCase):
    """v0.32. C0 scored 0.5991 vs persistence 0.4928 — worse by ~2x the MDE —
    and the identity control proved the scoring path was faithless of blame.
    Cause: `logits = f(input)` asks the net to synthesise the target from
    scratch, and with Dice(input,target) ~ 0.50 a loss-minimising model lands
    between them. The residual head makes persistence the FLOOR."""

    def _x(self):
        x = torch.zeros(1, 1, 32, 32, 32)
        x[:, :, 10:20, 10:20, 10:20] = 1
        return x

    def test_untrained_residual_model_is_exactly_persistence(self):
        m = ResidualUNet3D(); m.eval()
        x = self._x()
        with torch.no_grad():
            p = (torch.sigmoid(m(x)) > 0.5).float()
        self.assertTrue(torch.equal(p, x), "untrained model does not reproduce input")

    def test_head_is_zero_initialised(self):
        m = ResidualUNet3D()
        self.assertEqual(float(m.head.weight.abs().sum()), 0.0)
        self.assertEqual(float(m.head.bias.abs().sum()), 0.0)

    def test_non_residual_variant_does_not_reproduce_input(self):
        # The old formulation, kept so the difference is demonstrable.
        torch.manual_seed(0)
        m = ResidualUNet3D(residual=False); m.eval()
        x = self._x()
        with torch.no_grad():
            p = (torch.sigmoid(m(x)) > 0.5).float()
        self.assertFalse(torch.equal(p, x))

    def test_model_can_learn_away_from_persistence(self):
        # A floor is only useful if it is escapable.
        from sailor.stage4.loss import make_loss
        torch.manual_seed(0)
        x = self._x()
        tgt = torch.zeros_like(x); tgt[:, :, 10:22, 10:22, 10:22] = 1
        m = ResidualUNet3D(); L = make_loss()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        for _ in range(80):
            opt.zero_grad(); L(m(x), tgt).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = (torch.sigmoid(m(x)) > 0.5).float()
        d_target = float(2 * (p * tgt).sum() / (p.sum() + tgt.sum()))
        d_base = float(2 * (x * tgt).sum() / (x.sum() + tgt.sum()))
        self.assertGreater(d_target, d_base,
                           "residual model could not improve on persistence")

    def test_conditioning_effective_after_training(self):
        from sailor.stage4.loss import make_loss
        torch.manual_seed(0)
        x = self._x()
        tgt = torch.zeros_like(x); tgt[:, :, 10:22, 10:22, 10:22] = 1
        m = ResidualUNet3D(cond_dim=2); L = make_loss()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        for i in range(80):
            opt.zero_grad()
            grow = i % 2 == 0
            c = torch.tensor([[1., 0.]]) if grow else torch.tensor([[0., 1.]])
            L(m(x, c), tgt if grow else x).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            a = (torch.sigmoid(m(x, torch.tensor([[1., 0.]]))) > 0.5)
            b = (torch.sigmoid(m(x, torch.tensor([[0., 1.]]))) > 0.5)
        self.assertGreater(int((a ^ b).sum()), 0, "FiLM had no effect after training")

    def test_capacity_still_matched_across_rungs(self):
        c0, c4 = ResidualUNet3D(), ResidualUNet3D(cond_dim=4)
        self.assertLess(param_count(c4) - param_count(c0), 0.01 * param_count(c0))


class TestFingerprintCoversTheModel(unittest.TestCase):
    """v0.32. The residual change altered the architecture without altering the
    fingerprint, so a resume would have continued a non-residual checkpoint
    silently — the failure the fingerprint exists to prevent, one level up."""

    def test_model_config_is_in_the_fingerprint(self):
        from sailor.stage4 import train as TR, model as MD
        a = TR._config_fingerprint()
        old = MD.CONFIG["residual_prediction"]
        MD.CONFIG["residual_prediction"] = not old
        try:
            self.assertNotEqual(a, TR._config_fingerprint())
        finally:
            MD.CONFIG["residual_prediction"] = old

    def test_model_config_records_what_it_invalidates(self):
        from sailor.stage4.model import CONFIG
        self.assertIn("MUST be re-run", CONFIG["invalidates"])
        self.assertIn("AMD-007", CONFIG["fixed_by"])


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestResidualHead(unittest.TestCase):
    """ROS §8: `Residual Head | x_t, Z_cond, Δt | Δ̂` — the head outputs a
    CHANGE. Zero-init makes persistence the floor rather than a target the model
    has to rediscover."""

    def _x(self):
        torch.manual_seed(0)
        return (torch.rand(2, 1, 32, 32, 32) > 0.9).float()

    def test_untrained_residual_model_is_exactly_persistence(self):
        from sailor.stage4.model import ResidualUNet3D, PRIOR_SCALE
        x = self._x()
        with torch.no_grad():
            logits = ResidualUNet3D(residual=True)(x)
        self.assertTrue(torch.equal((torch.sigmoid(logits) > 0.5).float(), x))
        self.assertAlmostEqual(
            float((logits - PRIOR_SCALE * (2 * x - 1)).abs().max()), 0.0, places=9)

    def test_direct_model_does_not_start_at_persistence(self):
        from sailor.stage4.model import ResidualUNet3D
        x = self._x()
        with torch.no_grad():
            out = ResidualUNet3D(residual=False)(x)
        self.assertFalse(torch.equal((torch.sigmoid(out) > 0.5).float(), x))

    def test_both_modes_have_identical_capacity(self):
        # A3 compares residual vs direct; a capacity difference would confound it.
        from sailor.stage4.model import ResidualUNet3D, param_count
        self.assertEqual(param_count(ResidualUNet3D(residual=True)),
                         param_count(ResidualUNet3D(residual=False)))

    def test_prior_scale_is_reachable_by_the_correction(self):
        # A saturating prior (±20) would need a correction of ~40 to flip a
        # voxel, leaving the head nearly unable to move the prediction.
        from sailor.stage4.model import PRIOR_SCALE
        self.assertLessEqual(PRIOR_SCALE, 6.0)
        self.assertGreaterEqual(PRIOR_SCALE, 2.0)

    def test_zero_init_symmetry_breaks_after_one_step(self):
        from sailor.stage4.model import ResidualUNet3D
        from sailor.stage4.loss import make_loss
        torch.manual_seed(0)
        m = ResidualUNet3D(cond_dim=4, residual=True)
        L, x = make_loss(), self._x()
        y = (torch.rand(2, 1, 32, 32, 32) > 0.9).float()
        cond = torch.randn(2, 4)
        L(m(x, cond), y).backward()
        params = dict(m.named_parameters())
        self.assertGreater(float(params["head.weight"].grad.abs().max()), 0)
        self.assertEqual(float(params["film.weight"].grad.abs().max()), 0.0)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        opt.step(); opt.zero_grad()
        L(m(x, cond), y).backward()
        self.assertGreater(float(params["film.weight"].grad.abs().max()), 0)

    def test_config_records_that_c0_must_be_rerun(self):
        from sailor.stage4.model import CONFIG
        self.assertIn("MUST be re-run", CONFIG["invalidates"])
        self.assertIn("AMD-007", CONFIG["fixed_by"])
