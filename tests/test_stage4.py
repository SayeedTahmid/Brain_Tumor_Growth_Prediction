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

    def test_conditioning_actually_changes_the_output(self):
        torch.manual_seed(0)
        m = ResidualUNet3D(cond_dim=4)
        x = torch.randn(1, 1, 32, 32, 32)
        a = m(x, torch.zeros(1, 4))
        b = m(x, torch.ones(1, 4) * 5)
        self.assertFalse(torch.allclose(a, b), "FiLM had no effect")


if __name__ == "__main__":
    unittest.main(verbosity=2)
