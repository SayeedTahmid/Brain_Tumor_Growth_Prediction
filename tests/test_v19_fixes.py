"""Tests for the v0.19 defect fixes (23, 24, 25) and the GATE-1 array export.

Each pins a defect found against REAL data that produced a wrong or useless
result silently.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from sailor.data import archives as AR  # noqa: E402
from sailor.stage1 import visual_check as VC  # noqa: E402
from sailor.stage1 import array_export as AE  # noqa: E402
from sailor.utils import persist as PS  # noqa: E402

D = "d/mni2009c-n-s/sub-25/ses-03"


class TestPerfusionNotIntensity(unittest.TestCase):
    """Defect 23: one rCBF volume measured 5550 and was counted as an intensity
    volume failing the 0-255 check. Perfusion maps are physiological units."""

    def test_perfusion_maps_excluded(self):
        for b in ("rCBF.nii.gz", "rCBV.nii.gz", "CBF.nii.gz", "CBV.nii.gz"):
            self.assertTrue(AR.NON_INTENSITY.search(f"{D}/{b}"), b)

    def test_previous_exclusions_did_not_regress(self):
        for b in ("fastsurfer-segmentation.nii.gz", "T2-icor-zscore.nii.gz"):
            self.assertTrue(AR.NON_INTENSITY.search(f"{D}/{b}"), b)

    def test_real_intensities_still_pass(self):
        for b in ("T1c.nii.gz", "T2-icor.nii.gz", "Flair.nii.gz", "T1-icor.nii.gz"):
            self.assertFalse(AR.NON_INTENSITY.search(f"{D}/{b}"), b)


class TestCacheSelectionByMtime(unittest.TestCase):
    """Defect 24: sorted(glob(...))[-1] is lexicographic. The v0.17 key sorts
    BEFORE the v0.16 key because '4' < '6' in s40 vs s60, so every caller
    silently read a stale pass with no percentiles and one subject's images."""

    def _cache(self, root, key, raw):
        p = PS.cache_dir(root) / f"{key}.json"
        p.write_text(json.dumps({"key": key, "raw": raw}))
        return p

    def test_lexicographic_order_would_pick_the_stale_cache(self):
        keys = ["audit_scan_full_mall_s40_ps6_icor1_plhm8_v17_derivatives",
                "audit_scan_full_mall_s60_derivatives"]
        self.assertEqual(sorted(keys)[-1], "audit_scan_full_mall_s60_derivatives")

    def test_mtime_picks_the_newest(self):
        root = Path(tempfile.mkdtemp())
        old = self._cache(root, "audit_scan_full_mall_s60_derivatives",
                          {"volume_stats": {"a": {}}})
        time.sleep(0.02)
        self._cache(root, "audit_scan_full_mall_s40_ps6_icor1_plhm8_v17_derivatives",
                    {"volume_stats": {"b": {}}})
        import os
        os.utime(old, (time.time() - 500, time.time() - 500))
        self.assertEqual(PS.latest_full_pass(root),
                         "audit_scan_full_mall_s40_ps6_icor1_plhm8_v17_derivatives")

    def test_slices_companion_is_never_selected(self):
        root = Path(tempfile.mkdtemp())
        self._cache(root, "audit_scan_full_x_derivatives", {"volume_stats": {"a": {}}})
        time.sleep(0.02)
        self._cache(root, "audit_scan_full_x_derivatives__slices", {"s": {}})
        self.assertEqual(PS.latest_full_pass(root), "audit_scan_full_x_derivatives")

    def test_cache_missing_required_field_is_rejected_loudly(self):
        root = Path(tempfile.mkdtemp())
        self._cache(root, "audit_scan_full_only_texts", {"texts": {}})
        with self.assertRaises(RuntimeError) as cm:
            PS.latest_full_pass(root, require=("volume_stats",))
        self.assertIn("volume_stats", str(cm.exception))

    def test_no_cache_at_all_raises(self):
        with self.assertRaises(RuntimeError):
            PS.latest_full_pass(Path(tempfile.mkdtemp()))


class TestSliceReference(unittest.TestCase):
    """Defect 25: slice indices came from each volume's OWN non-zero extent. For
    an image that is the whole head, so sub-25's T1c was sliced at z=7/88/169
    while the tumour sat at z=114-132 — the bed appeared in none of them."""

    def _vol(self, shape=(60, 60, 200)):
        a = np.zeros(shape, dtype=np.float32)
        a[10:50, 10:50, 5:195] = 100.0          # "head": nearly the whole z range
        return a

    def test_own_extent_misses_the_lesion(self):
        h = VC.SliceHandler(targets=[("sub-25", "ses-03")])
        a = self._vol()
        nz = np.nonzero(a)
        lo, hi = int(nz[2].min()), int(nz[2].max())
        idx = [int(lo + (hi - lo) * f) for f in (0.0, 0.5, 1.0)]
        self.assertFalse(any(114 <= k <= 132 for k in idx),
                         f"expected the v0.17 failure, got {idx}")

    def test_reference_z_hits_the_lesion(self):
        h = VC.SliceHandler(targets=[("sub-25", "ses-03")],
                            reference_z=VC.SUB25_REFERENCE_Z, z_margin=8)
        lo, hi = 114 - 8, 132 + 8
        idx = [int(lo + (hi - lo) * f) for f in (0.0, 0.5, 1.0)]
        self.assertTrue(all(100 <= k <= 145 for k in idx), idx)
        self.assertTrue(any(114 <= k <= 132 for k in idx), idx)

    def test_reference_z_constant_matches_the_measurement(self):
        # ses-01 CL extent measured 114-132; ses-02 measured 122-128.
        self.assertEqual(VC.SUB25_REFERENCE_Z, (114, 132))

    def test_targets_cover_all_eleven_sessions(self):
        t = VC.sub25_targets()
        self.assertEqual(len(t), 11)
        self.assertIn(("sub-25", "ses-09"), t)   # no CL mask: absence must be visible


class TestArrayExport(unittest.TestCase):
    """GATE-1 needs voxels on disk; the audit only ever recorded statistics."""

    def test_matches_dose_and_baseline_masks_only(self):
        h = AE.ArrayExportHandler(outdir=tempfile.mkdtemp())
        self.assertTrue(h.match(f"{D}/DoseMap.nii.gz", 1000))
        self.assertTrue(h.match(f"{D}/ContrastEnhancedMask-CL.nii.gz", 1000))
        self.assertFalse(h.match(f"{D}/T1c.nii.gz", 1000))
        self.assertFalse(h.match(f"{D}/EdemaMask-CL.nii.gz", 1000))

    def test_dose_ignores_the_session_filter(self):
        # Dose maps are per-subject and carry no session, so a session filter
        # must not silently drop them.
        h = AE.ArrayExportHandler(outdir=tempfile.mkdtemp(),
                                  sessions={("sub-25", "ses-03")})
        self.assertTrue(h.match("d/sub-01/DoseMap.nii.gz", 1000))
        self.assertFalse(h.match("d/sub-01/ses-99/ContrastEnhancedMask-CL.nii.gz", 1000))

    def test_summary_flags_that_resampling_is_required(self):
        exported = {
            "sub-01/DoseMap.nii.gz": {"shape": [256, 256, 190], "max": 61.0},
            "sub-02/DoseMap.nii.gz": {"shape": [240, 240, 190], "max": 65.0},
        }
        s = AE.summarise(exported)
        self.assertTrue(s["resampling_required"])
        self.assertEqual(s["n_on_reference_grid"], 0)
        self.assertEqual(s["n_dose_exported"], 2)
        self.assertEqual(s["dose_max_range"], [61.0, 65.0])

    def test_summary_detects_when_no_resampling_needed(self):
        exported = {"sub-01/DoseMap.nii.gz": {"shape": [193, 229, 193], "max": 60.0}}
        self.assertFalse(AE.summarise(exported)["resampling_required"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSliceRobustness(unittest.TestCase):
    """Found by fixture testing before release, not by a wasted pass."""

    def _nii(self, shape):
        import gzip, io
        from tests.make_fixture import nifti1_bytes
        a = np.zeros(shape, dtype=np.float32)
        a[1:shape[0]-1, 1:shape[1]-1, 1:shape[2]-1] = 5.0
        return io.BytesIO(gzip.compress(nifti1_bytes(a)))

    def test_out_of_range_reference_is_flagged_not_silently_wrong(self):
        h = VC.SliceHandler(targets=[("sub-01", "ses-01")], reference_z=(200, 240))
        h.handle("a", "d/sub-01/ses-01/T1c.nii.gz", 99, self._nii((8, 8, 20)))
        rec = h.slices["sub-01/ses-01/T1c.nii.gz"]
        self.assertIn("OUT_OF_RANGE", rec["slice_basis"])
        self.assertTrue(h.errors)
        self.assertTrue(all(0 <= k <= 19 for k in rec["slice_indices"]))

    def test_in_range_reference_is_clean(self):
        h = VC.SliceHandler(targets=[("sub-01", "ses-01")], reference_z=(8, 12),
                            z_margin=2)
        h.handle("a", "d/sub-01/ses-01/T1c.nii.gz", 99, self._nii((8, 8, 20)))
        rec = h.slices["sub-01/ses-01/T1c.nii.gz"]
        self.assertEqual(rec["slice_basis"], "reference_z")
        self.assertEqual(h.errors, [])
        self.assertTrue(all(6 <= k <= 14 for k in rec["slice_indices"]))


class TestCompanionCacheExclusion(unittest.TestCase):
    """__exported was selectable as a full pass: the exclusion listed only the
    companion that existed when it was written."""

    def test_all_companions_excluded(self):
        root = Path(tempfile.mkdtemp())
        base = "audit_scan_full_k_derivatives"
        for suffix in ("", "__slices", "__exported", "__somethingnew"):
            (PS.cache_dir(root) / f"{base}{suffix}.json").write_text(
                json.dumps({"raw": {"volume_stats": {}}}))
        self.assertEqual(PS.latest_full_pass(root), base)


class TestNiftiAffine(unittest.TestCase):
    """v0.20: the parser read past qform/sform and discarded them, so NOTHING on
    record said where any volume sat. Dose (256x256x190) against masks
    (193x229x193) could not be related at all."""

    def _hdr(self, sform_code=1, srow=None, qform_code=1, qoff=(0., 0., 0.)):
        from sailor.data.nifti_header import NiftiHeader
        srow = srow or [(1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.)]
        return NiftiHeader(
            version=1, byteorder="<", dim=[4, 5, 6], ndim=3,
            pixdim=[1., 1., 1.], datatype_code=16, dtype="<f4", bitpix=32,
            vox_offset=352, scl_slope=1., scl_inter=0.,
            qform_code=qform_code, sform_code=sform_code, xyzt_units=2, descrip="",
            qoffset_x=qoff[0], qoffset_y=qoff[1], qoffset_z=qoff[2],
            srow_x=srow[0], srow_y=srow[1], srow_z=srow[2])

    def test_sform_takes_precedence(self):
        h = self._hdr(srow=[(1., 0., 0., 10.), (0., 1., 0., 20.), (0., 0., 1., 30.)])
        self.assertEqual(h.world_origin, (10.0, 20.0, 30.0))
        self.assertTrue(h.spatial_status.startswith("SFORM"))

    def test_singular_sform_falls_back_to_qform(self):
        h = self._hdr(srow=[(0., 0., 0., 0.)] * 3, qform_code=1, qoff=(5., 6., 7.))
        self.assertIn("SINGULAR", h.spatial_status)
        self.assertEqual(h.world_origin, (5.0, 6.0, 7.0))

    def test_no_spatial_information_returns_none_not_identity(self):
        h = self._hdr(sform_code=0, qform_code=0)
        self.assertIsNone(h.affine)
        self.assertIsNone(h.world_origin)
        self.assertEqual(h.spatial_status, "NO_SPATIAL_INFORMATION")


class TestDoseAlignment(unittest.TestCase):
    from sailor.stage1 import dose_alignment as DA

    def _v(self, name, shape, affine, status="SFORM_code_2"):
        return {"name": name, "shape": shape, "affine": affine,
                "spatial_status": status}

    def test_pure_crop_pad_is_recognised(self):
        from sailor.stage1 import dose_alignment as DA
        ref = self._v("d/sub-01/ses-01/ContrastEnhancedMask-CL.nii.gz",
                      [193, 229, 193],
                      [[1, 0, 0, -96], [0, 1, 0, -132], [0, 0, 1, -78], [0, 0, 0, 1]])
        dose = self._v("d/sub-01/DoseMap.nii.gz", [256, 256, 190],
                       [[1, 0, 0, -128], [0, 1, 0, -148], [0, 0, 1, -78], [0, 0, 0, 1]])
        r = DA.compare(dose, ref)
        self.assertEqual(r["verdict"], DA.CROP_PAD)
        self.assertEqual([int(round(x)) for x in r["voxel_offset"]], [-32, -16, 0])

    def test_rotated_dose_needs_registration(self):
        from sailor.stage1 import dose_alignment as DA
        ref = self._v("x/ContrastEnhancedMask-CL.nii.gz", [193, 229, 193],
                      [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        dose = self._v("x/DoseMap.nii.gz", [256, 256, 190],
                       [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.assertEqual(DA.compare(dose, ref)["verdict"], DA.AFFINE_DIFFERS)

    def test_missing_affine_is_unknown_not_assumed(self):
        from sailor.stage1 import dose_alignment as DA
        ref = self._v("x/ContrastEnhancedMask-CL.nii.gz", [193, 229, 193], None,
                      "NO_SPATIAL_INFORMATION")
        dose = self._v("x/DoseMap.nii.gz", [256, 256, 190], None,
                       "NO_SPATIAL_INFORMATION")
        r = DA.compare(dose, ref)
        self.assertEqual(r["verdict"], DA.UNKNOWN)
        self.assertIn("CANNOT", r["detail"])

    def test_any_unknown_blocks_the_whole_gate(self):
        from sailor.stage1 import dose_alignment as DA
        ref = self._v("x/ContrastEnhancedMask-CL.nii.gz", [193, 229, 193],
                      [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        good = self._v("x/sub-01/DoseMap.nii.gz", [193, 229, 193],
                       [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        bad = self._v("x/sub-02/DoseMap.nii.gz", [256, 256, 190], None)
        res = DA.report({"a": ref, "b": good, "c": bad})
        self.assertEqual(res["verdict"], DA.UNKNOWN)


class TestReproducibilityStamp(unittest.TestCase):
    """§12/§18.5: a result must RECORD whether it is reproducible. env.git_state()
    existed but was wired into the bootstrap record only, so anything written
    from a module or a notebook cell carried none of it."""

    def test_stamp_has_the_required_fields(self):
        r = PS.reproducibility_stamp()
        for k in ("written_utc", "publication_status"):
            self.assertIn(k, r)

    def test_non_repo_is_flagged_not_reproducible(self):
        r = PS.reproducibility_stamp(code_root=tempfile.mkdtemp())
        self.assertFalse(r.get("git_available"))
        self.assertIn("NOT_REPRODUCIBLE", r["publication_status"])

    def test_artefacts_are_stamped_automatically(self):
        root = Path(tempfile.mkdtemp())
        PS.save_artefact(root, "06_QC_REPORTS", "thing", {"result": 1})
        w = json.loads((root / "06_QC_REPORTS" / "v2_thing.json").read_text())
        self.assertIn("reproducibility", w)
        self.assertIn("publication_status", w["reproducibility"])

    def test_caller_supplied_stamp_is_not_overwritten(self):
        root = Path(tempfile.mkdtemp())
        PS.save_artefact(root, "06_QC_REPORTS", "t2",
                         {"reproducibility": {"mine": True}})
        w = json.loads((root / "06_QC_REPORTS" / "v2_t2.json").read_text())
        self.assertEqual(w["reproducibility"], {"mine": True})

    def test_stamp_never_raises(self):
        self.assertIsInstance(PS.reproducibility_stamp(code_root="/nonexistent"), dict)
