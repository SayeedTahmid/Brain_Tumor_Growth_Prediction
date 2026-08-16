"""Tests for the v0.17 defect fixes.

Each test pins a defect that was found against REAL data and silently produced a
wrong result in v0.16. They are written so a regression fails loudly rather than
reappearing as a plausible-looking number.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sailor.data import archives as AR  # noqa: E402
from sailor.stage1 import adjudicate as A  # noqa: E402
from sailor.stage1 import plhm_check as P  # noqa: E402
from sailor.qc import guards as G  # noqa: E402

D = "d/mni2009c-n-s/sub-01/ses-01"


class TestDoseIsRead(unittest.TestCase):
    """v0.16 read 0 of 26 dose arrays: `dose` demanded a trailing separator, so
    `DoseMap.nii.gz` never matched. GATE-1 was blocked by a regex boundary."""

    def test_dosemap_is_a_mask_candidate(self):
        self.assertTrue(AR.MASKISH.search(f"{D}/DoseMap.nii.gz"))
        self.assertTrue(AR.MASKISH.search(f"{D}/DoseMap_unscaled.nii.gz"))

    def test_existing_mask_matches_did_not_regress(self):
        for base in ("Mask.nii.gz", "ContrastEnhancedMask-CL.nii.gz",
                     "ContrastEnhancedMask-ONCO.nii.gz", "EdemaMask-CL.nii.gz",
                     "NecrosisMask-ONCO.nii.gz"):
            self.assertTrue(AR.MASKISH.search(f"{D}/{base}"), base)

    def test_plain_images_are_still_not_mask_candidates(self):
        for base in ("T1c.nii.gz", "rCBV.nii.gz", "T2-icor.nii.gz"):
            self.assertFalse(AR.MASKISH.search(f"{D}/{base}"), base)


class TestIntensityClassification(unittest.TestCase):
    """13 of 13 G10 'failures' were label maps and z-scores, neither of which
    carries intensities the 0-255 claim could apply to."""

    def test_label_maps_and_zscores_are_not_intensities(self):
        self.assertTrue(AR.NON_INTENSITY.search(f"{D}/fastsurfer-segmentation.nii.gz"))
        self.assertTrue(AR.NON_INTENSITY.search(f"{D}/T2-icor-zscore.nii.gz"))

    def test_real_intensities_are_intensities(self):
        for base in ("T1c.nii.gz", "T2-icor.nii.gz", "Flair.nii.gz"):
            self.assertFalse(AR.NON_INTENSITY.search(f"{D}/{base}"), base)

    def test_icor_regex_excludes_the_zscore_variant(self):
        self.assertTrue(AR.ICOR_RE.search(f"{D}/T1c-icor.nii.gz"))
        self.assertFalse(AR.ICOR_RE.search(f"{D}/T1c-icor-zscore.nii.gz"))

    def test_range_check_ignores_non_intensity_volumes(self):
        vs = {
            "a::x/fastsurfer-segmentation.nii.gz": {
                "name": "x/sub-01/ses-01/fastsurfer-segmentation.nii.gz",
                "role": "image_sample", "min": 0.0, "max": 2035.0,
                "is_intensity": False},
            "a::x/T2-icor-zscore.nii.gz": {
                "name": "x/sub-01/ses-01/T2-icor-zscore.nii.gz",
                "role": "image_sample", "min": -2.1, "max": 9.1,
                "is_intensity": False},
            "a::x/T1c.nii.gz": {
                "name": "x/sub-02/ses-01/T1c.nii.gz",
                "role": "image_sample", "min": 0.0, "max": 255.0,
                "is_intensity": True},
        }
        r = A.nonfinite_report(vs)
        self.assertEqual(r["n_outside_0_255"], 0)
        self.assertEqual(r["n_excluded_non_intensity"], 2)


class TestSamplingCoverage(unittest.TestCase):
    """All 60 v0.16 samples came from sub-13, so G10 described one patient."""

    def _rec(self, sub, base, mn, mx):
        return {"name": f"d/{sub}/ses-01/{base}", "archive": "d.tar.bz2",
                "role": "image_sample", "dtype": "<f8", "min": mn, "max": mx,
                "mean": 1.0, "n_nonfinite": 0, "is_intensity": True,
                "has_fractional_values": True}

    def test_single_subject_sample_is_inconclusive_not_pass(self):
        vs = {f"k{i}": self._rec("sub-13", f"T{i}.nii.gz", 0.0, 255.0)
              for i in range(5)}
        rec = G.g10_intensity_sanity(vs, mni_archive="d.tar.bz2")
        self.assertEqual(rec["status"], G.INCONCLUSIVE)
        self.assertTrue(rec["evidence"]["single_subject_sample"])

    def test_multi_subject_sample_can_pass(self):
        vs = {f"k{i}": self._rec(f"sub-{i:02d}", "T1c.nii.gz", 0.0, 255.0)
              for i in range(1, 6)}
        rec = G.g10_intensity_sanity(vs, mni_archive="d.tar.bz2")
        self.assertEqual(rec["status"], G.PASS)
        self.assertEqual(rec["evidence"]["n_subjects_covered"], 5)

    def test_per_subject_quota_spreads_the_budget(self):
        h = AR.VolumeStatsHandler(sample_images=100, sample_per_subject=2)
        h.begin_archive("d.tar.bz2")
        taken = []
        for sub in ("sub-13", "sub-14"):
            for i in range(4):
                n = f"d/{sub}/ses-01/img{i}.nii.gz"
                if h.match(n, 1000):
                    taken.append(n)
                    h._sampled += 1
                    h._per_subject[sub] = h._per_subject.get(sub, 0) + 1
        self.assertEqual(sum("sub-13" in t for t in taken), 2)
        self.assertEqual(sum("sub-14" in t for t in taken), 2)


class TestRanoNotUsedAsEvidence(unittest.TestCase):
    """1 = complete response is refuted by measurement; the mapping is unknown."""

    def vol(self, entries):
        return {f"a::d/{s}/{e}/{b}": {"name": f"d/{s}/{e}/{b}", "n_nonzero": n,
                                      "role": "mask_candidate"}
                for s, e, b, n in entries}

    def test_rano_never_appears_in_evidence_lists(self):
        vs = self.vol([("sub-25", "ses-02", "ContrastEnhancedMask-CL.nii.gz", 82),
                       ("sub-25", "ses-03", "ContrastEnhancedMask-CL.nii.gz", 0),
                       ("sub-25", "ses-03", "ContrastEnhancedMask-ONCO.nii.gz", 407),
                       ("sub-25", "ses-03", "EdemaMask-CL.nii.gz", 2245),
                       ("sub-25", "ses-04", "ContrastEnhancedMask-CL.nii.gz", 0)])
        r = A.adjudicate(vs, [{"subject": "sub-25", "session": "ses-03", "rano": 2}])
        v = r["verdicts"][0]
        joined = " ".join(v["supports_true_response"] +
                          v["supports_segmentation_failure"])
        self.assertNotIn("RANO", joined)
        self.assertFalse(v["rano_used_as_evidence"])
        self.assertEqual(v["rano"], 2)          # still reported

    def test_mapping_is_marked_unverified_and_refutation_recorded(self):
        r = A.adjudicate(self.vol([("s", "ses-01",
                                    "ContrastEnhancedMask-CL.nii.gz", 0)]), [])
        self.assertEqual(r["rano_mapping_status"], "UNVERIFIED")
        self.assertIn("REFUTED", r["rano_coding_note"])


class TestNeighbourSemantics(unittest.TestCase):
    """A session with NO CL mask is neither empty nor a neighbour."""

    def vol(self, entries):
        return {f"a::d/{s}/{e}/{b}": {"name": f"d/{s}/{e}/{b}", "n_nonzero": n,
                                      "role": "mask_candidate"}
                for s, e, b, n in entries}

    def test_absent_mask_is_skipped_not_counted_as_zero(self):
        # ses-09 has only an edema mask: it must not enter the CL series at all.
        vs = self.vol([("sub-25", "ses-08", "ContrastEnhancedMask-CL.nii.gz", 0),
                       ("sub-25", "ses-09", "EdemaMask-CL.nii.gz", 2595),
                       ("sub-25", "ses-10", "ContrastEnhancedMask-CL.nii.gz", 0)])
        r = A.adjudicate(vs, [])
        v8 = next(v for v in r["verdicts"] if v["session"] == "ses-08")
        self.assertEqual(v8["neighbour_after"], 0)      # ses-10, not ses-09
        self.assertEqual(v8["neighbour_gap_after"], 2)  # and the gap is reported

    def test_contiguous_run_is_reported(self):
        vs = self.vol([("sub-25", f"ses-{i:02d}",
                        "ContrastEnhancedMask-CL.nii.gz", 0) for i in range(3, 9)])
        r = A.adjudicate(vs, [])
        self.assertEqual(r["clustering_by_subject"]["sub-25"]["longest_consecutive_run"], 6)


class TestPlhmCheck(unittest.TestCase):
    """The threshold is fixed in source before any measurement runs."""

    def _stats(self, spread_icor):
        vs = {}
        for ses in range(1, 6):
            for variant, spread in (("", 1.0), ("-icor", spread_icor)):
                base = f"T1c{variant}.nii.gz"
                vs[f"a::d/sub-01/ses-{ses:02d}/{base}"] = {
                    "name": f"d/sub-01/ses-{ses:02d}/{base}",
                    "percentiles_nonzero": {p: 100.0 + ses * 10 * spread
                                            for p in ("25", "50", "75")}}
        return vs

    def test_shared_landmarks_read_as_joint_fit(self):
        # Six modality series would be needed for a verdict; one gives INCONCLUSIVE,
        # which is the correct answer for insufficient data.
        r = P.measure(self._stats(spread_icor=0.01))
        self.assertEqual(r["verdict"], P.INCONCLUSIVE)
        self.assertEqual(r["n_series_compared"], 1)

    def test_plain_counterpart_is_read_for_paired_subjects(self):
        # Without this the check silently compares nothing: the first build read
        # every -icor volume but only 3 plain volumes per subject, so ZERO
        # (patient, modality) series had both variants and the verdict was
        # INCONCLUSIVE for a reason that looked like missing data.
        h = AR.VolumeStatsHandler(sample_images=0, sample_per_subject=0,
                                  read_icor=True, plhm_subjects=2)
        h.begin_archive("d.tar.bz2")
        self.assertEqual(h._role("d/sub-01/ses-01/T1c-icor.nii.gz"), "icor_full")
        self.assertEqual(h._role("d/sub-01/ses-01/T1c.nii.gz"), "plain_paired")
        h._role("d/sub-02/ses-01/T1c-icor.nii.gz")
        h._role("d/sub-03/ses-01/T1c-icor.nii.gz")   # beyond the quota
        self.assertIsNone(h._role("d/sub-03/ses-01/T1c.nii.gz"))

    def test_threshold_is_pre_registered_in_source(self):
        r = P.measure({})
        self.assertEqual(r["pre_registered"]["joint_fit_ratio"], P.JOINT_FIT_RATIO)
        self.assertEqual(r["provenance_status"], "UNVERIFIED")
        self.assertTrue(r["blocking"])

    def test_ratio_direction(self):
        tight = P.measure(self._stats(0.01))["comparisons"]
        loose = P.measure(self._stats(1.0))["comparisons"]
        self.assertLess(tight[0]["median_ratio"], loose[0]["median_ratio"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
