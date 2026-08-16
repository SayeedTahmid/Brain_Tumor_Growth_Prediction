"""Tests for degenerate-mask adjudication (GATE-0 follow-up)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sailor.stage1 import adjudicate as A  # noqa: E402


def vol(entries):
    out = {}
    for sub, ses, base, nz in entries:
        out[f"a::d/{sub}/{ses}/{base}"] = {
            "name": f"d/{sub}/{ses}/{base}", "n_nonzero": nz,
            "role": "mask_candidate"}
    return out


class TestAdjudication(unittest.TestCase):
    def test_agreement_with_automated_and_rano_reads_as_true_response(self):
        vs = vol([("sub-01", "ses-03", "ContrastEnhancedMask-CL.nii.gz", 900),
                  ("sub-01", "ses-04", "ContrastEnhancedMask-CL.nii.gz", 0),
                  ("sub-01", "ses-04", "ContrastEnhancedMask-ONCO.nii.gz", 0),
                  ("sub-01", "ses-04", "EdemaMask-CL.nii.gz", 4200)])
        r = A.adjudicate(vs, [{"subject": "sub-01", "session": "ses-04", "rano": 1}])
        self.assertEqual(r["verdicts"][0]["verdict"], A.TRUE_ZERO)

    def test_automated_finds_tumour_reads_as_failure(self):
        vs = vol([("sub-02", "ses-02", "ContrastEnhancedMask-CL.nii.gz", 1500),
                  ("sub-02", "ses-03", "ContrastEnhancedMask-CL.nii.gz", 0),
                  ("sub-02", "ses-03", "ContrastEnhancedMask-ONCO.nii.gz", 1300),
                  ("sub-02", "ses-04", "ContrastEnhancedMask-CL.nii.gz", 1700),
                  ("sub-02", "ses-03", "EdemaMask-CL.nii.gz", 0)])
        r = A.adjudicate(vs, [{"subject": "sub-02", "session": "ses-03", "rano": 2}])
        self.assertEqual(r["verdicts"][0]["verdict"], A.FAILURE)

    def test_conflicting_evidence_is_ambiguous_not_forced(self):
        vs = vol([("sub-03", "ses-02", "ContrastEnhancedMask-CL.nii.gz", 800),
                  ("sub-03", "ses-03", "ContrastEnhancedMask-CL.nii.gz", 0),
                  ("sub-03", "ses-03", "ContrastEnhancedMask-ONCO.nii.gz", 0),
                  ("sub-03", "ses-04", "ContrastEnhancedMask-CL.nii.gz", 950)])
        r = A.adjudicate(vs, [{"subject": "sub-03", "session": "ses-03", "rano": 3}])
        self.assertEqual(r["verdicts"][0]["verdict"], A.AMBIGUOUS)

    def test_true_responses_are_retained_by_policy(self):
        r = A.adjudicate(vol([("s", "ses-01", "ContrastEnhancedMask-CL.nii.gz", 0)]), [])
        self.assertIn("retained", r["policy"])
        self.assertIn("bias the cohort toward progressors", r["policy"])

    def test_rano_coding_is_flagged_unverified(self):
        r = A.adjudicate(vol([("s", "ses-01", "ContrastEnhancedMask-CL.nii.gz", 0)]), [])
        self.assertIn("UNVERIFIED", r["rano_coding_note"])

    def test_undefined_dice_consequence_is_stated(self):
        r = A.adjudicate(vol([("s", "ses-01", "ContrastEnhancedMask-CL.nii.gz", 0)]), [])
        self.assertIn("Dice is undefined", r["metric_consequence"])

    def test_nonfinite_report_lists_offending_volumes(self):
        vs = {"a::x/T1c.nii.gz": {"name": "x/T1c.nii.gz", "n_nonfinite": 12,
                                  "dtype": "<f8", "role": "image_sample",
                                  "min": 0.0, "max": 255.0},
              "a::x/T2.nii.gz": {"name": "x/T2.nii.gz", "n_nonfinite": 0,
                                 "dtype": "<f8", "role": "image_sample",
                                 "min": -3.0, "max": 900.0}}
        r = A.nonfinite_report(vs)
        self.assertEqual(r["n_volumes_with_nonfinite"], 1)
        self.assertEqual(r["n_outside_0_255"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
