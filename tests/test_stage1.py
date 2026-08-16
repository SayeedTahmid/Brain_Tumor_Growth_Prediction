"""Unit tests for logic that could be silently wrong (§16).

Run: python3 -m unittest discover -s tests -p 'test_*.py'
"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sailor.data import inventory, naming, tables, treatment  # noqa: E402
from sailor.data.nifti_header import read_array, read_header  # noqa: E402
from sailor.qc import guards  # noqa: E402
from tests.make_fixture import gz, nifti1_bytes  # noqa: E402


class TestNifti(unittest.TestCase):
    def test_roundtrip_uint8(self):
        arr = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
        hdr, got = read_array(io.BytesIO(nifti1_bytes(arr)), "x.nii")
        self.assertEqual(hdr.shape, (2, 3, 4))
        self.assertEqual(hdr.spacing, (1.0, 1.0, 1.0))
        np.testing.assert_array_equal(got, arr)

    def test_roundtrip_gzip_float(self):
        arr = (np.random.default_rng(1).random((3, 3, 3)) * 60).astype(np.float32)
        hdr, got = read_array(io.BytesIO(gz(nifti1_bytes(arr))), "x.nii.gz")
        self.assertEqual(hdr.dtype[-2:], "f4")
        np.testing.assert_allclose(got, arr)

    def test_header_only_does_not_need_full_stream(self):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        payload = nifti1_bytes(arr)[:400]  # truncated body
        hdr = read_header(io.BytesIO(payload), "x.nii")
        self.assertEqual(hdr.shape, (4, 4, 4))

    def test_fortran_order_is_respected(self):
        arr = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
        _, got = read_array(io.BytesIO(nifti1_bytes(arr)), "x.nii")
        self.assertEqual(got[1, 0, 0], arr[1, 0, 0])
        self.assertEqual(got[0, 1, 0], arr[0, 1, 0])


class TestNaming(unittest.TestCase):
    def test_longest_sequence_token_wins(self):
        self.assertEqual(naming.resolve_sequence("sub-01_ses-01_t2wflair.nii.gz", {}, "t2wflair"),
                         "t2wflair")
        self.assertEqual(naming.resolve_sequence("sub-01_ses-01_t2w.nii.gz", {}, "t2w"), "t2w")
        self.assertEqual(naming.resolve_sequence("sub-01_dscprea.nii.gz", {}, "dscprea"),
                         "dscprea")

    def test_cl_and_onco_are_distinguished(self):
        cl = naming.classify_annotation("sub-01_ses-01_desc-CL_label-enh_mask.nii.gz")
        onco = naming.classify_annotation("sub-01_ses-01_desc-ONCO_label-necro_mask.nii.gz")
        self.assertEqual(cl["kind"], "CL")
        self.assertEqual(cl["component"], "enhancing_t1wc")
        self.assertEqual(onco["kind"], "ONCO")
        self.assertEqual(onco["component"], "necrosis")

    def test_flair_component_not_swallowed_by_enhancement_token(self):
        c = naming.classify_annotation("sub-01_desc-CL_label-flair_ce_mask.nii.gz")
        self.assertEqual(c["component"], "t2wflair_hyperintensity")

    def test_masklike_token_is_bounded(self):
        """An unbounded `wm` token matched any path containing those letters."""
        from sailor.data.archives import MASKISH
        self.assertIsNone(MASKISH.search("sub-01_ses-01_flowmap.nii.gz"))
        self.assertIsNotNone(MASKISH.search("sub-01_desc-nawm_mask.nii.gz"))
        self.assertIsNotNone(MASKISH.search("sub-01_desc-dose_rtdose.nii.gz"))

    def test_unknown_masklike_is_flagged_not_guessed(self):
        c = naming.classify_annotation("sub-01_ses-01_seg-xyz.nii.gz")
        self.assertEqual(c["kind"], "UNRESOLVED_MASKLIKE")

    def test_target_report_flags_missing_component(self):
        rows = [{"path": "a_CL_mask.nii.gz", "subject": "sub-01", "session": "ses-01",
                 "sequence": None, "annotation_kind": "CL",
                 "annotation_component": None, "entities": {}, "suffix": None}]
        rep = naming.target_resolution_report(rows)
        self.assertEqual(rep["status"], "UNRESOLVED_COMPONENT")


class TestTreatment(unittest.TestCase):
    def test_unknown_is_missing_not_a_class(self):
        status, observed, token = treatment.canonicalise("unknown")
        self.assertIsNone(status)
        self.assertFalse(observed)
        self.assertEqual(token, "unknown")

    def test_unrecognised_token_is_not_coerced(self):
        status, observed, token = treatment.canonicalise("RT+TMZ maybe")
        self.assertIsNone(status)
        self.assertFalse(observed)
        self.assertEqual(token, "RT+TMZ maybe")

    def test_known_values_canonicalise(self):
        for raw, want in [("crt", "CRT"), ("TMZ", "TMZ"), ("no", "no"),
                          ("Temozolomide", "TMZ")]:
            self.assertEqual(treatment.canonicalise(raw)[0], want)

    def test_missing_indicator_is_emitted(self):
        ov = tables.parse_overview(
            "subject\tsession\ttreatment\n sub-01\tses-01\tunknown\n")
        out = treatment.extract(ov)
        rec = out["records"]["sub-01"]["ses-01"]
        self.assertEqual(rec["missing_indicator"], 1)
        self.assertIsNone(rec["status"])


class TestTables(unittest.TestCase):
    def test_column_resolution_reported_when_it_fails(self):
        out = tables.parse_raw_mni_link("colA\tcolB\n1\t2\n")
        self.assertEqual(out["status"], "UNVERIFIED_COLUMNS")

    def test_subject_normalisation(self):
        ov = tables.parse_overview("patient\tsession\n3\t7\n")
        self.assertEqual(ov["records"][0]["subject"], "sub-03")
        self.assertEqual(ov["records"][0]["session"], "ses-07")


class TestGuards(unittest.TestCase):
    def _rows(self, n):
        return [{"path": f"m{i}.nii.gz", "subject": "sub-01", "session": f"ses-{i:02d}",
                 "sequence": None, "annotation_kind": "CL",
                 "annotation_component": "enhancing_t1wc", "entities": {},
                 "suffix": None} for i in range(n)]

    def test_g1_inconclusive_when_target_absent(self):
        rec = guards.g1_degenerate_labels([], {})
        self.assertEqual(rec["status"], "INCONCLUSIVE")

    def test_g1_fails_on_all_zero(self):
        rows = self._rows(2)
        stats = {"m0.nii.gz": {"n_nonzero": 0, "n_voxels": 100, "labels": [0.0],
                               "binary": True},
                 "m1.nii.gz": {"n_nonzero": 50, "n_voxels": 100, "labels": [0.0, 1.0],
                               "binary": True}}
        rec = guards.g1_degenerate_labels(rows, stats)
        self.assertEqual(rec["status"], "FAIL")
        self.assertEqual(len(rec["evidence"]["primary"]["all_zero"]), 1)

    def test_g5_fails_on_quarantined_input(self):
        rec = guards.g5_leakage_stage1(["/x/ckpt_dose/model.pt"], {}, {})
        self.assertEqual(rec["status"], "FAIL")

    def test_g5_fails_on_prior_split(self):
        rec = guards.g5_leakage_stage1(["/x/split_v1.json"], {}, {})
        self.assertEqual(rec["status"], "FAIL")

    def test_g7_fails_when_any_interval_approximate(self):
        rec = guards.g7_delta_t_provenance([], {
            "sub-01/ses-01": {"source": "raw_scans_tsv_acq_time"},
            "sub-01/ses-02": {"source": "overview_days_column"}})
        self.assertEqual(rec["status"], "FAIL")

    def test_g7_inconclusive_when_nothing_recovered(self):
        self.assertEqual(guards.g7_delta_t_provenance([], {})["status"], "INCONCLUSIVE")

    def test_g8_inconclusive_without_link_table(self):
        rec = guards.g8_session_correspondence({"status": "ABSENT", "pairs": []}, {}, {})
        self.assertEqual(rec["status"], "INCONCLUSIVE")

    def test_g8_rejects_non_unique_join(self):
        link = {"status": "OK", "n_rows": 2, "pairs": [
            {"subject": "sub-01", "raw_session": "ses-01", "mni_session": "ses-01"},
            {"subject": "sub-01", "raw_session": "ses-02", "mni_session": "ses-01"}]}
        rec = guards.g8_session_correspondence(link, {"sub-01": ["ses-01"]},
                                               {"sub-01": ["ses-01", "ses-02"]})
        self.assertEqual(rec["status"], "FAIL")

    def test_g9_counts_survivors_not_nominal(self):
        missing = {"status": "OK", "n_rows": 1, "index": {"sub-01/ses-02": ["t1wc"]}}
        sessions = {"sub-01": ["ses-01", "ses-02"]}
        present = {("sub-01", "ses-01"): {"t1wc"}, ("sub-01", "ses-02"): {"t1wc"}}
        rec = guards.g9_missing_tsv(missing, sessions, present)
        self.assertEqual(rec["evidence"]["n_sessions_primary"], 1)

    def test_g9_zero_cohort_is_never_pass(self):
        """A guard must not render green over an empty dataset (§15.5)."""
        missing = {"status": "OK", "n_rows": 0, "index": {}}
        sessions = {"sub-01": ["ses-01", "ses-02"]}
        present = {("sub-01", "ses-01"): set(), ("sub-01", "ses-02"): set()}
        rec = guards.g9_missing_tsv(missing, sessions, present,
                                    raw_to_mni={("sub-01", "ses-01"): "ses-01"})
        self.assertEqual(rec["status"], "FAIL")
        self.assertEqual(rec["evidence"]["n_sessions_primary"], 0)

    def test_g9_refuses_to_apply_raw_space_list_without_mapping(self):
        """missing.tsv is raw-space; the masks are MNI-space (§3.1(3))."""
        missing = {"status": "OK", "n_rows": 2,
                   "index": {"sub-01/ses-07": ["t1wc"]}, "layout": "wide"}
        rec = guards.g9_missing_tsv(missing, {"sub-01": ["ses-01"]},
                                    {("sub-01", "ses-01"): {"t1wc"}})
        self.assertEqual(rec["status"], "INCONCLUSIVE")
        self.assertFalse(rec["evidence"]["mapped_through_raw_mni_link"])
        self.assertIn("raw/source session numbering", rec["detail"])

    def test_g9_translates_through_raw_mni_link(self):
        missing = {"status": "OK", "n_rows": 2, "layout": "wide",
                   "index": {"sub-01/ses-07": ["t1wc"]},
                   "index_inverted": {"sub-01/ses-07": ["t2w"]}}
        rec = guards.g9_missing_tsv(
            missing, {"sub-01": ["ses-03"]}, {("sub-01", "ses-03"): {"t1wc"}},
            raw_to_mni={("sub-01", "ses-07"): "ses-03"})
        self.assertTrue(rec["evidence"]["mapped_through_raw_mni_link"])
        self.assertEqual(rec["evidence"]["n_raw_keys_unmapped"], 0)

    def test_g9_reports_raw_keys_with_no_mni_counterpart(self):
        missing = {"status": "OK", "n_rows": 2, "layout": "wide",
                   "index": {"sub-01/ses-07": ["t1wc"], "sub-01/ses-21": ["t1wc"]}}
        rec = guards.g9_missing_tsv(
            missing, {"sub-01": ["ses-03"]}, {("sub-01", "ses-03"): {"t1wc"}},
            raw_to_mni={("sub-01", "ses-07"): "ses-03"})
        self.assertEqual(rec["evidence"]["n_raw_keys_unmapped"], 1)
        self.assertIn("sub-01/ses-21",
                      rec["evidence"]["raw_keys_without_mni_counterpart"])

    def test_g9_no_sessions_observed_is_inconclusive_not_fail(self):
        rec = guards.g9_missing_tsv({"status": "OK", "n_rows": 0, "index": {}}, {}, {})
        self.assertEqual(rec["status"], "INCONCLUSIVE")

    def test_g9_wide_layout_detected(self):
        seqs = ["t1w", "t1wc", "t2w", "t2wflair", "dce", "dsc", "dti", "adc"]
        txt = "\t".join(["subject", "session"] + seqs) + "\n"
        txt += "\t".join(["sub-01", "ses-01"] + ["n"] * 4 + ["y"] * 4) + "\n"
        out = tables.parse_missing(txt)
        self.assertEqual(out["layout"], "wide")
        self.assertEqual(out["status"], "OK")
        self.assertEqual(set(out["index"]["sub-01/ses-01"]), {"dce", "dsc", "dti", "adc"})

    def test_g9_polarity_settled_by_observed_files_not_filename(self):
        """An inverted file must be read correctly despite being named `missing`."""
        seqs = ["t1w", "t1wc", "t2w", "t2wflair"]
        head = "\t".join(["subject", "session"] + seqs)
        # Inverted convention: y = PRESENT. t1wc is flagged y and really is there.
        txt = head + "\n" + "\t".join(["sub-01", "ses-01", "y", "y", "n", "n"]) + "\n"
        mt = tables.parse_missing(txt)
        present = {("sub-01", "ses-01"): {"t1w", "t1wc"}}
        rec = guards.g9_missing_tsv(mt, {"sub-01": ["ses-01"]}, present)
        self.assertEqual(rec["evidence"]["polarity_adopted"], "n_means_missing")

    def test_g9_structural_pass_does_not_guess_polarity(self):
        seqs = ["t1w", "t1wc", "t2w", "t2wflair"]
        txt = ("\t".join(["subject", "session"] + seqs) + "\n"
               + "\t".join(["sub-01", "ses-01", "n", "n", "y", "y"]) + "\n")
        rec = guards.g9_missing_tsv(tables.parse_missing(txt), {}, {})
        self.assertEqual(rec["status"], "INCONCLUSIVE")

    def test_g10_inconclusive_without_measurements(self):
        self.assertEqual(guards.g10_intensity_sanity({})["status"], "INCONCLUSIVE")

    def test_summary_separates_inconclusive_from_pass(self):
        s = guards.summarise([
            {"guard": "GA", "status": "PASS"}, {"guard": "GB", "status": "INCONCLUSIVE"}])
        self.assertEqual(s["passed"], ["GA"])
        self.assertEqual(s["inconclusive"], ["GB"])
        self.assertFalse(s["stop_protocol_triggered"])


class TestSrcToRaw(unittest.TestCase):
    YAML = """Files:
   -
    in_dir:           sourcedata/sub-01/ses-01/t1wc
    out_dir:          rawdata/sub-01/ses-01
    filename:         t1wc
   -
    in_dir:           sourcedata/sub-02/ses-03/adc
    out_dir:          rawdata/sub-02/ses-03
    filename:         adc
"""

    def test_parses_conversion_records(self):
        out = tables.parse_src_to_raw(self.YAML)
        self.assertEqual(out["n_conversion_records"], 2)
        self.assertEqual(out["n_sessions"], 2)
        self.assertTrue(out["source_raw_indices_identical"])

    def test_index_mismatch_is_reported(self):
        bad = self.YAML.replace("out_dir:          rawdata/sub-01/ses-01",
                                "out_dir:          rawdata/sub-01/ses-07")
        out = tables.parse_src_to_raw(bad)
        self.assertFalse(out["source_raw_indices_identical"])


class TestTreatmentSource(unittest.TestCase):
    def test_absent_column_is_not_reported_as_all_unknown(self):
        ov = tables.parse_overview("subject\tsession\nsub-01\tses-01\n")
        out = treatment.extract(ov)
        self.assertEqual(out["status"], "NO_TREATMENT_COLUMN_RESOLVED")
        self.assertIsNone(out["resolved_column"])


class TestInventory(unittest.TestCase):
    def test_spacing_counted_for_annotations_not_only_sequences(self):
        """Spacing was silently empty whenever sequence resolution failed."""
        rows = [{"path": "sub-01/ses-01/x_desc-CL_label-enh_mask.nii.gz",
                 "subject": "sub-01", "session": "ses-01", "sequence": None,
                 "annotation_kind": "CL", "annotation_component": "enhancing_t1wc",
                 "entities": {}, "suffix": None}]
        hdrs = {"sub-01/ses-01/x_desc-CL_label-enh_mask.nii.gz":
                {"shape": [193, 229, 193], "spacing": [1.0, 1.0, 1.0], "dtype": "<f8"}}
        built = inventory.build_sessions(rows, hdrs)
        summ = inventory.summarise(built["sessions"])
        self.assertEqual(summ["spacing_frequency"], [[[1.0, 1.0, 1.0], 1]])
        self.assertEqual(summ["dtype_frequency"], [("<f8", 1)])


class TestPipelineVocabulary(unittest.TestCase):
    """The MNI derivatives use ONCOHabitats names, not BIDS suffixes."""

    CASES = [
        ("T1.nii.gz", "t1w", "raw", "not_annotation", None),
        ("T1c.nii.gz", "t1wc", "raw", "not_annotation", None),
        ("T2.nii.gz", "t2w", "raw", "not_annotation", None),
        ("Flair.nii.gz", "t2wflair", "raw", "not_annotation", None),
        ("T1c-icor.nii.gz", "t1wc", "icor", "not_annotation", None),
        ("T1c-icor-zscore.nii.gz", "t1wc", "icor-zscore", "not_annotation", None),
        ("ContrastEnhancedMask-CL.nii.gz", None, "raw", "CL", "enhancing_t1wc"),
        ("ContrastEnhancedMask-ONCO.nii.gz", None, "raw", "ONCO", "enhancing_t1wc"),
        ("EdemaMask-CL.nii.gz", None, "raw", "CL", "edema"),
        ("NecrosisMask-ONCO.nii.gz", None, "raw", "ONCO", "necrosis"),
        ("Segmentation-ONCO.nii.gz", None, "raw", "multilabel_segmentation", None),
        ("fastsurfer-segmentation.nii.gz", None, "raw", "multilabel_segmentation", None),
        ("BrainExtractionMask.nii.gz", None, "raw", "brain_mask", None),
        ("NAWMask.nii.gz", None, "raw", "nawm_mask", None),
        ("rCBV.nii.gz", "rcbv", "raw", "not_annotation", None),
    ]

    def test_real_filenames_resolve(self):
        for base, seq, variant, kind, comp in self.CASES:
            row = naming.build_file_table([f"d/sub-01/ses-01/{base}"])[0]
            self.assertEqual(row["sequence"], seq, base)
            self.assertEqual(row["intensity_variant"], variant, base)
            self.assertEqual(row["annotation_kind"], kind, base)
            self.assertEqual(row["annotation_component"], comp, base)

    def test_t1c_not_swallowed_by_t1(self):
        rows = naming.build_file_table(["d/sub-01/ses-01/T1.nii.gz",
                                        "d/sub-01/ses-01/T1c.nii.gz"])
        self.assertEqual({r["sequence"] for r in rows}, {"t1w", "t1wc"})

    def test_image_never_carries_an_annotation_component(self):
        row = naming.build_file_table(["d/sub-01/ses-01/Flair.nii.gz"])[0]
        self.assertIsNone(row["annotation_component"])

    def test_ambiguous_mask_is_flagged_not_guessed(self):
        row = naming.build_file_table(["d/sub-01/ses-01/Mask.nii.gz"])[0]
        self.assertEqual(row["annotation_kind"], "UNRESOLVED_MASKLIKE")

    def test_intervals_filename_hyphen_and_underscore_both_match(self):
        from sailor.stage1.clinical_table import INTERVALS_FILE
        for n in ("sub-01/intervals-days.txt", "sub-01/intervals_days.txt",
                  "sub-01/intevals_days.txt"):
            self.assertTrue(INTERVALS_FILE.search(n), n)


class TestRawMniLink(unittest.TestCase):
    LINK = ("subject\traw session\tmni session\n"
            "sub-01\tses-01\tno\n"
            "sub-01\tses-02\tno\n"
            "sub-01\tses-03\tses-01\n"
            "sub-01\tses-04\tses-02\n")

    def test_no_counterpart_rows_are_counted_not_discarded(self):
        r = tables.parse_raw_mni_link(self.LINK)
        self.assertEqual(r["n_raw_without_mni"], 2)
        self.assertEqual(r["n_mapped"], 2)

    def test_mni_ses01_is_not_the_first_examination(self):
        r = tables.parse_raw_mni_link(self.LINK)
        self.assertEqual(r["mni_ses01_maps_to_raw"]["sub-01"], "ses-03")
        self.assertEqual(r["n_raw_exams_dropped_before_mni_ses01"]["sub-01"], 2)

    def test_unexpected_mni_value_is_flagged_not_silently_dropped(self):
        bad = self.LINK + "sub-01\tses-05\tmaybe\n"
        r = tables.parse_raw_mni_link(bad)
        self.assertEqual(len(r["unparsed_mni_values"]), 1)

    def test_g8_reports_dropped_pre_mni_exams(self):
        link = tables.parse_raw_mni_link(self.LINK)
        rec = guards.g8_session_correspondence(
            link, {"sub-01": ["ses-01", "ses-02"]}, {}, raw_side_scanned=False)
        self.assertEqual(rec["evidence"]["n_raw_without_mni_counterpart"], 2)
        self.assertIn("NOT", rec["detail"])


class TestKnownIssues(unittest.TestCase):
    def test_leakage_sessions_are_excluded_by_default(self):
        from sailor.data import known_issues as K
        ex = K.excluded_sessions()
        self.assertIn(("sub-04", "ses-01"), ex)
        self.assertIn(("sub-05", "ses-01"), ex)

    def test_leakage_can_be_listed_separately_from_hard_exclusions(self):
        from sailor.data import known_issues as K
        without = K.excluded_sessions(include_leakage=False)
        self.assertNotIn(("sub-04", "ses-01"), without)
        self.assertIn(("sub-05", "ses-05"), without)

    def test_primary_cohort_is_26_not_27(self):
        from sailor.data import known_issues as K
        s = K.summary()
        self.assertEqual(s["n_patients_primary_cohort"], 26)
        self.assertEqual(s["subjects_without_primary_target"], ["sub-24"])

    def test_dose_eligibility_excludes_missing_and_unscaled(self):
        from sailor.data import known_issues as K
        d = K.dose_eligible_subjects([f"sub-{i:02d}" for i in range(1, 28)])
        self.assertEqual(d["n_eligible"], 25)
        self.assertEqual(d["blocked"], ["sub-19", "sub-26"])

    def test_delta_t_flags_interpolated_subjects(self):
        from sailor.data import known_issues as K
        self.assertEqual(K.delta_t_flag("sub-13")["kind"], "ESTIMATED")
        self.assertEqual(K.delta_t_flag("sub-01")["kind"], "DOCUMENTED_APPROXIMATE")

    def test_every_issue_carries_its_source_quote(self):
        from sailor.data import known_issues as K
        for e in K.SESSION_ISSUES + K.SUBJECT_ISSUES:
            self.assertTrue(e.get("quote"), f"{e} has no verbatim quote")


class TestG7Documented(unittest.TestCase):
    def test_intervals_present_but_never_reported_exact(self):
        ct = {"rows": [{"subject": "sub-13", "session": "ses-01"},
                       {"subject": "sub-01", "session": "ses-01"}],
              "n_sessions_with_days_from_first": 2, "interval_problems": []}
        rec = guards.g7_delta_t_provenance([], {}, clinical_table=ct)
        self.assertEqual(rec["status"], "FAIL")
        self.assertFalse(rec["evidence"]["exact_source_available"])
        self.assertIn("sub-13", rec["evidence"]["subjects_interpolated"])

    def test_unread_intervals_are_inconclusive_not_approximate(self):
        ct = {"rows": [{"subject": "sub-01", "session": "ses-01"}],
              "n_sessions_with_days_from_first": 0, "interval_problems": []}
        rec = guards.g7_delta_t_provenance([], {}, clinical_table=ct)
        self.assertEqual(rec["status"], "INCONCLUSIVE")


class TestPartialScan(unittest.TestCase):
    def test_unscanned_raw_side_is_not_reported_as_zero_unmatched(self):
        link = {"status": "OK", "n_rows": 1, "pairs": [
            {"subject": "sub-01", "raw_session": "ses-01", "mni_session": "ses-01"}]}
        rec = guards.g8_session_correspondence(
            link, {"sub-01": ["ses-01"]}, {}, raw_side_scanned=False)
        self.assertEqual(rec["evidence"]["unmatched_raw_sessions"], "UNSCANNED")
        self.assertIn("UNVERIFIED", rec["detail"])

    def test_scanned_raw_side_reports_real_counts(self):
        link = {"status": "OK", "n_rows": 1, "pairs": [
            {"subject": "sub-01", "raw_session": "ses-01", "mni_session": "ses-01"}]}
        rec = guards.g8_session_correspondence(
            link, {"sub-01": ["ses-01"]}, {"sub-01": ["ses-01", "ses-09"]},
            raw_side_scanned=True)
        self.assertEqual(rec["evidence"]["unmatched_raw_sessions"], ["sub-01/ses-09"])


class TestPersistence(unittest.TestCase):
    """A 90-minute pass must survive a kernel restart (§15.4)."""

    def _fixture(self):
        import tempfile
        from pathlib import Path as P
        from sailor.config import Paths
        from tests import make_fixture
        tmp = P(tempfile.mkdtemp())
        make_fixture.build(tmp / "sailor_v1")
        return Paths(project_name="T", dataset_root=tmp / "T",
                     legacy_root=tmp / "sailor_v1", code_root=P("."))

    def test_clinical_collect_writes_artefacts(self):
        from sailor.stage1.clinical_table import collect
        from pathlib import Path as P
        paths = self._fixture()
        t = collect(paths, verbose=False)
        self.assertTrue(P(t["artefacts"]["json"]["latest"]).exists())
        self.assertTrue(P(t["artefacts"]["csv"]).exists())
        self.assertTrue(P(t["artefacts"]["history_txt"]).exists())

    def test_second_collect_reuses_cache_and_matches(self):
        from sailor.stage1.clinical_table import collect
        paths = self._fixture()
        a = collect(paths, verbose=False)
        b = collect(paths, verbose=False)
        self.assertFalse(a["from_cache"])
        self.assertTrue(b["from_cache"])
        self.assertEqual(a["rows"], b["rows"])

    def test_force_rescan_bypasses_cache(self):
        from sailor.stage1.clinical_table import collect
        paths = self._fixture()
        collect(paths, verbose=False)
        c = collect(paths, verbose=False, force_rescan=True)
        self.assertFalse(c["from_cache"])

    def test_cache_write_is_atomic(self):
        """A killed kernel must not leave a truncated cache that loads as valid."""
        from sailor.utils.persist import cache_path, load_cache, save_cache
        import tempfile
        from pathlib import Path as P
        root = P(tempfile.mkdtemp())
        save_cache(root, "k", {"a": "b"})
        self.assertEqual(load_cache(root, "k")["raw"], {"a": "b"})
        cache_path(root, "k").write_text("{ truncated")
        self.assertIsNone(load_cache(root, "k"))


class TestContracts(unittest.TestCase):
    def test_target_lock_assertion_rejects_mismatch(self):
        from sailor.contracts import assert_target_lock
        with self.assertRaises(AssertionError):
            assert_target_lock({"primary_target_mask": "ONCO",
                                "primary_target_component": "enhancing_t1wc"},
                               "CL", "enhancing_t1wc")


if __name__ == "__main__":
    unittest.main(verbosity=2)
