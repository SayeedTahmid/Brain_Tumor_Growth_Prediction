"""Tests for Phase 4: pairs and patient-level folds."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sailor.data import pairs as P  # noqa: E402
from sailor.data import splits as S  # noqa: E402


def rows(subject="sub-01", n=5, start_ordinal=1, day_step=30.0):
    return [{"subject": subject, "session": f"ses-{i:02d}", "session_ordinal": i,
             "days_from_first": (i - start_ordinal) * day_step,
             "treatment_status": "CRT" if i <= 3 else "TMZ",
             "treatment_observed": True, "rano": 2}
            for i in range(start_ordinal, start_ordinal + n)]


class TestPairs(unittest.TestCase):
    def test_consecutive_gives_n_minus_one(self):
        b = P.build_pairs(rows(n=5), apply_known_exclusions=False)
        self.assertEqual(b["n_pairs"], 4)
        self.assertTrue(all(p["gap_in_ordinals"] == 1 for p in b["pairs"]))

    def test_all_ordered_gives_n_choose_two(self):
        b = P.build_pairs(rows(n=5), mode=P.ALL_ORDERED,
                          apply_known_exclusions=False)
        self.assertEqual(b["n_pairs"], 10)

    def test_pair_touching_a_leakage_session_is_dropped(self):
        """sub-04/ses-01's mask came from its own ses-02 — the pair is poisoned."""
        b = P.build_pairs(rows("sub-04", n=3))
        self.assertNotIn(("ses-01", "ses-02"),
                         [(p["input_session"], p["target_session"]) for p in b["pairs"]])
        self.assertIn("touches_excluded_session", b["rejected"])

    def test_subject_without_primary_target_contributes_nothing(self):
        b = P.build_pairs(rows("sub-24", n=6))
        self.assertEqual(b["n_pairs"], 0)
        self.assertIn("subject_has_no_primary_target", b["rejected"])

    def test_missing_target_mask_rejects_the_pair(self):
        r = rows(n=3)
        have = {("sub-01", "ses-01"), ("sub-01", "ses-02")}  # ses-03 has no mask
        b = P.build_pairs(r, sessions_with_target=have,
                          apply_known_exclusions=False)
        self.assertEqual(b["n_pairs"], 1)
        self.assertIn("target_session_lacks_primary_target", b["rejected"])

    def test_missing_delta_t_rejects_rather_than_defaults(self):
        r = rows(n=3)
        r[1]["days_from_first"] = None
        b = P.build_pairs(r, apply_known_exclusions=False)
        self.assertIn("delta_t_unavailable", b["rejected"])

    def test_non_positive_delta_t_rejected(self):
        r = rows(n=2)
        r[1]["days_from_first"] = r[0]["days_from_first"]
        b = P.build_pairs(r, apply_known_exclusions=False)
        self.assertEqual(b["n_pairs"], 0)
        self.assertIn("non_positive_delta_t", b["rejected"])

    def test_interpolated_delta_t_is_flagged_on_the_pair(self):
        b = P.build_pairs(rows("sub-13", n=4), apply_known_exclusions=False)
        self.assertTrue(all(p["delta_t_kind"] == "INTERPOLATED" for p in b["pairs"]))
        b2 = P.build_pairs(rows("sub-01", n=4), apply_known_exclusions=False)
        self.assertTrue(all(p["delta_t_kind"] == "DOCUMENTED_APPROXIMATE"
                            for p in b2["pairs"]))

    def test_day_window_filters_are_reported(self):
        b = P.build_pairs(rows(n=5), apply_known_exclusions=False, max_days=1.0)
        self.assertEqual(b["n_pairs"], 0)
        self.assertIn("above_max_days", b["rejected"])

    def test_unknown_target_availability_is_reported_not_assumed(self):
        b = P.build_pairs(rows(n=3), apply_known_exclusions=False)
        self.assertFalse(b["target_availability_known"])


class TestFolds(unittest.TestCase):
    PATIENTS = [f"sub-{i:02d}" for i in range(1, 27)]

    def test_split_unit_is_patient(self):
        f = S.make_folds(self.PATIENTS, n_folds=5, n_repeats=2)
        self.assertEqual(f["unit_of_split"], "patient")
        for rep in f["repeats"]:
            for fold in rep["folds"]:
                self.assertFalse(set(fold["train_patients"])
                                 & set(fold["test_patients"]))

    def test_every_patient_is_tested_at_least_once(self):
        f = S.make_folds(self.PATIENTS, n_folds=5, n_repeats=3)
        tested = set()
        for rep in f["repeats"]:
            for fold in rep["folds"]:
                tested.update(fold["test_patients"])
        self.assertEqual(tested, set(self.PATIENTS))

    def test_same_seed_reproduces_folds(self):
        a = S.make_folds(self.PATIENTS, seed=99)
        b = S.make_folds(self.PATIENTS, seed=99)
        self.assertEqual(a["repeats"], b["repeats"])

    def test_different_seed_changes_folds(self):
        a = S.make_folds(self.PATIENTS, seed=1)
        b = S.make_folds(self.PATIENTS, seed=2)
        self.assertNotEqual(a["repeats"], b["repeats"])

    def test_too_many_folds_raises(self):
        with self.assertRaises(ValueError):
            S.make_folds(["sub-01", "sub-02"], n_folds=5)

    def test_g5_catches_a_patient_in_both_sides(self):
        f = S.make_folds(self.PATIENTS, n_folds=5, n_repeats=1)
        f["repeats"][0]["folds"][0]["train_patients"].append(
            f["repeats"][0]["folds"][0]["test_patients"][0])
        rec = S.g5_fold_leakage(f, [])
        self.assertEqual(rec["status"], "FAIL")
        self.assertIn("patient_in_train_and_test",
                      {p["type"] for p in rec["evidence"]["problems"]})

    def test_g5_catches_a_pair_touching_an_excluded_session(self):
        f = S.make_folds(self.PATIENTS, n_folds=5, n_repeats=1)
        bad = [{"subject": "sub-04", "input_session": "ses-01",
                "target_session": "ses-02"}]
        rec = S.g5_fold_leakage(f, bad, excluded_sessions={("sub-04", "ses-01")})
        self.assertEqual(rec["status"], "FAIL")

    def test_g5_passes_on_a_clean_split(self):
        f = S.make_folds(self.PATIENTS, n_folds=5, n_repeats=2)
        rec = S.g5_fold_leakage(f, [{"subject": "sub-01",
                                     "input_session": "ses-01",
                                     "target_session": "ses-02"}])
        self.assertEqual(rec["status"], "PASS")


class TestFreeze(unittest.TestCase):
    def _build(self):
        r = []
        for i in range(1, 27):
            r += rows(f"sub-{i:02d}", n=5)
        b = P.build_pairs(r, apply_known_exclusions=False)
        f = S.make_folds([f"sub-{i:02d}" for i in range(1, 27)],
                         n_folds=5, n_repeats=2)
        return b, f, S.assign_pairs(b["pairs"], f)

    def test_freeze_writes_a_hashed_manifest(self):
        b, f, a = self._build()
        out = Path(tempfile.mkdtemp())
        res = S.freeze(f, b, a, out, decisions={"pair_mode": "consecutive"})
        self.assertTrue(Path(res["path"]).exists())
        self.assertEqual(len(res["content_sha256"]), 64)

    def test_frozen_manifest_verifies(self):
        b, f, a = self._build()
        out = Path(tempfile.mkdtemp())
        res = S.freeze(f, b, a, out, decisions={})
        self.assertTrue(S.verify_frozen(res["path"])["intact"])

    def test_tampered_manifest_fails_verification(self):
        import json
        b, f, a = self._build()
        out = Path(tempfile.mkdtemp())
        res = S.freeze(f, b, a, out, decisions={})
        m = json.loads(Path(res["path"]).read_text())
        m["folds"]["repeats"][0]["folds"][0]["test_patients"].append("sub-99")
        Path(res["path"]).write_text(json.dumps(m))
        self.assertFalse(S.verify_frozen(res["path"])["intact"])

    def test_manifest_carries_the_target_lock(self):
        from sailor.contracts import assert_target_lock
        b, f, a = self._build()
        res = S.freeze(f, b, a, Path(tempfile.mkdtemp()), decisions={})
        assert_target_lock(res["manifest"], "CL", "enhancing_t1wc")


if __name__ == "__main__":
    unittest.main(verbosity=2)
