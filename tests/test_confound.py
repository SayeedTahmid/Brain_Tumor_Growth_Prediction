"""Unit tests for the confound quantification (§5, G2).

The estimator's job is to be *right about the bad case*: a cohort where every
patient follows the same protocol must come out UNTENABLE, and the P1 control
must be reported as powerless there.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sailor.experiments import confound  # noqa: E402

SCHEDULE = [0, 2, 5, 7, 9, 12, 16, 20, 24, 28]


def stupp_label(week: float) -> str:
    return "no" if week < 4 else ("CRT" if week < 10 else "TMZ")


def cohort(n_patients=27, jitter=0.3, seed=0):
    rng = np.random.default_rng(seed)
    status, weeks, patient = [], [], []
    for p in range(n_patients):
        for w in SCHEDULE:
            status.append(stupp_label(w))
            weeks.append(w + rng.normal(0, jitter))
            patient.append(f"sub-{p:02d}")
    return status, weeks, patient


class TestConfound(unittest.TestCase):
    def test_uniform_protocol_is_untenable(self):
        r = confound.quantify(*cohort(), n_permutations=200)
        self.assertEqual(r.verdict, "UNTENABLE")
        self.assertGreater(r.uncertainty_coefficient, confound.U_UNTENABLE)
        self.assertGreaterEqual(r.time_only_balanced_accuracy, 0.95)
        self.assertLess(r.permutation_p, 0.05)

    def test_uniform_protocol_makes_P1_powerless(self):
        r = confound.quantify(*cohort(), n_permutations=200)
        self.assertFalse(r.p1_control_has_power)
        self.assertGreaterEqual(r.schedule_homogeneity, 0.80)

    def test_random_labels_are_tractable(self):
        status, weeks, patient = cohort()
        rng = np.random.default_rng(7)
        status = [rng.choice(["no", "CRT", "TMZ"]) for _ in status]
        r = confound.quantify(status, weeks, patient, n_permutations=200)
        self.assertEqual(r.verdict, "TRACTABLE")
        self.assertLess(r.uncertainty_coefficient, 0.2)
        self.assertGreater(r.permutation_p, 0.05)

    def test_homogeneity_undefined_when_no_association(self):
        status, weeks, patient = cohort()
        rng = np.random.default_rng(7)
        status = [rng.choice(["no", "CRT", "TMZ"]) for _ in status]
        r = confound.quantify(status, weeks, patient, n_permutations=200)
        self.assertTrue(np.isnan(r.schedule_homogeneity))

    def test_missing_status_rows_are_dropped_not_imputed(self):
        status, weeks, patient = cohort(n_patients=6)
        status = [None if i % 5 == 0 else s for i, s in enumerate(status)]
        r = confound.quantify(status, weeks, patient, n_permutations=50)
        self.assertEqual(r.n_timepoints, sum(1 for s in status if s is not None))
        self.assertNotIn("unknown", r.class_counts)

    def test_missing_time_rows_are_dropped(self):
        status, weeks, patient = cohort(n_patients=6)
        weeks = [None if i % 7 == 0 else w for i, w in enumerate(weeks)]
        r = confound.quantify(status, weeks, patient, n_permutations=50)
        self.assertEqual(r.n_timepoints, sum(1 for w in weeks if w is not None))

    def test_thresholds_are_pre_specified_and_reported(self):
        r = confound.quantify(*cohort(n_patients=6), n_permutations=50)
        pol = r.threshold_policy
        self.assertEqual(pol["U_untenable"], confound.U_UNTENABLE)
        self.assertIn("immutability", pol)
        self.assertIn("p1_powerlessness", pol)

    def test_wiring_covers_every_rung_and_control(self):
        w = confound.wiring()
        for key in ["C-1", "C0", "C1", "C2", "C3", "C4", "P1", "P2", "P3"]:
            self.assertTrue(any(k.startswith(key + "_") or k.startswith(key)
                                for k in w), f"{key} not wired")

    def test_time_bins_are_clinical_not_data_driven(self):
        b = confound.discretise_time(np.array([0.0, 3.9, 4.0, 9.9, 10.0, 60.0]))
        self.assertEqual(b.tolist(), [0, 0, 1, 1, 2, 5])

    def test_ordinal_basis_recorded_not_silently_substituted(self):
        rows_ = [{"subject": f"sub-{p:02d}", "session": f"ses-{i:02d}",
                  "session_ordinal": i,
                  "treatment_status": "CRT" if i <= 3 else "TMZ"}
                 for p in range(8) for i in range(1, 7)]
        import tempfile
        from pathlib import Path
        out = Path(tempfile.mkdtemp())
        payload = confound.run_and_write(rows_, out)
        self.assertEqual(payload["confound_measurement"]["time_basis"],
                         confound.ORDINAL)
        self.assertTrue((out / "v2_confound_prereg.json").exists())

    def test_weeks_basis_used_when_available(self):
        rows_ = [{"subject": f"sub-{p:02d}", "session": f"ses-{i:02d}",
                  "session_ordinal": i,
                  "treatment_status": "CRT" if i <= 3 else "TMZ"}
                 for p in range(8) for i in range(1, 7)]
        weeks = {f"{r['subject']}/{r['session']}": r["session_ordinal"] * 4.0
                 for r in rows_}
        import tempfile
        from pathlib import Path
        payload = confound.run_and_write(rows_, Path(tempfile.mkdtemp()),
                                         weeks_by_session=weeks)
        self.assertEqual(payload["confound_measurement"]["time_basis"],
                         confound.WEEKS)

    def test_trivial_rule_audit_separates_pure_from_mixed_ordinals(self):
        rows_ = ([{"subject": f"sub-{p:02d}", "session_ordinal": 1,
                   "treatment_status": "CRT"} for p in range(10)]
                 + [{"subject": f"sub-{p:02d}", "session_ordinal": 2,
                     "treatment_status": "CRT" if p < 5 else "TMZ"}
                    for p in range(10)])
        a = confound.trivial_rule_audit(rows_)
        self.assertEqual(sorted(a["unambiguous_ordinals"]), [1])
        self.assertEqual(sorted(a["mixed_ordinals"]), [2])
        self.assertEqual(a["n_determined_by_ordinal_alone"], 10)

    def test_prereg_payload_carries_the_binding_not_just_the_number(self):
        rows_ = [{"subject": f"sub-{p:02d}", "session": f"ses-{i:02d}",
                  "session_ordinal": i,
                  "treatment_status": "CRT" if i <= 3 else "TMZ"}
                 for p in range(10) for i in range(1, 7)]
        import tempfile
        from pathlib import Path
        payload = confound.run_and_write(rows_, Path(tempfile.mkdtemp()))
        self.assertEqual(payload["confound_measurement"]["verdict"], "UNTENABLE")
        self.assertIn("dose maps", payload["verdict_binding"])
        self.assertIn("ordinal_time_caveat",
                      payload["confound_measurement"]["threshold_policy"])

    def test_bin_count_alone_can_move_the_verdict(self):
        """The reason a basis comparison must be at matched resolution."""
        import numpy as np
        status, weeks, patient = cohort()
        ordv = [SCHEDULE.index(min(SCHEDULE, key=lambda s: abs(s - w))) + 1
                for w in weeks]
        saved = confound.discretise_time
        try:
            confound.discretise_time = lambda w, edges=None: np.asarray(w).astype(int)
            fine = confound.quantify(status, ordv, patient, n_permutations=100)
            confound.discretise_time = lambda w, edges=None: np.digitize(w, [4, 7])
            coarse = confound.quantify(status, ordv, patient, n_permutations=100)
        finally:
            confound.discretise_time = saved
        self.assertGreater(fine.uncertainty_coefficient,
                           coarse.uncertainty_coefficient)
        self.assertGreater(fine.time_only_balanced_accuracy,
                           coarse.time_only_balanced_accuracy)

    def test_binding_verdict_is_the_most_confounded_estimate(self):
        import tempfile
        from pathlib import Path
        status, weeks, patient = cohort(n_patients=12)
        rows_ = [{"subject": p, "session": f"ses-{i:02d}", "session_ordinal": i,
                  "treatment_status": s}
                 for p, s, i in zip(patient, status,
                                    [SCHEDULE.index(min(SCHEDULE,
                                     key=lambda x: abs(x - w))) + 1 for w in weeks])]
        wk = {f"{r['subject']}/{r['session']}": r["session_ordinal"] * 4.0
              for r in rows_}
        payload = confound.run_and_write(rows_, Path(tempfile.mkdtemp()),
                                         weeks_by_session=wk)
        sens = payload["basis_sensitivity"]
        order = {"TRACTABLE": 0, "SEVERE": 1, "UNTENABLE": 2}
        worst = max(order[r["verdict"]] for r in sens["rows"])
        self.assertEqual(order[payload["binding_verdict"]], worst)

    def test_conservative_rule_is_stated_in_the_policy(self):
        pol = confound.policy()
        self.assertIn("conservative_basis_rule", pol)
        self.assertIn("basis_and_resolution", pol)

    def test_delta_t_that_separates_phases_flags_c1_contamination(self):
        rng = np.random.default_rng(0)
        pairs = []
        for p in range(20):
            for _ in range(4):
                pairs.append({"subject": f"sub-{p:02d}", "input_treatment": "CRT",
                              "delta_days": float(rng.integers(7, 21))})
            for _ in range(5):
                pairs.append({"subject": f"sub-{p:02d}", "input_treatment": "TMZ",
                              "delta_days": float(rng.integers(70, 100))})
        r = confound.delta_t_encodes_treatment(pairs, n_permutations=200)
        self.assertTrue(r["c1_is_treatment_contaminated"])
        self.assertGreater(r["median_ratio_across_phases"], 2.0)

    def test_uniform_intervals_leave_c1_usable(self):
        rng = np.random.default_rng(1)
        pairs = [{"subject": f"sub-{p:02d}",
                  "input_treatment": rng.choice(["CRT", "TMZ"]),
                  "delta_days": float(rng.integers(28, 35))}
                 for p in range(20) for _ in range(9)]
        r = confound.delta_t_encodes_treatment(pairs, n_permutations=200)
        self.assertFalse(r["c1_is_treatment_contaminated"])

    def test_no_data_is_reported_not_zero(self):
        r = confound.delta_t_encodes_treatment([{"subject": "s"}])
        self.assertEqual(r["status"], "NO_DATA")

    def test_empty_input_raises_rather_than_returning_a_number(self):
        with self.assertRaises(ValueError):
            confound.quantify([None, None], [1.0, 2.0], ["a", "b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
