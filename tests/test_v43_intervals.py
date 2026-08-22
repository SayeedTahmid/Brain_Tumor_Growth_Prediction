"""v0.43 — a point estimate against a threshold is not an interval.

`v2_ladder_complete.json` reported `gap` and `paired_sd` per rung and no
interval, so "no rung beats persistence" rested on comparing a point estimate to
the frozen MDE. These tests pin the interval machinery and the property that
made it worth adding: every rung's interval crosses zero, which is a stronger
and more honest statement than failing to exceed a threshold.

Also pins the partial-coverage rule for audit selection. The 11 Aug pass
reported G1 = PASS having read 54 of 240 masks; reading all 240 surfaced seven
all-zero masks and turned it FAIL.
"""

import numpy as np
import pytest

from sailor.stage1 import audit
from sailor.stage4 import ladder_analysis as LA


def _pp(values_by_rung):
    """Build a per_patient() shaped result from {rung: [26 gaps]}."""
    rungs = list(values_by_rung)
    n = len(next(iter(values_by_rung.values())))
    rows = []
    for i in range(n):
        per_rung = {r: values_by_rung[r][i] for r in rungs}
        rows.append({"subject": f"sub-{i + 1:02d}", "per_rung": per_rung,
                     "mean_across_rungs": float(np.mean(list(per_rung.values())))})
    return {"rungs": rungs, "n_patients": n, "per_patient": rows}


class TestIntervals:

    def test_mean_matches_the_per_patient_table(self):
        vals = list(np.linspace(-0.02, 0.03, 26))
        r = LA.intervals(_pp({"C0": vals}))
        assert r["per_rung"]["C0"]["mean"] == pytest.approx(float(np.mean(vals)))

    def test_a_null_rung_crosses_zero(self):
        rng = np.random.default_rng(0)
        vals = list(rng.normal(0.001, 0.02, 26))
        r = LA.intervals(_pp({"C0": vals}))
        assert r["per_rung"]["C0"]["crosses_zero"] is True
        assert r["rungs_beating_persistence"] == []

    def test_a_clearly_better_rung_is_detected(self):
        """Guards against an interval that can never exclude zero."""
        vals = [-0.20] * 26
        r = LA.intervals(_pp({"C0": vals}))
        assert r["per_rung"]["C0"]["crosses_zero"] is False
        assert r["rungs_beating_persistence"] == ["C0"]

    def test_sign_convention_is_recorded(self):
        r = LA.intervals(_pp({"C0": [0.001] * 26}))
        assert "WORSE" in r["per_rung"]["C0"]["sign_convention"]

    def test_mde_is_read_not_recomputed(self):
        assert LA.FROZEN_MDE == 0.0555
        r = LA.intervals(_pp({"C0": [0.001] * 26}))
        assert r["frozen_mde"] == 0.0555
        assert r["per_rung"]["C0"]["exceeds_mde"] is False

    def test_exceeds_mde_is_two_sided(self):
        """A rung far BELOW persistence also exceeds the MDE."""
        r = LA.intervals(_pp({"C0": [-0.10] * 26}))
        assert r["per_rung"]["C0"]["exceeds_mde"] is True

    def test_missing_rung_values_are_skipped_not_zeroed(self):
        pp = _pp({"C0": [0.01] * 26})
        pp["per_patient"][0]["per_rung"]["C0"] = None
        r = LA.intervals(pp)
        assert r["per_rung"]["C0"]["n_patients"] == 25

    def test_reuses_the_persistence_bootstrap(self):
        """Rungs and floor must not drift apart in seed or convention."""
        from sailor.stage3 import persistence as P
        vals = list(np.linspace(-0.02, 0.03, 26))
        mine = LA.intervals(_pp({"C0": vals}))["per_rung"]["C0"]
        theirs = P._patient_bootstrap(
            {f"sub-{i + 1:02d}": [v] for i, v in enumerate(vals)})
        assert mine["ci_low"] == theirs["ci_low"]
        assert mine["ci_high"] == theirs["ci_high"]

    def test_power_caveat_is_present(self):
        r = LA.intervals(_pp({"C0": [0.001] * 26}))
        assert "NOT evidence that the true effect is zero" in r["power_caveat"]


class TestPartialCoverageRejected:

    def _audit(self, name, g1_status, g1_n, g1_files, g10_status, g10_n):
        return name, {"guards": {"records": [
            {"guard": "G1", "status": g1_status,
             "evidence": {"primary": {"n_measured": g1_n, "n_files": g1_files}}},
            {"guard": "G10", "status": g10_status,
             "evidence": {"n_images_measured": g10_n}},
        ]}}

    @pytest.fixture
    def qc(self, tmp_path):
        import json
        d = tmp_path / "06_QC_REPORTS"
        d.mkdir(parents=True)

        def write(pairs):
            for name, body in pairs:
                (d / name).write_text(json.dumps(body))
            return tmp_path
        return write

    def test_pass_on_partial_coverage_is_rejected(self):
        """The 11 Aug case: PASS on 54 of 240."""
        rec = {"guard": "G1", "status": "PASS",
               "evidence": {"primary": {"n_measured": 54, "n_files": 240}}}
        assert audit._available_count(rec) == 240
        assert audit._measured_count(rec) == 54

    def test_partial_audit_does_not_win(self, qc):
        root = qc([
            self._audit("v2_stage1_audit_20260819T000000Z.json",
                        "PASS", 54, 240, "PASS", 100),
            self._audit("v2_stage1_audit_20260814T142238Z.json",
                        "FAIL", 240, 240, "FAIL", 1479),
        ])
        r = audit.latest_measured_audit(root)
        assert r["verdict"] == "RESOLVED"
        assert "20260814" in r["deciding_audit"]["file"]

    def test_full_coverage_is_reported_per_guard(self, qc):
        root = qc([self._audit("v2_stage1_audit_20260814T142238Z.json",
                               "FAIL", 240, 240, "FAIL", 1479)])
        g = audit.latest_measured_audit(root)["deciding_audit"]["guards"]
        assert g["G1"]["full_coverage"] is True

    def test_unknown_coverage_does_not_disqualify(self, qc):
        """G10 records no total; absence of a total is not evidence of partiality."""
        root = qc([self._audit("v2_stage1_audit_20260814T142238Z.json",
                               "FAIL", 240, 240, "FAIL", 1479)])
        g = audit.latest_measured_audit(root)["deciding_audit"]["guards"]
        assert g["G10"]["full_coverage"] is None
        assert audit.latest_measured_audit(root)["verdict"] == "RESOLVED"


class TestGateDatesMatchTheirArtefacts:

    def test_decided_dates_are_not_before_the_evidence(self):
        from sailor.experiments import gates
        o = gates.protocol()["gate_outcomes"]
        assert o["GATE-0"]["decided_utc"].startswith("2026-08-22")
        assert o["GATE-4"]["decided_utc"].startswith("2026-08-22")
