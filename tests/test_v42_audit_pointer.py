"""v0.42 — an audit pointer must resolve by completeness, not recency.

`persist.latest_full_pass` fixed the cache-key version of this (defect 24).
This is the artefact version: `v2_stage1_audit_latest.json` names the most
recent audit, and the most recent audit is often a structural pass in which G1
and G10 are INCONCLUSIVE by construction. A caller asking "did G1 pass?" gets
INCONCLUSIVE while a full pass measuring 240 masks sits two files away.
"""

import json

import pytest

from sailor.stage1 import audit


def _audit(name, g1_status, g1_n, g10_status, g10_n, when):
    return name, {
        "generated_utc": when,
        "guards": {"records": [
            {"guard": "G1", "status": g1_status,
             "evidence": {"primary": {"n_measured": g1_n}}},
            {"guard": "G10", "status": g10_status,
             "evidence": {"n_images_measured": g10_n}},
        ]},
    }


@pytest.fixture
def qc(tmp_path):
    d = tmp_path / "06_QC_REPORTS"
    d.mkdir(parents=True)

    def write(pairs):
        for name, body in pairs:
            (d / name).write_text(json.dumps(body))
        return tmp_path
    return write


class TestSelectsByCompleteness:

    def test_newest_structural_pass_does_not_win(self, qc):
        root = qc([
            _audit("v2_stage1_audit_20260820T223430Z.json",
                   "INCONCLUSIVE", 0, "INCONCLUSIVE", 0, "2026-08-20"),
            _audit("v2_stage1_audit_20260814T142238Z.json",
                   "FAIL", 240, "FAIL", 1479, "2026-08-14"),
        ])
        r = audit.latest_measured_audit(root)
        assert r["verdict"] == "RESOLVED"
        assert "20260814" in r["deciding_audit"]["file"]

    def test_fail_counts_as_measured(self, qc):
        """GATE-0 asks for a measured verdict, not a clean one."""
        root = qc([_audit("v2_stage1_audit_20260814T142238Z.json",
                          "FAIL", 240, "FAIL", 1479, "2026-08-14")])
        assert audit.latest_measured_audit(root)["verdict"] == "RESOLVED"

    def test_partial_measurement_is_not_enough(self, qc):
        """G1 measured, G10 not: the 15 Aug pass. Must not be selected."""
        root = qc([_audit("v2_stage1_audit_20260815T210904Z.json",
                          "FAIL", 240, "INCONCLUSIVE", 0, "2026-08-15")])
        r = audit.latest_measured_audit(root)
        assert r["verdict"] == "UNRESOLVED"

    def test_status_without_count_is_rejected(self, qc):
        """A guard claiming PASS having read nothing is not measurement."""
        root = qc([_audit("v2_stage1_audit_20260811T205528Z.json",
                          "PASS", 0, "PASS", 0, "2026-08-11")])
        assert audit.latest_measured_audit(root)["verdict"] == "UNRESOLVED"

    def test_newest_qualifying_wins_among_several(self, qc):
        root = qc([
            _audit("v2_stage1_audit_20260814T142238Z.json",
                   "FAIL", 240, "FAIL", 1479, "2026-08-14"),
            _audit("v2_stage1_audit_20260812T213527Z.json",
                   "FAIL", 240, "FAIL", 60, "2026-08-12"),
        ])
        assert "20260814" in (audit.latest_measured_audit(root)
                              ["deciding_audit"]["file"])


class TestFailsLoudly:

    def test_no_audits_at_all_is_unresolved(self, tmp_path):
        (tmp_path / "06_QC_REPORTS").mkdir(parents=True)
        r = audit.latest_measured_audit(tmp_path)
        assert r["verdict"] == "UNRESOLVED"
        assert "read_volumes=True" in r["detail"]

    def test_unreadable_file_does_not_crash_the_scan(self, qc):
        root = qc([_audit("v2_stage1_audit_20260814T142238Z.json",
                          "FAIL", 240, "FAIL", 1479, "2026-08-14")])
        (root / "06_QC_REPORTS"
         / "v2_stage1_audit_20260899T000000Z.json").write_text("{ not json")
        r = audit.latest_measured_audit(root)
        assert r["verdict"] == "RESOLVED"
        assert any(not c["usable"] and "unreadable" in c.get("why", "")
                   for c in r["considered"])

    def test_every_candidate_is_reported(self, qc):
        root = qc([
            _audit("v2_stage1_audit_20260820T223430Z.json",
                   "INCONCLUSIVE", 0, "INCONCLUSIVE", 0, "2026-08-20"),
            _audit("v2_stage1_audit_20260814T142238Z.json",
                   "FAIL", 240, "FAIL", 1479, "2026-08-14"),
        ])
        assert len(audit.latest_measured_audit(root)["considered"]) == 2


class TestGateOutcomesAreComplete:

    def test_no_gate_is_left_unverified(self):
        from sailor.experiments import gates
        outcomes = gates.protocol()["gate_outcomes"]
        assert all(o["verdict"] != "UNVERIFIED" for o in outcomes.values())

    def test_gate4_discloses_the_ordering_violation(self):
        from sailor.experiments import gates
        g4 = gates.protocol()["gate_outcomes"]["GATE-4"]
        assert g4["verdict"] == "NO_GO"
        assert "ordering_violation_disclosed" in g4

    def test_gate0_does_not_cite_the_latest_pointer(self):
        """The deciding artefact is the 14 Aug full pass, not _latest."""
        from sailor.experiments import gates
        ev = gates.protocol()["gate_outcomes"]["GATE-0"]["evidence"]
        assert any("20260814T142238Z" in e for e in ev)
        assert not any("audit_latest" in e for e in ev)
