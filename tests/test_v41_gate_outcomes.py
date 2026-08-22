"""v0.41 — the pre-registration document must not assert a falsehood.

Before this, `protocol()['status']` read "No C-rung has been run" while C0, C1,
C2 and P1 were complete and GATE-3 carried a recorded GO. The document a
reviewer is pointed at contradicted the project state.

The fix keeps criteria immutable and adds a separate outcome register. These
tests pin both halves: criteria cannot drift, and the status line cannot
contradict the outcomes.
"""

import pytest

from sailor.experiments import gates


def _doc():
    return gates.protocol()


class TestCriteriaAreNotEdited:
    """The pre-registered half. Editing these retroactively is the one thing
    this document exists to prevent."""

    def test_pre_registration_status_inside_gates_is_untouched(self):
        for g in _doc()["gates"]:
            assert g["status"].startswith("OPEN"), (
                f"{g['id']} pre-registration status was edited. Resolutions "
                "belong in GATE_OUTCOMES, not here.")

    def test_criteria_frozen_date_is_recorded(self):
        assert _doc()["criteria_frozen_utc"] == "2026-08-12T20:29:49+00:00"

    def test_every_gate_has_criteria_and_a_decision_rule(self):
        for g in _doc()["gates"]:
            assert g.get("criteria"), f"{g['id']} lost its criteria"
            assert g.get("decision"), f"{g['id']} lost its decision rule"


class TestOutcomeRegister:

    def test_every_gate_has_an_outcome_entry(self):
        d = _doc()
        assert set(d["gate_outcomes"]) == set(d["gate_order"])

    def test_outcome_ids_are_real_gates(self):
        ids = {g["id"] for g in _doc()["gates"]}
        assert set(_doc()["gate_outcomes"]) <= ids

    def test_a_resolved_gate_names_its_evidence(self):
        for gid, o in _doc()["gate_outcomes"].items():
            if o["verdict"] == "UNVERIFIED":
                assert o["evidence"] is None
            else:
                assert o["evidence"], (
                    f"{gid} claims {o['verdict']} with no artefact. A verdict "
                    "without evidence is an assertion.")

    def test_unverified_is_used_rather_than_a_guess(self):
        """Section 2.2: unknowns are marked, never filled in."""
        assert _doc()["gate_outcomes"]["GATE-0"]["verdict"] == "UNVERIFIED"
        assert _doc()["gate_outcomes"]["GATE-4"]["verdict"] == "UNVERIFIED"


class TestStatusCannotContradictOutcomes:

    def test_status_does_not_deny_completed_rungs(self):
        d = _doc()
        resolved = [g for g, o in d["gate_outcomes"].items()
                    if o["verdict"] != "UNVERIFIED"]
        if resolved:
            assert "No C-rung has been run" not in d["status"], (
                "status denies work that gate_outcomes records as done")

    def test_status_still_asserts_pre_registration(self):
        assert "PRE-REGISTERED" in _doc()["status"]

    def test_status_points_at_the_outcome_register(self):
        assert "gate_outcomes" in _doc()["status"]


class TestAmendments:

    def test_all_nine_are_present(self):
        ids = [a["id"] for a in _doc()["amendments"]]
        assert ids == [f"AMD-{n:03d}" for n in range(1, 10)]

    def test_each_amendment_denies_following_a_model_result(self):
        for a in _doc()["amendments"]:
            assert a.get("results_seen_before_amendment"), (
                f"{a['id']} does not state what had been seen when it was made")

    @pytest.mark.parametrize("field", ["change", "prompted_by", "nature"])
    def test_amendment_fields_are_populated(self, field):
        for a in _doc()["amendments"]:
            assert a.get(field), f"{a['id']} missing {field}"
