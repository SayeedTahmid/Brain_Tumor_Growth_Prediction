# =============================================================================
# ID CORRECTION — the code's identifiers had drifted from the ROS
#
# TWO ERRORS, BOTH FOUND BY READING §11.1 AND §11.2 RATHER THAN RECALLING THEM:
#
#   1. ARCHITECTURAL ABLATIONS. gates.py used a SUBTRACTIVE scheme ("removes X")
#      with its own numbering; ROS §11.2 is ADDITIVE (A0 = winning rung, then
#      + component). Under the code list the residual comparison was A5; under
#      the ROS it is A3. Everything reported so far called it A3, which the ROS
#      makes CORRECT — the code list was wrong, not the report.
#
#   2. PERMUTATION CONTROLS. ROS §11.1: P1 is the TREATMENT shuffle, P2 the DOSE
#      shuffle. CONTROL_SHAPES had these INVERTED, written from memory of rung
#      ordering rather than read from the constitution. The completed treatment
#      shuffle was therefore named P2_v2 when it is the ROS's P1.
#
# The experiment is unaffected — treatment labels shuffled between patients at
# matched session ordinal is exactly the ROS's P1. Only the label was wrong, and
# it is corrected by a RECORD rather than by editing the artefact, so the Drive
# history stays honest.
# =============================================================================
import json, os
from sailor.utils.persist import save_artefact
from sailor.experiments.gates import (ARCHITECTURAL_ABLATIONS,
                                      PERMUTATION_CONTROLS,
                                      ABLATION_ID_CORRECTION)

L = lambda n: json.load(open(os.path.join(ROOT, '10_EXPERIMENTS', f'v2_rung_{n}.json')))
c2, p_shuffle = L('C2_v2'), L('P2_v2')
g = lambda r: r['gap_vs_persistence']

rec = {
  "manifest": "id_correction",
  "authority": "ROS §11.1 and §11.2 bind; the code had drifted",
  "renames": [{
      "artefact_on_disk": "v2_rung_P2_v2.json",
      "recorded_as": "P2_v2",
      "correct_ros_id": "P1",
      "why": ("ROS §11.1 defines P1 as the treatment shuffle and P2 as the dose "
              "shuffle. CONTROL_SHAPES had them inverted. The run IS a treatment "
              "shuffle (labels permuted between patients at matched session "
              "ordinal), so it is the ROS's P1."),
      "artefact_NOT_edited": ("The file keeps its original name. This record is "
                              "the mapping, so the history is not rewritten."),
  }],
  "ablation_ids": ABLATION_ID_CORRECTION,
  "ros_p1_prespecified_interpretation": {
      "quote": ("P1 — treatment shuffle. C2 must degrade relative to its "
                "unpermuted self. If it does not, the treatment branch is "
                "reading position, not treatment."),
      "C2_gap": g(c2),
      "P1_shuffled_gap": g(p_shuffle),
      "C2_degraded_under_shuffle": bool(g(p_shuffle) > g(c2)),
      "verdict": ("C2 did NOT degrade — the shuffled run is marginally BETTER "
                  f"({g(p_shuffle):+.4f} vs {g(c2):+.4f}). By the ROS's own "
                  "pre-specified interpretation, the treatment branch is "
                  "reading POSITION, not treatment. This is a stronger and "
                  "pre-registered conclusion than 'no signal'."),
      "caveat": ("Only 34% of labels changed under the shuffle, because "
                 "treatment is ~92% predictable from session index. The control "
                 "is weak by construction and bounds rather than eliminates a "
                 "treatment effect."),
  },
  "p2_dose_shuffle": {
      "status": "UNRUNNABLE",
      "why": "requires C3, blocked by GATE-1 (all 26 dose maps in a different space)",
  },
  "p3_time_only": {"status": "NOT RUN",
                   "note": ("ROS: 'C2 must beat P3. If C2 ≈ P3, treatment status "
                            "is a re-encoding of the protocol schedule.' C1 is "
                            "time-conditioned and gave +0.0077 vs C2's +0.0031, "
                            "so C2 ≈ C1 ≈ P3 in effect — but P3 as specified was "
                            "not run and this is an inference, not a result.")},
}
art = save_artefact(ROOT, '10_EXPERIMENTS', 'id_correction', rec)
print('P2_v2 on disk  ->  ROS id P1')
print('A3             ->  + residual formulation (ROS §11.2), as reported')
print('\nROS P1 interpretation:')
print(' ', rec['ros_p1_prespecified_interpretation']['verdict'])
print('artefact:', art['latest'])
