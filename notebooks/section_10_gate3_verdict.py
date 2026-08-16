# =============================================================================
# GATE-3 — assessment and verdict.
#
# The gate's literal criterion (compare the MDE against the largest gain in
# comparable literature) is UNSATISFIABLE: TaDiff, the most comparable work,
# reports no persistence baseline. AMD-008 replaces it with a self-contained
# criterion — the fraction of total available headroom a model must capture.
#
# Cell A assesses and issues NO verdict. Cell B records yours, append-only.
# =============================================================================

# --- CELL A — assess ---------------------------------------------------------
import json, os
from sailor.stage3 import persistence
from sailor.experiments import gate3

res = persistence.run(ROOT)
prereg = json.load(open(os.path.join(
    ROOT, '10_EXPERIMENTS', 'v2_gate3_primary_metric.json')))

a = gate3.assess(res, prereg)
gate3.print_assessment(a)


# --- CELL B — record the verdict. Edit both, then run ONCE. -----------------
VERDICT = "GO"          # or "NO_GO"

JUSTIFICATION = (
    "EDIT THIS. State the reasoning that a reviewer without this conversation "
    "would need: what fraction of total headroom must be captured, why that is "
    "or is not achievable for this cohort, and what the consequence is. Note "
    "that the MDE is an upper bound and the true requirement is lower by an "
    "unknown amount, and that no external benchmark exists because the "
    "comparable literature omits the persistence comparison."
)

rec = gate3.record_verdict(ROOT, VERDICT, JUSTIFICATION, a)
print("GATE-3:", rec["verdict"])
print(rec["consequence"])
print("artefact:", rec["artefact"]["latest"])
