# =============================================================================
# LADDER ANALYSIS — no training, no archive reads. Minutes.
#
# The pooled result is a null: spread 0.0071 across four rungs against a frozen
# MDE of 0.0555. A null invites two fair questions, and both are answerable from
# artefacts already on disk.
#
#   1. IS THE NULL UNIFORM, OR AN AVERAGE OVER OPPOSING EFFECTS?
#      A pooled mean of zero is consistent with "nothing anywhere" and with
#      "helps some patients, hurts others". Those are different findings.
#
#   2. IS IT UNIFORM ACROSS Δt?
#      AMD-002 stratified PERSISTENCE by frozen bands because copy-forward is
#      near-ceiling at short intervals. The RUNGS were never stratified. A
#      pooled mean dominated by the 83 short-interval pairs could hide an effect
#      that exists only where persistence is weak.
#
# NEITHER IS A SUBGROUP SEARCH. The bands are frozen (AMD-002) and the patient
# is the unit (AMD-003). Anything found here is a HYPOTHESIS for a future
# cohort — the pooled MDE does not transfer to a subgroup.
# =============================================================================
from sailor.stage4 import ladder_analysis as LA
from sailor.utils.persist import save_artefact

pp = LA.per_patient(ROOT)
LA.print_per_patient(pp)

bands = LA.by_delta_band(ROOT)
LA.print_bands(bands)

art = save_artefact(ROOT, '10_EXPERIMENTS', 'ladder_analysis',
                    {"per_patient": pp, "by_delta_band": bands})
print('artefact:', art['latest'])
