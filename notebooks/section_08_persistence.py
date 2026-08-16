# =============================================================================
# STAGE 3 — PERSISTENCE BASELINE (rung C-1), the input to GATE-3
#
# No archive read. All 240 CL masks are already exported to
# 01_DATA_FOUNDATION/v2_arrays/. Runs in seconds.
#
# Run the git bootstrap cell first, so the artefact is stamped REPRODUCIBLE.
# =============================================================================
from sailor.stage3 import persistence

res = persistence.run(ROOT)
persistence.print_report(res)

# GATE-3 headroom. Read from volume_change_error, NOT change-region Dice:
# persistence predicts no change, so its change-region Dice is structurally
# 0.0 or undefined and carries no between-patient variance for an MDE.
mde = persistence.minimum_detectable_effect(res, metric="volume_change_error")
print('\nMINIMUM DETECTABLE EFFECT (stated before any model runs):')
for k in ('metric', 'n_patients', 'per_patient_sd', 'mde'):
    print(f'  {k:<16} {mde.get(k)}')
print('\n ', mde.get('interpretation'))
print('\n ', mde.get('approximation_note'))

print('\nartefact:', res['artefact']['latest'])
