# =============================================================================
# OFFICIAL RUNG C0 — 25 fits (5 folds x 5 repeats) of the frozen split
#
# This is not a probe. It produces a number that enters the paper.
#
# STEP BUDGET must be frozen before this runs. Evidence from the multi-fold
# probe: training loss flattens by step 1000 on every fold, and validation
# drifts slightly WORSE from there (mild overfitting). 2000 gives 2x margin
# over the flattening point at ~11 min/fit.
#
# RESUMABLE: each fit checkpoints independently and completed fits are skipped,
# so a dropped session costs one fit, not the rung.
#
# NEW GUARD: checkpoints carry a config fingerprint. If the loss or patch
# settings change, resume REFUSES rather than silently mixing objectives —
# the failure that hit fold 0 when the loss changed.
# =============================================================================
import json, os
from sailor.stage4 import mask_cache as MC
from sailor.stage4.model import ResidualUNet3D
from sailor.stage4 import rung as RG
from sailor.stage4.train import _config_fingerprint

STEPS_PER_FIT = 2000        # FROZEN — applies identically to every rung

cache = MC.CachedMasks(ROOT)
split = json.load(open(os.path.join(ROOT, '01_DATA_FOUNDATION',
                                    'v2_pairs_and_folds.json')))
print('config fingerprint:', _config_fingerprint())
print('steps/fit:', STEPS_PER_FIT, '| expected ~4.6 h for 25 fits')

res = RG.run_rung(ROOT, cache, split, model_fn=lambda: ResidualUNet3D(),
                  rung='C0', steps_per_fit=STEPS_PER_FIT,
                  batch_size=8, device='cuda', amp=True)

print("""
READ IN THIS ORDER
  beats persistence   the C0 result. Paired comparison on identical held-out
                      pairs, patient-level CIs over 26 units.
  paired difference   the SD GATE-3's MDE was an UPPER BOUND for. Recompute the
                      MDE from it ONCE, record it, do not re-derive per rung.
  CI overlap          if model and persistence CIs overlap heavily, the rung is
                      indistinguishable from the floor — which at this n is a
                      finding, not a failure.
""")
