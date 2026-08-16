# =============================================================================
# MULTI-FOLD CONVERGENCE — with the corrected loss
#
# TWO CHANGES, BOTH FORCED BY MEASUREMENT, BOTH BEFORE ANY OFFICIAL RUNG:
#
# 1. LOSS: BCEWithLogits -> BCEWithLogits + soft Dice.
#    The C0 probe under-predicted volume by ~15% (median pred/target 0.846) and
#    scored WORSE than copy-forward on Dice-vs-target (0.4923 vs 0.5041).
#    Measured mechanism: under BCE the background gradient mass is 63x the
#    foreground's — exactly the class ratio — so the cheapest descent direction
#    is to shrink. Soft Dice cuts that imbalance to 22x.
#    (An earlier explanation, that BCE penalises under-prediction more leniently,
#     was TESTED AND REFUTED: it scores equal under/over errors identically.)
#
# 2. VALIDATION: one fold (5 patients) -> all folds pooled (26 patients).
#    Single-fold between-patient SD was 0.276, with per-patient means from 0.20
#    to 0.76. The step-to-step swings (0.45-0.87) were the SAME order — so the
#    "oscillation" was largely re-drawn sampling noise, not training
#    instability. A validation estimate noisier than the effect cannot decide a
#    step budget.
#
# Cost: 5x the single-fold probe. That is the honest price of a usable curve.
# Nothing locked changes: architecture, patches, split, CV and the pre-registered
# metric are untouched.
# =============================================================================
import json, os
from sailor.stage4 import mask_cache as MC
from sailor.stage4.model import ResidualUNet3D
from sailor.stage4 import convergence as CV
from sailor.stage4.loss import CONFIG as LOSS_CONFIG

print('LOSS NOW FIXED AS:', json.dumps(LOSS_CONFIG, indent=2))

cache = MC.CachedMasks(ROOT)
split = json.load(open(os.path.join(ROOT, '01_DATA_FOUNDATION',
                                    'v2_pairs_and_folds.json')))

res = CV.run_multifold(ROOT, cache, split, model_fn=lambda: ResidualUNet3D(),
                       repeat=0, max_steps=4000, validate_every=500,
                       batch_size=8, device='cuda', amp=True)

print("""
WHAT TO READ
  between-pt SD   now over 26 patients, not 5. If it is still ~0.28 the metric
                  is intrinsically variable across patients and the ladder's
                  own CIs will be wide — that is a finding, not a bug.
  beats?          model vs persistence on the SAME pooled patients.
  plateau         a budget is recommended ONLY if converged=True. If the curve
                  still oscillates at 26 patients, the instability is real and
                  must be diagnosed rather than averaged away.
""")
