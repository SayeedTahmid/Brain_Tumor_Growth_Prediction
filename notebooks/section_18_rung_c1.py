# =============================================================================
# RUNG C1 — mask + Δt
#
# AMD-001, EASY TO FORGET: C1 IS NOT A TREATMENT-FREE REFERENCE. Δt separates
# treatment phases almost completely (U = 0.629; median 14 d during CRT vs 91 d
# during TMZ, a 6.5x ratio). Supplying Δt already supplies most of the treatment
# signal, so C2 − C1 UNDERSTATES the treatment effect. C0 is the treatment-free
# reference.
#
# ENCODING: log(Δt), standardised to zero mean and unit variance using
# statistics fitted on each fold's TRAINING pairs ONLY. Fitting on all pairs
# would let the held-out Δt distribution shape the inputs seen during training.
# The per-fold statistics are recorded in the rung artefact.
#
# ARCHITECTURE UNCHANGED (AMD-007): same residual U-Net, same patches, same
# loss, same 2000 steps. C1 differs from C0 in ONE way — cond_dim 1 instead of
# 0 — so the rung gap measures information, not structure.
#
# BENCHMARKS TO BEAT
#   persistence   0.4928   C1 must reach 0.4373 (floor − MDE 0.0555) to beat it
#   C0-residual   0.5723   C1 must reach 0.5168 to beat C0
# =============================================================================
import json, os
from sailor.stage4 import mask_cache as MC
from sailor.stage4.model import ResidualUNet3D
from sailor.stage4 import rung as RG
from sailor.stage4.conditioning import cond_dim, describe

RUNG = 'C1'
CD = cond_dim(RUNG)
print(json.dumps(describe(RUNG), indent=2))

cache = MC.CachedMasks(ROOT)
split = json.load(open(os.path.join(ROOT, '01_DATA_FOUNDATION',
                                    'v2_pairs_and_folds.json')))

res = RG.run_rung(ROOT, cache, split,
                  model_fn=lambda: ResidualUNet3D(cond_dim=CD, residual=True),
                  rung=RUNG, steps_per_fit=2000,
                  batch_size=8, device='cuda', amp=True)

print("""
READ AGAINST TWO BENCHMARKS
  vs persistence 0.4928   does Δt let the model beat copy-forward at all?
  vs C0-res      0.5723   does Δt add information the mask alone lacks?
The MDE stays FROZEN at 0.0555. Each rung prints an MDE from its own paired SD;
that is informational only. Re-deriving it per rung would make it post-hoc.
""")
