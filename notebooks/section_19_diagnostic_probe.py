# =============================================================================
# DIAGNOSTIC — is the ladder measuring an optimisation artefact?
#
# THREE RUNGS, ONE SIDE OF THE FLOOR, MONOTONE:
#     persistence   0.4928
#     C1 (+Δt)      0.5592   gap +0.0664
#     C0-residual   0.5723   gap +0.0795
#     C0-direct     0.5991   gap +0.1063
#
# The residual rungs were VERIFIED to start exactly at 0.4928 and trained AWAY
# from it. Δt moved the endpoint by 0.0131 — under a quarter of the frozen MDE.
# Two rungs departing a guaranteed floor under different information points at
# the OBJECTIVE, not the information.
#
# HYPOTHESIS: BCE + soft Dice rewards spatial overlap; the pre-registered metric
# is a volume ratio. Nothing in training penalises getting volume wrong, so a
# model can cut its loss while drifting on the metric that decides the rung.
# Every rung has shown exactly that signature — model Dice ABOVE persistence,
# model log-ratio BELOW it.
#
# THIS IS A PROBE, NOT A CHANGE. sailor.stage4.loss is untouched. One repeat,
# 5 fits, ~1 h. Adopting this loss would invalidate C0 and C1 under AMD-007 and
# require re-running both (~10 h) — hence measuring first.
# =============================================================================
import json, os
from sailor.stage4 import mask_cache as MC
from sailor.stage4.model import ResidualUNet3D
from sailor.stage4 import diagnostic_loss as DL
from sailor.stage4.conditioning import FoldStandardiser, make_cond_fn, cond_dim

cache = MC.CachedMasks(ROOT)
split = json.load(open(os.path.join(ROOT, '01_DATA_FOUNDATION',
                                    'v2_pairs_and_folds.json')))

# Run the diagnostic on C0-residual's configuration: no conditioning, so the
# ONLY difference from the completed C0res rung is the loss.
res = DL.run_probe(ROOT, cache, split,
                   model_fn=lambda: ResidualUNet3D(residual=True),
                   rung_name='DIAGvol', steps_per_fit=2000,
                   batch_size=8, device='cuda', amp=True, repeat=0)
DL.print_report(res)

from sailor.utils.persist import save_artefact
art = save_artefact(ROOT, '10_EXPERIMENTS', 'diagnostic_volume_loss', res)
print('artefact:', art['latest'])

print("""
NOTE ON COMPARABILITY. C0res's headline 0.5723 pools all 5 repeats; this probe
runs repeat 0 only. Compare the GAP vs persistence computed on the same pairs,
which both report, rather than the raw log-ratio against the 5-repeat figure.
""")
