# =============================================================================
# DIAGNOSTIC PROBE B — a volume term that is NOT the evaluation metric
#
# PROBE A RESULT: adding a log-volume-ratio term collapsed the persistence gap
# from +0.0795 to +0.0052 — a 94% reduction. The ladder's rung deficits were
# largely an OPTIMISATION ARTEFACT: BCE + soft Dice rewards overlap while the
# pre-registered metric measures volume, so the model drifted on the second
# while improving the first.
#
# For scale: the loss change bought 0.0743. Δt — the conditioning variable C1
# exists to test — bought 0.0131. The artefact was 5.7x the effect being
# measured.
#
# WHY NOT SIMPLY ADOPT PROBE A. Its term is a differentiable analogue of
# `log_volume_ratio_error`, the metric the rung is SCORED by. Adopting it makes
# "beats persistence" weaker evidence and a reviewer will say so.
#
# VARIANT B corrects the same drift with a term sharing no functional form with
# the metric:
#     metric   |log((V_true+1)/(V_pred+1))|     logarithmic, SYMMETRIC
#     variant  |V_pred - V_true| / (V_true+eps) linear, ASYMMETRIC
# Verified: halving vs doubling the volume costs 0.68/0.68 under the metric and
# 0.33/0.67 under variant B. It is not the metric rescaled.
#
# One repeat, 5 fits, ~1 h. The frozen loss remains untouched.
# =============================================================================
import json, os
from sailor.stage4 import mask_cache as MC
from sailor.stage4.model import ResidualUNet3D
from sailor.stage4 import diagnostic_loss as DL
from sailor.utils.persist import save_artefact

cache = MC.CachedMasks(ROOT)
split = json.load(open(os.path.join(ROOT, '01_DATA_FOUNDATION',
                                    'v2_pairs_and_folds.json')))

res = DL.run_probe(ROOT, cache, split,
                   model_fn=lambda: ResidualUNet3D(residual=True),
                   rung_name='DIAGvolB', steps_per_fit=2000,
                   batch_size=8, device='cuda', amp=True, repeat=0,
                   loss_fn=DL.make_relative_volume_loss(),
                   loss_config=DL.CONFIG_B)
DL.print_report(res)
print('artefact:', save_artefact(ROOT, '10_EXPERIMENTS',
                                 'diagnostic_volume_loss_B', res)['latest'])

print("""
DECISION THIS INFORMS
  gap similar to probe A (~+0.005)  adopt variant B. Same correction, and
                                    "beats persistence" stays fully meaningful.
                                    Amend AMD-007, re-run C0 and C1 (~10 h).
  gap much larger (~+0.05-0.08)     the correction depended on mirroring the
                                    metric. Then the choice is variant A with
                                    the asterisk, or the frozen loss with the
                                    artefact documented.
""")
