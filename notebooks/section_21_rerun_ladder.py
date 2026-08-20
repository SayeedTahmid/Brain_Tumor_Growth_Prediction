# =============================================================================
# RE-RUN C0 AND C1 UNDER AMD-009
#
# THE LOSS CHANGED AFTER RUNG RESULTS WERE SEEN. That is disclosed, not hidden:
# AMD-009 carries a `post_result_correction` block answering what was seen, why
# it is a defect rather than a preference, that prior results are retained, and
# what the correction costs. The amendment guard REJECTS a post-rung amendment
# that does not declare all four.
#
# WHY: three rungs sat on one side of the persistence floor, monotone
# (+0.1063 / +0.0795 / +0.0664), and the residual rungs were VERIFIED to start
# exactly at the floor and train away from it. A probe adding a volume term
# collapsed the gap to +0.0052 — 94%. Δt, the variable C1 exists to test, moved
# the result by 0.0131. The objective artefact was 5.7x the conditioning effect.
#
# THE COST: the adopted term is a differentiable analogue of the scoring metric,
# so "beats persistence" is WEAKER evidence under this ladder. Both ladders are
# reported. The uncorrected runs are retained under their original names.
#
# CHECKPOINT SAFETY: the config fingerprint changed (ce442e00 -> 4906bce6), so
# any attempt to resume an old checkpoint REFUSES rather than silently mixing
# objectives. New rung names keep the two ladders separate on disk.
# =============================================================================
import json, os
from sailor.stage4 import mask_cache as MC
from sailor.stage4.model import ResidualUNet3D
from sailor.stage4 import rung as RG
from sailor.stage4.loss import CONFIG as LOSS_CONFIG
from sailor.stage4.conditioning import cond_dim
from sailor.stage4.train import _config_fingerprint

print('loss       :', LOSS_CONFIG['loss'])
print('amendment  :', LOSS_CONFIG['amendment'])
print('fingerprint:', _config_fingerprint())
print('cost       :', LOSS_CONFIG['known_cost'])

cache = MC.CachedMasks(ROOT)
split = json.load(open(os.path.join(ROOT, '01_DATA_FOUNDATION',
                                    'v2_pairs_and_folds.json')))

# C0 first: no conditioning, so it isolates the loss change against the
# retained C0res result (0.5723, gap +0.0795).
res0 = RG.run_rung(ROOT, cache, split,
                   model_fn=lambda: ResidualUNet3D(residual=True),
                   rung='C0res_v2', steps_per_fit=2000,
                   batch_size=8, device='cuda', amp=True)

# Then C1. Compare against the retained C1 (0.5592, gap +0.0664).
CD = cond_dim('C1')
res1 = RG.run_rung(ROOT, cache, split,
                   model_fn=lambda: ResidualUNet3D(cond_dim=CD, residual=True),
                   rung='C1_v2', steps_per_fit=2000,
                   batch_size=8, device='cuda', amp=True)

print("""
BOTH LADDERS, FOR THE PAPER
  uncorrected   C0-direct +0.1063 | C0-residual +0.0795 | C1 +0.0664
  corrected     printed above
The MDE stays FROZEN at 0.0555. C1_v2 - C0res_v2 is the conditioning effect
measured under an objective aligned with the metric.
""")
