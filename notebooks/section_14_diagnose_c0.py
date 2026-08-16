# =============================================================================
# C0 DIAGNOSIS — why did validation never beat persistence?
#
# THE OBSERVATION (repeat 0 / fold 0, 16 validations):
#   model log-ratio never beat persistence (best 0.4480 vs 0.4218)
#   training loss 0.72 -> 0.018 by step 500, then FLAT for 3,500 steps
#   validation OSCILLATED between 0.45 and 0.87 — it never converged
#   model Dice 0.53 vs cohort persistence ~0.47 — BETTER on the demoted metric,
#   WORSE on the pre-registered one. Exactly the divergence AMD-005 predicted.
#
# THREE EXPLANATIONS, NOT INTERCHANGEABLE:
#   A  degenerate task — BCE on ~99.9%-background patches has an easy attractor:
#      reproduce the input. Good Dice, poor volume ratio.        IMPLEMENTATION
#   B  loss/metric mismatch — BCE optimises voxels, the metric is a volume
#      ratio; nothing pushes toward correct volume.              IMPLEMENTATION
#   C  no signal at C0 — mask-only input may contain nothing beyond
#      copy-forward.                                             REAL RESULT
#
# Reporting C when the truth is A or B is a false negative dressed as a finding.
# This cell MEASURES which it is. It changes no loss, architecture or lock.
# =============================================================================
import json, os, torch
from sailor.stage4 import mask_cache as MC
from sailor.stage4.model import ResidualUNet3D
from sailor.stage4.train import load_checkpoint
from sailor.stage4 import convergence as CV
from sailor.stage4 import diagnose_c0 as DG

cache = MC.CachedMasks(ROOT)
split = json.load(open(os.path.join(ROOT, '01_DATA_FOUNDATION',
                                    'v2_pairs_and_folds.json')))
_, test_pairs = CV.fold_pairs(split)

# Reload the trained probe model rather than retraining it.
model = ResidualUNet3D()
opt = torch.optim.AdamW(model.parameters())
scaler = torch.amp.GradScaler('cuda', enabled=False)
ck = os.path.join(ROOT, '11_CHECKPOINTS', 'probe_r0f0.pt')
step, meta = load_checkpoint(ck, model, opt, scaler)
print(f'loaded checkpoint at step {step}')

d = DG.diagnose(model.cuda(), cache, test_pairs, patch=96, batch_size=8,
                device='cuda', amp=True)
spread = DG.per_patient_spread(d)
DG.print_report(d, spread)

from sailor.utils.persist import save_artefact
art = save_artefact(ROOT, '10_EXPERIMENTS', 'c0_diagnosis',
                    {k: v for k, v in d.items() if k != 'per_pair'} |
                    {'per_patient_spread': spread, 'checkpoint_step': step})
print('artefact:', art['latest'])
