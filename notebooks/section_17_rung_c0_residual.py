# =============================================================================
# RUNG C0-RESIDUAL — re-run of C0 with the ROS §8 residual head
#
# WHY THE RE-RUN. The direct-prediction C0 scored 0.5991 against persistence
# 0.4928 — worse by +0.1063, about 2x the measured MDE of 0.0555. The identity
# pipeline control returned delta EXACTLY 0.000000 over all 208 pairs, so the
# scoring path was not at fault: the deficit belonged to the model.
#
# ROS §8 specifies `Residual Head | x_t, Z_cond, Δt | Δ̂` — the head outputs a
# CHANGE. Our model predicted the target directly, which is a specification gap.
# With a zero-initialised residual head the UNTRAINED model is exactly
# persistence, so persistence becomes the FLOOR rather than something the model
# must rediscover from scratch.
#
# NOTHING IS DISCARDED. The completed direct-prediction C0 is retained as
# ablation A3's comparison arm — ROS §7 stage 8: "beats direct prediction, or is
# dropped". The ablation now has its data instead of an assumption.
#
# AMD-007 requires ONE architecture across rungs, so C0 is redone rather than
# the ladder being run on two different models.
# =============================================================================
import json, os
from sailor.stage4 import mask_cache as MC
from sailor.stage4.model import ResidualUNet3D, CONFIG as MODEL_CONFIG
from sailor.stage4 import rung as RG, identity_control as IC
from sailor.stage4.train import _config_fingerprint

print('architecture:', json.dumps(MODEL_CONFIG, indent=2)[:600])
print('\nconfig fingerprint:', _config_fingerprint())

cache = MC.CachedMasks(ROOT)
split = json.load(open(os.path.join(ROOT, '01_DATA_FOUNDATION',
                                    'v2_pairs_and_folds.json')))

# --- PRE-FLIGHT: an untrained residual model must score EXACTLY persistence ---
# This is the residual guarantee, checked on real data before 4.75 h is spent.
pre = IC.run(cache, split['pairs']['pairs'][:40], patch=96, batch_size=8,
             device='cuda', amp=True)
print(f"\nidentity control (scoring path): delta {pre['delta']:+.6f}")

from sailor.stage4.inference import evaluate_fold
untrained = evaluate_fold(ResidualUNet3D(residual=True).cuda(), cache,
                          split['pairs']['pairs'][:40], batch_size=8,
                          device='cuda', amp=True)
gap = (untrained['model']['log_ratio']['mean']
       - untrained['persistence']['log_ratio']['mean'])
print(f"untrained residual model      : {untrained['model']['log_ratio']['mean']:.6f}")
print(f"persistence on the same pairs : {untrained['persistence']['log_ratio']['mean']:.6f}")
print(f"gap                           : {gap:+.6f}")
assert abs(gap) < 1e-6, (
    f"RESIDUAL GUARANTEE VIOLATED: untrained model differs from persistence by "
    f"{gap:+.6f}. Do not spend 4.75 h until this is zero.")
print("residual guarantee holds — the model starts AT the persistence floor")

# --- the rung ---------------------------------------------------------------
res = RG.run_rung(ROOT, cache, split,
                  model_fn=lambda: ResidualUNet3D(residual=True),
                  rung='C0res', steps_per_fit=2000,
                  batch_size=8, device='cuda', amp=True)

print("""
READ IN THIS ORDER
  beats persistence   with the floor guaranteed at init, a rung that still
                      fails to beat persistence has learned something actively
                      unhelpful — a much stronger statement than before.
  gap                 compare against direct-prediction C0's +0.1063. The
                      difference IS ablation A3.
  paired difference   recompute nothing. The MDE is frozen at 0.0555.
""")
