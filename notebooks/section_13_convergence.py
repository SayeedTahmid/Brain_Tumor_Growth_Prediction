# =============================================================================
# CONVERGENCE PROBE + PIPELINE VERIFICATION
#
# STEPS_PER_FIT = 2000 was a PLACEHOLDER. The entire 24 h budget rests on it and
# nothing has established it is enough — or more than enough. This probe:
#   1. proves checkpoint/resume is BIT-EXACT (not merely "works")
#   2. trains ONE pre-specified fold to 4000 steps
#   3. validates every 250 steps at VOLUME level on held-out patients
#   4. applies a plateau rule FIXED BEFORE the curve is seen
#
# Nothing locked changes: architecture, 96^3 patches, sampling config, the
# frozen 5x5 split and the pre-registered metric are all unchanged.
#
# Do NOT start the 200 official fits until this reports a plateau AND
# bit-exact resume.
# =============================================================================
import json, os
from sailor.stage4 import mask_cache as MC
from sailor.stage4.model import ResidualUNet3D
from sailor.stage4 import convergence as CV

cache = MC.CachedMasks(ROOT)
split = json.load(open(os.path.join(ROOT, '01_DATA_FOUNDATION',
                                    'v2_pairs_and_folds.json')))
print(f'cache {len(cache)} volumes, {cache.ram_mb():.0f} MB | '
      f'split {split["content_sha256"][:16]}…')

# ~4000 steps at the measured 0.332 s/step (bs=8) is ~22 min of training,
# plus periodic whole-volume validation on the held-out patients.
res = CV.run(ROOT, cache, split, model_fn=lambda: ResidualUNet3D(),
             max_steps=4000, validate_every=250, batch_size=8,
             device='cuda', amp=True, verify_resume_first=True)

print("""
WHAT TO CHECK BEFORE FREEZING THE BUDGET
  resume        must read BIT-EXACT. Anything else and a disconnected run
                silently diverges from the run it claims to continue.
  plateau_step  the step budget. If it is None the metric was still improving
                at 4000 and the probe must be EXTENDED, not rounded down.
  beats?        whether the model beats persistence on the held-out patients of
                this ONE fold. Informative, not a result — one fold is not the
                cohort and this is not GATE-3.
""")

p = res.get('plateau', {})
if p.get('plateau_step'):
    from sailor.stage4 import benchmark as B
    bench = {"sec_per_step_total": 0.332}     # measured, bs=8, L4
    print('Budget at the measured plateau:')
    B.print_report({"benchmark": dict(bench, gpu=B.gpu_info(), batch_size=8,
                                      amp=True, channels_last=False,
                                      sec_data=0.007, sec_compute=0.325,
                                      steps_per_sec=1/0.332, peak_vram_gb=4.4,
                                      data_fraction=0.02, io_bound_warning=None,
                                      device='cuda', steps_timed=80,
                                      samples_per_sec=24.1),
                    "budget": B.budget(bench, p['plateau_step'])})
