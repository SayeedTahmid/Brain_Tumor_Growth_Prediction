# =============================================================================
# TRAINING THROUGHPUT BENCHMARK — run BEFORE any official C0 fit.
#
# No official run starts from an estimate. This measures steps/sec, VRAM and the
# data/compute split on the ACTUAL Colab GPU, then computes whether the planned
# ladder fits a 24 h per-job limit.
#
# THE LADDER IS 200 FOLD-FITS: (C0-C4 + P1-P3) x 5 folds x 5 repeats.
# Binding target: <= 45 min/fit keeps a complete rung (25 fits) inside one job.
# =============================================================================
import json, os
from sailor.stage4.patches import PairPatchSampler, CONFIG
from sailor.stage4.model import ResidualUNet3D, param_count
from sailor.stage4 import benchmark as B

print('GPU:', B.gpu_info())
print('patch config (FIXED across rungs):', json.dumps(CONFIG, indent=2))

split = json.load(open(os.path.join(ROOT, '01_DATA_FOUNDATION',
                                    'v2_pairs_and_folds.json')))
pairs = split['pairs']['pairs']
ARRAYS = os.path.join(ROOT, '01_DATA_FOUNDATION', 'v2_arrays')

model = ResidualUNet3D()
print('\nparams:', f'{param_count(model):,}')
sampler = PairPatchSampler(ARRAYS, pairs)

# ---- sweep batch size to find the VRAM ceiling -----------------------------
for bs in (2, 4, 8, 16):
    try:
        r = B.benchmark(ResidualUNet3D(), sampler, batch_size=bs,
                        steps=20, warmup=5, amp=True)
        print(f'  bs={bs:>2}  {r["sec_per_step_total"]:.3f} s/step  '
              f'{r["samples_per_sec"]:.1f} samp/s  '
              f'VRAM {r["peak_vram_gb"]} GB  data {r["data_fraction"]:.0%}')
    except RuntimeError as e:
        print(f'  bs={bs:>2}  OOM or error: {str(e)[:70]}')
        break

# ---- official benchmark at the chosen batch size ---------------------------
BATCH = 8            # set from the sweep above
STEPS_PER_FIT = 2000 # planned optimisation steps per fold-fit

res = B.run(ROOT, ResidualUNet3D(), sampler,
            steps_per_fit=STEPS_PER_FIT, batch_size=BATCH,
            steps=80, amp=True)

print("""
READ THE VERDICT
  FITS         a rung completes in <75% of the 24 h limit — restart headroom
  TIGHT        inside the limit but a disconnect costs the whole rung
  DOES_NOT_FIT split the rung across jobs, or apply the listed savings

Safe savings change compute, not results. Anything marked NEEDS APPROVAL or
CHANGES A LOCK must be agreed before it is applied.
""")
