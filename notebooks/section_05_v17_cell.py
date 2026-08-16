# =============================================================================
# SAILOR — v18 / sailor 0.17.0-stage2
# ONE widened pass. bz2 is sequential: the ~90 min decompression is paid the
# moment the archive is opened, so everything that needs voxels is read now.
#
# NOTHING BELOW CHANGES A LOCK. No cohort change, no re-freeze, no target or
# intensity-variant decision. It measures; you decide afterwards.
# =============================================================================

# --- CELL 1 — bootstrap ------------------------------------------------------
VERSION = 18
import sys, os, shutil, zipfile
ZIP  = f'/content/drive/MyDrive/sailor_stage1_v{VERSION}.zip'
CODE = '/content/sailor_stage1'
assert os.path.isfile(ZIP), f'not found: {ZIP}'
shutil.rmtree(CODE, ignore_errors=True)
with zipfile.ZipFile(ZIP) as z:
    z.extractall('/content')
for m in [k for k in sys.modules if k == 'sailor' or k.startswith('sailor.')]:
    del sys.modules[m]
sys.path = [p for p in sys.path if p != CODE]
sys.path.insert(0, CODE)
os.environ['SAILOR_PROJECT_NAME'] = 'SAILOR_Longitudinal_Research'
os.environ['SAILOR_LEGACY_ROOT']  = '/content/drive/MyDrive/sailor_v1'
os.environ['SAILOR_CODE_ROOT']    = CODE

import sailor
from sailor.config import get_paths
paths = get_paths(); ROOT = paths.dataset_root
print('sailor', sailor.__version__)
assert sailor.__version__ == '0.17.0-stage2', f'stale package: {sailor.__version__}'
sailor.require('dose_arrays_read', 'plhm_icor_check', 'stratified_sampling',
               'volume_percentiles', 'rano_refuted', 'visual_check')
print('capabilities OK')


# --- CELL 2 — the widened pass (~90-120 min; do not let the tab idle) --------
# force_rescan=True is REQUIRED: the v0.16 cache has no percentiles, no dose
# arrays and a single-subject image sample. The cache key also changed, so a
# stale cache cannot be silently reused.
from sailor.stage1.audit import run_stage1_audit
from sailor.stage1.visual_check import sub25_targets

report = run_stage1_audit(
    paths,
    verify_hashes=False,          # hashes already verified in an earlier pass
    force_rescan=True,
    archives_to_scan=['derivatives.tar.bz2'],
    read_volumes=True,
    sample_per_subject=6,         # stratified: replaces first-60-in-archive-order
    read_icor=True,               # locked variant, all 270 sessions
    plhm_subjects=8,              # paired plain+icor for the PLHM check
    slice_targets=sub25_targets(),
    verbose=True)

print('\nguards:', report['guards']['summary'])


# --- CELL 3 — the three checks, all off the cache, seconds each --------------
from sailor.stage1 import adjudicate, plhm_check

print('\n' + '=' * 78 + '\n  PLHM / -icor  (BLOCKING before Phase 5)\n' + '=' * 78)
plhm = plhm_check.run(ROOT)

print('\n' + '=' * 78 + '\n  DOSE — GATE-1 input\n' + '=' * 78)
d = report['dose']
print(f"  files={d['n_dose_files']}  patients={d['n_patients_with_dose']}  "
      f"registration={d['registration_status']}")
for vr in d.get('value_ranges', [])[:5]:
    print(f"    {vr['path'].rsplit('/', 1)[-1]}: "
          f"min={vr['min']} max={vr['max']} nonzero={vr['n_nonzero']}")

print('\n' + '=' * 78 + '\n  ADJUDICATION — RANO no longer scored\n' + '=' * 78)
adj = adjudicate.run(ROOT)
print('\n  clustering:', adj['clustering_by_subject'])

print('\n' + '=' * 78 + '\n  G10 — intensity, now cohort-wide\n' + '=' * 78)
import json
from sailor.utils.persist import cache_dir, load_cache
key = sorted(p.stem for p in cache_dir(ROOT).glob('audit_scan_full_*_v17_*.json')
             if not p.stem.endswith('__slices'))[-1]
vs = load_cache(ROOT, key)['raw']['volume_stats']
nf = adjudicate.nonfinite_report(vs, top=40)
print(json.dumps({k: v for k, v in nf.items()
                  if k not in ('volumes_with_nonfinite', 'outside_0_255')}, indent=2))


# --- CELL 4 — sub-25 slices. LOOK at these; do not infer from them. ----------
from sailor.stage1.visual_check import write_png
sl = load_cache(ROOT, key + '__slices')
if not sl:
    print('no slices captured — was slice_targets passed to the pass?')
else:
    pngs = write_png(ROOT, sl['raw'])
    print(f'{len(pngs)} PNG(s) written under 06_QC_REPORTS/v2_sub25_slices/')
    for p in pngs:
        print('  ', p)
    print('\nThe question: at ses-05..ses-10, does T1c show enhancing tissue?')
    print('  no enhancement visible  -> the all-zero CL mask is CORRECT, real data')
    print('  enhancement clearly visible -> segmentation failure, excluded per 3.2')
    print('Nothing is decided by this script.')
