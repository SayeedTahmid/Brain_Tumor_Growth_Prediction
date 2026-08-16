# =============================================================================
# SAILOR — v19 / sailor 0.19.0-stage2
#
# ONE targeted pass. Two purposes, both needing voxels from the same archive:
#   1. sub-25 T1c slices at the ACTUAL tumour anatomy (z=114-132), so the
#      adjudication can be decided by looking rather than inferring.
#   2. Dose maps + baseline CL masks exported to .npz, after which GATE-1 is
#      pure CPU work off disk and needs no further archive read.
#
# NOTHING BELOW CHANGES A LOCK. Cohort 26 in the frozen manifest, split
# 40e674ee52... untouched, sub-25 deferred per v2_phase5_analysis_cohort.json.
# =============================================================================

# --- CELL 1 — bootstrap ------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

VERSION = 19
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
# Capability check, NOT a version-equality assert: an equality assert fails on
# the next upgrade as loudly as on a stale copy, and says "stale" either way.
sailor.require('slice_reference_z', 'mask_overlay', 'cache_by_mtime',
               'cache_field_require', 'array_export', 'perfusion_not_intensity')
print('capabilities OK |', ROOT)


# --- CELL 2 — targeted pass (~90-120 min, cannot resume) --------------------
# read_icor=False and sample_per_subject=0: the intensity statistics are already
# measured and cached from v18. This pass reads only what is still missing.
from sailor.stage1.audit import run_stage1_audit
from sailor.stage1.visual_check import sub25_targets, SUB25_REFERENCE_Z

report = run_stage1_audit(
    paths,
    verify_hashes=False,
    force_rescan=True,
    archives_to_scan=['derivatives.tar.bz2'],
    read_volumes=True,
    sample_per_subject=0,          # intensity stats already cached from v18
    read_icor=False,
    plhm_subjects=0,               # PLHM already answered: PER_VOLUME, 1.564
    slice_targets=sub25_targets(),
    slice_reference_z=SUB25_REFERENCE_Z,   # (114, 132) — measured, not guessed
    export_arrays=True,
    verbose=True)

print('\nguards:', report['guards']['summary'])
# Watch for: "[slices] captured N volume(s)" and "[export] wrote N array(s)".
# A "[slices] WARNING" line means the capture produced nothing — read it.


# --- CELL 3 — sub-25 overlays. LOOK at these. -------------------------------
from sailor.utils.persist import latest_full_pass, load_cache
from sailor.stage1.visual_check import overlay_png, write_png

KEY = latest_full_pass(ROOT, require=('volume_stats',))
sl = load_cache(ROOT, KEY + '__slices')
print('slice volumes:', len(sl['raw']) if sl else 0)
print('capture errors:', sl['meta'].get('errors') if sl else 'n/a')

# Sanity: every image must be sliced through z ~ 114-132, not the whole head.
for k, r in sorted(sl['raw'].items()):
    if 'T1c.nii' in k:
        print(f"  {k:<44} idx={r['slice_indices']} basis={r['slice_basis']}")

pngs = overlay_png(ROOT, sl['raw'], subject='sub-25',
                   image='T1c.nii.gz', mask='ContrastEnhancedMask-CL.nii.gz')
print(f'\n{len(pngs)} overlay PNG(s) written:')
for p in pngs: print('  ', p)

print("""
THE QUESTION, for ses-03 and ses-04 specifically:
  Is there focal enhancing tissue in the resection bed that the CL mask missed,
  or is the enhancement diffuse post-treatment change that a human reader
  correctly declined to label as tumour?
    no focal enhancement  -> CL mask is CORRECT; all 7 are true complete response
    clear focal tumour    -> segmentation failure; exclude per 3.2
ses-01 and ses-02 (546 and 82 voxels) show what this patient's enhancement
looked like when it WAS there. Nothing in this code decides the answer.
""")


# --- CELL 4 — GATE-1 inputs now on disk -------------------------------------
import numpy as np
from sailor.stage1.array_export import summarise

ex = load_cache(ROOT, KEY + '__exported')
print('arrays exported:', len(ex['raw']))
print('dose summary:', ex['meta'])

d = sorted((ROOT / '01_DATA_FOUNDATION' / 'v2_arrays').glob('*DoseMap.npz'))
print(f'\ndose .npz on disk: {len(d)}')
for p in d[:5]:
    z = np.load(p)
    print(f"  {p.name:<34} shape={tuple(z['shape'])} spacing={tuple(z['spacing'])} "
          f"max={z['array'].max():.1f}")
print('\nGATE-1 can now run off disk. Resampling to 193x229x193 is a PREREQUISITE '
      'and its interpolation choice must be recorded as a decision.')
