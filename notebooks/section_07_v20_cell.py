# =============================================================================
# SAILOR — v20 / sailor 0.20.0-stage2
#
# The v0.19 export saved every dose voxel but NOT the affine, so GATE-1 still
# could not relate a 256x256x190 dose map to a 193x229x193 mask. v0.20 parses
# the spatial terms the reader previously discarded.
#
# CELL 2 IS FREE — it re-reads HEADERS ONLY (read_volumes=False), which is a few
# KB per member instead of a full voxel read. Minutes, not hours.
#
# NOTHING BELOW CHANGES A LOCK.
# =============================================================================

# --- CELL 1 — bootstrap ------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

VERSION = 20
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
sailor.require('nifti_affine', 'affine_validation', 'dose_alignment')
print('capabilities OK |', ROOT)


# --- CELL 2 — header-only pass. NO voxel reads, so minutes not hours. --------
from sailor.stage1.audit import run_stage1_audit
report = run_stage1_audit(
    paths, verify_hashes=False, force_rescan=True,
    archives_to_scan=['derivatives.tar.bz2'],
    read_volumes=False,          # <-- headers only: this is what makes it cheap
    verbose=True)
print('\nguards:', report['guards']['summary'])
# G10 will read INCONCLUSIVE here by design: no volume was read. The intensity
# evidence lives in the v18 cache and is unaffected.


# --- CELL 3 — is the dose in the same space as the masks? -------------------
from sailor.utils.persist import load_cache, cache_dir
from sailor.stage1 import dose_alignment

keys = sorted(p.stem for p in cache_dir(ROOT).glob('audit_scan_structural_*.json')
              if '__' not in p.stem)
print('structural caches:', keys)
hdr = load_cache(ROOT, keys[-1])['raw']['headers']
print('headers:', len(hdr))

from collections import Counter
print('spatial_status across all volumes:',
      dict(Counter(v.get('spatial_status') for v in hdr.values())))

res = dose_alignment.report(hdr)
dose_alignment.print_report(res)

import json, os
out = os.path.join(ROOT, '06_QC_REPORTS', 'v2_dose_alignment.json')
tmp = out + '.tmp'
with open(tmp, 'w') as f: json.dump(res, f, indent=2, default=str)
os.replace(tmp, out)
print('\nwritten:', out)

print("""
READING THE VERDICT
  SAME_SPACE_CROP_PAD -> field-of-view difference only. The dose goes onto the
      reference grid by an INDEX SHIFT with zero interpolation, so no smoothing
      enters the GATE-1 isodose boundary. Best case.
  DIFFERENT_SPACE_REGISTRATION_REQUIRED -> a real registration problem. Matching
      by shape would displace the dose by centimetres and GATE-1 would return a
      plausible-looking number measuring nothing.
  UNKNOWN_NO_AFFINE_ON_RECORD -> the files themselves carry no position. GATE-1
      cannot proceed from this evidence and the escalation is the authors.
""")
