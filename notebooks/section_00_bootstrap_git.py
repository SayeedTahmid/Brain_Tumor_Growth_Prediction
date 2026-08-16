# =============================================================================
# BOOTSTRAP — clone, not zip.  Replaces every earlier zip-upload cell.
#
# WHY THIS MATTERS BEYOND TIDINESS: §12 requires every experiment to record
# git_commit / git_branch / git_dirty. Run from a zip, git_state() returns
# git_available=False and every artefact written is stamped NOT_REPRODUCIBLE.
# Run from a clone at a clean commit and the SAME artefacts are publishable.
# Nothing else about the code changes.
#
# §18.2: a cell that hardcodes /content/drive/MyDrive/... is a policy violation.
# Paths belong in configs/local_paths.py (gitignored) or environment variables.
# =============================================================================
import os, sys, subprocess

REMOTE    = os.environ.get('SAILOR_REMOTE', '<your-remote-url>')
CODE_ROOT = os.environ.get('SAILOR_CODE_ROOT', '/content/SAILOR')

from google.colab import drive
drive.mount('/content/drive')

if os.path.isdir(os.path.join(CODE_ROOT, '.git')):
    subprocess.run(['git', '-C', CODE_ROOT, 'pull', '--ff-only'], check=False)
else:
    subprocess.run(['git', 'clone', REMOTE, CODE_ROOT], check=True)

for m in [k for k in sys.modules if k == 'sailor' or k.startswith('sailor.')]:
    del sys.modules[m]
sys.path = [p for p in sys.path if p != CODE_ROOT]
sys.path.insert(0, CODE_ROOT)
os.environ['SAILOR_CODE_ROOT'] = CODE_ROOT

import sailor
from sailor.config import get_paths
from sailor.utils.persist import reproducibility_stamp

paths = get_paths(); ROOT = paths.dataset_root
print('sailor', sailor.__version__)
print('ROOT  ', ROOT)

r = reproducibility_stamp()
print('\ngit_commit :', (r.get('git_commit') or 'NONE')[:12])
print('git_branch :', r.get('git_branch'))
print('git_dirty  :', r.get('git_dirty'))
print('status     :', r['publication_status'])

# A dirty tree does not stop work — it changes what the work may be used for.
# Phase 5 produces the persistence baseline every C-rung is measured against,
# so it is worth running from a clean commit rather than re-running later.
if r.get('git_dirty') or not r.get('git_available'):
    print('\n*** Results from this runtime are NOT publishable as-is (§18.5). ***')
