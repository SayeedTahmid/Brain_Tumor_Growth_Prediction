# SAILOR — treatment-aware longitudinal brain-MRI prediction

Code for the SAILOR study. **Code lives here; data never does** (§18.1, §18.2).

---

## What is and is not in this repository

Tracked: `sailor/`, `tests/`, `configs/`, `notebooks/`, docs.

Never tracked (`.gitignore` enforces it): medical images, masks, dose maps,
checkpoints, generated features, secrets, and personal Drive paths. A `.git`
directory synced by Drive corrupts under concurrent access, so the repository is
cloned into the runtime and Drive holds data only.

```
CODE_ROOT   = this repository, cloned into the runtime
DATA_ROOT   = Google Drive — canonical data, outputs, checkpoints
LEGACY_ROOT = Google Drive — sailor_v1, READ-ONLY
```

## First-time setup

```bash
git init
git add -A
git commit -m "Initial commit: sailor 0.20.0-stage2, Stage 1 complete"
git branch -M main
git remote add origin <your-remote>
git push -u origin main
```

Then, locally and untracked:

```bash
cp configs/local_paths.example.py configs/local_paths.py   # edit; never commit
```

Environment variables override both:
`SAILOR_PROJECT_NAME`, `SAILOR_LEGACY_ROOT`, `SAILOR_CODE_ROOT`.

## Running from Colab

Replace the zip-upload bootstrap with a clone. A notebook cell that hardcodes
`/content/drive/MyDrive/...` is a policy violation, not a convenience (§18.2) —
the path belongs in `configs/local_paths.py` or an environment variable.

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone -q <your-remote> /content/SAILOR || (cd /content/SAILOR && git pull -q)

import sys, os
sys.path.insert(0, '/content/SAILOR')
os.environ['SAILOR_CODE_ROOT'] = '/content/SAILOR'

import sailor
from sailor.config import get_paths
paths = get_paths(); ROOT = paths.dataset_root
print(sailor.__version__, ROOT)
```

**Why this matters beyond tidiness.** §12 requires every experiment to record
`git_commit`, `git_branch` and `git_dirty`. Run from a zip, `git_state()` returns
`git_available: False` and every artefact is stamped `NOT_REPRODUCIBLE`. Run from
a clone at a clean commit and the same artefacts are publishable. Nothing else
changes.

## Pre-merge gate (§18.4)

All six must hold:

1. `python -m pytest tests/ -q` passes
2. synthetic smoke test passes — `python tests/smoke_stage1.py`
3. the affected notebook section runs
4. contract and shape assertions pass
5. **no provenance or leakage guard is weakened** — checked, not assumed
6. the commit is recorded in the relevant completion JSON

Condition 5 is the one to be careful about. A diff that changes a guard's return
status, removes a guard, or turns a `FAIL`/`INCONCLUSIVE` path into `PASS` must
be called out explicitly and approved on its own terms. Weakening a guard to make
a section green is the single most damaging change anyone can make here.

## Branches (§18.3)

`member-1/<section>` … `member-4/<section>`. Shared modules are never forked:
copying `guards.py` into a personal variant to make one section pass is exactly
the failure this rule prevents.

## Current state

`sailor 0.20.0-stage2` · 175 tests · Stage 1 substantively complete.

| item | state |
|---|---|
| Frozen split | `40e674ee52…`, 26 patients, 208 pairs, frozen 12 Aug 2026 |
| Primary target | `CL` / `enhancing_t1wc` — LOCKED (§3.2) |
| sub-25 adjudication | CLOSED — true sustained complete response, retained |
| PLHM / `-icor` leakage | CLEARED — 1.564 vs a pre-registered 0.50 |
| Treatment confound | **UNTENABLE** — U = 0.814, balanced accuracy 0.921 (§5) |
| RANO coding | `UNVERIFIED` — 1 = complete response REFUTED; not used as evidence |
| G1 | FAIL by count; adjudicated. 7 all-zero masks, all sub-25, retained |
| G7 | FAIL — all Δt approximate; sensitivity analysis required |
| G10 | cohort-wide statistics measured; normalisation not yet chosen |
| GATE-1 | blocked on dose alignment (header-only pass, minutes) |

Full state: `SAILOR_HANDOFF_COMPLETE_STATE_v3.md`.

## Two rules binding on Phase 5

Recorded in `10_EXPERIMENTS/v2_sub25_adjudication_final.json`, **not yet
implemented in code**:

1. **Empty targets.** 5 of 208 pairs are empty→empty. Dice is undefined there and
   must not silently drop out of a mean. Report exact-zero agreement as a
   separate count, volume-change error (defined at 0), and change-region error
   per AMD-005. Any Dice mean must state how many pairs it excluded and why.
2. **Consecutive.** Consecutive *ordinal* sessions where both ends carry the
   primary target. No pair is bridged across a missing target.
