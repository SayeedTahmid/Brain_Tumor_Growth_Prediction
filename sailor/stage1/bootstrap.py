"""Project root creation (§14.1, §14.2).

Creates `DATASET_ROOT` and its subtree, writes the README that records creation
date, DATA_VERSION, the target lock and the legacy source path, and verifies
every directory is writable before the audit proceeds.

The legacy folder is opened read-only. This module contains no code path that
writes, renames, moves, deletes, or extracts into it.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ..config import (DATA_VERSION, PROJECT_SUBDIRS, Paths, target_lock)

README_TEMPLATE = """# {project_name}

Created (UTC): {created}
DATA_VERSION: {data_version}

## Target lock (§3.2 — locked before Stage 1, not re-openable)

    PRIMARY_TARGET_MASK      = {primary_target_mask}
    PRIMARY_TARGET_COMPONENT = {primary_target_component}
    SECONDARY_TARGET_MASK    = {secondary_target_mask}
    SENSITIVITY_TARGETS      = {sensitivity_targets}

A run whose manifest target does not match this lock is invalid.

## Provenance

Legacy source (READ-ONLY for the duration of the project):

    {legacy_root}

`00_CANONICAL/` holds read-only pointers to the verified EBRAINS files in the
legacy folder. It never holds copies (§4.1). `00_QUARANTINE/` holds pointers to
prior TaDiff artefacts, which are comparison targets only and are never inputs.

All new outputs use DATA_VERSION {data_version} and the `v2_` filename prefix.
"""


def ensure_project_root(paths: Paths, dry_run: bool = False) -> dict:
    """Create the tree and verify it is writable. Idempotent."""
    root = Path(paths.dataset_root)
    created, existing, failed = [], [], []
    if dry_run:
        return {"status": "DRY_RUN", "dataset_root": str(root),
                "would_create": [str(root / d) for d in PROJECT_SUBDIRS]}

    for rel in ["", *PROJECT_SUBDIRS]:
        d = root / rel if rel else root
        try:
            if d.exists():
                existing.append(str(d))
            else:
                d.mkdir(parents=True, exist_ok=True)
                created.append(str(d))
        except OSError as e:
            failed.append({"path": str(d), "error": str(e)})

    not_writable = []
    for rel in ["", *PROJECT_SUBDIRS]:
        d = root / rel if rel else root
        if not d.exists():
            continue
        probe = d / ".sailor_write_probe"
        try:
            probe.write_text("ok")
            probe.unlink()
        except OSError as e:
            not_writable.append({"path": str(d), "error": str(e)})

    legacy = Path(paths.legacy_root)
    legacy_state = {
        "path": str(legacy),
        "exists": legacy.exists(),
        "readable": os.access(legacy, os.R_OK) if legacy.exists() else False,
        "policy": "READ_ONLY — this module never writes into it",
    }

    status = "OK" if not failed and not not_writable else "FAIL"
    return {"status": status, "dataset_root": str(root),
            "created": created, "already_existed": existing,
            "mkdir_failures": failed, "not_writable": not_writable,
            "legacy": legacy_state}


def write_readme(paths: Paths) -> str:
    lock = target_lock()
    text = README_TEMPLATE.format(
        project_name=paths.project_name,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        data_version=DATA_VERSION,
        legacy_root=paths.legacy_root,
        **{k: lock[k] for k in ("primary_target_mask", "primary_target_component",
                                "secondary_target_mask", "sensitivity_targets")})
    p = Path(paths.dataset_root) / "README.md"
    p.write_text(text)
    return str(p)


def write_completion_record(paths: Paths, section: int, stage: int, status: str,
                            payload: dict) -> str:
    """`section_XX_complete.json` per the §15.6 schema."""
    from ..utils.env import git_state, hardware
    lock = target_lock()
    gs = git_state(Path(paths.code_root))
    hw = hardware()
    rec = {
        "section": section, "stage": stage, "status": status,
        "owner": payload.get("owner", "unassigned"),
        "implementation_id": os.environ.get("SAILOR_IMPLEMENTATION_ID", "UNSET"),
        "data_version": DATA_VERSION,
        "model_version": None, "preprocessing_version": None,
        "feature_shape": [],
        "primary_target_mask": lock["primary_target_mask"],
        "primary_target_component": lock["primary_target_component"],
        "conditioning_rung": None, "fold_scheme": None,
        "guards_passed": payload.get("guards_passed", []),
        "guards_failed": payload.get("guards_failed", []),
        "guards_inconclusive": payload.get("guards_inconclusive", []),
        "n_patients": payload.get("n_patients"),
        "n_sessions": payload.get("n_sessions"),
        "n_pairs": payload.get("n_pairs"),
        "seed": payload.get("seed", 1337),
        "gpu": hw.get("gpu"),
        "git_commit": gs.get("git_commit"), "git_branch": gs.get("git_branch"),
        "git_dirty": gs.get("git_dirty"),
        "artefacts": payload.get("artefacts", {}),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path(paths.dataset_root) / "01_DATA_FOUNDATION" / f"section_{section:02d}_complete.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    return str(out)
