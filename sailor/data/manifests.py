"""Draft manifests (§4: rebuilt from canonical files only).

Draft, because they are proposals until the Phase 1 gate approves them. Every
manifest carries `PRIMARY_TARGET_MASK` and `PRIMARY_TARGET_COMPONENT`; a run
whose manifest target does not match the lock is invalid (§3.2).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import DATA_VERSION, FILE_PREFIX, target_lock


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def session_manifest(sessions: dict, treatment: dict, delta_t: dict,
                     g9: dict) -> dict:
    survivors = set(g9["evidence"].get("surviving_sessions_primary", []))
    rows = []
    for sub, ses_map in sessions.items():
        for ses, node in ses_map.items():
            sid = f"{sub}/{ses}"
            t = treatment.get("records", {}).get(sub, {}).get(ses, {})
            dt = delta_t["per_session"].get(sid, {})
            primary = [a for a in node["annotations"]
                       if a["annotation_kind"] == "CL"
                       and a["annotation_component"] == "enhancing_t1wc"]
            rows.append({
                "subject": sub, "session": ses,
                "sequences_present": sorted(node["sequences"].keys()),
                "n_annotation_files": len(node["annotations"]),
                "primary_target_path": primary[0]["path"] if primary else None,
                "primary_target_available": bool(primary),
                "treatment_status": t.get("status"),
                "treatment_observed": t.get("observed", False),
                "treatment_missing_indicator": t.get("missing_indicator", 1),
                "days_from_prev": dt.get("days_from_prev"),
                "delta_t_kind": dt.get("kind", "UNAVAILABLE"),
                "delta_t_source": dt.get("source", "none"),
                "passes_primary_modality_requirement": sid in survivors,
            })
    rows.sort(key=lambda r: (r["subject"], r["session"]))
    return {
        "manifest": "session_inventory_draft",
        "status": "DRAFT — not approved for training use",
        "generated_utc": _stamp(),
        "data_version": DATA_VERSION,
        **target_lock(),
        "n_rows": len(rows),
        "n_with_primary_target": sum(1 for r in rows if r["primary_target_available"]),
        "rows": rows,
    }


def target_manifest(target_resolution: dict, g1: dict) -> dict:
    lock = target_lock()
    return {
        "manifest": "primary_target_registration",
        "status": "DRAFT",
        "generated_utc": _stamp(),
        **lock,
        "resolution_on_disk": target_resolution,
        "degeneracy": g1["evidence"]["primary"],
        "inventory_only": g1["evidence"]["inventory_only"],
        "binding_rule": ("ONCO and CL:t2wflair are inventoried only; they may not "
                         "influence cohort, preprocessing, hyperparameters, model "
                         "selection, or any experimental decision (§3.2)."),
    }


def write_all(project_root: Path, manifests: dict[str, dict]) -> dict:
    out_dir = Path(project_root) / "01_DATA_FOUNDATION"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, obj in manifests.items():
        p = out_dir / f"{FILE_PREFIX}{name}.json"
        p.write_text(json.dumps(obj, indent=2, default=str))
        written[name] = str(p)
    return written
