# ── MASTER_SAILOR_PIPELINE.ipynb § 04 — Pairs and folds (Phase 4) ─────────────
# CPU-only, seconds. Reads persisted manifests; touches no archive.

import json
import sailor
sailor.require("pairs", "patient_folds", "g5_fold_level", "frozen_splits",
               "target_session_manifest", "delta_t_phase_check")

from sailor.config import get_paths
from sailor.data import pairs as P, splits as S
from sailor.data.known_issues import excluded_sessions
from sailor.experiments.confound import delta_t_encodes_treatment

paths = get_paths(); ROOT = paths.dataset_root

DECISIONS = {
    "apply_known_exclusions": True,   # 26 patients; leakage sessions dropped
    "intensity_variant": "icor",      # bias-corrected, not z-scored
    "pair_mode": P.CONSECUTIVE,       # primary; all_ordered is a sensitivity
    "enforce_target_availability": True,
    "n_folds": 5, "n_repeats": 5, "seed": 1337,
}

clin = json.loads((ROOT / "01_DATA_FOUNDATION" / "v2_clinical_table.json").read_text())

# --- target availability -----------------------------------------------------
# Only sessions carrying ContrastEnhancedMask-CL may end a pair. Requires the
# audit to have written v2_target_sessions.json (Stage-1 §03e).
have_target = P.load_target_sessions(ROOT)
if have_target is None:
    raise RuntimeError(
        "v2_target_sessions.json missing — rerun the §03 audit (it reads from "
        "cache in seconds) before freezing, or pairs will be built where the "
        "primary target does not exist.")
print(f"sessions carrying the primary target: {len(have_target)}")

built = P.build_pairs(clin["rows"], sessions_with_target=have_target,
                      mode=DECISIONS["pair_mode"],
                      apply_known_exclusions=DECISIONS["apply_known_exclusions"])
P.print_report(built)

# --- does Δt itself encode treatment phase? ----------------------------------
# If it does, C1 (MRI + Δt) is not a treatment-free reference and the C2 - C1
# gap is not the quantity §11.1 assumes.
dtp = delta_t_encodes_treatment(built["pairs"])
print(f"\nΔt vs treatment phase: U={dtp.get('uncertainty_coefficient')} "
      f"acc={dtp.get('balanced_accuracy_of_delta_t_alone')} "
      f"median ratio={dtp.get('median_ratio_across_phases')}")
print(f"C1 treatment-contaminated: {dtp.get('c1_is_treatment_contaminated')}")
print(dtp.get("implication"))

# --- folds -------------------------------------------------------------------
meta = {s: {"n_sessions": sum(1 for r in clin["rows"] if r["subject"] == s)}
        for s in {r["subject"] for r in clin["rows"]}}
folds = S.make_folds(sorted(built["pairs_per_patient"]),
                     n_folds=DECISIONS["n_folds"], n_repeats=DECISIONS["n_repeats"],
                     seed=DECISIONS["seed"], patient_meta=meta)
assignment = S.assign_pairs(built["pairs"], folds)

g5 = S.g5_fold_leakage(folds, built["pairs"], excluded_sessions())
print(f"\n[guard] {g5['guard']} {g5['status']}: {g5['detail']}")
assert g5["status"] == "PASS", "fold-level leakage — do not freeze"

# --- freeze ------------------------------------------------------------------
frozen = S.freeze(folds, built, assignment, ROOT / "01_DATA_FOUNDATION",
                  decisions={**DECISIONS, "delta_t_phase_check": dtp})
print(f"\nfrozen: {frozen['path']}")
print(f"sha256: {frozen['content_sha256']}")
print(f"test pairs per fold: {assignment['min_test_pairs']}-{assignment['max_test_pairs']}")
