"""Patient-level cross-validation, frozen once (Phase 4, §11-12, G5).

Splitting is at the PATIENT level, never the session or pair level. With 27
patients contributing 5-19 sessions each, a session-level split would put the
same tumour in train and test, and every metric would measure memorisation.

Folds are written once to a manifest and reused by every downstream rung. A
split regenerated per experiment is not a split; it is a hyperparameter, and the
project has 14 phases in which to accidentally tune it.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SEED = 1337


def _rng(seed: int):
    import numpy as np
    return np.random.default_rng(seed)


def _strata(patient_meta: dict[str, dict], key: str, n_bins: int = 3) -> dict[str, int]:
    """Bin a continuous patient-level covariate for balanced folds.

    Stratifying on the number of sessions keeps a fold from receiving only short
    or only long series, which would make its Δt distribution unlike the others'.
    """
    vals = [(p, m.get(key)) for p, m in patient_meta.items()]
    numeric = sorted((v for _, v in vals if v is not None))
    if not numeric:
        return {p: 0 for p, _ in vals}
    cuts = [numeric[int(len(numeric) * (i + 1) / n_bins) - 1]
            for i in range(n_bins - 1)]
    out = {}
    for p, v in vals:
        if v is None:
            out[p] = 0
            continue
        b = 0
        for c in cuts:
            if v > c:
                b += 1
        out[p] = b
    return out


def make_folds(patients: list[str], n_folds: int = 5, n_repeats: int = 5,
               seed: int = DEFAULT_SEED,
               patient_meta: dict[str, dict] | None = None,
               stratify_on: str = "n_sessions") -> dict:
    """Repeated stratified K-fold over PATIENTS.

    Repeats exist because with 26-27 patients a single 5-fold split has test
    folds of ~5 patients, and one unusual patient can move a fold mean further
    than any modelling choice. Repeats give the patient-level bootstrap something
    to average over; they do not create independent samples.
    """
    patients = sorted(set(patients))
    if n_folds < 2 or n_folds > len(patients):
        raise ValueError(f"n_folds={n_folds} invalid for {len(patients)} patient(s)")

    strata = ({p: 0 for p in patients} if not patient_meta
              else _strata(patient_meta, stratify_on))
    rng = _rng(seed)
    repeats = []
    for rep in range(n_repeats):
        by_stratum: dict[int, list[str]] = defaultdict(list)
        for p in patients:
            by_stratum[strata.get(p, 0)].append(p)
        assignment: dict[str, int] = {}
        # Deal each stratum round-robin from a rotating start, so no fold
        # systematically receives the first patient of every stratum.
        for s_idx, (s, members) in enumerate(sorted(by_stratum.items())):
            order = list(members)
            rng.shuffle(order)
            offset = int(rng.integers(0, n_folds))
            for i, p in enumerate(order):
                assignment[p] = (i + offset + s_idx) % n_folds
        folds = []
        for k in range(n_folds):
            test = sorted(p for p, f in assignment.items() if f == k)
            train = sorted(p for p in patients if p not in test)
            folds.append({"fold": k, "train_patients": train,
                          "test_patients": test,
                          "n_train": len(train), "n_test": len(test)})
        repeats.append({"repeat": rep, "folds": folds})

    return {
        "scheme": f"repeated_stratified_patient_kfold_{n_folds}x{n_repeats}",
        "n_folds": n_folds, "n_repeats": n_repeats, "seed": seed,
        "n_patients": len(patients), "patients": patients,
        "stratify_on": stratify_on if patient_meta else None,
        "strata": strata if patient_meta else None,
        "repeats": repeats,
        "unit_of_split": "patient",
        "rationale": ("Patients contribute many sessions of the same tumour; a "
                      "session- or pair-level split would place the same tumour "
                      "on both sides and measure memorisation."),
    }


def assign_pairs(pairs: list[dict], folds: dict) -> dict:
    """Attach (repeat, fold, role) to every pair from the patient-level split."""
    out = []
    counts = Counter()
    for rep in folds["repeats"]:
        for f in rep["folds"]:
            test = set(f["test_patients"])
            for p in pairs:
                role = "test" if p["subject"] in test else "train"
                out.append({**p, "repeat": rep["repeat"], "fold": f["fold"],
                            "role": role})
                counts[(rep["repeat"], f["fold"], role)] += 1
    per_fold = []
    for rep in folds["repeats"]:
        for f in rep["folds"]:
            per_fold.append({
                "repeat": rep["repeat"], "fold": f["fold"],
                "n_train_pairs": counts[(rep["repeat"], f["fold"], "train")],
                "n_test_pairs": counts[(rep["repeat"], f["fold"], "test")],
                "n_test_patients": f["n_test"]})
    test_sizes = [x["n_test_pairs"] for x in per_fold]
    return {"assigned": out, "per_fold": per_fold,
            "min_test_pairs": min(test_sizes) if test_sizes else 0,
            "max_test_pairs": max(test_sizes) if test_sizes else 0}


# ------------------------------------------------------------------ G5 (folds)

def g5_fold_leakage(folds: dict, pairs: list[dict],
                    excluded_sessions: set | None = None) -> dict:
    """Fold-level leakage checks (§9 G5, Stage-2 scope).

    Five ways a split can be wrong, all checked:
      1. a patient in both train and test of the same fold
      2. a patient never appearing in any test fold
      3. a pair whose two ends land in different folds
      4. an excluded session reappearing inside a pair
      5. an empty test fold
    """
    problems = []
    subject_test_count = Counter()

    for rep in folds["repeats"]:
        for f in rep["folds"]:
            tr, te = set(f["train_patients"]), set(f["test_patients"])
            both = tr & te
            if both:
                problems.append({"type": "patient_in_train_and_test",
                                 "repeat": rep["repeat"], "fold": f["fold"],
                                 "patients": sorted(both)})
            if not te:
                problems.append({"type": "empty_test_fold",
                                 "repeat": rep["repeat"], "fold": f["fold"]})
            for p in te:
                subject_test_count[p] += 1

    never_tested = [p for p in folds["patients"] if subject_test_count[p] == 0]
    if never_tested:
        problems.append({"type": "patient_never_in_test", "patients": never_tested})

    # A pair is a single unit; both ends belong to one patient, so a split by
    # patient cannot straddle a pair. Verified rather than assumed.
    straddling = [p for p in pairs
                  if p.get("subject") is None
                  or p["input_session"] == p["target_session"]]
    if straddling:
        problems.append({"type": "degenerate_pair", "n": len(straddling)})

    if excluded_sessions:
        touching = [f"{p['subject']}: {p['input_session']}->{p['target_session']}"
                    for p in pairs
                    if (p["subject"], p["input_session"]) in excluded_sessions
                    or (p["subject"], p["target_session"]) in excluded_sessions]
        if touching:
            problems.append({"type": "pair_touches_excluded_session",
                             "n": len(touching), "examples": touching[:10]})

    status = "FAIL" if problems else "PASS"
    detail = ("; ".join(sorted({p["type"] for p in problems})) if problems else
              f"{folds['n_folds']}x{folds['n_repeats']} patient-level folds are "
              f"disjoint; every one of {folds['n_patients']} patient(s) appears in "
              "a test fold; no pair touches an excluded session.")
    return {"guard": "G5", "title": "Leakage (fold-level)", "status": status,
            "detail": detail,
            "evidence": {"problems": problems,
                         "n_test_appearances_per_patient": dict(subject_test_count),
                         "scheme": folds["scheme"]}}


# ------------------------------------------------------------------- freezing

def freeze(folds: dict, pairs_built: dict, assignment: dict, out_dir: Path,
           decisions: dict, prefix: str = "v2_") -> dict:
    """Write the split manifest with a content hash. Frozen means frozen."""
    from ..config import target_lock, DATA_VERSION

    core = {"scheme": folds["scheme"], "seed": folds["seed"],
            "patients": folds["patients"],
            "repeats": [[f["test_patients"] for f in r["folds"]]
                        for r in folds["repeats"]],
            "pairs": [[p["subject"], p["input_session"], p["target_session"]]
                      for p in pairs_built["pairs"]]}
    digest = hashlib.sha256(
        json.dumps(core, sort_keys=True).encode()).hexdigest()

    manifest = {
        "manifest": "longitudinal_pairs_and_folds",
        "status": "FROZEN",
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "content_sha256": digest,
        "data_version": DATA_VERSION,
        **target_lock(),
        "decisions": decisions,
        "folds": folds,
        "pairs": pairs_built,
        "fold_assignment_summary": assignment["per_fold"],
        "reuse_rule": ("Every downstream rung reads this manifest. A run whose "
                       "content_sha256 differs from the one recorded in its "
                       "completion record used a different split and its numbers "
                       "are not comparable with any other rung."),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{prefix}pairs_and_folds.json"
    p.write_text(json.dumps(manifest, indent=2, default=str))
    return {"path": str(p), "content_sha256": digest, "manifest": manifest}


def verify_frozen(path: Path) -> dict:
    """Recompute the hash of a frozen manifest and compare."""
    m = json.loads(Path(path).read_text())
    core = {"scheme": m["folds"]["scheme"], "seed": m["folds"]["seed"],
            "patients": m["folds"]["patients"],
            "repeats": [[f["test_patients"] for f in r["folds"]]
                        for r in m["folds"]["repeats"]],
            "pairs": [[p["subject"], p["input_session"], p["target_session"]]
                      for p in m["pairs"]["pairs"]]}
    digest = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
    return {"path": str(path), "recorded": m["content_sha256"],
            "recomputed": digest, "intact": digest == m["content_sha256"]}
