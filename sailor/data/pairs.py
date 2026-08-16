"""Longitudinal pair construction (Phase 4, §11-12).

A pair is (t, t+Δt) for one patient: the model sees the history up to t and must
predict the target at t+Δt. Getting this wrong is how a project produces an
impressive number that means nothing, so three rules are enforced structurally
rather than by convention.

1. A pair may not touch an excluded session. Sessions carrying future-information
   leakage (`sub-04/ses-01` and friends, whose masks were derived from their own
   ses-02) are not merely dropped as inputs — any pair spanning them is dropped,
   because the contamination travels with the session.

2. A pair requires the locked primary target at BOTH ends. A pair whose target is
   missing is not a pair with a missing label; it is not a pair.

3. Δt is carried on every pair with its provenance flag. Eight subjects have
   interpolated intervals (G7), and a pair built from an interpolated interval
   inherits that, so it is marked rather than silently averaged in.

Nothing here chooses hyperparameters, and nothing looks at a target value.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .known_issues import (DELTA_T_ESTIMATED, excluded_sessions,
                           excluded_subjects)

CONSECUTIVE = "consecutive"
ALL_ORDERED = "all_ordered"


def _ordinal(row: dict) -> int:
    return int(row.get("session_ordinal") or 0)


def build_pairs(clinical_rows: list[dict],
                sessions_with_target: set | None = None,
                mode: str = CONSECUTIVE,
                apply_known_exclusions: bool = True,
                min_days: float | None = None,
                max_days: float | None = None) -> dict:
    """Construct (t -> t+Δt) pairs.

    `sessions_with_target` is the set of (subject, session) tuples for which the
    locked primary target exists on disk. When None, target availability is
    assumed unknown and reported as such rather than assumed present.
    """
    if mode not in (CONSECUTIVE, ALL_ORDERED):
        raise ValueError(f"unknown pair mode {mode!r}")

    excluded_sess = excluded_sessions() if apply_known_exclusions else set()
    blocked_subs = excluded_subjects("CL manual masks") if apply_known_exclusions else set()

    by_subject: dict[str, list[dict]] = defaultdict(list)
    for r in clinical_rows:
        if r.get("subject") and r.get("session"):
            by_subject[r["subject"]].append(r)
    for rows in by_subject.values():
        rows.sort(key=_ordinal)

    pairs = []
    rejected = Counter()
    rejected_examples: dict[str, list] = defaultdict(list)

    def reject(reason: str, sub: str, a: str, b: str):
        rejected[reason] += 1
        if len(rejected_examples[reason]) < 10:
            rejected_examples[reason].append(f"{sub}: {a} -> {b}")

    for sub, rows in sorted(by_subject.items()):
        if sub in blocked_subs:
            rejected["subject_has_no_primary_target"] += len(rows)
            rejected_examples["subject_has_no_primary_target"].append(sub)
            continue
        n = len(rows)
        for i in range(n):
            js = [i + 1] if mode == CONSECUTIVE else range(i + 1, n)
            for j in js:
                if j >= n:
                    continue
                a, b = rows[i], rows[j]
                sa, sb = a["session"], b["session"]

                if (sub, sa) in excluded_sess or (sub, sb) in excluded_sess:
                    reject("touches_excluded_session", sub, sa, sb)
                    continue
                if sessions_with_target is not None:
                    if (sub, sa) not in sessions_with_target:
                        reject("input_session_lacks_primary_target", sub, sa, sb)
                        continue
                    if (sub, sb) not in sessions_with_target:
                        reject("target_session_lacks_primary_target", sub, sa, sb)
                        continue

                da, db = a.get("days_from_first"), b.get("days_from_first")
                delta_days = (None if (da is None or db is None) else round(db - da, 3))
                if delta_days is None:
                    reject("delta_t_unavailable", sub, sa, sb)
                    continue
                if delta_days <= 0:
                    reject("non_positive_delta_t", sub, sa, sb)
                    continue
                if min_days is not None and delta_days < min_days:
                    reject("below_min_days", sub, sa, sb)
                    continue
                if max_days is not None and delta_days > max_days:
                    reject("above_max_days", sub, sa, sb)
                    continue

                interpolated = sub in DELTA_T_ESTIMATED
                pairs.append({
                    "subject": sub,
                    "input_session": sa,
                    "target_session": sb,
                    "input_ordinal": _ordinal(a),
                    "target_ordinal": _ordinal(b),
                    "gap_in_ordinals": _ordinal(b) - _ordinal(a),
                    "delta_days": delta_days,
                    "delta_weeks": round(delta_days / 7.0, 4),
                    "delta_t_kind": ("INTERPOLATED" if interpolated
                                     else "DOCUMENTED_APPROXIMATE"),
                    "input_treatment": a.get("treatment_status"),
                    "target_treatment": b.get("treatment_status"),
                    "input_treatment_observed": bool(a.get("treatment_observed")),
                    "target_treatment_observed": bool(b.get("treatment_observed")),
                    "input_rano": a.get("rano"),
                    "target_rano": b.get("rano"),
                    "n_history_sessions": _ordinal(a),
                })

    per_subject = Counter(p["subject"] for p in pairs)
    deltas = [p["delta_days"] for p in pairs]
    n_interp = sum(1 for p in pairs if p["delta_t_kind"] == "INTERPOLATED")
    return {
        "pairs": pairs,
        "mode": mode,
        "n_pairs": len(pairs),
        "n_patients": len(per_subject),
        "pairs_per_patient": dict(per_subject),
        "min_pairs_per_patient": min(per_subject.values()) if per_subject else 0,
        "max_pairs_per_patient": max(per_subject.values()) if per_subject else 0,
        "n_pairs_with_interpolated_delta_t": n_interp,
        "fraction_interpolated": (round(n_interp / len(pairs), 4) if pairs else None),
        "delta_days_summary": _summary(deltas),
        "rejected": dict(rejected),
        "rejected_examples": {k: v for k, v in rejected_examples.items()},
        "exclusions_applied": {
            "sessions": sorted(f"{a}/{b}" for a, b in excluded_sess),
            "subjects": sorted(blocked_subs),
            "policy": ("A pair TOUCHING an excluded session is dropped, not just "
                       "the session itself: leakage travels with the session."),
        },
        "target_availability_known": sessions_with_target is not None,
    }


def _summary(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    import statistics as st
    s = sorted(vals)
    def q(p):
        k = (len(s) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 2)
    return {"n": len(s), "min": s[0], "q25": q(.25), "median": q(.5),
            "q75": q(.75), "max": s[-1],
            "mean": round(st.fmean(s), 2),
            "sd": round(st.pstdev(s), 2) if len(s) > 1 else 0.0}


def target_sessions_from_file_table(file_rows: list[dict],
                                    mask: str = "CL",
                                    component: str = "enhancing_t1wc") -> set:
    """(subject, session) tuples where the locked primary target exists."""
    return {(r["subject"], r["session"]) for r in file_rows
            if r.get("subject") and r.get("session")
            and r.get("annotation_kind") == mask
            and r.get("annotation_component") == component}


def load_target_sessions(project_root, prefix: str = "v2_") -> set | None:
    """Read the persisted set of sessions carrying the primary target.

    Returns None when the manifest is absent, so callers can report target
    availability as UNKNOWN rather than silently treating every session as
    having a mask.
    """
    import json
    from pathlib import Path
    p = Path(project_root) / "01_DATA_FOUNDATION" / f"{prefix}target_sessions.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    out = set()
    for s in data.get("sessions", []):
        sub, _, ses = s.partition("/")
        if sub and ses:
            out.add((sub, ses))
    return out or None


def print_report(built: dict) -> None:
    line = "-" * 78
    print(line)
    print(f"LONGITUDINAL PAIRS ({built['mode']})")
    print(line)
    print(f"  pairs: {built['n_pairs']}   patients: {built['n_patients']}")
    print(f"  pairs per patient: min {built['min_pairs_per_patient']}, "
          f"max {built['max_pairs_per_patient']}")
    d = built["delta_days_summary"]
    if d.get("n"):
        print(f"  Δt days: min {d['min']}  q25 {d['q25']}  median {d['median']}  "
              f"q75 {d['q75']}  max {d['max']}  (mean {d['mean']} sd {d['sd']})")
    print(f"  pairs with INTERPOLATED Δt: {built['n_pairs_with_interpolated_delta_t']}"
          f" ({built['fraction_interpolated']})")
    if not built["target_availability_known"]:
        print("  target availability UNKNOWN — pass sessions_with_target to enforce it")
    print("  rejected:")
    for k, v in sorted(built["rejected"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:<38} {v}")
        for ex in built["rejected_examples"].get(k, [])[:3]:
            print(f"        e.g. {ex}")
    print(line)
