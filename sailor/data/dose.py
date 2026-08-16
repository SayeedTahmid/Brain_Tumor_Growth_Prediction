"""Dose-map prerequisites for rung C3 (§11.1, §19.2).

Four questions the audit must answer with measurements, not assumptions:
  how many of the 27 patients have a dose map; what space and resolution it is
  in; whether it needs registration to MNI; how it should be represented at
  TMZ-phase timepoints.

The fourth is a design question, not a measurement, so this module reports the
evidence that constrains it — in particular that a CRT-derived map is constant
across a patient's sessions, which is why P2 exists (§11.1).
"""

from __future__ import annotations

from collections import Counter, defaultdict

# Grid of the descriptor's MNI152 ICBM 2009c 1 mm derivative space. A dose map
# matching this grid needs no resampling; anything else does. The reference grid
# itself is measured from the derivatives during the audit and this literal is
# only the fallback used when no derivative volume was measured.
MNI_1MM_REFERENCE_SHAPE = [193, 229, 193]
MNI_1MM_REFERENCE_SPACING = [1.0, 1.0, 1.0]


def collect(file_rows: list[dict], volume_stats: dict,
            reference_shape=None, reference_spacing=None,
            header_stats: dict | None = None) -> dict:
    """`header_stats` lets a structural pass answer the registration question.

    Grid and spacing come from the NIfTI header; only value ranges need the full
    array. Requiring volume statistics for both meant a header-only pass reported
    UNVERIFIED when it had in fact measured everything registration depends on.
    """
    ref_shape = list(reference_shape) if reference_shape else MNI_1MM_REFERENCE_SHAPE
    ref_spacing = [round(float(x), 3) for x in
                   (reference_spacing or MNI_1MM_REFERENCE_SPACING)]

    dose_rows = [r for r in file_rows if r["annotation_kind"] == "dose_map"]
    by_subject: dict[str, list] = defaultdict(list)
    shapes, spacings, dtypes = Counter(), Counter(), Counter()
    value_ranges = []
    needs_registration = []
    header_stats = header_stats or {}
    for r in dose_rows:
        st = (volume_stats.get(r["path"])
              or volume_stats.get(f"{r.get('archive')}::{r['path']}")
              or header_stats.get(r["path"])
              or header_stats.get(f"{r.get('archive')}::{r['path']}"))
        shape = st.get("shape") if st else None
        spacing = [round(float(x), 3) for x in st["spacing"]] if st and st.get("spacing") else None
        if shape:
            shapes[tuple(shape)] += 1
        if spacing:
            spacings[tuple(spacing)] += 1
        if st and st.get("dtype"):
            dtypes[st["dtype"]] += 1
        if st and not st.get("geometry_only"):
            value_ranges.append({"path": r["path"], "min": st.get("min"),
                                 "max": st.get("max"), "mean": st.get("mean"),
                                 "n_nonzero": st.get("n_nonzero"),
                                 "has_fractional_values": st.get("has_fractional_values")})
        on_grid = (shape is not None and list(shape) == ref_shape
                   and spacing is not None and spacing == ref_spacing)
        if shape is not None:
            needs_registration.append({"path": r["path"], "on_mni_grid": bool(on_grid),
                                       "shape": shape, "spacing": spacing})
        if r["subject"]:
            by_subject[r["subject"]].append(r["path"])

    n_measured = len([x for x in needs_registration])
    n_on_grid = len([x for x in needs_registration if x["on_mni_grid"]])
    if not dose_rows:
        reg_status = "NO_DOSE_FILES_FOUND"
    elif n_measured == 0:
        reg_status = "UNVERIFIED_NO_GEOMETRY_MEASURED"
    elif n_on_grid == n_measured:
        reg_status = "ALL_ON_MNI_REFERENCE_GRID"
    elif n_on_grid == 0:
        reg_status = "NONE_ON_MNI_REFERENCE_GRID_REGISTRATION_REQUIRED"
    else:
        reg_status = "MIXED_REGISTRATION_REQUIRED_FOR_SOME"

    per_subject_temporal = {
        sub: {"n_dose_files": len(paths),
              "time_varying_evidence": ("MULTIPLE_FILES_CHECK_IF_PER_SESSION"
                                        if len(paths) > 1 else "SINGLE_STATIC_MAP")}
        for sub, paths in by_subject.items()
    }

    return {
        "n_dose_files": len(dose_rows),
        "n_patients_with_dose": len(by_subject),
        "patients_with_dose": sorted(by_subject.keys()),
        "reference_grid": {"shape": ref_shape, "spacing": ref_spacing,
                           "source": "measured" if reference_shape else "fallback_literal"},
        "shape_frequency": [[list(k), v] for k, v in shapes.most_common()],
        "spacing_frequency": [[list(k), v] for k, v in spacings.most_common()],
        "dtype_frequency": dtypes.most_common(),
        "value_ranges": value_ranges[:50],
        "registration_status": reg_status,
        "n_measured": n_measured,
        "n_on_mni_grid": n_on_grid,
        "per_subject": per_subject_temporal,
        "tmz_representation": {
            "question": "How is dose represented at TMZ-phase timepoints?",
            "measured_constraint": (
                "Dose maps derive from CRT. Where a patient has one map, it is "
                "constant across that patient's sessions."),
            "implication": (
                "A time-constant map is a spatial prior modulated by dt, not a "
                "time-varying treatment signal; this must be stated in methods "
                "and is the reason control P2 is mandatory before any C3 claim."),
            "decision": "DEFERRED_TO_PHASE_GATE — not chosen by the audit",
        },
        "status": "OK" if dose_rows else "ABSENT",
    }
