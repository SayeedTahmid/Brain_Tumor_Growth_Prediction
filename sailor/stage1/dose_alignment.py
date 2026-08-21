"""Are the dose maps in the same space as the derivative grid? (GATE-1 input)

GATE-1 compares the >=95% isodose region against the baseline tumour mask
dilated at 5/10/15/20 mm. That comparison is meaningless unless both volumes are
in the same anatomical space. The audit reported
`NONE_ON_MNI_REFERENCE_GRID_REGISTRATION_REQUIRED` on SHAPE alone, and shape
mismatch has two very different causes:

  crop/pad     same space, different field of view -> a pure index offset,
               recoverable exactly from the two affines, no interpolation
  different    the dose sits in the patient's planning space -> needs a real
  space        registration, and matching by shape would displace the dose by
               centimetres while still producing a plausible-looking Dice

Both give 1 mm spacing and a differing shape, so they are indistinguishable
without the affine. This module measures the difference rather than assuming
either.

MEASURED so far: 25 dose maps carry physical Gy (max 56-66, consistent with a
60 Gy prescription); sub-26 has only `DoseMap_unscaled` (max 32389, scale
unknown, NOT convertible by a guessed constant); sub-19 has no dose map. So
GATE-1 covers at most 25 patients, 24 if sub-26 is excluded.

This module does NOT resample and does NOT register. It reports what
relationship holds, so the resampling decision is recorded rather than buried.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

_SUB = re.compile(r"(sub-[A-Za-z0-9]+)")

SAME_GRID = "SAME_GRID"
CROP_PAD = "SAME_SPACE_CROP_PAD"
AFFINE_DIFFERS = "DIFFERENT_SPACE_REGISTRATION_REQUIRED"
UNKNOWN = "UNKNOWN_NO_AFFINE_ON_RECORD"

#: An index offset is a crop/pad only if it is (near) integral in voxels.
INTEGER_TOL = 0.05
#: Direction cosines must match this closely to call it the same orientation.
ROTATION_TOL = 1e-4
#: Off-axis residual above which R^-1 D is not a signed permutation. Fixed here
#: BEFORE the measurement it judges. A pure relabel gives exactly 0.0; an 8
#: degree obliquity gives 0.139, so 1e-3 separates them by two orders of
#: magnitude.
RELABEL_TOL = 1e-3

AXIS_RELABEL = "AXIS_RELABEL_SAME_ORIENTATION_FAMILY"
OBLIQUE = "OBLIQUE_TRUE_ROTATION"


def compare(dose_hdr: dict, ref_hdr: dict) -> dict:
    """Relationship between one dose volume and the reference grid."""
    da, ra = dose_hdr.get("affine"), ref_hdr.get("affine")
    out = {
        "subject": (_SUB.search(dose_hdr.get("name", "")) or [None])
                   and (_SUB.search(dose_hdr["name"]).group(1)
                        if _SUB.search(dose_hdr.get("name", "")) else None),
        "dose_shape": dose_hdr.get("shape"), "ref_shape": ref_hdr.get("shape"),
        "dose_spatial_status": dose_hdr.get("spatial_status"),
        "ref_spatial_status": ref_hdr.get("spatial_status"),
    }
    if not da or not ra:
        out.update(verdict=UNKNOWN, detail=(
            "at least one volume has no affine on record. Shape and spacing "
            "alone CANNOT distinguish a crop/pad from a different space, so no "
            "resampling may be attempted from this evidence."))
        return out

    D, R = np.array(da, dtype=float), np.array(ra, dtype=float)
    rot_diff = float(np.abs(D[:3, :3] - R[:3, :3]).max())
    out["max_direction_cosine_difference"] = round(rot_diff, 8)

    if rot_diff > ROTATION_TOL:
        out.update(verdict=AFFINE_DIFFERS, detail=(
            f"direction cosines differ by up to {rot_diff:.4g}. The volumes are "
            "oriented differently, so this is a genuine registration problem, "
            "not a field-of-view difference. Reshaping the array would place the "
            "dose in the wrong anatomy."))
        return out

    # Same orientation and scale: is the origin offset a whole number of voxels?
    try:
        idx = np.linalg.solve(R[:3, :3], D[:3, 3] - R[:3, 3])
    except np.linalg.LinAlgError:
        out.update(verdict=UNKNOWN, detail="reference affine is singular")
        return out
    frac = float(np.abs(idx - np.round(idx)).max())
    out["voxel_offset"] = [round(float(x), 4) for x in idx]
    out["max_fractional_offset"] = round(frac, 6)

    if list(dose_hdr.get("shape") or []) == list(ref_hdr.get("shape") or []) \
            and frac <= INTEGER_TOL and np.allclose(idx, 0, atol=INTEGER_TOL):
        out.update(verdict=SAME_GRID, detail="identical grid; no resampling needed")
    elif frac <= INTEGER_TOL:
        out.update(verdict=CROP_PAD, detail=(
            f"same orientation and spacing, origin offset by "
            f"{[int(round(x)) for x in idx]} voxels. This is a field-of-view "
            "difference: the dose can be placed on the reference grid by INDEX "
            "SHIFT with zero interpolation, which introduces no smoothing into "
            "a GATE-1 isodose boundary."))
    else:
        out.update(verdict=AFFINE_DIFFERS, detail=(
            f"origin offset is {frac:.4g} voxels from integral, so the grids are "
            "not aligned to voxel centres. Sub-voxel resampling is required and "
            "its interpolation order must be recorded as a decision."))
    return out


def relabel_check(dose_hdr: dict, ref_hdr: dict) -> dict:
    """Is D a signed axis permutation of R, or a genuine rotation?

    `compare()` returns AFFINE_DIFFERS as soon as max|D-R| exceeds ROTATION_TOL
    and returns early, so the integer-offset test is never reached. That
    statistic cannot separate a true rotation from a mere axis relabel: a pure
    permutation with flips gives max|D-R| = 1.0 exactly, and an 8 degree
    obliquity on top of one gives 1.139. Both read as AFFINE_DIFFERS.

    This decomposes M = R^-1 D instead. A relabel is recoverable by transpose
    and flip with zero interpolation; a rotation is not.

    A relabel verdict does NOT place the dose in MNI. It means the offset test
    that `compare()` skipped can be run on the relabelled affine, and only that
    test decides whether the space is shared.
    """
    da, ra = dose_hdr.get("affine"), ref_hdr.get("affine")
    name = dose_hdr.get("name", "")
    subject = _SUB.search(name).group(1) if _SUB.search(name) else None
    if not da or not ra:
        return {"subject": subject, "verdict": UNKNOWN,
                "detail": "at least one volume has no affine on record"}

    D = np.array(da, dtype=float)[:3, :3]
    R = np.array(ra, dtype=float)[:3, :3]
    try:
        M = np.linalg.solve(R, D)
    except np.linalg.LinAlgError:
        return {"subject": subject, "verdict": UNKNOWN,
                "detail": "reference affine is singular"}

    scale = np.linalg.norm(M, axis=0)
    if float(scale.min()) < 1e-12:
        return {"subject": subject, "verdict": UNKNOWN,
                "detail": "degenerate mapping; a dose axis collapses to zero"}

    P = M / scale
    resid, axes, signs = 0.0, [], []
    for j in range(3):
        col = P[:, j]
        i = int(np.argmax(np.abs(col)))
        axes.append(i)
        signs.append(int(np.sign(col[i])))
        resid = max(resid,
                    float(np.abs(np.delete(col, i)).max()),
                    float(abs(abs(col[i]) - 1.0)))

    out = {
        "subject": subject,
        "dose_shape": dose_hdr.get("shape"),
        "relative_scale": [round(float(s), 6) for s in scale],
        "axis_permutation": axes,
        "axis_signs": signs,
        "max_off_axis_residual": round(resid, 8),
        "is_permutation": sorted(axes) == [0, 1, 2],
    }
    out["verdict"] = (AXIS_RELABEL
                      if out["is_permutation"] and resid <= RELABEL_TOL
                      else OBLIQUE)
    out["detail"] = (
        "signed axis permutation: same orientation family, recoverable by "
        "transpose and flip with no interpolation. This does NOT establish a "
        "shared space -- run the offset test on the relabelled affine."
        if out["verdict"] == AXIS_RELABEL else
        f"off-axis residual {resid:.4g} exceeds {RELABEL_TOL}; the volumes are "
        "genuinely rotated relative to one another, so no relabelling recovers "
        "the reference grid.")
    return out


def relabel_report(headers: dict,
                   ref_basename: str = "ContrastEnhancedMask-CL.nii.gz",
                   dose_token: str = "DoseMap") -> dict:
    """Run `relabel_check` over every dose volume. Reports; never resamples."""
    ref = next((v for k, v in headers.items()
                if v.get("name", k).endswith(ref_basename)), None)
    if ref is None:
        return {"check": "dose_relabel", "verdict": UNKNOWN,
                "detail": f"no reference volume matching {ref_basename}"}

    rows = [relabel_check(v, ref) for k, v in sorted(headers.items())
            if dose_token in v.get("name", k)]
    verdicts = Counter(r["verdict"] for r in rows)
    if not rows:
        overall = UNKNOWN
    elif verdicts.get(UNKNOWN) or verdicts.get(OBLIQUE):
        overall = OBLIQUE if not verdicts.get(UNKNOWN) else UNKNOWN
    else:
        overall = AXIS_RELABEL

    return {
        "check": "dose_relabel",
        "refines": "dose_alignment",
        "blocking_for": "GATE-1",
        "verdict": overall,
        "counts": dict(verdicts),
        "n_dose_volumes": len(rows),
        "relabel_tol": RELABEL_TOL,
        "reference": {"name": ref.get("name"), "shape": ref.get("shape")},
        "per_volume": rows,
        "note": (
            "The conservative reading binds: a single OBLIQUE or UNKNOWN volume "
            "means no cohort-wide relabelling exists. A cohort-wide "
            "AXIS_RELABEL verdict does not by itself unblock GATE-1; it only "
            "makes the skipped offset test computable."),
        "does_not_resample": True,
    }


def report(headers: dict, ref_basename: str = "ContrastEnhancedMask-CL.nii.gz",
           dose_token: str = "DoseMap") -> dict:
    """Compare every dose map against the reference grid."""
    ref = next((v for k, v in headers.items()
                if v.get("name", k).endswith(ref_basename)), None)
    if ref is None:
        return {"check": "dose_alignment", "verdict": UNKNOWN,
                "detail": f"no reference volume matching {ref_basename}"}

    rows = [compare(v, ref) for k, v in sorted(headers.items())
            if dose_token in v.get("name", k)]
    verdicts = Counter(r["verdict"] for r in rows)
    if not rows:
        overall = UNKNOWN
    elif verdicts.get(UNKNOWN):
        overall = UNKNOWN            # any unknown blocks the whole gate
    elif verdicts.get(AFFINE_DIFFERS):
        overall = AFFINE_DIFFERS
    elif verdicts.get(CROP_PAD):
        overall = CROP_PAD
    else:
        overall = SAME_GRID

    return {
        "check": "dose_alignment",
        "blocking_for": "GATE-1",
        "verdict": overall,
        "counts": dict(verdicts),
        "n_dose_volumes": len(rows),
        "reference": {"name": ref.get("name"), "shape": ref.get("shape"),
                      "spatial_status": ref.get("spatial_status")},
        "per_volume": rows,
        "note": (
            "The conservative reading binds: if ANY dose map is UNKNOWN or in a "
            "different space, GATE-1 cannot be computed cohort-wide from a shape "
            "match. Per-patient handling must then be recorded, not assumed."),
        "does_not_resample": True,
    }


def print_report(res: dict) -> None:
    line = "-" * 78
    print(line)
    print("DOSE ALIGNMENT vs REFERENCE GRID  (GATE-1 prerequisite)")
    print(line)
    print(f"  verdict: {res['verdict']}")
    print(f"  counts : {res.get('counts')}")
    r = res.get("reference", {})
    print(f"  ref    : {r.get('shape')}  {r.get('spatial_status')}")
    for row in res.get("per_volume", [])[:8]:
        print(f"    {str(row.get('subject')):<9} {str(row.get('dose_shape')):<20} "
              f"{row['verdict']:<38} offset={row.get('voxel_offset')}")
    if len(res.get("per_volume", [])) > 8:
        print(f"    ... {len(res['per_volume']) - 8} more")
    print(f"\n  {res.get('note', '')}")
    print(line)
