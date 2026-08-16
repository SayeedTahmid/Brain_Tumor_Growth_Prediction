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
