"""v0.41 — axis relabel must not be reported as a true rotation.

`compare()` flags any direction-cosine difference as AFFINE_DIFFERS. These tests
pin the failure mode that hides behind it: a pure signed permutation produces
max|D-R| = 1.0 exactly, which is where all 26 measured SAILOR dose maps sit.
"""

import numpy as np

from sailor.stage1 import dose_alignment as DA


def _aff(m):
    a = np.eye(4)
    a[:3, :3] = m
    return a.tolist()


_REF = _aff(np.diag([-1.0, 1.0, 1.0]))          # MNI-like, 1 mm isotropic
_PERM = np.array([[0, 0, 1.0], [-1, 0, 0], [0, -1, 0]])


def test_identical_grids_are_a_relabel_of_themselves():
    r = DA.relabel_check({"affine": _REF}, {"affine": _REF})
    assert r["verdict"] == DA.AXIS_RELABEL
    assert r["axis_permutation"] == [0, 1, 2]


def test_pure_relabel_is_not_a_rotation():
    r = DA.relabel_check({"affine": _aff(_PERM)}, {"affine": _REF})
    assert r["verdict"] == DA.AXIS_RELABEL
    assert r["max_off_axis_residual"] == 0.0
    assert r["is_permutation"]


def test_the_existing_statistic_cannot_see_a_relabel():
    """The reason relabel_check exists. Guards the diagnosis itself."""
    rot_diff = float(np.abs(_PERM - np.diag([-1.0, 1.0, 1.0])).max())
    assert abs(rot_diff - 1.0) < 1e-12
    assert rot_diff > DA.ROTATION_TOL          # compare() returns AFFINE_DIFFERS
    assert DA.relabel_check({"affine": _aff(_PERM)},
                            {"affine": _REF})["verdict"] == DA.AXIS_RELABEL


def test_relabel_survives_anisotropic_spacing():
    m = _PERM @ np.diag([0.9375, 0.9375, 1.0])
    r = DA.relabel_check({"affine": _aff(m)}, {"affine": _REF})
    assert r["verdict"] == DA.AXIS_RELABEL
    assert r["relative_scale"][:2] == [0.9375, 0.9375]


def test_small_obliquity_is_caught():
    t = np.deg2rad(8.0)
    rot = np.array([[np.cos(t), -np.sin(t), 0.0],
                    [np.sin(t), np.cos(t), 0.0],
                    [0.0, 0.0, 1.0]])
    r = DA.relabel_check({"affine": _aff(rot @ _PERM)}, {"affine": _REF})
    assert r["verdict"] == DA.OBLIQUE
    assert r["max_off_axis_residual"] > 10 * DA.RELABEL_TOL


def test_missing_affine_is_unknown_not_a_guess():
    r = DA.relabel_check({"name": "sub-09/DoseMap.nii.gz"}, {"affine": _REF})
    assert r["verdict"] == DA.UNKNOWN
    assert r["subject"] == "sub-09"


def test_report_conservative_reading_binds():
    t = np.deg2rad(8.0)
    rot = np.array([[np.cos(t), -np.sin(t), 0.0],
                    [np.sin(t), np.cos(t), 0.0],
                    [0.0, 0.0, 1.0]])
    hdr = {
        "r": {"name": "sub-01/ContrastEnhancedMask-CL.nii.gz", "affine": _REF},
        "a": {"name": "sub-01/DoseMap.nii.gz", "affine": _aff(_PERM)},
        "b": {"name": "sub-02/DoseMap.nii.gz", "affine": _aff(rot @ _PERM)},
    }
    res = DA.relabel_report(hdr)
    assert res["verdict"] == DA.OBLIQUE       # one oblique volume binds
    assert res["n_dose_volumes"] == 2
    assert res["does_not_resample"] is True


def test_capability_is_registered():
    import sailor
    sailor.require("relabel_check", "dose_alignment")
