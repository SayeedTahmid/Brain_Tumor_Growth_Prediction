"""Export dose maps and baseline masks to disk for GATE-1.

GATE-1 asks whether the RT dose distribution is substantially a smoothed
dilation of the baseline tumour — Dice between the >=95% isodose region and the
baseline CL mask dilated at 5/10/15/20 mm, plus surface distances and centroid
displacement. That needs VOXELS, and the audit only ever recorded statistics.

bz2 is sequential, so re-reading 26 dose maps costs another ~3 h decompression of
a 43 GB archive. This handler therefore rides along in a pass being run anyway
and writes the arrays to `.npz`, after which GATE-1 is pure CPU work off disk and
can be re-run as often as the analysis needs.

MEASURED, not assumed: the 26 dose maps are on `256x256x190` and `240x240x190`
grids, while the derivatives are `193x229x193`. NOT ONE dose map is on the
reference grid, so resampling is a prerequisite of GATE-1 rather than a detail of
it. This module exports; it does not resample, because the resampling choice
(interpolation order, whether dose is treated as an extensive quantity) is a
decision to record, not one to bury in an exporter.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..data.archives import MemberHandler, classify

_SUB = re.compile(r"(sub-[A-Za-z0-9]+)")
_SES = re.compile(r"(ses-[A-Za-z0-9]+)")

#: Dose maps, and the masks GATE-1 compares them against.
DOSE_BASENAMES = ("DoseMap.nii.gz", "DoseMap_unscaled.nii.gz")
MASK_BASENAMES = ("ContrastEnhancedMask-CL.nii.gz", "ContrastEnhancedMask-ONCO.nii.gz")


class ArrayExportHandler(MemberHandler):
    """Writes selected volumes to `.npz` during a pass, one file per volume.

    Streaming to disk rather than accumulating in memory: 26 dose maps at
    ~12 M voxels each would be several GB held live, and a Colab runtime that
    dies at hour two would take the whole pass with it.
    """

    name = "array_export"

    def __init__(self, outdir, basenames: tuple = DOSE_BASENAMES + MASK_BASENAMES,
                 sessions: set | None = None, max_bytes: int = 400 << 20,
                 compress: bool = True):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.basenames = {b.lower() for b in basenames}
        #: Optional (subject, session) filter. Dose maps are per-subject and
        #: carry no session, so they are always exported when the basename matches.
        self.sessions = sessions
        self.max_bytes = max_bytes
        self.compress = compress
        self.exported: dict[str, dict] = {}
        self.errors: list[dict] = []

    def match(self, name: str, size: int) -> bool:
        if classify(name) != "nifti" or size > self.max_bytes:
            return False
        base = name.rsplit("/", 1)[-1].lower()
        if base not in self.basenames:
            return False
        if self.sessions is None or base.startswith("dosemap"):
            return True
        s, e = _SUB.search(name), _SES.search(name)
        return bool(s and e and (s.group(1), e.group(1)) in self.sessions)

    def handle(self, archive, name, size, fh):
        import numpy as np
        from ..data.nifti_header import read_array, scaled
        hdr, arr = read_array(fh, name)
        vals = np.asarray(scaled(arr, hdr))
        s = _SUB.search(name)
        e = _SES.search(name)
        stem = "__".join(x for x in (s.group(1) if s else "nosub",
                                     e.group(1) if e else "noses",
                                     name.rsplit("/", 1)[-1].replace(".nii.gz", "")))
        out = self.outdir / f"{stem}.npz"
        save = np.savez_compressed if self.compress else np.savez
        # Geometry travels WITH the array. A dose map separated from its shape
        # and spacing cannot be resampled onto the reference grid later, and
        # these maps are on three different grids.
        aff = hdr.affine
        # v0.20 — the affine travels with the array. Without it a 256x256x190
        # dose map cannot be placed on a 193x229x193 grid at all, and the v0.19
        # export was insufficient for GATE-1 despite holding every voxel.
        save(out, array=vals.astype("float32"),
             shape=np.array(hdr.shape), spacing=np.array(hdr.spacing),
             affine=np.array(aff if aff is not None else np.full((4, 4), np.nan)),
             spatial_status=np.array(hdr.spatial_status))
        self.exported[f"{s.group(1) if s else '?'}/{name.rsplit('/', 1)[-1]}"] = {
            "path": str(out), "name": name,
            "shape": list(hdr.shape), "spacing": list(hdr.spacing),
            "dtype": hdr.dtype,
            "n_nonzero": int((vals != 0).sum()),
            "spatial_status": hdr.spatial_status,
            "affine": None if hdr.affine is None else
                      [[round(float(x), 6) for x in r] for r in hdr.affine],
            "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals)),
            "bytes_on_disk": out.stat().st_size,
        }

    def result(self):
        return {"n_exported": len(self.exported), "errors": self.errors}


def summarise(exported: dict, reference_shape=(193, 229, 193)) -> dict:
    """What GATE-1 must resolve before it can compute anything."""
    from collections import Counter
    dose = {k: v for k, v in exported.items() if "DoseMap" in k}
    shapes = Counter(tuple(v["shape"]) for v in dose.values())
    on_grid = [k for k, v in dose.items() if tuple(v["shape"]) == tuple(reference_shape)]
    maxima = sorted(v["max"] for v in dose.values())
    return {
        "n_dose_exported": len(dose),
        "shape_frequency": {str(k): n for k, n in shapes.most_common()},
        "reference_shape": list(reference_shape),
        "n_on_reference_grid": len(on_grid),
        "dose_max_range": [maxima[0], maxima[-1]] if maxima else None,
        "resampling_required": len(on_grid) < len(dose),
        "note": (
            "Dose maxima clustering near 60-65 are consistent with a standard 60 Gy "
            "prescription, so the values are physical Gy. Grids are NOT the derivative "
            "grid, so GATE-1 needs an explicit resampling step whose interpolation "
            "choice is recorded as a decision. This module does not resample."),
    }
