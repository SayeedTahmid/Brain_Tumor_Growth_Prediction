"""Slice extraction for direct visual verification of degenerate masks.

The sub-25 adjudication cannot be settled by any log, table or descriptor:
`code.tar.bz2` has no annotation scripts, `history.txt` never mentions sub-25,
and the descriptor says nothing about empty masks or the RANO integer coding.
What remains is the imaging itself. If sub-25's T1c shows no enhancing tissue at
ses-05..ses-10, the all-zero CL mask is CORRECT and those sessions are real data.
If enhancing tissue is plainly visible, the mask is a segmentation failure.

That question is answered by looking, which is why this module extracts slices
rather than computing another statistic. It rides along in the same archive pass
as everything else: bz2 is sequential, so a separate pass for a handful of
volumes would cost another full decompression.

This module MEASURES AND SHOWS. It classifies nothing and excludes nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..data.archives import MemberHandler, classify

_SUB = re.compile(r"(sub-[A-Za-z0-9]+)")
_SES = re.compile(r"(ses-[A-Za-z0-9]+)")

#: What to pull for a visual adjudication: the post-contrast image the manual
#: label is drawn from, the label itself, the automated comparator, and the
#: edema label that shows the session was processed at all.
DEFAULT_BASENAMES = (
    "T1c.nii.gz", "T1c-icor.nii.gz",
    "ContrastEnhancedMask-CL.nii.gz", "ContrastEnhancedMask-ONCO.nii.gz",
    "EdemaMask-CL.nii.gz",
)


class SliceHandler(MemberHandler):
    """Collects a few orthogonal slices for named (subject, session) targets."""

    name = "slices"

    def __init__(self, targets: list[tuple[str, str]],
                 basenames: tuple = DEFAULT_BASENAMES,
                 n_slices: int = 3, max_bytes: int = 200 << 20,
                 reference_z: tuple[int, int] | None = None,
                 z_margin: int = 8):
        self.targets = {(s, e) for s, e in targets}
        self.basenames = {b.lower() for b in basenames}
        self.n_slices = n_slices
        self.max_bytes = max_bytes
        # v0.19 (defect 25) — slice indices were taken from EACH VOLUME's own
        # non-zero extent. For a mask that is the lesion; for an image it is the
        # whole head, so the three slices landed at the skull base, mid-brain and
        # vertex. sub-25's tumour sits at z=114-132 and the captured T1c slices
        # were z=7-9/88/168-169: the tumour bed appeared in NONE of them and the
        # whole capture was unusable for its only purpose.
        #
        # `reference_z` is the anatomical range of interest, supplied by the
        # caller from a mask that actually contains the lesion. Every volume is
        # then sliced through the SAME anatomy, which is also what makes the
        # image and its mask comparable side by side.
        self.reference_z = tuple(reference_z) if reference_z else None
        self.z_margin = z_margin
        self.slices: dict[str, dict] = {}
        self.errors: list[dict] = []

    def match(self, name: str, size: int) -> bool:
        if classify(name) != "nifti" or size > self.max_bytes:
            return False
        if name.rsplit("/", 1)[-1].lower() not in self.basenames:
            return False
        s, e = _SUB.search(name), _SES.search(name)
        if not (s and e):
            return False
        return (s.group(1), e.group(1)) in self.targets

    def handle(self, archive, name, size, fh):
        import numpy as np
        from ..data.nifti_header import read_array, scaled
        hdr, arr = read_array(fh, name)
        vals = np.asarray(scaled(arr, hdr), dtype=float)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        if vals.ndim != 3:
            vals = vals.reshape(hdr.shape[:3])
        # Slice through the axial extent of the non-zero content when there is
        # any, otherwise through the geometric centre. An all-zero mask has no
        # content to centre on, and its companion image does — which is exactly
        # the comparison being made.
        nz = np.nonzero(vals)
        zmax = vals.shape[2] - 1
        if self.reference_z is not None:
            lo = max(0, min(zmax, self.reference_z[0] - self.z_margin))
            hi = max(lo, min(zmax, self.reference_z[1] + self.z_margin))
            basis = "reference_z"
            if not (0 <= self.reference_z[0] <= zmax and 0 <= self.reference_z[1] <= zmax):
                # A reference outside the volume means the caller measured it on
                # a different grid. Clamping keeps the capture usable, but the
                # slices are NOT the requested anatomy and must say so: v0.17
                # lost a 3-hour pass to slices that silently showed the wrong z.
                basis = "reference_z_OUT_OF_RANGE_CLAMPED"
                self.errors.append({
                    "name": name,
                    "error": f"reference_z {self.reference_z} outside z extent "
                             f"0-{zmax}; clamped to {lo}-{hi}. Slices are NOT the "
                             "requested anatomy."})
        elif nz[2].size:
            lo, hi = int(nz[2].min()), int(nz[2].max())
            basis = "own_content"
        else:
            lo, hi = 0, vals.shape[2] - 1
            basis = "whole_volume"
        idx = [int(lo + (hi - lo) * f) for f in
               ([0.5] if self.n_slices == 1 else
                [i / (self.n_slices - 1) for i in range(self.n_slices)])]
        s, e = _SUB.search(name).group(1), _SES.search(name).group(1)
        self.slices[f"{s}/{e}/{name.rsplit('/', 1)[-1]}"] = {
            "subject": s, "session": e, "name": name,
            "shape": list(vals.shape),
            "n_nonzero": int((vals != 0).sum()),
            "slice_indices": idx,
            "slice_basis": basis,
            "reference_z": list(self.reference_z) if self.reference_z else None,
            "axial_extent_of_content": [int(nz[2].min()), int(nz[2].max())] if nz[2].size else None,
            "slices": [vals[:, :, k].astype("float32").tolist() for k in idx],
        }

    def result(self):
        return {"n_slices": len(self.slices), "errors": self.errors}


def write_png(project_root, slice_record: dict, outdir: str = "06_QC_REPORTS") -> list:
    """Render collected slices to PNG so they can actually be looked at."""
    import numpy as np
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return [{"status": "MATPLOTLIB_ABSENT",
                 "note": "pip install matplotlib, or read the arrays from the JSON"}]
    d = Path(project_root) / outdir / "v2_sub25_slices"
    d.mkdir(parents=True, exist_ok=True)
    written = []
    for key, rec in sorted(slice_record.items()):
        arrs = [np.array(a) for a in rec["slices"]]
        fig, axes = plt.subplots(1, len(arrs), figsize=(4 * len(arrs), 4))
        axes = np.atleast_1d(axes)
        for ax, a, k in zip(axes, arrs, rec["slice_indices"]):
            ax.imshow(np.rot90(a), cmap="gray")
            ax.set_title(f"z={k}", fontsize=8)
            ax.axis("off")
        fig.suptitle(f"{key}   nonzero={rec['n_nonzero']}", fontsize=9)
        p = d / (key.replace("/", "__").replace(".nii.gz", "") + ".png")
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(str(p))
    return written


#: Measured axial extent of sub-25's enhancing tumour, from the ses-01 and ses-02
#: CL masks (114-132 and 122-128). Recorded as a literal because it is a
#: MEASUREMENT from this dataset, not a guess, and because the v0.17 capture
#: failed precisely for want of it.
SUB25_REFERENCE_Z = (114, 132)


def sub25_targets() -> list:
    """The sessions under adjudication, plus the two before for comparison.

    ses-01 and ses-02 carry non-empty CL masks (546 and 82 voxels) and are the
    reference for what this patient's enhancement looked like before it
    disappeared. ses-09 and ses-11 have NO CL mask at all and are included so the
    absent-vs-empty distinction is visible rather than inferred.
    """
    return [("sub-25", f"ses-{i:02d}") for i in range(1, 12)]


def overlay_png(project_root, slice_record: dict, subject: str = "sub-25",
                image: str = "T1c.nii.gz", mask: str = "ContrastEnhancedMask-CL.nii.gz",
                outdir: str = "06_QC_REPORTS") -> list:
    """One row per session: the image, and the image with its mask outlined.

    A mask rendered on its own tells a reader nothing about whether the
    underlying tissue enhances. The adjudication question is whether tissue is
    visible where the mask says nothing is, so image and mask must be seen
    together, at the same anatomy, across sessions.
    """
    import numpy as np
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return [{"status": "MATPLOTLIB_ABSENT"}]

    d = Path(project_root) / outdir / f"v2_{subject}_overlays"
    d.mkdir(parents=True, exist_ok=True)
    sessions = sorted({r["session"] for r in slice_record.values()
                       if r["subject"] == subject})
    written = []
    for ses in sessions:
        img = next((r for r in slice_record.values()
                    if r["subject"] == subject and r["session"] == ses
                    and r["name"].endswith(image)), None)
        if img is None:
            continue
        msk = next((r for r in slice_record.values()
                    if r["subject"] == subject and r["session"] == ses
                    and r["name"].endswith(mask)), None)
        arrs = [np.array(a) for a in img["slices"]]
        n = len(arrs)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8.4))
        axes = np.atleast_2d(axes)
        for col, (a, k) in enumerate(zip(arrs, img["slice_indices"])):
            lo, hi = np.percentile(a[a > 0], [1, 99]) if (a > 0).any() else (0, 1)
            for row in (0, 1):
                axes[row, col].imshow(np.rot90(a), cmap="gray", vmin=lo, vmax=hi)
                axes[row, col].axis("off")
            axes[0, col].set_title(f"z={k}", fontsize=9)
            if msk is not None and col < len(msk["slices"]):
                m = np.array(msk["slices"][col])
                if m.any():
                    axes[1, col].contour(np.rot90(m), levels=[0.5],
                                         colors="red", linewidths=1.2)
                else:
                    axes[1, col].set_title("mask EMPTY here", fontsize=8, color="red")
            elif msk is None:
                axes[1, col].set_title("NO MASK FILE", fontsize=8, color="orange")
        nz = "absent" if msk is None else msk["n_nonzero"]
        fig.suptitle(f"{subject} / {ses}   {image}   mask nonzero = {nz}\n"
                     f"top: image   bottom: image + {mask} outline", fontsize=10)
        p_out = d / f"{subject}__{ses}__overlay.png"
        fig.savefig(p_out, dpi=115, bbox_inches="tight")
        plt.close(fig)
        written.append(str(p_out))
    return written
