"""Streaming access to `*.tar.bz2` / `*.tar` without full extraction.

§4.1 requires selective or streaming extraction rather than decompressing
~43 GB into a Colab disk. bz2 is not randomly seekable, so the unit of work is
one forward pass over the archive during which every interested handler sees the
members it wants. Passes are expensive and are therefore cached: the scan writes
a JSONL index under the project root and a later run reuses it unless
`force=True` (§15.4 idempotence).

Nothing here decides what a file means. Handlers collect measurements; naming
and semantics are resolved afterwards in `naming.py` from what was observed.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, BinaryIO


@dataclass
class MemberRecord:
    archive: str
    name: str
    size: int
    kind: str  # 'nifti' | 'text' | 'other'


class MemberHandler:
    """Sees each member once, in archive order, as a forward-only stream."""

    name = "handler"

    def match(self, name: str, size: int) -> bool:
        raise NotImplementedError

    def handle(self, archive: str, name: str, size: int, fh: BinaryIO) -> None:
        raise NotImplementedError

    def result(self) -> dict:
        return {}


TEXT_SUFFIXES = (".tsv", ".csv", ".txt", ".json", ".yaml", ".yml", ".bval", ".bvec")
NIFTI_SUFFIXES = (".nii", ".nii.gz")


def classify(name: str) -> str:
    low = name.lower()
    if low.endswith(NIFTI_SUFFIXES):
        return "nifti"
    if low.endswith(TEXT_SUFFIXES):
        return "text"
    return "other"


def _open_tar(path: Path) -> tarfile.TarFile:
    mode = "r|bz2" if str(path).endswith(".bz2") else "r|*"
    return tarfile.open(path, mode=mode)  # '|' = stream mode, no seeking


def _run(handler: "MemberHandler", archive: str, name: str, size: int,
         fh: BinaryIO) -> None:
    """Call one handler; a measurement failure is recorded, not raised."""
    try:
        handler.handle(archive, name, size, fh)
    except Exception as exc:
        errs = getattr(handler, "errors", None)
        if errs is not None:
            errs.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})


def scan_archive(path: Path, handlers: Iterable[MemberHandler],
                 index_path: Path | None = None,
                 force: bool = False,
                 max_members: int | None = None,
                 inmem_limit: int = 512 << 20,
                 progress: Callable[[int, float], None] | None = None) -> dict:
    """One forward pass. Returns measured pass statistics.

    If `index_path` exists and `force` is False the pass is skipped and the
    cached member list is returned with `from_cache: True`; handlers do not run,
    so callers that need handler output must pass force=True or check the flag.
    """
    path = Path(path)
    if not path.exists():
        return {"archive": path.name, "status": "ABSENT", "members": 0}

    if index_path and index_path.exists() and not force:
        members = [json.loads(l) for l in index_path.read_text().splitlines() if l]
        return {"archive": path.name, "status": "OK", "from_cache": True,
                "members": len(members), "index_path": str(index_path),
                "seconds": 0.0}

    handlers = list(handlers)
    for h in handlers:
        if hasattr(h, "begin_archive"):
            h.begin_archive(path.name)
    t0 = time.time()
    n = 0
    out = None
    if index_path:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        out = index_path.open("w")
    try:
        with _open_tar(path) as tf:
            for member in tf:
                if not member.isfile():
                    continue
                n += 1
                kind = classify(member.name)
                if out:
                    out.write(json.dumps({"archive": path.name,
                                          "name": member.name,
                                          "size": member.size,
                                          "kind": kind}) + "\n")
                wanted = [h for h in handlers if h.match(member.name, member.size)]
                if wanted:
                    # A stream-mode tar cannot seek backwards, so a member may be
                    # extracted only once. Members below `inmem_limit` are buffered
                    # so every interested handler sees the same bytes; larger ones
                    # go to the first handler only and the rest are recorded as
                    # skipped rather than silently dropped.
                    fh = tf.extractfile(member)
                    if fh is not None:
                        try:
                            if member.size <= inmem_limit:
                                data = fh.read()
                                for h in wanted:
                                    _run(h, path.name, member.name, member.size,
                                         io.BytesIO(data))
                            else:
                                _run(h := wanted[0], path.name, member.name,
                                     member.size, fh)
                                for h in wanted[1:]:
                                    errs = getattr(h, "errors", None)
                                    if errs is not None:
                                        errs.append({"name": member.name,
                                                     "error": "SKIPPED_MEMBER_TOO_LARGE_TO_BUFFER"})
                        finally:
                            fh.close()
                if progress and n % 500 == 0:
                    progress(n, time.time() - t0)
                if max_members and n >= max_members:
                    break
    finally:
        if out:
            out.close()
    return {"archive": path.name, "status": "OK", "from_cache": False,
            "members": n, "seconds": round(time.time() - t0, 2),
            "index_path": str(index_path) if index_path else None}


def load_index(index_path: Path) -> list[MemberRecord]:
    recs = []
    if not Path(index_path).exists():
        return recs
    for line in Path(index_path).read_text().splitlines():
        if not line:
            continue
        d = json.loads(line)
        recs.append(MemberRecord(**d))
    return recs


# ------------------------------------------------------------------- handlers

class TextHandler(MemberHandler):
    """Collects small text members verbatim (tsv/json/yaml/txt)."""

    name = "text"

    def __init__(self, max_bytes: int = 4 << 20, patterns: list[str] | None = None):
        self.max_bytes = max_bytes
        self.patterns = [re.compile(p) for p in (patterns or [r".*"])]
        self.texts: dict[str, str] = {}
        self.errors: list[dict] = []

    def match(self, name: str, size: int) -> bool:
        if classify(name) != "text" or size > self.max_bytes:
            return False
        return any(p.search(name) for p in self.patterns)

    def handle(self, archive, name, size, fh):
        self.texts[f"{archive}::{name}"] = fh.read().decode("utf-8", "replace")

    def result(self):
        return {"n_texts": len(self.texts), "errors": self.errors}


class HeaderHandler(MemberHandler):
    """Reads NIfTI headers only: shape, spacing, dtype, scaling."""

    name = "nifti_header"

    def __init__(self):
        self.headers: dict[str, dict] = {}
        self.errors: list[dict] = []

    def match(self, name: str, size: int) -> bool:
        return classify(name) == "nifti"

    def handle(self, archive, name, size, fh):
        from .nifti_header import read_header
        hdr = read_header(fh, name)
        aff = hdr.affine
        self.headers[f"{archive}::{name}"] = {
            "archive": archive, "name": name, "size_bytes": size,
            "shape": list(hdr.shape), "spacing": list(hdr.spacing),
            "dtype": hdr.dtype, "datatype_code": hdr.datatype_code,
            "scl_slope": hdr.scl_slope, "scl_inter": hdr.scl_inter,
            "qform_code": hdr.qform_code, "sform_code": hdr.sform_code,
            "nifti_version": hdr.version, "descrip": hdr.descrip,
            # v0.20 — without these there is no record of WHERE a volume sits,
            # so two grids of different shape cannot be related at all.
            "spatial_status": hdr.spatial_status,
            "affine": None if aff is None else [[round(float(x), 6) for x in row]
                                                for row in aff],
            "world_origin": hdr.world_origin,
            "qoffset": [hdr.qoffset_x, hdr.qoffset_y, hdr.qoffset_z],
            "quatern": [hdr.quatern_b, hdr.quatern_c, hdr.quatern_d],
            "qfac": hdr.qfac,
        }

    def result(self):
        return {"n_headers": len(self.headers), "errors": self.errors}


# Generous candidate matcher: naming is discovered, not assumed (§2.2). Anything
# that could be a label map is read in full during the same pass, because a
# second pass over a bz2 archive costs another full decompression.
# Token-bounded: an unbounded `wm` matched any path containing those two
# letters, so every volume looked like a mask candidate and the image sample
# budget G10 depends on was never spent.
#
# v0.17 — `dose` required a separator AFTER the token, so `DoseMap.nii.gz`
# (token followed by `M`) never matched and NO dose array was ever read: 0 of 26,
# while `naming.DOSE_TOKENS` classified them correctly. Two regexes, one right.
# The volume-reading side is what GATE-1 depends on, so the dose-geometry audit
# was blocked by a boundary assertion. Dose now matches as a prefix.
MASKISH = re.compile(
    r"(?:^|[_\-/.])(?:"
    r"(?:mask|seg|label|roi|lesion|tumou?r|necro|edema|oedema|enh|"
    r"CL|ONCO|habitat|brain|nawm|wm)(?:$|[_\-/.])"
    r"|(?:dosemap|rtdose|dose)"          # prefix match: DoseMap, DoseMap_unscaled
    r")", re.IGNORECASE)

# Volumes whose values are NOT intensities and must never be range-checked
# against the descriptor's 0-255 claim: FreeSurfer label IDs reach 2035, a
# z-scored variant is negative by construction, and rCBF/rCBV are quantitative
# perfusion values in their own physiological units (one rCBF volume measured
# 5550). v0.19 (defect 23) adds the perfusion maps — the same category error as
# the label maps, missed because only one slipped past the earlier fix.
NON_INTENSITY = re.compile(
    r"(fastsurfer|segmentation|-zscore|(?:^|[_\-/.])r?(?:CBF|CBV)(?:$|[_\-/.]))",
    re.IGNORECASE)

# The locked intensity variant (§ intensity-variant decision). Read in full so
# normalisation is chosen from the cohort, not from whichever subject happens to
# sit first in the archive.
ICOR_RE = re.compile(r"-icor(?!-zscore)\.nii", re.IGNORECASE)

# The PLHM check compares within-patient landmark spread between the plain and
# `-icor` variants, so it needs BOTH for the SAME sessions. Reading every plain
# structural volume cohort-wide would double the per-volume numpy work for a
# test that saturates after a handful of patients, so plain variants are read in
# full for a bounded number of subjects only (`plhm_subjects`).
PLAIN_STRUCTURAL_RE = re.compile(r"/(T1|T1c|T2|Flair)\.nii", re.IGNORECASE)


class VolumeStatsHandler(MemberHandler):
    """Full-array statistics for mask-like members, and for sampled images.

    Mask candidates are matched generously and read in full (they are small).
    Non-candidate images are sampled up to `sample_images` so G10 has measured
    dtype and value ranges without decompressing every volume.
    """

    name = "volume_stats"

    #: percentiles recorded for every volume read. The PLHM check compares the
    #: within-patient spread of these landmarks between the plain and `-icor`
    #: variants; min/max/mean alone cannot answer it.
    PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)

    def __init__(self, sample_images: int = 40, max_bytes: int = 200 << 20,
                 max_labels: int = 32, sample_per_subject: int | None = None,
                 read_icor: bool = True, plhm_subjects: int = 8):
        self.sample_images = sample_images
        self.max_bytes = max_bytes
        self.max_labels = max_labels
        # v0.17 — the old budget was "first N members in archive order", which
        # spent all 60 samples inside sub-13: every intensity statistic in the
        # project, and G10's whole verdict, rested on ONE subject. A per-subject
        # quota is the most a forward-only stream can do without a second pass.
        self.sample_per_subject = sample_per_subject
        self.read_icor = read_icor
        self.plhm_subjects = plhm_subjects
        self.stats: dict[str, dict] = {}
        self.errors: list[dict] = []
        self._sampled = 0
        self._per_subject: dict[str, int] = {}
        #: Subjects whose PLAIN structural volumes are read in full so the PLHM
        #: check has paired data. Filled in archive order — the first N subjects
        #: encountered, which is NOT a random sample and is reported as such.
        self._plhm_subs: set = set()

    def begin_archive(self, archive: str) -> None:
        # The image sample budget is per archive: the descriptor's intensity
        # claim concerns the MNI derivatives, so a budget spent inside the raw
        # archive would leave G10 with nothing to measure where it matters.
        self._sampled = 0
        self._per_subject = {}
        self._plhm_subs = set()

    @staticmethod
    def _subject(name: str) -> str:
        m = re.search(r"(sub-[A-Za-z0-9]+)", name)
        return m.group(1) if m else "_nosubject"

    def _role(self, name: str) -> str | None:
        if MASKISH.search(name):
            return "mask_candidate"
        sub = self._subject(name)
        if self.read_icor and ICOR_RE.search(name):
            # An icor volume admits its subject to the paired set, so the plain
            # counterpart of the SAME session is read rather than missed.
            if len(self._plhm_subs) < self.plhm_subjects:
                self._plhm_subs.add(sub)
            return "icor_full"
        if PLAIN_STRUCTURAL_RE.search(name) and sub in self._plhm_subs:
            return "plain_paired"
        if self.sample_per_subject is not None:
            if self._per_subject.get(sub, 0) < self.sample_per_subject:
                return "image_sample"
            return None
        if self._sampled < self.sample_images:
            return "image_sample"
        return None

    def match(self, name: str, size: int) -> bool:
        if classify(name) != "nifti" or size > self.max_bytes:
            return False
        return self._role(name) is not None

    def handle(self, archive, name, size, fh):
        import numpy as np
        from .nifti_header import read_array, scaled
        role = self._role(name) or "image_sample"
        if role == "image_sample":
            self._sampled += 1
            sub = self._subject(name)
            self._per_subject[sub] = self._per_subject.get(sub, 0) + 1
        hdr, arr = read_array(fh, name)
        vals = scaled(arr, hdr)
        finite = np.isfinite(vals)
        n_nonfinite = int(vals.size - finite.sum())
        v = vals[finite]
        uniq = np.unique(v)
        rec = {
            "archive": archive, "name": name, "key": f"{archive}::{name}", "role": role,
            "shape": list(hdr.shape), "spacing": list(hdr.spacing),
            "dtype": hdr.dtype, "n_voxels": int(hdr.n_voxels),
            "min": float(v.min()) if v.size else None,
            "max": float(v.max()) if v.size else None,
            "mean": float(v.mean()) if v.size else None,
            "n_nonfinite": n_nonfinite,
            "n_unique": int(uniq.size),
            "n_nonzero": int((v != 0).sum()),
            "is_intensity": not bool(NON_INTENSITY.search(name)),
            "has_fractional_values": bool(v.size and not np.all(np.equal(np.mod(v, 1), 0))),
        }
        # Landmarks are computed over NON-ZERO finite voxels: background is a
        # large constant plateau that would drag every low percentile to 0 and
        # hide exactly the shifts PLHM would introduce.
        nz = v[v != 0] if v.size else v
        if nz.size:
            pct = np.percentile(nz, self.PERCENTILES)
            rec["percentiles_nonzero"] = {str(p): float(x)
                                          for p, x in zip(self.PERCENTILES, pct)}
            rec["n_finite_nonzero"] = int(nz.size)
        else:
            rec["percentiles_nonzero"] = None
            rec["n_finite_nonzero"] = 0
        if uniq.size <= self.max_labels:
            rec["labels"] = [float(x) for x in uniq]
            rec["label_counts"] = {str(float(x)): int((v == x).sum()) for x in uniq}
            rec["binary"] = bool(set(rec["labels"]) <= {0.0, 1.0})
        else:
            rec["labels"] = None
            rec["label_counts"] = None
            rec["binary"] = False
        self.stats[f"{archive}::{name}"] = rec

    def result(self):
        return {"n_volumes": len(self.stats), "errors": self.errors}
