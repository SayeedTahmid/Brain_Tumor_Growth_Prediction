"""Synthetic EBRAINS-shaped fixture (§16: smoke test under a minute).

This is NOT SAILOR data and no measurement taken against it is a measurement of
SAILOR. Its only job is to exercise every code path — archive streaming, NIfTI
header parsing, naming resolution, guard logic, report rendering — before the
audit is pointed at 43 GB of real archives.

The fixture deliberately plants defects the guards must catch:
  * one all-zero primary mask                      -> G1 FAIL
  * one session absent from raw-mni-link.tsv       -> G8 FAIL
  * one session listed in missing.tsv              -> G9 exclusion
  * float-valued volumes outside the uint8 claim   -> G10 evidence
  * `unknown` treatment tokens                     -> §5 missingness handling
"""

from __future__ import annotations

import gzip
import io
import struct
import tarfile
from pathlib import Path

import numpy as np

SHAPE = (12, 14, 10)
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 5)]
SESSIONS = ["ses-01", "ses-02", "ses-03"]
SEQUENCES = ["t1w", "t1wc", "t2w", "t2wflair"]


def nifti1_bytes(arr: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> bytes:
    """Minimal valid NIfTI-1 single-file image."""
    dtype_map = {np.dtype("uint8"): (2, 8), np.dtype("int16"): (4, 16),
                 np.dtype("float32"): (16, 32)}
    code, bitpix = dtype_map[arr.dtype]
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)
    dim = [3, *arr.shape, 1, 1, 1, 1][:8]
    struct.pack_into("<8h", hdr, 40, *dim)
    struct.pack_into("<h", hdr, 70, code)
    struct.pack_into("<h", hdr, 72, bitpix)
    struct.pack_into("<8f", hdr, 76, 1.0, *spacing, 0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<f", hdr, 108, 352.0)   # vox_offset
    struct.pack_into("<f", hdr, 112, 1.0)     # scl_slope
    struct.pack_into("<f", hdr, 116, 0.0)     # scl_inter
    struct.pack_into("<h", hdr, 252, 1)       # qform_code
    struct.pack_into("<h", hdr, 254, 1)       # sform_code
    hdr[344:348] = b"n+1\x00"
    return bytes(hdr) + b"\x00" * 4 + arr.astype(arr.dtype).tobytes(order="F")


def gz(data: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(data)
    return buf.getvalue()


def _add(tf: tarfile.TarFile, name: str, payload: bytes):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tf.addfile(info, io.BytesIO(payload))


def _mask(nonzero: int) -> np.ndarray:
    m = np.zeros(SHAPE, dtype=np.uint8)
    if nonzero:
        flat = m.reshape(-1)
        flat[:nonzero] = 1
    return m


def build(root: Path) -> Path:
    """Create a fake legacy folder at `root` and return it."""
    rng = np.random.default_rng(0)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # --- derivatives (MNI): images + CL / ONCO / brain / dose ------------------
    deriv = root / "derivatives.tar.bz2"
    with tarfile.open(deriv, "w:bz2") as tf:
        for si, sub in enumerate(SUBJECTS):
            for sj, ses in enumerate(SESSIONS):
                sdir = f"derivatives/mni2009c-n-s/{sub}/{ses}"
                base = f"{sdir}/anat/{sub}_{ses}"
                # Real ONCOHabitats pipeline names, measured from the archive.
                for pipe, seq in [("T1", "t1w"), ("T1c", "t1wc"),
                                  ("T2", "t2w"), ("Flair", "t2wflair")]:
                    arr = rng.integers(0, 256, SHAPE).astype(np.uint8)
                    if si == 0 and seq == "t1wc":       # plants G10 evidence
                        arr = (arr.astype(np.float32) / 3.0).astype(np.float32)
                    _add(tf, f"{sdir}/{pipe}.nii.gz", gz(nifti1_bytes(arr)))
                    _add(tf, f"{sdir}/{pipe}-icor.nii.gz", gz(nifti1_bytes(arr)))
                # locked primary target; sub-02/ses-02 is degenerate on purpose
                n = 0 if (sub == "sub-02" and ses == "ses-02") else 40 + 7 * sj
                _add(tf, f"{sdir}/ContrastEnhancedMask-CL.nii.gz",
                     gz(nifti1_bytes(_mask(n))))
                _add(tf, f"{sdir}/EdemaMask-CL.nii.gz",
                     gz(nifti1_bytes(_mask(n + 25))))
                onco = _mask(n + 10).astype(np.uint8)
                onco[0, 0, 0] = 2
                _add(tf, f"{sdir}/NecrosisMask-ONCO.nii.gz",
                     gz(nifti1_bytes(onco)))
                _add(tf, f"{sdir}/ContrastEnhancedMask-ONCO.nii.gz",
                     gz(nifti1_bytes(_mask(n + 5))))
                _add(tf, f"{sdir}/Segmentation-ONCO.nii.gz",
                     gz(nifti1_bytes(onco)))
                _add(tf, f"{sdir}/BrainExtractionMask.nii.gz",
                     gz(nifti1_bytes(_mask(SHAPE[0] * SHAPE[1] * SHAPE[2] // 2))))
            if si < 3:  # one patient deliberately has no dose map
                dose = (rng.random(SHAPE) * 60).astype(np.float32)
                name = "DoseMap_unscaled" if si == 2 else "DoseMap"
                _add(tf, f"derivatives/mni2009c-n-s/{sub}/{name}.nii.gz",
                     gz(nifti1_bytes(dose)))
        _add(tf, "derivatives/raw-mni-link.tsv", RAW_MNI_LINK.encode())
        # Per-session clinical files, in the real EBRAINS layout.
        # NB: not `root` -- that is the function's Path argument.
        droot = "derivatives/mni2009c-n-s"
        _add(tf, f"{droot}/history.txt",
             b"Description:\n\nLongitudinal data of treatment of high-grade glioma.\n")
        _add(tf, f"{droot}/structure.txt",
             ("---\n" + "\n".join(f"{s}\n" + "  ".join(SESSIONS) for s in SUBJECTS)
              ).encode())
        _add(tf, f"{droot}/overview.tsv",
             ("subject\tsession\n" + "\n".join(f"{s}\t{e}" for s in SUBJECTS
                                                for e in SESSIONS)).encode())
        for si, sub in enumerate(SUBJECTS):
            _add(tf, f"{droot}/{sub}/overall-survival-months.txt",
                 f"{12.5 + si:.5f}".encode())
            # Intervals BETWEEN consecutive exams: n_sessions - 1 values.
            _add(tf, f"{droot}/{sub}/age-years.txt", f"{45 + si}".encode())
            # Hyphen, not underscore: this is what the archive actually contains.
            _add(tf, f"{droot}/{sub}/intervals-days.txt",
                 "\n".join(str(28 + 7 * j) for j in range(len(SESSIONS) - 1)).encode())
            for sj, ses in enumerate(SESSIONS):
                # Deliberately schedule-locked, like the real cohort.
                status = "CRT" if sj == 0 else ("no" if sj == 1 else "TMZ")
                if si == 3 and sj == 2:
                    status = "unknown"
                _add(tf, f"{droot}/{sub}/{ses}/treatment.txt", status.encode())
                _add(tf, f"{droot}/{sub}/{ses}/RANO.txt", str(1 + sj).encode())

    # --- rawdata_BIDS: scans.tsv carrying acq_time (the exact Δt source) -------
    bids = root / "rawdata_BIDS.tar.bz2"
    with tarfile.open(bids, "w:bz2") as tf:
        for sub in SUBJECTS:
            rows = ["filename\tacq_time"]
            for j, ses in enumerate(SESSIONS):
                for seq in SEQUENCES:
                    _add(tf, f"rawdata_BIDS/{sub}/{ses}/anat/{sub}_{ses}_{seq}.nii.gz",
                         gz(nifti1_bytes(rng.integers(0, 4096, SHAPE).astype(np.int16))))
                rows.append(f"{ses}/anat/{sub}_{ses}_t1wc.nii.gz\t"
                            f"2019-0{1 + j}-1{j}T09:00:00")
            _add(tf, f"rawdata_BIDS/{sub}/{sub}_scans.tsv", "\n".join(rows).encode())
        _add(tf, "rawdata_BIDS/dataset_description.json",
             b'{"Name": "SAILOR fixture", "BIDSVersion": "1.8.0"}')

    # --- loose canonical files -------------------------------------------------
    (root / "overview.tsv").write_text(OVERVIEW)
    (root / "missing.tsv").write_text(MISSING)
    (root / "README.txt").write_text("synthetic fixture — not EBRAINS data\n")
    (root / "src-to-raw.yaml").write_text("fixture: true\n")
    (root / "data-descriptor_a866425efff8.pdf").write_bytes(b"%PDF-1.4 fixture\n")
    (root / "code.tar.bz2").write_bytes(_tiny_tar_bz2())

    # --- quarantine + ambiguous artefacts (§4) --------------------------------
    (root / "split_v1.json").write_text('{"train": [], "test": []}')
    (root / "unet_v1.pt").write_bytes(b"\x00" * 64)
    (root / "tadiff_npy").mkdir(exist_ok=True)
    (root / "dosemaps.tar").write_bytes(_tiny_tar())

    # --- SHA512.txt over what exists ------------------------------------------
    import hashlib
    lines = []
    for name in ["overview.tsv", "missing.tsv", "README.txt", "src-to-raw.yaml",
                 "data-descriptor_a866425efff8.pdf", "code.tar.bz2",
                 "rawdata_BIDS.tar.bz2", "derivatives.tar.bz2"]:
        h = hashlib.sha512((root / name).read_bytes()).hexdigest()
        lines.append(f"{h}  {name}")
    (root / "SHA512.txt").write_text("\n".join(lines) + "\n")
    return root


def _tiny_tar() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        _add(tf, "placeholder.txt", b"fixture\n")
    return buf.getvalue()


def _tiny_tar_bz2() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tf:
        _add(tf, "code/placeholder.py", b"# fixture\n")
    return buf.getvalue()


OVERVIEW = "\n".join(
    ["subject\tsession\ttreatment\tdays_from_first\trano\tage\tsex\toverall_survival"]
    + [f"{sub}\t{ses}\t{tr}\t{d}\tSD\t56\tM\t19"
       for sub in SUBJECTS
       for ses, tr, d in [("ses-01", "no", 0), ("ses-02", "CRT", 31),
                          ("ses-03", "unknown", 74)]]
) + "\n"

# Wide y/n matrix, matching the real EBRAINS layout: one column per sequence.
_MISSING_SEQS = ["t1w", "t1wc", "t2w", "t2wflair", "dce", "dsc", "dscprea",
                 "dscprep", "dti", "dtiprea", "dtiprep", "t1wll", "trace", "adc"]


def _missing_rows():
    """RAW session numbering, like the real file.

    Raw ses-01/02 have no MNI counterpart; raw ses-03..05 map to MNI ses-01..03.
    The planted t1wc exclusion sits at raw ses-05, i.e. MNI ses-03, so a correct
    implementation must translate before it can honour the exclusion.
    """
    raw_sessions = [f"ses-{i:02d}" for i in range(1, len(SESSIONS) + 3)]
    lines = ["\t".join(["subject", "session"] + _MISSING_SEQS)]
    for sub in SUBJECTS:
        for ses in raw_sessions:
            flags = []
            for seq in _MISSING_SEQS:
                if seq in SEQUENCES:
                    # raw ses-05 == MNI ses-03 for sub-03
                    missing = (sub == "sub-03" and ses == "ses-05" and seq == "t1wc")
                else:
                    missing = True  # functional/diffusion absent in the fixture
                flags.append("y" if missing else "n")
            lines.append("\t".join([sub, ses] + flags))
    return "\n".join(lines) + "\n"


MISSING = _missing_rows()

# Mirrors the real file: two earliest raw exams per patient have NO MNI
# counterpart (literal "no"), and MNI ses-01 therefore sits at raw ses-03.
# sub-04/ses-03 is additionally omitted so G8 reports an unmatched MNI session.
def _raw_mni_link():
    lines = ["subject\traw session\tmni session"]
    for sub in SUBJECTS:
        lines.append(f"{sub}\tses-01\tno")
        lines.append(f"{sub}\tses-02\tno")
        for j, ses in enumerate(SESSIONS):
            if sub == "sub-04" and ses == "ses-03":
                continue
            lines.append(f"{sub}\tses-{j + 3:02d}\t{ses}")
    return "\n".join(lines) + "\n"


RAW_MNI_LINK = _raw_mni_link()


if __name__ == "__main__":
    import sys
    out = build(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sailor_fixture/sailor_v1"))
    print(out)
