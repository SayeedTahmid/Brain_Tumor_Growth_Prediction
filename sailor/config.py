"""Centralised configuration (§12: no magic numbers outside this module).

PROJECT_NAME is read from the environment or an untracked local config and is
never hardcoded in tracked code (§18.2). Nothing here contains a personal Drive
path; `configs/local_paths.example.py` shows the shape of the untracked file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

DATA_VERSION = "v2.0"
FILE_PREFIX = "v2_"  # §4: never collide with a prior artefact

# ---------------------------------------------------------------- target lock
# §3.2 — LOCKED before Stage 1. Changing these after a result is seen invalidates
# the project. They are written into every manifest and every completion record.
PRIMARY_TARGET_MASK = "CL"
PRIMARY_TARGET_COMPONENT = "enhancing_t1wc"
SECONDARY_TARGET_MASK = "ONCO"
SENSITIVITY_TARGETS = ["CL:t2wflair_hyperintensity"]

# ------------------------------------------------------------------ provenance
# §4 — canonical EBRAINS artefacts (immutable input).
CANONICAL_FILES = [
    "data-descriptor_a866425efff8.pdf",
    "README.txt",
    "SHA512.txt",
    "overview.tsv",
    "missing.tsv",
    "src-to-raw.yaml",
    "code.tar.bz2",
    "rawdata_BIDS.tar.bz2",
    "derivatives.tar.bz2",
]

# EBRAINS artefacts the spec expects to be absent from the legacy folder; their
# absence is measured, not assumed (§4, G7).
CANONICAL_EXPECTED_ABSENT = [
    "sourcedata.tar.bz2",
    "rawdata.tar.bz2",
    "rawdata_BIDS_ext.tar.bz2",
]

QUARANTINE_FILES = [
    "tadiff_npy", "ckpt_dose", "ckpt_nodose", "ckpt_finetune", "_workdir",
    "unet_v1.pt", "unet_timecond_v1.pt", "unet_timecond_v2.pt",
    "autoencoder_v1.pt", "latents_v1.npz", "pairs_v1.npz",
    "sailor_slices_v1.h5", "sub-17_image.npy",
    "split_v1.json", "split_tadiff.json", "session_whitelist.json",
    "persistence_baseline.json",
]

# §4 — repackaged by the user (.tar not .tar.bz2); origin must be verified.
AMBIGUOUS_FILES = ["raw_needed.tar", "dosemaps.tar"]

# Prior splits that may never be reused as input (§4).
FORBIDDEN_INPUT_FILES = ["split_v1.json", "split_tadiff.json", "session_whitelist.json"]

# ------------------------------------------------------------------- tree §14.1
PROJECT_SUBDIRS = [
    "sailor", "tests", "notebooks/members",
    "00_CANONICAL", "00_QUARANTINE",
    "01_DATA_FOUNDATION", "02_PREPROCESSED_MRI", "03_TUMOR_MASKS",
    "04_LONGITUDINAL_WINDOWS", "05_TREATMENT_DATA", "06_QC_REPORTS",
    "07_BASELINE_RESULTS", "08_FEATURES", "09_MODEL_OUTPUTS", "10_EXPERIMENTS",
    "CHECKPOINTS", "LOGS", "RESULTS",
]

# ------------------------------------------------------------------- treatment
# §3 descriptor values. `unknown` is missing data, not a category (§5).
TREATMENT_VALUES = ["CRT", "TMZ", "no", "unknown"]
TREATMENT_MISSING_TOKEN = "unknown"

# Sequences named by the descriptor (§3). Presence per session is measured.
STRUCTURAL_SEQUENCES = ["t1w", "t1wc", "t2w", "t2wflair", "dti", "adc", "trace",
                        "dtiprea", "dtiprep", "t1wll"]
FUNCTIONAL_SEQUENCES = ["dce", "dsc", "dscprea", "dscprep"]

# Minimum modality set a session must have to enter the primary cohort (G9).
# Stated here so the survivor count is attributable to a written rule.
REQUIRED_SEQUENCES_PRIMARY = ["t1wc"]
REQUIRED_SEQUENCES_EXTENDED = ["t1wc", "t2wflair"]


@dataclass
class Paths:
    project_name: str
    dataset_root: Path
    legacy_root: Path
    code_root: Path

    def sub(self, name: str) -> Path:
        return self.dataset_root / name

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: str(v) for k, v in d.items()}


def _from_local_config(key: str):
    try:
        import configs.local_paths as lp  # type: ignore
        return getattr(lp, key, None)
    except Exception:
        return None


def get_paths(project_name: str | None = None,
              drive_root: str | None = None,
              legacy_root: str | None = None,
              code_root: str | None = None) -> Paths:
    """Resolve roots from explicit args > environment > untracked local config.

    Raises if PROJECT_NAME cannot be resolved: guessing it would write a real
    project into an arbitrary folder.
    """
    name = (project_name
            or os.environ.get("SAILOR_PROJECT_NAME")
            or _from_local_config("PROJECT_NAME"))
    if not name:
        raise RuntimeError(
            "PROJECT_NAME unresolved. Set SAILOR_PROJECT_NAME, or create "
            "configs/local_paths.py from configs/local_paths.example.py.")

    drive = (drive_root
             or os.environ.get("SAILOR_DRIVE_ROOT")
             or _from_local_config("DRIVE_ROOT")
             or "/content/drive/MyDrive")
    legacy = (legacy_root
              or os.environ.get("SAILOR_LEGACY_ROOT")
              or _from_local_config("LEGACY_ROOT")
              or f"{drive}/sailor_v1")
    code = (code_root
            or os.environ.get("SAILOR_CODE_ROOT")
            or _from_local_config("CODE_ROOT")
            or str(Path.cwd()))
    return Paths(project_name=name,
                 dataset_root=Path(drive) / name,
                 legacy_root=Path(legacy),
                 code_root=Path(code))


def target_lock() -> dict:
    return {
        "primary_target_mask": PRIMARY_TARGET_MASK,
        "primary_target_component": PRIMARY_TARGET_COMPONENT,
        "secondary_target_mask": SECONDARY_TARGET_MASK,
        "sensitivity_targets": list(SENSITIVITY_TARGETS),
        "data_version": DATA_VERSION,
    }
