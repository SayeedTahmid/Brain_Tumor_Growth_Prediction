"""Filename vocabulary discovery.

The descriptor names sequences and annotation sets; it does not tell us how they
appear on disk. §19.2 requires the audit to *resolve* how `CL` sub-masks are
named rather than assume a pattern. So this module derives the vocabulary from
observed member paths, then maps it onto the descriptor's names, and reports
anything it could not resolve as `UNRESOLVED` instead of guessing.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from ..config import FUNCTIONAL_SEQUENCES, STRUCTURAL_SEQUENCES

SUB_RE = re.compile(r"(sub-[A-Za-z0-9]+)")
SES_RE = re.compile(r"(ses-[A-Za-z0-9]+)")
ENTITY_RE = re.compile(r"([A-Za-z]+)-([A-Za-z0-9]+)")
SUFFIX_RE = re.compile(r"_([A-Za-z0-9]+)\.nii(\.gz)?$")

ALL_SEQUENCES = STRUCTURAL_SEQUENCES + FUNCTIONAL_SEQUENCES

# ---------------------------------------------------------------- real vocabulary
# The MNI derivatives do NOT use BIDS suffixes. They use ONCOHabitats pipeline
# names, measured from the archive itself:
#
#   T1.nii.gz  T1c.nii.gz  T2.nii.gz  Flair.nii.gz
#   T1c-icor.nii.gz            intensity-corrected variant
#   T1c-icor-zscore.nii.gz     z-scored variant
#   ContrastEnhancedMask-CL.nii.gz   ContrastEnhancedMask-ONCO.nii.gz
#   EdemaMask-CL.nii.gz              NecrosisMask-ONCO.nii.gz
#   Segmentation-ONCO.nii.gz         fastsurfer-segmentation.nii.gz
#   BrainExtractionMask.nii.gz  Mask.nii.gz  NAWMask.nii.gz  rCBV / rCBF
#
# Mapping is longest-first so `T1c` is never swallowed by `T1`.
PIPELINE_SEQUENCE_NAMES = {
    "t1c": "t1wc",
    "t1": "t1w",
    "t2": "t2w",
    "flair": "t2wflair",
    "adc": "adc",
    "trace": "trace",
    "rcbv": "rcbv",
    "rcbf": "rcbf",
}

# Intensity variant of an image. Which one enters the model is a preprocessing
# DECISION (§3.1(4)), not something this module may make silently, so the variant
# is recorded on every row and the choice is deferred to the phase gate.
VARIANT_RE = re.compile(r"-(icor-zscore|icor)(?=\.nii)", re.I)


def resolve_pipeline_sequence(basename: str) -> tuple[str | None, str]:
    """Map a pipeline-style basename to (descriptor sequence, variant).

    Returns (None, variant) when the stem is not an image sequence -- masks and
    segmentations resolve through `classify_annotation` instead.
    """
    stem = basename.split(".nii")[0]
    m = VARIANT_RE.search(basename)
    variant = m.group(1).lower() if m else "raw"
    core = re.sub(r"-(icor-zscore|icor)$", "", stem, flags=re.I)
    key = core.lower()
    for cand in sorted(PIPELINE_SEQUENCE_NAMES, key=len, reverse=True):
        if key == cand:
            return PIPELINE_SEQUENCE_NAMES[cand], variant
    return None, variant

# Tokens that indicate an annotation rather than an image. Matching is by token,
# and every unmatched mask-like file is reported.
CL_TOKENS = re.compile(r"(?:^|[_\-/.])(cl)(?:$|[_\-/.])", re.IGNORECASE)
# `Segmentation-ONCO` and `fastsurfer-segmentation` are multi-label maps, not
# binary component masks; they must not be mistaken for the primary target.
MULTILABEL_TOKENS = re.compile(r"(segmentation)", re.IGNORECASE)
ONCO_TOKENS = re.compile(r"onco", re.IGNORECASE)
ENH_TOKENS = re.compile(r"(contrastenhanced|enh|ce|t1wc|gd|contrast)", re.IGNORECASE)
FLAIR_TOKENS = re.compile(r"(flair|t2wflair|hyper)", re.IGNORECASE)
NECRO_TOKENS = re.compile(r"(necro)", re.IGNORECASE)
EDEMA_TOKENS = re.compile(r"(edema|oedema)", re.IGNORECASE)
BRAIN_TOKENS = re.compile(
    r"(brainextractionmask|brainmask|_brain|mask-brain|desc-brain)", re.IGNORECASE)
NAWM_TOKENS = re.compile(r"(nawm|normal[-_]?appearing)", re.IGNORECASE)
# Measured filenames: DoseMap.nii.gz and DoseMap_unscaled.nii.gz. A bare `gy`
# token matched far too much, so the pattern is anchored to real names.
DOSE_TOKENS = re.compile(r"(dosemap|rtdose|rt[-_]?plan|(?:^|[_\-/.])dose(?:$|[_\-/.]))",
                         re.IGNORECASE)
DOSE_UNSCALED = re.compile(r"unscaled", re.IGNORECASE)
MASKISH = re.compile(
    r"(?:^|[_\-/.])(mask|seg|label|roi|lesion|tumou?r)(?:$|[_\-/.])", re.IGNORECASE)


def parse_path(name: str) -> dict:
    """Extract subject, session, entities and suffix from one member path."""
    sub = SUB_RE.search(name)
    ses = SES_RE.search(name)
    base = name.rsplit("/", 1)[-1]
    entities = {k.lower(): v for k, v in ENTITY_RE.findall(base)}
    suffix = SUFFIX_RE.search(base)
    return {
        "path": name,
        "subject": sub.group(1) if sub else None,
        "session": ses.group(1) if ses else None,
        "entities": entities,
        "suffix": suffix.group(1) if suffix else None,
        "basename": base,
    }


def resolve_sequence(name: str, entities: dict, suffix: str | None) -> str | None:
    """Map a filename to one of the descriptor's sequence names, or None.

    Tries the measured pipeline vocabulary first, then BIDS-style tokens.
    Longest-token-first so `t2wflair` is not swallowed by `t2w`, `dscprea` not
    by `dsc`, and `T1c` not by `T1`.
    """
    base = name.rsplit("/", 1)[-1]
    seq, _variant = resolve_pipeline_sequence(base)
    if seq:
        return seq
    hay = name.lower()
    for candidate in sorted(ALL_SEQUENCES, key=len, reverse=True):
        if re.search(rf"(?:^|[_\-/.]){candidate}(?:$|[_\-/.])", hay):
            return candidate
    if suffix and suffix.lower() in ALL_SEQUENCES:
        return suffix.lower()
    for key in ("acq", "desc", "suffix", "mod"):
        v = entities.get(key, "").lower()
        if v in ALL_SEQUENCES:
            return v
    return None


def classify_annotation(name: str) -> dict:
    """Classify a mask-like path into annotation set and component."""
    is_cl = bool(CL_TOKENS.search(name))
    is_onco = bool(ONCO_TOKENS.search(name))
    dose = bool(DOSE_TOKENS.search(name))
    brain = bool(BRAIN_TOKENS.search(name))
    nawm = bool(NAWM_TOKENS.search(name))
    component = None
    if FLAIR_TOKENS.search(name):
        component = "t2wflair_hyperintensity"
    if ENH_TOKENS.search(name) and component is None:
        component = "enhancing_t1wc"
    if NECRO_TOKENS.search(name):
        component = "necrosis"
    if EDEMA_TOKENS.search(name):
        component = "edema"
    if MULTILABEL_TOKENS.search(name) and not FLAIR_TOKENS.search(name):
        kind = "multilabel_segmentation"
    elif dose:
        kind = "dose_map"
    elif nawm:
        kind = "nawm_mask"
    elif brain:
        kind = "brain_mask"
    elif is_cl:
        kind = "CL"
    elif is_onco:
        kind = "ONCO"
    elif MASKISH.search(name):
        kind = "UNRESOLVED_MASKLIKE"
    else:
        kind = "not_annotation"
    # A component only means something on an annotation. `Flair.nii.gz` is an
    # image, not a FLAIR-hyperintensity mask, and labelling it with a component
    # would let an image be mistaken for a target.
    if kind == "not_annotation":
        component = None
    if kind == "dose_map" and DOSE_UNSCALED.search(name):
        component = "unscaled"
    return {"kind": kind, "component": component,
            "cl_token": is_cl, "onco_token": is_onco}


def discover_vocabulary(member_names: list[str]) -> dict:
    """Summarise what naming actually exists, for human inspection."""
    entity_keys = Counter()
    entity_values = defaultdict(Counter)
    suffixes = Counter()
    subjects, sessions = set(), set()
    dirs = Counter()
    for n in member_names:
        p = parse_path(n)
        if p["subject"]:
            subjects.add(p["subject"])
        if p["session"]:
            sessions.add(p["session"])
        for k, v in p["entities"].items():
            entity_keys[k] += 1
            entity_values[k][v] += 1
        if p["suffix"]:
            suffixes[p["suffix"]] += 1
        parts = n.split("/")
        if len(parts) > 1:
            dirs[parts[0]] += 1
    return {
        "n_members": len(member_names),
        "n_subjects_in_paths": len(subjects),
        "n_sessions_in_paths": len(sessions),
        "top_level_dirs": dirs.most_common(20),
        "entity_keys": entity_keys.most_common(),
        "entity_values": {k: v.most_common(25) for k, v in entity_values.items()},
        "suffixes": suffixes.most_common(40),
    }


def build_file_table(member_names: list[str]) -> list[dict]:
    """One row per NIfTI member with resolved subject/session/sequence/annotation."""
    rows = []
    for n in member_names:
        if not n.lower().endswith((".nii", ".nii.gz")):
            continue
        p = parse_path(n)
        ann = classify_annotation(n)
        base = n.rsplit("/", 1)[-1]
        seq, variant = resolve_pipeline_sequence(base)
        if not seq:
            seq = resolve_sequence(n, p["entities"], p["suffix"])
        rows.append({
            "path": n, "subject": p["subject"], "session": p["session"],
            "sequence": seq, "intensity_variant": variant,
            "annotation_kind": ann["kind"],
            "annotation_component": ann["component"],
            "entities": p["entities"], "suffix": p["suffix"],
        })
    return rows


def target_resolution_report(file_rows: list[dict]) -> dict:
    """Does the locked primary target exist on disk, and under what name?"""
    cl = [r for r in file_rows if r["annotation_kind"] == "CL"]
    cl_enh = [r for r in cl if r["annotation_component"] == "enhancing_t1wc"]
    cl_flair = [r for r in cl if r["annotation_component"] == "t2wflair_hyperintensity"]
    cl_unknown = [r for r in cl if r["annotation_component"] is None]
    onco = [r for r in file_rows if r["annotation_kind"] == "ONCO"]
    unresolved = [r for r in file_rows if r["annotation_kind"] == "UNRESOLVED_MASKLIKE"]
    status = "RESOLVED" if cl_enh else (
        "UNRESOLVED_COMPONENT" if cl else "UNRESOLVED_NO_CL_FILES")
    return {
        "status": status,
        "n_cl_files": len(cl),
        "n_cl_enhancing_t1wc": len(cl_enh),
        "n_cl_t2wflair": len(cl_flair),
        "n_cl_component_unknown": len(cl_unknown),
        "n_onco_files": len(onco),
        "n_unresolved_masklike": len(unresolved),
        "cl_example_paths": [r["path"] for r in cl[:10]],
        "cl_enhancing_example_paths": [r["path"] for r in cl_enh[:10]],
        "cl_component_unknown_examples": [r["path"] for r in cl_unknown[:10]],
        "onco_example_paths": [r["path"] for r in onco[:10]],
        "unresolved_masklike_examples": [r["path"] for r in unresolved[:20]],
        "note": ("Component resolution is token-based and must be confirmed "
                 "against the descriptor before the primary cohort is frozen."),
    }
