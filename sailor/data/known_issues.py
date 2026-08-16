"""Documented data issues, transcribed from `derivatives/mni2009c-n-s/history.txt`.

These are not measurements. They are the dataset authors' own record of what they
estimated, imputed, reconstructed or excluded. Transcribing them into code makes
them enforceable: a session flagged here cannot quietly enter the primary cohort
because nobody re-read an 8 KB text file.

Every entry carries the verbatim justification so a reader can check the
transcription against the source rather than trusting this module.

Severity:
  EXCLUDE    must not enter the primary cohort
  LEAKAGE    excluding is mandatory for any prediction spanning this timepoint;
             the artefact contains information derived from a later session
  DEGRADED   usable but flagged; the limitation must appear in the write-up
  ESTIMATED  the value exists but was interpolated, not observed
"""

from __future__ import annotations

from collections import defaultdict

EXCLUDE, LEAKAGE, DEGRADED, ESTIMATED = "EXCLUDE", "LEAKAGE", "DEGRADED", "ESTIMATED"

SOURCE = "derivatives/mni2009c-n-s/history.txt"


# ---------------------------------------------------------------- session-level

SESSION_ISSUES = [
    # --- future-information leakage -------------------------------------------
    {"subject": "sub-04", "session": "ses-01", "severity": LEAKAGE,
     "issue": "contrast-enhanced mask built by dilating the NEXT timepoint's segmentation",
     "quote": "Created contrast enhanced mask using binary dilation from "
              "coregistered next time point segmentation.",
     "implication": "The ses-01 target is derived from ses-02. Any pair "
                    "(ses-01 -> ses-02) is trained on part of its own answer."},
    {"subject": "sub-05", "session": "ses-01", "severity": LEAKAGE,
     "issue": "masks re-used from the next timepoint as approximations",
     "quote": "re-used masks from next time point as approximations.",
     "implication": "Same as sub-04/ses-01: the ses-01 label IS the ses-02 label."},
    {"subject": "sub-04", "session": "ses-01", "severity": LEAKAGE,
     "issue": "intensities histogram-matched to ses-02",
     "quote": "histograms matched with corresponding ses-02 mri to ensure "
              "comparable intensity values",
     "implication": "Image intensities also carry ses-02 information, so the "
                    "leakage is not confined to the mask."},
    {"subject": "sub-05", "session": "ses-01", "severity": LEAKAGE,
     "issue": "intensities histogram-matched to ses-02",
     "quote": "histograms matched with corresponding ses-02 mri",
     "implication": "See sub-04/ses-01."},
    {"subject": "sub-21", "session": "ses-01", "severity": LEAKAGE,
     "issue": "intensities histogram-matched to ses-02",
     "quote": "histograms matched with corresponding ses-02 mri",
     "implication": "See sub-04/ses-01."},

    # --- acquisition / registration failures ----------------------------------
    {"subject": "sub-05", "session": "ses-05", "severity": EXCLUDE,
     "issue": "T1c missing coronal slices; coregistration elongated laterally",
     "quote": "has an error in t1c, missing coronal slices in the middle of the "
              "brain. This leads to not good co-registration",
     "implication": "Geometry is wrong, so volume and Dice against this session "
                    "are not comparable with the rest of the cohort."},
    {"subject": "sub-21", "session": "ses-12", "severity": DEGRADED,
     "issue": "incorrectly coregistered T2",
     "quote": "has incorrectly coregistered T2",
     "implication": "Excludable only for rungs that use T2; T1c-only targets "
                    "may survive, which must be decided and recorded."},
    {"subject": "sub-22", "session": "ses-05", "severity": DEGRADED,
     "issue": "large parts of FLAIR missing",
     "quote": "has missing large parts of Flair",
     "implication": "Blocks the CL:t2wflair sensitivity target for this session."},

    # --- pipeline coverage ----------------------------------------------------
    {"subject": "sub-21", "session": "ses-01", "severity": DEGRADED,
     "issue": "not processed by ONCOHabitats; not denoised or bias-corrected",
     "quote": "are not processed by ONCOHab. and not denoised and field "
              "inhomogeneity corrected.",
     "implication": "Preprocessing differs from every other session (§3.1(4))."},
    {"subject": "sub-04", "session": "ses-01", "severity": DEGRADED,
     "issue": "not processed by ONCOHabitats; not denoised or bias-corrected",
     "quote": "are not processed by ONCOHab.",
     "implication": "See sub-21/ses-01."},
    {"subject": "sub-05", "session": "ses-01", "severity": DEGRADED,
     "issue": "not processed by ONCOHabitats; not denoised or bias-corrected",
     "quote": "are not processed by ONCOHab.",
     "implication": "See sub-21/ses-01."},
    {"subject": "sub-08", "session": "ses-04", "severity": DEGRADED,
     "issue": "manual edema mask missing",
     "quote": "sub-08/ses-04 is missing manual edema mask.",
     "implication": "Affects edema components only, not the locked primary target."},
    {"subject": "sub-04", "session": "ses-01", "severity": DEGRADED,
     "issue": "NAWM mask and CBV missing",
     "quote": "sub-04/ses-01 and sub-05/ses-01 is missing NAWMask.nii and CBV",
     "implication": "Blocks perfusion normalisation for this session."},
    {"subject": "sub-05", "session": "ses-01", "severity": DEGRADED,
     "issue": "NAWM mask and CBV missing",
     "quote": "sub-04/ses-01 and sub-05/ses-01 is missing NAWMask.nii and CBV",
     "implication": "See sub-04/ses-01."},
]


# ---------------------------------------------------------------- subject-level

SUBJECT_ISSUES = [
    {"subject": "sub-24", "severity": EXCLUDE,
     "scope": "CL manual masks",
     "issue": "manual segmentation labels were never added",
     "quote": "manual masks for sub-24 not added because of preprocessing "
              "trouble by Atle.",
     "implication": "The locked primary target does not exist for this patient. "
                    "sub-24 cannot contribute to any CL-target rung, so the "
                    "primary cohort is 26 patients, not 27."},
    {"subject": "sub-19", "severity": EXCLUDE,
     "scope": "dose map",
     "issue": "no dose map",
     "quote": "sub-19 is missing dose map",
     "implication": "Excluded from C3 and C4; does not affect C0-C2."},
    {"subject": "sub-26", "severity": EXCLUDE,
     "scope": "dose map",
     "issue": "dose map is not scaled",
     "quote": "sub-26 has not scaled dose map (TODO calculate)",
     "implication": "Values are not in the same units as the other 25 maps. "
                    "Including it would mix scales inside one covariate; it is "
                    "excluded from C3/C4 unless the scaling is recovered."},
    {"subject": "sub-26", "severity": DEGRADED,
     "scope": "RANO",
     "issue": "RANO missing for examinations 12, 13, 14",
     "quote": "sub-26 is missing RANO for latest examinations 12, 13 and 14 "
              "(contrast-enhanced recurrence on 13 and 14)",
     "implication": "RANO is not fully observed; relevant only if RANO is used."},
]


# ---- Δt estimation, per the interval-extraction log in history.txt ------------
# These subjects' inter-exam intervals are partly INTERPOLATED, not measured.
DELTA_T_ESTIMATED = {
    "sub-13": "estimated 3 intervals by dividing a large interval at the end by 4",
    "sub-14": "estimated 2 intervals by dividing a large interval at the end by 3",
    "sub-15": "estimated 3 intervals by dividing 3 large intervals by 2",
    "sub-20": "estimated 3 intervals by dividing a large interval at the end by 4",
    "sub-21": "estimated 1 interval by dividing a large interval by 2, estimated "
              "2 intervals by dividing a large interval by 3",
    "sub-23": "estimated 2 intervals by dividing a large interval by 3, estimated "
              "1 interval by dividing a large interval by 2",
    "sub-24": "estimated 3 intervals by dividing a large interval at the end by 4",
    "sub-26": "estimated 4 intervals by dividing a large interval by 3, estimated "
              "1 interval by dividing a large interval by 2",
}

DELTA_T_DATE_RECOVERY = {
    "sub-13": "two extra data points; dates estimated from a spreadsheet",
    "sub-14": "dates fixed from a spreadsheet",
    "sub-15": "two timepoints estimated",
    "sub-21": "one missing date found in the DICOM StudyDate header",
    "sub-23": "two dates estimated based on pattern",
    "sub-24": "two dates found from StudyDate, one estimated based on pattern",
    "sub-26": "three dates estimated based on pattern",
    "sub-22": "multiple dates removed",
}

DELTA_T_GLOBAL_CAVEAT = (
    "Time intervals in days between examinations (intervals_days.txt) manually "
    "extracted from DICOM header information and/or Excel sheet... Some day "
    "estimations and modifications of the processed dataset were made, and the "
    "numbers may be inaccurate.")


# ---- documented intensity expectation, for G10 -------------------------------
INTENSITY_CLAIM = {
    "quote": "intensity values are generally scaled to between 0-255 (uint8 "
             "format), but still contain decimal values in most cases",
    "reading": ("The descriptor's own text expects a 0-255 RANGE carried in a "
                "FLOATING-POINT dtype. A measured float64 array with values in "
                "0-255 is therefore consistent with the documentation, not a "
                "contradiction of it; a measured range outside 0-255 would be."),
}

# ---- what the annotation sets are --------------------------------------------
ANNOTATION_PROVENANCE = {
    "CL": {"meaning": "manual contrast-enhancement and edema segmentation labels",
           "quote": "Added contrast enhancement and edema manual segmentation "
                    "labels (by Christopher Larsson and provided/preprocessed by Atle).",
           "note": "Manual, and the locked primary target. Absent for sub-24."},
    "ONCO": {"meaning": "automated ONCOHabitats pipeline segmentation",
             "quote": "Coregistered by Elies Fuster-Garcia (ONCOHabitats ...)",
             "note": "Automated. Secondary/inventory only under the §3.2 lock."},
}

# Manual labels came from a different session numbering; the authors intersected
# them with the MNI sessions and dropped the surplus. Recorded because it means
# CL coverage is not simply 'every MNI session'.
MANUAL_LABEL_INTERSECTION = {
    "sub-02": {"mni": 15, "atle": 16, "excluded_exams": [4]},
    "sub-03": {"mni": 6, "atle": 7, "excluded_exams": [3]},
    "sub-06": {"mni": 15, "atle": 16, "excluded_exams": [6]},
    "sub-11": {"mni": 9, "atle": 10, "excluded_exams": [3]},
    "sub-13": {"mni": 16, "atle": 18, "excluded_exams": [12, 18]},
    "sub-14": {"mni": 14, "atle": 15, "excluded_exams": [15]},
    "sub-18": {"mni": 6, "atle": 9, "excluded_exams": [4, 6, 7]},
    "sub-20": {"mni": 10, "atle": 13, "excluded_exams": [4, 5, 7]},
    "sub-21": {"mni": 13, "atle": 15, "excluded_exams": [6, 14]},
    "sub-22": {"mni": 6, "atle": 9, "excluded_exams": [7, 8, 9]},
    "sub-23": {"mni": 14, "atle": 15, "excluded_exams": [14]},
    "sub-24": {"mni": None, "atle": None, "excluded_exams": [],
               "note": "row left blank in history.txt; sub-24 has no manual masks"},
    "sub-25": {"mni": 11, "atle": 14, "excluded_exams": [8, 9, 10]},
}


# --------------------------------------------------------------------- queries

def session_index() -> dict[tuple[str, str], list[dict]]:
    idx: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in SESSION_ISSUES:
        idx[(e["subject"], e["session"])].append(e)
    return dict(idx)


def excluded_sessions(include_leakage: bool = True) -> set[tuple[str, str]]:
    """Sessions that must not enter the primary cohort."""
    bad = {EXCLUDE} | ({LEAKAGE} if include_leakage else set())
    return {(e["subject"], e["session"]) for e in SESSION_ISSUES
            if e["severity"] in bad}


def excluded_subjects(scope: str = "CL manual masks") -> set[str]:
    return {e["subject"] for e in SUBJECT_ISSUES
            if e["severity"] == EXCLUDE and e.get("scope") == scope}


def dose_eligible_subjects(all_subjects) -> dict:
    """Callers pass row-level subject lists, so de-duplicate before counting.
    Counting 270 session rows as 270 patients is the kind of arithmetic error
    that reads as plausible and silently inflates every per-patient statistic."""
    blocked = excluded_subjects(scope="dose map")
    eligible = sorted({s for s in all_subjects} - blocked)
    return {"eligible": eligible, "n_eligible": len(eligible),
            "blocked": sorted(blocked),
            "reasons": {e["subject"]: e["issue"] for e in SUBJECT_ISSUES
                        if e.get("scope") == "dose map"}}


def delta_t_flag(subject: str) -> dict:
    est = DELTA_T_ESTIMATED.get(subject)
    rec = DELTA_T_DATE_RECOVERY.get(subject)
    return {"subject": subject,
            "kind": ESTIMATED if est else "DOCUMENTED_APPROXIMATE",
            "interval_estimation": est,
            "date_recovery": rec,
            "global_caveat": DELTA_T_GLOBAL_CAVEAT}


def summary(all_subjects=None) -> dict:
    sessions = excluded_sessions()
    leakage = {(e["subject"], e["session"]) for e in SESSION_ISSUES
               if e["severity"] == LEAKAGE}
    cl_blocked = excluded_subjects("CL manual masks")
    subs = sorted(set(all_subjects)) if all_subjects else [
        f"sub-{i:02d}" for i in range(1, 28)]
    return {
        "source": SOURCE,
        "n_session_issues": len(SESSION_ISSUES),
        "n_subject_issues": len(SUBJECT_ISSUES),
        "sessions_excluded": sorted(f"{a}/{b}" for a, b in sessions),
        "sessions_with_future_leakage": sorted(f"{a}/{b}" for a, b in leakage),
        "subjects_without_primary_target": sorted(cl_blocked),
        "n_patients_primary_cohort": len(subs) - len(cl_blocked),
        "dose": dose_eligible_subjects(subs),
        "delta_t_subjects_with_estimated_intervals": sorted(DELTA_T_ESTIMATED),
        "n_delta_t_estimated_subjects": len(DELTA_T_ESTIMATED),
        "intensity_claim": INTENSITY_CLAIM,
        "annotation_provenance": ANNOTATION_PROVENANCE,
        "manual_label_intersection": MANUAL_LABEL_INTERSECTION,
    }
