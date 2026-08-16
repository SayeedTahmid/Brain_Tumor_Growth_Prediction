"""Is `-icor` the output of LONGITUDINAL intensity normalisation? (BLOCKING)

The data descriptor lists the MNI preprocessing chain and step 10 is
"Longitudinal intensity normalization (PLHM)", implemented with
PLHM/nyul-normalize. PLHM fits histogram landmarks JOINTLY across a patient's
timepoints and maps every volume onto the shared scale.

If `-icor` is that output, then the intensity values at session t were computed
using sessions t+1, t+2, ... — every FUTURE scan in that patient's series. For a
framework whose target is `MRI_t+dt = MRI_t + dMRI`, that is future information
inside the model INPUT, present in all 270 sessions rather than the three ses-01
cases history.txt documents.

This matters more than the treatment confound. The confound limits what may be
CLAIMED about treatment; input leakage of this kind would compromise C_REALISM
and C_PREDICTION as well — the results current evidence expects to be positive.

WHAT IS AND IS NOT ESTABLISHED
------------------------------
The preprocessing code is NOT in the distribution: `code.tar.bz2` holds only
curation and BIDS-conversion scripts (16 members, zero hits for icor / plhm /
nyul / zscore), and history.txt mentions normalisation only for NAWM/CBV. The
chain lives in the external `crai-lproc` repository, available on request.

So this module cannot prove provenance. It measures a CONSEQUENCE and reports a
signal, and the verdict vocabulary keeps that distinction:

    CONSISTENT_WITH_JOINT_FIT      landmarks align across a patient's sessions
                                   far more in `-icor` than in the plain variant
    CONSISTENT_WITH_PER_VOLUME     no such alignment
    INCONCLUSIVE                   not enough paired volumes measured

None of these is a provenance verdict. A CONSISTENT_WITH_JOINT_FIT result raises
the intensity-variant lock for reconsideration; it does not by itself decide it,
and the escalation path (asking the dataset authors, or reading crai-lproc) stays
open.

METHOD
------
Nyul/PLHM maps each volume onto a SHARED set of landmark percentiles. So if the
fit is joint across a patient's timepoints, the landmark values of `-icor` will
vary far less across that patient's sessions than the same landmarks of the
plain variant do. Per patient and modality:

    cv(p) = stdev over sessions / mean over sessions,  for each percentile p

then compare cv_icor against cv_plain on the SAME patient/modality/percentile.
A large systematic drop indicates shared landmarks.

The threshold is pre-registered here, before the measurement is run, so it
cannot be chosen after seeing the numbers (§2.2 in spirit).
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------- pre-registered
#: Median ratio cv(-icor) / cv(plain) at or below this is read as joint fitting.
#: Rationale: per-volume processing (N4, denoise) changes scale modestly and
#: leaves between-session variation largely intact, so a ratio near 1 is
#: expected. Shared landmarks collapse the variation by construction. 0.5 is a
#: deliberately conservative midpoint: it demands the icor spread be at most half
#: the plain spread before the stronger reading is allowed.
JOINT_FIT_RATIO = 0.50
#: Minimum (patient, modality) series with >=3 sessions on both variants.
MIN_SERIES = 8

JOINT = "CONSISTENT_WITH_JOINT_FIT"
PER_VOLUME = "CONSISTENT_WITH_PER_VOLUME"
INCONCLUSIVE = "INCONCLUSIVE"

_SUB = re.compile(r"(sub-[A-Za-z0-9]+)")
_SES = re.compile(r"(ses-[A-Za-z0-9]+)")
#: Plain vs icor for the four structural modalities. `-icor-zscore` is excluded:
#: it is a further transform and, on sub-13, is destroyed (every finite voxel 0).
_PLAIN = re.compile(r"^(T1|T1c|T2|Flair)\.nii", re.IGNORECASE)
_ICOR = re.compile(r"^(T1|T1c|T2|Flair)-icor\.nii", re.IGNORECASE)


def _cv(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if len(vals) < 3:
        return None
    m = statistics.fmean(vals)
    if m == 0:
        return None
    return statistics.stdev(vals) / abs(m)


def measure(volume_stats: dict) -> dict:
    """Compare within-patient landmark spread: plain variant vs `-icor`."""
    # (subject, modality, variant) -> {session: {percentile: value}}
    series: dict[tuple, dict] = defaultdict(dict)
    n_with_pct = 0
    for key, st in volume_stats.items():
        name = st.get("name", key.split("::", 1)[-1])
        base = name.rsplit("/", 1)[-1]
        pct = st.get("percentiles_nonzero")
        if not pct:
            continue
        n_with_pct += 1
        sub, ses = _SUB.search(name), _SES.search(name)
        if not (sub and ses):
            continue
        if _ICOR.match(base):
            variant, mod = "icor", _ICOR.match(base).group(1).lower()
        elif _PLAIN.match(base):
            variant, mod = "plain", _PLAIN.match(base).group(1).lower()
        else:
            continue
        series[(sub.group(1), mod, variant)][ses.group(1)] = pct

    comparisons, ratios = [], []
    keys = {(s, m) for (s, m, _) in series}
    for sub, mod in sorted(keys):
        plain = series.get((sub, mod, "plain"), {})
        icor = series.get((sub, mod, "icor"), {})
        shared = sorted(set(plain) & set(icor))
        if len(shared) < 3:
            continue
        percentiles = sorted(
            {p for s in shared for p in plain[s]} & {p for s in shared for p in icor[s]},
            key=float)
        row = {"subject": sub, "modality": mod, "n_sessions": len(shared),
               "per_percentile": {}}
        for p in percentiles:
            cv_p = _cv([plain[s].get(p) for s in shared])
            cv_i = _cv([icor[s].get(p) for s in shared])
            if cv_p is None or cv_i is None or cv_p == 0:
                continue
            r = cv_i / cv_p
            row["per_percentile"][p] = {"cv_plain": round(cv_p, 5),
                                        "cv_icor": round(cv_i, 5),
                                        "ratio": round(r, 5)}
            ratios.append(r)
        if row["per_percentile"]:
            row["median_ratio"] = round(
                statistics.median(v["ratio"] for v in row["per_percentile"].values()), 5)
            comparisons.append(row)

    if len(comparisons) < MIN_SERIES or not ratios:
        verdict, detail = INCONCLUSIVE, (
            f"only {len(comparisons)} (patient, modality) series had >=3 sessions "
            f"measured on BOTH the plain and -icor variants; {MIN_SERIES} were "
            "pre-registered as the minimum. Re-run the pass with read_icor=True "
            "and a per-subject sample budget that also reads the plain variants.")
    else:
        med = statistics.median(ratios)
        if med <= JOINT_FIT_RATIO:
            verdict = JOINT
            detail = (
                f"median cv(-icor)/cv(plain) = {med:.3f} over {len(ratios)} "
                f"landmark comparisons across {len(comparisons)} series, at or "
                f"below the pre-registered {JOINT_FIT_RATIO}. Landmark values "
                "align across a patient's sessions far more in -icor than in the "
                "plain variant, which is what a JOINT longitudinal fit produces. "
                "This does not prove PLHM was used; it is consistent with it.")
        else:
            verdict = PER_VOLUME
            detail = (
                f"median cv(-icor)/cv(plain) = {med:.3f} over {len(ratios)} "
                f"landmark comparisons across {len(comparisons)} series, above "
                f"the pre-registered {JOINT_FIT_RATIO}. Between-session spread "
                "survives in -icor, which is what per-volume processing leaves "
                "behind. No evidence here of a joint longitudinal fit.")

    return {
        "check": "plhm_icor_provenance",
        "blocking": True,
        "verdict": verdict,
        "detail": detail,
        "pre_registered": {"joint_fit_ratio": JOINT_FIT_RATIO,
                           "min_series": MIN_SERIES,
                           "note": "thresholds fixed in source before the measurement ran"},
        "n_volumes_with_percentiles": n_with_pct,
        "n_series_compared": len(comparisons),
        "n_landmark_comparisons": len(ratios),
        "median_ratio": round(statistics.median(ratios), 5) if ratios else None,
        "comparisons": comparisons,
        "provenance_status": "UNVERIFIED",
        "provenance_note": (
            "The MNI preprocessing code is NOT in the distribution. code.tar.bz2 "
            "contains 16 curation/BIDS scripts with zero references to icor, "
            "plhm, nyul or zscore; history.txt mentions normalisation only for "
            "NAWM/CBV. The chain lives in the external crai-lproc repository. "
            "This check therefore measures a consequence, not provenance."),
        "if_joint": (
            "If joint, every session's intensities were computed with that "
            "patient's FUTURE sessions in scope: future information inside the "
            "model input, in all 270 sessions. The intensity-variant lock would "
            "need reconsideration. Mitigations: use the plain variants (0-255, "
            "no PLHM), or re-normalise from a within-training-fold fit. Neither "
            "is enacted here."),
    }


def run(project_root, audit_cache_key: str | None = None) -> dict:
    """Measure from a cached pass. No archive read."""
    from ..utils.persist import latest_full_pass, load_cache, save_artefact
    project_root = Path(project_root)
    if audit_cache_key is None:
        audit_cache_key = latest_full_pass(project_root, require=("volume_stats",))
    cached = load_cache(project_root, audit_cache_key)
    if not cached:
        raise RuntimeError(f"cache {audit_cache_key} unreadable")
    res = measure(cached["raw"].get("volume_stats", {}))
    res["audit_cache_key"] = audit_cache_key
    res["artefact"] = save_artefact(project_root, "06_QC_REPORTS",
                                    "plhm_icor_check", res)
    print_report(res)
    return res


def print_report(res: dict) -> None:
    line = "-" * 78
    print(line)
    print("PLHM / -icor PROVENANCE CHECK  (BLOCKING before Phase 5)")
    print(line)
    print(f"  verdict:   {res['verdict']}")
    print(f"  series:    {res['n_series_compared']}  "
          f"landmark comparisons: {res['n_landmark_comparisons']}")
    print(f"  median cv(-icor)/cv(plain): {res['median_ratio']}  "
          f"(pre-registered threshold {JOINT_FIT_RATIO})")
    print(f"\n  {res['detail']}")
    print(f"\n  provenance: {res['provenance_status']}")
    print(f"  {res['provenance_note']}")
    if res["verdict"] == JOINT:
        print(f"\n  {res['if_joint']}")
    print(line)
