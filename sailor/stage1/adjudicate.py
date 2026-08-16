"""Adjudicating degenerate primary masks (GATE-0 follow-up).

G1 found 7 of 240 `ContrastEnhancedMask-CL` volumes to be all-zero. §3.2 says
degenerate masks are missing data and must be excluded — and for a segmentation
failure that is right. But an empty enhancing-tumour mask can also be *correct*:
a glioma patient who achieves complete radiological response after resection and
chemoradiation genuinely has no enhancing tumour.

Excluding those would delete complete responders from the cohort, biasing it
toward progressors and inflating every growth metric by removing the hardest
outcome to predict. So the two cases are separated on evidence before anything
is dropped.

Four independent signals, all already measured and cached:

  ONCO   does the automated segmentation ALSO read zero at that session?
         Two independent methods agreeing on zero is strong evidence of a true
         absence; the automated pipeline failing where the manual one succeeded
         (or vice versa) points at a processing failure.
  RANO   the response class recorded for that session. RANO 1 is complete
         response, which predicts a zero mask.
  NEIGH  the same patient's masks immediately before and after. A zero between
         two substantial masks is implausible as a true response.
  EDEMA  is the CL edema mask also empty? Enhancement can resolve while edema
         persists; both empty is more consistent with a failed segmentation.

This module classifies and reports. It does not exclude anything: the decision
is recorded as a phase-gate action with the evidence attached.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

TRUE_ZERO = "LIKELY_TRUE_COMPLETE_RESPONSE"
FAILURE = "LIKELY_SEGMENTATION_FAILURE"
AMBIGUOUS = "AMBIGUOUS"

# RANO coding — REFUTED ASSUMPTION (v0.17)
#
# v0.16 took 1 = complete response and scored every other code as evidence
# AGAINST a true response. Measurement refutes it. Across 240 masks:
#
#     code 1: n=67   0% empty   median CL enhancing volume 13573
#     code 2: n=101  7% empty   median  1790
#     code 3: n=21   0% empty   median   390
#     code 5: n=23   0% empty   median 13507
#
# Complete response is an ABSOLUTE criterion — no enhancing disease. Code 1
# carries the LARGEST median volume in the dataset and not one empty mask. The
# token set is {1,2,3,5} with no 4, so it is not a contiguous four-class scale
# either. Two further facts from history.txt:
#
#   - RANO was transcribed from a clinical Excel sheet, so it is genuinely
#     independent of the masks (it is not circular evidence), BUT
#   - "Some subjects missed RANO on the very latest examination -> filled in
#     RANO based on the previous RANO, if visually similar T1c images."
#     Values are therefore PARTLY CARRIED FORWARD by documented policy, and
#     which ones cannot be recovered.
#
# The descriptor names RANO but prints no mapping. code.tar.bz2 contains no
# script that writes RANO.txt. The mapping is UNVERIFIED and no meaning is
# inferred here. RANO is consequently NOT used as adjudication evidence: a code
# of unknown meaning and partly unknown provenance cannot support excluding a
# patient.
RANO_MAPPING_STATUS = "UNVERIFIED"
RANO_NOTE = (
    "RANO integer coding is UNVERIFIED. The v0.16 assumption 1 = complete "
    "response is REFUTED by measurement: code 1 has the largest median enhancing "
    "volume (13573) and zero empty masks, which is the opposite of what an "
    "absolute complete-response criterion implies. Observed tokens {1,2,3,5} with "
    "no 4. history.txt records that RANO came from an Excel sheet and that some "
    "values were filled in from the previous session. No meaning is inferred, and "
    "RANO is not used as evidence for or against any verdict.")


def _key(path: str) -> tuple[str | None, str | None]:
    s = re.search(r"(sub-[A-Za-z0-9]+)", path)
    e = re.search(r"(ses-[A-Za-z0-9]+)", path)
    return (s.group(1) if s else None), (e.group(1) if e else None)


def _ordinal(ses: str | None) -> int:
    m = re.search(r"(\d+)$", ses or "")
    return int(m.group(1)) if m else 0


def collect_mask_volumes(volume_stats: dict) -> dict:
    """(subject, session) -> {kind: n_nonzero} from cached volume statistics."""
    out: dict[tuple, dict] = defaultdict(dict)
    for key, st in volume_stats.items():
        name = st.get("name", key.split("::", 1)[-1])
        sub, ses = _key(name)
        if not (sub and ses):
            continue
        base = name.rsplit("/", 1)[-1]
        nz = st.get("n_nonzero")
        if nz is None:
            continue
        low = base.lower()
        if low.startswith("contrastenhancedmask-cl"):
            out[(sub, ses)]["CL_enhancing"] = nz
        elif low.startswith("contrastenhancedmask-onco"):
            out[(sub, ses)]["ONCO_enhancing"] = nz
        elif low.startswith("edemamask-cl"):
            out[(sub, ses)]["CL_edema"] = nz
        elif low.startswith("edemamask-onco"):
            out[(sub, ses)]["ONCO_edema"] = nz
        elif low.startswith("necrosismask-onco"):
            out[(sub, ses)]["ONCO_necrosis"] = nz
    return dict(out)


def adjudicate(volume_stats: dict, clinical_rows: list[dict]) -> dict:
    vols = collect_mask_volumes(volume_stats)
    rano = {(r["subject"], r["session"]): r.get("rano") for r in clinical_rows}

    by_subject: dict[str, list] = defaultdict(list)
    for (sub, ses), v in vols.items():
        if "CL_enhancing" in v:
            by_subject[sub].append((_ordinal(ses), ses, v["CL_enhancing"]))
    for lst in by_subject.values():
        lst.sort()

    zeros = [(sub, ses) for (sub, ses), v in vols.items()
             if v.get("CL_enhancing") == 0]
    verdicts = []
    for sub, ses in sorted(zeros):
        v = vols[(sub, ses)]
        onco = v.get("ONCO_enhancing")
        edema = v.get("CL_edema")
        r = rano.get((sub, ses))
        o = _ordinal(ses)
        series = by_subject.get(sub, [])
        prev_e = next(((oo, n) for (oo, _, n) in reversed(series) if oo < o), None)
        nxt_e = next(((oo, n) for (oo, _, n) in series if oo > o), None)
        prev = prev_e[1] if prev_e else None
        nxt = nxt_e[1] if nxt_e else None
        # `by_subject` only holds sessions that HAVE a CL mask, so a session
        # without one is skipped rather than counted as empty — absent != missing
        # != empty. But the skip is silent and unbounded: sub-25 ses-08's "next"
        # neighbour is ses-10, two ordinals and ~174 days away. The gap is now
        # reported so "adjacent" is never read as "immediately adjacent".
        gap_before = (o - prev_e[0]) if prev_e else None
        gap_after = (nxt_e[0] - o) if nxt_e else None

        support_true, support_fail = [], []
        if onco == 0:
            support_true.append("automated ONCO enhancing mask is also empty")
        elif onco:
            support_fail.append(f"automated ONCO mask is NOT empty ({onco} voxels)")
        # RANO deliberately NOT scored — mapping UNVERIFIED, provenance partly
        # carried forward. See RANO_NOTE. The value is still reported.
        if prev and nxt and prev > 0 and nxt > 0:
            support_fail.append(
                f"neighbours non-empty ({prev} before, {nxt} after)")
        elif prev == 0 or nxt == 0:
            support_true.append(
                "an adjacent session with a CL mask is also empty"
                + (f" (ordinal gap {gap_before or gap_after})"
                   if (gap_before or 1) > 1 or (gap_after or 1) > 1 else ""))
        if edema == 0:
            support_fail.append("CL edema mask is ALSO empty — both labels absent")
        elif edema:
            support_true.append(f"CL edema present ({edema} voxels) — "
                                "enhancement resolved but edema did not")

        if support_true and not support_fail:
            verdict = TRUE_ZERO
        elif support_fail and not support_true:
            verdict = FAILURE
        else:
            verdict = AMBIGUOUS

        verdicts.append({
            "subject": sub, "session": ses, "session_ordinal": o,
            "verdict": verdict,
            "n_nonzero_CL_enhancing": 0,
            "n_nonzero_ONCO_enhancing": onco,
            "n_nonzero_CL_edema": edema,
            "rano": r,
            "rano_used_as_evidence": False,
            "neighbour_before": prev, "neighbour_after": nxt,
            "neighbour_gap_before": gap_before, "neighbour_gap_after": gap_after,
            "supports_true_response": support_true,
            "supports_segmentation_failure": support_fail,
        })

    # A contiguous run of empties in ONE patient is a different object from the
    # same count scattered across a cohort: annotation failure is a per-session
    # accident, a sustained run is the signature of a trajectory. Reported so the
    # per-session verdicts are never read without it.
    runs: dict[str, dict] = {}
    for sub in {v["subject"] for v in verdicts}:
        ords = sorted(v["session_ordinal"] for v in verdicts if v["subject"] == sub)
        longest = cur = 1
        for a, b in zip(ords, ords[1:]):
            cur = cur + 1 if b - a == 1 else 1
            longest = max(longest, cur)
        runs[sub] = {"n_zero_sessions": len(ords), "ordinals": ords,
                     "longest_consecutive_run": longest}

    counts = {k: sum(1 for v in verdicts if v["verdict"] == k)
              for k in (TRUE_ZERO, FAILURE, AMBIGUOUS)}
    return {
        "n_primary_masks_measured": sum(1 for v in vols.values()
                                        if "CL_enhancing" in v),
        "n_all_zero": len(zeros),
        "verdicts": verdicts,
        "counts": counts,
        "clustering_by_subject": runs,
        "rano_mapping_status": RANO_MAPPING_STATUS,
        "rano_coding_note": RANO_NOTE,
        "resolution_path": (
            "Neither the descriptor, code.tar.bz2 nor history.txt addresses empty "
            "masks or the RANO coding, so no document can settle an AMBIGUOUS "
            "verdict. The remaining internal evidence is the imaging: extract T1c "
            "at the sessions in question via sailor.stage1.visual_check and look. "
            "Author contact is the final escalation, not the first."),
        "policy": (
            "A session judged LIKELY_TRUE_COMPLETE_RESPONSE is REAL DATA and is "
            "retained: excluding it would delete complete responders and bias the "
            "cohort toward progressors. A session judged "
            "LIKELY_SEGMENTATION_FAILURE is missing data and is excluded per "
            "§3.2. AMBIGUOUS sessions are excluded from the primary analysis and "
            "reported, with a sensitivity analysis that retains them."),
        "metric_consequence": (
            "Dice is undefined when both prediction and reference are empty. Any "
            "retained true-zero session must be scored with a metric that is "
            "defined there — report volume error and count exact-zero agreement "
            "separately rather than letting an undefined Dice silently drop out "
            "of the mean."),
    }


def print_report(res: dict) -> None:
    line = "-" * 78
    print(line)
    print(f"DEGENERATE PRIMARY MASKS — {res['n_all_zero']} of "
          f"{res['n_primary_masks_measured']} are all-zero")
    print(line)
    for v in res["verdicts"]:
        print(f"  {v['subject']}/{v['session']}  ->  {v['verdict']}")
        print(f"    ONCO enhancing: {v['n_nonzero_ONCO_enhancing']}   "
              f"CL edema: {v['n_nonzero_CL_edema']}   RANO: {v['rano']}   "
              f"neighbours: {v['neighbour_before']} / {v['neighbour_after']}")
        for s in v["supports_true_response"]:
            print(f"      + true response: {s}")
        for s in v["supports_segmentation_failure"]:
            print(f"      + failure:       {s}")
    print(line)
    print(f"  {res['counts']}")
    print(f"  {res['policy']}")
    print(f"  NOTE: {res['rano_coding_note']}")
    print(line)


def run(project_root, audit_cache_key: str | None = None) -> dict:
    """Adjudicate from the cached full pass. No archive read."""
    from ..utils.persist import latest_full_pass, load_cache, save_artefact
    project_root = Path(project_root)
    if audit_cache_key is None:
        audit_cache_key = latest_full_pass(project_root, require=("volume_stats",))
    cached = load_cache(project_root, audit_cache_key)
    if not cached:
        raise RuntimeError(f"cache {audit_cache_key} unreadable")
    vol = cached["raw"].get("volume_stats", {})
    clin = json.loads(
        (project_root / "01_DATA_FOUNDATION" / "v2_clinical_table.json").read_text())
    res = adjudicate(vol, clin["rows"])
    res["audit_cache_key"] = audit_cache_key
    res["artefact"] = save_artefact(project_root, "06_QC_REPORTS",
                                    "degenerate_mask_adjudication", res)
    print_report(res)
    return res


def nonfinite_report(volume_stats: dict, top: int = 40) -> dict:
    """Which volumes carry NaN/Inf, and what are the measured ranges? (G10)"""
    bad, ranges = [], []
    for key, st in volume_stats.items():
        name = st.get("name", key.split("::", 1)[-1])
        if st.get("n_nonfinite"):
            bad.append({"name": name, "n_nonfinite": st["n_nonfinite"],
                        "dtype": st.get("dtype"), "role": st.get("role")})
        # v0.17 — the range check ran on every sampled volume, including
        # FreeSurfer label maps (IDs up to 2035) and -icor-zscore variants
        # (negative by construction). That produced 13 of 13 false positives:
        # neither file class carries intensities, so the descriptor's 0-255
        # claim does not apply to them. `is_intensity` is recorded at read time.
        if (st.get("role") in ("image_sample", "icor_full", "plain_paired")
                and st.get("min") is not None):
            rec = {"name": name, "min": st["min"], "max": st["max"],
                   "mean": st.get("mean"), "role": st.get("role"),
                   "is_intensity": st.get("is_intensity", True)}
            ranges.append(rec)
    intensity = [r for r in ranges if r.get("is_intensity", True)]
    excluded = [r for r in ranges if not r.get("is_intensity", True)]
    out_of_range = [r for r in intensity if r["min"] < 0 or r["max"] > 255]
    # Coverage matters as much as the counts: if every sampled volume comes from
    # one subject, a prevalence figure is a statement about that subject only.
    subs = sorted({m.group(1) for r in ranges
                   for m in [re.search(r"(sub-[A-Za-z0-9]+)", r["name"])] if m})
    return {
        "n_volumes_with_nonfinite": len(bad),
        "volumes_with_nonfinite": bad[:top],
        "n_image_samples": len(ranges),
        "n_intensity_volumes_checked": len(intensity),
        "n_excluded_non_intensity": len(excluded),
        "excluded_non_intensity_note": (
            "FreeSurfer label maps and -icor-zscore variants are excluded from "
            "the 0-255 range check: their values are label IDs and z-scores, not "
            "intensities. In v0.16 these were 13 of 13 'failures'."),
        "subjects_covered": subs,
        "n_subjects_covered": len(subs),
        "coverage_warning": (
            None if len(subs) > 1 else
            f"ALL sampled volumes come from {subs[0] if subs else 'one subject'}. "
            "Every intensity statistic, and any prevalence figure derived from "
            "it, describes that subject and not the cohort."),
        "n_outside_0_255": len(out_of_range),
        "outside_0_255": sorted(out_of_range,
                                key=lambda r: -abs(r["max"]))[:top],
        "implication": (
            "Non-finite voxels propagate silently through convolution and "
            "normalisation and surface as NaN loss many hours into training. "
            "They must be handled explicitly in the preprocessing version — "
            "masked, clipped or the volume excluded — and the choice recorded. "
            "Values outside 0-255 contradict history.txt, so normalisation is "
            "chosen from these measured statistics, not from the descriptor."),
    }
