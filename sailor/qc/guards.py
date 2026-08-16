"""Integrity guards (§9). Stage-1 subset: G1, G5, G7, G8, G9, G10.

Every guard returns the same record shape and one of three statuses:

  PASS          the check ran on real measurements and succeeded
  FAIL          the check ran and failed -> STOP protocol (§2.3)
  INCONCLUSIVE  the check could not run because an input was absent

INCONCLUSIVE is never rendered as PASS. §18.4 condition 5 makes converting one
into the other the most damaging possible change to this codebase, so the three
statuses are kept structurally distinct and the report prints them separately.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from ..config import (FORBIDDEN_INPUT_FILES, PRIMARY_TARGET_COMPONENT,
                      PRIMARY_TARGET_MASK, QUARANTINE_FILES,
                      REQUIRED_SEQUENCES_EXTENDED, REQUIRED_SEQUENCES_PRIMARY)

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"


def _rec(gid, title, status, detail, evidence=None):
    return {"guard": gid, "title": title, "status": status, "detail": detail,
            "evidence": evidence or {}}


# ------------------------------------------------------------------------ G1

def g1_degenerate_labels(file_rows, volume_stats, near_empty_voxels: int = 10):
    """Degenerate masks for the locked primary target; inventory for the rest."""
    def stat_for(row):
        return (volume_stats.get(row["path"])
                or volume_stats.get(f"{row.get('archive')}::{row['path']}"))

    def audit(rows):
        out = {"n_files": len(rows), "n_measured": 0, "all_zero": [],
               "all_one": [], "near_empty": [], "non_binary": [],
               "voxel_counts": {}}
        for r in rows:
            st = stat_for(r)
            if not st:
                continue
            out["n_measured"] += 1
            sid = f"{r['subject']}/{r['session']}"
            nz = st.get("n_nonzero")
            out["voxel_counts"][sid] = nz
            if nz == 0:
                out["all_zero"].append({"session": sid, "path": r["path"]})
            elif nz == st.get("n_voxels"):
                out["all_one"].append({"session": sid, "path": r["path"]})
            elif nz is not None and nz < near_empty_voxels:
                out["near_empty"].append({"session": sid, "path": r["path"],
                                          "n_nonzero": nz})
            if st.get("labels") is not None and not st.get("binary"):
                out["non_binary"].append({"session": sid, "path": r["path"],
                                          "labels": st["labels"]})
        return out

    from ..data.known_issues import excluded_subjects, excluded_sessions
    blocked_subs = excluded_subjects("CL manual masks")
    blocked_sess = excluded_sessions()

    primary = [r for r in file_rows
               if r["annotation_kind"] == PRIMARY_TARGET_MASK
               and r["annotation_component"] == PRIMARY_TARGET_COMPONENT]
    flair = [r for r in file_rows if r["annotation_kind"] == "CL"
             and r["annotation_component"] == "t2wflair_hyperintensity"]
    onco = [r for r in file_rows if r["annotation_kind"] == "ONCO"]

    p = audit(primary)
    inventory = {"CL_t2wflair": audit(flair), "ONCO": audit(onco)}

    if p["n_files"] == 0:
        status, detail = INCONCLUSIVE, (
            f"No files resolved as {PRIMARY_TARGET_MASK}/{PRIMARY_TARGET_COMPONENT}. "
            "Cannot count degeneracy for a target that was not located.")
    elif p["n_measured"] == 0:
        status, detail = INCONCLUSIVE, "Primary target files found but no volume statistics measured."
    else:
        bad = len(p["all_zero"]) + len(p["all_one"]) + len(p["near_empty"])
        status = PASS if bad == 0 else FAIL
        detail = (f"{p['n_measured']} primary-target masks measured; "
                  f"{len(p['all_zero'])} all-zero, {len(p['all_one'])} all-one, "
                  f"{len(p['near_empty'])} near-empty (<{near_empty_voxels} voxels). "
                  "Degenerate masks are missing data and must be excluded, not trained on.")
    detail += (f" Documented in history.txt: {sorted(blocked_subs)} have no manual "
               f"CL masks at all, and {len(blocked_sess)} session(s) are excluded "
               "or carry future-information leakage; those are absences, not "
               "degeneracies, and are enforced separately.")
    return _rec("G1", "Degenerate labels", status, detail,
                {"primary": p, "inventory_only": inventory,
                 "documented_subjects_without_target": sorted(blocked_subs),
                 "documented_excluded_sessions": sorted(f"{a}/{b}" for a, b in blocked_sess),
                 "inventory_note": "CL_t2wflair and ONCO counts are reported and "
                                   "may not influence cohort or preprocessing (§3.2)."})


# ------------------------------------------------------------------------ G5

def g5_leakage_stage1(manifest_inputs, legacy_classification, subject_sessions):
    """Stage-1 scope of the leakage guard.

    Split-level leakage is not checkable before splits exist; what is checkable
    now is that no quarantined artefact and no prior split is on the input path,
    and that no session is claimed by two subjects.
    """
    quarantined_on_path = [p for p in manifest_inputs
                           if any(q in str(p) for q in QUARANTINE_FILES)]
    forbidden_on_path = [p for p in manifest_inputs
                         if any(f in str(p) for f in FORBIDDEN_INPUT_FILES)]
    seen: dict[str, set] = defaultdict(set)
    for sub, ses_map in subject_sessions.items():
        for ses in ses_map:
            seen[ses].add(sub)
    collisions = {ses: sorted(subs) for ses, subs in seen.items() if len(subs) > 1}

    failures = []
    if quarantined_on_path:
        failures.append(f"{len(quarantined_on_path)} quarantined artefact(s) on the input path")
    if forbidden_on_path:
        failures.append(f"{len(forbidden_on_path)} prior split file(s) on the input path")
    status = FAIL if failures else PASS
    detail = ("; ".join(failures) if failures else
              "No quarantined artefact and no prior split is an input. "
              "Split-level leakage checks run at Stage 2 sections 11-12.")
    return _rec("G5", "Leakage (Stage-1 scope)", status, detail,
                {"quarantined_on_path": quarantined_on_path,
                 "forbidden_on_path": forbidden_on_path,
                 "session_id_collisions": collisions,
                 "scope": "input-path purity only; fold-level checks deferred to §15.3 s.11-12",
                 "prior_splits_present_in_legacy":
                     legacy_classification.get("forbidden_as_input", [])})


# ------------------------------------------------------------------------ G7

def g7_delta_t_provenance(recovery_report, per_session_dt, clinical_table=None):
    """Which version did each Δt come from, and is it exact or approximate?

    `history.txt` settles this: the intervals in `intervals_days.txt` were
    manually transcribed from DICOM headers and a spreadsheet, with some dates
    estimated and some intervals interpolated by dividing a long gap into equal
    parts. That is documented, not inferred, so Δt is APPROXIMATE for the whole
    cohort and INTERPOLATED for the subjects the log names. No exact source
    exists in the artefacts present.
    """
    from ..data.known_issues import (DELTA_T_ESTIMATED, DELTA_T_GLOBAL_CAVEAT,
                                     delta_t_flag)

    if clinical_table is not None:
        n_sessions = len(clinical_table.get("rows", []))
        n_with_dt = clinical_table.get("n_sessions_with_days_from_first") or 0
        subs = sorted({r["subject"] for r in clinical_table.get("rows", [])})
        n_interp = len([s for s in subs if s in DELTA_T_ESTIMATED])
        problems = clinical_table.get("interval_problems", [])
        if n_with_dt == 0:
            return _rec("G7", "Δt provenance", INCONCLUSIVE,
                        "intervals_days.txt was not read, so no Δt exists yet. "
                        "Δt is UNAVAILABLE, not approximate.",
                        {"n_sessions": n_sessions, "source": "intervals_days.txt"})
        detail = (
            f"Δt recovered for {n_with_dt}/{n_sessions} session(s) from "
            f"intervals_days.txt. history.txt documents these as manually "
            f"transcribed with estimated dates and interpolated intervals: "
            f"{n_interp} of {len(subs)} subject(s) have at least one interval "
            "produced by dividing a long gap into equal parts. Δt is APPROXIMATE "
            "cohort-wide and INTERPOLATED for those subjects. Declare this in the "
            "limitations and run the Δt sensitivity analysis before any "
            "Δt-conditioned claim (C1-C4).")
        if problems:
            detail += (f" {len(problems)} subject(s) have an interval count that "
                       "does not match their session count.")
        return _rec("G7", "Δt provenance", FAIL, detail,
                    {"source": "intervals_days.txt",
                     "n_sessions": n_sessions,
                     "n_sessions_with_delta_t": n_with_dt,
                     "n_subjects_interpolated": n_interp,
                     "subjects_interpolated": sorted(
                         s for s in subs if s in DELTA_T_ESTIMATED),
                     "per_subject_flags": {s: delta_t_flag(s) for s in subs
                                           if s in DELTA_T_ESTIMATED},
                     "interval_count_problems": problems,
                     "documented_caveat": DELTA_T_GLOBAL_CAVEAT,
                     "exact_source_available": False,
                     "why_not_exact": ("No acq_time survives in rawdata_BIDS and no "
                                       "DICOM sourcedata is present; the authors "
                                       "themselves reconstructed the dates.")})

    sources = Counter(v.get("source", "none") for v in per_session_dt.values())
    exact_sources = {"raw_scans_tsv_acq_time", "source_dicom", "raw_exam_date"}
    n_exact = sum(c for s, c in sources.items() if s in exact_sources)
    n_total = sum(sources.values())
    if n_total == 0:
        status = INCONCLUSIVE
        detail = ("No Δt value could be constructed from artefacts present. "
                  "All downstream Δt is UNAVAILABLE, not approximate.")
    elif n_exact == n_total:
        status = PASS
        detail = f"All {n_total} intervals derive from an exact-date source."
    else:
        status = FAIL
        detail = (f"{n_total - n_exact} of {n_total} intervals are approximate. "
                  "Declare Δt approximate in limitations and run the Δt "
                  "sensitivity analysis before any Δt-conditioned claim.")
    return _rec("G7", "Δt provenance", status, detail,
                {"source_counts": dict(sources),
                 "recovery_attempts": recovery_report,
                 "exact_sources_recognised": sorted(exact_sources)})


# ------------------------------------------------------------------------ G8

def g8_session_correspondence(link_table, mni_sessions, raw_sessions,
                              raw_side_scanned: bool = True):
    """MNI <-> raw join legality. `ses-XX` is never assumed to align."""
    if link_table.get("status") != "OK" or not link_table.get("pairs"):
        return _rec("G8", "Session correspondence", INCONCLUSIVE,
                    "raw-mni-link.tsv absent or its columns did not resolve; "
                    "no legal join exists, so MNI and raw sessions may not be "
                    "combined at all.", {"link_status": link_table.get("status"),
                                         "resolved_columns": link_table.get("resolved_columns")})
    fwd, rev = defaultdict(set), defaultdict(set)
    for p in link_table["pairs"]:
        if p["subject"] and p["mni_session"] and p["raw_session"]:
            fwd[(p["subject"], p["mni_session"])].add(p["raw_session"])
            rev[(p["subject"], p["raw_session"])].add(p["mni_session"])
    non_unique_fwd = {f"{k[0]}/{k[1]}": sorted(v) for k, v in fwd.items() if len(v) > 1}
    non_unique_rev = {f"{k[0]}/{k[1]}": sorted(v) for k, v in rev.items() if len(v) > 1}
    linked_mni = {k for k in fwd}
    unmatched_mni = sorted(f"{s}/{ses}" for s, sess in mni_sessions.items()
                           for ses in sess if (s, ses) not in linked_mni)
    linked_raw = {k for k in rev}
    unmatched_raw = (sorted(f"{s}/{ses}" for s, sess in raw_sessions.items()
                            for ses in sess if (s, ses) not in linked_raw)
                     if raw_side_scanned else [])
    identical = sum(1 for p in link_table["pairs"]
                    if p["mni_session"] and p["mni_session"] == p["raw_session"])
    dropped = link_table.get("n_raw_exams_dropped_before_mni_ses01", {}) or {}
    n_raw_no_mni = link_table.get("n_raw_without_mni", 0)
    problems = []
    if not raw_side_scanned:
        # An unscanned archive is not an empty one. Reporting 0 unmatched raw
        # sessions here would be a measurement nobody made.
        pass
    if non_unique_fwd or non_unique_rev:
        problems.append("join is not one-to-one")
    if unmatched_mni:
        problems.append(f"{len(unmatched_mni)} MNI session(s) unmatched")
    status = FAIL if problems else PASS
    detail = ("; ".join(problems) if problems else
              f"{len(fwd)} MNI sessions join one-to-one through raw-mni-link.tsv.")
    if not raw_side_scanned:
        detail += (" The raw archive was not scanned, so the raw side of the join "
                   "is UNVERIFIED: unmatched raw sessions are unknown, not zero.")
    if n_raw_no_mni:
        vals = sorted(dropped.values())
        detail += (f" {n_raw_no_mni} raw exam(s) have NO MNI counterpart, and they "
                   f"are systematically the earliest per patient: MNI ses-01 sits "
                   f"at raw exam {min(vals) + 1 if vals else '?'}-"
                   f"{max(vals) + 1 if vals else '?'}. MNI ses-01 is therefore NOT "
                   "a patient's first examination, and weeks-from-first is measured "
                   "from the first MNI exam, not from diagnosis or surgery.")
    return _rec("G8", "Session correspondence", status, detail,
                {"n_link_rows": link_table.get("n_rows"),
                 "n_joined_mni_sessions": len(fwd),
                 "non_unique_mni_to_raw": non_unique_fwd,
                 "non_unique_raw_to_mni": non_unique_rev,
                 "unmatched_mni_sessions": unmatched_mni[:200],
                 "unmatched_raw_sessions": (unmatched_raw[:200] if raw_side_scanned
                                            else "UNSCANNED"),
                 "raw_side_scanned": raw_side_scanned,
                 "n_index_identical_pairs": identical,
                 "n_raw_without_mni_counterpart": n_raw_no_mni,
                 "mni_ses01_maps_to_raw": link_table.get("mni_ses01_maps_to_raw"),
                 "n_raw_exams_dropped_before_mni_ses01": dropped,
                 "note": "Index equality is coincidence, not correspondence."})


# ------------------------------------------------------------------------ G9

def g9_missing_tsv(missing_table, sessions_by_subject, present_sequences,
                   raw_to_mni: dict | None = None, link_table=None):
    """Honour the official exclusion list and report the real survivor count.

    For the wide y/n layout the polarity is not taken on faith. Both readings
    are scored against the sequences actually observed on disk, and the one that
    agrees better is adopted -- with the agreement rate reported, so a weak
    margin is visible rather than hidden. If the observed inventory is empty
    (a structural or truncated pass), no polarity is adopted and the guard
    reports INCONCLUSIVE rather than guessing.
    """
    if missing_table.get("status") != "OK":
        return _rec("G9", "Honour missing.tsv", INCONCLUSIVE,
                    "missing.tsv columns did not resolve; no exclusion list is in force.",
                    {"resolved_columns": missing_table.get("resolved_columns"),
                     "layout": missing_table.get("layout")})

    layout = missing_table.get("layout", "long")
    idx_y = {tuple(k.split("/")): set(v) for k, v in missing_table["index"].items()}
    idx_n = {tuple(k.split("/")): set(v)
             for k, v in missing_table.get("index_inverted", {}).items()}

    # Derive the raw->MNI map from the link table when one was not passed in.
    if raw_to_mni is None and link_table and link_table.get("status") == "OK" \
            and link_table.get("pairs"):
        raw_to_mni = {(p_["subject"], p_["raw_session"]): p_["mni_session"]
                      for p_ in link_table["pairs"]
                      if p_.get("subject") and p_.get("raw_session")
                      and p_.get("mni_session")} or None

    # missing.tsv is indexed in RAW/SOURCE session numbering (337 rows) while the
    # masks and volumes live in MNI numbering (270 sessions). Applying one to the
    # other directly is the §3.1(3) hazard: `ses-XX` indices must never be assumed
    # to align. The only legal translation is raw-mni-link.tsv, so without it the
    # exclusion list cannot be applied at all.
    n_raw_keys = len(idx_y)
    mapped = False
    unmapped_keys: list[str] = []
    if raw_to_mni:
        def translate(idx):
            out = {}
            for (sub, raw_ses), v in idx.items():
                tgt = raw_to_mni.get((sub, raw_ses))
                if tgt is None:
                    unmapped_keys.append(f"{sub}/{raw_ses}")
                    continue
                out[(sub, tgt)] = v
            return out
        idx_y, idx_n = translate(idx_y), translate(idx_n)
        unmapped_keys = sorted(set(unmapped_keys))
        mapped = True

    # ---- settle polarity against observed files -------------------------
    polarity = "y_means_missing"
    agreement = None
    margin = None
    n_comparable = sum(1 for k in idx_y if k in present_sequences)
    if layout == "wide" and n_comparable:
        def score(missing_idx):
            hit = tot = 0
            for key, declared_missing in missing_idx.items():
                have = {s.lower() for s in present_sequences.get(key, set())}
                if not have:
                    continue
                for seq in declared_missing:
                    tot += 1
                    if seq not in have:
                        hit += 1
            return (hit / tot) if tot else None
        a_y, a_n = score(idx_y), score(idx_n)
        if a_y is not None and a_n is not None:
            polarity = "y_means_missing" if a_y >= a_n else "n_means_missing"
            agreement = round(max(a_y, a_n), 4)
            margin = round(abs(a_y - a_n), 4)
        elif a_y is not None:
            agreement = round(a_y, 4)
    elif layout == "wide":
        polarity = "UNDETERMINED"

    idx = idx_y if polarity in ("y_means_missing", "UNDETERMINED") else idx_n

    def survivors(req):
        keep, drop = [], []
        for sub, ses_map in sessions_by_subject.items():
            for ses in ses_map:
                declared_missing = idx.get((sub, ses), set())
                have = {s.lower() for s in present_sequences.get((sub, ses), set())}
                blocked = [q for q in req
                           if q.lower() in declared_missing or q.lower() not in have]
                (drop if blocked else keep).append(
                    {"session": f"{sub}/{ses}", "blocked_by": blocked})
        return keep, drop

    def declared_survivors(req):
        """Survivors by the exclusion list alone, independent of what was scanned.

        This is the number that matters while the archives are only partly read:
        it is the cohort the official list permits, before file-level checks.
        """
        keep = []
        for key, declared_missing in idx.items():
            if not any(q.lower() in declared_missing for q in req):
                keep.append(f"{key[0]}/{key[1]}")
        return sorted(keep)

    keep_p, drop_p = survivors(REQUIRED_SEQUENCES_PRIMARY)
    keep_e, _ = survivors(REQUIRED_SEQUENCES_EXTENDED)
    decl_p = declared_survivors(REQUIRED_SEQUENCES_PRIMARY)
    decl_e = declared_survivors(REQUIRED_SEQUENCES_EXTENDED)
    n_patients_declared = len({s.split("/")[0] for s in decl_p})
    n_sessions_seen = sum(len(v) for v in sessions_by_subject.values())

    status = PASS
    detail = (f"By the exclusion list alone: {len(decl_p)} session(s) across "
              f"{n_patients_declared} patient(s) satisfy {REQUIRED_SEQUENCES_PRIMARY}, "
              f"{len(decl_e)} satisfy {REQUIRED_SEQUENCES_EXTENDED}. "
              f"Cross-checked against observed files: {len(keep_p)} of "
              f"{n_sessions_seen} scanned session(s) survive.")
    if layout == "wide":
        detail += (f" Polarity {polarity}"
                   + (f" (agreement {agreement}, margin {margin})."
                      if agreement is not None else " — UNVERIFIED, no observed "
                      "files to check against."))
    if not mapped and n_sessions_seen:
        status = INCONCLUSIVE
        detail = (
            f"missing.tsv is indexed in raw/source session numbering "
            f"({n_raw_keys} key(s)) but the observed sessions are in MNI numbering "
            f"({n_sessions_seen}). No raw-mni-link.tsv mapping was supplied, so the "
            "exclusion list CANNOT be applied: `ses-XX` indices do not align across "
            "versions (§3.1(3)). Supply the mapping or the cohort stays undefined. "
            + detail)
    elif not idx and n_sessions_seen == 0:
        status = INCONCLUSIVE
        detail = ("The exclusion list carries no session entries and no session was "
                  "observed on disk. Nothing was measured, so no cohort exists to "
                  "report -- this is an unread inventory, not an empty dataset.")
    elif polarity == "UNDETERMINED":
        status = INCONCLUSIVE
        detail = ("y/n polarity of the wide exclusion matrix could not be settled "
                  "against observed files, so no exclusion is in force. " + detail)
    elif len(decl_p) == 0:
        status = FAIL
        detail = ("ZERO sessions satisfy the primary modality requirement by the "
                  "exclusion list. The cohort is empty. " + detail)
    elif n_sessions_seen and len(keep_p) == 0:
        any_seq = any(present_sequences.get(k) for k in present_sequences)
        status = INCONCLUSIVE if not any_seq else FAIL
        if not any_seq:
            detail = ("No sequence was resolved for ANY scanned session, so the "
                      "modality requirement could not be evaluated. This is a "
                      "filename-resolution failure, not an empty cohort: the "
                      "survivor count here means nothing. " + detail)
        else:
            detail = ("ZERO of the scanned sessions satisfy the requirement though "
                      "the exclusion list permits some. " + detail)
    elif margin is not None and margin < 0.10:
        status = INCONCLUSIVE
        detail = ("Polarity margin is too small to adopt a reading confidently. "
                  + detail)

    return _rec("G9", "Honour missing.tsv", status, detail,
                {"n_missing_tsv_entries": missing_table.get("n_rows"),
                 "layout": layout,
                 "raw_space_keys": n_raw_keys,
                 "mapped_through_raw_mni_link": mapped,
                 "raw_keys_without_mni_counterpart": unmapped_keys[:100],
                 "n_raw_keys_unmapped": len(unmapped_keys),
                 "polarity_adopted": polarity,
                 "polarity_agreement": agreement,
                 "polarity_margin": margin,
                 "n_sessions_observed": n_sessions_seen,
                 "required_primary": REQUIRED_SEQUENCES_PRIMARY,
                 "required_extended": REQUIRED_SEQUENCES_EXTENDED,
                 "n_sessions_declared_primary": len(decl_p),
                 "n_patients_declared_primary": n_patients_declared,
                 "n_sessions_declared_extended": len(decl_e),
                 "n_sessions_primary": len(keep_p),
                 "n_sessions_extended": len(keep_e),
                 "declared_surviving_sessions_primary": decl_p,
                 "surviving_sessions_primary": [k["session"] for k in keep_p],
                 "excluded_primary": drop_p[:200]})


# ----------------------------------------------------------------------- G10

def g10_intensity_sanity(volume_stats, mni_archive: str | None = None):
    """Measured dtype and range vs the descriptor's uint8 0-255 claim.

    The claim concerns the MNI derivatives, so when `mni_archive` is given the
    verdict is computed on those volumes and other archives are reported
    separately rather than averaged in.
    """
    # v0.17 — two corrections. (1) `icor_full` volumes are images too and were
    # invisible to this guard. (2) The 0-255 claim concerns INTENSITIES; running
    # it over FreeSurfer label maps (IDs to 2035) and -icor-zscore variants
    # (negative by construction) produced 13 of 13 false positives.
    all_images = [v for v in volume_stats.values()
                  if v.get("role") in ("image_sample", "icor_full", "plain_paired")]
    all_images = [v for v in all_images if v.get("is_intensity", True)]
    images = ([v for v in all_images if v.get("archive") == mni_archive]
              if mni_archive else all_images)
    other = [v for v in all_images if v not in images]
    subs = sorted({m.group(1) for v in images
                   for m in [re.search(r"(sub-[A-Za-z0-9]+)", v.get("name", ""))] if m})
    if not images:
        return _rec("G10", "Intensity sanity", INCONCLUSIVE,
                    ("No image volume from the MNI derivatives was read in full; "
                     "dtype and range unmeasured where the claim applies."),
                    {"n_images_measured": 0, "mni_archive": mni_archive,
                     "n_images_other_archives": len(other)})
    dtypes = Counter(v["dtype"] for v in images)
    frac = [v["name"] for v in images if v.get("has_fractional_values")]
    out_of_range = [v["name"] for v in images
                    if v.get("min") is not None
                    and (v["min"] < 0 or v["max"] > 255)]
    nonfinite = [v["name"] for v in images if v.get("n_nonfinite")]
    matches_claim = (not frac and not out_of_range
                     and all(d.endswith("u1") for d in dtypes))
    # A prevalence figure drawn from ONE subject is not a cohort statement. In
    # v0.16 all 60 samples came from sub-13, so G10 FAIL described one patient.
    single_subject = len(subs) <= 1
    status = INCONCLUSIVE if single_subject else (PASS if not nonfinite else FAIL)
    detail = (f"{len(images)} intensity volume(s) measured across "
              f"{len(subs)} subject(s). dtypes={dict(dtypes)}. "
              f"{len(frac)} contain fractional values, {len(out_of_range)} fall "
              f"outside 0-255. Descriptor's uint8 0-255 claim "
              f"{'holds' if matches_claim else 'does NOT hold'} on this sample; "
              "normalisation must be chosen from these statistics, not the claim.")
    if nonfinite:
        detail += f" {len(nonfinite)} volume(s) contain non-finite values."
    if single_subject:
        detail += (f" INCONCLUSIVE: every measured volume comes from "
                   f"{subs[0] if subs else 'a single subject'}, so neither the "
                   "pass nor the failure generalises to the cohort. Re-run with "
                   "sample_per_subject set.")
    return _rec("G10", "Intensity sanity", status, detail,
                {"n_images_measured": len(images),
                 "dtype_frequency": dict(dtypes),
                 "with_fractional_values": frac[:50],
                 "outside_0_255": out_of_range[:50],
                 "with_nonfinite": nonfinite[:50],
                 "descriptor_claim_holds": bool(matches_claim),
                 "mni_archive": mni_archive,
                 "subjects_covered": subs,
                 "n_subjects_covered": len(subs),
                 "single_subject_sample": bool(single_subject),
                 "n_images_other_archives": len(other),
                 "range_summary": {
                     "min_of_mins": min(v["min"] for v in images if v["min"] is not None),
                     "max_of_maxes": max(v["max"] for v in images if v["max"] is not None)}})


def summarise(guards: list[dict]) -> dict:
    counts = Counter(g["status"] for g in guards)
    return {"passed": [g["guard"] for g in guards if g["status"] == PASS],
            "failed": [g["guard"] for g in guards if g["status"] == FAIL],
            "inconclusive": [g["guard"] for g in guards if g["status"] == INCONCLUSIVE],
            "counts": dict(counts),
            "stop_protocol_triggered": bool(counts.get(FAIL))}
