"""Stage 1 audit orchestrator — notebook section 03 (§15.3).

CPU-only. Reads canonical EBRAINS artefacts, measures what is there, runs guards
G1/G5/G7/G8/G9/G10, writes draft manifests and a QC report, and ends with a gap
report. It downloads nothing and writes nothing into the legacy folder.

Entry point:  run_stage1_audit(paths)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import (DATA_VERSION, PRIMARY_TARGET_COMPONENT,
                      PRIMARY_TARGET_MASK, target_lock)
from ..data import archives, delta_t as dt_mod, dose as dose_mod
from ..data import inventory, manifests, naming, provenance, tables, treatment
from ..qc import guards, report as report_mod
from ..utils.env import Stopwatch, git_state, hardware, peak_rss_gb, resource_card
from . import bootstrap

BIDS_ARCHIVE = "rawdata_BIDS.tar.bz2"
DERIV_ARCHIVE = "derivatives.tar.bz2"
AMBIGUOUS_ARCHIVES = ["raw_needed.tar", "dosemaps.tar"]

STOP_IMPACTS = {
    "G1": "Degenerate masks in the locked primary target: every Dice, volume and "
          "growth metric computed over them is undefined or silently zero-inflated.",
    "G5": "A quarantined or prior-split artefact is on the input path: any split "
          "derived from this run inherits tuning from the previous project.",
    "G7": "Δt is approximate: C1-C4 all condition on time, so every conditioning "
          "claim inherits an unquantified time error.",
    "G8": "MNI and raw sessions cannot be joined legally: any Δt attached to an "
          "MNI session may belong to a different exam.",
    "G9": "The surviving cohort is smaller than the nominal one: n, power and every "
          "confidence interval change.",
    "G10": "Measured intensities contradict the assumed format: normalisation chosen "
           "from the descriptor would distort every input volume.",
}
STOP_FIXES = {
    "G1": "Exclude the listed sessions from the primary cohort, report the reduced n, "
          "and do not substitute another target (§3.2).",
    "G5": "Remove the artefact from the input path and rebuild the manifest from "
          "canonical files only.",
    "G7": "Declare Δt approximate in the limitations, run the Δt sensitivity analysis, "
          "and request approval for the smallest artefact that restores exact dates.",
    "G8": "Join only through raw-mni-link.tsv; drop unmatched sessions and report them.",
    "G9": "Adopt the measured survivor count as the sample size and re-derive the "
          "power statement in §6 from it.",
    "G10": "Choose normalisation from the measured statistics and record the measured "
           "dtype and range in the preprocessing version.",
}


def _texts_for(handler, archive_name):
    return {k: v for k, v in handler.texts.items() if k.startswith(f"{archive_name}::")}


def _load_loose_table(legacy_root: Path, filename: str, collected: dict) -> str | None:
    p = Path(legacy_root) / filename
    if p.exists():
        return p.read_text(errors="replace")
    for key, text in collected.items():
        if key.split("::", 1)[-1].rsplit("/", 1)[-1] == filename:
            return text
    return None


def run_stage1_audit(paths,
                     verify_hashes: bool = True,
                     max_hash_gb: float | None = 2.0,
                     force_rescan: bool = False,
                     sample_images: int = 40,
                     max_members: int | None = None,
                     read_volumes: bool = True,
                     archives_to_scan: list | None = None,
                     sample_per_subject: int | None = None,
                     read_icor: bool = True,
                     plhm_subjects: int = 8,
                     slice_targets: list | None = None,
                     slice_reference_z: tuple | None = None,
                     export_arrays: bool = False,
                     export_sessions: set | None = None,
                     verbose: bool = True) -> dict:
    """Run the audit. Returns the report dict and writes it under the project root.

    verify_hashes / max_hash_gb: SHA-512 over a 43 GB archive is a long CPU job.
    Files above `max_hash_gb` are recorded SKIPPED_TOO_LARGE, never as verified.
    Set max_hash_gb=None to hash everything.

    read_volumes=False runs a STRUCTURAL pass: member listing, NIfTI headers and
    index tables, but no full array reads. It answers what exists, what it is
    named, what shape and dtype it has, and whether raw-mni-link.tsv and the
    acq_time columns are present -- without decompressing gigabytes of voxels.
    G1 and G10 need voxel values, so they report INCONCLUSIVE in this mode and
    are never presented as passed.

    archives_to_scan restricts which archives are read. bz2 is sequential, so
    cost tracks archive size: derivatives alone is ~90 min over Drive FUSE while
    all four is ~2.5 h. Passing ["derivatives.tar.bz2"] answers G7 (intervals),
    G8 (raw-mni-link.tsv) and the clinical table in one pass; the raw side of the
    G8 join then reports as unscanned rather than as absent.
    """
    sw = Stopwatch()
    log = print if verbose else (lambda *a, **k: None)

    # 1 — project root (§14.2). Legacy is never written to.
    root_state = bootstrap.ensure_project_root(paths)
    if root_state["status"] != "OK":
        return {"status": "FAIL_PROJECT_ROOT", "root_state": root_state}
    readme = bootstrap.write_readme(paths)
    log(f"[root] {root_state['dataset_root']}  created={len(root_state['created'])} "
        f"existing={len(root_state['already_existed'])}")

    project_root = Path(root_state["dataset_root"])
    # `from_cache` is reported in the resource card so a fast run is never
    # mistaken for a fast archive.
    index_dir = project_root / "01_DATA_FOUNDATION" / "v2_archive_index"

    # 2 — provenance firewall (§4)
    legacy = provenance.scan_legacy_root(paths.legacy_root)
    if legacy["status"] != "OK":
        return {"status": "FAIL_LEGACY_ABSENT", "legacy": legacy}
    verification = provenance.verify_canonical(
        paths.legacy_root, legacy["entries"], verify_hashes=verify_hashes,
        max_hash_bytes=None if max_hash_gb is None else int(max_hash_gb * 2**30))
    cls = verification["classification"]
    log(f"[provenance] canonical present={len(cls['canonical_present'])} "
        f"missing={cls['canonical_missing']} quarantine={len(cls['quarantine_present'])}")
    pointer_manifest = provenance.canonical_pointer_manifest(paths.legacy_root, verification)
    sw.mark("provenance")

    # 3 — archive passes. One forward pass per archive; handler output cached.
    from ..utils.persist import load_cache, save_cache
    from .clinical_table import CACHE_KEY as CLIN_KEY_EARLY
    text_h = archives.TextHandler()
    hdr_h = archives.HeaderHandler()
    vol_h = archives.VolumeStatsHandler(sample_images=sample_images,
                                        sample_per_subject=sample_per_subject,
                                        read_icor=read_icor,
                                        plhm_subjects=plhm_subjects)
    from .clinical_table import ClinicalFileHandler
    clin_h = ClinicalFileHandler()
    # The clinical files ride along in the SAME pass. bz2 is sequential, so a
    # separate pass for them would cost a second full decompression of the same
    # 43 GB for a few hundred KB of text.
    handlers = [text_h, hdr_h, clin_h] + ([vol_h] if read_volumes else [])
    export_h = None
    if export_arrays and read_volumes:
        from .array_export import ArrayExportHandler
        export_h = ArrayExportHandler(
            outdir=project_root / "01_DATA_FOUNDATION" / "v2_arrays",
            sessions=export_sessions)
        handlers.append(export_h)
        log("[export] dose maps and baseline masks -> 01_DATA_FOUNDATION/v2_arrays")
    # Slice capture for visual adjudication rides along too, for the same reason.
    slice_h = None
    if slice_targets and read_volumes:
        from .visual_check import SliceHandler
        slice_h = SliceHandler(targets=[tuple(t) for t in slice_targets],
                               reference_z=slice_reference_z)
        if slice_reference_z is None:
            log("[slices] WARNING: no reference_z given — slices come from each "
                "volume's OWN extent, which for an image is the whole head. "
                "Pass slice_reference_z=(lo, hi) from a mask containing the lesion.")
        handlers.append(slice_h)
        log(f"[slices] capturing {len(slice_h.targets)} target session(s)")
    if not read_volumes:
        log("[mode] STRUCTURAL pass: headers and tables only, no voxel reads. "
            "G1 and G10 will report INCONCLUSIVE by construction.")
    scans = {}
    # A cached member index alone is not enough: handlers produce headers and
    # volume statistics that the index does not contain. So the cache stores the
    # handler OUTPUT, and is invalidated by scan mode, member cap and sample
    # size — anything that would change what a pass actually reads.
    arc_list = archives_to_scan or [BIDS_ARCHIVE, DERIV_ARCHIVE, *AMBIGUOUS_ARCHIVES]
    arc_tag = "-".join(sorted(a.split(".")[0] for a in arc_list))
    # The cache key must encode EVERYTHING that changes what a pass reads, or a
    # v0.16 cache (no percentiles, no dose arrays, first-60 sampling) would be
    # silently reused by a v0.17 run and every new check would report on old
    # bytes. `_v17` bumps the key for the widened read scope.
    audit_cache_key = (f"audit_scan_{'full' if read_volumes else 'structural'}"
                       f"_m{max_members or 'all'}_s{sample_images}"
                       f"_ps{sample_per_subject or 'off'}"
                       f"_icor{'1' if read_icor else '0'}"
                       f"_plhm{plhm_subjects}_v17_{arc_tag}")
    cached = None if force_rescan else load_cache(project_root, audit_cache_key)
    if cached:
        text_h.texts = cached["raw"].get("texts", {})
        hdr_h.headers = cached["raw"].get("headers", {})
        vol_h.stats = cached["raw"].get("volume_stats", {})
        scans = cached.get("meta", {}).get("scans", {})
        log(f"[scan] reusing cache from {cached.get('saved_utc')}: "
            f"{len(hdr_h.headers)} header(s), {len(vol_h.stats)} volume stat(s). "
            "force_rescan=True to re-read the archives.")
    else:
        for arc in arc_list:
            p = Path(paths.legacy_root) / arc
            if not p.exists():
                scans[arc] = {"archive": arc, "status": "ABSENT", "members": 0}
                log(f"[scan] {arc}: ABSENT")
                continue
            res = archives.scan_archive(
                p, handlers,
                index_path=index_dir / f"{arc}.index.jsonl",
                force=True, max_members=max_members)
            scans[arc] = res
            log(f"[scan] {arc}: {res['members']} members in {res.get('seconds')} s")
        save_cache(project_root, CLIN_KEY_EARLY, clin_h.raw,
                   meta={"source": "audit_single_pass"}) if clin_h.raw else None
        save_cache(project_root, audit_cache_key,
                   {"texts": text_h.texts, "headers": hdr_h.headers,
                    "volume_stats": vol_h.stats},
                   meta={"scans": scans, "read_volumes": read_volumes,
                         "max_members": max_members,
                         "sample_images": sample_images,
                         "sample_per_subject": sample_per_subject,
                         "read_icor": read_icor,
                         "plhm_subjects": plhm_subjects})
        # Slices are bulky float arrays; they go to their own cache so reloading
        # the main handler output stays cheap.
        if export_h and export_h.exported:
            from .array_export import summarise as _dsum
            save_cache(project_root, f"{audit_cache_key}__exported", export_h.exported,
                       meta=_dsum(export_h.exported))
            log(f"[export] wrote {len(export_h.exported)} array(s); "
                f"dose summary: {_dsum(export_h.exported)['shape_frequency']}")
        if slice_h and slice_h.slices:
            save_cache(project_root, f"{audit_cache_key}__slices", slice_h.slices,
                       meta={"targets": sorted(map(list, slice_h.targets)),
                             "reference_z": list(slice_reference_z) if slice_reference_z else None,
                             "errors": slice_h.errors[:20]})
            log(f"[slices] captured {len(slice_h.slices)} volume(s)")
        if slice_h and slice_h.errors:
            # Handler exceptions are collected rather than raised, so without
            # this they are invisible and a capture that produced nothing looks
            # like a capture that found nothing.
            log(f"[slices] WARNING: {len(slice_h.errors)} error(s) during capture; "
                f"first: {slice_h.errors[0]}")
        if slice_h and slice_targets and not slice_h.slices:
            log("[slices] WARNING: targets were requested but NO slice was captured. "
                "Check that the target sessions exist and that reference_z is on "
                "the same grid as the volumes.")
        log(f"[scan] cached handler output under {audit_cache_key}")
    sw.mark("archive_scan")

    if archives_to_scan:
        log(f"[mode] scanning {arc_list} only; other archives report as UNSCANNED, "
            "not as absent.")
    member_names = {arc: [] for arc in scans}
    for arc in scans:
        idx = index_dir / f"{arc}.index.jsonl"
        member_names[arc] = [r.name for r in archives.load_index(idx)]

    # 4 — index tables (§3)
    all_texts = text_h.texts
    overview_txt = _load_loose_table(paths.legacy_root, "overview.tsv", all_texts)
    missing_txt = _load_loose_table(paths.legacy_root, "missing.tsv", all_texts)
    link_txt = _load_loose_table(paths.legacy_root, "raw-mni-link.tsv", all_texts)
    overview = tables.parse_overview(overview_txt) if overview_txt else {
        "status": "ABSENT", "records": [], "header": [], "n_rows": 0,
        "resolved_columns": {}}
    missing = tables.parse_missing(missing_txt) if missing_txt else {
        "status": "ABSENT", "entries": [], "index": {}, "n_rows": 0,
        "resolved_columns": {}}
    link = tables.parse_raw_mni_link(link_txt) if link_txt else {
        "status": "ABSENT", "pairs": [], "n_rows": 0, "resolved_columns": {}}
    s2r_txt = _load_loose_table(paths.legacy_root, "src-to-raw.yaml", all_texts)
    src_to_raw = tables.parse_src_to_raw(s2r_txt) if s2r_txt else {
        "status": "ABSENT", "sequences_per_session": {}}
    log(f"[src-to-raw] {src_to_raw.get('status')} "
        f"sessions={src_to_raw.get('n_sessions')} "
        f"identical_indices={src_to_raw.get('source_raw_indices_identical')}")
    log(f"[tables] overview={overview['status']} missing={missing['status']} "
        f"raw-mni-link={link['status']}")

    # 5 — naming vocabulary and file table
    file_rows = []
    for arc, names in member_names.items():
        for row in naming.build_file_table(names):
            row["archive"] = arc
            file_rows.append(row)
    vocabulary = naming.discover_vocabulary(
        [n for names in member_names.values() for n in names])
    target_resolution = naming.target_resolution_report(file_rows)
    log(f"[target] {target_resolution['status']} "
        f"CL/enhancing_t1wc files={target_resolution['n_cl_enhancing_t1wc']}")
    sw.mark("naming")

    # 6 — inventory. MNI and raw sides kept separate: never assume ses-XX aligns (G8).
    header_index = hdr_h.headers
    deriv_rows = [r for r in file_rows if r["archive"] == DERIV_ARCHIVE]
    bids_rows = [r for r in file_rows if r["archive"] == BIDS_ARCHIVE]
    mni_built = inventory.build_sessions(deriv_rows, header_index)
    raw_built = inventory.build_sessions(bids_rows, header_index)
    mni_summary = inventory.summarise(mni_built["sessions"])
    raw_summary = inventory.summarise(raw_built["sessions"])
    log(f"[inventory] MNI patients={mni_summary['n_patients']} "
        f"sessions={mni_summary['n_sessions']} | raw patients={raw_summary['n_patients']} "
        f"sessions={raw_summary['n_sessions']}")

    # Legal raw -> MNI translation, built only from raw-mni-link.tsv (G8).
    raw_to_mni = {}
    for pr in link.get("pairs", []):
        if pr.get("subject") and pr.get("raw_session") and pr.get("mni_session"):
            raw_to_mni[(pr["subject"], pr["raw_session"])] = pr["mni_session"]
    log(f"[link] raw->MNI mapping entries: {len(raw_to_mni)}")

    mni_sessions = {s: sorted(v.keys()) for s, v in mni_built["sessions"].items()}
    raw_sessions = {s: sorted(v.keys()) for s, v in raw_built["sessions"].items()}
    present_sequences = {}
    for sub, ses_map in mni_built["sessions"].items():
        for ses, node in ses_map.items():
            present_sequences[(sub, ses)] = set(node["sequences"].keys())
    sw.mark("inventory")

    # 7 — treatment (§5) and Δt (G7)
    treat = treatment.extract(overview)
    log(f"[treatment] {treat['status']} counts={treat.get('status_counts')}")
    delta = dt_mod.build(
        texts_bids=_texts_for(text_h, BIDS_ARCHIVE),
        texts_legacy={k: v for k, v in all_texts.items()
                      if not k.startswith(f"{BIDS_ARCHIVE}::")},
        overview=overview,
        raw_needed_present=(Path(paths.legacy_root) / "raw_needed.tar").exists())
    for line in delta["summary_lines"]:
        log(f"[delta_t] {line}")
    sw.mark("treatment_delta_t")

    # 7b — per-session clinical files (treatment / RANO / survival / intervals)
    from .clinical_table import build_table
    if clin_h.raw:
        clinical = build_table(clin_h.raw)          # collected in the pass above
    else:
        clin_cached = load_cache(project_root, CLIN_KEY_EARLY)
        clinical = build_table(clin_cached["raw"] if clin_cached else {})
        if clin_cached:
            log(f"[clinical] reusing cache from {clin_cached.get('saved_utc')}")
    log(f"[clinical] {clinical['n_subjects']} subject(s), {clinical['n_sessions']} "
        f"session(s); treatment={clinical['treatment_counts']}; "
        f"Δt sessions={clinical['n_sessions_with_days_from_first']}")

    # 8 — dose maps (C3 prerequisite)
    ref_shape = None
    ref_spacing = None
    if mni_summary["shape_frequency"]:
        ref_shape = mni_summary["shape_frequency"][0][0]
    if mni_summary["spacing_frequency"]:
        ref_spacing = mni_summary["spacing_frequency"][0][0]
    vol_by_path = {}
    for key, st in vol_h.stats.items():
        vol_by_path[key] = st
        vol_by_path[st["name"]] = st
    hdr_by_path = {}
    for key, h in header_index.items():
        hdr_by_path[key] = h
        hdr_by_path[h.get("name", key)] = h
    dose = dose_mod.collect(file_rows, vol_by_path, ref_shape, ref_spacing,
                            header_stats=hdr_by_path)
    from ..data.known_issues import dose_eligible_subjects, summary as issues_summary
    # Deduplicate: clinical rows are per SESSION, so passing them raw counted
    # every session as a patient.
    clinical_subjects = sorted({r["subject"] for r in clinical["rows"]})
    dose["documented_eligibility"] = dose_eligible_subjects(
        clinical_subjects or [f"sub-{i:02d}" for i in range(1, 28)])
    log(f"[dose] files={dose['n_dose_files']} patients={dose['n_patients_with_dose']} "
        f"registration={dose['registration_status']}")

    # 9 — guards
    manifest_inputs = [r.get("path") for r in verification["results"] if r.get("path")]
    g_records = [
        guards.g1_degenerate_labels(file_rows, vol_by_path),
        guards.g5_leakage_stage1(manifest_inputs, cls, mni_built["sessions"]),
        guards.g7_delta_t_provenance(delta["attempts"], delta["per_session"],
                                     clinical_table=clinical),
        guards.g8_session_correspondence(
            link, mni_sessions, raw_sessions,
            raw_side_scanned=(BIDS_ARCHIVE in arc_list)),
        guards.g9_missing_tsv(missing, mni_sessions, present_sequences,
                              raw_to_mni=raw_to_mni),
        guards.g10_intensity_sanity(vol_h.stats, mni_archive=DERIV_ARCHIVE),
    ]
    g_summary = guards.summarise(g_records)
    for g in g_records:
        log(f"[guard] {g['guard']} {g['status']}: {g['detail'][:160]}")
    sw.mark("guards")

    # 10 — draft manifests
    g1 = g_records[0]
    g9 = g_records[4]
    session_mf = manifests.session_manifest(mni_built["sessions"], treat, delta, g9)
    target_mf = manifests.target_manifest(target_resolution, g1)
    # The file table is what Phase 4 needs to know which sessions actually carry
    # the locked primary target. Persisted so pair construction never has to
    # re-read the archive, and never has to assume the mask is present.
    target_sessions = sorted(
        f"{r['subject']}/{r['session']}" for r in file_rows
        if r.get("subject") and r.get("session")
        and r.get("annotation_kind") == PRIMARY_TARGET_MASK
        and r.get("annotation_component") == PRIMARY_TARGET_COMPONENT)
    log(f"[target] {len(target_sessions)} session(s) carry the locked primary target")

    written = manifests.write_all(project_root, {
        "target_sessions": {
            "manifest": "sessions_with_primary_target",
            **target_lock(),
            "n_sessions": len(target_sessions),
            "sessions": target_sessions,
            "note": ("Phase 4 builds a pair only where BOTH ends appear here. "
                     "A pair whose target mask is absent is not a pair with a "
                     "missing label; it is not a pair."),
        },
        "canonical_pointers": pointer_manifest,
        "session_inventory_draft": session_mf,
        "primary_target_registration": target_mf,
        "naming_vocabulary": {"vocabulary": vocabulary,
                              "file_table_sample": file_rows[:200]},
    })
    log(f"[manifests] wrote {len(written)} draft manifest(s)")

    # 11 — gap report (§4.1). Nothing is fetched.
    disk = provenance.disk_report([paths.legacy_root, project_root, Path("/content"),
                                   Path.cwd()])
    gap = __import__("sailor.stage1.gap_report", fromlist=["build"]).build(
        verification, delta, target_resolution, dose, disk, g_records)
    for line in gap["lines"]:
        log(f"[gap] {line}")

    # 12 — report + completion record
    card = resource_card("03_stage1_audit", profiled=True, measured={
        "wall_seconds": sw.elapsed,
        "stage_marks": dict(sw.marks),
        "peak_rss_gb": peak_rss_gb(),
        "disk_required_gb": "0 (streaming reads; no extraction performed)",
        "safe_on_fresh_runtime": "YES (mounts Drive, reads canonical, no in-memory deps)",
        "checkpoint_resume": ("handler output cached under "
                              "01_DATA_FOUNDATION/v2_scan_cache; re-runs skip the "
                              "archive pass unless force_rescan=True"),
        "used_cache": bool(cached),
        "measured_against": "real archives" if any(
            s.get("members") for s in scans.values()) else "NOTHING — no archive was read",
    })

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_version": DATA_VERSION,
        "paths": paths.to_dict(),
        "root_state": root_state,
        "readme": readme,
        "target_lock": target_lock(),
        "target_resolution": target_resolution,
        "provenance": {"legacy": legacy, "verification": verification,
                       "pointer_manifest": pointer_manifest},
        "archive_scans": scans,
        "naming_vocabulary": vocabulary,
        "inventory": mni_summary,
        "inventory_raw_side": raw_summary,
        "treatment": treat,
        "treatment_authority": {
            "source": "derivatives/mni2009c-n-s/sub-XX/ses-YY/treatment.txt",
            "counts": clinical["treatment_counts"],
            "n_observed": clinical["n_observed"],
            "note": ("The loose overview.tsv carries no treatment column; the "
                     "per-session files are the only source. The TREATMENT (§5) "
                     "section below reports the index-table lookup, which is "
                     "expected to find nothing."),
        },
        "clinical": {k: v for k, v in clinical.items()
                     if k not in ("history_text", "structure_text")},
        "history_text": clinical.get("history_text"),
        "known_issues": issues_summary(clinical_subjects or None),
        "src_to_raw": src_to_raw,
        "delta_t": delta,
        "dose": dose,
        "guards": {"records": g_records, "summary": g_summary},
        "manifests": written,
        "gap_report": gap,
        "resource_card": card,
        "scan_mode": "FULL" if read_volumes else "STRUCTURAL",
        "archives_scanned": arc_list,
        "archives_unscanned": [a for a in
                               [BIDS_ARCHIVE, DERIV_ARCHIVE, *AMBIGUOUS_ARCHIVES]
                               if a not in arc_list],
        "max_members": max_members,
        "bounded_pass": max_members is not None,
        "git": git_state(Path(paths.code_root)),
        "hardware": hardware(),
        "stop_impacts": STOP_IMPACTS,
        "stop_fixes": STOP_FIXES,
        "downloads_performed": 0,
    }
    paths_written = report_mod.write(report, project_root / "06_QC_REPORTS")
    report["report_paths"] = paths_written

    completion = bootstrap.write_completion_record(
        paths, section=3, stage=1,
        status="complete_with_failures" if g_summary["stop_protocol_triggered"] else "complete",
        payload={"owner": "stage1_audit",
                 "guards_passed": g_summary["passed"],
                 "guards_failed": g_summary["failed"],
                 "guards_inconclusive": g_summary["inconclusive"],
                 "n_patients": mni_summary["n_patients"],
                 "n_sessions": mni_summary["n_sessions"],
                 "artefacts": {**written, **paths_written}})
    report["completion_record"] = completion
    log(f"[report] {paths_written['latest_text']}")
    log(f"[completion] {completion}")
    return report


def print_report(report: dict) -> None:
    print(report_mod.render_text(report))


#: Guards that need voxel values and therefore report INCONCLUSIVE in a
#: structural pass. Fixed here so callers cannot quietly widen the set.
VOXEL_GUARDS = ("G1", "G10")

#: A guard has spoken on measurement when it returns one of these. FAIL counts:
#: GATE-0 asks for a measured verdict, not a clean one, which is why its
#: criteria name INCONCLUSIVE as the disqualifier rather than FAIL.
MEASURED_STATUSES = ("PASS", "FAIL")


def _guard_records(audit: dict) -> dict:
    recs = (audit.get("guards") or {}).get("records") or []
    return {r.get("guard"): r for r in recs if isinstance(r, dict)}


def _measured_count(rec: dict) -> int:
    """How many things the guard actually read. 0 means it read nothing."""
    ev = rec.get("evidence") or {}
    if "n_images_measured" in ev:                      # G10 shape
        return int(ev.get("n_images_measured") or 0)
    primary = ev.get("primary") or {}                  # G1 shape
    return int(primary.get("n_measured") or 0)


def latest_measured_audit(project_root,
                          require_guards: tuple = VOXEL_GUARDS,
                          subdir: str = "06_QC_REPORTS") -> dict:
    """Newest stage-1 audit in which `require_guards` all ran on voxels.

    The twin of `persist.latest_full_pass` (defect 24), one layer up. That
    defect was a cache pointer that resolved by name; this is an artefact
    pointer that resolves by RECENCY, which is just as wrong when the newest
    pass is the least complete one.

    `v2_stage1_audit_latest.json` names the most recent audit. A structural pass
    (read_volumes=False) is a perfectly valid audit and is often the most recent
    one, but G1 and G10 are INCONCLUSIVE in it BY CONSTRUCTION -- see
    run_stage1_audit. A caller asking "did G1 pass?" and reading _latest gets
    INCONCLUSIVE and may conclude the data is unmeasured when a full pass
    measuring 240 masks sits two files away.

    Returns the deciding audit, or verdict UNRESOLVED naming what was missing.
    Never returns a partial answer that reads like a finding.
    """
    q = Path(project_root) / subdir
    cands = sorted(q.glob("v2_stage1_audit_2026*.json"), reverse=True)
    considered, chosen = [], None

    for f in cands:
        try:
            a = json.loads(f.read_text())
        except (OSError, ValueError) as e:
            considered.append({"file": f.name, "usable": False,
                               "why": f"unreadable: {type(e).__name__}"})
            continue
        recs = _guard_records(a)
        per = {g: {"status": (recs.get(g) or {}).get("status", "ABSENT"),
                   "n_measured": _measured_count(recs.get(g) or {})}
               for g in require_guards}
        ok = all(v["status"] in MEASURED_STATUSES and v["n_measured"] > 0
                 for v in per.values())
        considered.append({"file": f.name, "usable": ok,
                           "generated_utc": a.get("generated_utc"),
                           "guards": per})
        if ok and chosen is None:
            chosen = {"path": str(f), "file": f.name,
                      "generated_utc": a.get("generated_utc"), "guards": per}

    if chosen is None:
        missing = sorted(require_guards)
        return {"verdict": "UNRESOLVED",
                "require_guards": list(require_guards),
                "detail": (f"No audit in {subdir} has {missing} on measurement. "
                           "Run run_stage1_audit(read_volumes=True); a "
                           "structural pass cannot decide these."),
                "considered": considered}

    return {"verdict": "RESOLVED", "require_guards": list(require_guards),
            "deciding_audit": chosen,
            "note": ("Selected by completeness, not recency. The newest audit "
                     "may be a structural pass in which these guards are "
                     "INCONCLUSIVE by construction."),
            "considered": considered}
