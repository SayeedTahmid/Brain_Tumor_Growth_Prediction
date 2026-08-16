"""QC report rendering (§15.5: failed guards at the top, never suppressed)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

RULE = "=" * 78
THIN = "-" * 78


def _fmt(v):
    if v is None:
        return "UNMEASURED"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def render_text(report: dict) -> str:
    L = []
    a = L.append
    a(RULE)
    a("SAILOR STAGE 1 — PROVENANCE & DATASET AUDIT")
    a(f"generated: {report['generated_utc']}   data_version: {report['data_version']}")
    a(f"project_root: {report['paths']['dataset_root']}")
    a(f"legacy_root (read-only): {report['paths']['legacy_root']}")
    a(RULE)

    g = report["guards"]
    a("")
    a("GUARDS")
    a(THIN)
    for rec in g["records"]:
        if rec["status"] == "FAIL":
            a(f"  [FAIL] {rec['guard']} {rec['title']}: {rec['detail']}")
    for rec in g["records"]:
        if rec["status"] == "INCONCLUSIVE":
            a(f"  [INCONCLUSIVE] {rec['guard']} {rec['title']}: {rec['detail']}")
    for rec in g["records"]:
        if rec["status"] == "PASS":
            a(f"  [PASS] {rec['guard']} {rec['title']}: {rec['detail']}")
    if g["summary"]["stop_protocol_triggered"]:
        a("")
        a("  STOP PROTOCOL TRIGGERED — see §2.3 block below.")

    a("")
    a("TARGET LOCK (§3.2)")
    a(THIN)
    tl = report["target_lock"]
    a(f"  primary:   {tl['primary_target_mask']} / {tl['primary_target_component']}")
    a(f"  secondary: {tl['secondary_target_mask']}   sensitivity: {tl['sensitivity_targets']}")
    tr = report["target_resolution"]
    a(f"  resolution on disk: {tr['status']}")
    a(f"  files: CL={tr['n_cl_files']} (enhancing_t1wc={tr['n_cl_enhancing_t1wc']}, "
      f"t2wflair={tr['n_cl_t2wflair']}, component-unknown={tr['n_cl_component_unknown']}), "
      f"ONCO={tr['n_onco_files']}, unresolved mask-like={tr['n_unresolved_masklike']}")

    a("")
    a("PROVENANCE (§4)")
    a(THIN)
    cls = report["provenance"]["verification"]["classification"]
    a(f"  canonical present: {len(cls['canonical_present'])} -> {cls['canonical_present']}")
    a(f"  canonical missing: {cls['canonical_missing']}")
    a(f"  expected-absent confirmed absent: {cls['expected_absent_confirmed']}")
    a(f"  quarantine present: {len(cls['quarantine_present'])} entries")
    a(f"  ambiguous (verify origin): {cls['ambiguous_present']}")
    a(f"  unclassified entries: {cls['unclassified']}")
    for r in report["provenance"]["verification"]["results"]:
        a(f"    {r['status']:<20} {r['file']}"
          + (f"  ({r['size_bytes']/2**20:.1f} MiB)" if r.get("size_bytes") else ""))

    a("")
    a("COHORT (measured)")
    a(THIN)
    inv = report["inventory"]
    a(f"  patients with files: {_fmt(inv.get('n_patients'))}")
    a(f"  sessions with files: {_fmt(inv.get('n_sessions'))}")
    a(f"  sequence frequency: {inv.get('sequence_frequency')}")
    a(f"  shape frequency (top): {inv.get('shape_frequency')}")
    a(f"  spacing frequency (top): {inv.get('spacing_frequency')}")
    a(f"  dtype frequency: {inv.get('dtype_frequency')}")

    a("")
    a("TREATMENT (§5)")
    a(THIN)
    c = report.get("clinical") or {}
    if c.get("treatment_counts"):
        a(f"  SOURCE OF RECORD: per-session treatment.txt in derivatives")
        a(f"  {c['n_subjects']} subject(s), {c['n_sessions']} session(s)")
        a(f"  counts: {c['treatment_counts']}   observed: {c.get('n_observed')}")
        a(f"  RANO available: {c.get('rano_available')}   "
          f"survival: {c.get('survival_available')}")
        a(f"  Δt: {c.get('n_subjects_with_intervals')} subject(s) with an intervals "
          f"file; {c.get('n_sessions_with_days_from_first')} session(s) with "
          "days_from_first")
        if c.get("interval_value_counts"):
            a(f"  interval value counts: {c['interval_value_counts']}")
        if c.get("interval_problems"):
            a(f"  interval problems: {c['interval_problems'][:6]}")
        a("")
        a("  (the loose overview.tsv below is a session inventory, NOT the "
          "treatment source)")
    t = report["treatment"]
    a(f"  source status: {t.get('status')}   resolved column: {t.get('resolved_column')}")
    if t.get("status") == "NO_TREATMENT_COLUMN_RESOLVED":
        a("  NO treatment variable exists in the parsed index tables. This is not "
          "'all unknown' -- the variable was never found. C2/C4 and the §5 confound "
          "measurement cannot run until its source is located.")
    a(f"  status counts: {t.get('status_counts')}")
    a(f"  timepoints: {_fmt(t.get('n_timepoints'))}   missing fraction: {_fmt(t.get('missing_fraction'))}")
    if t.get("unrecognised_tokens"):
        a(f"  unrecognised tokens (not coerced): {t['unrecognised_tokens']}")

    if report.get("known_issues"):
        k = report["known_issues"]
        a("")
        a("DOCUMENTED ISSUES (history.txt — authors' own record)")
        a(THIN)
        a(f"  primary cohort: {k['n_patients_primary_cohort']} patient(s) "
          f"(no manual CL masks for {k['subjects_without_primary_target']})")
        a(f"  sessions excluded: {k['sessions_excluded']}")
        a(f"  FUTURE-INFORMATION LEAKAGE: {k['sessions_with_future_leakage']}")
        a("    these ses-01 labels were derived from ses-02, so a ses-01 -> ses-02")
        a("    pair is partly trained on its own answer; exclusion is mandatory")
        a(f"  Δt interpolated for {k['n_delta_t_estimated_subjects']} subject(s): "
          f"{k['delta_t_subjects_with_estimated_intervals']}")
        a(f"  CL = {k['annotation_provenance']['CL']['meaning']}")
        a(f"  ONCO = {k['annotation_provenance']['ONCO']['meaning']}")
        a(f"  intensity: {k['intensity_claim']['reading']}")

    a("")
    a("DOSE MAPS (prerequisite for C3)")
    a(THIN)
    d = report["dose"]
    a(f"  dose files: {d['n_dose_files']}   patients with dose: {d['n_patients_with_dose']}")
    a(f"  registration: {d['registration_status']}")
    a(f"  shapes: {d['shape_frequency']}   spacings: {d['spacing_frequency']}")
    a(f"  TMZ representation: {d['tmz_representation']['decision']}")
    if d.get("documented_eligibility"):
        de = d["documented_eligibility"]
        a(f"  documented eligibility: {de['n_eligible']} patient(s); "
          f"blocked {de['blocked']} ({de['reasons']})")

    if report.get("src_to_raw"):
        st = report["src_to_raw"]
        a("")
        a("src-to-raw.yaml (conversion plan)")
        a(THIN)
        a(f"  {st.get('n_conversion_records')} conversion record(s); "
          f"{st.get('n_subjects')} subject(s), {st.get('n_sessions')} source session(s)")
        a(f"  source and raw session indices identical: "
          f"{st.get('source_raw_indices_identical')}")
        a(f"  {st.get('caveat')}")

    a("")
    a("Δt (G7)")
    a(THIN)
    for line in report["delta_t"]["summary_lines"]:
        a(f"  {line}")

    a("")
    a("GAP REPORT (§4.1 — no file was fetched)")
    a(THIN)
    for line in report["gap_report"]["lines"]:
        a(f"  {line}")

    a("")
    a("RESOURCE CARD (§15.7)")
    a(THIN)
    for k, v in report["resource_card"].items():
        a(f"  {k}: {v}")

    if g["summary"]["stop_protocol_triggered"]:
        a("")
        a(RULE)
        for rec in g["records"]:
            if rec["status"] == "FAIL":
                a(f"PROBLEM:         {rec['guard']} — {rec['detail']}")
                a(f"IMPACT:          {report['stop_impacts'].get(rec['guard'], 'UNSPECIFIED')}")
                a(f"RECOMMENDED FIX: {report['stop_fixes'].get(rec['guard'], 'UNSPECIFIED')}")
                a(THIN)
        a(RULE)
    a("")
    return "\n".join(L)


def write(report: dict, qc_dir: Path, prefix: str = "v2_") -> dict:
    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    j = qc_dir / f"{prefix}stage1_audit_{stamp}.json"
    t = qc_dir / f"{prefix}stage1_audit_{stamp}.txt"
    latest_j = qc_dir / f"{prefix}stage1_audit_latest.json"
    latest_t = qc_dir / f"{prefix}stage1_audit_latest.txt"
    j.write_text(json.dumps(report, indent=2, default=str))
    text = render_text(report)
    t.write_text(text)
    latest_j.write_text(json.dumps(report, indent=2, default=str))
    latest_t.write_text(text)
    return {"json": str(j), "text": str(t),
            "latest_json": str(latest_j), "latest_text": str(latest_t)}
