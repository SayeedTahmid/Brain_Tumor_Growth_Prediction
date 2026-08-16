"""Locate the clinical variables (§3) — treatment status, Δt, RANO, age/sex/OS.

`overview.tsv` turned out to be a two-column session inventory, so the clinical
variables the descriptor promises are not in the loose index files. They must
live inside one of the archives. Without them, C2 and C4 cannot run and the §5
confound cannot be measured at all, so finding them is the single highest-value
thing a scan can do right now.

This is a cheap targeted pass: it matches only small text members whose names
look like they could carry clinical data, reads them in full, and reports which
descriptor variables each one appears to contain. It reads no voxel data.

It reports candidates. It does not decide that a file is the source of a
variable — that is a judgement to make from the printed contents.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..data.archives import MemberHandler, scan_archive
from ..data.tables import read_delimited

# Names worth opening. Deliberately broad: a missed clinical file costs another
# multi-hour pass, while a false positive costs a few kilobytes.
CANDIDATE_NAME = re.compile(
    r"(participants|sessions|scans|clinical|treatment|therapy|demograph|"
    r"meta[-_]?data|history|structure|rano|response|survival|dose|overview|"
    r"phenotype|subject)", re.IGNORECASE)

TEXTY = (".tsv", ".csv", ".txt", ".json", ".yaml", ".yml", ".xlsx")

# What each descriptor variable would look like in a header or body.
SIGNATURES = {
    "treatment_status": re.compile(r"\b(crt|tmz|temozolomid|chemorad|treatment|therapy)\b", re.I),
    "acquisition_time": re.compile(r"\b(acq_time|acquisition|studydate|exam_?date|scan_?date)\b", re.I),
    "days_between": re.compile(r"\b(days?|interval|delta|dt|weeks?)\b", re.I),
    "rano": re.compile(r"\brano\b", re.I),
    "age": re.compile(r"\bage\b", re.I),
    "sex": re.compile(r"\b(sex|gender)\b", re.I),
    "survival": re.compile(r"\b(survival|os_months|overall_survival|death|deceased)\b", re.I),
    "dose": re.compile(r"\b(dose|gy|fraction)\b", re.I),
}


class ClinicalHandler(MemberHandler):
    name = "clinical"

    def __init__(self, max_bytes: int = 8 << 20):
        self.max_bytes = max_bytes
        self.found: dict[str, dict] = {}
        self.errors: list[dict] = []

    def match(self, name: str, size: int) -> bool:
        low = name.lower()
        return (low.endswith(TEXTY) and size <= self.max_bytes
                and bool(CANDIDATE_NAME.search(low)))

    def handle(self, archive, name, size, fh):
        raw = fh.read()
        if name.lower().endswith(".xlsx"):
            self.found[f"{archive}::{name}"] = {
                "archive": archive, "name": name, "size_bytes": size,
                "kind": "xlsx", "signatures": ["BINARY — open separately"],
                "header": None, "n_rows": None, "preview": []}
            return
        text = raw.decode("utf-8", "replace")
        hits = [k for k, rx in SIGNATURES.items() if rx.search(text[:200_000])]
        header, rows = ([], [])
        if name.lower().endswith((".tsv", ".csv")):
            header, rows = read_delimited(text)
        self.found[f"{archive}::{name}"] = {
            "archive": archive, "name": name, "size_bytes": size,
            "kind": Path(name).suffix.lstrip("."),
            "signatures": hits,
            "header": header or None,
            "n_rows": len(rows) if rows else None,
            "preview": text.splitlines()[:12],
        }

    def result(self):
        return {"n_found": len(self.found), "errors": self.errors}


def locate(paths, archives_to_scan=None, max_members=None,
           verbose: bool = True, persist: bool = True) -> dict:
    """Scan archives for clinical-variable candidates. No voxel reads."""
    log = print if verbose else (lambda *a, **k: None)
    legacy = Path(paths.legacy_root)
    archives_to_scan = archives_to_scan or [
        "rawdata_BIDS.tar.bz2", "derivatives.tar.bz2", "raw_needed.tar"]

    h = ClinicalHandler()
    scans = {}
    for arc in archives_to_scan:
        p = legacy / arc
        if not p.exists():
            scans[arc] = {"status": "ABSENT"}
            log(f"[clinical] {arc}: ABSENT")
            continue
        res = scan_archive(p, [h], index_path=None, force=True,
                           max_members=max_members)
        scans[arc] = res
        log(f"[clinical] {arc}: {res['members']} members in {res.get('seconds')} s")

    by_variable: dict[str, list[str]] = {k: [] for k in SIGNATURES}
    for key, rec in h.found.items():
        for sig in rec["signatures"]:
            by_variable[sig].append(rec["name"])

    line = "-" * 78
    log(line)
    log(f"CLINICAL VARIABLE CANDIDATES — {len(h.found)} file(s) opened")
    log(line)
    for key, rec in sorted(h.found.items()):
        log(f"  {rec['name']}")
        log(f"    size={rec['size_bytes']}B  rows={rec['n_rows']}  "
            f"signatures={rec['signatures'] or 'none'}")
        if rec["header"]:
            log(f"    header: {rec['header']}")
        for ln in rec["preview"][:4]:
            log(f"      | {ln[:100]}")
    log(line)
    log("COVERAGE BY DESCRIPTOR VARIABLE")
    for var, files in by_variable.items():
        mark = "FOUND   " if files else "NOT FOUND"
        log(f"  {mark} {var:<20} {files[:3] if files else ''}")
    log(line)
    missing_vars = [v for v, f in by_variable.items() if not f]
    if missing_vars:
        log("Variables with no candidate file are unavailable from what was "
            "scanned. Treatment status and Δt in particular gate C2/C4 and the "
            "§5 confound measurement; if they stay unfound after a full pass, "
            "that is a STOP-protocol finding, not a reason to substitute a proxy.")
    result = {"files": h.found, "by_variable": by_variable,
              "variables_not_found": missing_vars, "scans": scans}
    if persist:
        from ..utils.persist import save_artefact
        result["artefact"] = save_artefact(paths.dataset_root, "06_QC_REPORTS",
                                           "clinical_variable_locations", result)
        log(f"  written: {result['artefact']['latest']}")
    return result
