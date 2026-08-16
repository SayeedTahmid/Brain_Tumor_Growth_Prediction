"""Per-session clinical files (§3, §5).

The clinical variables are not a table. They are one tiny file per subject-session
scattered through `derivatives.tar.bz2`:

    derivatives/mni2009c-n-s/sub-XX/ses-YY/treatment.txt        CRT | TMZ | no | unknown
    derivatives/mni2009c-n-s/sub-XX/ses-YY/RANO.txt             integer response class
    derivatives/mni2009c-n-s/sub-XX/overall-survival-months.txt float, per patient
    derivatives/mni2009c-n-s/history.txt                        free-text descriptor
    derivatives/mni2009c-n-s/overview.tsv                       MNI session inventory
    derivatives/mni2009c-n-s/structure.txt                      sessions per subject

This module turns those into one table and nothing more. It does not impute, does
not order sessions by anything other than the index printed in the path, and does
not convert `unknown` into a category (§5).

Duplicate copies of the same logical file appear in `raw_needed.tar` under
`./`-prefixed paths. Those are user-repackaged and unverifiable against
`SHA512.txt`, so canonical archives win and every disagreement is reported rather
than silently resolved.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

from ..data.archives import MemberHandler, scan_archive
from ..data.treatment import canonicalise

SUB_RE = re.compile(r"(sub-[A-Za-z0-9]+)")
SES_RE = re.compile(r"(ses-[A-Za-z0-9]+)")

# Canonical archive: values read from anywhere else are cross-checks, not inputs.
CANONICAL_ARCHIVE = "derivatives.tar.bz2"

TREATMENT_FILE = re.compile(r"/treatment\.txt$", re.I)
RANO_FILE = re.compile(r"/RANO\.txt$", re.I)
SURVIVAL_FILE = re.compile(r"overall-survival-months\.txt$", re.I)
AGE_FILE = re.compile(r"age[-_]years?\.txt$", re.I)
HISTORY_FILE = re.compile(r"/history\.txt$", re.I)
STRUCTURE_FILE = re.compile(r"/structure\.txt$", re.I)
OVERVIEW_FILE = re.compile(r"/overview\.tsv$", re.I)
# The Δt source. Named in history.txt; my clinical-name matcher did not match it,
# which is why the first locator pass reported acquisition_time NOT FOUND.
# history.txt spells this "intevals_days.txt" twice, so the on-disk name may
# carry the typo. Match any subject-level *days*.txt rather than one spelling;
# a missed Δt source costs another 90-minute pass to discover.
INTERVALS_FILE = re.compile(r"(inte?r?vals?|interval)[_-]?days?\.txt$", re.I)
INTERVALS_FALLBACK = re.compile(r"days?\.txt$", re.I)

WANTED = [TREATMENT_FILE, RANO_FILE, SURVIVAL_FILE, HISTORY_FILE,
          STRUCTURE_FILE, OVERVIEW_FILE, INTERVALS_FILE, AGE_FILE]


class ClinicalFileHandler(MemberHandler):
    """Reads only the tiny clinical files. Whole set is a few hundred KB."""

    name = "clinical_files"

    def __init__(self, max_bytes: int = 1 << 20):
        self.max_bytes = max_bytes
        self.raw: dict[str, str] = {}
        self.errors: list[dict] = []

    def match(self, name: str, size: int) -> bool:
        if size > self.max_bytes:
            return False
        return (any(rx.search(name) for rx in WANTED)
                or bool(INTERVALS_FALLBACK.search(name)))

    def handle(self, archive, name, size, fh):
        self.raw[f"{archive}::{name}"] = fh.read().decode("utf-8", "replace")

    def result(self):
        return {"n_files": len(self.raw), "errors": self.errors}


def _sub_ses(path: str) -> tuple[str | None, str | None]:
    s, e = SUB_RE.search(path), SES_RE.search(path)
    return (s.group(1) if s else None), (e.group(1) if e else None)


def _ordinal(session: str | None) -> int | None:
    if not session:
        return None
    m = re.search(r"(\d+)$", session)
    return int(m.group(1)) if m else None


def build_table(raw: dict[str, str]) -> dict:
    """Assemble subject/session rows from the collected file contents."""
    treatment: dict[tuple[str, str], dict] = {}
    rano: dict[tuple[str, str], dict] = {}
    survival: dict[str, dict] = {}
    conflicts: list[dict] = []
    non_canonical_only: list[str] = []
    age: dict[str, dict] = {}
    intervals: dict[str, dict] = {}
    history_text = None
    structure_text = None
    mni_overview_rows = None

    def record(store, key, value, archive, path, label):
        canon = archive == CANONICAL_ARCHIVE
        if key not in store:
            store[key] = {"value": value, "archive": archive, "path": path}
            if not canon:
                non_canonical_only.append(path)
            return
        prev = store[key]
        if prev["value"] != value:
            conflicts.append({"field": label, "key": "/".join(x for x in key if x)
                              if isinstance(key, tuple) else key,
                              "canonical_value": prev["value"] if prev["archive"] == CANONICAL_ARCHIVE else value,
                              "other_value": value if prev["archive"] == CANONICAL_ARCHIVE else prev["value"],
                              "paths": [prev["path"], path]})
        if canon and prev["archive"] != CANONICAL_ARCHIVE:
            store[key] = {"value": value, "archive": archive, "path": path}
            if prev["path"] in non_canonical_only:
                non_canonical_only.remove(prev["path"])

    for key, text in raw.items():
        archive, path = key.split("::", 1)
        body = text.strip()
        sub, ses = _sub_ses(path)
        if HISTORY_FILE.search(path) and archive == CANONICAL_ARCHIVE:
            history_text = text
        elif STRUCTURE_FILE.search(path) and archive == CANONICAL_ARCHIVE:
            structure_text = text
        elif OVERVIEW_FILE.search(path) and archive == CANONICAL_ARCHIVE:
            mni_overview_rows = [l for l in text.splitlines()[1:] if l.strip()]
        elif TREATMENT_FILE.search(path) and sub and ses:
            record(treatment, (sub, ses), body, archive, path, "treatment")
        elif RANO_FILE.search(path) and sub and ses:
            record(rano, (sub, ses), body, archive, path, "RANO")
        elif SURVIVAL_FILE.search(path) and sub:
            record(survival, sub, body, archive, path, "overall_survival_months")
        elif AGE_FILE.search(path) and sub:
            record(age, sub, body, archive, path, "age_years")
        elif (INTERVALS_FILE.search(path) or INTERVALS_FALLBACK.search(path)) and sub:
            record(intervals, sub, body, archive, path, "intervals_days")

    # Parse intervals into per-subject day offsets. The file lists intervals
    # BETWEEN consecutive examinations, so n_intervals should be n_sessions - 1;
    # any mismatch is reported rather than padded or truncated.
    from .. data.known_issues import delta_t_flag
    interval_days: dict[str, list[float]] = {}
    interval_problems: list[dict] = []
    for sub, rec in intervals.items():
        vals = []
        for tok in re.split(r"[\s,;]+", rec["value"].strip()):
            if not tok:
                continue
            try:
                vals.append(float(tok))
            except ValueError:
                interval_problems.append({"subject": sub, "unparsed_token": tok})
        interval_days[sub] = vals

    rows = []
    keys = sorted(set(treatment) | set(rano),
                  key=lambda k: (k[0], _ordinal(k[1]) or 0))
    cum: dict[str, float] = {}
    for sub, ses in keys:
        t_raw = treatment.get((sub, ses), {}).get("value")
        status, observed, token = canonicalise(t_raw)
        r_raw = rango = rano.get((sub, ses), {}).get("value")
        try:
            rano_val = int(str(r_raw).strip()) if r_raw not in (None, "") else None
        except ValueError:
            rano_val = None
        age_raw = age.get(sub, {}).get("value")
        try:
            age_val = float(age_raw) if age_raw not in (None, "") else None
        except ValueError:
            age_val = None
        surv = survival.get(sub, {}).get("value")
        try:
            surv_val = float(surv) if surv not in (None, "") else None
        except ValueError:
            surv_val = None
        ordinal = _ordinal(ses)
        ivals = interval_days.get(sub)
        days_from_prev = None
        if ivals and ordinal and 2 <= ordinal <= len(ivals) + 1:
            days_from_prev = ivals[ordinal - 2]
        if ordinal == 1:
            cum[sub] = 0.0
        elif days_from_prev is not None and sub in cum:
            cum[sub] = cum[sub] + days_from_prev
        days_from_first = cum.get(sub) if (ordinal == 1 or days_from_prev is not None) else None
        flag = delta_t_flag(sub)
        rows.append({
            "subject": sub,
            "session": ses,
            "session_ordinal": ordinal,
            "days_from_prev": days_from_prev,
            "days_from_first": days_from_first,
            "weeks_from_first": (round(days_from_first / 7.0, 4)
                                 if days_from_first is not None else None),
            "delta_t_kind": flag["kind"],
            "delta_t_estimation_note": flag["interval_estimation"],
            "treatment_status": status,
            "treatment_observed": observed,
            "treatment_missing_indicator": int(not observed),
            "treatment_raw_token": token,
            "rano": rano_val,
            "rano_raw": r_raw,
            "age_years": age_val,
            "overall_survival_months": surv_val,
            "treatment_source_path": treatment.get((sub, ses), {}).get("path"),
        })

    # Interval-count consistency, per subject.
    sess_per_sub = Counter(r["subject"] for r in rows)
    for sub, n_ses in sess_per_sub.items():
        n_iv = len(interval_days.get(sub, []))
        if sub in interval_days and n_iv != n_ses - 1:
            interval_problems.append({"subject": sub, "n_sessions": n_ses,
                                      "n_intervals": n_iv,
                                      "expected_intervals": n_ses - 1})

    counts = Counter(r["treatment_status"] or "MISSING" for r in rows)
    per_ordinal: dict[int, Counter] = defaultdict(Counter)
    for r in rows:
        if r["session_ordinal"]:
            per_ordinal[r["session_ordinal"]][r["treatment_status"] or "MISSING"] += 1

    subjects = sorted({r["subject"] for r in rows})
    return {
        "rows": rows,
        "n_subjects": len(subjects),
        "n_sessions": len(rows),
        "subjects": subjects,
        "sessions_per_subject": {s: sum(1 for r in rows if r["subject"] == s)
                                 for s in subjects},
        "treatment_counts": dict(counts),
        "n_observed": sum(1 for r in rows if r["treatment_observed"]),
        "status_by_session_ordinal": {k: dict(v) for k, v in sorted(per_ordinal.items())},
        "rano_available": sum(1 for r in rows if r["rano"] is not None),
        "survival_available": len(survival),
        "age_available": len(age),
        "n_subjects_with_intervals": len(interval_days),
        "interval_source_paths": {s: r["path"] for s, r in intervals.items()},
        "interval_value_counts": {s: len(v) for s, v in interval_days.items()},
        "n_sessions_with_days_from_first": sum(
            1 for r in rows if r["days_from_first"] is not None),
        "interval_problems": interval_problems,
        "delta_t_provenance": ("DOCUMENTED_APPROXIMATE — history.txt states the "
                               "intervals were manually extracted and partly "
                               "estimated; several subjects have interpolated "
                               "intervals (see known_issues.DELTA_T_ESTIMATED)."),
        "history_text": history_text,
        "structure_text": structure_text,
        "n_mni_overview_rows": len(mni_overview_rows) if mni_overview_rows else None,
        "conflicts_between_archives": conflicts,
        "values_only_in_non_canonical_archive": non_canonical_only[:50],
        "policy": ("`unknown` is status=None with missing_indicator=1, never a class. "
                   "Canonical archive wins; disagreements are reported, not merged."),
    }


CACHE_KEY = "clinical_files"


def save_existing(paths, table: dict, verbose: bool = True) -> dict:
    """Persist a clinical table that is already in memory.

    For the case where a long pass has completed in a notebook and the result
    would otherwise be lost to a kernel restart. It writes the derived artefacts
    only: the raw scanned bytes are not recoverable from a built table, so this
    does NOT seed the scan cache and a later re-parse still needs a real pass.
    That limitation is stated in the artefact rather than left implicit.
    """
    from ..utils.persist import save_artefact, save_table_csv, save_text
    log = print if verbose else (lambda *a, **k: None)
    root = Path(paths.dataset_root)
    payload = {k: v for k, v in table.items()
               if k not in ("history_text", "structure_text")}
    payload["provenance_note"] = (
        "Written from an in-memory table via save_existing(). Derived artefacts "
        "only; the scan cache was NOT seeded, so re-parsing requires a real pass.")
    written = {"json": save_artefact(root, "01_DATA_FOUNDATION",
                                     "clinical_table", payload),
               "csv": save_table_csv(root, "01_DATA_FOUNDATION",
                                     "clinical_table", table.get("rows", []))}
    if table.get("history_text"):
        written["history_txt"] = save_text(root, "01_DATA_FOUNDATION",
                                           "history.txt", table["history_text"])
    if table.get("structure_text"):
        written["structure_txt"] = save_text(root, "01_DATA_FOUNDATION",
                                             "structure.txt", table["structure_text"])
    for k, v in written.items():
        log(f"  {k}: {v['latest'] if isinstance(v, dict) else v}")
    return written


def collect(paths, archives=None, verbose: bool = True,
            use_cache: bool = True, force_rescan: bool = False) -> dict:
    """Scan for the clinical files and build the table. No voxel reads.

    The scanned bytes are cached under the project root and the derived table is
    written as an artefact, so a kernel restart costs nothing and re-parsing does
    not require another ~90-minute pass over the archive.
    """
    from ..utils.persist import (load_cache, save_artefact, save_cache,
                                 save_table_csv)
    log = print if verbose else (lambda *a, **k: None)
    legacy = Path(paths.legacy_root)
    archives = archives or [CANONICAL_ARCHIVE]
    project_root = Path(paths.dataset_root)

    raw, scans, from_cache = {}, {}, False
    cached = None if force_rescan else (load_cache(project_root, CACHE_KEY)
                                        if use_cache else None)
    if cached and cached.get("raw"):
        raw = cached["raw"]
        scans = cached.get("meta", {}).get("scans", {})
        from_cache = True
        log(f"[clinical-files] reusing cache from {cached.get('saved_utc')} "
            f"({len(raw)} file(s)) — pass skipped. force_rescan=True to re-read.")
    else:
        h = ClinicalFileHandler()
        for arc in archives:
            p = legacy / arc
            if not p.exists():
                scans[arc] = {"status": "ABSENT"}
                log(f"[clinical-files] {arc}: ABSENT")
                continue
            log(f"[clinical-files] {arc}: streaming (this is the slow pass) ...")
            res = scan_archive(p, [h], index_path=None, force=True)
            scans[arc] = res
            log(f"[clinical-files] {arc}: {res['members']} members in "
                f"{res.get('seconds')} s")
        raw = h.raw
        if use_cache and raw:
            cp = save_cache(project_root, CACHE_KEY, raw,
                            meta={"scans": scans, "archives": archives})
            log(f"[clinical-files] cached {len(raw)} file(s) -> {cp}")

    table = build_table(raw)
    table["scans"] = scans
    table["from_cache"] = from_cache

    log("-" * 78)
    log(f"CLINICAL TABLE — {table['n_subjects']} subject(s), {table['n_sessions']} session(s)")
    log("-" * 78)
    log(f"  treatment: {table['treatment_counts']}  ({table['n_observed']} observed)")
    log(f"  Δt: {table['n_subjects_with_intervals']} subject(s) have intervals_days.txt; "
        f"{table['n_sessions_with_days_from_first']} session(s) have days_from_first")
    if table["interval_problems"]:
        log(f"  interval problems: {table['interval_problems']}")
    log(f"  RANO available: {table['rano_available']}   "
        f"survival: {table['survival_available']}   age: {table['age_available']}")
    if table["n_mni_overview_rows"] is not None:
        log(f"  MNI overview.tsv rows: {table['n_mni_overview_rows']} "
            f"(compare with the loose raw-space overview.tsv — they differ by design, §3.1(3))")
    log("  status by session ordinal:")
    for k, v in table["status_by_session_ordinal"].items():
        log(f"    ses-{k:02d}  n={sum(v.values()):>3}  {v}")
    if table["conflicts_between_archives"]:
        log(f"  CONFLICTS between archives: {len(table['conflicts_between_archives'])} "
            "— canonical value kept, all disagreements listed in the artefact")
    if table["history_text"]:
        log("-" * 78)
        log("history.txt (verbatim)")
        log("-" * 78)
        for line in table["history_text"].splitlines():
            log(f"  {line}")
    if table["structure_text"]:
        log("-" * 78)
        log("structure.txt (verbatim)")
        log("-" * 78)
        for line in table["structure_text"].splitlines():
            log(f"  {line}")
    # --- persist -----------------------------------------------------------
    written = {}
    payload = {k: v for k, v in table.items()
               if k not in ("history_text", "structure_text")}
    written["json"] = save_artefact(project_root, "01_DATA_FOUNDATION",
                                    "clinical_table", payload)
    written["csv"] = save_table_csv(project_root, "01_DATA_FOUNDATION",
                                    "clinical_table", table["rows"])
    if table.get("history_text"):
        from ..utils.persist import save_text
        written["history_txt"] = save_text(project_root, "01_DATA_FOUNDATION",
                                           "history.txt", table["history_text"])
    if table.get("structure_text"):
        from ..utils.persist import save_text
        written["structure_txt"] = save_text(project_root, "01_DATA_FOUNDATION",
                                             "structure.txt", table["structure_text"])
    table["artefacts"] = written
    log("-" * 78)
    log("WRITTEN")
    for k, v in written.items():
        log(f"  {k}: {v['latest'] if isinstance(v, dict) else v}")
    log("-" * 78)
    return table
