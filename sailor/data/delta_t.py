"""Δt recovery and provenance (§4.1 applied to the known gap, §9 G7).

The descriptor says exact inter-exam intervals exist only in the source and raw
versions; MNI intervals were partly hand-reconstructed and may be inaccurate.
Before any download is proposed, this module tries, in order of trustworthiness,
every source already present:

  1. BIDS `*_scans.tsv` `acq_time` inside `rawdata_BIDS.tar.bz2`   -> EXACT
  2. `<meta-data>.txt` / `history.txt` acquisition dates            -> EXACT if dated
  3. `raw_needed.tar` (user-repackaged; origin must be verified)    -> EXACT if dated
  4. `overview.tsv` day columns                                     -> APPROXIMATE
  5. MNI-version derived intervals                                  -> APPROXIMATE

Each attempt records what it looked for, what it found, and why it failed. The
result of that record is what justifies (or does not justify) a download — §4.1
requires the attempt to be reported, not merely claimed.
"""

from __future__ import annotations

import re
from collections import defaultdict

from .tables import parse_scans_tsv
from .inventory import _parse_time

SCANS_TSV_RE = re.compile(r"(sub-[A-Za-z0-9]+)[_/].*?(ses-[A-Za-z0-9]+)?.*_scans\.tsv$")
DATE_RE = re.compile(r"(\d{4}[-/]\d{2}[-/]\d{2})")

EXACT = "EXACT"
APPROX = "APPROXIMATE"


def _subject_session_from_path(path: str) -> tuple[str | None, str | None]:
    sub = re.search(r"(sub-[A-Za-z0-9]+)", path)
    ses = re.search(r"(ses-[A-Za-z0-9]+)", path)
    return (sub.group(1) if sub else None), (ses.group(1) if ses else None)


def attempt_scans_tsv(texts: dict[str, str]) -> dict:
    """Attempt 1 — BIDS scans.tsv acq_time."""
    found: dict[str, dict[str, str]] = defaultdict(dict)
    files_seen = []
    for key, text in texts.items():
        name = key.split("::", 1)[-1]
        if not name.endswith("_scans.tsv"):
            continue
        files_seen.append(name)
        sub, ses_from_path = _subject_session_from_path(name)
        rows = parse_scans_tsv(text)
        for r in rows:
            if not r.get("acq_time"):
                continue
            _, ses_from_row = _subject_session_from_path(r.get("filename") or "")
            ses = ses_from_row or ses_from_path
            if sub and ses:
                found[sub].setdefault(ses, r["acq_time"])
    n_times = sum(len(v) for v in found.values())
    return {"attempt": "bids_scans_tsv_acq_time", "kind": EXACT,
            "n_files_seen": len(files_seen), "n_subjects": len(found),
            "n_session_times": n_times,
            "result": "OK" if n_times else "NO_ACQ_TIME_VALUES",
            "why": ("acq_time present in BIDS scans.tsv" if n_times else
                    "no _scans.tsv member carried a populated acq_time column"),
            "times": {k: dict(v) for k, v in found.items()},
            "examples": files_seen[:10]}


def attempt_metadata_text(texts: dict[str, str]) -> dict:
    """Attempt 2 — dates embedded in meta-data/history text files."""
    hits = defaultdict(dict)
    scanned = []
    for key, text in texts.items():
        name = key.split("::", 1)[-1]
        base = name.rsplit("/", 1)[-1].lower()
        if not (base.endswith(".txt") and
                ("meta" in base or "history" in base or "structure" in base)):
            continue
        scanned.append(name)
        for line in text.splitlines():
            sub, ses = _subject_session_from_path(line)
            d = DATE_RE.search(line)
            if sub and ses and d:
                hits[sub].setdefault(ses, d.group(1))
    n = sum(len(v) for v in hits.values())
    return {"attempt": "metadata_text_dates", "kind": EXACT,
            "n_files_scanned": len(scanned), "n_session_dates": n,
            "result": "OK" if n else "NO_DATED_SESSION_LINES",
            "why": ("dates parsed from metadata text" if n else
                    "no line carried subject, session and an ISO-like date together"),
            "times": {k: dict(v) for k, v in hits.items()},
            "files_scanned": scanned[:20]}


def attempt_raw_needed(texts: dict[str, str], present: bool) -> dict:
    """Attempt 3 — the user-repackaged raw_needed.tar."""
    if not present:
        return {"attempt": "raw_needed_tar", "kind": EXACT, "result": "ARCHIVE_ABSENT",
                "why": "raw_needed.tar not present in the legacy folder", "times": {}}
    inner = attempt_scans_tsv(texts)
    meta = attempt_metadata_text(texts)
    times = {k: dict(v) for k, v in inner["times"].items()}
    for sub, m in meta["times"].items():
        times.setdefault(sub, {}).update(m)
    n = sum(len(v) for v in times.values())
    return {"attempt": "raw_needed_tar", "kind": EXACT,
            "n_session_dates": n,
            "result": "OK" if n else "NO_DATES_INSIDE",
            "why": ("dates recovered from repackaged raw archive; origin must be "
                    "verified against SHA512.txt before these are treated as canonical"
                    if n else "archive present but carried no dated scans.tsv or metadata"),
            "times": times}


def attempt_overview_days(overview: dict) -> dict:
    """Attempt 4 — numeric day columns in overview.tsv. APPROXIMATE by §3.1(2)."""
    per_sub = defaultdict(dict)
    n_numeric = 0
    for rec in overview.get("records", []):
        sub, ses, raw = rec.get("subject"), rec.get("session"), rec.get("days_raw")
        if not (sub and ses) or raw in (None, ""):
            continue
        try:
            per_sub[sub][ses] = float(str(raw).strip())
            n_numeric += 1
        except ValueError:
            continue
    return {"attempt": "overview_days_column", "kind": APPROX,
            "n_numeric_values": n_numeric,
            "resolved_column": overview.get("resolved_columns", {}).get("days"),
            "result": "OK" if n_numeric else "NO_NUMERIC_DAY_VALUES",
            "why": ("day offsets parsed from overview.tsv; the descriptor states "
                    "MNI-version intervals may be inaccurate, so these are APPROXIMATE"
                    if n_numeric else "no column resolved to numeric days"),
            "days": {k: dict(v) for k, v in per_sub.items()}}


def build(texts_bids: dict, texts_legacy: dict, overview: dict,
          raw_needed_present: bool) -> dict:
    """Run every attempt, choose per-session Δt provenance, never silently mix."""
    attempts = [
        attempt_scans_tsv(texts_bids),
        attempt_metadata_text({**texts_bids, **texts_legacy}),
        attempt_raw_needed(texts_legacy, raw_needed_present),
        attempt_overview_days(overview),
    ]

    exact_times: dict[str, dict[str, str]] = {}
    exact_source = None
    for a in attempts:
        if a["kind"] == EXACT and a.get("result") == "OK":
            exact_times = a["times"]
            exact_source = a["attempt"]
            break

    per_session: dict[str, dict] = {}
    if exact_times:
        for sub, ses_map in exact_times.items():
            ordered = sorted(
                ((s, _parse_time(t)) for s, t in ses_map.items()),
                key=lambda kv: (kv[1] is None, kv[1]))
            prev = None
            for ses, t in ordered:
                if t is None:
                    per_session[f"{sub}/{ses}"] = {"days_from_prev": None,
                                                   "source": "none",
                                                   "kind": "UNPARSEABLE"}
                    continue
                days = None if prev is None else round((t - prev).total_seconds() / 86400.0, 3)
                per_session[f"{sub}/{ses}"] = {
                    "days_from_prev": days,
                    "source": ("raw_scans_tsv_acq_time"
                               if exact_source == "bids_scans_tsv_acq_time"
                               else "raw_exam_date"),
                    "kind": EXACT}
                prev = t
    else:
        approx = attempts[3]
        for sub, ses_map in approx.get("days", {}).items():
            ordered = sorted(ses_map.items(), key=lambda kv: kv[1])
            prev = None
            for ses, d in ordered:
                per_session[f"{sub}/{ses}"] = {
                    "days_from_prev": None if prev is None else round(d - prev, 3),
                    "source": "overview_days_column", "kind": APPROX}
                prev = d

    n = len(per_session)
    n_exact = sum(1 for v in per_session.values() if v["kind"] == EXACT)
    lines = []
    for a in attempts:
        lines.append(f"{a['attempt']:<26} [{a['kind']:<11}] {a['result']:<24} {a['why']}")
    if n == 0:
        lines.append("NO Δt VALUES RECOVERED — Δt is UNAVAILABLE, not approximate.")
    else:
        lines.append(f"{n} session intervals constructed; {n_exact} EXACT, "
                     f"{n - n_exact} APPROXIMATE.")
    return {"attempts": attempts, "per_session": per_session,
            "n_intervals": n, "n_exact": n_exact,
            "exact_source": exact_source,
            "summary_lines": lines,
            "downloads_justified": bool(n_exact == 0)}
