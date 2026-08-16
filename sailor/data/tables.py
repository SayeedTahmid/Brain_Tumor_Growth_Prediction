"""Index-table parsing (§3): `overview.tsv`, `missing.tsv`, `raw-mni-link.tsv`.

Column names are not assumed. Each parser reports the header it actually found
and resolves the columns it needs by matching against a small set of candidate
names; if resolution fails it returns `UNVERIFIED` with the observed header
rather than guessing (§2.2).
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

SUB_RE = re.compile(r"(sub-[A-Za-z0-9]+)")
SES_RE = re.compile(r"(ses-[A-Za-z0-9]+)")


def read_delimited(text: str) -> tuple[list[str], list[dict]]:
    """Sniff delimiter, return (header, rows)."""
    sample = text[:8192]
    delim = "\t"
    if sample.count("\t") == 0:
        delim = "," if sample.count(",") >= sample.count(";") else ";"
    rdr = csv.reader(io.StringIO(text), delimiter=delim)
    rows = [r for r in rdr if any(c.strip() for c in r)]
    if not rows:
        return [], []
    header = [h.strip() for h in rows[0]]
    out = []
    for r in rows[1:]:
        r = list(r) + [""] * (len(header) - len(r))
        out.append({header[i]: r[i].strip() for i in range(len(header))})
    return header, out


def _find_col(header: list[str], candidates: list[str]) -> str | None:
    low = {h.lower().strip(): h for h in header}
    for c in candidates:
        if c in low:
            return low[c]
    for c in candidates:
        for h_low, h in low.items():
            if c in h_low:
                return h
    return None


def _norm_sub(value: str) -> str | None:
    if not value:
        return None
    m = SUB_RE.search(value)
    if m:
        return m.group(1)
    v = value.strip()
    if v.isdigit():
        return f"sub-{int(v):02d}"
    return None


def _norm_ses(value: str) -> str | None:
    if not value:
        return None
    m = SES_RE.search(value)
    if m:
        return m.group(1)
    v = value.strip()
    if v.isdigit():
        return f"ses-{int(v):02d}"
    return None


def parse_overview(text: str) -> dict:
    header, rows = read_delimited(text)
    sub_c = _find_col(header, ["subject", "sub", "patient", "participant_id", "id"])
    ses_c = _find_col(header, ["session", "ses", "timepoint", "time_point", "exam"])
    treat_c = _find_col(header, ["treatment", "therapy", "treat", "status"])
    days_c = _find_col(header, ["days", "delta", "interval", "dt", "date"])
    rano_c = _find_col(header, ["rano", "response"])
    age_c = _find_col(header, ["age"])
    sex_c = _find_col(header, ["sex", "gender"])
    os_c = _find_col(header, ["overall_survival", "survival", "os"])
    recs = []
    for r in rows:
        rec = {
            "subject": _norm_sub(r.get(sub_c, "")) if sub_c else None,
            "session": _norm_ses(r.get(ses_c, "")) if ses_c else None,
            "treatment_raw": r.get(treat_c) if treat_c else None,
            "days_raw": r.get(days_c) if days_c else None,
            "rano": r.get(rano_c) if rano_c else None,
            "age": r.get(age_c) if age_c else None,
            "sex": r.get(sex_c) if sex_c else None,
            "overall_survival": r.get(os_c) if os_c else None,
            "row": r,
        }
        recs.append(rec)
    resolved = {"subject": sub_c, "session": ses_c, "treatment": treat_c,
                "days": days_c, "rano": rano_c, "age": age_c, "sex": sex_c,
                "overall_survival": os_c}
    return {"header": header, "n_rows": len(rows), "records": recs,
            "resolved_columns": resolved,
            "status": "OK" if sub_c else "UNVERIFIED_COLUMNS"}


def parse_missing(text: str) -> dict:
    """Official exclusion list (G9). Supports both layouts it may arrive in.

    WIDE:  subject, session, then one column per sequence holding a y/n flag.
    LONG:  subject, session, and a column naming the missing sequence.

    The layout is detected from whether the non-key columns are descriptor
    sequence names, not assumed. For the wide layout the *polarity* of y/n is
    not decided here: the file is named `missing`, which suggests y = missing,
    but a filename is not evidence. Both readings are returned along with the
    marginal counts, and G9 settles the polarity by comparing each reading
    against the files actually observed on disk.
    """
    from ..config import FUNCTIONAL_SEQUENCES, STRUCTURAL_SEQUENCES
    known_seqs = {s.lower() for s in STRUCTURAL_SEQUENCES + FUNCTIONAL_SEQUENCES}

    header, rows = read_delimited(text)
    sub_c = _find_col(header, ["subject", "sub", "patient", "participant_id"])
    ses_c = _find_col(header, ["session", "ses", "timepoint", "exam"])
    key_cols = {c for c in (sub_c, ses_c) if c}
    seq_cols = [h for h in header if h not in key_cols and h.lower() in known_seqs]
    layout = "wide" if len(seq_cols) >= 3 else "long"

    entries = []
    index: dict[tuple[str, str], set[str]] = {}
    index_inverted: dict[tuple[str, str], set[str]] = {}
    flag_counts: dict[str, int] = {}
    per_sequence: dict[str, dict[str, int]] = {c: {} for c in seq_cols}

    if layout == "wide":
        for r in rows:
            sub = _norm_sub(r.get(sub_c, "")) if sub_c else None
            ses = _norm_ses(r.get(ses_c, "")) if ses_c else None
            flagged_y, flagged_n = [], []
            for c in seq_cols:
                v = (r.get(c) or "").strip().lower()
                flag_counts[v] = flag_counts.get(v, 0) + 1
                per_sequence[c][v] = per_sequence[c].get(v, 0) + 1
                if v in ("y", "yes", "1", "true"):
                    flagged_y.append(c.lower())
                elif v in ("n", "no", "0", "false"):
                    flagged_n.append(c.lower())
            entries.append({"subject": sub, "session": ses,
                            "flagged_y": flagged_y, "flagged_n": flagged_n,
                            "missing_sequences": flagged_y, "row": r})
            if sub and ses:
                index.setdefault((sub, ses), set()).update(flagged_y)
                index_inverted.setdefault((sub, ses), set()).update(flagged_n)
    else:
        seq_c = _find_col(header, ["sequence", "modality", "series", "missing", "scan"])
        for r in rows:
            sub = _norm_sub(r.get(sub_c, "")) if sub_c else None
            ses = _norm_ses(r.get(ses_c, "")) if ses_c else None
            raw_seq = (r.get(seq_c) or "") if seq_c else ""
            seqs = [x.strip() for x in re.split(r"[,\s;/]+", raw_seq) if x.strip()]
            entries.append({"subject": sub, "session": ses,
                            "missing_sequences": seqs, "row": r})
            if sub and ses:
                index.setdefault((sub, ses), set()).update(x.lower() for x in seqs)
        seq_cols = [seq_c] if seq_c else []

    resolved_ok = bool(sub_c and ses_c and (seq_cols or layout == "long"))
    return {
        "header": header, "n_rows": len(rows), "entries": entries, "layout": layout,
        "sequence_columns": [c.lower() for c in seq_cols],
        "index": {f"{k[0]}/{k[1]}": sorted(v) for k, v in index.items()},
        "index_inverted": {f"{k[0]}/{k[1]}": sorted(v) for k, v in index_inverted.items()},
        "flag_counts": flag_counts,
        "per_sequence_flag_counts": per_sequence,
        "polarity": "UNDETERMINED — settled by G9 against observed files"
                    if layout == "wide" else "N/A",
        "resolved_columns": {"subject": sub_c, "session": ses_c,
                             "sequence": None if layout == "wide" else
                             _find_col(header, ["sequence", "modality", "series",
                                                "missing", "scan"])},
        "status": "OK" if resolved_ok else "UNVERIFIED_COLUMNS",
    }


def parse_src_to_raw(text: str) -> dict:
    """`src-to-raw.yaml` — the dcm2niix conversion plan.

    Not a substitute for `raw-mni-link.tsv` under G8, but it does two useful
    things for free: it enumerates which sequence directories existed per source
    session, and it shows whether source and raw session indices coincide. Parsed
    with a small indent reader rather than PyYAML so bootstrap keeps one fewer
    pinned dependency.
    """
    in_dir = out_dir = fname = None
    records = []
    for line in text.splitlines():
        st = line.strip()
        if st.startswith("in_dir:"):
            in_dir = st.split(":", 1)[1].strip()
        elif st.startswith("out_dir:"):
            out_dir = st.split(":", 1)[1].strip()
        elif st.startswith("filename:"):
            fname = st.split(":", 1)[1].strip()
            if in_dir and out_dir:
                records.append({"in_dir": in_dir, "out_dir": out_dir,
                                "filename": fname})
            in_dir = out_dir = fname = None

    per_session: dict[tuple[str, str], set[str]] = {}
    index_mismatch = []
    for rec in records:
        sub = SUB_RE.search(rec["in_dir"])
        ses_src = SES_RE.search(rec["in_dir"])
        ses_out = SES_RE.search(rec["out_dir"])
        if not (sub and ses_src):
            continue
        per_session.setdefault((sub.group(1), ses_src.group(1)), set()).add(
            rec["filename"].lower())
        if ses_out and ses_out.group(1) != ses_src.group(1):
            index_mismatch.append({"in": rec["in_dir"], "out": rec["out_dir"]})

    subs = sorted({k[0] for k in per_session})
    return {
        "n_conversion_records": len(records),
        "n_subjects": len(subs),
        "n_sessions": len(per_session),
        "subjects": subs,
        "sequences_per_session": {f"{k[0]}/{k[1]}": sorted(v)
                                  for k, v in per_session.items()},
        "source_raw_index_mismatches": index_mismatch,
        "source_raw_indices_identical": not index_mismatch,
        "caveat": ("Describes source -> raw only. The MNI join is defined solely "
                   "by raw-mni-link.tsv and this file may not stand in for it (G8)."),
        "status": "OK" if records else "NO_RECORDS_PARSED",
    }


NO_COUNTERPART = {"no", "none", "-", "", "na", "n/a"}


def parse_raw_mni_link(text: str) -> dict:
    """Session correspondence MNI <-> raw/source. The only legal join (G8).

    A literal `no` in the MNI column means that raw exam has no MNI derivative
    at all. Those rows are counted, not discarded: they are the difference
    between the 337 raw sessions and the 270 MNI sessions, and they are
    systematically the EARLIEST exams per patient. Treating them as unparseable
    would hide the fact that MNI `ses-01` is not a patient's first examination.
    """
    header, rows = read_delimited(text)
    sub_c = _find_col(header, ["subject", "sub", "patient", "participant_id"])
    raw_c = _find_col(header, ["raw", "source", "src", "orig"])
    mni_c = _find_col(header, ["mni", "derivative", "deriv", "target"])
    pairs, unmapped, unparsed = [], [], []
    for r in rows:
        sub = _norm_sub(r.get(sub_c, "")) if sub_c else None
        raw_ses = _norm_ses(r.get(raw_c, "")) if raw_c else None
        raw_mni = (r.get(mni_c) or "").strip() if mni_c else ""
        mni_ses = _norm_ses(raw_mni)
        no_counterpart = raw_mni.lower() in NO_COUNTERPART
        if mni_ses is None and not no_counterpart and raw_mni:
            unparsed.append({"subject": sub, "raw_session": raw_ses,
                             "mni_value": raw_mni})
        if no_counterpart and sub and raw_ses:
            unmapped.append({"subject": sub, "raw_session": raw_ses})
        pairs.append({"subject": sub, "raw_session": raw_ses,
                      "mni_session": mni_ses,
                      "no_mni_counterpart": no_counterpart, "row": r})

    # Which raw ordinal does each patient's MNI ses-01 correspond to?
    first_mni_at_raw = {}
    for p_ in pairs:
        if p_["mni_session"] and p_["subject"] and p_["raw_session"]:
            key = p_["subject"]
            if p_["mni_session"] == "ses-01":
                first_mni_at_raw[key] = p_["raw_session"]
    dropped_before_first = {}
    for sub, raw_ses in first_mni_at_raw.items():
        m = re.search(r"(\d+)$", raw_ses)
        if m:
            dropped_before_first[sub] = int(m.group(1)) - 1

    return {"header": header, "n_rows": len(rows), "pairs": pairs,
            "n_mapped": sum(1 for p_ in pairs if p_["mni_session"]),
            "n_raw_without_mni": len(unmapped),
            "raw_sessions_without_mni": unmapped[:400],
            "unparsed_mni_values": unparsed[:50],
            "mni_ses01_maps_to_raw": first_mni_at_raw,
            "n_raw_exams_dropped_before_mni_ses01": dropped_before_first,
            "resolved_columns": {"subject": sub_c, "raw": raw_c, "mni": mni_c},
            "status": "OK" if (sub_c and raw_c and mni_c) else "UNVERIFIED_COLUMNS"}


def parse_scans_tsv(text: str) -> list[dict]:
    """BIDS `*_scans.tsv`: filename + acq_time, the exact-date candidate (G7)."""
    header, rows = read_delimited(text)
    fn_c = _find_col(header, ["filename", "file", "scan"])
    at_c = _find_col(header, ["acq_time", "acquisition_time", "acqtime", "date"])
    out = []
    for r in rows:
        out.append({"filename": r.get(fn_c) if fn_c else None,
                    "acq_time": r.get(at_c) if at_c else None})
    return out
