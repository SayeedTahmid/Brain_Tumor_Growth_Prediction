"""Patient / session inventory (§19.2).

Counts are derived from what the scan measured, never from the descriptor's
nominal numbers. Where a quantity cannot be measured from the artefacts present,
the field is `None` and the accompanying status says why.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime


def build_sessions(file_rows: list[dict], header_index: dict) -> dict:
    """Group NIfTI members into subject -> session -> sequences/annotations."""
    sessions: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(
        lambda: {"sequences": {}, "annotations": [], "unassigned": []}))
    orphans = []
    for r in file_rows:
        sub, ses = r["subject"], r["session"]
        if not sub or not ses:
            orphans.append(r["path"])
            continue
        key = f"{r['archive']}::{r['path']}" if "archive" in r else r["path"]
        hdr = header_index.get(key) or header_index.get(r["path"])
        entry = {"path": r["path"], "sequence": r["sequence"],
                 "annotation_kind": r["annotation_kind"],
                 "annotation_component": r["annotation_component"],
                 "shape": hdr.get("shape") if hdr else None,
                 "spacing": hdr.get("spacing") if hdr else None,
                 "dtype": hdr.get("dtype") if hdr else None,
                 "header_measured": hdr is not None}
        node = sessions[sub][ses]
        if r["annotation_kind"] in ("CL", "ONCO", "brain_mask", "nawm_mask",
                                    "dose_map", "UNRESOLVED_MASKLIKE"):
            node["annotations"].append(entry)
        elif r["sequence"]:
            node["sequences"].setdefault(r["sequence"], []).append(entry)
        else:
            node["unassigned"].append(entry)
    return {"sessions": {s: dict(v) for s, v in sessions.items()},
            "orphan_paths": orphans}


def summarise(sessions: dict) -> dict:
    subs = sorted(sessions.keys())
    per_patient = {}
    seq_counter = Counter()
    shape_counter = Counter()
    spacing_counter = Counter()
    dtype_counter = Counter()
    n_sessions_total = 0
    n_unassigned = 0
    for sub in subs:
        ses_map = sessions[sub]
        n_sessions_total += len(ses_map)
        seqs_here = Counter()
        for ses, node in ses_map.items():
            for seq, entries in node["sequences"].items():
                seqs_here[seq] += 1
                seq_counter[seq] += 1
                for e in entries:
                    if e["shape"]:
                        shape_counter[tuple(e["shape"])] += 1
                    if e["spacing"]:
                        spacing_counter[tuple(round(x, 3) for x in e["spacing"])] += 1
                    if e["dtype"]:
                        dtype_counter[e["dtype"]] += 1
            # Annotations are measured on the same footing as sequences: when
            # sequence resolution fails, counting spacing only inside the
            # sequences loop silently reports an empty spacing table for a
            # dataset whose spacings were in fact measured.
            for e in node["annotations"] + node["unassigned"]:
                if e["shape"]:
                    shape_counter[tuple(e["shape"])] += 1
                if e["spacing"]:
                    spacing_counter[tuple(round(x, 3) for x in e["spacing"])] += 1
                if e["dtype"]:
                    dtype_counter[e["dtype"]] += 1
        n_unassigned += sum(len(n["unassigned"]) for n in ses_map.values())
        per_patient[sub] = {
            "n_sessions": len(ses_map),
            "sessions": sorted(ses_map.keys()),
            "sequence_counts": dict(seqs_here),
        }
    return {
        "n_patients": len(subs),
        "n_sessions": n_sessions_total,
        "n_files_with_unresolved_sequence": n_unassigned,
        "patients": per_patient,
        "sequence_frequency": seq_counter.most_common(),
        "shape_frequency": [[list(k), v] for k, v in shape_counter.most_common(20)],
        "spacing_frequency": [[list(k), v] for k, v in spacing_counter.most_common(20)],
        "dtype_frequency": dtype_counter.most_common(),
    }


# --------------------------------------------------------------------- delta-t

def _parse_time(value: str):
    if not value:
        return None
    v = value.strip().replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(v, fmt)
        except ValueError:
            continue
    return None


def gaps_from_acq_times(acq_by_session: dict[str, dict[str, str]]) -> dict:
    """Inter-session gaps in days from BIDS acq_time, where present.

    `acq_by_session[subject][session] = acq_time string`.
    """
    out = {}
    for sub, ses_map in acq_by_session.items():
        parsed = {s: _parse_time(t) for s, t in ses_map.items()}
        usable = {s: t for s, t in parsed.items() if t is not None}
        ordered = sorted(usable.items(), key=lambda kv: kv[1])
        gaps = []
        for i in range(1, len(ordered)):
            gaps.append({"from": ordered[i - 1][0], "to": ordered[i][0],
                         "days": round((ordered[i][1] - ordered[i - 1][1]).total_seconds()
                                       / 86400.0, 3),
                         "source": "bids_scans_tsv_acq_time"})
        out[sub] = {"n_sessions_with_time": len(usable),
                    "n_sessions_seen": len(ses_map),
                    "gaps": gaps}
    return out


def gaps_from_overview(overview: dict) -> dict:
    """Inter-session gaps from an overview column, if one parses as numeric days."""
    per_sub: dict[str, list] = defaultdict(list)
    for rec in overview.get("records", []):
        sub, ses, raw = rec.get("subject"), rec.get("session"), rec.get("days_raw")
        if not sub or not ses:
            continue
        val = None
        if raw not in (None, ""):
            try:
                val = float(str(raw).strip())
            except ValueError:
                val = None
        per_sub[sub].append({"session": ses, "days_raw": raw, "days": val})
    out = {}
    for sub, rows in per_sub.items():
        n_num = sum(1 for r in rows if r["days"] is not None)
        out[sub] = {"n_rows": len(rows), "n_numeric": n_num, "rows": rows}
    return out
