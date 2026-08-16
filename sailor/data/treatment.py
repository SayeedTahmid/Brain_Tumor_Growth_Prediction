"""Per-time-point treatment status (§3, §5).

Two rules from the specification are enforced here rather than downstream:

1. `unknown` is missing data, not a fourth category. It is emitted as
   `status=None` plus `observed=False`, so no encoder can learn a semantics for
   it by accident (§5).
2. Nothing is imputed from the Stupp schedule. Inferring status from
   weeks-since-surgery is exactly the confound under investigation; imputing it
   would make G2 unfalsifiable.
"""

from __future__ import annotations

from collections import Counter

from ..config import TREATMENT_VALUES

CANON = {"crt": "CRT", "chemoradiotherapy": "CRT", "rt": "CRT", "radiotherapy": "CRT",
         "tmz": "TMZ", "temozolomide": "TMZ",
         "no": "no", "none": "no", "off": "no", "0": "no",
         "unknown": "unknown", "unk": "unknown", "na": "unknown",
         "n/a": "unknown", "": "unknown", "-": "unknown"}


def canonicalise(raw: str | None) -> tuple[str | None, bool, str | None]:
    """-> (status, observed, raw_token). Unrecognised tokens are not coerced."""
    if raw is None:
        return None, False, None
    token = str(raw).strip()
    mapped = CANON.get(token.lower())
    if mapped is None:
        return None, False, token  # unrecognised: reported, never guessed
    if mapped == "unknown":
        return None, False, token
    return mapped, True, token


def extract(overview: dict) -> dict:
    """Build subject -> session -> treatment record from overview.tsv.

    If no treatment column exists in the source table, that is reported as
    NO_TREATMENT_COLUMN_RESOLVED. It is emphatically not the same fact as "every
    timepoint is unknown": one says the variable was not found, the other says it
    was found and is missing, and conflating them would let a table that never
    carried treatment data masquerade as a fully-missing one.
    """
    resolved = overview.get("resolved_columns", {}).get("treatment")
    records: dict[str, dict[str, dict]] = {}
    counts = Counter()
    unrecognised = Counter()
    for rec in overview.get("records", []):
        sub, ses = rec.get("subject"), rec.get("session")
        if not sub or not ses:
            continue
        status, observed, token = canonicalise(rec.get("treatment_raw"))
        if token is not None and status is None and token.lower() not in CANON:
            unrecognised[token] += 1
        counts[status if status else "MISSING"] += 1
        records.setdefault(sub, {})[ses] = {
            "status": status,
            "observed": observed,
            "missing_indicator": int(not observed),
            "raw_token": token,
            "rano": rec.get("rano"),
        }
    n_total = sum(counts.values())
    if resolved is None:
        status = "NO_TREATMENT_COLUMN_RESOLVED"
    elif n_total == 0:
        status = "NO_ROWS"
    else:
        status = "OK"
    return {
        "resolved_column": resolved,
        "records": records,
        "status_counts": dict(counts),
        "n_timepoints": n_total,
        "missing_fraction": (counts.get("MISSING", 0) / n_total) if n_total else None,
        "unrecognised_tokens": dict(unrecognised),
        "canonical_values": TREATMENT_VALUES,
        "policy": ("`unknown` is encoded as status=None with missing_indicator=1; "
                   "it is never a fourth class and is never imputed from schedule."),
        "status": status,
        "n_timepoints_with_status": n_total - counts.get("MISSING", 0),
    }


def weeks_since_first(session_days: dict[str, dict[str, float]]) -> dict:
    """Convert per-subject cumulative day offsets to weeks since the first exam.

    Input: subject -> session -> days-from-first-exam (already measured).
    This is the covariate G2/P3 tests treatment status against. It is deliberately
    *not* called "weeks since surgery": whether the first exam is the pre- or
    post-surgical scan must be established from the descriptor and recorded, not
    assumed here.
    """
    out = {}
    for sub, ses_map in session_days.items():
        out[sub] = {s: (None if d is None else round(d / 7.0, 4))
                    for s, d in ses_map.items()}
    return out
