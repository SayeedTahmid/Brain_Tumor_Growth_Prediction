"""Loose-file inspection — seconds, no archive decompression.

Three questions the bounded pass left open can be answered from files totalling
a few hundred kilobytes, without touching a single `.tar.bz2`:

  1. What does `SHA512.txt` actually cover? (every canonical file came back
     NOT_IN_SHA512_TXT, which is either a name-format mismatch or genuinely
     partial coverage)
  2. What are `overview.tsv`'s real column names, and why did the treatment
     column resolve to 337 rows of MISSING?
  3. Is `raw-mni-link.tsv` loose anywhere on disk, or only inside an archive?

It prints what it finds and resolves nothing on its own.
"""

from __future__ import annotations

from pathlib import Path

from ..data.tables import parse_missing, parse_overview, read_delimited
from ..data.treatment import CANON


def inspect(legacy_root: Path, n_rows: int = 8, project_root=None) -> dict:
    """Print and, when `project_root` is given, persist the loose-file audit."""
    legacy_root = Path(legacy_root)
    out = {}
    line = "-" * 78

    # -- 1. SHA512.txt ------------------------------------------------------
    print(line)
    print("SHA512.txt")
    print(line)
    p = legacy_root / "SHA512.txt"
    if not p.exists():
        print("  ABSENT")
        out["sha512"] = {"status": "ABSENT"}
    else:
        from ..data.provenance import parse_sha512_file
        raw = p.read_text(errors="replace")
        entries = parse_sha512_file(p)
        print(f"  {len(raw)} bytes, {len(raw.splitlines())} line(s), "
              f"{len(entries)} parsed entry/entries.")
        print("  covered filenames (digest elided -- the name is what matters):")
        for name, dig in entries.items():
            print(f"    {dig[:12]}...  {name}")
        unparsed = [ln for ln in raw.splitlines()
                    if ln.strip() and not any(n in ln for n in entries)]
        if unparsed:
            print("  lines that did NOT parse as <digest>  <name> "
                  "(this is why files report NOT_IN_SHA512_TXT):")
            for ln in unparsed[:12]:
                print(f"    {ln[:120]}")
        out["sha512"] = {"status": "OK", "n_lines": len(raw.splitlines()),
                         "covered": sorted(entries), "n_unparsed": len(unparsed)}

    # -- 2. overview.tsv ----------------------------------------------------
    print(line)
    print("overview.tsv")
    print(line)
    p = legacy_root / "overview.tsv"
    if not p.exists():
        print("  ABSENT")
        out["overview"] = {"status": "ABSENT"}
    else:
        text = p.read_text(errors="replace")
        header, rows = read_delimited(text)
        parsed = parse_overview(text)
        print(f"  {len(rows)} data row(s)")
        print(f"  header: {header}")
        print(f"  resolved columns: {parsed['resolved_columns']}")
        print(f"  first {n_rows} row(s):")
        for r in rows[:n_rows]:
            print(f"    {r}")
        tcol = parsed["resolved_columns"].get("treatment")
        if tcol:
            vals = {}
            for r in rows:
                vals[r.get(tcol, "")] = vals.get(r.get(tcol, ""), 0) + 1
            print(f"  distinct values in resolved treatment column {tcol!r}:")
            for v, c in sorted(vals.items(), key=lambda kv: -kv[1])[:20]:
                known = "recognised" if v.strip().lower() in CANON else "UNRECOGNISED"
                print(f"    {v!r:<24} n={c:<5} {known}")
        else:
            print("  no treatment column resolved -- candidate columns above")
        # every column, so a mis-resolution is visible rather than inferred
        print("  per-column distinct-value counts:")
        for h in header:
            distinct = {r.get(h, "") for r in rows}
            sample = sorted(distinct)[:6]
            print(f"    {h:<28} {len(distinct):>5} distinct  e.g. {sample}")
        out["overview"] = {"status": "OK", "header": header, "n_rows": len(rows),
                           "resolved_columns": parsed["resolved_columns"]}

    # -- 3. missing.tsv -----------------------------------------------------
    print(line)
    print("missing.tsv")
    print(line)
    p = legacy_root / "missing.tsv"
    if not p.exists():
        print("  ABSENT")
        out["missing"] = {"status": "ABSENT"}
    else:
        text = p.read_text(errors="replace")
        header, rows = read_delimited(text)
        parsed = parse_missing(text)
        print(f"  {len(rows)} data row(s)")
        print(f"  header: {header}")
        print(f"  resolved columns: {parsed['resolved_columns']}")
        for r in rows[:n_rows]:
            print(f"    {r}")
        out["missing"] = {"status": "OK", "header": header, "n_rows": len(rows)}

    # -- 4. is raw-mni-link.tsv loose anywhere? -----------------------------
    print(line)
    print("raw-mni-link.tsv")
    print(line)
    hits = [q for q in legacy_root.rglob("*") if q.is_file()
            and "raw" in q.name.lower() and "mni" in q.name.lower()]
    loose = list(legacy_root.glob("*.tsv"))
    print(f"  name matches under legacy root: {[str(h) for h in hits] or 'NONE'}")
    print(f"  all loose .tsv files: {[q.name for q in loose]}")
    print("  if absent here, it can only be an archive member -- a full "
          "derivatives listing is required to confirm, and G8 stays "
          "INCONCLUSIVE until then")
    out["raw_mni_link"] = {"loose_matches": [str(h) for h in hits],
                           "loose_tsv": [q.name for q in loose]}

    # -- 5. src-to-raw.yaml: a possible session-correspondence substitute ----
    print(line)
    print("src-to-raw.yaml (possible G8 substitute)")
    print(line)
    p = legacy_root / "src-to-raw.yaml"
    if not p.exists():
        print("  ABSENT")
        out["src_to_raw"] = {"status": "ABSENT"}
    else:
        text = p.read_text(errors="replace")
        lines = text.splitlines()
        print(f"  {len(text)} bytes, {len(lines)} line(s). First 25:")
        for ln in lines[:25]:
            print(f"    {ln[:110]}")
        print("  NOTE: this maps source->raw. It is NOT the MNI join and may not "
              "be substituted for raw-mni-link.tsv under G8.")
        out["src_to_raw"] = {"status": "OK", "n_lines": len(lines)}

    if project_root is not None:
        from ..utils.persist import save_artefact
        out["artefact"] = save_artefact(project_root, "06_QC_REPORTS",
                                        "loose_file_inspection", out)
        print(f"  written: {out['artefact']['latest']}")
    print(line)
    return out
