"""Cache-backed diagnostics — answers questions in seconds, no archive read.

The ~90-minute pass wrote its handler output to `01_DATA_FOUNDATION/v2_scan_cache/`.
Anything about what was *seen* can therefore be re-examined instantly. This module
exists so that a surprising guard verdict is investigated against the real
collected bytes rather than by guessing at filenames and shipping a speculative
parser fix.

It reports. It changes nothing and concludes nothing.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ..utils.persist import cache_dir, load_cache

NIFTI = (".nii", ".nii.gz")


def list_caches(project_root) -> list[dict]:
    d = cache_dir(project_root)
    out = []
    for p in sorted(d.glob("*.json")):
        out.append({"key": p.stem, "bytes": p.stat().st_size, "path": str(p)})
    return out


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def filename_vocabulary(project_root, cache_key: str, top: int = 40) -> dict:
    """What do the NIfTI members actually look like?

    Sequence resolution assumes BIDS-style suffixes (`_t1wc.nii.gz`). If the
    derivatives use pipeline-style names (`T1c.nii.gz`) instead, resolution
    silently returns nothing and every downstream modality requirement fails.
    This prints the observed names so the resolver can be corrected against
    reality rather than against an assumption.
    """
    cached = load_cache(project_root, cache_key)
    if not cached:
        return {"status": "NO_CACHE", "key": cache_key}
    headers = cached["raw"].get("headers", {})
    names = [v.get("name", k.split("::", 1)[-1]) for k, v in headers.items()]
    bases = Counter(_basename(n) for n in names)
    stems = Counter(_basename(n).split(".")[0] for n in names)
    depth = Counter(n.count("/") for n in names)
    has_bids_suffix = sum(1 for n in names if re.search(r"_[A-Za-z0-9]+\.nii", n))
    return {
        "status": "OK",
        "n_nifti_headers": len(names),
        "distinct_basenames": len(bases),
        "top_basenames": bases.most_common(top),
        "top_stems": stems.most_common(top),
        "path_depth_frequency": sorted(depth.items()),
        "n_with_bids_style_suffix": has_bids_suffix,
        "example_paths": names[:15],
        "reading": ("If n_with_bids_style_suffix is near zero, the resolver's "
                    "BIDS suffix assumption does not hold for this archive and "
                    "the observed stems above are the real vocabulary."),
    }


def clinical_files_present(project_root, cache_key: str = "clinical_files") -> dict:
    """Which of the small clinical files were actually collected?"""
    cached = load_cache(project_root, cache_key)
    if not cached:
        return {"status": "NO_CACHE", "key": cache_key}
    raw = cached["raw"]
    names = [k.split("::", 1)[-1] for k in raw]
    kinds = Counter()
    for n in names:
        b = _basename(n).lower()
        kinds[b] += 1
    # Match the same spellings the collector does; an underscore-only filter here
    # would report zero even when the hyphenated file had been collected.
    ivrx = re.compile(r"inte?r?vals?[_-]?days?\.txt$", re.I)
    intervals = {k: v for k, v in raw.items() if ivrx.search(k)}
    sample = None
    if intervals:
        key = sorted(intervals)[0]
        sample = {"path": key.split("::", 1)[-1],
                  "first_200_chars": intervals[key][:200],
                  "n_lines": len(intervals[key].strip().splitlines()),
                  "n_whitespace_tokens": len(intervals[key].split())}
    return {
        "status": "OK",
        "n_files_cached": len(raw),
        "basename_counts": kinds.most_common(20),
        "n_intervals_days_files": len(intervals),
        "intervals_sample": sample,
        "reading": ("n_intervals_days_files should equal the number of subjects. "
                    "Zero means the Δt source was never collected; a non-zero "
                    "count with empty content means it was collected but does not "
                    "parse as numbers."),
    }


def text_members(project_root, cache_key: str, pattern: str = "",
                 top: int = 40) -> dict:
    """List cached text members, optionally filtered, with sizes."""
    cached = load_cache(project_root, cache_key)
    if not cached:
        return {"status": "NO_CACHE", "key": cache_key}
    texts = cached["raw"].get("texts", {})
    rx = re.compile(pattern, re.I) if pattern else None
    hits = {k: v for k, v in texts.items() if not rx or rx.search(k)}
    return {"status": "OK", "n_texts_cached": len(texts), "n_matching": len(hits),
            "matches": [{"path": k.split("::", 1)[-1], "bytes": len(v)}
                        for k in sorted(hits)][:top]}


def show_text(project_root, cache_key: str, pattern: str,
              max_lines: int = 40) -> None:
    cached = load_cache(project_root, cache_key)
    if not cached:
        print(f"NO_CACHE for {cache_key}")
        return
    texts = cached["raw"].get("texts", {})
    rx = re.compile(pattern, re.I)
    for k in sorted(texts):
        if rx.search(k):
            print("-" * 78)
            print(k.split("::", 1)[-1])
            print("-" * 78)
            for line in texts[k].splitlines()[:max_lines]:
                print(f"  {line[:140]}")
            return
    print(f"no cached text matches {pattern!r}")


def report(project_root, audit_cache_key: str | None = None) -> dict:
    """Print everything needed to explain a surprising Stage-1 verdict."""
    line = "-" * 78
    caches = list_caches(project_root)
    print(line)
    print("CACHES ON DISK")
    print(line)
    for c in caches:
        print(f"  {c['key']:<52} {c['bytes'] / 1024:>10.1f} KiB")
    if audit_cache_key is None:
        audit = [c["key"] for c in caches if c["key"].startswith("audit_scan_")]
        audit_cache_key = audit[0] if audit else None
    out = {"caches": caches, "audit_cache_key": audit_cache_key}

    print(line)
    print("CLINICAL FILES COLLECTED")
    print(line)
    cf = clinical_files_present(project_root)
    out["clinical_files"] = cf
    if cf["status"] == "OK":
        print(f"  cached files: {cf['n_files_cached']}")
        print(f"  intervals_days.txt files: {cf['n_intervals_days_files']}")
        if cf["intervals_sample"]:
            s = cf["intervals_sample"]
            print(f"    sample: {s['path']}")
            print(f"    lines={s['n_lines']} tokens={s['n_whitespace_tokens']}")
            print(f"    content: {s['first_200_chars']!r}")
        for b, n in cf["basename_counts"]:
            print(f"    {b:<40} {n}")
    else:
        print(f"  {cf['status']}")

    if audit_cache_key:
        print(line)
        print("NIfTI FILENAME VOCABULARY")
        print(line)
        v = filename_vocabulary(project_root, audit_cache_key)
        out["vocabulary"] = v
        if v["status"] == "OK":
            print(f"  headers: {v['n_nifti_headers']}   distinct basenames: "
                  f"{v['distinct_basenames']}")
            print(f"  with BIDS-style suffix: {v['n_with_bids_style_suffix']}")
            print("  top basenames:")
            for b, n in v["top_basenames"][:30]:
                print(f"    {b:<44} {n}")
            print("  example paths:")
            for p in v["example_paths"][:8]:
                print(f"    {p}")
        else:
            print(f"  {v['status']}")

        print(line)
        print("raw-mni-link.tsv (first lines)")
        print(line)
        show_text(project_root, audit_cache_key, r"raw-mni-link", max_lines=12)

    print(line)
    return out
