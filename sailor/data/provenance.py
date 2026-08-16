"""Provenance firewall (§4).

The legacy folder mixes the EBRAINS download with artefacts from a previous
TaDiff reproduction. This module classifies what is there, verifies canonical
archives against `SHA512.txt`, and produces the read-only pointer manifest that
`00_CANONICAL/` holds instead of a 43 GB copy (§4.1).

It never writes into the legacy folder.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from ..config import (AMBIGUOUS_FILES, CANONICAL_EXPECTED_ABSENT, CANONICAL_FILES,
                      FORBIDDEN_INPUT_FILES, QUARANTINE_FILES)

SHA512_LINE = re.compile(r"^([0-9a-fA-F]{128})\s+[*]?(.+)$")


def sha512_of(path: Path, chunk: int = 8 << 20,
              progress=None) -> tuple[str, int, float]:
    """Streaming SHA-512. Returns (hexdigest, bytes_read, seconds)."""
    import time
    h = hashlib.sha512()
    n = 0
    t0 = time.time()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            n += len(b)
            if progress:
                progress(n)
    return h.hexdigest(), n, round(time.time() - t0, 2)


def parse_sha512_file(path: Path) -> dict[str, str]:
    """Map basename -> expected digest. Absent file yields an empty map."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        m = SHA512_LINE.match(line.strip())
        if m:
            out[Path(m.group(2).strip()).name] = m.group(1).lower()
    return out


def scan_legacy_root(legacy_root: Path) -> dict:
    """Inventory the legacy folder without touching it."""
    legacy_root = Path(legacy_root)
    if not legacy_root.exists():
        return {"status": "ABSENT", "legacy_root": str(legacy_root), "entries": []}
    entries = []
    for p in sorted(legacy_root.iterdir()):
        try:
            st = p.stat()
            entries.append({"name": p.name, "is_dir": p.is_dir(),
                            "size_bytes": (st.st_size if p.is_file() else None),
                            "mtime": int(st.st_mtime)})
        except OSError as e:
            entries.append({"name": p.name, "error": str(e)})
    return {"status": "OK", "legacy_root": str(legacy_root), "entries": entries}


def classify_entries(entries: list[dict]) -> dict:
    names = {e["name"] for e in entries}
    present_canonical = [n for n in CANONICAL_FILES if n in names]
    missing_canonical = [n for n in CANONICAL_FILES if n not in names]
    expected_absent_present = [n for n in CANONICAL_EXPECTED_ABSENT if n in names]
    expected_absent_absent = [n for n in CANONICAL_EXPECTED_ABSENT if n not in names]
    quarantine = [n for n in QUARANTINE_FILES if n in names]
    ambiguous = [n for n in AMBIGUOUS_FILES if n in names]
    known = set(CANONICAL_FILES) | set(CANONICAL_EXPECTED_ABSENT) | \
        set(QUARANTINE_FILES) | set(AMBIGUOUS_FILES)
    unclassified = sorted(n for n in names if n not in known)
    return {
        "canonical_present": present_canonical,
        "canonical_missing": missing_canonical,
        "expected_absent_but_present": expected_absent_present,
        "expected_absent_confirmed": expected_absent_absent,
        "quarantine_present": quarantine,
        "ambiguous_present": ambiguous,
        "unclassified": unclassified,
        "forbidden_as_input": [n for n in FORBIDDEN_INPUT_FILES if n in names],
    }


def verify_canonical(legacy_root: Path, entries: list[dict],
                     verify_hashes: bool = True,
                     max_hash_bytes: int | None = None) -> dict:
    """Verify present canonical files against SHA512.txt.

    `verify_hashes=False` records SKIPPED rather than a fabricated PASS.
    `max_hash_bytes` skips files above a size and records why, so a 43 GB hash
    can be deferred to an explicit long run instead of blocking the audit.
    """
    legacy_root = Path(legacy_root)
    expected = parse_sha512_file(legacy_root / "SHA512.txt")
    cls = classify_entries(entries)
    results = []
    for name in cls["canonical_present"] + cls["ambiguous_present"]:
        p = legacy_root / name
        if p.is_dir():
            results.append({"file": name, "status": "SKIPPED_DIR"})
            continue
        exp = expected.get(name)
        size = p.stat().st_size
        rec = {"file": name, "size_bytes": size,
               "in_sha512_txt": exp is not None,
               "path": str(p)}
        if exp is None:
            rec["status"] = "NOT_IN_SHA512_TXT"
        elif not verify_hashes:
            rec["status"] = "SKIPPED_BY_FLAG"
        elif max_hash_bytes is not None and size > max_hash_bytes:
            rec["status"] = "SKIPPED_TOO_LARGE"
            rec["max_hash_bytes"] = max_hash_bytes
        else:
            digest, nbytes, secs = sha512_of(p)
            rec["sha512"] = digest
            rec["seconds"] = secs
            rec["status"] = "MATCH" if digest == exp else "MISMATCH"
        results.append(rec)
    for name in cls["canonical_missing"]:
        results.append({"file": name, "status": "ABSENT",
                        "in_sha512_txt": name in expected})
    return {"sha512_txt_present": (legacy_root / "SHA512.txt").exists(),
            "n_expected_entries": len(expected),
            "results": results,
            "classification": cls}


def canonical_pointer_manifest(legacy_root: Path, verification: dict) -> dict:
    """00_CANONICAL holds read-only pointers, never copies (§4.1)."""
    return {
        "legacy_root": str(legacy_root),
        "policy": "read-only pointers; canonical data is never copied or modified",
        "files": [
            {"file": r["file"], "path": r.get("path"),
             "size_bytes": r.get("size_bytes"), "sha512": r.get("sha512"),
             "verification": r["status"]}
            for r in verification["results"] if r["status"] != "ABSENT"
        ],
        "absent": [r["file"] for r in verification["results"] if r["status"] == "ABSENT"],
    }


def disk_report(paths: list[Path]) -> list[dict]:
    """Measured free space for each existing path's filesystem."""
    import shutil
    out = []
    seen = set()
    for p in paths:
        p = Path(p)
        probe = p
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            key = os.stat(probe).st_dev
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        total, used, free = shutil.disk_usage(probe)
        out.append({"path": str(probe), "total_gb": round(total / 2**30, 1),
                    "used_gb": round(used / 2**30, 1),
                    "free_gb": round(free / 2**30, 1)})
    return out
