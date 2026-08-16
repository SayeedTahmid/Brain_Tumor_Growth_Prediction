"""Result persistence (§15.4 idempotence, §15.6 artefacts).

A pass over `derivatives.tar.bz2` costs ~90 minutes over Drive FUSE. Holding its
output only in a notebook variable means a kernel restart, a browser disconnect
or an idle timeout throws that away. Worse, it makes a re-run tempting, and a
re-run that silently differs from the first is exactly the reproducibility
failure §15.4 is written against.

So: every expensive scan writes two things.

  cache/     the raw collected bytes, keyed by archive. Cheap to reload, and a
             later run reuses it instead of decompressing 43 GB again.
  artefact   the derived, human-readable result under the project root, with the
             provenance stamp that makes it citable.

The cache stores what was *read*, never what was *concluded*. Reloading a cache
therefore re-runs all parsing and guard logic against the same bytes, so fixing
a parser does not require another 90-minute pass — but changing a parser cannot
be hidden by a stale conclusion either.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

CACHE_DIRNAME = "v2_scan_cache"


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cache_dir(project_root) -> Path:
    d = Path(project_root) / "01_DATA_FOUNDATION" / CACHE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path(project_root, key: str) -> Path:
    return cache_dir(project_root) / f"{key}.json"


def save_cache(project_root, key: str, raw: dict, meta: dict | None = None) -> str:
    """Persist raw scanned file contents. Small: text members only."""
    p = cache_path(project_root, key)
    payload = {"key": key, "saved_utc": _stamp(),
               "n_entries": len(raw), "meta": meta or {}, "raw": raw}
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(p)  # atomic: a killed kernel never leaves a half-written cache
    return str(p)


def load_cache(project_root, key: str) -> dict | None:
    p = cache_path(project_root, key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save_artefact(project_root, subdir: str, name: str, obj: dict,
                  prefix: str = "v2_", stamp_repro: bool = True) -> dict:
    """Write a derived result, plus a timestamped copy so runs are comparable.

    v0.21 — every artefact now carries §12 reproducibility metadata
    (`git_commit`, `git_branch`, `git_dirty`, `data_version`, versions) unless
    the caller opts out. Previously `env.git_state()` existed and was wired into
    the bootstrap record only, so results written straight from a module or a
    notebook cell carried none of it. §18.5: a result from a dirty tree is not
    reproducible and must be RECORDED as such — which requires the record to
    exist in the first place.
    """
    out = Path(project_root) / subdir
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if stamp_repro and isinstance(obj, dict) and "reproducibility" not in obj:
        obj = dict(obj)
        obj["reproducibility"] = reproducibility_stamp()
    body = json.dumps(obj, indent=2, default=str)
    latest = out / f"{prefix}{name}.json"
    dated = out / f"{prefix}{name}_{stamp}.json"
    latest.write_text(body)
    dated.write_text(body)
    return {"latest": str(latest), "dated": str(dated),
            "bytes": len(body)}


def reproducibility_stamp(code_root=None) -> dict:
    """§12 metadata for any record. Safe to call anywhere; never raises."""
    from datetime import datetime as _dt, timezone as _tz
    rec = {"written_utc": _dt.now(_tz.utc).isoformat(timespec="seconds")}
    try:
        from .env import git_state
        from ..config import get_paths, DATA_VERSION
        root = Path(code_root) if code_root else Path(get_paths().code_root)
        rec.update(git_state(root))
        rec["data_version"] = DATA_VERSION
    except Exception as exc:                      # never block a write
        rec["repro_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import sailor
        rec["sailor_version"] = sailor.__version__
    except Exception:
        pass
    # The point of recording this is that it CHANGES how the result may be used.
    if rec.get("git_available") is False:
        rec["publication_status"] = (
            "NOT_REPRODUCIBLE — code was not run from a git repository, so no "
            "commit identifies it. §18.5: must be reproduced from a clean commit "
            "before publication.")
    elif rec.get("git_dirty"):
        rec["publication_status"] = (
            "NOT_REPRODUCIBLE — git_dirty: uncommitted changes were present. "
            "§18.5: must be reproduced from a clean commit before publication.")
    else:
        rec["publication_status"] = "REPRODUCIBLE — clean tree at a known commit"
    return rec


def save_text(project_root, subdir: str, name: str, text: str,
              prefix: str = "v2_") -> str:
    out = Path(project_root) / subdir
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{prefix}{name}"
    p.write_text(text)
    return str(p)


def save_table_csv(project_root, subdir: str, name: str,
                   rows: list[dict], prefix: str = "v2_") -> str | None:
    """Write a row table as CSV. Columns are the union of keys, sorted stably."""
    if not rows:
        return None
    import csv
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    out = Path(project_root) / subdir
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{prefix}{name}.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return str(p)


class Timer:
    def __init__(self, label: str, log=print):
        self.label, self.log = label, log

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = round(time.time() - self.t0, 2)
        self.log(f"[{self.label}] {self.seconds} s")
        return False


def latest_full_pass(project_root, require: tuple = ()) -> str:
    """Newest full-pass cache key, by MODIFICATION TIME.

    v0.19 (defect 24). Callers previously used
    ``sorted(glob("audit_scan_full_*.json"))[-1]``, which is LEXICOGRAPHIC, not
    chronological. The v0.17 key `audit_scan_full_mall_s40_ps6_icor1_plhm8_v17_...`
    sorts BEFORE the v0.16 key `audit_scan_full_mall_s60_...` because '4' < '6',
    so every caller silently read a three-month-old pass that contained no
    percentiles and one subject's images. The PLHM check reported 0 series and
    INCONCLUSIVE from a cache that simply lacked the fields.

    The irony is worth recording: the key was lengthened precisely so a stale
    cache could not be reused silently, and lengthening it is what made the stale
    one sort last.

    `require` names fields that must be present in the payload; a cache lacking
    them is rejected with a message naming what was missing, rather than
    producing an empty result that looks like a finding.
    """
    import json as _json
    # Companion caches (__slices, __exported, and anything added later) are
    # written alongside the pass and are NOT full passes. Excluding them by an
    # explicit list would need updating every time one is added, so any stem
    # carrying a `__` suffix is treated as a companion.
    cands = [p for p in cache_dir(project_root).glob("audit_scan_full_*.json")
             if "__" not in p.stem]
    if not cands:
        raise RuntimeError(
            "no full-pass cache found — run run_stage1_audit(read_volumes=True) first")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    problems = []
    for p in cands:
        if not require:
            break
        try:
            raw = _json.loads(p.read_text()).get("raw", {})
        except Exception as exc:
            problems.append(f"{p.stem}: unreadable ({type(exc).__name__})")
            continue
        missing = [f for f in require if f not in raw]
        if not missing:
            break
        problems.append(f"{p.stem}: missing {missing}")
    else:
        raise RuntimeError(
            "no full-pass cache satisfies " + repr(list(require)) + ":\n  "
            + "\n  ".join(problems)
            + "\nRe-run the pass, or pass audit_cache_key= explicitly.")
    if len(cands) > 1:
        others = ", ".join(c.stem for c in cands[1:4])
        print(f"[cache] {len(cands)} full-pass caches present; using the NEWEST "
              f"by mtime: {p.stem}\n[cache] older: {others}")
    return p.stem
