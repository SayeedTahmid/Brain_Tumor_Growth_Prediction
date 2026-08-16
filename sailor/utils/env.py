"""Runtime facts, measured (§12, §13, §15.7, §18.5)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def git_state(code_root: Path) -> dict:
    """§18.5 — a result from a dirty tree is not reproducible."""
    def run(*args):
        try:
            out = subprocess.run(["git", "-C", str(code_root), *args],
                                 capture_output=True, text=True, timeout=15)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None
    commit = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain")
    if commit is None:
        return {"git_available": False, "git_commit": None, "git_branch": None,
                "git_dirty": None, "note": "not a git repository or git unavailable"}
    return {"git_available": True, "git_commit": commit, "git_branch": branch,
            "git_dirty": bool(status), "dirty_paths": (status or "").splitlines()[:50]}


def hardware() -> dict:
    """CPU/RAM measured; GPU reported only if torch is importable and sees one."""
    rec = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": None,
        "gpu": "NOT_QUERIED",
    }
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        rec["ram_total_gb"] = round(pages * size / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    try:
        import torch  # noqa
        if torch.cuda.is_available():
            rec["gpu"] = torch.cuda.get_device_name(0)
            rec["vram_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        else:
            rec["gpu"] = "NONE_VISIBLE"
    except Exception:
        rec["gpu"] = "TORCH_NOT_IMPORTED"
    return rec


class Stopwatch:
    def __init__(self):
        self.t0 = time.time()
        self.marks: list[tuple[str, float]] = []

    def mark(self, label: str):
        self.marks.append((label, round(time.time() - self.t0, 2)))
        return self.marks[-1]

    @property
    def elapsed(self) -> float:
        return round(time.time() - self.t0, 2)


def peak_rss_gb() -> float | None:
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS reports bytes.
        return round((kb / 2**20) if sys.platform != "darwin" else (kb / 2**30), 3)
    except Exception:
        return None


def resource_card(section_id: str, profiled: bool, measured: dict) -> dict:
    """§15.7 — an unprofiled card shows UNMEASURED, never a plausible estimate."""
    if not profiled:
        return {"section": section_id, "compute_mode": "CPU", "profiled": "NO",
                "wall_seconds": "UNMEASURED", "peak_rss_gb": "UNMEASURED",
                "gpu_required": "NO", "vram_required": "N/A (CPU-only)",
                "disk_required_gb": "UNMEASURED",
                "safe_on_fresh_runtime": "UNVERIFIED",
                "checkpoint_resume": "UNVERIFIED",
                "measured_against": "nothing"}
    card = {"section": section_id, "compute_mode": "CPU", "profiled": "YES",
            "gpu_required": "NO", "vram_required": "N/A (CPU-only)"}
    card.update(measured)
    return card
