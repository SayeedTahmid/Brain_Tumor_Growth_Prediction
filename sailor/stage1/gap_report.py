"""Gap report (§19.2 final paragraph, §4.1).

Ends Stage 1 with a ranked list of the minimum EBRAINS artefacts that would
close each measured gap, the projected decompressed footprint against measured
free space, and the justification block §4.1 requires before approval.

This module downloads nothing and contains no network call.
"""

from __future__ import annotations

# bzip2 on NIfTI-like medical volumes typically expands 2.5-4x. The band is
# reported as a band, and it is labelled an assumption, because the true ratio
# is only known after a real extraction (assumption ledger, §2.4).
BZ2_EXPANSION_LOW = 2.5
BZ2_EXPANSION_HIGH = 4.0

# Ranked candidates. Sizes are UNKNOWN until read from the EBRAINS listing; the
# audit reports them as UNKNOWN rather than inventing figures (§2.2).
CANDIDATES = [
    {"artefact": "rawdata_BIDS.tar.bz2",
     "needed_by": "G7 (exact Δt), G8 (raw side of the join), G9",
     "gap": "exact inter-exam acquisition times",
     "size_bytes": None,
     "note": "already present in the legacy folder in the expected case; "
             "only a candidate if verification found it absent or mismatched"},
    {"artefact": "rawdata.tar.bz2",
     "needed_by": "G7 exact Δt if BIDS scans.tsv carries no acq_time",
     "gap": "exact inter-exam intervals per §3.1(2)",
     "size_bytes": None,
     "note": "smaller than sourcedata; prefer over DICOM if it resolves Δt"},
    {"artefact": "sourcedata.tar.bz2",
     "needed_by": "G7 exact Δt of last resort (DICOM StudyDate/AcquisitionDate)",
     "gap": "exact exam dates",
     "size_bytes": None,
     "note": "largest option; justified only if raw-level sources all fail"},
    {"artefact": "rawdata_BIDS_ext.tar.bz2",
     "needed_by": "extended BIDS metadata if sequence resolution stays UNRESOLVED",
     "gap": "sequence/annotation naming resolution",
     "size_bytes": None,
     "note": "only if naming cannot be resolved from present members"},
]


def project_footprint(canonical_results: list[dict]) -> dict:
    """Projected decompressed size band for present compressed archives."""
    rows = []
    total_c = 0
    for r in canonical_results:
        name = r.get("file", "")
        size = r.get("size_bytes")
        if not name.endswith((".tar.bz2", ".tar")) or not size:
            continue
        total_c += size
        if name.endswith(".bz2"):
            lo, hi = size * BZ2_EXPANSION_LOW, size * BZ2_EXPANSION_HIGH
        else:
            lo = hi = size
        rows.append({"archive": name,
                     "compressed_gb": round(size / 2**30, 2),
                     "projected_decompressed_gb_low": round(lo / 2**30, 2),
                     "projected_decompressed_gb_high": round(hi / 2**30, 2)})
    return {
        "archives": rows,
        "total_compressed_gb": round(total_c / 2**30, 2),
        "total_projected_low_gb": round(sum(r["projected_decompressed_gb_low"] for r in rows), 2),
        "total_projected_high_gb": round(sum(r["projected_decompressed_gb_high"] for r in rows), 2),
        "expansion_assumption": f"{BZ2_EXPANSION_LOW}x-{BZ2_EXPANSION_HIGH}x "
                                "(ASSUMPTION — unverified until a real extraction)",
        "policy": "streaming/selective member reads are preferred to full extraction (§4.1)",
    }


def build(verification: dict, delta_t: dict, target_resolution: dict,
          dose: dict, disk: list[dict], guards: list[dict]) -> dict:
    cls = verification["classification"]
    results = verification["results"]
    footprint = project_footprint(results)

    gaps = []
    if cls["canonical_missing"]:
        gaps.append({"gap": "canonical artefact absent",
                     "detail": cls["canonical_missing"],
                     "blocks": "provenance verification and any claim of an unmodified input"})
    if delta_t["n_exact"] == 0:
        gaps.append({"gap": "no EXACT Δt source recovered",
                     "detail": [a["attempt"] + ": " + a["result"] for a in delta_t["attempts"]],
                     "blocks": "G7; every Δt-conditioned rung (C1-C4) inherits approximate time"})
    if target_resolution["status"] != "RESOLVED":
        gaps.append({"gap": "primary target not resolved on disk",
                     "detail": target_resolution["status"],
                     "blocks": "G1 and the entire primary cohort definition"})
    if dose["status"] != "OK":
        gaps.append({"gap": "dose maps not located",
                     "detail": dose["status"],
                     "blocks": "rung C3 and control P2"})
    for g in guards:
        if g["status"] == "INCONCLUSIVE":
            gaps.append({"gap": f"{g['guard']} inconclusive", "detail": g["detail"],
                         "blocks": f"{g['guard']} cannot be reported as passed"})

    ranked = []
    for c in CANDIDATES:
        relevant = False
        if c["artefact"] in cls["canonical_missing"]:
            relevant = True
        if delta_t["n_exact"] == 0 and "Δt" in c["needed_by"]:
            relevant = True
        if (target_resolution["status"] != "RESOLVED"
                and "naming" in c["gap"]):
            relevant = True
        if relevant:
            ranked.append(c)

    free = min((d["free_gb"] for d in disk), default=None)
    lines = []
    lines.append(f"measured free space (min across filesystems): "
                 f"{'UNMEASURED' if free is None else str(free) + ' GB'}")
    lines.append(f"compressed archives present: {footprint['total_compressed_gb']} GB")
    lines.append(f"projected decompressed: {footprint['total_projected_low_gb']}"
                 f"-{footprint['total_projected_high_gb']} GB "
                 f"[{footprint['expansion_assumption']}]")
    if free is not None and footprint["total_projected_high_gb"] > free:
        lines.append("FULL EXTRACTION DOES NOT FIT at the upper projection — "
                     "streaming member reads are mandatory, not preferred.")
    for g in gaps:
        lines.append(f"GAP: {g['gap']} -> blocks {g['blocks']}")
    if not ranked:
        lines.append("No additional EBRAINS artefact is required by a measured gap. "
                     "No download is proposed.")
    else:
        lines.append("Minimum additional artefacts, ranked (APPROVAL REQUIRED — "
                     "nothing was fetched):")
        for i, c in enumerate(ranked, 1):
            lines.append(f"  {i}. {c['artefact']} — needed by {c['needed_by']}; "
                         f"size UNKNOWN until the EBRAINS listing is read; {c['note']}")

    justification = [{
        "artefact": c["artefact"],
        "stage_and_guard": c["needed_by"],
        "tried_first": [f"{a['attempt']}: {a['result']} ({a['why']})"
                        for a in delta_t["attempts"]],
        "download_size": "UNKNOWN — read from the EBRAINS listing before approval",
        "free_space_gb": free,
        "fits": "UNDETERMINED until size is known",
        "approval": "PENDING — no file fetched this turn",
    } for c in ranked]

    return {"gaps": gaps, "ranked_candidates": ranked, "footprint": footprint,
            "disk": disk, "justifications": justification, "lines": lines,
            "downloads_performed": 0}
