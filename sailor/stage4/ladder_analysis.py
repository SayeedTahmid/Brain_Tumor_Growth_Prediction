"""Analysis of the completed ladder. No training, no archive reads.

The pooled numbers say no rung beats persistence and no rung differs from
another: spread 0.0071 across four rungs against a frozen MDE of 0.0555. That is
the headline, and it is a null.

A null invites two fair questions, and this module answers both from artefacts
already on disk rather than from new runs:

  1. IS THE NULL UNIFORM, OR AN AVERAGE OVER OPPOSING EFFECTS? A pooled mean of
     zero is consistent with "nothing anywhere" and with "helps some patients,
     hurts others". Those are different findings. The per-patient breakdown
     distinguishes them.

  2. IS IT UNIFORM ACROSS Δt? AMD-002 stratified persistence by frozen bands
     (<=21 / 22-90 / >90 days) because copy-forward is near-ceiling at short
     intervals. The RUNGS were never stratified. Conditioning could plausibly
     help only where persistence is weak — the long-interval band — and a pooled
     mean dominated by the 83 short-interval pairs would hide it.

WHAT THIS IS NOT. It is not a search for a subgroup where the model wins. Both
breakdowns are pre-specified by the ROS (bands frozen under AMD-002, patient as
the unit under AMD-003), and neither is used to redefine a result. A subgroup
effect found here would be a HYPOTHESIS for a future cohort, not a claim from
this one — with 26 patients and three bands, a per-band comparison has a fraction
of the power of the pooled one, and the pooled MDE of 0.0555 does not transfer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: AMD-002 bands, frozen before any result was seen.
DELTA_BANDS = (("<=21d", 0.0, 21.0), ("22-90d", 21.0, 90.0), (">90d", 90.0, 1e9))
CORRECTED = ("C0res_v2", "C1_v2", "C2_v2", "P2_v2")
UNCORRECTED = ("C0res", "C1")


def _band(delta_days) -> str:
    if delta_days is None:
        return "unknown"
    d = float(delta_days)
    for name, lo, hi in DELTA_BANDS:
        if (lo == 0.0 and d <= hi) or (lo < d <= hi):
            return name
    return "unknown"


def load_rung(project_root, rung: str) -> dict:
    p = Path(project_root) / "10_EXPERIMENTS" / f"v2_rung_{rung}.json"
    if not p.is_file():
        raise FileNotFoundError(f"{p} — has rung {rung} been run?")
    return json.loads(p.read_text())


def _per_pair(project_root, rung: str) -> list:
    """Per-pair rows from the fold eval files the rung runner wrote.

    The rung artefact omits `per_pair` to keep it small, so the rows come from
    the checkpoint-side eval JSONs. Those are the SAME rows the rung aggregated,
    not a recomputation.
    """
    d = Path(project_root) / "11_CHECKPOINTS"
    rows = []
    for f in sorted(d.glob(f"{rung}_r*f*_eval.json")):
        ev = json.loads(f.read_text())
        stem = f.stem.replace(f"{rung}_", "").replace("_eval", "")
        rep, fold = stem.split("f")
        for r in ev.get("per_pair", []):
            rows.append(dict(r, repeat=int(rep[1:]), fold=int(fold)))
    return rows


def per_patient(project_root, rungs=CORRECTED) -> dict:
    """Model-minus-persistence per patient, per rung. Is the null uniform?"""
    out, ref = {}, None
    for rung in rungs:
        rows = _per_pair(project_root, rung)
        by = {}
        for r in rows:
            if r.get("model_log_ratio") is None or r.get("pers_log_ratio") is None:
                continue
            by.setdefault(r["subject"], []).append(
                float(r["model_log_ratio"]) - float(r["pers_log_ratio"]))
        out[rung] = {k: float(np.mean(v)) for k, v in sorted(by.items())}
        ref = ref or out[rung]

    patients = sorted(ref)
    summary = []
    for p in patients:
        vals = [out[r].get(p) for r in rungs if out[r].get(p) is not None]
        summary.append({
            "subject": p,
            "per_rung": {r: out[r].get(p) for r in rungs},
            "mean_across_rungs": float(np.mean(vals)) if vals else None,
            "helped_by_any_rung": bool(any(v < 0 for v in vals)),
        })

    helped = [s for s in summary if s["mean_across_rungs"] is not None
              and s["mean_across_rungs"] < 0]
    return {
        "rungs": list(rungs),
        "n_patients": len(patients),
        "per_patient": summary,
        "n_patients_where_model_beat_persistence": len(helped),
        "patients_helped": [s["subject"] for s in helped],
        "reading": (
            "A pooled mean near zero is consistent with 'nothing anywhere' and "
            "with 'helps some, hurts others'. If roughly half the patients are "
            "helped and half hurt, with magnitudes far above the pooled mean, "
            "the null is an AVERAGE over opposing effects and should be "
            "described that way. If almost all patients sit near zero, the null "
            "is uniform."),
        "not_a_subgroup_search": (
            "A patient-level effect found here is a HYPOTHESIS for a future "
            "cohort, not a claim from this one. The pooled MDE of 0.0555 does "
            "not transfer to a subgroup."),
    }


def by_delta_band(project_root, rungs=CORRECTED) -> dict:
    """Each rung stratified by the frozen AMD-002 bands.

    Persistence is near-ceiling at short intervals, so a pooled mean dominated
    by the 83 short-interval pairs could hide a conditioning effect that only
    exists where persistence is weak.
    """
    out = {}
    for rung in rungs:
        rows = _per_pair(project_root, rung)
        bands = {}
        for name, _, _ in DELTA_BANDS:
            sub = [r for r in rows if _band(r.get("delta_days")) == name]
            by_m, by_p = {}, {}
            for r in sub:
                if r.get("model_log_ratio") is not None:
                    by_m.setdefault(r["subject"], []).append(float(r["model_log_ratio"]))
                if r.get("pers_log_ratio") is not None:
                    by_p.setdefault(r["subject"], []).append(float(r["pers_log_ratio"]))
            m = [float(np.mean(v)) for v in by_m.values()]
            p = [float(np.mean(v)) for v in by_p.values()]
            bands[name] = {
                "n_pairs": len(sub), "n_patients": len(m),
                "model": float(np.mean(m)) if m else None,
                "persistence": float(np.mean(p)) if p else None,
                "gap": (float(np.mean(m)) - float(np.mean(p))) if (m and p) else None,
            }
        out[rung] = bands
    return {
        "bands": [b[0] for b in DELTA_BANDS],
        "frozen_by": "AMD-002, before any result was seen",
        "per_rung": out,
        "power_caveat": (
            "Each band holds a fraction of the 208 pairs and fewer than 26 "
            "patients. The frozen MDE of 0.0555 was computed on the FULL "
            "cohort and does NOT apply within a band. No band-level comparison "
            "should be read as significant."),
    }


def print_per_patient(r: dict, top: int = 30) -> None:
    line = "-" * 78
    print(line)
    print("PER-PATIENT: model minus persistence (negative = model better)")
    print(line)
    rungs = r["rungs"]
    print("  patient " + "".join(f"{x:>11}" for x in rungs) + f"{'mean':>11}")
    for s in r["per_patient"][:top]:
        cells = "".join(
            (f"{s['per_rung'][x]:>+11.4f}" if s["per_rung"].get(x) is not None
             else f"{'—':>11}") for x in rungs)
        mark = "  <-- helped" if (s["mean_across_rungs"] or 0) < 0 else ""
        print(f"  {s['subject']:<8}{cells}{s['mean_across_rungs']:>+11.4f}{mark}")
    print(f"\n  patients where the model beat persistence on average: "
          f"{r['n_patients_where_model_beat_persistence']} / {r['n_patients']}")
    if r["patients_helped"]:
        print(f"  {r['patients_helped']}")
    print(f"\n  {r['reading']}")
    print(f"\n  {r['not_a_subgroup_search']}")
    print(line)


def print_bands(r: dict) -> None:
    line = "-" * 78
    print(line)
    print("BY Δt BAND (frozen, AMD-002)")
    print(line)
    for rung, bands in r["per_rung"].items():
        print(f"\n  {rung}")
        print(f"    {'band':<9}{'pairs':>7}{'patients':>10}{'model':>10}"
              f"{'persist':>10}{'gap':>10}")
        for name, b in bands.items():
            f = lambda v: f"{v:>10.4f}" if v is not None else f"{'—':>10}"
            print(f"    {name:<9}{b['n_pairs']:>7}{b['n_patients']:>10}"
                  f"{f(b['model'])}{f(b['persistence'])}{f(b['gap'])}")
    print(f"\n  {r['power_caveat']}")
    print(line)
