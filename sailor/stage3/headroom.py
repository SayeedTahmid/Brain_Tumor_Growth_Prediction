"""GATE-3 — headroom above persistence, and the pre-registered primary metric.

GATE-3 asks whether there is enough headroom above persistence to detect any
plausible model gain at n = 26 patients. It needs one number: the minimum
detectable effect on a named metric. Which metric is named is the whole
question, because the MDE varies enormously between them.

WHY THIS MODULE EXISTS SEPARATELY. Four candidate headroom metrics now exist,
and they were not all available when the first baseline ran. Adding metrics
after seeing a result is exactly the forking-paths pattern that turns a null
into a false positive (§17, §2.2). Two facts make it defensible here and both
are recorded rather than assumed:

  1. No C-rung, no model, and no conditioning comparison has been run. What has
     been seen is the BASELINE ONLY. A metric chosen now cannot have been chosen
     to favour a model, because no model exists.
  2. The reason for adding them is stated in advance and is structural, not
     result-driven: absolute volume-change error is scale-dependent, and
     change-region Dice is degenerate for a baseline that predicts no change.

`compare()` reports the MDE for every candidate side by side. It deliberately
does NOT pick one. `preregister()` writes the choice, with its justification and
the commit that produced it, and refuses to run if a choice already exists —
the record is append-only, so a second thought is visible rather than silent.

THE MDE IS AN UPPER BOUND. It assumes per-patient model-minus-baseline
differences have spread comparable to the between-patient spread of the baseline
alone. For a PAIRED comparison that is pessimistic: a model and persistence make
correlated errors on the same patient, so the SD of their difference is normally
smaller than either. The honest paired MDE needs a model to exist. Until one
does, treat every number here as a ceiling on the true requirement, and say so
in the paper rather than quoting it as the requirement itself.
"""

from __future__ import annotations

import json
from pathlib import Path

CANDIDATES = {
    "volume_change_error": {
        "key": "primary_volume_change_error",
        "scale_free": False,
        "defined_on_empty_pairs": True,
        "note": ("Absolute voxels. Interpretable, but between-patient spread is "
                 "dominated by tumour size rather than predictive difficulty, "
                 "which inflates the MDE."),
    },
    "relative_volume_change_error": {
        "key": "headroom_relative_volume_change_error",
        "scale_free": True,
        "defined_on_empty_pairs": False,
        "note": ("Bounded [0, 1] for persistence. Comparable across patients. "
                 "UNDEFINED on the 5 empty->empty pairs, so its mean describes "
                 "203 of 208."),
    },
    "log_volume_ratio_error": {
        "key": "headroom_log_volume_ratio_error",
        "scale_free": True,
        "defined_on_empty_pairs": True,
        "note": ("Scale-free, symmetric in growth and shrinkage, and the only "
                 "headroom candidate defined on ALL 208 pairs."),
    },
    "dice": {
        "key": "secondary_whole_mask_dice",
        "scale_free": True,
        "defined_on_empty_pairs": False,
        "note": ("Secondary per AMD-005: dominated by the unchanged core. "
                 "Reported for comparability with the literature, not as the "
                 "headroom metric."),
    },
}

#: change_region_dice is excluded as a headroom candidate ON PURPOSE.
EXCLUDED = {
    "change_region_dice": (
        "Structurally 0.0 or undefined for persistence, which predicts no "
        "change. A constant baseline has no between-patient variance and cannot "
        "support an MDE. It remains the primary metric for comparing MODELS, "
        "where it is not degenerate."),
}


def compare(result: dict, power: float = 0.80, alpha: float = 0.05) -> dict:
    """MDE for every candidate, side by side. Picks nothing."""
    from .persistence import minimum_detectable_effect
    rows = []
    for name, meta in CANDIDATES.items():
        agg = result["overall"].get(meta["key"])
        if agg is None:
            continue
        mde = minimum_detectable_effect(result, metric=name, power=power, alpha=alpha)
        mean = agg.get("mean")
        rows.append({
            "metric": name,
            "mean": mean,
            "ci": [agg.get("ci_low"), agg.get("ci_high")],
            "per_patient_sd": agg.get("per_patient_sd"),
            "n_pairs_defined": agg.get("n_pairs_defined"),
            "n_pairs_undefined": agg.get("n_pairs_undefined"),
            "mde": mde.get("mde"),
            "mde_as_fraction_of_baseline": (
                None if not (mde.get("mde") and mean) else abs(mde["mde"] / mean)),
            "scale_free": meta["scale_free"],
            "defined_on_empty_pairs": meta["defined_on_empty_pairs"],
            "note": meta["note"],
        })
    return {
        "check": "gate3_headroom_comparison",
        "power": power, "alpha": alpha,
        "n_patients": result.get("n_patients"),
        "candidates": rows,
        "excluded": EXCLUDED,
        "mde_is_an_upper_bound": (
            "Assumes model-minus-baseline per-patient differences have spread "
            "comparable to the between-patient spread of the baseline. For a "
            "paired comparison this is pessimistic, because a model and "
            "persistence err together on the same patient. Treat as a ceiling."),
        "no_metric_selected": (
            "This function reports; it does not choose. Call preregister() with "
            "an explicit metric and justification BEFORE any C-rung runs."),
    }


def preregister(project_root, metric: str, justification: str,
                comparison: dict, overwrite: bool = False) -> dict:
    """Record the GATE-3 primary metric. Append-only; refuses to overwrite.

    A pre-registration that can be quietly revised is not a pre-registration.
    If the choice genuinely needs to change, pass `overwrite=True` — the prior
    choice is preserved in `superseded` so the change is visible in the record.
    """
    from ..utils.persist import save_artefact
    if metric not in CANDIDATES:
        raise ValueError(
            f"{metric!r} is not a headroom candidate. Choices: "
            f"{sorted(CANDIDATES)}. Excluded: {sorted(EXCLUDED)}")
    if not justification or len(justification.strip()) < 40:
        raise ValueError(
            "a substantive justification is required — this is the record that "
            "the choice was made on structural grounds and not to suit a result")

    out = Path(project_root) / "10_EXPERIMENTS" / "v2_gate3_primary_metric.json"
    prior = json.loads(out.read_text()) if out.is_file() else None
    if prior and not overwrite:
        raise RuntimeError(
            f"GATE-3 primary metric is ALREADY pre-registered as "
            f"{prior.get('primary_metric')!r} at {prior.get('written_utc')}. "
            "Changing it after the fact is a forking path. Pass overwrite=True "
            "only if the change is deliberate; the prior choice will be kept in "
            "the record.")

    chosen = next((c for c in comparison["candidates"] if c["metric"] == metric), None)
    rec = {
        "manifest": "gate3_primary_metric",
        "status": "PREREGISTERED",
        "primary_metric": metric,
        "justification": justification,
        "chosen_metric_summary": chosen,
        "all_candidates_at_time_of_choice": comparison["candidates"],
        "excluded_candidates": EXCLUDED,
        "state_when_chosen": (
            "Persistence baseline had been run and seen. NO model, NO C-rung and "
            "NO conditioning comparison had been run. A metric chosen at this "
            "point cannot have been chosen to favour a model that does not exist."),
        "mde_caveat": comparison["mde_is_an_upper_bound"],
        "binding": (
            "GATE-3's GO/NO_GO verdict is read from this metric. Every C-rung "
            "reports it. A different metric may be reported alongside, never "
            "instead."),
    }
    if prior:
        rec["superseded"] = prior
    rec["artefact"] = save_artefact(project_root, "10_EXPERIMENTS",
                                    "gate3_primary_metric", rec)
    return rec


def print_comparison(cmp: dict) -> None:
    line = "-" * 92
    print(line)
    print("GATE-3 HEADROOM CANDIDATES  —  no metric is selected by this report")
    print(line)
    print(f"  n patients: {cmp['n_patients']}   power: {cmp['power']:.0%}   "
          f"alpha: {cmp['alpha']}")
    print(f"\n  {'metric':<32}{'mean':>10}{'sd':>10}{'MDE':>10}"
          f"{'MDE/mean':>10}{'undef':>7}")
    for r in cmp["candidates"]:
        f = lambda v, w=10, p=4: (f"{v:>{w}.{p}f}" if isinstance(v, (int, float))
                                  else f"{'—':>{w}}")
        print(f"  {r['metric']:<32}{f(r['mean'])}{f(r['per_patient_sd'])}"
              f"{f(r['mde'])}{f(r['mde_as_fraction_of_baseline'], 10, 3)}"
              f"{r['n_pairs_undefined']:>7}")
    print(f"\n  EXCLUDED:")
    for k, v in cmp["excluded"].items():
        print(f"    {k}: {v}")
    print(f"\n  {cmp['mde_is_an_upper_bound']}")
    print(f"\n  {cmp['no_metric_selected']}")
    print(line)
