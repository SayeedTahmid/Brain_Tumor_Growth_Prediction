"""GATE-3 — closing the gate when its literal criterion cannot be satisfied.

GATE-3 asks whether there is enough headroom above persistence to detect a
plausible model gain at n = 26, and its fourth criterion reads: *"if the MDE
exceeds the largest gain reported in the comparable literature, that is recorded
as a power limitation in advance."*

**That criterion is unsatisfiable as written.** The most comparable work is
TaDiff (Liu et al., IEEE TMI 2025, 44(6):2449-2462; arXiv 2309.05406) — same
institution, overlapping authors, and it cites Larsson et al. 2020, the same
annotation lineage as SAILOR's CL masks. It reports future-tumour-prediction
DSC 0.719 ± 0.13 against ground truth and **reports no persistence,
copy-forward, or last-observation-carried-forward baseline at all**. There is
therefore no published "gain over persistence" for this task to compare an MDE
against.

WHY TaDiff's 0.719 CANNOT SIMPLY BE USED. Four incompatibilities, each
disqualifying on its own:

  1. It is measured on 2D slices filtered to tumour >= 1 cm^2 (3,352 test
     slices). Excluding small tumours removes the hardest cases.
  2. Statistics are reported over slices from 5 test patients — the exact
     unit-of-analysis error §6 forbids, and roughly sqrt(slices/patients) too
     narrow.
  3. Its target is edema AND enhancing tumour; SAILOR's locked primary is the
     enhancing-only CL component. Larger, smoother masks score higher Dice.
  4. It uses per-channel z-score normalisation, the variant SAILOR rejected.

Reading "0.719 vs 0.4697" as a 53% gain compares different tasks on different
units. That is the single most likely misreading of this project's results and
is recorded here so it cannot be made accidentally.

WHAT THIS MODULE DOES. It records the literature finding, states the amended
criterion, and computes the one quantity the amended gate turns on — the
fraction of total available headroom a model must capture to be detectable. It
does NOT invent a numerical pass threshold. Setting one now, after the fraction
is already known, would be a post-hoc threshold of exactly the kind
pre-registration exists to prevent. The verdict is a recorded human judgement
with the number in hand.
"""

from __future__ import annotations

import json
from pathlib import Path

LITERATURE_FINDING = {
    "reference": ("Liu Q, Fuster-Garcia E, Hovden IT, MacIntosh BJ, Grødem EOS, "
                  "Brandal P, Lopez-Mateu C, Sederevičius D, Skogen K, "
                  "Schellhorn T, Bjørnerud A, Eeg Emblem K. Treatment-Aware "
                  "Diffusion Probabilistic Model for Longitudinal MRI Generation "
                  "and Diffuse Glioma Growth Prediction. IEEE Trans Med Imaging. "
                  "2025;44(6):2449-2462. doi:10.1109/TMI.2025.3533038"),
    "preprint_read": "arXiv:2309.05406v3",
    "reported_future_tumour_dsc": 0.719,
    "reported_future_tumour_dsc_sd": 0.13,
    "reported_source_segmentation_dsc": 0.849,
    "persistence_baseline_reported": False,
    "finding": (
        "TaDiff reports future tumour prediction DSC 0.719 ± 0.13 against ground "
        "truth and reports NO persistence or copy-forward baseline. The most "
        "comparable work does not establish whether a learned model beats "
        "copying the last mask forward."),
    "not_comparable_because": [
        "2D slices filtered to tumour >= 1 cm^2 (3,352 test slices), vs SAILOR's "
        "3D volumes with no size filter",
        "statistics over slices from 5 test patients, vs patient-level bootstrap "
        "over 26 — the §6 unit-of-analysis error",
        "target is edema AND enhancing tumour, vs SAILOR's locked enhancing-only "
        "CL component",
        "per-channel z-score normalisation, the variant SAILOR rejected",
    ],
    "verification_caveat": (
        "Read from the arXiv v3 preprint (2023). The published TMI 2025 version "
        "grew to 225 exams; if a persistence baseline was added in review it "
        "would be there and not in what was read. VERIFY against the published "
        "version before writing this up as settled."),
    "what_it_does_not_license": (
        "It is NOT established that TaDiff fails to beat persistence. It is "
        "established that nobody checked. Claiming the former from the latter "
        "would be the same error in the opposite direction."),
}

AMENDMENT = {
    "id": "AMD-008",
    "section": "§9 G3, GATE-3 criterion 4",
    "date": "2026-08-16",
    "change": (
        "GATE-3's external-benchmark criterion is replaced. The MDE is no longer "
        "compared against 'the largest gain reported in the comparable "
        "literature', because no such figure exists: the most comparable work "
        "reports no persistence baseline. The gate instead turns on the fraction "
        "of TOTAL AVAILABLE HEADROOM a model must capture to be detectable at "
        "n = 26, which is self-contained and needs no external reference."),
    "prompted_by": (
        "TaDiff (IEEE TMI 2025) reports DSC 0.719 for future tumour prediction "
        "with no copy-forward comparison, and its numbers are not commensurable "
        "with SAILOR's in any case (2D size-filtered slices, slice-level "
        "statistics over 5 patients, edema+enhancing target, z-score "
        "normalisation)."),
    "nature": (
        "Replaces an unsatisfiable criterion with a measurable one. Does NOT "
        "weaken the gate: the amended criterion can still return NO_GO, and the "
        "quantity it turns on was computed before the amendment was written."),
    "results_seen_before_amendment": (
        "Persistence baseline and the MDE. NO model, NO C-rung, NO conditioning "
        "comparison. The amendment cannot favour a model that does not exist."),
    "no_threshold_invented": (
        "The amended criterion deliberately carries NO numerical pass threshold. "
        "The headroom fraction is already known (0.322), so any threshold set now "
        "would be post-hoc. The verdict is a recorded judgement with the number "
        "in hand, and its reasoning is part of the artefact."),
}


def headroom_fraction(baseline_mean: float, mde: float) -> dict:
    """What share of the total achievable improvement must a model capture?

    On `log_volume_ratio_error` a perfect prediction scores 0, so the TOTAL
    available headroom is the baseline mean itself. The MDE expressed as a
    fraction of that is the whole amended criterion: it says how much of
    everything there is to win must be won before the cohort can see it.

    This works because the metric is a distance from a perfect score. It would
    NOT work on a similarity metric like Dice, where the ceiling is 1.0 and the
    available headroom is (1 - baseline).
    """
    if not baseline_mean or mde is None:
        return {"fraction": None, "note": "baseline mean or MDE unavailable"}
    frac = mde / baseline_mean
    return {
        "baseline_mean": baseline_mean,
        "total_available_headroom": baseline_mean,
        "mde": mde,
        "fraction_of_headroom_required": frac,
        "reading": (
            f"A perfect model scores 0, so the total winnable improvement is "
            f"{baseline_mean:.4f}. A model must capture at least {frac:.1%} of "
            f"everything there is to win before n = 26 can resolve it as an "
            f"improvement over persistence."),
        "caveat": (
            "The MDE is an UPPER BOUND — it assumes model-minus-baseline "
            "per-patient differences spread as widely as the baseline does "
            "between patients. Paired comparisons normally do better, so the "
            "true requirement is lower by an unknown amount. Recompute from the "
            "paired difference SD once the first C-rung exists, ONCE, and record "
            "it; re-deriving per rung turns a power threshold post-hoc."),
    }


def assess(persistence_result: dict, preregistered_metric_record: dict) -> dict:
    """Everything GATE-3 needs, with no verdict attached."""
    from ..stage3.persistence import minimum_detectable_effect
    metric = preregistered_metric_record["primary_metric"]
    mde_rec = minimum_detectable_effect(persistence_result, metric=metric)
    agg = None
    for v in persistence_result["overall"].values():
        if isinstance(v, dict) and v.get("metric") == metric:
            agg = v
            break
    hf = headroom_fraction(agg.get("mean") if agg else None, mde_rec.get("mde"))
    return {
        "gate": "GATE-3",
        "status": "ASSESSED_NO_VERDICT",
        "preregistered_metric": metric,
        "metric_preregistered_at": preregistered_metric_record.get("written_utc"),
        "baseline": agg,
        "mde": mde_rec,
        "headroom": hf,
        "literature": LITERATURE_FINDING,
        "amendment": AMENDMENT,
        "decision_is_human": (
            "This function reports. Call record_verdict() with GO or NO_GO and a "
            "substantive justification. GO -> conditioning ladder. NO_GO -> "
            "report baselines and the identifiability analysis, and do not run "
            "an ablation ladder whose rungs cannot be resolved (§17: a "
            "well-characterised negative result is a legitimate primary "
            "contribution at this n)."),
    }


def record_verdict(project_root, verdict: str, justification: str,
                   assessment: dict, overwrite: bool = False) -> dict:
    """Write the GATE-3 verdict. Append-only, like the metric pre-registration."""
    from ..utils.persist import save_artefact
    if verdict not in ("GO", "NO_GO"):
        raise ValueError("verdict must be 'GO' or 'NO_GO'")
    if not justification or len(justification.strip()) < 60:
        raise ValueError(
            "a substantive justification is required — this is the record of why "
            "the gate was closed the way it was, and it will be read by a "
            "reviewer who does not have this conversation")

    out = Path(project_root) / "10_EXPERIMENTS" / "v2_gate3_verdict.json"
    prior = json.loads(out.read_text()) if out.is_file() else None
    if prior and not overwrite:
        raise RuntimeError(
            f"GATE-3 verdict is ALREADY recorded as {prior.get('verdict')!r} at "
            f"{prior.get('written_utc')}. Pass overwrite=True only if the change "
            "is deliberate; the prior verdict is preserved in the record.")

    rec = {
        "manifest": "gate3_verdict",
        "gate": "GATE-3",
        "verdict": verdict,
        "justification": justification,
        "assessment_at_time_of_verdict": assessment,
        "consequence": (
            "Conditioning ladder §11.1 is warranted. Recompute the MDE from the "
            "paired difference SD once C0 exists, once, and record it."
            if verdict == "GO" else
            "The conditioning ladder is NOT run. Report the baselines and the "
            "identifiability analysis as the primary contribution (§17). Do not "
            "rescue this with architecture changes, target changes or added "
            "conditioning before it has been reported as-is (§11.1)."),
    }
    if prior:
        rec["superseded"] = prior
    rec["artefact"] = save_artefact(project_root, "10_EXPERIMENTS",
                                    "gate3_verdict", rec)
    return rec


def print_assessment(a: dict) -> None:
    line = "-" * 78
    print(line)
    print("GATE-3 ASSESSMENT  —  no verdict is issued by this report")
    print(line)
    b, m, h = a["baseline"] or {}, a["mde"], a["headroom"]
    print(f"  pre-registered metric : {a['preregistered_metric']}")
    print(f"  persistence baseline  : {b.get('mean')}  "
          f"[{b.get('ci_low')}, {b.get('ci_high')}]")
    print(f"  patients              : {b.get('n_patients')}   "
          f"pairs defined: {b.get('n_pairs_defined')}")
    print(f"  MDE (80% power)       : {m.get('mde')}")
    print(f"\n  {h.get('reading')}")
    print(f"\n  LITERATURE: {LITERATURE_FINDING['finding']}")
    print(f"\n  {LITERATURE_FINDING['what_it_does_not_license']}")
    print(f"\n  {LITERATURE_FINDING['verification_caveat']}")
    print(f"\n  AMENDMENT {AMENDMENT['id']}: {AMENDMENT['change']}")
    print(f"\n  {a['decision_is_human']}")
    print(line)
