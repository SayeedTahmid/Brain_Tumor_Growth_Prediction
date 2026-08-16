"""Why did C0's validation never beat persistence? Measure before concluding.

THE OBSERVATION. Over 16 validations on repeat 0 / fold 0 the model never beat
persistence on the pre-registered `log_volume_ratio_error` (best 0.4480 vs
0.4218). Training loss fell from 0.72 to 0.018 by step 500 and then sat flat for
3,500 steps while validation oscillated between 0.45 and 0.87. Meanwhile Dice
ran 0.53 for the model against ~0.47 for cohort persistence — better on the
demoted metric, worse on the pre-registered one.

THREE EXPLANATIONS, AND THEY ARE NOT INTERCHANGEABLE.

  A. DEGENERATE TASK. Predicting a mask from the same mask under BCE on patches
     that are ~99.9% background has an easy attractor: reproduce the input. That
     yields good overlap (Dice) and poor volume estimation (log-ratio), which is
     what was observed. This is an IMPLEMENTATION artefact.

  B. LOSS/METRIC MISMATCH. BCE optimises per-voxel classification; the
     pre-registered metric is a volume ratio. Nothing in training pushes toward
     correct total volume. Also an IMPLEMENTATION issue.

  C. NO SIGNAL AT C0. Mask-only input with no intensity, no Δt and no treatment
     may contain nothing learnable beyond copy-forward. This is a genuine
     SCIENTIFIC RESULT and precisely what the ladder exists to establish.

Reporting C when the truth is A or B would be a false negative dressed as a
finding. This module distinguishes them by measurement and asserts nothing.

DECISIVE QUANTITY: Dice(prediction, INPUT). If the model reproduces its input,
that runs near 1.0 while Dice(prediction, target) tracks the input-target
similarity — explanation A, and no conclusion about signal may be drawn until it
is fixed. If prediction differs materially from input yet still fails to beat
persistence, A is excluded and C becomes credible.

This module MEASURES ONLY. It changes no loss, no architecture, no lock.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def diagnose(model, cache, pairs: list, patch: int = 96, batch_size: int = 8,
             device: str = "cuda", amp: bool = True) -> dict:
    """Per-pair anatomy of what the model actually produced."""
    from .inference import predict_mask
    from ..stage3.persistence import dice, log_volume_ratio_error

    rows = []
    for p in pairs:
        a = cache.get(p["subject"], p["input_session"])
        b = cache.get(p["subject"], p["target_session"])
        if a is None or b is None:
            continue
        pred = predict_mask(model, a, patch=patch, batch_size=batch_size,
                            device=device, amp=amp)
        prev, ref = a.astype(bool), b.astype(bool)
        n_pred, n_prev, n_ref = int(pred.sum()), int(prev.sum()), int(ref.sum())
        rows.append({
            "subject": p["subject"],
            "input_session": p["input_session"],
            "target_session": p["target_session"],
            "n_input": n_prev, "n_pred": n_pred, "n_target": n_ref,
            # THE decisive number: how close is the prediction to the input?
            "dice_pred_vs_input": dice(pred, prev),
            "dice_pred_vs_target": dice(pred, ref),
            "dice_input_vs_target": dice(prev, ref),
            "identical_to_input": bool(np.array_equal(pred, prev)),
            "voxels_changed_from_input": int(np.logical_xor(pred, prev).sum()),
            "pred_over_target_ratio": (n_pred / n_ref) if n_ref else None,
            "pred_over_input_ratio": (n_pred / n_prev) if n_prev else None,
            "log_ratio_model": log_volume_ratio_error(pred, ref, prev),
            "log_ratio_persistence": log_volume_ratio_error(prev, ref, prev),
        })

    def m(key):
        v = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(v)) if v else None

    copy_dice = m("dice_pred_vs_input")
    changed = [r["voxels_changed_from_input"] for r in rows]
    over = [r["pred_over_target_ratio"] for r in rows
            if r["pred_over_target_ratio"] is not None]

    # Verdict thresholds are stated here rather than chosen after reading the
    # numbers. >0.95 mean Dice against the input is reproduction in all but
    # name; <0.70 means the model is doing something substantially different.
    if copy_dice is None:
        verdict, detail = "UNDETERMINED", "no defined comparisons"
    elif copy_dice > 0.95:
        verdict = "A_DEGENERATE_COPY"
        detail = (f"Mean Dice(prediction, INPUT) = {copy_dice:.4f}. The model is "
                  "reproducing its input. Its apparent Dice against the target is "
                  "inherited from input-target similarity, not learned. NO "
                  "conclusion about signal at C0 may be drawn until the loss or "
                  "task framing is fixed.")
    elif copy_dice > 0.70:
        verdict = "A_PARTIAL_COPY"
        detail = (f"Mean Dice(prediction, INPUT) = {copy_dice:.4f}. The model "
                  "departs from its input somewhat but stays close to it. "
                  "Explanation A is not excluded; treat any 'no signal' reading "
                  "as unsafe.")
    else:
        verdict = "NOT_A_COPY"
        detail = (f"Mean Dice(prediction, INPUT) = {copy_dice:.4f}. The model "
                  "produces something materially different from its input, so "
                  "explanation A is excluded. Failure to beat persistence is "
                  "then either a loss/metric mismatch (B) or genuine absence of "
                  "signal at C0 (C).")

    return {
        "check": "c0_degeneracy_diagnosis",
        "n_pairs": len(rows),
        "verdict": verdict,
        "detail": detail,
        "mean_dice_pred_vs_input": copy_dice,
        "mean_dice_pred_vs_target": m("dice_pred_vs_target"),
        "mean_dice_input_vs_target": m("dice_input_vs_target"),
        "n_predictions_identical_to_input": sum(
            1 for r in rows if r["identical_to_input"]),
        "mean_voxels_changed_from_input": float(np.mean(changed)) if changed else None,
        "median_pred_over_target_ratio": float(np.median(over)) if over else None,
        "volume_bias": (
            None if not over else
            "OVER-PREDICTS volume" if np.median(over) > 1.15 else
            "UNDER-PREDICTS volume" if np.median(over) < 0.85 else
            "volume roughly calibrated"),
        "mean_log_ratio_model": m("log_ratio_model"),
        "mean_log_ratio_persistence": m("log_ratio_persistence"),
        "verdict_thresholds": {
            "degenerate_copy": ">0.95 mean Dice(pred, input)",
            "partial_copy": ">0.70",
            "note": "stated in source, not chosen after reading the numbers"},
        "what_this_does_not_establish": (
            "Nothing about whether C0 has signal, unless the verdict is "
            "NOT_A_COPY. A degenerate or partial copy means the pipeline has "
            "not yet posed the question the ladder is meant to answer."),
        "per_pair": rows,
    }


def per_patient_spread(diag: dict) -> dict:
    """Validation variance across the 5 held-out patients.

    The probe's validation swung between 0.45 and 0.87 across steps. Some of
    that is genuine instability; some is 5 patients being a small sample. This
    separates them: large between-patient spread at a FIXED step means the
    validation estimate itself is noisy and step-to-step movement should not be
    over-read.
    """
    by = {}
    for r in diag["per_pair"]:
        if r["log_ratio_model"] is not None:
            by.setdefault(r["subject"], []).append(r["log_ratio_model"])
    per = {k: float(np.mean(v)) for k, v in by.items()}
    vals = list(per.values())
    return {
        "per_patient_mean_log_ratio": per,
        "n_patients": len(per),
        "between_patient_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        "range": [min(vals), max(vals)] if vals else None,
        "reading": (
            "If the between-patient SD is comparable to the step-to-step "
            "movement seen during the probe, the validation estimate is noise-"
            "dominated at 5 patients and no single validation point should be "
            "read as a plateau or as a result."),
    }


def print_report(d: dict, spread: dict | None = None) -> None:
    line = "-" * 78
    print(line)
    print("C0 DIAGNOSIS  —  is the model learning, or reproducing its input?")
    print(line)
    print(f"  pairs evaluated            : {d['n_pairs']}")
    print(f"  Dice(prediction, INPUT)    : {d['mean_dice_pred_vs_input']:.4f}   <-- decisive")
    print(f"  Dice(prediction, target)   : {d['mean_dice_pred_vs_target']:.4f}")
    print(f"  Dice(input, target)        : {d['mean_dice_input_vs_target']:.4f}")
    print(f"  predictions identical to input: "
          f"{d['n_predictions_identical_to_input']} / {d['n_pairs']}")
    print(f"  mean voxels changed        : {d['mean_voxels_changed_from_input']:.0f}")
    print(f"  median pred/target volume  : {d['median_pred_over_target_ratio']}"
          f"   ({d['volume_bias']})")
    print(f"  log-ratio  model {d['mean_log_ratio_model']:.4f}  vs  "
          f"persistence {d['mean_log_ratio_persistence']:.4f}")
    print(f"\n  VERDICT: {d['verdict']}")
    print(f"  {d['detail']}")
    if spread:
        print(f"\n  per-patient log-ratio: {spread['per_patient_mean_log_ratio']}")
        print(f"  between-patient SD   : {spread['between_patient_sd']}")
        print(f"  {spread['reading']}")
    print(f"\n  {d['what_this_does_not_establish']}")
    print(line)
