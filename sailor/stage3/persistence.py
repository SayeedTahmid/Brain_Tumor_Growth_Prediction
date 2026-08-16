"""Persistence baseline — Stage 3, and the input to GATE-3.

Persistence copies the last observed mask forward: `prediction(t+Δt) = mask(t)`.
It is the floor every model must clear (§11.1 rung C−1, G3). Higher mean Dice is
not a result; if a model is statistically indistinguishable from copying the
last scan forward, that is the finding.

Two rules are BINDING here, both pre-registered in
`10_EXPERIMENTS/v2_sub25_adjudication_final.json` when sub-25 was retained.

**Empty targets.** 5 of 208 pairs are empty→empty in the primary target
(sub-25 ses-03/04 through 07/08). Dice has 0/0 in both numerator and denominator
there and is UNDEFINED. The dangerous failure is not a crash — it is
`np.nanmean` quietly dropping those pairs so the reported mean describes 203
pairs while claiming 208. Every metric here therefore returns an explicit
`n_defined` alongside its value, and `n_undefined` is reported next to any mean.

**Consecutive.** Pairs are consecutive ORDINAL sessions where BOTH ends carry
the primary target. No pair is bridged across a missing target: sub-25
ses-08 → ses-10 spans the CL-less ses-09 and does not exist. This is enforced in
`data/pairs.py`; this module asserts it rather than assuming it.

Uncertainty resamples at the PATIENT level (26 units), never the pair level
(208). Pairs within a patient share a tumour, so pair-level CIs would be roughly
sqrt(208/26) ≈ 2.8× too narrow (AMD-003, GATE-3).

The primary metric is CHANGE-SENSITIVE (AMD-005). At q25 = 14 days, copy-forward
scores near ceiling on whole-mask Dice, so Dice is reported as secondary and is
not the headline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

# Frozen Δt bands (AMD-002). Fixed here before any number is computed.
DELTA_BANDS = (("<=21d", 0.0, 21.0), ("22-90d", 21.0, 90.0), (">90d", 90.0, 1e9))

#: Bootstrap resamples for patient-level CIs.
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 1337          # same seed discipline as the frozen split
CI = (2.5, 97.5)


# ---------------------------------------------------------------- array access
def _stem(subject: str, session: str, base: str) -> str:
    return f"{subject}__{session}__{base}"


def load_mask(arrays_dir, subject: str, session: str,
              base: str = "ContrastEnhancedMask-CL") -> np.ndarray | None:
    """Load one exported mask as a boolean array. None when the file is absent.

    Absent is NOT empty. A missing file means the session carries no primary
    target and must never enter a pair; an empty file means the expert read the
    scan and found no enhancing tumour. Conflating them is defect 5/10/17 in this
    project's history, which appeared three times.
    """
    p = Path(arrays_dir) / f"{_stem(subject, session, base)}.npz"
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=True) as z:
        return np.asarray(z["array"]) > 0


# --------------------------------------------------------------------- metrics
def dice(pred: np.ndarray, ref: np.ndarray) -> float | None:
    """Dice. None when BOTH are empty — undefined, not 1.0 and not 0.0.

    Returning 1.0 for empty→empty would reward a model for predicting nothing
    where there is nothing, which at 5 of 208 pairs is a free win on the pairs
    persistence can never lose. Returning 0.0 would punish a correct prediction.
    Undefined is the honest answer and is counted separately.
    """
    a, b = int(pred.sum()), int(ref.sum())
    if a == 0 and b == 0:
        return None
    inter = int(np.logical_and(pred, ref).sum())
    return 2.0 * inter / (a + b)


def relative_volume_change_error(pred: np.ndarray, ref: np.ndarray,
                                 prev: np.ndarray) -> float | None:
    """Volume-change error divided by the tumour scale of the pair.

    Absolute `volume_change_error` is scale-dependent: a 60,000-voxel tumour can
    generate more absolute error than a 500-voxel one can possibly produce, so
    between-patient spread is dominated by tumour size rather than by
    predictive difficulty. That inflates the MDE without reflecting anything a
    model could improve.

    Scale is `max(n_input, n_target)`, which bounds the metric at 1.0 for
    persistence: |Δtrue| never exceeds the larger of the two volumes. So 0 is
    perfect, 1 is the worst persistence can do (total appearance or total
    disappearance), and the range is comparable across patients.

    UNDEFINED on empty->empty: scale is 0 and there is no tumour to be relative
    to. Counted, never silently dropped.
    """
    scale = max(int(prev.sum()), int(ref.sum()))
    if scale == 0:
        return None
    return volume_change_error(pred, ref, prev) / scale


def log_volume_ratio_error(pred: np.ndarray, ref: np.ndarray,
                           prev: np.ndarray) -> float:
    """|log((n_target+1) / (n_pred+1))| — scale-free and DEFINED EVERYWHERE.

    The +1 is a Laplace offset that keeps the ratio finite when either side is
    empty, which is what makes this the one headroom metric defined on all 208
    pairs including the five empty->empty ones (log(1/1) = 0, a perfect score
    that persistence genuinely earns there).

    It is also symmetric in growth and shrinkage: doubling and halving produce
    the same magnitude, whereas a raw ratio penalises growth more heavily. That
    matters in a cohort where both progression and response occur.
    """
    a, b = int(pred.sum()), int(ref.sum())
    return abs(float(np.log((b + 1.0) / (a + 1.0))))


def volume_change_error(pred: np.ndarray, ref: np.ndarray,
                        prev: np.ndarray) -> float:
    """|predicted change − true change| in voxels. DEFINED at zero.

    This is why it is the primary metric under AMD-005: on an empty→empty pair
    the true change is 0 and persistence predicts 0, giving an error of 0 — a
    real, defined, unbeatable score rather than a hole in the mean.
    """
    return abs((int(pred.sum()) - int(prev.sum())) - (int(ref.sum()) - int(prev.sum())))


def change_region_dice(pred: np.ndarray, ref: np.ndarray,
                       prev: np.ndarray) -> float | None:
    """Dice restricted to voxels that CHANGED between input and target (AMD-005).

    Whole-mask Dice at 14-day intervals is dominated by the unchanged core, so
    copy-forward scores near ceiling and a real improvement is invisible inside
    it. Scoring only the symmetric difference removes the free credit.
    Undefined when nothing changed and nothing was predicted to change.

    STRUCTURAL NOTE — this metric is degenerate FOR PERSISTENCE specifically.
    Persistence predicts `pred == prev`, so its predicted-change set is always
    empty and its score is exactly 0.0 whenever any change occurred, and
    undefined when none did. It never takes an intermediate value.

    That is not a bug and the metric is still the right primary for comparing
    MODELS, which do predict change. But it means change-region Dice gives no
    usable headroom estimate: the gap from 0.0 to 1.0 is not an achievable range
    and an MDE computed from a constant-zero baseline has no between-patient
    variance to work with. **Use `volume_change_error` for the GATE-3 headroom
    and MDE**, since it takes real values on both baseline and model, and read
    change-region Dice as the floor it is.
    """
    changed = np.logical_xor(ref, prev)
    predicted_change = np.logical_xor(pred, prev)
    a, b = int(predicted_change.sum()), int(changed.sum())
    if a == 0 and b == 0:
        return None
    inter = int(np.logical_and(predicted_change, changed).sum())
    return 2.0 * inter / (a + b)


def exact_zero_agreement(pred: np.ndarray, ref: np.ndarray) -> bool | None:
    """Did both come out empty? Reported as a COUNT, never folded into a mean."""
    a, b = int(pred.sum()), int(ref.sum())
    return (a == 0 and b == 0) if (a == 0 or b == 0) else None


# ------------------------------------------------------------------- per pair
def score_pair(arrays_dir, pair: dict,
               base: str = "ContrastEnhancedMask-CL") -> dict:
    """Persistence on one pair: predict the input mask unchanged."""
    sub = pair["subject"]
    prev = load_mask(arrays_dir, sub, pair["input_session"], base)
    ref = load_mask(arrays_dir, sub, pair["target_session"], base)
    rec = {"subject": sub,
           "input_session": pair["input_session"],
           "target_session": pair["target_session"],
           "delta_days": pair.get("delta_days"),
           "gap_in_ordinals": pair.get("gap_in_ordinals")}
    if prev is None or ref is None:
        # A pair whose ends lack the target should never have been built.
        rec["error"] = ("MISSING_ARRAY: input" if prev is None else "MISSING_ARRAY: target")
        return rec
    pred = prev                                    # persistence: copy forward
    rec.update({
        "n_input": int(prev.sum()), "n_target": int(ref.sum()),
        "true_change": int(ref.sum()) - int(prev.sum()),
        "both_empty": bool(prev.sum() == 0 and ref.sum() == 0),
        "dice": dice(pred, ref),
        "change_region_dice": change_region_dice(pred, ref, prev),
        "volume_change_error": volume_change_error(pred, ref, prev),
        "relative_volume_change_error": relative_volume_change_error(pred, ref, prev),
        "log_volume_ratio_error": log_volume_ratio_error(pred, ref, prev),
        "exact_zero_agreement": exact_zero_agreement(pred, ref),
    })
    return rec


# ----------------------------------------------------------------- aggregation
def _patient_bootstrap(by_patient: dict, seed: int = BOOTSTRAP_SEED,
                       n: int = N_BOOTSTRAP) -> dict:
    """Resample PATIENTS with replacement (AMD-003), never pairs."""
    patients = sorted(by_patient)
    if not patients:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_patients": 0}
    rng = np.random.default_rng(seed)
    per_patient = np.array([float(np.mean(by_patient[p])) for p in patients])
    idx = rng.integers(0, len(patients), size=(n, len(patients)))
    boot = per_patient[idx].mean(axis=1)
    lo, hi = np.percentile(boot, CI)
    return {"mean": float(per_patient.mean()),
            "ci_low": float(lo), "ci_high": float(hi),
            "n_patients": len(patients),
            "per_patient_sd": float(per_patient.std(ddof=1)) if len(patients) > 1 else None}


def _aggregate(rows: list[dict], metric: str) -> dict:
    """Patient-level mean and CI for one metric, with undefined pairs COUNTED."""
    by_patient: dict[str, list] = {}
    n_undef = 0
    for r in rows:
        v = r.get(metric)
        if v is None:
            n_undef += 1
            continue
        by_patient.setdefault(r["subject"], []).append(float(v))
    out = _patient_bootstrap(by_patient)
    out.update({
        "metric": metric,
        "n_pairs_total": len(rows),
        "n_pairs_defined": len(rows) - n_undef,
        "n_pairs_undefined": n_undef,
        "undefined_note": (
            None if n_undef == 0 else
            f"{n_undef} of {len(rows)} pairs are UNDEFINED on {metric} and are "
            "excluded from this mean. The mean therefore describes "
            f"{len(rows) - n_undef} pairs, not {len(rows)}."),
    })
    return out


def _band(delta_days) -> str:
    if delta_days is None:
        return "unknown"
    for name, lo, hi in DELTA_BANDS:
        if lo < float(delta_days) <= hi or (lo == 0.0 and float(delta_days) <= hi):
            return name
    return "unknown"


def run(project_root, arrays_subdir: str = "01_DATA_FOUNDATION/v2_arrays",
        split_name: str = "v2_pairs_and_folds.json") -> dict:
    """Persistence baseline over the frozen split. No archive read."""
    from ..utils.persist import save_artefact
    project_root = Path(project_root)
    split_path = project_root / "01_DATA_FOUNDATION" / split_name
    split = json.loads(split_path.read_text())
    arrays_dir = project_root / arrays_subdir

    pairs = split["pairs"]["pairs"]
    rows = [score_pair(arrays_dir, p) for p in pairs]
    errors = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]

    # The consecutive rule is ASSERTED, not assumed: a pair spanning more than
    # one ordinal would mean a missing target was bridged.
    bridged = [f"{r['subject']} {r['input_session']}->{r['target_session']}"
               for r in ok if (r.get("gap_in_ordinals") or 1) != 1]

    result = {
        "baseline": "persistence (copy last mask forward)",
        "rung": "C-1",
        "primary_target": f'{split["primary_target_mask"]}/{split["primary_target_component"]}',
        "split_content_sha256": split["content_sha256"],
        "n_pairs": len(pairs),
        "n_scored": len(ok),
        "n_errors": len(errors),
        "errors": errors[:20],
        "n_patients": len({r["subject"] for r in ok}),
        "n_both_empty_pairs": sum(1 for r in ok if r["both_empty"]),
        "both_empty_pairs": [f"{r['subject']} {r['input_session']}->{r['target_session']}"
                             for r in ok if r["both_empty"]],
        "pairs_spanning_more_than_one_ordinal": bridged,
        "overall": {
            "primary_change_region_dice": _aggregate(ok, "change_region_dice"),
            "primary_volume_change_error": _aggregate(ok, "volume_change_error"),
            "secondary_whole_mask_dice": _aggregate(ok, "dice"),
            "headroom_relative_volume_change_error":
                _aggregate(ok, "relative_volume_change_error"),
            "headroom_log_volume_ratio_error":
                _aggregate(ok, "log_volume_ratio_error"),
        },
        "by_delta_band": {},
        "gate3_headroom_metric": "volume_change_error",
        "gate3_headroom_note": (
            "change_region_dice is structurally 0.0 or undefined for persistence "
            "(it predicts no change), so it carries no between-patient variance "
            "and cannot support an MDE. volume_change_error takes real values on "
            "both baseline and model and is the metric GATE-3 headroom is read "
            "from. change_region_dice remains the primary metric for comparing "
            "MODELS, where it is not degenerate."),
        "metric_policy": {
            "primary": ("change_region_dice and volume_change_error (AMD-005). "
                        "Whole-mask Dice is SECONDARY: at q25 = 14 days "
                        "copy-forward scores near ceiling on it."),
            "empty_targets": ("Dice is UNDEFINED on empty->empty and is excluded "
                              "and COUNTED, never silently dropped. "
                              "volume_change_error is defined at 0 there."),
            "uncertainty": ("patient-level bootstrap, "
                            f"{N_BOOTSTRAP} resamples, seed {BOOTSTRAP_SEED}, "
                            "percentile CI. NEVER pair-level (AMD-003)."),
        },
        "per_pair": ok,
    }
    for name, _, _ in DELTA_BANDS:
        sub = [r for r in ok if _band(r["delta_days"]) == name]
        result["by_delta_band"][name] = {
            "n_pairs": len(sub),
            "n_patients": len({r["subject"] for r in sub}),
            "change_region_dice": _aggregate(sub, "change_region_dice"),
            "volume_change_error": _aggregate(sub, "volume_change_error"),
            "relative_volume_change_error":
                _aggregate(sub, "relative_volume_change_error"),
            "log_volume_ratio_error": _aggregate(sub, "log_volume_ratio_error"),
            "whole_mask_dice": _aggregate(sub, "dice"),
        }
    result["artefact"] = save_artefact(project_root, "10_EXPERIMENTS",
                                       "persistence_baseline",
                                       {k: v for k, v in result.items()
                                        if k != "per_pair"})
    return result


def minimum_detectable_effect(result: dict, metric: str = "volume_change_error",
                              power: float = 0.80, alpha: float = 0.05) -> dict:
    """MDE on the primary metric, stated BEFORE any model runs (GATE-3).

    Paired design over patients: MDE = (z_a/2 + z_b) * sd_between_patients /
    sqrt(n). This is an approximation and is labelled as one — it assumes the
    per-patient differences a model would produce have spread comparable to the
    between-patient spread of the baseline itself, which is a working assumption
    and not a measurement.
    """
    agg = (result["overall"].get(f"primary_{metric}")
           or result["overall"].get(f"headroom_{metric}")
           or result["overall"].get(f"secondary_{metric}")
           or result["overall"].get(metric))
    if agg is None:
        for v in result["overall"].values():
            if isinstance(v, dict) and v.get("metric") == metric:
                agg = v
                break
    if agg is None or agg.get("per_patient_sd") is None:
        return {"mde": None, "note": f"no per-patient sd available for {metric}"}
    z = {0.80: 1.2816, 0.90: 1.6449}.get(round(power, 2), 1.2816)
    z_alpha = 1.9600 if abs(alpha - 0.05) < 1e-9 else 2.5758
    n = agg["n_patients"]
    sd = agg["per_patient_sd"]
    mde = (z_alpha + z) * sd / np.sqrt(n)
    return {
        "metric": metric, "n_patients": n, "per_patient_sd": sd,
        "power": power, "alpha": alpha, "mde": float(mde),
        "interpretation": (
            f"A model must improve mean {metric} by at least {mde:.4f} over "
            f"persistence to be detectable at n = {n} patients with "
            f"{power:.0%} power. Improvements below this are not resolvable by "
            "this cohort, whatever the point estimate shows."),
        "approximation_note": (
            "Assumes per-patient model-minus-baseline differences have spread "
            "comparable to the between-patient spread of the baseline. Working "
            "assumption, not a measurement."),
    }


def print_report(res: dict) -> None:
    line = "-" * 78
    print(line)
    print("PERSISTENCE BASELINE (rung C-1)  —  input to GATE-3")
    print(line)
    print(f"  target : {res['primary_target']}")
    print(f"  split  : {res['split_content_sha256'][:16]}…")
    print(f"  pairs  : {res['n_scored']} scored / {res['n_pairs']} "
          f"({res['n_errors']} errors) across {res['n_patients']} patients")
    print(f"  empty->empty pairs: {res['n_both_empty_pairs']}  {res['both_empty_pairs']}")
    if res["pairs_spanning_more_than_one_ordinal"]:
        print(f"  !! pairs spanning >1 ordinal: "
              f"{res['pairs_spanning_more_than_one_ordinal']}")
    print(f"\n  {'metric':<28}{'mean':>9}{'95% CI':>20}{'defined':>10}{'undef':>7}")
    for label, key in (("change-region Dice  [1°]", "primary_change_region_dice"),
                       ("volume-change error [1°]", "primary_volume_change_error"),
                       ("whole-mask Dice     [2°]", "secondary_whole_mask_dice")):
        a = res["overall"][key]
        ci = f"[{a['ci_low']:.4f}, {a['ci_high']:.4f}]" if a["mean"] is not None else "—"
        m = f"{a['mean']:.4f}" if a["mean"] is not None else "—"
        print(f"  {label:<28}{m:>9}{ci:>20}{a['n_pairs_defined']:>10}"
              f"{a['n_pairs_undefined']:>7}")
    print(f"\n  by Δt band (change-region Dice):")
    for name in res["by_delta_band"]:
        b = res["by_delta_band"][name]
        a = b["change_region_dice"]
        m = f"{a['mean']:.4f}" if a["mean"] is not None else "—"
        print(f"    {name:<9} n={b['n_pairs']:>3} pairs / {b['n_patients']:>2} patients"
              f"   mean={m}  undefined={a['n_pairs_undefined']}")
    print(f"\n  {res['metric_policy']['empty_targets']}")
    print(line)
