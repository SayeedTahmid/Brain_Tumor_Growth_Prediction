"""DIAGNOSTIC ONLY — a loss variant with a volume term, and the probe that uses it.

THE OBSERVATION THIS EXISTS TO EXPLAIN. Three rungs now sit on one side of the
persistence floor and the ordering is monotone:

    persistence     0.4928
    C1  (+Δt)       0.5592   gap +0.0664
    C0-residual     0.5723   gap +0.0795
    C0-direct       0.5991   gap +0.1063

The residual rungs were VERIFIED to start exactly at 0.4928 and trained AWAY from
it. Adding Δt shifted the endpoint by 0.0131 — under a quarter of the frozen MDE
— and changed nothing structural. Two rungs departing a guaranteed floor under
different information points at the OBJECTIVE, not the information.

THE HYPOTHESIS. `BCE + soft Dice` rewards spatial overlap with the target. The
pre-registered metric is `log_volume_ratio_error`, a volume quantity. Nothing in
training penalises getting total volume wrong, so a model can reduce its loss
while drifting away on the metric that decides the rung — which is exactly the
Dice-versus-volume divergence AMD-005 anticipated and which every rung has shown
(model Dice ABOVE persistence, model log-ratio BELOW it).

WHAT THIS MODULE IS NOT. It is not a new frozen loss and it does not touch
`loss.py`. AMD-007 fixes one loss across rungs, so adopting this would invalidate
C0 and C1 and require re-running both (~10 h). This runs ONE repeat (5 fits,
~1 h) to decide whether that cost is justified.

    model holds the floor with the volume term
        -> the ladder has been measuring an optimisation artefact. Every rung
           needs re-running under a corrected objective before any rung result
           can be interpreted.
    model still drifts
        -> the finding is robust to the objective. C2-C4 proceed as designed and
           the null is reportable as a property of the data rather than of the
           training criterion.

THE RISK OF THIS EXPERIMENT, STATED. Adding a term that directly penalises volume
error moves the loss closer to the evaluation metric. That is legitimate as a
DIAGNOSTIC — it asks whether the model CAN hold the floor when told to — but it
would need care if adopted: a loss containing the evaluation metric makes
"beats persistence" less meaningful, because the model is being trained on the
thing it is scored by. Any adoption must therefore be recorded as an amendment
with that trade-off written down, not slipped in as a bug fix.
"""

from __future__ import annotations

import numpy as np

#: Weight on the volume term. Deliberately modest: the point is to test whether
#: a volume signal changes the trajectory at all, not to optimise the metric.
VOLUME_WEIGHT = 0.5
SMOOTH = 1.0

# --------------------------------------------------------------------------
# VARIANT B — a volume term that is NOT the evaluation metric.
#
# Probe A (log-ratio term) collapsed the persistence gap from +0.0795 to
# +0.0052, a 94% reduction, confirming that the ladder's deficits were largely
# an optimisation artefact. But that term is a differentiable analogue of
# `log_volume_ratio_error`, the metric the rung is scored by. Adopting it would
# make "beats persistence" weaker evidence and a reviewer would rightly say so.
#
# This variant corrects the same drift with a term that shares NO functional
# form with the metric: a RELATIVE ABSOLUTE volume error, normalised by target
# volume rather than expressed as a log ratio.
#
#     metric   |log((V_true + 1) / (V_pred + 1))|      log of a ratio
#     variant  |V_pred - V_true| / (V_true + eps)      linear, normalised
#
# They correlate — any volume signal must — but the variant is not the metric
# rescaled: it is asymmetric where the metric is symmetric (over-prediction by
# 2x costs 1.0, under-prediction to zero costs 1.0, whereas the log ratio gives
# 0.69 both ways), and it is unbounded above where the log ratio is not.
#
# WHY NORMALISED RATHER THAN RAW VOXELS. A raw |V_pred - V_true| would be
# dominated by the largest tumours — the same scale-dependence that inflated
# the absolute volume-error metric's between-patient SD to 3995 against a mean
# of 4464, and which motivated the scale-free primary metric in the first place.
VARIANT_B_WEIGHT = 0.5
VOLUME_EPS = 32.0   # ~ the smallest annotated lesion; keeps empty targets sane

CONFIG = {
    "loss": "BCEWithLogits + soft Dice + log-volume-ratio term",
    "volume_weight": VOLUME_WEIGHT,
    "status": "DIAGNOSTIC — NOT the frozen training loss",
    "frozen_loss_untouched": "sailor.stage4.loss.CONFIG is unchanged",
    "if_adopted": ("would invalidate C0 and C1 under AMD-007 and require "
                   "re-running both; must be recorded as an amendment carrying "
                   "the train-on-the-metric trade-off"),
}


def make_relative_volume_loss(volume_weight: float = VARIANT_B_WEIGHT):
    """BCE + soft Dice + RELATIVE ABSOLUTE volume error (variant B).

    The volume term is `|V_pred - V_true| / (V_true + eps)` on SOFT volumes, so
    it is differentiable and scale-free without being the evaluation metric in
    disguise. `eps` keeps the denominator finite for the five empty->empty pairs
    retained with sub-25, where V_true is 0 and any pure ratio would explode.
    """
    import torch
    import torch.nn as nn

    bce = nn.BCEWithLogitsLoss()

    def soft_dice(logits, target):
        p = torch.sigmoid(logits)
        dims = tuple(range(1, p.ndim))
        inter = (p * target).sum(dims)
        denom = p.sum(dims) + target.sum(dims)
        return 1.0 - ((2.0 * inter + SMOOTH) / (denom + SMOOTH)).mean()

    def rel_volume(logits, target):
        p = torch.sigmoid(logits)
        dims = tuple(range(1, p.ndim))
        v_pred, v_true = p.sum(dims), target.sum(dims)
        return (torch.abs(v_pred - v_true) / (v_true + VOLUME_EPS)).mean()

    def loss(logits, target):
        return (bce(logits, target)
                + soft_dice(logits, target)
                + volume_weight * rel_volume(logits, target))

    return loss


CONFIG_B = {
    "loss": "BCEWithLogits + soft Dice + relative absolute volume error",
    "volume_weight": VARIANT_B_WEIGHT,
    "volume_eps": VOLUME_EPS,
    "status": "DIAGNOSTIC variant B — NOT the frozen training loss",
    "why_not_variant_a": (
        "Variant A's term is a differentiable analogue of "
        "log_volume_ratio_error, the metric the rung is scored by. Adopting it "
        "would make 'beats persistence' weaker evidence. This term corrects the "
        "same drift without sharing the metric's functional form: linear and "
        "asymmetric where the metric is logarithmic and symmetric."),
    "probe_a_result": {"gap_before": 0.0795, "gap_after": 0.0052,
                       "reduction": "94%"},
}


def make_diagnostic_loss(volume_weight: float = VOLUME_WEIGHT):
    """BCE + soft Dice + a differentiable analogue of the primary metric.

    The volume term is `|log((V_target+1)/(V_pred+1))|` with V_pred the SOFT
    volume (sum of probabilities), which keeps it differentiable. It mirrors
    `log_volume_ratio_error` without being identical to it: the metric
    thresholds at 0.5 and this does not.
    """
    import torch
    import torch.nn as nn

    bce = nn.BCEWithLogitsLoss()

    def soft_dice(logits, target):
        p = torch.sigmoid(logits)
        dims = tuple(range(1, p.ndim))
        inter = (p * target).sum(dims)
        denom = p.sum(dims) + target.sum(dims)
        return 1.0 - ((2.0 * inter + SMOOTH) / (denom + SMOOTH)).mean()

    def volume_term(logits, target):
        p = torch.sigmoid(logits)
        dims = tuple(range(1, p.ndim))
        v_pred = p.sum(dims)
        v_true = target.sum(dims)
        return torch.abs(torch.log((v_true + 1.0) / (v_pred + 1.0))).mean()

    def loss(logits, target):
        return (bce(logits, target)
                + soft_dice(logits, target)
                + volume_weight * volume_term(logits, target))

    return loss


def run_probe(project_root, cache, split: dict, model_fn, rung_name: str = "DIAG",
              steps_per_fit: int = 2000, batch_size: int = 8,
              device: str = "cuda", amp: bool = True, repeat: int = 0,
              cond_fn_factory=None, loss_fn=None, loss_config=None) -> dict:
    """One repeat (5 fits) under a diagnostic loss. Compares against C0res."""
    import torch
    from ..utils.persist import save_artefact
    from pathlib import Path
    from .patches import CachedPairPatchSampler
    from .inference import evaluate_fold
    from .convergence import fold_pairs
    from .train import train_fold
    from . import loss as frozen_loss

    root = Path(project_root)
    lossf = make_diagnostic_loss() if loss_fn is None else loss_fn
    rows, fits = [], []
    n_folds = len(split["folds"]["repeats"][repeat]["folds"])

    for fold in range(n_folds):
        tr_pairs, te_pairs = fold_pairs(split, repeat, fold)
        tr_subj = {p["subject"] for p in tr_pairs}
        te_subj = {p["subject"] for p in te_pairs}
        if tr_subj & te_subj:
            raise RuntimeError(f"LEAKAGE r{repeat}f{fold}")
        cfn = None if cond_fn_factory is None else cond_fn_factory(tr_pairs)
        sampler = CachedPairPatchSampler(cache, tr_pairs, cond_fn=cfn)
        print(f"[{rung_name} r{repeat}f{fold}] train {len(tr_subj)}p "
              f"test {len(te_subj)}p")
        model = model_fn()
        # Checkpoint path is distinct so a diagnostic never collides with, or
        # resumes from, an official rung.
        train_fold(model, sampler, steps=steps_per_fit, batch_size=batch_size,
                   device=device, amp=amp,
                   checkpoint_path=root / "11_CHECKPOINTS" /
                                   f"{rung_name}_r{repeat}f{fold}.pt",
                   resume=True, log_every=1000, loss_fn=lossf)
        ev = evaluate_fold(model, cache, te_pairs, batch_size=batch_size,
                           device=device, amp=amp, cond_fn=cfn)
        fits.append({"fold": fold, "n_test_patients": len(te_subj)})
        rows.extend(ev["per_pair"])

    def agg(key):
        by = {}
        for r in rows:
            if r.get(key) is not None:
                by.setdefault(r["subject"], []).append(float(r[key]))
        if not by:
            return None
        per = [float(np.mean(v)) for v in by.values()]
        return {"mean": float(np.mean(per)), "n_patients": len(per),
                "sd": float(np.std(per, ddof=1)) if len(per) > 1 else None}

    m, p_ = agg("model_log_ratio"), agg("pers_log_ratio")
    gap = None if not (m and p_) else m["mean"] - p_["mean"]
    return {
        "probe": "diagnostic_volume_loss",
        "rung_name": rung_name,
        "repeat": repeat,
        "n_fits": len(fits),
        "diagnostic_loss": loss_config or CONFIG,
        "frozen_loss": frozen_loss.CONFIG["loss"],
        "model": m, "persistence": p_,
        "gap_vs_persistence": gap,
        "model_dice": agg("model_dice"),
        "persistence_dice": agg("pers_dice"),
        "reference_gaps": {
            "C0_direct": 0.1063, "C0_residual": 0.0795, "C1": 0.0664},
        "frozen_mde": 0.0555,
        "interpretation": (
            None if gap is None else
            ("HOLDS THE FLOOR — the volume term keeps the model at or below "
             "persistence. The ladder has been measuring an optimisation "
             "artefact; every rung needs re-running under a corrected objective "
             "before any rung result can be interpreted."
             if gap <= 0 else
             "STILL DRIFTS — the model departs the floor even when the loss "
             "penalises volume error directly. The finding is robust to the "
             "objective, C2-C4 proceed as designed, and the null is reportable "
             "as a property of the data rather than of the training criterion."
             if gap > 0.0555 else
             "PARTIAL — the gap shrank below the MDE but did not reach the "
             "floor. Neither reading is clean; report the number and decide "
             "explicitly rather than rounding it either way.")),
        "adoption_caveat": (
            "This loss contains a differentiable analogue of the evaluation "
            "metric. Adopting it would make 'beats persistence' weaker evidence, "
            "because the model would be trained on what it is scored by. Any "
            "adoption must be an amendment carrying that trade-off, not a bug "
            "fix."),
    }


def print_report(r: dict) -> None:
    line = "-" * 78
    print(line)
    print("DIAGNOSTIC — does a volume term keep the model at the floor?")
    print(line)
    print(f"  loss  : {r['diagnostic_loss']['loss']}")
    print(f"  frozen: {r['frozen_loss']}  (UNCHANGED)")
    print(f"  fits  : {r['n_fits']} (repeat {r['repeat']} only)")
    print(f"\n  model log-ratio       : {r['model']['mean']:.4f}")
    print(f"  persistence log-ratio : {r['persistence']['mean']:.4f}")
    print(f"  gap                   : {r['gap_vs_persistence']:+.4f}")
    print(f"\n  for reference — gaps under the frozen loss:")
    for k, v in r["reference_gaps"].items():
        print(f"    {k:<12} {v:+.4f}")
    print(f"    frozen MDE   {r['frozen_mde']:.4f}")
    print(f"\n  {r['interpretation']}")
    print(f"\n  {r['adoption_caveat']}")
    print(line)
