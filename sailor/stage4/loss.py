"""The training loss — fixed once, identical across C0-C4 and P1-P3.

WHY IT CHANGED, AND WHEN. The C0 convergence probe produced a model that:

    Dice(prediction, INPUT)   0.8000     — close to, but not identical to, input
    Dice(prediction, target)  0.4923     — SLIGHTLY WORSE than copy-forward
    Dice(input, target)       0.5041
    median predicted/target volume  0.846  — systematically UNDER-predicts by ~15%
    voxels moved from input   6,055 (mean)

So the model learned something — a bias toward predicting less tumour — and that
something made it worse than doing nothing. Pure BCE on patches that are ~99.9%
background rewards predicting fewer positives: the cheapest way to cut
cross-entropy on a sparse target is to shrink the prediction. The model obliged,
and the pre-registered `log_volume_ratio_error` punished exactly that.

That is a LOSS/METRIC MISALIGNMENT (explanation B in the diagnosis), not an
absence of signal. Reporting "C0 has no signal" from a run whose loss pushed
toward shrinkage would be a false negative published as a finding.

THE MECHANISM, MEASURED RATHER THAN ASSERTED. An initial explanation — that BCE
penalises under-prediction more leniently than over-prediction — was tested and
REFUTED: on a 1.6%-foreground target, BCE scores equal-magnitude under- and
over-errors identically (ratio 1.000). The real cause is gradient mass. At a
neutral starting point (probability 0.5 everywhere):

    BCE only    foreground grad -0.0078 | background +0.4922 | ratio 63.0x
    BCE + Dice  foreground grad -0.0227 | background +0.5069 | ratio 22.3x

The background gradient is 63x the foreground's under BCE, which is exactly the
class ratio: the optimiser's cheapest descent direction is to drive everything
toward background, and that is under-prediction. Adding soft Dice triples the
foreground pull and cuts the imbalance to 22x, because soft Dice is normalised
by predicted plus target volume and so does not weaken as the prediction shrinks.

THE FIX IS THE FIELD STANDARD, NOT A TUNED CHOICE. `BCE + soft Dice` is the
default compound loss for sparse 3D segmentation. The weighting is 1:1 and
unweighted, because any other split would be a hyperparameter fitted to the data
being scored.

WHAT THIS IS NOT. It is NOT training on the evaluation metric. Soft Dice is an
overlap term; the pre-registered metric is a volume ratio. They are related but
distinct, and no term here optimises `log_volume_ratio_error` directly.

TIMING, RECORDED SO IT CAN BE JUDGED. This was fixed after a diagnostic probe
and BEFORE any official rung. No C-rung, no permutation control and no
conditioning comparison had been run. AMD-007 requires one fixed architecture
across rungs; the loss is part of that and is now frozen here rather than passed
as a tunable argument.
"""

from __future__ import annotations

#: 1:1 and unweighted. Any other ratio would be a hyperparameter fitted on the
#: data used to score the rungs.
BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
#: Laplace term keeping soft Dice finite when both prediction and target are
#: empty — the five empty->empty pairs retained with sub-25 hit this.
SMOOTH = 1.0

CONFIG = {
    "loss": "BCEWithLogits + soft Dice",
    "bce_weight": BCE_WEIGHT,
    "dice_weight": DICE_WEIGHT,
    "smooth": SMOOTH,
    "fixed_by": "AMD-007 — identical across C0-C4 and P1-P3",
    "changed_from": "BCEWithLogits alone",
    "reason": (
        "The C0 probe measured a systematic ~15% volume under-prediction "
        "(median pred/target 0.846) and Dice-vs-target 0.4923 against "
        "copy-forward's 0.5041. Measured mechanism: under BCE the background "
        "gradient mass is 63x the foreground's (the class ratio), so the "
        "cheapest descent direction is to shrink. Adding soft Dice cuts that "
        "imbalance to 22x. NOTE: an earlier explanation — that BCE penalises "
        "under-prediction more leniently — was tested and refuted; BCE scores "
        "equal-magnitude under and over errors identically."),
    "results_seen_when_fixed": (
        "One diagnostic convergence probe on repeat 0 / fold 0. NO official "
        "C-rung, NO permutation control, NO conditioning comparison."),
    "not_training_on_the_metric": (
        "Soft Dice is an overlap term. The pre-registered metric is a volume "
        "ratio. No term optimises log_volume_ratio_error directly."),
}


def make_loss():
    """The frozen training criterion. Takes logits, returns a scalar."""
    import torch
    import torch.nn as nn

    bce = nn.BCEWithLogitsLoss()

    def soft_dice(logits, target):
        p = torch.sigmoid(logits)
        dims = tuple(range(1, p.ndim))
        inter = (p * target).sum(dims)
        denom = p.sum(dims) + target.sum(dims)
        return 1.0 - ((2.0 * inter + SMOOTH) / (denom + SMOOTH)).mean()

    def loss(logits, target):
        return BCE_WEIGHT * bce(logits, target) + DICE_WEIGHT * soft_dice(logits, target)

    return loss
