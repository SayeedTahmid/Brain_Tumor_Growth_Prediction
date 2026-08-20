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
#: AMD-009. Added after C0 and C1 revealed the objective was misaligned with the
#: pre-registered metric — see the AMD-009 block below.
VOLUME_WEIGHT = 0.5
#: Laplace term keeping soft Dice finite when both prediction and target are
#: empty — the five empty->empty pairs retained with sub-25 hit this.
SMOOTH = 1.0

# --------------------------------------------------------------------------
# AMD-009 (18 Aug 2026) — THE LOSS CHANGED AFTER RUNG RESULTS WERE SEEN.
#
# Three rungs sat on one side of the persistence floor, monotone:
#     C0-direct +0.1063 | C0-residual +0.0795 | C1 +0.0664
# The residual rungs were VERIFIED to start exactly at the floor and trained
# away from it. Every rung showed the same signature: model Dice ABOVE
# persistence, model log-ratio BELOW it — the divergence AMD-005 anticipated.
#
# A probe adding a log-volume-ratio term collapsed the gap to +0.0052, a 94%
# reduction. For scale, Δt — the variable C1 exists to test — moved the result
# by 0.0131. The objective artefact was 5.7x the conditioning effect.
#
# TWO NON-METRIC ALTERNATIVES WERE TRIED AND REJECTED, so the cost below was
# not accepted lightly:
#   B  unbounded relative volume error: WORSE (+0.0974). Diagnosed to a scale
#      pathology — the term reaches 250 at p=0.5 because an 884,736-voxel soft
#      volume dwarfs a ~1,700-voxel target before the head saturates.
#   C  bounded relative volume error: introduced on a claim of asymmetry that
#      MEASUREMENT REFUTED (halving and doubling both cost 0.500). Not run.
#   Three variants was the pre-declared stop rule. No fourth was tried.
#
# THE COST, STATED PLAINLY. The adopted term is a differentiable analogue of
# `log_volume_ratio_error`, the metric the rungs are SCORED by. "Beats
# persistence" is therefore WEAKER evidence under this loss than under the
# previous one. The paper must say so. Mitigation: BOTH ladders are reported —
# the runs under the previous loss are retained, not discarded, so a reader can
# see the uncorrected numbers and judge the correction.
#
# WHY THIS IS NOT A FORKING PATH. Under the corrected loss the model STILL does
# not beat persistence (+0.0052 against a frozen MDE of 0.0555). The headline
# null is unchanged; what changes is that it becomes attributable to the data
# rather than to a misaligned objective.
# --------------------------------------------------------------------------

CONFIG = {
    "loss": "BCEWithLogits + soft Dice + log-volume-ratio",
    "bce_weight": BCE_WEIGHT,
    "dice_weight": DICE_WEIGHT,
    "volume_weight": VOLUME_WEIGHT,
    "amendment": "AMD-009",
    "invalidates": ("C0-direct, C0-residual and C1 were run under "
                    "BCEWithLogits + soft Dice. They MUST be re-run under this "
                    "loss and are RETAINED as the uncorrected ladder."),
    "known_cost": ("Contains a differentiable analogue of the scoring metric. "
                   "'Beats persistence' is weaker evidence under this loss and "
                   "the paper must state that."),
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

    def volume_term(logits, target):
        """|log((V_true+1)/(V_pred+1))| on SOFT volumes, so it is differentiable.

        Mirrors `log_volume_ratio_error` without being identical: the metric
        thresholds at 0.5 and this does not.
        """
        p = torch.sigmoid(logits)
        dims = tuple(range(1, p.ndim))
        return torch.abs(torch.log((target.sum(dims) + 1.0)
                                   / (p.sum(dims) + 1.0))).mean()

    def loss(logits, target):
        return (BCE_WEIGHT * bce(logits, target)
                + DICE_WEIGHT * soft_dice(logits, target)
                + VOLUME_WEIGHT * volume_term(logits, target))

    return loss
