"""Can the evaluation pipeline reproduce persistence? A control never run.

THE ANOMALY. C0 scored 0.5991 against persistence 0.4928 — worse by +0.1063,
roughly twice the measured MDE of 0.0555. But persistence IS the identity
function on the input mask, and the model is a U-Net WITH SKIP CONNECTIONS that
receives that mask as input. Copying the input to the output is the easiest
function such a network can represent. A model that cannot match it is either
badly under-trained, or something between the model's output and the score is
adding error.

THE UNTESTED PATH. Training happens on 96^3 patches; scoring happens on whole
volumes via sliding-window inference with half-patch stride, logit averaging in
the overlaps, and a fixed 0.5 threshold. That path has never been validated
against a known-correct predictor. If reassembly systematically blurs
boundaries, every rung inherits the same penalty and every rung-vs-persistence
comparison is biased — while rung-vs-rung differences stay valid.

THE CONTROL. Feed an EXACT identity model — one whose forward returns its input
as saturated logits — through the identical evaluation path used for every rung.
Its score should equal persistence exactly, because it computes the same
function.

    identity == persistence      the pipeline is faithful; C0's deficit is real
    identity  > persistence      the pipeline ADDS error. C0's deficit is partly
                                 or wholly an artefact of sliding-window
                                 reassembly and thresholding, and the
                                 rung-vs-persistence comparison must be
                                 corrected before it is reported

This is a validity check on every number the ladder will produce, and it costs
one inference pass. It should have been run before C0.
"""

from __future__ import annotations

import numpy as np


class IdentityModel:
    """Returns its input as saturated logits — exactly the persistence rule.

    Not an `nn.Module` subclass by accident: it must go through the same
    `predict_mask` path as a real model, including autocast and the 0.5
    threshold, so any error introduced there is exposed rather than bypassed.
    """

    SATURATION = 20.0

    def __init__(self):
        self._training = False

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def to(self, *a, **kw):
        return self

    def __call__(self, x, cond=None):
        # logits: +20 where the input is 1, -20 where 0 -> sigmoid ~1 / ~0
        return (x * (2 * self.SATURATION)) - self.SATURATION


def run(cache, pairs: list, patch: int = 96, batch_size: int = 8,
        device: str = "cuda", amp: bool = True) -> dict:
    """Score the identity model through the full rung evaluation path."""
    from .inference import evaluate_fold, predict_mask
    from ..stage3.persistence import log_volume_ratio_error, dice

    ev = evaluate_fold(IdentityModel(), cache, pairs, patch=patch,
                       batch_size=batch_size, device=device, amp=amp)

    # Voxel-level check: does the pipeline return the input unchanged?
    exact, drift = 0, []
    for p in pairs[:20]:
        a = cache.get(p["subject"], p["input_session"])
        if a is None:
            continue
        pred = predict_mask(IdentityModel(), a, patch=patch,
                            batch_size=batch_size, device=device, amp=amp)
        same = bool(np.array_equal(pred, a.astype(bool)))
        exact += int(same)
        if not same:
            drift.append({
                "subject": p["subject"], "session": p["input_session"],
                "n_input": int(a.sum()), "n_returned": int(pred.sum()),
                "voxels_differing": int(np.logical_xor(pred, a.astype(bool)).sum()),
            })

    m = ev["model"]["log_ratio"]["mean"]
    p_ = ev["persistence"]["log_ratio"]["mean"]
    delta = None if (m is None or p_ is None) else m - p_
    # Tolerance is tight on purpose: these compute the same function, so any
    # difference at all is pipeline error rather than sampling noise.
    faithful = delta is not None and abs(delta) < 1e-6

    return {
        "check": "identity_pipeline_control",
        "n_pairs": len(pairs),
        "identity_log_ratio": m,
        "persistence_log_ratio": p_,
        "delta": delta,
        "identity_dice": ev["model"]["dice"]["mean"],
        "persistence_dice": ev["persistence"]["dice"]["mean"],
        "volumes_returned_exactly": exact,
        "volumes_checked": min(20, len(pairs)),
        "drift_examples": drift[:5],
        "pipeline_faithful": faithful,
        "verdict": (
            "PIPELINE FAITHFUL — the identity model scores exactly persistence, "
            "so sliding-window inference and thresholding add no error. C0's "
            "deficit of +0.1063 is a real property of the trained model."
            if faithful else
            "PIPELINE ADDS ERROR — the identity model does NOT score persistence, "
            "although it computes the same function. Every rung inherits this "
            "penalty, so rung-vs-persistence comparisons are biased and must be "
            "corrected before they are reported. Rung-vs-rung differences remain "
            "valid because the bias is common to all of them."),
        "what_it_does_not_establish": (
            "A faithful pipeline does not mean C0 is well trained — only that "
            "its deficit is attributable to the model rather than the scoring "
            "path."),
    }


def print_report(r: dict) -> None:
    line = "-" * 78
    print(line)
    print("IDENTITY PIPELINE CONTROL  —  can the scoring path reproduce persistence?")
    print(line)
    print(f"  pairs                    : {r['n_pairs']}")
    print(f"  identity model log-ratio : {r['identity_log_ratio']:.6f}")
    print(f"  persistence log-ratio    : {r['persistence_log_ratio']:.6f}")
    print(f"  delta                    : {r['delta']:+.6f}")
    print(f"  identity Dice            : {r['identity_dice']:.6f}")
    print(f"  persistence Dice         : {r['persistence_dice']:.6f}")
    print(f"  volumes returned exactly : {r['volumes_returned_exactly']} / "
          f"{r['volumes_checked']}")
    if r["drift_examples"]:
        print("  drift examples:")
        for d in r["drift_examples"]:
            print(f"    {d['subject']}/{d['session']}  in {d['n_input']} -> "
                  f"out {d['n_returned']}  ({d['voxels_differing']} differ)")
    print(f"\n  {r['verdict']}")
    print(f"\n  {r['what_it_does_not_establish']}")
    print(line)
