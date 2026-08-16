"""Convergence probe — settle the step budget before 200 fits depend on it.

`STEPS_PER_FIT = 2000` was a placeholder. The whole 24-hour budget rests on it,
and nothing has established it is enough to converge or more than is needed.
This module runs ONE fold with periodic volume-level validation past 2000 steps
and applies a plateau rule fixed in advance.

WHY THE FOLD IS CHOSEN BY RULE, NOT BY HAND. Picking a "representative" fold by
eye is how a convenient result gets selected. Repeat 0 / fold 0 is used because
it is first in the frozen split — an arbitrary but pre-specified choice that
nobody tuned.

WHAT THIS PROBE MAY AND MAY NOT DECIDE. It may set the step budget. It may NOT
select a learning rate, patch size, threshold or architecture: those are locked
or fixed, and the held-out patients used here are the same ones that will score
rungs later. Choosing anything else on this data would fit a parameter to the
scoring set.

THE FOLD USED HERE IS NOT DISCARDED AFTERWARDS. Its test patients will appear in
the official C0 run. That is acceptable ONLY because the sole quantity taken
from this probe is a step count, which is a compute decision applied identically
to every rung and control — not a modelling choice tuned per fold. This is
recorded so a reader can judge it rather than discover it.
"""

from __future__ import annotations

import json
from pathlib import Path

PROBE_REPEAT, PROBE_FOLD = 0, 0
#: Validate past the placeholder so a plateau can be seen rather than assumed.
MAX_STEPS = 4000
VALIDATE_EVERY = 250
#: Fixed BEFORE the curve is seen.
REL_TOL, PATIENCE = 0.01, 3


def fold_pairs(split: dict, repeat: int = PROBE_REPEAT, fold: int = PROBE_FOLD):
    """(train_pairs, test_pairs) for one fold of the FROZEN split."""
    f = split["folds"]["repeats"][repeat]["folds"][fold]
    tr, te = set(f["train_patients"]), set(f["test_patients"])
    pairs = split["pairs"]["pairs"]
    return ([p for p in pairs if p["subject"] in tr],
            [p for p in pairs if p["subject"] in te])


def run(project_root, cache, split: dict, model_fn,
        max_steps: int = MAX_STEPS, validate_every: int = VALIDATE_EVERY,
        batch_size: int = 8, device: str = "cuda", amp: bool = True,
        verify_resume_first: bool = True) -> dict:
    from ..utils.persist import save_artefact
    from .patches import CachedPairPatchSampler, CONFIG
    from .inference import evaluate_fold
    from .train import train_fold, verify_resume, find_plateau

    root = Path(project_root)
    train_pairs, test_pairs = fold_pairs(split)
    tr_sampler = CachedPairPatchSampler(cache, train_pairs)

    # A patient appearing on both sides would invalidate everything downstream.
    tr_subj = {p["subject"] for p in train_pairs}
    te_subj = {p["subject"] for p in test_pairs}
    overlap = tr_subj & te_subj
    if overlap:
        raise RuntimeError(f"LEAKAGE: patients on both sides of the fold: {overlap}")

    out = {
        "probe": "convergence_and_pipeline",
        "fold": {"repeat": PROBE_REPEAT, "fold": PROBE_FOLD,
                 "n_train_patients": len(tr_subj), "n_test_patients": len(te_subj),
                 "n_train_pairs": len(train_pairs), "n_test_pairs": len(test_pairs),
                 "patient_overlap": sorted(overlap)},
        "split_content_sha256": split["content_sha256"],
        "patch_config": CONFIG,
        "plateau_rule": {"rel_tol": REL_TOL, "patience": PATIENCE,
                         "fixed_before_curve_seen": True},
        "scope_limits": (
            "This probe sets the STEP BUDGET only. It does not select learning "
            "rate, patch size, threshold or architecture — those are locked or "
            "fixed, and the held-out patients here also score rungs later."),
    }

    if verify_resume_first:
        print("[probe] verifying checkpoint/resume is bit-exact ...")
        out["resume"] = verify_resume(model_fn, tr_sampler, steps=40,
                                      batch_size=2, device=device, amp=amp)
        print("  " + out["resume"]["verdict"])
        if not out["resume"]["bit_exact"]:
            out["aborted"] = "resume is not bit-exact; probe not run"
            out["artefact"] = save_artefact(root, "10_EXPERIMENTS",
                                            "convergence_probe", out)
            return out

    def validate(model):
        return evaluate_fold(model, cache, test_pairs, batch_size=batch_size,
                             device=device, amp=amp)

    ck = root / "11_CHECKPOINTS" / "probe_r0f0.pt"
    print(f"[probe] training to {max_steps} steps, validating every "
          f"{validate_every} ...")
    tr = train_fold(model_fn(), tr_sampler, steps=max_steps,
                    batch_size=batch_size, device=device, amp=amp,
                    checkpoint_path=ck, validate_every=validate_every,
                    validate_fn=validate, resume=True)
    out["training"] = {k: v for k, v in tr.items() if k != "validation_history"}
    out["validation_history"] = [
        {"step": h["step"], "train_loss": h["train_loss"],
         "model_log_ratio": h["model"]["log_ratio"]["mean"],
         "persistence_log_ratio": h["persistence"]["log_ratio"]["mean"],
         "model_dice": h["model"]["dice"]["mean"],
         "persistence_dice": h["persistence"]["dice"]["mean"],
         "beats_persistence": h["beats_persistence_on_primary"]}
        for h in tr["validation_history"]]
    out["plateau"] = find_plateau(tr["validation_history"], "log_ratio",
                                  rel_tol=REL_TOL, patience=PATIENCE)

    p = out["plateau"]
    # A budget is recommended ONLY on a converged curve. The first version of
    # this probe reported "Plateau at 500 steps" from a curve whose own best was
    # at 4000 — a contradiction produced by a rule that could not tell
    # oscillation from convergence.
    out["recommended_steps_per_fit"] = p["plateau_step"] if p.get("converged") else None
    out["budget_note"] = (
        f"Plateau at {p['plateau_step']} steps. Freeze the step budget there "
        "and recompute the 24 h budget from it before the official fits."
        if p.get("converged") else
        "NO BUDGET RECOMMENDED — the validation curve has not converged. "
        f"{p.get('note', '')} Freezing a step count from this curve would pick "
        "a number the data does not support.")
    out["artefact"] = save_artefact(root, "10_EXPERIMENTS",
                                    "convergence_probe", out)
    print_report(out)
    return out


def print_report(o: dict) -> None:
    line = "-" * 78
    print(line)
    print("CONVERGENCE PROBE  —  repeat 0 / fold 0 of the frozen split")
    print(line)
    f = o["fold"]
    print(f"  train: {f['n_train_patients']} patients / {f['n_train_pairs']} pairs")
    print(f"  test : {f['n_test_patients']} patients / {f['n_test_pairs']} pairs")
    print(f"  patient overlap: {f['patient_overlap'] or 'none'}")
    if "resume" in o:
        print(f"\n  resume: {o['resume']['verdict']}")
    print(f"\n  {'step':>6} {'train loss':>11} {'val log-ratio':>14} "
          f"{'persistence':>12} {'val dice':>9}  beats?")
    for h in o.get("validation_history", []):
        mb = "yes" if h["beats_persistence"] else "no"
        print(f"  {h['step']:>6} {h['train_loss']:>11.4f} "
              f"{h['model_log_ratio']:>14.4f} {h['persistence_log_ratio']:>12.4f} "
              f"{h['model_dice']:>9.4f}  {mb}")
    p = o.get("plateau", {})
    print(f"\n  plateau rule : {p.get('rule')}")
    print(f"  plateau step : {p.get('plateau_step')}   "
          f"best {p.get('best_value')} at step {p.get('best_step')}")
    print(f"  trailing spread: {p.get('trailing_spread', float('nan')):.0%}   "
          f"trend: {p.get('trailing_trend', float('nan')):+.1%}")
    if not p.get("converged"):
        print(f"  ! {p.get('note')}")
    if o.get("budget_note"):
        print(f"\n  {o['budget_note']}")
    print(f"\n  {o['scope_limits']}")
    print(line)
