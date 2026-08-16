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

import numpy as np

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


# --------------------------------------------------------------- multi-fold
def run_multifold(project_root, cache, split: dict, model_fn,
                  repeat: int = 0, max_steps: int = MAX_STEPS,
                  validate_every: int = VALIDATE_EVERY, batch_size: int = 8,
                  device: str = "cuda", amp: bool = True) -> dict:
    """Convergence measured on ALL folds of one repeat — 26 patients, not 5.

    WHY. The single-fold probe validated on 5 held-out patients and measured a
    between-patient SD of 0.276 on the pre-registered metric, with per-patient
    means spanning 0.20 to 0.76 — nearly a factor of four. The step-to-step
    swings during that probe (0.45 to 0.87) are the SAME order as the
    between-patient spread, so the oscillation was largely re-drawn sampling
    noise rather than instability in training.

    A validation estimate noisier than the effect it measures cannot support a
    step-budget decision. Training every fold of a repeat and pooling the
    held-out predictions gives one validation point over all 26 patients at each
    step, which is the cohort-level quantity the ladder will report anyway.

    Cost is 5x the single-fold probe and is the honest price of a usable curve.
    """
    from ..utils.persist import save_artefact
    from .patches import CachedPairPatchSampler, CONFIG
    from .loss import CONFIG as LOSS_CONFIG
    from .inference import evaluate_fold
    from .train import train_fold, find_plateau

    root = Path(project_root)
    n_folds = len(split["folds"]["repeats"][repeat]["folds"])
    per_fold, curves = [], {}

    for fold in range(n_folds):
        tr_pairs, te_pairs = fold_pairs(split, repeat, fold)
        tr_subj = {p["subject"] for p in tr_pairs}
        te_subj = {p["subject"] for p in te_pairs}
        if tr_subj & te_subj:
            raise RuntimeError(f"LEAKAGE fold {fold}: {tr_subj & te_subj}")

        sampler = CachedPairPatchSampler(cache, tr_pairs)
        hist = []

        def validate(model, _te=te_pairs):
            return evaluate_fold(model, cache, _te, batch_size=batch_size,
                                 device=device, amp=amp)

        print(f"\n[fold {fold}] train {len(tr_subj)} patients / {len(tr_pairs)} "
              f"pairs | test {len(te_subj)} / {len(te_pairs)}")
        ck = root / "11_CHECKPOINTS" / f"probe_r{repeat}f{fold}.pt"
        tr = train_fold(model_fn(), sampler, steps=max_steps,
                        batch_size=batch_size, device=device, amp=amp,
                        checkpoint_path=ck, validate_every=validate_every,
                        validate_fn=validate, resume=True, log_every=500)
        for h in tr["validation_history"]:
            hist.append({"step": h["step"], "pairs": h["per_pair"]})
        curves[fold] = [(h["step"], h["model"]["log_ratio"]["mean"])
                        for h in tr["validation_history"]]
        per_fold.append({"fold": fold, "history": hist,
                         "n_test_patients": len(te_subj)})

    # Pool across folds at each step: every patient appears in exactly one test
    # fold, so a pooled point is a 26-patient estimate.
    steps = sorted({h["step"] for f in per_fold for h in f["history"]})
    pooled = []
    for step in steps:
        by_patient, pers_by_patient = {}, {}
        for f in per_fold:
            for h in f["history"]:
                if h["step"] != step:
                    continue
                for r in h["pairs"]:
                    if r["model_log_ratio"] is not None:
                        by_patient.setdefault(r["subject"], []).append(
                            r["model_log_ratio"])
                    if r["pers_log_ratio"] is not None:
                        pers_by_patient.setdefault(r["subject"], []).append(
                            r["pers_log_ratio"])
        if not by_patient:
            continue
        per = [float(np.mean(v)) for v in by_patient.values()]
        pers = [float(np.mean(v)) for v in pers_by_patient.values()]
        pooled.append({
            "step": step,
            "n_patients": len(per),
            "model": {"log_ratio": {"mean": float(np.mean(per))}},
            "model_log_ratio": float(np.mean(per)),
            "persistence_log_ratio": float(np.mean(pers)) if pers else None,
            "between_patient_sd": float(np.std(per, ddof=1)) if len(per) > 1 else None,
            "beats_persistence": (bool(np.mean(per) < np.mean(pers))
                                  if pers else None),
        })

    plateau = find_plateau(pooled, "log_ratio", rel_tol=REL_TOL, patience=PATIENCE)
    out = {
        "probe": "convergence_multifold",
        "repeat": repeat, "n_folds": n_folds,
        "split_content_sha256": split["content_sha256"],
        "patch_config": CONFIG, "loss_config": LOSS_CONFIG,
        "pooled_curve": pooled,
        "per_fold_curves": {str(k): v for k, v in curves.items()},
        "plateau": plateau,
        "recommended_steps_per_fit": (plateau["plateau_step"]
                                      if plateau.get("converged") else None),
        "why_multifold": (
            "The single-fold probe validated on 5 patients with a between-"
            "patient SD of 0.276 — the same order as its step-to-step swings. "
            "Pooling all folds of a repeat gives a 26-patient estimate per "
            "validation point."),
    }
    out["artefact"] = save_artefact(root, "10_EXPERIMENTS",
                                    "convergence_multifold", out)
    print_multifold(out)
    return out


def print_multifold(o: dict) -> None:
    line = "-" * 78
    print(line)
    print(f"MULTI-FOLD CONVERGENCE  —  repeat {o['repeat']}, "
          f"{o['n_folds']} folds pooled")
    print(line)
    print(f"  loss: {o['loss_config']['loss']}  "
          f"(was {o['loss_config']['changed_from']})")
    print(f"\n  {'step':>6} {'patients':>9} {'model':>9} {'persistence':>12} "
          f"{'between-pt SD':>14}  beats?")
    for p in o["pooled_curve"]:
        sd = f"{p['between_patient_sd']:.4f}" if p["between_patient_sd"] else "—"
        print(f"  {p['step']:>6} {p['n_patients']:>9} {p['model_log_ratio']:>9.4f} "
              f"{p['persistence_log_ratio']:>12.4f} {sd:>14}  "
              f"{'yes' if p['beats_persistence'] else 'no'}")
    pl = o["plateau"]
    print(f"\n  plateau: {pl.get('plateau_step')}  converged={pl.get('converged')}")
    if pl.get("note"):
        print(f"  ! {pl['note']}")
    print(f"\n  {o['why_multifold']}")
    print(line)
