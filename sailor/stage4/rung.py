"""Run one official rung of the conditioning ladder: 25 fits, 5x5 CV.

WHAT MAKES THIS "OFFICIAL" RATHER THAN A PROBE. A probe answers a question about
the pipeline; a rung produces a number that enters the paper. The difference is
enforced here:

  - all 25 fits (5 folds x 5 repeats) of the FROZEN split, not one repeat
  - the step budget is an explicit argument, recorded, never inferred at runtime
  - every held-out prediction is scored with the PRE-REGISTERED metric
  - uncertainty resamples PATIENTS (26 units), never pairs (208) — AMD-003
  - persistence is scored on the identical pairs, so the comparison is paired
  - the config fingerprint is recorded, so a rung cannot silently mix objectives

RESUMABLE ACROSS JOBS. 25 fits at ~11 min is ~4.6 h, inside a 24 h limit, but
this runtime has disconnected repeatedly. Each fit checkpoints independently and
completed fits are skipped on restart, so a dropped session costs one fit rather
than the rung.

WHAT THIS MODULE DOES NOT DO. It does not choose the step budget, compare rungs,
or decide anything. It runs a specified configuration and records what happened.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 1337
CI = (2.5, 97.5)


def _patient_bootstrap(by_patient: dict, seed: int = BOOTSTRAP_SEED,
                       n: int = N_BOOTSTRAP) -> dict:
    """Resample PATIENTS with replacement (AMD-003). Never pairs."""
    patients = sorted(by_patient)
    if not patients:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_patients": 0}
    rng = np.random.default_rng(seed)
    per = np.array([float(np.mean(by_patient[p])) for p in patients])
    boot = per[rng.integers(0, len(patients), size=(n, len(patients)))].mean(axis=1)
    lo, hi = np.percentile(boot, CI)
    return {"mean": float(per.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n_patients": len(patients),
            "per_patient_sd": float(per.std(ddof=1)) if len(patients) > 1 else None}


def run_rung(project_root, cache, split: dict, model_fn, rung: str,
             steps_per_fit: int, batch_size: int = 8, device: str = "cuda",
             amp: bool = True, cond_fn=None, repeats=None, folds=None) -> dict:
    """Train and evaluate every fold of every repeat for one rung."""
    from ..utils.persist import save_artefact
    from .patches import CachedPairPatchSampler, CONFIG as PATCH_CONFIG
    from .loss import CONFIG as LOSS_CONFIG
    from .inference import evaluate_fold
    from .train import train_fold, _config_fingerprint
    from .convergence import fold_pairs

    root = Path(project_root)
    all_repeats = range(len(split["folds"]["repeats"])) if repeats is None else repeats
    fingerprint = _config_fingerprint()
    t0 = time.perf_counter()
    fits, rows = [], []

    for rep in all_repeats:
        n_folds = len(split["folds"]["repeats"][rep]["folds"])
        for fold in (range(n_folds) if folds is None else folds):
            tr_pairs, te_pairs = fold_pairs(split, rep, fold)
            tr_subj = {p["subject"] for p in tr_pairs}
            te_subj = {p["subject"] for p in te_pairs}
            if tr_subj & te_subj:
                raise RuntimeError(
                    f"LEAKAGE r{rep}f{fold}: {sorted(tr_subj & te_subj)}")

            ck = root / "11_CHECKPOINTS" / f"{rung}_r{rep}f{fold}.pt"
            res_path = root / "11_CHECKPOINTS" / f"{rung}_r{rep}f{fold}_eval.json"
            if res_path.is_file():
                # Completed fits are skipped so a dropped session costs one fit.
                ev = json.loads(res_path.read_text())
                print(f"[{rung} r{rep}f{fold}] already complete, skipping")
            else:
                sampler = CachedPairPatchSampler(cache, tr_pairs)
                print(f"[{rung} r{rep}f{fold}] train {len(tr_subj)}p/"
                      f"{len(tr_pairs)}pr  test {len(te_subj)}p/{len(te_pairs)}pr")
                model = model_fn()
                train_fold(model, sampler, steps=steps_per_fit,
                           batch_size=batch_size, device=device, amp=amp,
                           checkpoint_path=ck, resume=True, log_every=1000)
                ev = evaluate_fold(model, cache, te_pairs, batch_size=batch_size,
                                   device=device, amp=amp)
                tmp = res_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(ev, default=str))
                tmp.replace(res_path)
            fits.append({"repeat": rep, "fold": fold,
                         "n_test_patients": len(te_subj),
                         "n_test_pairs": len(te_pairs)})
            for r in ev["per_pair"]:
                rows.append(dict(r, repeat=rep, fold=fold))

    # Aggregate. Every patient is held out once per repeat, so pooling within a
    # repeat gives a full-cohort estimate and averaging across repeats reduces
    # fold-assignment variance without inflating n.
    def agg(key):
        by = {}
        for r in rows:
            if r.get(key) is not None:
                by.setdefault(r["subject"], []).append(float(r[key]))
        return _patient_bootstrap(by)

    out = {
        "rung": rung,
        "steps_per_fit": steps_per_fit,
        "batch_size": batch_size,
        "n_fits": len(fits),
        "fits": fits,
        "split_content_sha256": split["content_sha256"],
        "config_fingerprint": fingerprint,
        "patch_config": PATCH_CONFIG,
        "loss_config": LOSS_CONFIG,
        "primary_metric": "log_volume_ratio_error",
        "model": {
            "log_ratio": agg("model_log_ratio"),
            "dice": agg("model_dice"),
            "vol_err": agg("model_vol_err"),
            "change_dice": agg("model_change_dice"),
        },
        "persistence_same_pairs": {
            "log_ratio": agg("pers_log_ratio"),
            "dice": agg("pers_dice"),
            "vol_err": agg("pers_vol_err"),
        },
        "wall_hours": (time.perf_counter() - t0) / 3600.0,
        "uncertainty_note": (
            "Patient-level bootstrap over 26 units, never the 208 pairs "
            "(AMD-003). Persistence is scored on the IDENTICAL held-out pairs, "
            "so model-vs-persistence is a paired comparison."),
    }
    m = out["model"]["log_ratio"]["mean"]
    p = out["persistence_same_pairs"]["log_ratio"]["mean"]
    out["beats_persistence"] = (None if m is None or p is None else bool(m < p))
    out["gap_vs_persistence"] = (None if m is None or p is None else m - p)

    # Paired difference SD — the quantity GATE-3's MDE was an upper bound for.
    dif = {}
    for r in rows:
        if r.get("model_log_ratio") is not None and r.get("pers_log_ratio") is not None:
            dif.setdefault(r["subject"], []).append(
                float(r["model_log_ratio"]) - float(r["pers_log_ratio"]))
    if dif:
        per = np.array([np.mean(v) for v in dif.values()])
        sd = float(per.std(ddof=1)) if len(per) > 1 else None
        out["paired_difference"] = {
            "mean": float(per.mean()), "sd": sd, "n_patients": len(per),
            "mde_from_paired_sd": (None if sd is None else
                                   float((1.96 + 1.2816) * sd / np.sqrt(len(per)))),
            "note": ("GATE-3's MDE of 0.1585 assumed model-minus-baseline spread "
                     "matched the baseline's between-patient spread. This is the "
                     "measured paired SD. Recompute the MDE from it ONCE, record "
                     "it, and do not re-derive per rung."),
        }
    out["artefact"] = save_artefact(root, "10_EXPERIMENTS", f"rung_{rung}", out)
    print_rung(out)
    return out


def print_rung(o: dict) -> None:
    line = "-" * 78
    print(line)
    print(f"RUNG {o['rung']}  —  {o['n_fits']} fits, {o['steps_per_fit']} steps each")
    print(line)
    print(f"  split {o['split_content_sha256'][:16]}…  config {o['config_fingerprint']}")
    print(f"  loss  {o['loss_config']['loss']}")
    print(f"  wall  {o['wall_hours']:.2f} h")
    print(f"\n  {'metric':<14}{'model':>26}{'persistence':>26}")
    for k, label in (("log_ratio", "log-ratio [1°]"), ("dice", "Dice [2°]"),
                     ("vol_err", "volume error")):
        m, p = o["model"][k], o["persistence_same_pairs"][k]
        f = lambda a: ("—" if a["mean"] is None else
                       f"{a['mean']:.4f} [{a['ci_low']:.4f}, {a['ci_high']:.4f}]")
        print(f"  {label:<14}{f(m):>26}{f(p):>26}")
    print(f"\n  beats persistence: {o['beats_persistence']}   "
          f"gap {o['gap_vs_persistence']:+.4f}"
          if o["gap_vs_persistence"] is not None else "")
    pd = o.get("paired_difference")
    if pd:
        print(f"\n  paired difference: mean {pd['mean']:+.4f}  sd {pd['sd']:.4f}  "
              f"n {pd['n_patients']}")
        print(f"  MDE from paired SD: {pd['mde_from_paired_sd']:.4f}  "
              "(GATE-3 assumed 0.1585 as an upper bound)")
    print(f"\n  {o['uncertainty_note']}")
    print(line)
