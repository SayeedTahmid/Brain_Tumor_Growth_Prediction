"""Fold training with checkpoint/resume and volume-level validation.

RESUME MUST BE EXACT, NOT APPROXIMATE. This runtime has dropped four times in
one session. A resume that restores weights but not optimizer moments, AMP
scaler state, or RNG position produces a run that is *similar* to the
uninterrupted one — which is worse than an obvious failure, because the numbers
look fine and are not reproducible. `verify_resume()` proves bit-exactness
rather than assuming it: it trains N steps straight through, trains N/2 +
resume + N/2, and asserts every parameter is identical.

VALIDATION IS VOLUME-LEVEL. Patch loss is not the task. It can plateau while
whole-volume performance is still moving and it is blind to systematic over- or
under-prediction, which is exactly what the pre-registered
`log_volume_ratio_error` measures.

WHAT IS NOT TUNED HERE. Learning rate, patch size, foreground ratio, batch size
and the CV scheme are fixed inputs. Nothing in this module selects a
hyperparameter by looking at validation, because the same held-out patients are
used to score the rung.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

LR = 1e-4
WEIGHT_DECAY = 1e-5
CHECKPOINT_EVERY = 250


def _rng_state(torch):
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _restore_rng(torch, st):
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def save_checkpoint(path, model, opt, scaler, step: int, meta: dict) -> None:
    import torch
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    # Atomic replace: a runtime that dies mid-write must not leave a truncated
    # checkpoint where a valid one used to be.
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "step": step,
                "rng": _rng_state(torch), "meta": meta}, tmp)
    tmp.replace(p)


def load_checkpoint(path, model, opt, scaler):
    import torch
    p = Path(path)
    if not p.is_file():
        return 0, None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    scaler.load_state_dict(ck["scaler"])
    _restore_rng(torch, ck["rng"])
    return int(ck["step"]), ck.get("meta")


def train_fold(model, sampler, steps: int, batch_size: int = 8,
               device: str = "cuda", amp: bool = True,
               checkpoint_path=None, checkpoint_every: int = CHECKPOINT_EVERY,
               validate_every: int = 0, validate_fn=None,
               lr: float = LR, resume: bool = True, log_every: int = 100) -> dict:
    """Train one fold. Resumes from `checkpoint_path` when present."""
    import torch
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device == "cuda"))
    lossf = torch.nn.BCEWithLogitsLoss()

    start = 0
    if resume and checkpoint_path:
        start, _ = load_checkpoint(checkpoint_path, model, opt, scaler)
        if start:
            print(f"[resume] continuing from step {start}")

    history, losses = [], []
    t0 = time.perf_counter()
    for step in range(start, steps):
        model.train()
        x, y = sampler.batch(batch_size, epoch=step)
        xb = torch.from_numpy(x).to(device, non_blocking=True).float()
        yb = torch.from_numpy(y).to(device, non_blocking=True).float()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(amp and device == "cuda")):
            loss = lossf(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.detach().cpu()))

        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step+1:>5}  loss {np.mean(losses[-log_every:]):.4f}  "
                  f"{(time.perf_counter()-t0)/(step+1-start):.3f} s/step")
        if checkpoint_path and (step + 1) % checkpoint_every == 0:
            save_checkpoint(checkpoint_path, model, opt, scaler, step + 1,
                            {"steps_planned": steps})
        if validate_every and validate_fn and (step + 1) % validate_every == 0:
            v = validate_fn(model)
            v["step"] = step + 1
            v["train_loss"] = float(np.mean(losses[-validate_every:]))
            history.append(v)
            print(f"  [val] step {step+1:>5}  "
                  f"log_ratio {v['model']['log_ratio']['mean']:.4f}  "
                  f"(persistence {v['persistence']['log_ratio']['mean']:.4f})  "
                  f"dice {v['model']['dice']['mean']:.4f}")

    if checkpoint_path:
        save_checkpoint(checkpoint_path, model, opt, scaler, steps,
                        {"steps_planned": steps, "complete": True})
    return {"steps": steps, "final_train_loss": float(np.mean(losses[-100:])) if losses else None,
            "validation_history": history,
            "wall_seconds": time.perf_counter() - t0}


def verify_resume(model_fn, sampler, steps: int = 40, batch_size: int = 2,
                  device: str = "cuda", amp: bool = True, tmpdir=None) -> dict:
    """Prove that resume is BIT-EXACT, not merely similar.

    Straight-through vs interrupted-and-resumed must produce identical
    parameters. Anything less means a disconnected run silently diverges from
    the run it claims to continue.
    """
    import tempfile
    import torch
    d = Path(tmpdir or tempfile.mkdtemp())

    torch.manual_seed(0); np.random.seed(0)
    a = model_fn()
    train_fold(a, sampler, steps=steps, batch_size=batch_size, device=device,
               amp=amp, checkpoint_path=None, resume=False, log_every=0)

    torch.manual_seed(0); np.random.seed(0)
    b = model_fn()
    ck = d / "resume_test.pt"
    train_fold(b, sampler, steps=steps // 2, batch_size=batch_size, device=device,
               amp=amp, checkpoint_path=ck, checkpoint_every=steps // 2,
               resume=False, log_every=0)
    b2 = model_fn()
    train_fold(b2, sampler, steps=steps, batch_size=batch_size, device=device,
               amp=amp, checkpoint_path=ck, checkpoint_every=steps,
               resume=True, log_every=0)

    diffs = []
    sa, sb = a.state_dict(), b2.state_dict()
    for k in sa:
        d_max = float((sa[k].float() - sb[k].float()).abs().max())
        if d_max > 0:
            diffs.append({"param": k, "max_abs_diff": d_max})
    return {
        "steps": steps,
        "bit_exact": not diffs,
        "n_params_differing": len(diffs),
        "worst": sorted(diffs, key=lambda r: -r["max_abs_diff"])[:5],
        "verdict": ("RESUME IS BIT-EXACT — an interrupted run continues the run "
                    "it claims to continue"
                    if not diffs else
                    "RESUME DIVERGES — a disconnected run would produce numbers "
                    "that look fine and are not reproducible. DO NOT run the "
                    "ladder until this is fixed."),
    }


def find_plateau(history: list, key: str = "log_ratio",
                 rel_tol: float = 0.01, patience: int = 3) -> dict:
    """Smallest step after which the metric stops improving materially.

    A metric is 'still improving' while it betters the best-so-far by more than
    `rel_tol` relative. `patience` consecutive non-improvements define the
    plateau. Stated as a rule BEFORE the curve is seen, so the step budget is
    not chosen by eye from a graph.
    """
    if not history:
        return {"plateau_step": None, "note": "no validation history"}
    vals = [(h["step"], h["model"][key]["mean"]) for h in history
            if h["model"][key]["mean"] is not None]
    if not vals:
        return {"plateau_step": None, "note": f"metric {key} never defined"}
    best, best_step, stale, plateau = float("inf"), None, 0, None
    for step, v in vals:
        if v < best * (1 - rel_tol):
            best, best_step, stale = v, step, 0
        else:
            stale += 1
            if stale >= patience and plateau is None:
                plateau = best_step
    return {
        "plateau_step": plateau,
        "best_value": best, "best_step": best_step,
        "last_step": vals[-1][0],
        "rule": (f"improvement of <{rel_tol:.0%} relative for {patience} "
                 "consecutive validations, fixed before the curve was seen"),
        "still_improving_at_end": plateau is None,
        "note": (None if plateau is not None else
                 "The metric had not plateaued by the last validation — the "
                 "probe must be extended before a step budget is frozen."),
        "curve": vals,
    }
