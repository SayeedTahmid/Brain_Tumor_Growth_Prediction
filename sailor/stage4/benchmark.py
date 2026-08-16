"""Measure throughput on the ACTUAL runtime, then decide if the ladder fits.

The rule this module enforces: no official C0 run starts from an estimate.
Twice in this project a number was predicted before measurement and was wrong
(persistence Dice, and the direction of the GATE-3 verdict). Training cost is
not going to be the third.

WHAT THE LADDER ACTUALLY COSTS
    C0-C4                      5 rungs
    P1, P2, P3                 3 permutation controls
    5 folds x 5 repeats       25 fold-fits each
    TOTAL                    200 fold-fits

With a hard 24 h per job, the binding design target is per-FIT time:

    <= 45 min/fit   a complete rung (25 fits) fits inside one 24 h job
    <= 30 min/fit   a rung takes ~12.5 h, leaving headroom for a restart
    >  60 min/fit   a rung no longer fits in one job and must be split,
                    which is allowed but multiplies checkpoint/resume risk

The AGGREGATE across all 200 fits is unavoidable at 5x5: roughly 100 GPU-hours
at 30 min/fit. That is not a single job and does not breach the 24 h limit, but
it is the real calendar cost and is reported explicitly rather than buried.

WHAT MAY AND MAY NOT BE OPTIMISED
Safe — costs compute, not validity: AMP, channels-last, torch.compile,
precomputed arrays, memmap, dataloader workers, checkpoint/resume, patch-based
3D training.
NOT safe without approval: fewer CV repeats (widens CIs), patch size or
foreground ratio (shifts predictions), early stopping on anything touching a
test fold (leakage), and any feature cache built across folds rather than
within one (leakage — G5).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

RUNGS = ("C0", "C1", "C2", "C3", "C4")
PERMUTATIONS = ("P1", "P2", "P3")
N_FOLDS, N_REPEATS = 5, 5
FITS_PER_RUNG = N_FOLDS * N_REPEATS
TOTAL_FITS = (len(RUNGS) + len(PERMUTATIONS)) * FITS_PER_RUNG

HARD_JOB_LIMIT_HOURS = 24.0
#: Fraction of the job limit a rung may occupy before it is called tight. A run
#: at 100% of the limit has no room for a restart, and this runtime has dropped
#: three times in one session.
SAFETY_FRACTION = 0.75


def gpu_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch": False}
    info = {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(name=p.name,
                    total_vram_gb=round(p.total_memory / 1e9, 2),
                    capability=f"{p.major}.{p.minor}",
                    bf16_supported=bool(p.major >= 8))
    return info


def benchmark(model, sampler, batch_size: int = 4, steps: int = 60,
              warmup: int = 10, amp: bool = True, channels_last: bool = False,
              lr: float = 1e-4) -> dict:
    """Time real training steps: sampling, transfer, forward, backward, step.

    Warmup steps are discarded. The first CUDA calls include kernel autotuning
    and allocator growth, and counting them would understate throughput by a
    large and variable factor.
    """
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    if channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and dev == "cuda"))
    from .loss import make_loss
    lossf = make_loss()

    sample_s, step_s = [], []
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for i in range(warmup + steps):
        t0 = time.perf_counter()
        x, y = sampler.batch(batch_size, epoch=i)
        # Transfer as uint8, cast on device — 4x less PCIe traffic and the
        # cast costs nothing on the GPU.
        xb = torch.from_numpy(x).to(dev, non_blocking=True).float()
        yb = torch.from_numpy(y).to(dev, non_blocking=True).float()
        if channels_last:
            xb = xb.to(memory_format=torch.channels_last_3d)
        if dev == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(amp and dev == "cuda")):
            loss = lossf(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if dev == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        if i >= warmup:
            sample_s.append(t1 - t0)
            step_s.append(t2 - t1)

    n = len(step_s)
    mean_sample = sum(sample_s) / n
    mean_step = sum(step_s) / n
    total = mean_sample + mean_step
    return {
        "device": dev,
        "gpu": gpu_info(),
        "batch_size": batch_size,
        "amp": amp,
        "channels_last": channels_last,
        "steps_timed": n,
        "sec_per_step_total": total,
        "sec_data": mean_sample,
        "sec_compute": mean_step,
        "data_fraction": mean_sample / total if total else None,
        "steps_per_sec": 1.0 / total if total else None,
        "samples_per_sec": batch_size / total if total else None,
        "peak_vram_gb": (round(__import__("torch").cuda.max_memory_allocated() / 1e9, 2)
                         if dev == "cuda" else None),
        "io_bound_warning": (
            "Data loading is >30% of step time — precompute inputs to a local "
            "memmap rather than reading Drive per step, and raise dataloader "
            "workers. This is pure I/O and changes no result."
            if total and (mean_sample / total) > 0.30 else None),
    }


def budget(bench: dict, steps_per_fit: int,
           rungs: int = len(RUNGS) + len(PERMUTATIONS)) -> dict:
    """Does the planned ladder fit? Reports per-fit, per-rung and aggregate."""
    sec_fit = steps_per_fit * bench["sec_per_step_total"]
    h_fit = sec_fit / 3600.0
    h_rung = h_fit * FITS_PER_RUNG
    h_total = h_rung * rungs
    limit = HARD_JOB_LIMIT_HOURS
    return {
        "steps_per_fit": steps_per_fit,
        "hours_per_fit": h_fit,
        "fits_per_rung": FITS_PER_RUNG,
        "hours_per_rung": h_rung,
        "n_rungs_including_permutations": rungs,
        "total_fits": FITS_PER_RUNG * rungs,
        "aggregate_gpu_hours": h_total,
        "hard_job_limit_hours": limit,
        "rung_fits_in_one_job": h_rung <= limit,
        "rung_fits_with_safety_margin": h_rung <= limit * SAFETY_FRACTION,
        "job_utilisation": h_rung / limit,
        "verdict": (
            "FITS" if h_rung <= limit * SAFETY_FRACTION else
            "TIGHT" if h_rung <= limit else
            "DOES_NOT_FIT"),
        "note": (
            f"A complete rung is {h_rung:.1f} h against a {limit:.0f} h job "
            f"limit. Aggregate across all {FITS_PER_RUNG * rungs} fold-fits is "
            f"{h_total:.0f} GPU-hours — that is the whole experiment, spread "
            f"over {rungs} jobs, not a single run."),
        "if_it_does_not_fit": [
            "SAFE: raise batch size to the VRAM ceiling; enable AMP and "
            "channels-last; precompute inputs to a local memmap; more "
            "dataloader workers; torch.compile.",
            "SAFE: split one rung across several jobs with checkpoint/resume — "
            "identical results, more restart risk.",
            "NEEDS APPROVAL: fewer steps per fit (may underfit — verify on a "
            "validation curve, not by assumption).",
            "NEEDS APPROVAL — CHANGES A LOCK: fewer CV repeats. Widens CIs and "
            "weakens every comparison; the 5x5 scheme and seed 1337 are frozen.",
            "REFUSED: reducing patch size or foreground ratio to save time. "
            "Both shift predictions and would make rungs non-comparable "
            "(AMD-007).",
        ],
    }


def run(project_root, model, sampler, steps_per_fit: int,
        batch_size: int = 4, steps: int = 60, amp: bool = True,
        channels_last: bool = False) -> dict:
    from ..utils.persist import save_artefact
    b = benchmark(model, sampler, batch_size=batch_size, steps=steps,
                  amp=amp, channels_last=channels_last)
    res = {"check": "training_throughput_benchmark",
           "benchmark": b,
           "budget": budget(b, steps_per_fit),
           "patch_config": __import__(
               "sailor.stage4.patches", fromlist=["CONFIG"]).CONFIG}
    res["artefact"] = save_artefact(Path(project_root), "10_EXPERIMENTS",
                                    "training_benchmark", res)
    print_report(res)
    return res


def print_report(res: dict) -> None:
    b, g = res["benchmark"], res["benchmark"]["gpu"]
    d = res["budget"]
    line = "-" * 78
    print(line)
    print("TRAINING THROUGHPUT BENCHMARK  —  measured, not estimated")
    print(line)
    print(f"  device      : {g.get('name', b['device'])}  "
          f"{g.get('total_vram_gb', '?')} GB   bf16={g.get('bf16_supported')}")
    print(f"  batch       : {b['batch_size']}   amp={b['amp']}   "
          f"channels_last={b['channels_last']}")
    print(f"  sec/step    : {b['sec_per_step_total']:.4f}  "
          f"(data {b['sec_data']:.4f} + compute {b['sec_compute']:.4f})")
    print(f"  steps/sec   : {b['steps_per_sec']:.2f}   "
          f"peak VRAM: {b['peak_vram_gb']} GB")
    if b["io_bound_warning"]:
        print(f"  ! {b['io_bound_warning']}")
    print(f"\n  steps/fit   : {d['steps_per_fit']}")
    print(f"  hours/fit   : {d['hours_per_fit']:.2f}")
    print(f"  hours/rung  : {d['hours_per_rung']:.1f}  "
          f"({d['fits_per_rung']} fits)")
    print(f"  aggregate   : {d['aggregate_gpu_hours']:.0f} GPU-h over "
          f"{d['total_fits']} fold-fits")
    print(f"\n  VERDICT     : {d['verdict']}   "
          f"(job utilisation {d['job_utilisation']:.0%} of "
          f"{d['hard_job_limit_hours']:.0f} h)")
    print(f"  {d['note']}")
    if d["verdict"] != "FITS":
        print("\n  Options, in order:")
        for o in d["if_it_does_not_fit"]:
            print(f"    - {o}")
    print(line)
