"""Conditioning vectors for the ladder rungs, and the statistics they need.

WHAT EACH RUNG SUPPLIES (ROS §11.1). Rungs differ ONLY in what information the
conditioning vector carries. The architecture, patch config, loss and step budget
are identical throughout (AMD-007), and conditioning enters through one pathway —
FiLM on the bottleneck — so a rung gap measures information rather than
structure.

    C0   nothing                                    cond_dim 0
    C1   Δt                                          cond_dim 1
    C2   Δt + treatment status                       cond_dim 5
    C3   Δt + dose features                          cond_dim 1 + n_dose
    C4   Δt + treatment status + dose features

AMD-001 APPLIES HERE AND IS EASY TO FORGET. C1 is NOT a treatment-free
reference. Δt separates treatment phases almost completely — U = 0.629, median
interval ratio 6.5x (14 d during CRT, 91 d during TMZ) — so supplying Δt already
supplies most of the treatment signal. C0 is the treatment-free reference, and
C2 − C1 UNDERSTATES the treatment effect rather than measuring it.

THE LEAKAGE RULE, ENFORCED RATHER THAN INTENDED. Standardisation statistics are
computed on TRAINING pairs only and applied unchanged to held-out pairs. Fitting
a transform on all pairs would let the held-out Δt distribution influence the
inputs the model sees at training time — a small leak, but the kind that is
invisible in results and indefensible once noticed. `FoldStandardiser` cannot be
constructed except from a list of pairs, and `rung.py` passes it the training
split, so the leak is prevented structurally rather than by discipline.

WHY log(Δt). Intervals span 7 to 371 days, a 53x range with a heavy right tail
from the TMZ phase. Raw days would give the long intervals dominant magnitude in
a FiLM scale/shift; log compresses that to a ~3.9x range. Standardising after
the log puts the vector in the zero-mean, unit-variance regime FiLM expects.

MISSINGNESS IS A FEATURE, NOT AN IMPUTATION. 41 of 270 sessions have no
treatment label. The vector carries an explicit observed/missing flag rather
than silently imputing a class, so the model can learn that "unknown" differs
from any known status.
"""

from __future__ import annotations

import numpy as np

#: Treatment tokens observed in the clinical table, plus an explicit unknown.
TREATMENT_CLASSES = ("CRT", "TMZ", "no")
#: Fallback when a fold's Δt has no spread. Standardising by ~0 would produce
#: infinities; 1.0 leaves the centred value unscaled and is recorded.
MIN_SD = 1e-6


class FoldStandardiser:
    """Zero-mean unit-variance statistics fitted on TRAINING pairs only.

    Constructed from a pair list so it cannot accidentally be fitted on the
    whole cohort: the caller must decide which pairs to hand it, and `rung.py`
    hands it the training split.
    """

    def __init__(self, train_pairs: list, key: str = "delta_days"):
        vals = [float(np.log(max(p[key], 1.0))) for p in train_pairs
                if p.get(key) is not None]
        if not vals:
            raise ValueError(f"no usable {key} in the training pairs")
        self.key = key
        self.mean = float(np.mean(vals))
        self.sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
        self.degenerate_sd = self.sd < MIN_SD
        if self.degenerate_sd:
            self.sd = 1.0
        self.n_train = len(vals)

    def transform(self, delta_days) -> float:
        if delta_days is None:
            # Δt is present for all 208 pairs; 0.0 is the fold mean in
            # standardised space and is the least-assuming fallback.
            return 0.0
        return (float(np.log(max(delta_days, 1.0))) - self.mean) / self.sd

    def to_dict(self) -> dict:
        return {"key": self.key, "log_mean": self.mean, "log_sd": self.sd,
                "n_train_pairs": self.n_train,
                "degenerate_sd": self.degenerate_sd,
                "fitted_on": "TRAINING pairs only — never the held-out fold"}


def _treatment_onehot(token) -> list:
    """One-hot plus an explicit observed flag. Never imputes a class."""
    observed = token in TREATMENT_CLASSES
    vec = [1.0 if observed and token == c else 0.0 for c in TREATMENT_CLASSES]
    return vec + [1.0 if observed else 0.0]


def cond_dim(rung: str, n_dose: int = 0) -> int:
    r = rung.upper()
    if r.startswith("C0"):
        return 0
    d = 1                                   # Δt
    if r.startswith("C2") or r.startswith("C4"):
        d += len(TREATMENT_CLASSES) + 1     # one-hot + observed flag
    if r.startswith("C3") or r.startswith("C4"):
        d += n_dose
    return d


def make_cond_fn(rung: str, standardiser: FoldStandardiser | None = None,
                 dose_features: dict | None = None):
    """Returns `pair -> np.ndarray` for the given rung, or None for C0.

    The returned function is the ONLY place a rung's information content is
    decided, so the difference between rungs is one readable expression rather
    than scattered branches.
    """
    r = rung.upper()
    if r.startswith("C0"):
        return None
    if standardiser is None:
        raise ValueError(f"{rung} needs Δt, so a FoldStandardiser is required")

    want_treat = r.startswith("C2") or r.startswith("C4")
    want_dose = r.startswith("C3") or r.startswith("C4")

    def cond_fn(pair: dict) -> np.ndarray:
        v = [standardiser.transform(pair.get("delta_days"))]
        if want_treat:
            # INPUT-side treatment: what the patient was receiving when the
            # input scan was taken. Using the TARGET session's status would
            # supply information from the future being predicted.
            v += _treatment_onehot(pair.get("input_treatment"))
        if want_dose:
            feats = (dose_features or {}).get(pair["subject"])
            if feats is None:
                raise KeyError(f"no dose features for {pair['subject']}")
            v += list(feats)
        return np.asarray(v, dtype=np.float32)

    return cond_fn


def describe(rung: str, standardiser: FoldStandardiser | None = None) -> dict:
    r = rung.upper()
    return {
        "rung": rung,
        "cond_dim": cond_dim(rung),
        "carries": (["nothing"] if r.startswith("C0") else
                    ["log(delta_days), standardised on training folds only"]
                    + (["treatment one-hot (CRT/TMZ/no) + observed flag"]
                       if r.startswith(("C2", "C4")) else [])
                    + (["dose features"] if r.startswith(("C3", "C4")) else [])),
        "standardiser": standardiser.to_dict() if standardiser else None,
        "amd_001_note": (
            "C1 is NOT a treatment-free reference. Δt separates treatment "
            "phases (U = 0.629, interval ratio 6.5x), so C2 − C1 understates "
            "the treatment effect. C0 is the treatment-free reference."),
        "treatment_side": ("input session, never the target — the target's "
                           "status is information from the future"),
    }
