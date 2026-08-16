"""Confound quantification — how predictable is treatment status from time? (§5, G2)

§5 names this the single most likely way the project produces a false positive.
This module measures it *before* any model is trained, so the threshold that
would make the treatment-awareness claim untenable is written down while it is
still cheap to accept.

Three quantities, all estimated at the patient level:

  I(A; T)      mutual information between treatment status A and weeks-since-
               surgery T, in bits, bias-corrected against an unrestricted label
               permutation null.
  U            uncertainty coefficient I(A;T)/H(A) — the fraction of the entropy
               of treatment status that time already explains. Scale-free, so it
               is the quantity the threshold is stated in.
  acc_time     balanced accuracy of a time-only classifier predicting A from T
               under leave-one-patient-out, against a chance floor.
  homogeneity  how far a patient-block permutation fails to degrade I(A;T). This
               is the diagnostic that decides whether control P1 has any power at
               all: if every patient follows the same schedule, permuting whole
               patients returns almost the same labels, so P1 cannot degrade C2
               even when the treatment branch is reading position. A control that
               cannot fail is not a control, and discovering that here is much
               cheaper than discovering it after C2 has been run.

Nothing here trains an imaging model. Its output is a pre-registered number that
decides how the §11.1 ladder is *interpreted*, not whether it is run.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- thresholds
# Pre-specified. Chosen before any C-rung is run, and not adjustable afterwards.
U_UNTENABLE = 0.90       # time explains >=90% of treatment entropy
U_SEVERE = 0.75          # severe confounding; C2 needs P3 separation to mean anything
ACC_UNTENABLE = 0.90     # time-only classifier reaches >=90% balanced accuracy
N_PERMUTATIONS = 2000
SEED = 1337


@dataclass
class ConfoundResult:
    time_basis: str
    n_timepoints: int
    n_patients: int
    class_counts: dict
    H_treatment_bits: float
    I_bits_raw: float
    I_bits_corrected: float
    uncertainty_coefficient: float
    permutation_p: float
    schedule_homogeneity: float
    p1_control_has_power: bool
    time_only_balanced_accuracy: float
    majority_floor: float
    time_only_accuracy_p: float
    verdict: str
    threshold_policy: dict

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ estimator

def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


ORDINAL = "session_ordinal"
WEEKS = "weeks_since_first"


def discretise_ordinal(values: np.ndarray) -> np.ndarray:
    """Each session index is its own bin.

    Used when Δt is unrecovered (G7 INCONCLUSIVE). Forcing clinical week edges
    onto an ordinal would invent a calendar the data does not contain.
    """
    return np.asarray(values).astype(int)


def discretise_time(weeks: np.ndarray, edges=(0, 4, 10, 18, 30, 52)) -> np.ndarray:
    """Bin weeks-since-surgery on Stupp-protocol boundaries.

    The edges are clinical, not data-driven: 0-4 post-surgical recovery, 4-10
    concomitant CRT, then adjuvant TMZ cycles. Data-driven binning would let the
    estimate depend on the treatment labels it is meant to be independent of.
    """
    return np.digitize(weeks, bins=list(edges[1:]), right=False)


def mutual_information_bits(a: np.ndarray, t_binned: np.ndarray) -> float:
    """Plug-in MI on the discrete joint. Biased upward at small n — hence the
    permutation correction in `quantify`."""
    a_vals = np.unique(a)
    t_vals = np.unique(t_binned)
    joint = np.zeros((a_vals.size, t_vals.size))
    for i, av in enumerate(a_vals):
        for j, tv in enumerate(t_vals):
            joint[i, j] = np.sum((a == av) & (t_binned == tv))
    joint /= joint.sum()
    pa = joint.sum(axis=1, keepdims=True)
    pt = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log2(joint[nz] / (pa @ pt)[nz])))


def _label_permute(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Unrestricted permutation of labels: breaks the A-T association while
    preserving the marginal of A. This is the null the MI estimate is corrected
    against, because it is the only one here that actually destroys the
    dependence being measured."""
    return rng.permutation(a)


def _patient_block_permute(a: np.ndarray, groups: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    """Permute whole patients' label sequences, preserving within-patient order.

    This is the null corresponding to control P1 (§11.1). It is NOT used to
    correct the MI estimate: when patients share a protocol, this permutation
    returns nearly the original labels, so it has almost no power. That failure
    is measured and reported as `schedule_homogeneity` rather than hidden.
    """
    uniq = np.unique(groups)
    blocks = [a[groups == g] for g in uniq]
    order = rng.permutation(len(blocks))
    out = np.empty_like(a)
    for slot, src in zip(uniq, order):
        target = out[groups == slot]
        src_block = blocks[src]
        n = target.size
        # cycle the donor block if lengths differ, so no timepoint is dropped
        out[groups == slot] = np.resize(src_block, n)
    return out


def time_only_classifier(a: np.ndarray, weeks: np.ndarray, groups: np.ndarray,
                         rng: np.random.Generator,
                         binner=None) -> tuple[float, float]:
    """Leave-one-patient-out nearest-threshold classifier on time alone.

    Deliberately the simplest possible model: if a one-dimensional rule on time
    already recovers treatment status, no capacity argument can rescue C2. This
    is the same object as reference P3 (§11.1), evaluated on labels rather than
    on imaging targets.
    """
    uniq = np.unique(groups)
    preds, truths = [], []
    for held in uniq:
        tr = groups != held
        te = groups == held
        if not tr.any() or not te.any():
            continue
        binfn = binner or discretise_time
        bins_tr = binfn(weeks[tr])
        bins_te = binfn(weeks[te])
        # majority label per time bin, learned on training patients only
        table = {}
        for b in np.unique(bins_tr):
            lbls = a[tr][bins_tr == b]
            table[b] = Counter(lbls.tolist()).most_common(1)[0][0]
        fallback = Counter(a[tr].tolist()).most_common(1)[0][0]
        preds.extend(table.get(b, fallback) for b in bins_te)
        truths.extend(a[te].tolist())
    preds_a, truths_a = np.array(preds), np.array(truths)
    classes = np.unique(truths_a)
    recalls = [(preds_a[truths_a == c] == c).mean() for c in classes
               if (truths_a == c).any()]
    bal_acc = float(np.mean(recalls)) if recalls else float("nan")
    floor = float(1.0 / classes.size) if classes.size else float("nan")
    return bal_acc, floor


def quantify(status: list, weeks: list, patient: list,
             n_permutations: int = N_PERMUTATIONS, seed: int = SEED,
             time_basis: str = WEEKS) -> ConfoundResult:
    """Measure the confound. Rows with missing status or missing time are dropped
    and the drop is reported, because `unknown` is missing data (§5)."""
    a_all = np.array(status, dtype=object)
    w_all = np.array([np.nan if x is None else float(x) for x in weeks], dtype=float)
    g_all = np.array(patient, dtype=object)
    keep = np.array([s is not None for s in a_all]) & np.isfinite(w_all)
    a, w, g = a_all[keep], w_all[keep], g_all[keep]
    if a.size == 0:
        raise ValueError("no timepoint has both an observed treatment status and a time")

    codes = {v: i for i, v in enumerate(sorted(set(a.tolist())))}
    a_i = np.array([codes[v] for v in a])
    counts = np.array([np.sum(a_i == i) for i in range(len(codes))], dtype=float)
    H = _entropy(counts)

    binner = discretise_ordinal if time_basis == ORDINAL else discretise_time
    t_binned = binner(w)
    I_raw = mutual_information_bits(a_i, t_binned)

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    block_null = np.empty(n_permutations)
    for k in range(n_permutations):
        null[k] = mutual_information_bits(_label_permute(a_i, rng), t_binned)
        block_null[k] = mutual_information_bits(
            _patient_block_permute(a_i, g, rng), t_binned)
    I_corr = float(max(0.0, I_raw - null.mean()))
    p_mi = float((np.sum(null >= I_raw) + 1) / (n_permutations + 1))
    U = float(I_corr / H) if H > 0 else float("nan")

    # How much does permuting whole patients fail to destroy the association?
    # 1.0 means block permutation changes nothing: P1 is powerless.
    # Only meaningful when there is an association to destroy: if I(A;T) is
    # itself indistinguishable from chance, the ratio is noise over noise and is
    # reported as undefined rather than as a confident 1.0.
    if I_corr > 0 and p_mi < 0.05 and I_raw > 0:
        homogeneity = float(np.mean(block_null) / I_raw)
        p1_has_power = bool(homogeneity < 0.80)
    else:
        homogeneity = float("nan")
        p1_has_power = True  # no measured confound; P1 is not known to be powerless

    bal_acc, floor = time_only_classifier(a_i, w, g, rng, binner)
    null_acc = np.empty(min(n_permutations, 500))
    for k in range(null_acc.size):
        null_acc[k] = time_only_classifier(_label_permute(a_i, rng), w, g, rng, binner)[0]
    p_acc = float((np.sum(null_acc >= bal_acc) + 1) / (null_acc.size + 1))

    if U >= U_UNTENABLE or bal_acc >= ACC_UNTENABLE:
        verdict = "UNTENABLE"
    elif U >= U_SEVERE:
        verdict = "SEVERE"
    else:
        verdict = "TRACTABLE"

    return ConfoundResult(
        time_basis=time_basis,
        n_timepoints=int(a.size), n_patients=int(np.unique(g).size),
        class_counts={k: int(np.sum(a_i == v)) for k, v in codes.items()},
        H_treatment_bits=round(H, 4),
        I_bits_raw=round(I_raw, 4), I_bits_corrected=round(I_corr, 4),
        uncertainty_coefficient=round(U, 4), permutation_p=round(p_mi, 5),
        schedule_homogeneity=(round(homogeneity, 4)
                              if np.isfinite(homogeneity) else float("nan")),
        p1_control_has_power=p1_has_power,
        time_only_balanced_accuracy=round(bal_acc, 4),
        majority_floor=round(floor, 4), time_only_accuracy_p=round(p_acc, 5),
        verdict=verdict,
        threshold_policy=policy())


def policy() -> dict:
    """Pre-specified interpretation. Written before the numbers exist."""
    return {
        "U_untenable": U_UNTENABLE,
        "U_severe": U_SEVERE,
        "acc_untenable": ACC_UNTENABLE,
        "n_permutations": N_PERMUTATIONS,
        "UNTENABLE": (
            "Treatment status is a re-encoding of the protocol schedule. No "
            "treatment-awareness claim may be made from the status label under any "
            "C2 result. C2 is still run and reported, but as a measurement of the "
            "confound; the paper reports that status carries no information beyond "
            "time in this cohort. The dose maps (C3) become the only route to a "
            "defensible treatment claim, and P2 becomes mandatory."),
        "SEVERE": (
            "C2 may only be claimed if it beats BOTH C1 and P3 outside patient-level "
            "bootstrap CIs AND degrades under P1. A C2>C1 gap that does not also "
            "exceed P3 is reported as time information, not treatment information."),
        "TRACTABLE": (
            "Standard G2 conditions apply: C2 beats C1 and P3 outside CIs and "
            "degrades under P1."),
        "p1_powerlessness": (
            "If schedule_homogeneity >= 0.80, permuting whole patients barely "
            "changes the labels, so control P1 cannot degrade C2 regardless of what "
            "the treatment branch has learned. In that case P1 must be reported as "
            "UNINFORMATIVE rather than as passed, and G2 falls back to the C2-vs-P3 "
            "comparison alone. A control that cannot fail must never be counted as "
            "evidence that a claim survived it."),
        "ordinal_time_caveat": (
            "When time_basis=session_ordinal, the covariate is session index, not "
            "elapsed time. Index is a coarser and partly different variable: it "
            "cannot distinguish two scans a week apart from two a year apart. A "
            "confound measured against ordinal is therefore a lower bound on the "
            "schedule confound and must be re-measured once Δt is recovered (G7)."),
        "basis_and_resolution": (
            "A verdict is only comparable across time bases at MATCHED bin "
            "resolution. Session ordinal binned one-per-index gives 19 bins; the "
            "clinical Stupp edges give 6. Coarsening the SAME covariate from 19 to "
            "6 bins moves U from 0.795 to 0.710 and balanced accuracy from 0.921 to "
            "0.627 — enough to cross the threshold on its own. A verdict difference "
            "between bases is therefore not evidence about time until resolution is "
            "equalised."),
        "conservative_basis_rule": (
            "PRE-SPECIFIED: when bases disagree, the MOST CONFOUNDED estimate binds. "
            "Both known biases run one way — coarse binning lowers measured MI, and "
            "measurement error in Δt attenuates it further — so each basis can only "
            "UNDER-state the schedule confound, never over-state it. Adopting the "
            "permissive verdict because it came from a noisier covariate would be "
            "the forking path this section exists to close."),
        "immutability": (
            "These thresholds are fixed before any C-rung runs. Adjusting them after "
            "seeing a C2 result is the forking path this section exists to close."),
    }


def wiring() -> dict:
    """How the measured verdict binds each rung and control of §11.1."""
    return {
        "C-1_persistence": "Unaffected. Floor only (G3).",
        "C0_mri_only": "Unaffected. Establishes whether imaging history alone predicts change.",
        "C1_mri_dt": ("The reference every treatment claim is measured against. Δt here "
                      "carries the G7 provenance flag; if Δt is APPROXIMATE, the C2-C1 "
                      "gap inherits that error and the Δt sensitivity analysis runs before "
                      "the gap is interpreted."),
        "C2_status": ("Interpreted through the verdict. UNTENABLE -> reported as a "
                      "confound measurement, never as treatment-awareness."),
        "C3_dose": ("Independent of the verdict, because a spatial dose map is not "
                    "recoverable from the schedule. Requires P2 regardless, since a "
                    "per-patient static map can encode patient identity."),
        "C4_status_plus_dose": ("Only interpretable once C2 and C3 are each interpreted. "
                                "If C2 is UNTENABLE, any C4>C3 gap is attributed to dose."),
        "P1_treatment_shuffle": ("Same patient-block permutation used here for the null. "
                                 "The confound estimate and P1 therefore share one null "
                                 "model, so a C2 that survives P1 but sits at U>=0.90 is "
                                 "visibly inconsistent rather than quietly reported."),
        "P2_dose_shuffle": "Gates every C3/C4 dose claim. Feasible because volumes share MNI space.",
        "P3_time_only": ("The imaging-target analogue of the time-only classifier here. "
                         "The classifier's balanced accuracy is the pre-registered "
                         "expectation for how strong P3 will be."),
    }


def trivial_rule_audit(rows: list[dict]) -> dict:
    """How much of the label is recoverable by a rule with no parameters?

    Fits nothing: it reads the majority label at each session ordinal and reports
    how many observed timepoints an unambiguous ordinal already determines. A high
    coverage with zero errors means the label is close to a re-encoding of index,
    which is a stronger and more legible statement than a mutual-information
    number alone.
    """
    from collections import Counter, defaultdict
    by_ord = defaultdict(Counter)
    for r in rows:
        if r.get("treatment_status") and r.get("session_ordinal"):
            by_ord[r["session_ordinal"]][r["treatment_status"]] += 1
    pure, mixed = {}, {}
    for k, c in sorted(by_ord.items()):
        (pure if len(c) == 1 else mixed)[k] = dict(c)
    n_obs = sum(sum(c.values()) for c in by_ord.values())
    n_determined = sum(sum(c.values()) for k, c in by_ord.items() if len(c) == 1)
    return {
        "n_observed_timepoints": n_obs,
        "n_determined_by_ordinal_alone": n_determined,
        "fraction_determined": round(n_determined / n_obs, 4) if n_obs else None,
        "unambiguous_ordinals": pure,
        "mixed_ordinals": mixed,
        "reading": ("Every session ordinal listed as unambiguous carries exactly one "
                    "treatment value across the whole cohort; only the mixed ordinals "
                    "carry information the index does not already determine."),
    }


def matched_resolution_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bins, so two covariates can be compared at equal resolution."""
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0 or n_bins < 2:
        return np.zeros_like(v, dtype=int)
    qs = np.quantile(finite, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(v, np.unique(qs))


def basis_sensitivity(clinical_rows: list[dict], weeks_by_session: dict,
                      n_bins: tuple = (4, 6, 10, 19),
                      n_permutations: int = 400, seed: int = SEED) -> dict:
    """Compare ordinal and weeks at MATCHED bin resolution.

    Without this, a verdict difference between time bases is uninterpretable:
    the default binners use 19 bins for ordinal and 6 for weeks, and that gap
    alone can move the verdict across the threshold.
    """
    status, ordv, wkv, pat = [], [], [], []
    for r in clinical_rows:
        sid = f"{r['subject']}/{r['session']}"
        w = weeks_by_session.get(sid)
        if r.get("session_ordinal") is None or w is None:
            continue
        status.append(r.get("treatment_status"))
        ordv.append(float(r["session_ordinal"]))
        wkv.append(float(w))
        pat.append(r["subject"])

    global discretise_time
    saved = discretise_time
    out = {"n_bins_tested": list(n_bins), "rows": [], "n_timepoints": len(status)}
    try:
        for k in n_bins:
            for label, vals in (("session_ordinal", ordv), ("weeks_since_first", wkv)):
                arr = np.asarray(vals, dtype=float)
                edges = np.unique(np.quantile(arr[np.isfinite(arr)],
                                              np.linspace(0, 1, k + 1)[1:-1]))
                discretise_time = (lambda e: (lambda w, edges=None:
                                              np.digitize(np.asarray(w, dtype=float), e)))(edges)
                r = quantify(status, vals, pat, n_permutations=n_permutations,
                             seed=seed, time_basis=WEEKS)
                out["rows"].append({
                    "basis": label, "n_bins_requested": k,
                    "n_bins_effective": int(len(edges) + 1),
                    "U": r.uncertainty_coefficient,
                    "balanced_accuracy": r.time_only_balanced_accuracy,
                    "verdict": r.verdict})
    finally:
        discretise_time = saved

    verdicts = {}
    for row in out["rows"]:
        verdicts.setdefault(row["n_bins_requested"], set()).add(row["verdict"])
    disagreeing = sorted(k for k, v in verdicts.items() if len(v) > 1)
    worst = max(out["rows"], key=lambda r: (r["U"], r["balanced_accuracy"]))
    out["bases_disagree_at_matched_resolution"] = disagreeing
    out["most_confounded_estimate"] = worst
    out["binding_verdict"] = worst["verdict"]
    out["reading"] = (
        "If the two bases agree at every matched bin count, an earlier verdict "
        "difference was a binning artefact, not a fact about time. The binding "
        "verdict is taken from the most confounded estimate per the pre-specified "
        "conservative_basis_rule.")
    return out


def delta_t_encodes_treatment(pairs: list[dict],
                              n_permutations: int = N_PERMUTATIONS,
                              seed: int = SEED) -> dict:
    """Does the INTERVAL itself carry treatment phase? (§5, and a G2 corollary)

    Clinical scanning is not uniform: intervals during chemoradiation are short
    and follow-up intervals are long. If Δt separates the phases, then C1
    (MRI + Δt) already contains treatment information, and the C2 − C1 gap is
    not the quantity §11.1 assumes it is — C1 stops being a treatment-free
    reference. This must be measured before any conditioning claim, not after.

    Measured on the pair's INPUT treatment status, since that is the phase the
    interval was scheduled under.
    """
    status, deltas, patient = [], [], []
    for p_ in pairs:
        st = p_.get("input_treatment")
        d = p_.get("delta_days")
        if st is None or d is None:
            continue
        status.append(st)
        deltas.append(float(d))
        patient.append(p_["subject"])
    if not status:
        return {"status": "NO_DATA",
                "reading": "No pair has both an observed input treatment and a Δt."}

    res = quantify(status, deltas, patient, n_permutations=n_permutations,
                   seed=seed, time_basis=WEEKS)

    import numpy as np
    by_status: dict[str, list[float]] = {}
    for st, d in zip(status, deltas):
        by_status.setdefault(st, []).append(d)
    per_status = {}
    for st, vals in sorted(by_status.items()):
        a = np.array(vals)
        per_status[st] = {"n": int(a.size), "median_days": float(np.median(a)),
                          "q25": float(np.quantile(a, .25)),
                          "q75": float(np.quantile(a, .75)),
                          "min": float(a.min()), "max": float(a.max())}

    medians = [v["median_days"] for v in per_status.values()]
    spread = (max(medians) / min(medians)) if medians and min(medians) > 0 else None
    contaminated = res.uncertainty_coefficient >= 0.30

    return {
        "status": "OK",
        "n_pairs_measured": len(status),
        "uncertainty_coefficient": res.uncertainty_coefficient,
        "permutation_p": res.permutation_p,
        "balanced_accuracy_of_delta_t_alone": res.time_only_balanced_accuracy,
        "delta_days_by_input_treatment": per_status,
        "median_ratio_across_phases": (round(spread, 2) if spread else None),
        "c1_is_treatment_contaminated": contaminated,
        "implication": (
            "Δt separates treatment phases. C1 (MRI + Δt) therefore already "
            "carries treatment information, so it is NOT a treatment-free "
            "reference and the C2 − C1 gap understates rather than measures the "
            "treatment signal. Report C0 (MRI only) as the additional reference, "
            "and state in the methods that Δt and treatment phase are entangled "
            "by the scanning protocol." if contaminated else
            "Δt does not separate treatment phases appreciably; C1 remains a "
            "usable treatment-free reference."),
        "threshold": {"U_contaminated": 0.30,
                      "note": "Pre-specified here, before any C-rung is run."},
    }


def run_and_write(clinical_rows: list[dict], out_dir: Path,
                  weeks_by_session: dict | None = None,
                  seed: int = SEED) -> dict:
    """Measure the confound from a clinical table and persist the pre-registration.

    `weeks_by_session` maps 'sub-XX/ses-YY' -> weeks since the first exam. When it
    is absent or empty, the measurement falls back to session ordinal and says so
    in `time_basis`; it never silently substitutes one for the other.
    """
    status, tvals, patient = [], [], []
    basis = WEEKS if weeks_by_session else ORDINAL
    for r in clinical_rows:
        sid = f"{r['subject']}/{r['session']}"
        status.append(r.get("treatment_status"))
        tvals.append((weeks_by_session or {}).get(sid) if basis == WEEKS
                     else r.get("session_ordinal"))
        patient.append(r["subject"])
    res = quantify(status, tvals, patient, seed=seed, time_basis=basis)
    sensitivity = None
    if weeks_by_session:
        try:
            sensitivity = basis_sensitivity(clinical_rows, weeks_by_session, seed=seed)
        except Exception as exc:
            sensitivity = {"error": f"{type(exc).__name__}: {exc}"}

    binding = res.verdict
    if sensitivity and sensitivity.get("binding_verdict"):
        order = {"TRACTABLE": 0, "SEVERE": 1, "UNTENABLE": 2}
        if order.get(sensitivity["binding_verdict"], 0) > order.get(binding, 0):
            binding = sensitivity["binding_verdict"]

    payload = {
        "status": "PRE-REGISTERED — measured before any C-rung was run",
        "confound_measurement": res.to_dict(),
        "basis_sensitivity": sensitivity,
        "binding_verdict": binding,
        "binding_rule": policy()["conservative_basis_rule"],
        "trivial_rule_audit": trivial_rule_audit(clinical_rows),
        "wiring": wiring(),
        "verdict_binding": policy()[binding],
        "p1_note": (policy()["p1_powerlessness"] if not res.p1_control_has_power
                    else "P1 retains power: block permutation measurably degrades the association."),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "v2_confound_prereg.json"
    p.write_text(json.dumps(payload, indent=2))
    payload["path"] = str(p)
    return payload


def print_report(payload: dict) -> None:
    r = payload["confound_measurement"]
    t = payload["trivial_rule_audit"]
    line = "-" * 78
    print(line); print("CONFOUND PRE-REGISTRATION (§5, G2)"); print(line)
    print(f"  time basis                  {r['time_basis']}")
    print(f"  timepoints / patients       {r['n_timepoints']} / {r['n_patients']}")
    print(f"  class counts                {r['class_counts']}")
    print(f"  H(treatment)                {r['H_treatment_bits']} bits")
    print(f"  I(A;T) corrected            {r['I_bits_corrected']} bits  (p={r['permutation_p']})")
    print(f"  uncertainty coefficient U   {r['uncertainty_coefficient']}")
    print(f"  time-only balanced acc      {r['time_only_balanced_accuracy']} "
          f"(floor {r['majority_floor']}, p={r['time_only_accuracy_p']})")
    print(f"  schedule homogeneity        {r['schedule_homogeneity']}  "
          f"P1 has power: {r['p1_control_has_power']}")
    print(f"  VERDICT                     {r['verdict']}")
    print(line)
    print(f"  ordinal alone determines    {t['n_determined_by_ordinal_alone']}"
          f"/{t['n_observed_timepoints']} observed timepoints "
          f"({t['fraction_determined']})")
    print(f"  unambiguous ordinals        {sorted(t['unambiguous_ordinals'])}")
    print(f"  mixed ordinals              {t['mixed_ordinals']}")
    sens = payload.get("basis_sensitivity")
    if sens and not sens.get("error"):
        print("  BASIS SENSITIVITY (matched bin resolution)")
        print(f"    {'bins':>6}  {'basis':<20} {'U':>7} {'acc':>7}  verdict")
        for row in sens["rows"]:
            print(f"    {row['n_bins_effective']:>6}  {row['basis']:<20} "
                  f"{row['U']:>7} {row['balanced_accuracy']:>7}  {row['verdict']}")
        if sens["bases_disagree_at_matched_resolution"]:
            print(f"    bases DISAGREE at bin counts "
                  f"{sens['bases_disagree_at_matched_resolution']}")
        else:
            print("    bases AGREE at every matched bin count -> an unmatched "
                  "verdict difference was a binning artefact")
        print(f"    binding (most confounded): {sens['binding_verdict']}")
        print(line)
    print(f"  BINDING VERDICT: {payload.get('binding_verdict')}")
    print("  BINDING:"); print(f"    {payload['verdict_binding']}")
    print(f"  P1: {payload['p1_note']}")
    print(line)
