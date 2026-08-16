"""Formal decision gates (pre-Phase 5).

The confound findings changed what this project can claim. That change must be
recorded as an amendment with criteria fixed in advance, not absorbed silently
into the objective — a project whose goalposts drift to match its results has no
result at all.

Three things are formalised here:

  * a CLAIM TAXONOMY, because "the model works" conflates three separable claims
  * GATES with numeric criteria, each decided before the evidence that decides it
  * an AMENDMENT LOG, so a reader can see which decisions preceded which results

Nothing in this module runs an experiment. It states what would count.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------- claim taxonomy
# A model can satisfy C_REALISM and C_PREDICTION completely while failing
# C_TREATMENT. Reporting them as one claim is the central overclaim risk in this
# literature, and the confound findings make it the likely failure mode here.

CLAIMS = {
    "C_REALISM": {
        "question": "Can the model generate a plausible future MRI?",
        "evidence": ["SSIM", "PSNR", "visual assessment", "distributional metrics"],
        "does_not_establish": [
            "that the prediction is closer to THIS patient's future than a trivial baseline",
            "that any treatment variable contributed",
        ],
        "trap": ("A model that reproduces the input image scores well on realism. "
                 "Realism metrics are computed against the true future but are "
                 "dominated by unchanged anatomy, which is most of the volume."),
    },
    "C_PREDICTION": {
        "question": ("Does the model predict the future better than persistence "
                     "and the other baselines?"),
        "evidence": ["change-region error", "volume-change error",
                     "growth-velocity error", "whole-mask Dice (secondary)"],
        "does_not_establish": ["that any treatment variable contributed"],
        "trap": ("Whole-mask Dice at short Δt rewards copying the input. The "
                 "primary metric MUST be change-sensitive; whole-mask Dice is "
                 "reported as context, never as the headline."),
    },
    "C_TREATMENT": {
        "question": ("Does the treatment variable carry information beyond "
                     "protocol position, Δt, and baseline tumour geometry?"),
        "evidence": ["C2 vs C0 and C1", "C3 vs geometry-derived dose control",
                     "P1", "P2", "P3"],
        "does_not_establish": ["anything about realism or overall accuracy"],
        "trap": ("Every candidate treatment variable measured so far is largely "
                 "recoverable from protocol position. A gain over a treatment-"
                 "blind baseline is not evidence of treatment information unless "
                 "the baseline was itself free of protocol information."),
    },
}

CLAIM_INDEPENDENCE = (
    "These claims are logically independent and are reported separately. A "
    "positive C_REALISM and C_PREDICTION with a negative C_TREATMENT is a "
    "complete, publishable result and is the outcome the current evidence "
    "predicts. Merging them into a single narrative would be the overclaim this "
    "taxonomy exists to prevent.")


# ------------------------------------------------------------------ the ladder
# Two ablation families, deliberately NOT interleaved. The first asks what
# INFORMATION contributes; the second asks what ARCHITECTURE contributes. Mixing
# them produces a table in which no row answers either question.

SCIENTIFIC_RUNGS = [
    {"id": "C-1", "name": "Persistence", "inputs": "copy the input mask forward",
     "role": "floor (G3). Any configuration failing to beat this has produced nothing."},
    {"id": "C0", "name": "MRI only", "inputs": "imaging history",
     "role": ("PRIMARY treatment-free reference (AMD-001). Carries no protocol "
              "information of any kind.")},
    {"id": "C1", "name": "MRI + Δt", "inputs": "imaging history + interval",
     "role": ("Protocol-phase-conditioned, NOT treatment-free: Δt vs phase "
              "U = 0.629, median interval ratio 6.5x. Descriptive rung.")},
    {"id": "C2", "name": "MRI + treatment status", "inputs": "imaging + status",
     "role": ("Status is ~81% recoverable from protocol position (UNTENABLE). "
              "Reported as a confound measurement.")},
    {"id": "C2+Δt", "name": "MRI + treatment + Δt", "inputs": "imaging + status + interval",
     "role": ("Explicit entanglement demonstration. If C2+Δt ≈ C1 ≈ C2, the three "
              "encode one construct and the ladder says so directly.")},
    {"id": "C3-G", "name": "MRI + geometry-derived synthetic dose",
     "inputs": "imaging + synthetic field from baseline mask only",
     "role": ("The control that makes C3-R interpretable. Contains NO treatment "
              "information by construction.")},
    {"id": "C3-R", "name": "MRI + real spatial RT dose",
     "inputs": "imaging + delivered dose map",
     "role": "The only candidate for a treatment-specific claim."},
    {"id": "FULL", "name": "Complete multimodal architecture",
     "inputs": "all of the above, as designed",
     "role": ("The intended framework and the main demonstration. Its performance "
              "establishes C_REALISM and C_PREDICTION; it does not by itself "
              "establish C_TREATMENT.")},
]

ARCHITECTURAL_ABLATIONS = [
    {"id": "A1", "removes": "3D foundation encoder pretraining"},
    {"id": "A2", "removes": "temporal encoder (single timepoint only)"},
    {"id": "A3", "removes": "cross-attention fusion (concatenation instead)"},
    {"id": "A4", "removes": "treatment/protocol encoder"},
    {"id": "A5", "removes": "residual formulation (direct prediction)"},
    {"id": "A6", "removes": "diffusion head (deterministic regression)"},
]

ABLATION_SEPARATION = (
    "Family A (scientific variables) and Family B (architecture) are reported in "
    "separate tables and are never interleaved. Family B runs ONLY on the "
    "configuration selected by Family A, so an architectural gain is never "
    "confounded with an information gain, and vice versa.")

# ---- multiplicity -----------------------------------------------------------
# 8 scientific rungs and 6 architectural ablations admit up to 43 pairwise
# comparisons against 26 independent patients. Reporting the largest observed
# gap across that many comparisons is a near-guarantee of a false positive.

PRIMARY_COMPARISON = {
    "comparison": "C3-R vs C3-G",
    "why": ("This is the only comparison in the whole ladder that could establish "
            "C_TREATMENT. Real delivered dose against a field constructed from "
            "baseline geometry alone isolates treatment-plan information from "
            "tumour location. Every other comparison is descriptive given the "
            "measured confounding."),
    "metric": "change-sensitive primary metric (AMD-005)",
    "uncertainty": "patient-level bootstrap, 26 units (AMD-003)",
    "decision_rule": ("C3-R must beat C3-G outside patient-level bootstrap CIs "
                      "AND degrade under P2. Both, not either."),
}

SECONDARY_COMPARISONS = {
    "comparisons": ["C0 vs C-1", "C1 vs C0", "C2 vs C0", "C2+Δt vs C1",
                    "FULL vs C0", "all Family B ablations"],
    "handling": ("Reported with patient-level CIs and an explicit statement that "
                 "they are secondary. No secondary comparison may be promoted to "
                 "a headline claim after the fact. Where a family of secondary "
                 "comparisons is used to support a single conclusion, "
                 "Holm-Bonferroni correction is applied within that family and "
                 "the uncorrected values are also shown."),
    "prohibited": ("Selecting the largest observed gap across rungs and reporting "
                   "it as the finding. The primary comparison is fixed above, "
                   "before any rung has been run."),
}

# ---------------------------------------------------------------------- gates

GATES = [
    {
        "id": "GATE-0",
        "name": "Data validity",
        "runs_before": "everything",
        "question": "Are the frozen pairs built on measured, non-degenerate data?",
        "criteria": [
            "G1 returns PASS or FAIL on measured voxel statistics — not INCONCLUSIVE",
            "G10 returns a measured dtype and value range for the MNI derivatives",
            "any session with a degenerate primary mask is removed and the pair "
            "manifest re-frozen with a new content hash",
        ],
        "decision": {
            "GO": "proceed to GATE-1",
            "NO_GO": "re-freeze the cohort and repeat; no modelling on unmeasured masks",
        },
        "status": "OPEN — requires one voxel-reading pass",
    },
    {
        "id": "GATE-1",
        "name": "Dose–geometry audit",
        "runs_before": "any C3 work, including preprocessing or registration",
        "question": ("Is the dose map substantially a re-encoding of baseline "
                     "tumour geometry?"),
        "rationale": ("Radiotherapy dose is planned on the GTV plus a margin, so "
                      "a dose map is approximately a smoothed dilation of the "
                      "baseline tumour. Conditioning on it may re-supply the "
                      "input mask rather than add treatment information."),
        "measurements": [
            "Dice(>=95% isodose, baseline CL mask dilated at r = 5,10,15,20 mm)",
            "Dice at the dilation radius that maximises overlap, per patient",
            "mean and Hausdorff distance between isodose surface and mask surface",
            "centroid displacement, isodose vs baseline mask",
            "dose value as a function of distance from the mask surface",
        ],
        "criteria": {
            "GEOMETRY_DOMINATED": ("median best-radius Dice >= 0.70 -> dose is "
                                   "largely baseline geometry; C3 requires the "
                                   "geometry-matched control of GATE-2 and may "
                                   "not be claimed on P2 alone"),
            "PARTIALLY_INDEPENDENT": ("0.40 <= median Dice < 0.70 -> control "
                                      "still required; report the overlap"),
            "INDEPENDENT": ("median Dice < 0.40 -> dose carries substantial "
                            "non-geometric structure; standard P2 suffices"),
        },
        "decision": {
            "GO": "proceed to GATE-2 with the control mandated by the band above",
            "NO_GO": ("if dose is geometry-dominated AND the GATE-2 control shows "
                      "no incremental information, C3 is reported as a negative "
                      "result and no dose-based treatment claim is made"),
        },
        "status": "OPEN",
    },
    {
        "id": "GATE-2",
        "name": "Incremental information over a geometry-matched control",
        "runs_before": "any treatment-specific claim from dose",
        "question": ("Does REAL dose beat a synthetic dose field derived only "
                     "from baseline tumour geometry?"),
        "control_construction": [
            "dilate the baseline CL mask by the GATE-1 best radius",
            "apply a distance-decaying falloff from the mask surface",
            "rescale so the synthetic field matches the real map's value "
            "distribution (quantile matching) and gradient scale",
            "the control must match smoothness and magnitude, or real dose can "
            "win on low-level statistics rather than on treatment content",
        ],
        "criteria": [
            "C3(real dose) must beat C3(geometry-derived dose) outside "
            "patient-level bootstrap CIs on the change-sensitive primary metric",
            "P2 (dose shuffled between patients) must degrade C3(real dose)",
            "P2 alone is INSUFFICIENT: shuffling dose also shuffles tumour "
            "location, so a geometry effect degrades under P2 exactly as a "
            "treatment effect would",
        ],
        "decision": {
            "GO": "C3 may be reported as treatment-specific",
            "NO_GO": ("C3 is reported as a geometry effect; the paper states that "
                      "no treatment variable in this dataset carried information "
                      "beyond protocol position and baseline geometry"),
        },
        "status": "OPEN",
    },
    {
        "id": "GATE-3",
        "name": "Persistence headroom and minimum detectable effect",
        "runs_before": "any modelling beyond baselines",
        "question": ("Is there enough headroom above persistence to detect any "
                     "plausible model gain at n = 26 patients?"),
        "criteria": [
            "persistence reported overall AND stratified by the frozen Δt bands",
            "all uncertainty resampled at the PATIENT level (26 units), never "
            "at the pair level (208 units): pairs within a patient are not "
            "independent and pair-level CIs would be roughly sqrt(208/26) ~ 2.8x "
            "too narrow",
            "MDE stated as a number on the primary metric before any model runs",
            "if the MDE exceeds the largest gain reported in the comparable "
            "literature, that is recorded as a power limitation in advance",
        ],
        "decision": {
            "GO": "proceed to conditioning ladder",
            "NO_GO": ("report baselines and the identifiability analysis; do not "
                      "run an ablation ladder whose rungs cannot be resolved"),
        },
        "status": "OPEN",
    },
    {
        "id": "GATE-4",
        "name": "Architectural ablation warrant",
        "runs_before": "Family B architectural ablations (NOT the full model)",
        "question": ("Is the Family B ablation ladder resolvable at this sample "
                     "size?"),
        "note": ("REVISED per AMD-007. The full architecture is built and "
                 "evaluated regardless of this gate — it is the intended "
                 "framework and the main demonstration, and a negative treatment "
                 "result carries more weight coming from a strong model than a "
                 "weak one. This gate governs whether the six-way ARCHITECTURAL "
                 "ablation is interpretable, not whether the model is built."),
        "criteria": [
            "GATE-0 and GATE-3 passed",
            "the Family A winner is identified first; Family B runs only on it",
            "the MDE from GATE-3 is smaller than the smallest architectural gain "
            "the ablation is meant to detect",
        ],
        "decision": {
            "GO": "run the six Family B ablations and report with correction",
            "NO_GO": ("report the full model's performance, and state that "
                      "individual architectural components could not be resolved "
                      "at n = 26 patients. The framework result stands; the "
                      "component attribution does not."),
        },
        "status": "OPEN",
    },
]

GATE_ORDER = ["GATE-0", "GATE-1", "GATE-2", "GATE-3", "GATE-4"]

GOVERNING_PRINCIPLE = (
    "Do not assume a conditioning variable represents treatment because it is "
    "clinically labelled as treatment information. Demonstrate incremental "
    "information beyond protocol position, Δt, and baseline tumour geometry "
    "first. Every variable examined so far has failed that test.")

FRAMING = {
    "not_this": "Our model proves treatment-aware prediction works.",
    "this": ("We develop a multimodal longitudinal prediction framework and "
             "systematically determine how much predictive information comes from "
             "MRI history, protocol time, treatment status, and spatial dose "
             "under strong confounding."),
    "role_of_the_full_model": ("The full architecture remains the main "
                               "demonstration and is built and evaluated as "
                               "designed. The ablation and control ladder "
                               "determines which claims it supports."),
}

NO_GO_IS_A_RESULT = (
    "A NO_GO at any gate is a publishable finding, not a project failure. The "
    "identifiability analysis stands on its own: in a standard longitudinal "
    "glioma dataset, treatment status is largely determined by protocol "
    "position, scan intervals differ several-fold between phases, and dose maps "
    "may re-encode baseline geometry — so the conventional treatment-aware "
    "ablation design cannot separate treatment from time. Reaching that "
    "conclusion quickly is the efficient outcome.")


# ------------------------------------------------------------------ amendments

AMENDMENTS = [
    {
        "id": "AMD-001",
        "section": "§11.1",
        "date": "2026-08-12",
        "change": ("C0 (MRI only) becomes the primary treatment-free reference; "
                   "C1 is demoted to a descriptive rung and renamed "
                   "protocol-phase-conditioned."),
        "prompted_by": ("Measured: Δt vs input treatment status U = 0.629, "
                        "balanced accuracy 0.643, median interval ratio 6.5x "
                        "across phases. C1 (MRI + Δt) is therefore not a "
                        "treatment-free reference."),
        "nature": ("Corrects a planning error. The original ladder assumed Δt "
                   "was treatment-independent; it is not. No model result "
                   "influenced this."),
        "results_seen_before_amendment": "none — no C-rung has been run",
    },
    {
        "id": "AMD-002",
        "section": "§14",
        "date": "2026-08-12",
        "change": ("Persistence is reported stratified by frozen Δt bands "
                   "<=21 d / 22-90 d / >90 d, with n per band."),
        "prompted_by": ("Measured Δt distribution: q25 = 14 d, median = 70 d, "
                        "max = 371 d. Pooling produces a headline dominated by "
                        "the short-interval quarter, where persistence is "
                        "near-ceiling."),
        "nature": "Metric specification fixed before any baseline was computed.",
        "results_seen_before_amendment": "none",
    },
    {
        "id": "AMD-003",
        "section": "§6, §20",
        "date": "2026-08-12",
        "change": ("All uncertainty, CIs and MDE resample at the PATIENT level "
                   "(26 units), never the pair level (208 units)."),
        "prompted_by": ("208 pairs arise from 26 patients, 2-15 pairs each. "
                        "Pair-level resampling would understate CI width by "
                        "roughly sqrt(208/26) ~ 2.8x."),
        "nature": "Correctness requirement, not a design change.",
        "results_seen_before_amendment": "none",
    },
    {
        "id": "AMD-004",
        "section": "§17, §23",
        "date": "2026-08-12",
        "change": ("Scientific hierarchy reframed: the primary contribution is "
                   "the identifiability and confounding analysis. Prediction and "
                   "generation models are demonstrations of its consequences."),
        "prompted_by": ("Confound verdict UNTENABLE at matched resolution; Δt "
                        "entangled with phase; n = 26 patients insufficient to "
                        "resolve the ablation ladder."),
        "nature": ("Reframes the claim, not the analysis. Every planned rung is "
                   "still run and reported."),
        "results_seen_before_amendment": "none",
    },
    {
        "id": "AMD-005",
        "section": "§8, §21",
        "date": "2026-08-12",
        "change": ("The primary prediction metric is change-sensitive (error in "
                   "the changed region, volume-change error). Whole-mask Dice is "
                   "reported as secondary context."),
        "prompted_by": ("At q25 = 14 days a copy-forward prediction scores near "
                        "ceiling on whole-mask Dice, so that metric cannot "
                        "distinguish a model from the trivial solution."),
        "nature": ("Closes a measurement flaw that would have inflated every "
                   "downstream comparison."),
        "results_seen_before_amendment": "none",
    },
    {
        "id": "AMD-007",
        "section": "§11.1, §17, §19, §23",
        "date": "2026-08-12",
        "change": ("The full multimodal architecture is retained, built and "
                   "evaluated as designed. What changes is the experimental "
                   "hierarchy and interpretation, not the architecture. The "
                   "scientific ladder is expanded to eight rungs (C-1, C0, C1, "
                   "C2, C2+Δt, C3-G, C3-R, FULL); Family A (information) and "
                   "Family B (architecture) ablations are separated; one primary "
                   "comparison (C3-R vs C3-G) is named in advance."),
        "prompted_by": ("Investigator direction, plus the multiplicity problem: "
                        "8 rungs and 6 architectural ablations admit up to 43 "
                        "pairwise comparisons against 26 independent patients. "
                        "Without a named primary comparison the ladder becomes a "
                        "search for the largest gap."),
        "nature": ("Supersedes the earlier proposal to reframe the project as a "
                   "confound analysis with the model demoted. A negative "
                   "treatment result carries more evidential weight when it comes "
                   "from a strong model than a weak one, which is an argument for "
                   "keeping the architecture that the earlier proposal missed. "
                   "GATE-4 is revised accordingly: it now governs whether the "
                   "ARCHITECTURAL ablation is interpretable, not whether the "
                   "model is built."),
        "results_seen_before_amendment": "none — no C-rung has been run",
    },
    {
        "id": "AMD-006",
        "section": "§11.1, §17",
        "date": "2026-08-12",
        "change": ("P2 (dose shuffle) is insufficient alone for any C3 claim; a "
                   "geometry-matched synthetic-dose control is mandatory."),
        "prompted_by": ("Dose is planned on the GTV plus margin, so it may "
                        "re-encode baseline geometry. Shuffling dose between "
                        "patients also shuffles tumour location, so a geometry "
                        "effect degrades under P2 exactly as a treatment effect "
                        "would — P2 cannot distinguish them."),
        "nature": "Adds a control; removes nothing.",
        "results_seen_before_amendment": "none",
    },
]


# --------------------------------------------------------------------- output

def protocol() -> dict:
    return {
        "document": "pre_phase5_decision_gates",
        "scientific_rungs": SCIENTIFIC_RUNGS,
        "architectural_ablations": ARCHITECTURAL_ABLATIONS,
        "ablation_separation": ABLATION_SEPARATION,
        "primary_comparison": PRIMARY_COMPARISON,
        "secondary_comparisons": SECONDARY_COMPARISONS,
        "framing": FRAMING,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "governing_principle": GOVERNING_PRINCIPLE,
        "claims": CLAIMS,
        "claim_independence": CLAIM_INDEPENDENCE,
        "gates": GATES,
        "gate_order": GATE_ORDER,
        "no_go_is_a_result": NO_GO_IS_A_RESULT,
        "amendments": AMENDMENTS,
        "status": ("PRE-REGISTERED — every criterion above was fixed before the "
                   "evidence that decides it. No C-rung has been run."),
    }


def write(project_root, prefix: str = "v2_") -> dict:
    out = Path(project_root) / "10_EXPERIMENTS"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{prefix}decision_gates.json"
    p.write_text(json.dumps(protocol(), indent=2))
    return {"path": str(p), "n_gates": len(GATES), "n_amendments": len(AMENDMENTS)}


def print_protocol() -> None:
    line = "=" * 78
    thin = "-" * 78
    print(line); print("PRE-PHASE-5 DECISION GATES"); print(line)
    print(f"\nPRINCIPLE\n  {GOVERNING_PRINCIPLE}\n")
    print(thin); print("CLAIM TAXONOMY — three separable claims"); print(thin)
    for k, v in CLAIMS.items():
        print(f"  {k}: {v['question']}")
        print(f"    trap: {v['trap']}")
    print(f"\n  {CLAIM_INDEPENDENCE}\n")
    print(thin); print("FAMILY A — scientific rungs (what INFORMATION contributes)"); print(thin)
    for r in SCIENTIFIC_RUNGS:
        print(f"  {r['id']:<6} {r['name']}")
        print(f"         {r['role']}")
    print(f"\n{thin}\nFAMILY B — architectural ablations (what ARCHITECTURE contributes)\n{thin}")
    for a in ARCHITECTURAL_ABLATIONS:
        print(f"  {a['id']}  minus {a['removes']}")
    print(f"\n  {ABLATION_SEPARATION}\n")
    print(thin); print("MULTIPLICITY"); print(thin)
    print(f"  PRIMARY comparison: {PRIMARY_COMPARISON['comparison']}")
    print(f"    {PRIMARY_COMPARISON['why']}")
    print(f"    rule: {PRIMARY_COMPARISON['decision_rule']}")
    print(f"  SECONDARY: {', '.join(SECONDARY_COMPARISONS['comparisons'])}")
    print(f"    {SECONDARY_COMPARISONS['prohibited']}")
    print()
    print(thin); print("FRAMING"); print(thin)
    print(f"  NOT: {FRAMING['not_this']}")
    print(f"  THIS: {FRAMING['this']}")
    print(f"  full model: {FRAMING['role_of_the_full_model']}")
    print()
    print(thin); print("GATES"); print(thin)
    for g in GATES:
        print(f"  {g['id']} {g['name']}  [{g['status']}]")
        print(f"    {g['question']}")
        print(f"    GO   -> {g['decision']['GO']}")
        print(f"    NO_GO-> {g['decision']['NO_GO']}")
    print(f"\n  {NO_GO_IS_A_RESULT}\n")
    print(thin); print("AMENDMENTS"); print(thin)
    for a in AMENDMENTS:
        print(f"  {a['id']} {a['section']}: {a['change']}")
        print(f"    prompted by: {a['prompted_by']}")
        print(f"    results seen first: {a['results_seen_before_amendment']}")
    print(line)
