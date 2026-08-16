# =============================================================================
# GATE-3 — headroom above persistence, and the pre-registered primary metric.
#
# ORDER MATTERS. Cell A reports every candidate side by side and selects
# NOTHING. Cell B records your choice and refuses to run twice. Choosing after
# a C-rung has run would be a forking path; right now no model exists, so the
# choice cannot be tuned to a result.
# =============================================================================

# --- CELL A — recompute the baseline with the scale-free metrics, then compare
from sailor.stage3 import persistence, headroom

res = persistence.run(ROOT)
persistence.print_report(res)

cmp = headroom.compare(res, power=0.80, alpha=0.05)
headroom.print_comparison(cmp)

print("""
HOW TO READ THE MDE/mean COLUMN
  It is the fraction of the baseline a model must improve on to be detectable
  at n = 26. A value near or above 1.0 means the cohort cannot resolve anything
  short of the model halving the error, which is a power limitation to state in
  advance rather than discover afterwards (GATE-3 criterion 4).

  Every MDE here is an UPPER BOUND: it assumes model-minus-baseline differences
  spread as widely as the baseline does between patients. Paired comparisons
  usually do better. The true requirement is lower, by an unknown amount.
""")


# --- CELL B — pre-register. Edit METRIC and JUSTIFICATION, then run ONCE.
METRIC = "log_volume_ratio_error"       # or volume_change_error / relative_... / dice

JUSTIFICATION = (
    "Chosen on structural grounds while only the baseline has been seen and no "
    "model, C-rung or conditioning comparison exists. Absolute volume-change "
    "error is scale-dependent, so its between-patient spread is dominated by "
    "tumour size rather than predictive difficulty; the log volume ratio is "
    "scale-free, symmetric in growth and shrinkage, and is the only headroom "
    "candidate defined on all 208 pairs including the five empty->empty pairs "
    "retained with sub-25."
)

rec = headroom.preregister(ROOT, METRIC, JUSTIFICATION, cmp)
print("PRE-REGISTERED:", rec["primary_metric"])
print("artefact:", rec["artefact"]["latest"])
