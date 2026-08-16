# ── MASTER_SAILOR_PIPELINE.ipynb § 03 — Provenance / dataset audit (Stage 1) ──
# CPU-only. Downloads nothing; writes nothing into sailor_v1/.
#
# ONE pass over derivatives.tar.bz2 answers G7 (intervals_days.txt), G8
# (raw-mni-link.tsv) and builds the clinical table. bz2 is sequential, so a
# second pass would cost another full decompression of the same 43 GB.

import sailor
sailor.require("single_pass", "archive_selection", "known_issues",
               "intervals_days", "result_persistence")

from sailor.config import get_paths
paths = get_paths(); ROOT = paths.dataset_root

# 03a — loose files. Seconds; artefact -> 06_QC_REPORTS/
from sailor.stage1.inspect_tables import inspect
inspect(paths.legacy_root, project_root=ROOT)

# 03b — documented issues from history.txt. Instant; no I/O.
from sailor.data.known_issues import summary
import json; print(json.dumps(summary(), indent=2)[:4000])

# 03c — THE ONE LONG RUN (~90 min). Everything after this reads the cache.
from sailor.stage1.audit import run_stage1_audit, print_report as print_audit
report = run_stage1_audit(paths, verify_hashes=False, read_volumes=False,
                          archives_to_scan=["derivatives.tar.bz2"])
print_audit(report)

# 03d — clinical table. Reads the cache the pass above just wrote: seconds.
from sailor.stage1.clinical_table import collect
clin = collect(paths)

# 03e — §5 confound, now on real weeks. Artefact -> 10_EXPERIMENTS/
from sailor.experiments.confound import run_and_write, print_report
weeks = {f"{r['subject']}/{r['session']}": r["weeks_from_first"]
         for r in clin["rows"] if r.get("weeks_from_first") is not None}
prereg = run_and_write(clin["rows"], ROOT / "10_EXPERIMENTS",
                       weeks_by_session=weeks or None)
print_report(prereg)

# 03f — later, when the raw side is needed for the full G8 join, and for G1/G10:
# report = run_stage1_audit(paths, verify_hashes=True, max_hash_gb=None)
