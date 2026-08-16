"""End-to-end Stage 1 smoke test on the synthetic fixture (§16).

Runs in seconds and asserts that the audit catches every defect the fixture
plants. A green smoke test says the code works; it says nothing about SAILOR.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sailor.config import Paths  # noqa: E402
from sailor.stage1.audit import run_stage1_audit  # noqa: E402
from tests import make_fixture  # noqa: E402


def main(keep: bool = False) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sailor_smoke_"))
    legacy = tmp / "sailor_v1"
    make_fixture.build(legacy)
    paths = Paths(project_name="SAILOR_SMOKE",
                  dataset_root=tmp / "SAILOR_SMOKE",
                  legacy_root=legacy,
                  code_root=Path(__file__).resolve().parents[1])

    report = run_stage1_audit(paths, verify_hashes=True, max_hash_gb=None,
                              force_rescan=True, sample_images=8, verbose=True)

    g = {r["guard"]: r for r in report["guards"]["records"]}
    inv = report["inventory"]
    ok = True

    def check(label, cond, got=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {label} {got}")
        ok = ok and bool(cond)

    print("\n--- smoke assertions ---")
    check("project root created", report["root_state"]["status"] == "OK")
    check("no download performed", report["downloads_performed"] == 0)
    check("canonical checksums verified",
          all(r["status"] in ("MATCH", "ABSENT", "NOT_IN_SHA512_TXT", "SKIPPED_DIR")
              for r in report["provenance"]["verification"]["results"]),
          str([r["status"] for r in report["provenance"]["verification"]["results"]]))
    check("expected-absent archives confirmed absent",
          len(report["provenance"]["verification"]["classification"]
              ["expected_absent_confirmed"]) == 3)
    check("quarantine detected",
          len(report["provenance"]["verification"]["classification"]["quarantine_present"]) >= 3)
    check("primary target resolved",
          report["target_resolution"]["status"] == "RESOLVED",
          report["target_resolution"]["status"])
    check("4 patients / 12 MNI sessions measured",
          inv["n_patients"] == 4 and inv["n_sessions"] == 12,
          f"{inv['n_patients']}/{inv['n_sessions']}")
    check("G1 catches the planted all-zero mask",
          g["G1"]["status"] == "FAIL"
          and len(g["G1"]["evidence"]["primary"]["all_zero"]) == 1,
          str(g["G1"]["evidence"]["primary"]["all_zero"]))
    check("G1 inventories ONCO without acting on it",
          g["G1"]["evidence"]["inventory_only"]["ONCO"]["n_measured"] > 0)
    check("G5 finds no quarantined input", g["G5"]["status"] == "PASS")
    check("G7 reads intervals_days.txt but refuses to call it exact",
          g["G7"]["status"] == "FAIL"
          and g["G7"]["evidence"]["source"] == "intervals_days.txt"
          and g["G7"]["evidence"]["exact_source_available"] is False,
          f"n_with_dt={g['G7']['evidence'].get('n_sessions_with_delta_t')}")
    check("clinical table built with Δt and treatment",
          report["clinical"]["n_sessions"] > 0
          and report["clinical"]["n_sessions_with_days_from_first"] > 0,
          f"{report['clinical']['n_sessions_with_days_from_first']} sessions with Δt")
    check("documented leakage sessions surfaced",
          "sub-04/ses-01" in report["known_issues"]["sessions_with_future_leakage"])
    check("primary cohort reduced by documented missing masks",
          report["known_issues"]["n_patients_primary_cohort"]
          < len(report["known_issues"]["dose"]["eligible"]) + 2)
    check("G8 surfaces raw exams with no MNI counterpart",
          g["G8"]["evidence"]["n_raw_without_mni_counterpart"] == 2 * 4
          and all(v == 2 for v in
                  g["G8"]["evidence"]["n_raw_exams_dropped_before_mni_ses01"].values()),
          f"n_raw_without_mni={g['G8']['evidence']['n_raw_without_mni_counterpart']}")
    check("G8 reports the unmatched MNI session",
          g["G8"]["status"] == "FAIL"
          and "sub-04/ses-03" in g["G8"]["evidence"]["unmatched_mni_sessions"],
          str(g["G8"]["evidence"]["unmatched_mni_sessions"]))
    check("G9 excludes the missing.tsv session",
          any(e["session"] == "sub-03/ses-03"
              for e in g["G9"]["evidence"]["excluded_primary"]),
          str(g["G9"]["evidence"]["excluded_primary"]))
    check("G10 measured intensities and contradicted the uint8 claim",
          g["G10"]["evidence"]["n_images_measured"] > 0
          and g["G10"]["evidence"]["descriptor_claim_holds"] is False)
    check("treatment `unknown` handled as missing, not a class",
          report["treatment"]["status_counts"].get("MISSING", 0) == 4
          and "unknown" not in report["treatment"]["status_counts"])
    check("dose prerequisites measured",
          report["dose"]["n_patients_with_dose"] == 3
          and report["dose"]["registration_status"] != "NO_DOSE_FILES_FOUND",
          report["dose"]["registration_status"])
    check("gap report produced, nothing fetched",
          report["gap_report"]["downloads_performed"] == 0
          and len(report["gap_report"]["lines"]) > 0)
    check("resource card is measured, not estimated",
          report["resource_card"]["profiled"] == "YES"
          and isinstance(report["resource_card"]["wall_seconds"], float))
    check("manifests carry the target lock",
          all(Path(p).exists() for p in report["manifests"].values()))

    print("\n--- rendered report ---")
    from sailor.qc.report import render_text
    print(render_text(report))

    if not keep:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nSMOKE {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(keep="--keep" in sys.argv))
