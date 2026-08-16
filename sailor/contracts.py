"""Shape and record contracts (§8, §16).

Stage 1 uses the manifest and guard-record contracts below. The tensor contracts
for the encoder/fusion/diffusion modules are declared here as symbolic shapes
and asserted on entry and exit once those modules exist; they are intentionally
not implemented before the Phase gate that specifies them.
"""

from __future__ import annotations

GUARD_STATUSES = ("PASS", "FAIL", "INCONCLUSIVE")

GUARD_RECORD_KEYS = ("guard", "title", "status", "detail", "evidence")

SESSION_MANIFEST_ROW_KEYS = (
    "subject", "session", "sequences_present", "n_annotation_files",
    "primary_target_path", "primary_target_available",
    "treatment_status", "treatment_observed", "treatment_missing_indicator",
    "days_from_prev", "delta_t_kind", "delta_t_source",
    "passes_primary_modality_requirement",
)

# §8 symbolic tensor contracts. B=batch, T=timepoints, C=channels,
# (D,H,W)=volume, Fd=feature dim. Asserted by assert_shape once modules land.
TENSOR_CONTRACTS = {
    "spatial_encoder":  {"in": "(B, C, D, H, W)", "out": "(B, Fd)"},
    "temporal_encoder": {"in": "(B, T, Fd) + (B, T)", "out": "(B, Fd)"},
    "treatment_encoder": {"in": "(B, T, Ft) + (B, T) missingness", "out": "(B, Fd)"},
    "cross_attention":  {"in": "(B, Fd) query + (B, Fd) key/value", "out": "(B, Fd)"},
    "residual_head":    {"in": "(B, C, D, H, W) + (B, Fd) + (B, 1)", "out": "(B, C, D, H, W)"},
    "diffusion":        {"in": "(B, C, D, H, W) + residual + (B, Fd) + (B, 1)",
                         "out": "(B, C, D, H, W)"},
}


def assert_guard_record(rec: dict) -> None:
    missing = [k for k in GUARD_RECORD_KEYS if k not in rec]
    assert not missing, f"guard record missing keys: {missing}"
    assert rec["status"] in GUARD_STATUSES, f"illegal guard status {rec['status']!r}"


def assert_manifest_row(row: dict) -> None:
    missing = [k for k in SESSION_MANIFEST_ROW_KEYS if k not in row]
    assert not missing, f"manifest row missing keys: {missing}"


def assert_target_lock(obj: dict, expected_mask: str, expected_component: str) -> None:
    """A run whose manifest target does not match the lock is invalid (§3.2)."""
    assert obj.get("primary_target_mask") == expected_mask, (
        f"target lock violated: manifest says {obj.get('primary_target_mask')!r}")
    assert obj.get("primary_target_component") == expected_component, (
        f"target lock violated: manifest says {obj.get('primary_target_component')!r}")


def assert_shape(tensor, expected, name: str = "tensor") -> None:
    """Shape assertion with symbolic dims: None matches any size."""
    shape = tuple(tensor.shape)
    assert len(shape) == len(expected), (
        f"{name}: rank {len(shape)} != expected {len(expected)} ({shape} vs {expected})")
    for i, (got, want) in enumerate(zip(shape, expected)):
        if want is not None:
            assert got == want, f"{name}: dim {i} = {got}, expected {want}"
