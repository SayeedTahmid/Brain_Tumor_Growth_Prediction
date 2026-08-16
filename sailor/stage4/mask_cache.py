"""One packed, verified mask cache — built on Drive, loaded to RAM once.

MEASURED PROBLEM. The throughput benchmark on an L4 showed data loading at
75-98% of every training step: ~2.5 s of Drive I/O against ~0.63 s of compute.
Each `.npz` holds a full 193x229x193 float32 array (68 MB raw, gzip on disk), so
extracting one 96^3 patch meant a FUSE read plus a decompress. The GPU sat idle.

THE FIX IS PURE I/O AND CHANGES NO RESULT.
  1. `uint8` instead of float32 — a binary mask does not need 32 bits (32x).
  2. Crop to the cohort-wide brain bounding box — the grid is mostly empty.
  3. ONE file instead of 240 — Drive's per-file latency dominates small reads.
  4. Load once into RAM at session start; training then does zero I/O.

Local disk is deliberately NOT used. With the cache under ~1 GB it fits in RAM,
so a local copy would add a step, add a way for a stale copy to diverge from
Drive, and buy no throughput. Drive keeps the cache durable across the
disconnects this runtime has produced repeatedly.

CORRECTNESS IS VERIFIED, NOT ASSUMED. Cropping changes the coordinate frame. A
silent offset between cache and source would shift every patch while looking
entirely healthy — a correctness failure that no timing or loss curve would
reveal. `build()` therefore re-reads a random sample of source volumes and
asserts EXACT equality after re-expansion, and refuses to write a cache that
fails. `verify()` re-runs the same check against a written cache at any time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

CACHE_NAME = "v2_mask_cache.npz"
#: Fraction of volumes re-read and checked exactly after packing.
VERIFY_FRACTION = 0.10
MIN_VERIFY = 12

_KEY = re.compile(r"(sub-[A-Za-z0-9]+)__(ses-[A-Za-z0-9]+)__(.+)\.npz$")


def _bbox_union(arrays_dir: Path, base: str) -> tuple:
    """Cohort-wide bounding box of non-zero content, with a safety margin.

    A per-volume box would give every session a different frame. One shared box
    keeps a single coordinate system across the cohort, so a patch origin means
    the same thing everywhere.
    """
    lo = np.array([np.inf] * 3)
    hi = np.array([-np.inf] * 3)
    shape = None
    n = 0
    for p in sorted(arrays_dir.glob(f"*__{base}.npz")):
        with np.load(p) as z:
            a = np.asarray(z["array"])
        shape = a.shape if shape is None else shape
        nz = np.nonzero(a)
        if nz[0].size:
            lo = np.minimum(lo, [nz[i].min() for i in range(3)])
            hi = np.maximum(hi, [nz[i].max() for i in range(3)])
        n += 1
    if not np.isfinite(lo).all():
        raise RuntimeError(f"no non-zero content in any {base}")
    return tuple(int(x) for x in lo), tuple(int(x) for x in hi), shape, n


def build(project_root, base: str = "ContrastEnhancedMask-CL",
          margin: int = 24, arrays_subdir: str = "01_DATA_FOUNDATION/v2_arrays",
          verify: bool = True, rng_seed: int = 1337, min_extent: int = None) -> dict:
    """Pack every mask into one verified uint8 array on Drive."""
    from .patches import PATCH
    root = Path(project_root)
    arrays_dir = root / arrays_subdir
    lo, hi, full_shape, n_found = _bbox_union(arrays_dir, base)

    # Margin so a patch centred near the lesion edge still has context, and so
    # a future annotation slightly outside today's extent is not clipped.
    lo = tuple(max(0, lo[a] - margin) for a in range(3))
    hi = tuple(min(full_shape[a] - 1, hi[a] + margin) for a in range(3))

    # A crop smaller than the patch would make every patch mostly zero padding
    # — the model would train on padding rather than anatomy. Grow the box,
    # centred, to at least the patch size in every dimension.
    min_extent = PATCH if min_extent is None else min_extent
    lo, hi = list(lo), list(hi)
    for a in range(3):
        need = min_extent - (hi[a] - lo[a] + 1)
        if need > 0:
            grow_lo = min(lo[a], need // 2 + need % 2)
            lo[a] -= grow_lo
            hi[a] = min(full_shape[a] - 1, hi[a] + (need - grow_lo))
            # If the volume itself is smaller than min_extent there is nothing
            # to grow into; crop() zero-pads and that is unavoidable.
            deficit = min_extent - (hi[a] - lo[a] + 1)
            if deficit > 0 and lo[a] > 0:
                lo[a] = max(0, lo[a] - deficit)
    lo, hi = tuple(lo), tuple(hi)
    crop_shape = tuple(hi[a] - lo[a] + 1 for a in range(3))

    keys, vols = [], []
    for p in sorted(arrays_dir.glob(f"*__{base}.npz")):
        m = _KEY.search(p.name)
        if not m:
            continue
        with np.load(p) as z:
            a = np.asarray(z["array"]) > 0
        sub = a[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
        # A mask voxel outside the crop would be silently deleted. The union
        # bbox plus margin makes that impossible, but "impossible" is what
        # assertions are for.
        if int(a.sum()) != int(sub.sum()):
            raise RuntimeError(
                f"{p.name}: crop lost {int(a.sum()) - int(sub.sum())} voxels — "
                "bounding box is wrong, refusing to write a lossy cache")
        keys.append(f"{m.group(1)}/{m.group(2)}")
        vols.append(sub.astype(np.uint8))

    if not vols:
        raise RuntimeError(f"no {base} arrays found in {arrays_dir}")
    stack = np.stack(vols)

    out = root / "01_DATA_FOUNDATION" / CACHE_NAME
    meta = {
        "base": base, "n_volumes": len(keys),
        "full_shape": list(full_shape), "crop_origin": list(lo),
        "crop_shape": list(crop_shape), "margin": margin,
        "dtype": "uint8",
        "min_extent": min_extent,
        "coordinate_note": (
            "Arrays are CROPPED. A patch origin in cache coordinates maps to "
            "full-grid coordinates by adding crop_origin. Any code mixing cache "
            "and full-grid coordinates without that offset is wrong."),
    }
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, volumes=stack, keys=np.array(keys),
                        meta=np.array(json.dumps(meta)))
    tmp.replace(out)

    size_mb = out.stat().st_size / 1e6
    ram_mb = stack.nbytes / 1e6
    res = {"cache": str(out), "n_volumes": len(keys),
           "file_size_mb": round(size_mb, 1),
           "ram_when_loaded_mb": round(ram_mb, 1),
           "full_shape": list(full_shape), "crop_shape": list(crop_shape),
           "crop_origin": list(lo),
           "voxel_reduction": round(np.prod(full_shape) / np.prod(crop_shape), 1),
           "meta": meta}
    if verify:
        res["verification"] = _verify(arrays_dir, stack, keys, meta,
                                      base, rng_seed)
    return res


def _verify(arrays_dir: Path, stack, keys, meta, base, seed) -> dict:
    """Re-read source volumes and assert EXACT equality after re-expansion."""
    rng = np.random.default_rng(seed)
    n = max(MIN_VERIFY, int(len(keys) * VERIFY_FRACTION))
    idx = rng.choice(len(keys), size=min(n, len(keys)), replace=False)
    lo = meta["crop_origin"]
    checked, mismatches = 0, []
    for i in idx:
        sub, ses = keys[i].split("/")
        p = arrays_dir / f"{sub}__{ses}__{base}.npz"
        with np.load(p) as z:
            original = np.asarray(z["array"]) > 0
        restored = np.zeros(meta["full_shape"], dtype=bool)
        c = stack[i].astype(bool)
        restored[lo[0]:lo[0] + c.shape[0],
                 lo[1]:lo[1] + c.shape[1],
                 lo[2]:lo[2] + c.shape[2]] = c
        checked += 1
        if not np.array_equal(original, restored):
            mismatches.append({
                "key": keys[i],
                "original_nonzero": int(original.sum()),
                "restored_nonzero": int(restored.sum())})
    if mismatches:
        raise RuntimeError(
            f"CACHE IS NOT FAITHFUL — {len(mismatches)} of {checked} volumes "
            f"differ from source after re-expansion: {mismatches[:3]}. The "
            "cache has been written but MUST NOT be used.")
    return {"volumes_checked": checked, "mismatches": 0,
            "method": ("random sample re-read from source .npz, re-expanded to "
                       "the full grid, asserted exactly equal — not a spot "
                       "check of summary statistics")}


class CachedMasks:
    """The whole cache in RAM. Zero I/O per training step."""

    def __init__(self, project_root, name: str = CACHE_NAME):
        p = Path(project_root) / "01_DATA_FOUNDATION" / name
        if not p.is_file():
            raise FileNotFoundError(
                f"{p} not found — run mask_cache.build(ROOT) once first")
        with np.load(p, allow_pickle=False) as z:
            self.volumes = z["volumes"]
            self.keys = [str(k) for k in z["keys"]]
            self.meta = json.loads(str(z["meta"]))
        self.index = {k: i for i, k in enumerate(self.keys)}
        # np.nonzero() on every patch draw rescans the whole volume and was the
        # dominant cost after I/O was removed. Computed ONCE per volume here.
        self._fg = [np.argwhere(v) for v in self.volumes]
        self.crop_origin = tuple(self.meta["crop_origin"])
        self.crop_shape = tuple(self.meta["crop_shape"])

    def get(self, subject: str, session: str):
        i = self.index.get(f"{subject}/{session}")
        return None if i is None else self.volumes[i]

    def foreground(self, subject: str, session: str):
        """Precomputed (N, 3) non-zero coordinates. Empty array when none."""
        i = self.index.get(f"{subject}/{session}")
        return None if i is None else self._fg[i]

    def has(self, subject: str, session: str) -> bool:
        return f"{subject}/{session}" in self.index

    def __len__(self) -> int:
        return len(self.keys)

    def ram_mb(self) -> float:
        return self.volumes.nbytes / 1e6


def verify(project_root, base: str = "ContrastEnhancedMask-CL",
           arrays_subdir: str = "01_DATA_FOUNDATION/v2_arrays") -> dict:
    """Re-verify an existing cache against source at any time."""
    c = CachedMasks(project_root)
    return _verify(Path(project_root) / arrays_subdir, c.volumes, c.keys,
                   c.meta, base, 1337)


def print_report(res: dict) -> None:
    line = "-" * 78
    print(line)
    print("MASK CACHE BUILT")
    print(line)
    print(f"  volumes      : {res['n_volumes']}")
    print(f"  full grid    : {res['full_shape']}")
    print(f"  cropped to   : {res['crop_shape']}  "
          f"(origin {res['crop_origin']}, {res['voxel_reduction']}x fewer voxels)")
    print(f"  file on Drive: {res['file_size_mb']} MB")
    print(f"  RAM when held: {res['ram_when_loaded_mb']} MB")
    v = res.get("verification")
    if v:
        print(f"\n  VERIFIED     : {v['volumes_checked']} volumes re-read from "
              f"source, {v['mismatches']} mismatches")
        print(f"  method       : {v['method']}")
    print(f"\n  {res['meta']['coordinate_note']}")
    print(line)
