"""3D patch sampling for the conditioning ladder.

WHY PATCHES, AND WHY THIS IS NOT A COMPROMISE. The primary target occupies a
few thousand voxels in an 8,530,021-voxel grid — sub-25's lesion spans roughly
6x11 voxels in-plane across ~18 slices. Training on whole volumes spends
>99% of every step on background the metric never scores. Patch sampling around
the lesion keeps every voxel the target contains and discards the empty half of
the head. It is 3D training, not an approximation of it.

WHAT IS FIXED HERE AND WHY IT MATTERS (AMD-007). Patch size, the
foreground/background sampling ratio, and jitter are hyperparameters that
CHANGE PREDICTIONS, not just speed. A ratio favouring foreground biases a model
toward predicting tumour everywhere. They must be fixed before C0 and held
identical across C0-C4 and P1-P3, or the rungs differ in two ways at once and
the ladder measures nothing. They are module constants, not call arguments with
defaults that a later caller could quietly change.

LEAKAGE. Patches from one pair never split across folds — the unit of the split
is the PATIENT (§6, AMD-003), and patch sampling happens inside a fold after the
split is applied. Any caching of computed features must likewise be per-fold: a
feature cache built once over the whole cohort and reused across folds would
leak test-fold information into training, which is the exact failure G5 exists
to catch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- FIXED (AMD-007)
#: Patch edge in voxels. 96^3 covers the largest lesion in the cohort with
#: context; smaller would clip, larger wastes compute on background.
PATCH = 96
#: Probability a patch is CENTRED on target foreground rather than sampled
#: uniformly. 0.5 is deliberately neutral: it neither starves the model of
#: positives nor teaches it that tumour is everywhere.
#:
#: MEASURED, not assumed: this is the centring probability, NOT the fraction of
#: patches that contain foreground. A uniformly-sampled 96^3 patch from a
#: 193x229x193 volume frequently contains a centrally-located lesion by chance,
#: so the observed foreground-containing rate runs higher — ~75% at ratio 0.5 on
#: a central lesion. The name describes the draw, not the outcome, and the
#: distinction is recorded because this constant is fixed across all rungs.
FOREGROUND_RATIO = 0.5
#: Random offset applied to a foreground-centred patch, so the lesion does not
#: always sit at the patch centre and the model cannot learn that prior.
JITTER = 16
#: Sampling is seeded per (fold, epoch) so a resumed run reproduces exactly.
SAMPLING_SEED = 1337

CONFIG = {
    "patch": PATCH,
    "foreground_ratio": FOREGROUND_RATIO,
    "jitter": JITTER,
    "sampling_seed": SAMPLING_SEED,
    "fixed_by": "AMD-007 — identical across C0-C4 and P1-P3",
    "why_fixed": ("These change predictions, not only speed. A rung that "
                  "differed in patch size or foreground ratio would differ from "
                  "its neighbour in two ways at once."),
}


def load_pair_arrays(arrays_dir, pair: dict,
                     base: str = "ContrastEnhancedMask-CL") -> tuple | None:
    """(input_mask, target_mask) for one pair, or None if either is absent."""
    d = Path(arrays_dir)
    sub = pair["subject"]
    out = []
    for ses in (pair["input_session"], pair["target_session"]):
        p = d / f"{sub}__{ses}__{base}.npz"
        if not p.is_file():
            return None
        with np.load(p) as z:
            out.append((np.asarray(z["array"]) > 0).astype(np.float32))
    return tuple(out)


def sample_patch_origin(shape, foreground: np.ndarray, rng: np.random.Generator,
                        patch: int = PATCH, jitter: int = JITTER) -> tuple:
    """Top-left-front corner of one patch.

    Foreground-centred when the target is non-empty and the draw says so;
    otherwise uniform over the volume. An EMPTY target has no foreground to
    centre on, so those pairs always sample uniformly — which is correct, and
    is why the five empty->empty pairs contribute background statistics rather
    than being silently skipped.
    """
    half = patch // 2
    nz = np.nonzero(foreground)
    if nz[0].size and rng.random() < FOREGROUND_RATIO:
        i = rng.integers(nz[0].size)
        centre = [int(nz[a][i]) + int(rng.integers(-jitter, jitter + 1))
                  for a in range(3)]
    else:
        centre = [int(rng.integers(half, max(half + 1, shape[a] - half)))
                  for a in range(3)]
    return tuple(int(np.clip(centre[a] - half, 0, max(0, shape[a] - patch)))
                 for a in range(3))


def sample_origin_from_fg(shape, fg_coords, rng: np.random.Generator,
                          patch: int = PATCH, jitter: int = JITTER) -> tuple:
    """Same rule as `sample_patch_origin`, from PRECOMPUTED coordinates.

    Identical semantics — foreground centring with probability
    FOREGROUND_RATIO, otherwise uniform — but without rescanning the volume on
    every draw. An empty `fg_coords` (the five empty->empty pairs) always
    samples uniformly, which is correct rather than a skip.
    """
    half = patch // 2
    if fg_coords is not None and len(fg_coords) and rng.random() < FOREGROUND_RATIO:
        c = fg_coords[int(rng.integers(len(fg_coords)))]
        centre = [int(c[a]) + int(rng.integers(-jitter, jitter + 1))
                  for a in range(3)]
    else:
        centre = [int(rng.integers(half, max(half + 1, shape[a] - half)))
                  for a in range(3)]
    return tuple(int(np.clip(centre[a] - half, 0, max(0, shape[a] - patch)))
                 for a in range(3))


def crop(vol: np.ndarray, origin, patch: int = PATCH) -> np.ndarray:
    """Crop, zero-padding only when the patch runs past an edge.

    The in-bounds case returns a VIEW and allocates nothing. It is the common
    case — the cache is grown to at least `patch` in every dimension — and the
    allocate-and-copy path was measurable once I/O was removed. Callers must
    treat the result as read-only; `np.stack` in `batch()` copies it.
    """
    if all(origin[a] + patch <= vol.shape[a] for a in range(3)):
        return vol[origin[0]:origin[0] + patch,
                   origin[1]:origin[1] + patch,
                   origin[2]:origin[2] + patch]
    out = np.zeros((patch, patch, patch), dtype=vol.dtype)
    sl = tuple(slice(origin[a], min(origin[a] + patch, vol.shape[a]))
               for a in range(3))
    got = vol[sl]
    out[:got.shape[0], :got.shape[1], :got.shape[2]] = got
    return out


class CachedPairPatchSampler:
    """Patch sampler backed by the in-RAM mask cache. ZERO I/O per step.

    The Drive-backed `PairPatchSampler` measured 75-98% of step time in data
    loading on an L4 — 2.5 s of FUSE reads against 0.63 s of compute. This
    class holds every mask in memory as uint8 and does no disk access at all
    once constructed.

    Coordinates are CACHE coordinates throughout: the cache is cropped to the
    cohort bounding box, so a patch origin here is not a full-grid origin. That
    is consistent because every array in the cache shares the same frame, and
    nothing in training needs full-grid positions. Anything that later maps back
    to the full grid must add `cache.crop_origin`.
    """

    def __init__(self, cache, pairs: list, patch: int = PATCH,
                 seed: int = SAMPLING_SEED):
        self.cache = cache
        self.patch = patch
        self.seed = seed
        # Pairs whose ends are absent from the cache are dropped HERE, once,
        # with a count — not skipped silently inside the sampling loop where a
        # shrinking effective dataset would be invisible.
        self.pairs, self.dropped = [], []
        for p in pairs:
            if cache.has(p["subject"], p["input_session"]) and \
               cache.has(p["subject"], p["target_session"]):
                self.pairs.append(p)
            else:
                self.dropped.append(
                    f'{p["subject"]} {p["input_session"]}->{p["target_session"]}')
        if not self.pairs:
            raise RuntimeError("no pairs survive the cache — wrong cache or base")

    def batch(self, n: int, epoch: int = 0) -> tuple:
        rng = np.random.default_rng((self.seed, epoch, n))
        xs, ys = [], []
        for _ in range(n):
            p = self.pairs[int(rng.integers(len(self.pairs)))]
            a = self.cache.get(p["subject"], p["input_session"])
            b = self.cache.get(p["subject"], p["target_session"])
            # Foreground coordinates come precomputed from the cache; scanning
            # the volume per draw was the dominant cost once I/O was removed.
            origin = sample_origin_from_fg(
                a.shape, self.cache.foreground(p["subject"], p["target_session"]),
                rng, self.patch)
            xs.append(crop(a, origin, self.patch))
            ys.append(crop(b, origin, self.patch))
        # Returned as UINT8 on purpose. Converting to float32 here was 67% of
        # sampling time once I/O was removed, and it also quadruples the bytes
        # crossing PCIe. The training loop casts on-device after transfer,
        # where the cast is effectively free.
        return np.stack(xs)[:, None], np.stack(ys)[:, None]


class PairPatchSampler:
    """Yields (input_patch, target_patch) for pairs belonging to ONE fold.

    Construct with the training pairs of a single fold. Nothing here has access
    to the split, so it cannot cross fold boundaries by construction.
    """

    def __init__(self, arrays_dir, pairs: list, patch: int = PATCH,
                 seed: int = SAMPLING_SEED):
        self.arrays_dir = Path(arrays_dir)
        self.pairs = list(pairs)
        self.patch = patch
        self.seed = seed
        self._cache: dict = {}

    def _arrays(self, idx: int):
        if idx not in self._cache:
            self._cache[idx] = load_pair_arrays(self.arrays_dir, self.pairs[idx])
        return self._cache[idx]

    def batch(self, n: int, epoch: int = 0) -> tuple:
        rng = np.random.default_rng((self.seed, epoch, n))
        xs, ys = [], []
        tries = 0
        while len(xs) < n and tries < n * 20:
            tries += 1
            idx = int(rng.integers(len(self.pairs)))
            arrs = self._arrays(idx)
            if arrs is None:
                continue
            a, b = arrs
            origin = sample_patch_origin(a.shape, b, rng, self.patch)
            xs.append(crop(a, origin, self.patch))
            ys.append(crop(b, origin, self.patch))
        if not xs:
            raise RuntimeError("no usable pairs — check arrays_dir")
        return (np.stack(xs)[:, None], np.stack(ys)[:, None])
