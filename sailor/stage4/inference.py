"""Whole-volume prediction from a patch-trained model.

WHY THIS EXISTS. The model sees 96^3 patches; the task is a whole volume, and
the pre-registered GATE-3 metric (`log_volume_ratio_error`) is a volume-level
quantity. Validating on patch loss would measure something the project never
claims — patch loss can plateau while volume-level performance is still moving,
and it is insensitive to the systematic over- or under-prediction that
volume-ratio error exists to catch.

Sliding-window inference with overlap, averaging logits in the overlap regions.
Averaging LOGITS rather than probabilities keeps the combination linear in the
model's own output space; averaging probabilities would weight confident and
uncertain predictions differently depending only on where a tile boundary fell.

The threshold is fixed at 0.5 and is NOT tuned. An optimised threshold per fold
would be a free parameter fitted on the data being scored, which is the same
class of error as tuning on a test fold.
"""

from __future__ import annotations

import numpy as np

#: Fixed. A tuned threshold would be a parameter fitted on the scored data.
THRESHOLD = 0.5
#: Half-patch stride: every interior voxel is covered by multiple tiles, so a
#: prediction never depends on which tile happened to contain it.
STRIDE_FRACTION = 0.5


def _starts(extent: int, patch: int, stride: int) -> list:
    if extent <= patch:
        return [0]
    xs = list(range(0, extent - patch + 1, stride))
    if xs[-1] != extent - patch:
        xs.append(extent - patch)          # always cover the far edge
    return xs


def predict_volume(model, volume: np.ndarray, patch: int = 96,
                   batch_size: int = 8, device: str = "cuda",
                   cond=None, amp: bool = True) -> np.ndarray:
    """Sliding-window logits for one whole volume. Returns float32, same shape."""
    import torch
    stride = max(1, int(patch * STRIDE_FRACTION))
    shape = volume.shape
    acc = np.zeros(shape, dtype=np.float32)
    cnt = np.zeros(shape, dtype=np.float32)

    origins = [(z, y, x)
               for z in _starts(shape[0], patch, stride)
               for y in _starts(shape[1], patch, stride)
               for x in _starts(shape[2], patch, stride)]

    model.eval()
    with torch.no_grad():
        for i in range(0, len(origins), batch_size):
            chunk = origins[i:i + batch_size]
            tiles = np.stack([
                volume[o[0]:o[0] + patch, o[1]:o[1] + patch, o[2]:o[2] + patch]
                for o in chunk])[:, None]
            xb = torch.from_numpy(tiles).to(device, non_blocking=True).float()
            with torch.amp.autocast("cuda", enabled=(amp and device == "cuda")):
                if cond is not None:
                    c = torch.as_tensor(cond, dtype=torch.float32,
                                        device=device).repeat(len(chunk), 1)
                    out = model(xb, c)
                else:
                    out = model(xb)
            out = out.float().cpu().numpy()[:, 0]
            for j, o in enumerate(chunk):
                acc[o[0]:o[0] + patch, o[1]:o[1] + patch,
                    o[2]:o[2] + patch] += out[j]
                cnt[o[0]:o[0] + patch, o[1]:o[1] + patch,
                    o[2]:o[2] + patch] += 1.0
    return acc / np.maximum(cnt, 1.0)


def predict_mask(model, volume: np.ndarray, **kw) -> np.ndarray:
    """Binary prediction at the fixed threshold."""
    logits = predict_volume(model, volume, **kw)
    return (1.0 / (1.0 + np.exp(-logits))) > THRESHOLD


def evaluate_fold(model, cache, pairs: list, patch: int = 96,
                  batch_size: int = 8, device: str = "cuda",
                  amp: bool = True) -> dict:
    """Volume-level metrics on held-out pairs, against persistence on the same.

    Reporting the model WITHOUT persistence on the identical pairs would leave
    a number with nothing to compare it to: the val set is a handful of
    patients and its persistence score is not the cohort's 0.4928.
    """
    from ..stage3.persistence import (dice, log_volume_ratio_error,
                                      volume_change_error, change_region_dice)
    rows = []
    for p in pairs:
        a = cache.get(p["subject"], p["input_session"])
        b = cache.get(p["subject"], p["target_session"])
        if a is None or b is None:
            continue
        pred = predict_mask(model, a, patch=patch, batch_size=batch_size,
                            device=device, amp=amp)
        prev, ref = a.astype(bool), b.astype(bool)
        rows.append({
            "subject": p["subject"],
            "model_log_ratio": log_volume_ratio_error(pred, ref, prev),
            "pers_log_ratio": log_volume_ratio_error(prev, ref, prev),
            "model_dice": dice(pred, ref),
            "pers_dice": dice(prev, ref),
            "model_vol_err": volume_change_error(pred, ref, prev),
            "pers_vol_err": volume_change_error(prev, ref, prev),
            "model_change_dice": change_region_dice(pred, ref, prev),
            "n_pred": int(pred.sum()), "n_target": int(ref.sum()),
            "n_input": int(prev.sum()),
        })

    def agg(key):
        by = {}
        for r in rows:
            if r[key] is not None:
                by.setdefault(r["subject"], []).append(float(r[key]))
        if not by:
            return {"mean": None, "n_patients": 0, "n_defined": 0}
        per = [float(np.mean(v)) for v in by.values()]
        return {"mean": float(np.mean(per)), "n_patients": len(per),
                "n_defined": sum(len(v) for v in by.values())}

    return {
        "n_pairs": len(rows),
        "model": {k: agg(f"model_{k}") for k in
                  ("log_ratio", "dice", "vol_err", "change_dice")},
        "persistence": {k: agg(f"pers_{k}") for k in
                        ("log_ratio", "dice", "vol_err")},
        "beats_persistence_on_primary": (
            None if agg("model_log_ratio")["mean"] is None
            else agg("model_log_ratio")["mean"] < agg("pers_log_ratio")["mean"]),
        "threshold": THRESHOLD,
        "per_pair": rows,
    }
