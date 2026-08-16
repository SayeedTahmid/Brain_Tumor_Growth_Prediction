# =============================================================================
# MASK CACHE + RE-BENCHMARK
#
# MEASURED PROBLEM (L4, 208 pairs): data loading was 75-98% of every training
# step — ~2.5 s of Drive I/O against ~0.63 s of compute. The GPU sat idle.
#
# FIX, PURE I/O, NO EFFECT ON RESULTS:
#   uint8 instead of float32 (32x)     ... a binary mask needs 1 bit, not 32
#   crop to cohort bounding box        ... the grid is mostly empty
#   ONE file instead of 240            ... Drive per-file latency dominates
#   held in RAM, cast on GPU           ... zero I/O per step, 4x less PCIe
#
# The cache lives on DRIVE (durable across disconnects) and is read once per
# session into RAM. Local disk is deliberately not used: at this size it buys
# no throughput and adds a way for a stale copy to diverge.
# =============================================================================

# --- CELL A — build the cache ONCE (minutes; re-run only if arrays change) ---
from sailor.stage4 import mask_cache as MC

res = MC.build(ROOT)          # raises if the cache is not faithful to source
MC.print_report(res)


# --- CELL B — load into RAM and re-benchmark --------------------------------
import json, os
from sailor.stage4.patches import CachedPairPatchSampler, CONFIG
from sailor.stage4.model import ResidualUNet3D, param_count
from sailor.stage4 import benchmark as B

cache = MC.CachedMasks(ROOT)
print(f'cache: {len(cache)} volumes, {cache.ram_mb():.0f} MB RAM, '
      f'crop {cache.crop_shape}')

split = json.load(open(os.path.join(ROOT, '01_DATA_FOUNDATION',
                                    'v2_pairs_and_folds.json')))
sampler = CachedPairPatchSampler(cache, split['pairs']['pairs'])
print(f'pairs usable: {len(sampler.pairs)}  dropped: {len(sampler.dropped)}')
if sampler.dropped:
    print('  dropped:', sampler.dropped)

for bs in (8, 16, 24, 32):
    try:
        r = B.benchmark(ResidualUNet3D(), sampler, batch_size=bs,
                        steps=20, warmup=5, amp=True)
        print(f'  bs={bs:>2}  {r["sec_per_step_total"]:.3f} s/step  '
              f'{r["samples_per_sec"]:.1f} samp/s  '
              f'VRAM {r["peak_vram_gb"]:.1f} GB  data {r["data_fraction"]:.0%}')
    except RuntimeError as e:
        print(f'  bs={bs:>2}  failed: {str(e)[:70]}')
        break


# --- CELL C — official benchmark and 24 h budget ----------------------------
BATCH = 16            # set from the sweep
STEPS_PER_FIT = 2000  # planned optimisation steps per fold-fit

res = B.run(ROOT, ResidualUNet3D(), sampler,
            steps_per_fit=STEPS_PER_FIT, batch_size=BATCH,
            steps=80, amp=True)
