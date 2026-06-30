"""
06_landing_site_traverse.py

Preliminary landing-site and rover-traverse methodology using the DFSAR incidence-angle
product (gri_in) as a terrain/illumination safety proxy, combined with confirmed
spatially-coherent ice candidates from 02_labeling.py.

Safe zone = low-incidence-angle terrain (flatter / better illuminated), explicitly
excluding the ice-candidate region itself (to preserve the scientific target rather
than land on it). Traverse = shortest path from the nearest safe pixel to the ice
candidate cluster centroid.

This is a first-order methodology for Phase 1, not a full hazard-avoidance path planner.
Boulder distribution / slope-from-DEM analysis (per the PS8 brief) is noted as future work.
"""

import numpy as np
import rasterio
from pathlib import Path
from scipy.ndimage import label as cc_label, distance_transform_edt
import matplotlib.pyplot as plt

PIXEL_SIZE_M = 7.5  # DFSAR GRD nominal ground sample distance — confirm against scene XML


def read_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def compute_ice_candidates(cpr_scene: np.ndarray, min_size: int = 4) -> np.ndarray:
    """Same connected-component coherence filter as 02_labeling.py, applied at
    full-scene resolution rather than per-patch."""
    raw_mask = cpr_scene > 1.0
    labeled, n = cc_label(raw_mask)
    candidates = np.zeros_like(raw_mask, dtype=np.float32)
    for cid in range(1, n + 1):
        comp = (labeled == cid)
        if comp.sum() >= min_size:
            candidates[comp] = 1.0
    return candidates


def propose_landing_site(incidence: np.ndarray, ice_candidates: np.ndarray,
                          safe_percentile: float = 30.0):
    """Returns (landing_site, target_centroid, traverse_distance_px, safe_zone_mask)."""
    inc_norm = (incidence - incidence.min()) / (incidence.max() - incidence.min() + 1e-8)
    safe_zone = (inc_norm < np.percentile(inc_norm, safe_percentile)) & (ice_candidates == 0)

    if ice_candidates.sum() == 0:
        return None, None, None, safe_zone

    ys, xs = np.where(ice_candidates > 0)
    target = (int(ys.mean()), int(xs.mean()))

    safe_ys, safe_xs = np.where(safe_zone)
    d = np.sqrt((safe_ys - target[0]) ** 2 + (safe_xs - target[1]) ** 2)
    nearest_idx = np.argmin(d)
    landing = (safe_ys[nearest_idx], safe_xs[nearest_idx])
    traverse_dist_px = d[nearest_idx]

    return landing, target, traverse_dist_px, safe_zone


def plot_landing_traverse(incidence, safe_zone, ice_candidates, landing, target,
                           traverse_dist_px, window=200,
                           out_path='outputs/PS8_landing_traverse.png'):
    """Crops a readable window around the landing site before plotting — DFSAR scenes
    are very tall/narrow strips (e.g. 13758 x 191 px) and plotting the full scene
    renders as an unreadable sliver."""
    r0, r1 = max(0, landing[0] - window), min(incidence.shape[0], landing[0] + window)
    c0, c1 = 0, incidence.shape[1]

    inc_crop  = incidence[r0:r1, c0:c1]
    safe_crop = safe_zone[r0:r1, c0:c1]
    ice_crop  = ice_candidates[r0:r1, c0:c1]
    landing_local = (landing[0] - r0, landing[1] - c0)
    target_local  = (target[0] - r0, target[1] - c0)
    traverse_dist_m = traverse_dist_px * PIXEL_SIZE_M

    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    fig.patch.set_facecolor('#0D1B2A')
    for ax in axes:
        ax.set_facecolor('#0D1B2A'); ax.tick_params(colors='#999', labelsize=8)

    axes[0].imshow(inc_crop, cmap='gray', aspect='auto')
    axes[0].set_title('Incidence angle (cropped, terrain proxy)', color='white', fontsize=10)

    axes[1].imshow(safe_crop, cmap='Greens', alpha=0.5, aspect='auto')
    axes[1].imshow(ice_crop, cmap='Blues', alpha=0.7, aspect='auto')
    axes[1].plot(landing_local[1], landing_local[0], 'r*', markersize=18, label='Proposed landing site')
    axes[1].plot([landing_local[1], target_local[1]], [landing_local[0], target_local[0]],
                 'y--', linewidth=2, label=f'Traverse ({traverse_dist_m:.0f} m)')
    axes[1].legend(facecolor='#0D1B2A', labelcolor='white', fontsize=8, loc='upper right')
    axes[1].set_title('Safe zone + ice candidates + traverse', color='white', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='#0D1B2A')
    print(f"Traverse distance: {traverse_dist_px:.1f} px = {traverse_dist_m:.0f} m")
    print(f"Saved: {out_path}")
    return traverse_dist_m


if __name__ == '__main__':
    import sys
    scene_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if scene_dir is None:
        raise SystemExit("Usage: python 06_landing_site_traverse.py <path_to_extracted_scene>")

    date_str = scene_dir.name.split('_')[3][:8]
    cal = scene_dir / 'data' / 'calibrated' / date_str

    INC = read_tif(sorted(cal.glob('*_gri_in_fp_xx_*.tif'))[0])
    HH = read_tif(sorted(cal.glob('*_gri_xx_fp_hh_*.tif'))[0])
    HV = read_tif(sorted(cal.glob('*_gri_xx_fp_hv_*.tif'))[0])
    VV = read_tif(sorted(cal.glob('*_gri_xx_fp_vv_*.tif'))[0])

    SC = (HH - VV) ** 2 / 4.0 + HV ** 2
    OC = (HH + VV) ** 2 / 4.0
    CPR_scene = np.clip(SC / (OC + 1e-10), 0, 10)

    ice_candidates = compute_ice_candidates(CPR_scene)
    print(f"Confirmed candidate clusters in this scene: {int(ice_candidates.sum())} px")

    landing, target, traverse_dist_px, safe_zone = propose_landing_site(INC, ice_candidates)
    if landing is None:
        print("No confirmed candidate cluster in this scene — traverse target unavailable.")
    else:
        print(f"Proposed landing site (row,col): {landing}")
        print(f"Target crater-floor centroid    : {target}")
        print(f"Incidence angle at landing site  : {INC[landing[0],landing[1]]:.2f} deg")
        print(f"Incidence angle at target        : {INC[target[0],target[1]]:.2f} deg")
        plot_landing_traverse(INC, safe_zone, ice_candidates, landing, target, traverse_dist_px)
