"""
07_ice_volume_estimate.py

Order-of-magnitude ice volume estimate from CPR anomaly strength, using a simplified
linear dielectric-mixing scaling. This is EXPLICITLY ILLUSTRATIVE — the CPR-to-ice-fraction
relation used here is a placeholder, not a calibrated physical model. A defensible final
number would require either lab dielectric measurements or calibration against an
independent ground-truth ice estimate (e.g. Mini-RF cross-validated PSR studies).

Do not present this number as precise. State the caveat on any slide/report that uses it.
"""

import numpy as np

PIXEL_SIZE_M = 7.5     # confirm against scene XML metadata before treating as final
DEPTH_M = 5.0           # PS8-specified depth of interest (top 5 m of regolith)


def estimate_ice_volume(cpr_scene: np.ndarray, ice_candidates: np.ndarray,
                         pixel_size_m: float = PIXEL_SIZE_M, depth_m: float = DEPTH_M):
    """Returns a dict of intermediate and final quantities. Returns None values if
    no confirmed candidates exist in this scene (do not fabricate a number)."""
    if ice_candidates.sum() == 0:
        print("No confirmed candidates in this scene — volume estimate not computable. "
              "Report as 'pending denser anomaly detection across full crater mosaic.'")
        return None

    cpr_anomaly_vals = cpr_scene[ice_candidates > 0]
    mean_cpr_anomaly = float(cpr_anomaly_vals.mean())

    # Illustrative linear scaling (placeholder — replace with calibrated relation):
    # CPR ~1.0 -> ~5% ice fraction, CPR ~2.0 -> ~20% ice fraction
    ice_fraction_pct = float(np.clip((mean_cpr_anomaly - 1.0) * 15.0 + 5.0, 0, 100))

    n_pixels = int(ice_candidates.sum())
    area_m2 = n_pixels * (pixel_size_m ** 2)
    regolith_volume_m3 = area_m2 * depth_m
    ice_volume_m3 = regolith_volume_m3 * (ice_fraction_pct / 100.0)

    result = dict(
        mean_cpr_anomaly=mean_cpr_anomaly,
        ice_fraction_pct=ice_fraction_pct,
        n_candidate_pixels=n_pixels,
        area_m2=area_m2,
        regolith_volume_m3=regolith_volume_m3,
        ice_volume_m3=ice_volume_m3,
    )

    print(f"Mean CPR in candidate clusters : {mean_cpr_anomaly:.3f}")
    print(f"Estimated ice fraction         : {ice_fraction_pct:.1f}%  (illustrative scaling)")
    print(f"Candidate surface area         : {area_m2:.1f} m^2")
    print(f"Regolith volume (top {depth_m:.0f} m)     : {regolith_volume_m3:.1f} m^3")
    print(f"Estimated ice volume           : {ice_volume_m3:.2f} m^3")
    print("NOTE: illustrative dielectric scaling, requires lab/field calibration.")
    return result


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from importlib import import_module
    lst = import_module('06_landing_site_traverse')

    scene_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if scene_dir is None:
        raise SystemExit("Usage: python 07_ice_volume_estimate.py <path_to_extracted_scene>")

    date_str = scene_dir.name.split('_')[3][:8]
    cal = scene_dir / 'data' / 'calibrated' / date_str
    HH = lst.read_tif(sorted(cal.glob('*_gri_xx_fp_hh_*.tif'))[0])
    HV = lst.read_tif(sorted(cal.glob('*_gri_xx_fp_hv_*.tif'))[0])
    VV = lst.read_tif(sorted(cal.glob('*_gri_xx_fp_vv_*.tif'))[0])

    SC = (HH - VV) ** 2 / 4.0 + HV ** 2
    OC = (HH + VV) ** 2 / 4.0
    CPR_scene = np.clip(SC / (OC + 1e-10), 0, 10)

    ice_candidates = lst.compute_ice_candidates(CPR_scene)
    estimate_ice_volume(CPR_scene, ice_candidates)
