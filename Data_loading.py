"""
01_data_loading.py

Extracts Chandrayaan-2 DFSAR scene archives, derives CPR (Circular Polarization Ratio) and
DOP (Degree of Polarization) from quad-pol HH/HV/VH/VV bands, and tiles scenes into 128x128
patches for model training.

Key design decision: RAW (unfiltered) CPR/DOP are kept separate from Lee-FILTERED versions.
- RAW CPR/DOP -> used only for ground-truth label generation (preserves pixel-scale anomalies)
- FILTERED CPR/DOP -> used only as network input features (denoised, for stable training)
Conflating these two was the root cause of an early, incorrect "baseline = 0" result.
See docs/FINDINGS.md for the full diagnostic trace.
"""

import numpy as np
import rasterio
import zipfile
from pathlib import Path
from scipy.ndimage import uniform_filter

DATA_DIR    = Path('data/dfsar_data/')
EXTRACT_DIR = Path('data/dfsar_extracted/')
PATCH_SIZE, STRIDE = 128, 64


def lee_filter(img: np.ndarray, w: int = 7) -> np.ndarray:
    """Adaptive speckle-reduction filter. Used for INPUT FEATURES ONLY — never apply
    before computing the literal ISRO threshold label, as it suppresses the
    pixel-scale anomalies the criterion is designed to detect."""
    img = img.astype(np.float32)
    m   = uniform_filter(img,    size=w)
    m2  = uniform_filter(img**2, size=w)
    var = np.clip(m2 - m**2, 0, None)
    k   = var / (var + np.var(img) + 1e-10)
    return (m + k * (img - m)).astype(np.float32)


def read_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def load_scene(scene_root: Path):
    """Loads one DFSAR scene and returns both filtered (for model input) and
    raw (for label generation) CPR/DOP, plus the literal ISRO criterion mask."""
    scene_root = Path(scene_root)
    date_str   = scene_root.name.split('_')[3][:8]
    cal        = scene_root / 'data' / 'calibrated' / date_str
    print(f"\nLoading: {scene_root.name[-35:]}")

    HH = read_tif(sorted(cal.glob('*_gri_xx_fp_hh_*.tif'))[0])
    HV = read_tif(sorted(cal.glob('*_gri_xx_fp_hv_*.tif'))[0])
    VH = read_tif(sorted(cal.glob('*_gri_xx_fp_vh_*.tif'))[0])
    VV = read_tif(sorted(cal.glob('*_gri_xx_fp_vv_*.tif'))[0])
    print(f"  Shape: {HH.shape}")

    # --- RAW physical quantities (label generation only) ---
    SC_raw = (HH - VV) ** 2 / 4.0 + HV ** 2
    OC_raw = (HH + VV) ** 2 / 4.0
    CPR_raw = np.clip(SC_raw / (OC_raw + 1e-10), 0, 10)

    S0_raw = HH ** 2 + HV ** 2 + VH ** 2 + VV ** 2
    S1_raw = HH ** 2 - VV ** 2
    S2_raw = 2 * HH * VV
    S3_raw = 2 * HV * VH
    DOP_raw = np.clip(np.sqrt(S1_raw ** 2 + S2_raw ** 2 + S3_raw ** 2) / (S0_raw + 1e-10), 0, 1)

    ICE_literal = ((CPR_raw > 1.0) & (DOP_raw < 0.13)).astype(np.float32)
    print(f"  Literal ICE (raw CPR>1 & DOP<0.13): {100*ICE_literal.mean():.5f}% "
          f"({int(ICE_literal.sum())} px)")

    # --- FILTERED quantities (network input only) ---
    SC_f = lee_filter(SC_raw)
    OC_f = lee_filter(OC_raw)
    CPR_f = np.clip(SC_f / (OC_f + 1e-10), 0, 10)
    S0_f, S1_f, S2_f, S3_f = (lee_filter(S0_raw), lee_filter(S1_raw),
                               lee_filter(S2_raw), lee_filter(S3_raw))
    DOP_f = np.clip(np.sqrt(S1_f ** 2 + S2_f ** 2 + S3_f ** 2) / (S0_f + 1e-10), 0, 1)

    return SC_f, CPR_f, DOP_f, CPR_raw, DOP_raw, ICE_literal


def tile(SC_f, CPR_f, DOP_f, CPR_raw, DOP_raw, ICE_literal):
    """Tiles a scene into PATCH_SIZE x PATCH_SIZE patches with STRIDE overlap.
    Network-input channels are z-normalized per patch; raw CPR/DOP are kept
    un-normalized for downstream label generation."""
    patches, c_raw_p, d_raw_p, ice_p = [], [], [], []
    H, W = SC_f.shape
    for y in range(0, H - PATCH_SIZE, STRIDE):
        for x in range(0, W - PATCH_SIZE, STRIDE):
            s  = SC_f[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            cf = CPR_f[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            df = DOP_f[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            cr = CPR_raw[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            dr = DOP_raw[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            ic = ICE_literal[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            if s.max() == 0 or np.isnan(s).any():
                continue
            sn  = (s - s.mean()) / (s.std() + 1e-8)
            cfn = (cf - cf.mean()) / (cf.std() + 1e-8)
            dfn = (df - df.mean()) / (df.std() + 1e-8)
            patches.append(np.stack([sn, cfn, dfn], 0).astype(np.float32))
            c_raw_p.append(cr); d_raw_p.append(dr); ice_p.append(ic)
    return patches, c_raw_p, d_raw_p, ice_p


def build_dataset():
    """Extracts all scene zips and assembles the full patch dataset."""
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print("Extracting...")
    for zf in sorted(DATA_DIR.glob('*.zip')):
        out = EXTRACT_DIR / zf.stem
        if not out.exists():
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall(out)

    all_p, all_craw, all_draw, all_ice = [], [], [], []
    for sd in sorted(EXTRACT_DIR.iterdir()):
        if not sd.is_dir():
            continue
        try:
            SC_f, CPR_f, DOP_f, CPR_raw, DOP_raw, ICE_literal = load_scene(sd)
            p, cr, dr, ic = tile(SC_f, CPR_f, DOP_f, CPR_raw, DOP_raw, ICE_literal)
            all_p += p; all_craw += cr; all_draw += dr; all_ice += ic
        except Exception as e:
            print(f"  SKIP {sd.name[-25:]}: {e}")

    X     = np.array(all_p)
    C_raw = np.array(all_craw)
    D_raw = np.array(all_draw)
    ICE   = np.array(all_ice)

    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC — literal criterion across ALL {len(X)} patches")
    print(f"{'='*60}")
    print(f"Total positive pixels : {int(ICE.sum())}")
    print(f"Overall prevalence    : {100*ICE.mean():.5f}%")
    print(f"{'='*60}")

    return X, C_raw, D_raw, ICE


if __name__ == '__main__':
    X, C_raw, D_raw, ICE = build_dataset()
    np.savez_compressed('data/ps8_patches.npz', X=X, C_raw=C_raw, D_raw=D_raw, ICE=ICE)
    print("Saved data/ps8_patches.npz")
