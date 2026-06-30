# Findings & Diagnostic Log

This document records the actual investigation that shaped the final pipeline — including the
dead ends — because the process is itself evidence of methodological rigor.

## 1. Initial result (rejected)

First pass reported: `ISRO Baseline IoU = 0.0000`, `U-Net IoU = 0.9038`, presented as a
+0.9038 "improvement." This was wrong for two independent reasons:

- **Circularity**: the training label (`Y_test`) was, at various points, either a per-patch
  top-20th-percentile CPR mask or a relative CPR/DOP rank — not the literal ISRO criterion the
  baseline was being scored against. Comparing a model trained on label A against a "baseline"
  using definition B, then calling the gap an "improvement," is not a valid comparison.
- **Baseline computation bug**: in one iteration, `C_test`/`D_test` were normalized/scaled
  copies of CPR/DOP (range ~[0, 2.6] and [0.66, 1.0] respectively), so `CPR > 1.0` and
  `DOP < 0.13` were near-impossible to satisfy by construction, regardless of real ice content.

## 2. Root-cause trace on raw data

Diagnostics were run directly on raw, un-normalized CPR/DOP computed straight from HH/HV/VH/VV:

| Check | Result |
|---|---|
| HH/VV/HV/VH raw stats | Non-negative integers (DN-like), median ~400, max ~11,300. Not dB-encoded. |
| CPR_raw percentiles (1 scene) | median 0.127, 95th 0.35, 99.9th 0.81, max 2.70 |
| DOP_raw percentiles (5 scenes, pooled) | median 0.887, 1st pct 0.737, **0th pct 0.0** |
| Fraction CPR_raw > 1.0 | 0.0313% (9,818 / ~31M px) |
| Fraction DOP_raw < 0.13 | 0.4340% (136,320 / ~31M px) |
| Fraction BOTH (literal criterion) | **0.0000%** (0 px), vs ~42.6 px expected if independent |

Both CPR and DOP individually behave in a physically plausible range (sparse high-CPR tail,
mostly-high DOP with a real low tail) — but their joint satisfaction is essentially zero, far
below the ~43 px expected by chance under independence.

## 3. Spatial visualization

Plotting `CPR_raw` with `CPR>1` pixels contoured showed isolated single-pixel hits scattered
randomly across the patch — no spatial clustering, no correlation with any visible terrain
feature. This is the signature of residual speckle noise, not a geophysical signal: a genuine
ice anomaly should appear as a spatially coherent patch tracing real terrain structure.

## 4. Spatial coherence filter (the actual fix)

Applied connected-component labeling to the raw `CPR>1` mask, keeping only components ≥4
contiguous pixels:

| Quantity | Value |
|---|---|
| Raw CPR>1 pixels (pre-filter) | 9,818 |
| Removed as isolated speckle (<4px clusters) | 9,794 (99.76%) |
| Remaining spatially-coherent pixels | 24 |
| Patches with ≥1 coherent cluster | 6 / 1,917 |

**Interpretation**: >99.7% of naive pixel-wise threshold hits on this detected (non-SLC,
phase-discarded) DFSAR product are speckle artifacts, not real anomalies. This is the central,
defensible finding of the project. It is also why per-pixel segmentation (24 positive pixels
total) is not a statistically supportable training target — see Limitations below.

## 5. Why DOP likely can't be computed rigorously from this product

`DOP = sqrt(S1² + S2² + S3²) / S0` depends on Stokes parameters that require coherent
(phase-preserving) complex scattering data to compute correctly. The only products available
for this scene (`gri`, `sli`, `sri` — all magnitude/detected rasters; confirmed via full
directory listing) discard phase. No covariance (`C3`), coherency (`T3`), or SLC product was
present. Computing "DOP" from detected magnitudes alone is an approximation of unknown
reliability — consistent with DOP_raw sitting mostly near 1.0 (low apparent depolarization)
almost everywhere, which is not physically expected for varied lunar terrain.

## 6. Final, defensible scope

- **Detection**: CPR-based spatial anomaly detection with connected-component coherence
  filtering, evaluated honestly (literal criterion fraction reported as a finding, not a
  baseline to inflate against).
- **DOP**: flagged explicitly as unreliable without SLC/complex data; not used as a hard gate.
- **Landing site / traverse**: built from the incidence-angle (`gri_in`) product, a real,
  available terrain/illumination proxy — combined with the (sparse) confirmed ice candidates.
- **Ice volume**: explicitly labeled illustrative, using a placeholder linear CPR-to-ice-fraction
  scaling pending lab/field calibration. Not presented as a precise measurement.

## Limitations (stated explicitly, not hidden)

- Confirmed spatially-coherent ice candidates in the supplied scene are extremely sparse (24 px
  / 6 patches out of 1,917). This is too few examples for reliable per-pixel CNN segmentation;
  results should be read as patch-level anomaly screening, not pixel-accurate mapping.
- DOP could not be rigorously validated due to the absence of phase-preserving (SLC/complex)
  PRADAN products for this scene.
- Ice volume estimate uses an illustrative, uncalibrated dielectric scaling relation.
- Pixel ground sample distance (7.5 m, used in volume/area calculations) should be confirmed
  against the scene's accompanying XML metadata before being treated as final.

## Phase 2 plan

- Acquire SLC/complex PRADAN products to compute DOP/Stokes parameters rigorously.
- Reframe as patch-level (not pixel-level) anomaly classification, with oversampling/few-shot
  techniques suited to ~6 confirmed positive patches.
- Extend landing-site/traverse methodology with OHRC-derived slope and boulder-distribution data.
- Calibrate the dielectric ice-fraction relation against literature (e.g., Mini-RF PSR studies)
  or lab measurements rather than an illustrative linear scaling.
