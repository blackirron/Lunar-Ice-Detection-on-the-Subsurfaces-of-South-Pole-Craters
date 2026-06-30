# Lunar-Ice-Detection-Subsurfaces-of-South-Pole-Craters

**Bharatiya Antariksh Hackathon — Problem Statement 8**
Detection and Characterization of Subsurface Ice in Lunar South Polar Regions Using Chandrayaan-2 Radar and Imagery Data

## Overview

This repo implements a radar-polarimetric ice detection pipeline using real Chandrayaan-2 DFSAR
(Dual Frequency Synthetic Aperture Radar) data over a doubly shadowed crater in the lunar south
polar region, along with a preliminary landing-site and rover-traverse methodology.

The defining feature of this project is that it does not present a circular or inflated result.
Initial experiments produced a misleadingly high IoU (~0.90) because the model was evaluated
against a label algebraically derived from its own input features. That issue was diagnosed,
traced to root cause, and corrected — the process and findings are documented in
[`docs/FINDINGS.md`](docs/FINDINGS.md). The final pipeline reports what the data actually
supports, not what looks best on a slide.

## Key finding (short version)

The literal ISRO criterion (`CPR > 1 AND DOP < 0.13`) applied pixel-wise to the supplied
detected (`gri`, phase-discarded) DFSAR product is dominated by speckle noise. Of 9,818 raw
pixels satisfying `CPR > 1` across 5 scenes (~31M pixels), only 24 pixels survive a spatial
coherence filter (connected components ≥ 4px) — meaning >99.7% of naive threshold hits are
noise, not real anomalies. This is a genuine, citable limitation of detected-magnitude SAR
products (no phase ⇒ no rigorous Stokes-parameter DOP), not a bug. Full derivation in
`docs/FINDINGS.md`.

## Repo structure

```
ps8-repo/
├── README.md
├── requirements.txt
├── docs/
│   └── FINDINGS.md              # full diagnostic writeup — read this for the real story
└── src/
    ├── 01_data_loading.py       # DFSAR extraction, CPR/DOP/Stokes derivation, Lee filtering, tiling
    ├── 02_labeling.py           # label diagnostics + spatial-coherence-filtered ground truth
    ├── 03_model.py              # U-Net + ResNet-34 (transfer learning) definition
    ├── 04_train.py              # training loop, focal+dice loss for class imbalance
    ├── 05_evaluate.py           # honest evaluation — no circular baseline comparison
    ├── 06_landing_site_traverse.py  # incidence-angle safety map + traverse path
    └── 07_ice_volume_estimate.py    # dielectric-mixing volume estimate (illustrative)
```

## Pipeline order

1. `01_data_loading.py` — extract DFSAR zips, compute CPR/DOP from quad-pol bands, tile into
   128×128 patches. Keeps **raw** (unfiltered) CPR/DOP separate from **Lee-filtered** versions —
   this separation is the single most important design decision in the project (see Findings).
2. `02_labeling.py` — runs the literal ISRO criterion, diagnoses its sparsity, applies a
   connected-component spatial coherence filter to separate real anomalies from speckle.
3. `03_model.py` / `04_train.py` — U-Net with ResNet-34 ImageNet-pretrained encoder, partial
   encoder freezing, focal + Dice loss to handle extreme class imbalance.
4. `05_evaluate.py` — reports the literal criterion's pixel fraction and the model's fidelity to
   the spatially-filtered label **as two separate findings**, not as a baseline-vs-model contest.
5. `06_landing_site_traverse.py` — incidence-angle-based terrain safety proxy, safe-zone
   computation, shortest-path traverse to nearest confirmed ice candidate cluster.
6. `07_ice_volume_estimate.py` — order-of-magnitude ice volume from CPR anomaly strength via a
   simplified dielectric-mixing relation (explicitly flagged as illustrative, pending calibration).

## Setup

```bash
pip install -r requirements.txt
```

Requires Chandrayaan-2 DFSAR data (HH/HV/VH/VV `gri` calibrated products + incidence angle
product) sourced from ISRO's PRADAN portal, placed under `data/dfsar_data/` as zip archives.

## Data source

Chandrayaan-2 DFSAR Level-2 calibrated products, South Polar region, via ISRO PRADAN
(https://pradan.issdc.gov.in/). Only detected (`gri`) magnitude products were available for
this scene — no complex/SLC product was supplied, which is the root cause of the DOP limitation
documented in Findings.

## Status

Phase 1 (hackathon) submission. Ice detection results are scoped to what the supplied detected
product can rigorously support (CPR-based spatial anomaly detection); a full per-pixel DOP-based
criterion would require SLC/complex PRADAN products, noted as future work.
