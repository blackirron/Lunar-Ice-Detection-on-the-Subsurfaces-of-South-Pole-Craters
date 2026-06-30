"""
02_labeling.py

Generates the ground-truth label used for training. Critically, this is NOT the raw
pixel-wise ISRO criterion (CPR>1 AND DOP<0.13) applied directly — diagnostics showed
that >99.7% of raw CPR>1 hits are isolated single-pixel speckle, not real spatial
anomalies (see docs/FINDINGS.md, section 3-4). This module applies a connected-component
spatial coherence filter to separate signal from noise before it becomes a training label.

DOP is NOT used as a hard AND-gate in the final label: diagnostics showed CPR>1 and
DOP<0.13 have near-zero co-occurrence (0 observed vs ~43 expected under independence),
consistent with DOP being unreliable on this detected (non-SLC) product. CPR is treated
as the primary, more trustworthy indicator for this dataset.
"""

import numpy as np
from scipy.ndimage import label as cc_label

MIN_CLUSTER_SIZE = 4  # minimum connected pixels to count as a real anomaly, not speckle


def spatial_coherent_label(cpr_patch: np.ndarray, min_size: int = MIN_CLUSTER_SIZE) -> np.ndarray:
    """Keeps only CPR>1 pixels that belong to a connected component of >= min_size pixels.
    Removes isolated speckle hits that fail the spatial coherence test."""
    raw_mask = (cpr_patch > 1.0)
    labeled, n = cc_label(raw_mask)
    clean = np.zeros_like(raw_mask, dtype=np.float32)
    for comp_id in range(1, n + 1):
        comp_mask = (labeled == comp_id)
        if comp_mask.sum() >= min_size:
            clean[comp_mask] = 1.0
    return clean


def build_labels(C_raw: np.ndarray):
    """Builds the final spatially-coherent label set and reports diagnostics."""
    Y_list, isolated_removed, clusters_kept = [], 0, 0
    for k in range(len(C_raw)):
        raw_mask = (C_raw[k] > 1.0)
        clean = spatial_coherent_label(C_raw[k])
        isolated_removed += int(raw_mask.sum() - clean.sum())
        if clean.sum() > 0:
            clusters_kept += 1
        Y_list.append(clean)
    Y = np.array(Y_list)[:, np.newaxis, :, :]

    print(f"{'='*60}")
    print("SPATIAL COHERENCE FILTER — separating noise from real anomalies")
    print(f"{'='*60}")
    print(f"Raw CPR>1 pixels (pre-filter)      : {int((C_raw > 1.0).sum())}")
    print(f"Removed as isolated speckle (<{MIN_CLUSTER_SIZE}px) : {isolated_removed}")
    print(f"Remaining coherent-cluster pixels  : {int(Y.sum())}")
    print(f"Patches with >=1 coherent cluster  : {clusters_kept} / {len(C_raw)}")
    print(f"Final label prevalence             : {100*Y.mean():.5f}%")
    print(f"{'='*60}")
    return Y


def split_dataset(X, Y, C_raw, train_frac=0.80, val_frac=0.10, seed=42):
    """80/10/10 train/val/test split. C_raw (RAW, unfiltered) is carried through
    the split alongside X/Y so the evaluation stage can recompute the literal
    criterion fraction on the correct, matching test indices."""
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    X, Y, C_raw = X[idx], Y[idx], C_raw[idx]

    nt = int(train_frac * len(X))
    nv = int(val_frac * len(X))

    X_train, Y_train = X[:nt],        Y[:nt]
    X_val,   Y_val   = X[nt:nt+nv],   Y[nt:nt+nv]
    X_test,  Y_test  = X[nt+nv:],     Y[nt+nv:]
    C_test = C_raw[nt+nv:]

    print(f"Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)} "
          f"| Train prevalence: {100*Y_train.mean():.4f}%")
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), C_test


if __name__ == '__main__':
    data = np.load('data/ps8_patches.npz')
    X, C_raw = data['X'], data['C_raw']
    Y = build_labels(C_raw)
    train, val, test, C_test = split_dataset(X, Y, C_raw)
    np.savez_compressed(
        'data/ps8_split.npz',
        X_train=train[0], Y_train=train[1],
        X_val=val[0], Y_val=val[1],
        X_test=test[0], Y_test=test[1],
        C_test=C_test,
    )
    print("Saved data/ps8_split.npz")
