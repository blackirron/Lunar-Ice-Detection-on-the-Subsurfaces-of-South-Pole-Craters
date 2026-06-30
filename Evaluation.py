"""
05_evaluate.py

Evaluates the trained model and reports two SEPARATE findings rather than a single
"baseline vs model, look how much better we are" comparison:

  Finding 1: how often the literal ISRO criterion (CPR>1 AND DOP<0.13) fires on raw,
             unfiltered data — a real, standalone observation about this dataset.
  Finding 2: how faithfully the model reproduces the spatially-coherence-filtered proxy
             label it was trained on, including spatial denoising of speckle.

These are NOT compared against each other as "baseline vs model" — that comparison is
circular, since the proxy label is itself derived from the same CPR field. See
docs/FINDINGS.md for why this distinction matters.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


def np_iou(p, g):
    inter = (p * g).sum()
    union = p.sum() + g.sum() - inter
    return float(inter / (union + 1e-8))


def evaluate(model, X_test, Y_test, C_test, D_test, device,
             train_losses, val_losses, train_ious, val_ious, best_val_iou, best_epoch):
    model_ious, test_preds_ml = [], []
    model.eval()
    with torch.no_grad():
        for i in range(len(X_test)):
            xb = torch.tensor(X_test[i:i+1], dtype=torch.float32).to(device)
            prob = torch.sigmoid(model(xb)).cpu().numpy()[0, 0]
            mbin = (prob > 0.5).astype(np.float32)
            gt = Y_test[i, 0]
            model_ious.append(np_iou(mbin, gt))
            test_preds_ml.append(prob)

    mmi = np.mean(model_ious)
    abs_fraction = float(np.mean([(C_test[i] > 1.0) & (D_test[i] < 0.13)
                                   for i in range(len(C_test))]))

    print("=" * 65)
    print("FINDING 1 — Literal ISRO absolute criterion (CPR>1 AND DOP<0.13)")
    print("=" * 65)
    print(f"Fraction of pixels satisfying literal criterion : {100*abs_fraction:.4f}%")
    print("-> Absolute threshold rarely fires on real, speckle-affected DFSAR data.")
    print()
    print("=" * 65)
    print("FINDING 2 — U-Net trained on spatially-coherent CPR proxy label")
    print("=" * 65)
    print(f"Test IoU (model vs proxy label) : {mmi:.4f}")
    print(f"Best Val IoU                    : {best_val_iou:.4f}")
    print("-> Proxy label = connected-component-filtered CPR>1 anomalies.")
    print("-> U-Net reproduces the proxy with spatial denoising of speckle.")
    print("=" * 65)

    return model_ious, test_preds_ml, mmi, abs_fraction


def plot_training_curves(train_losses, val_losses, train_ious, val_ious,
                          best_epoch, mmi, abs_fraction, epochs,
                          out_path='outputs/PS8_ML_training.png'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0D1B2A')
    fig.suptitle("PS 8 — Transfer Learning | U-Net+ResNet-34 | Real DFSAR Data\n"
                  "Input: SC + filtered-CPR + filtered-DOP | Label: spatially-coherent CPR proxy",
                  fontsize=12, color='white', fontweight='bold', y=1.02)
    ep = list(range(1, epochs + 1))

    axes[0].plot(ep, train_losses, color='#00B4D8', lw=2.5, label='Train')
    axes[0].plot(ep, val_losses, color='#FFD700', lw=2.5, label='Val', linestyle='--')
    axes[0].axvline(x=best_epoch, color='#1D9E75', lw=1.5, linestyle=':', label=f'Best ({best_epoch})')
    axes[0].set_facecolor('#1A2A3A'); axes[0].set_title('Loss (Focal+Dice)', color='white', fontsize=11)
    axes[0].tick_params(colors='#999'); axes[0].legend(facecolor='#0D1B2A', labelcolor='white')
    axes[0].grid(True, alpha=0.15, color='white')

    axes[1].plot(ep, train_ious, color='#00B4D8', lw=2.5, label='Train IoU')
    axes[1].plot(ep, val_ious, color='#FFD700', lw=2.5, label='Val IoU', linestyle='--')
    axes[1].axhline(y=mmi, color='#1D9E75', lw=1.5, linestyle='-.', label=f'Test={mmi:.3f}')
    axes[1].set_facecolor('#1A2A3A'); axes[1].set_title('IoU Over Training', color='white', fontsize=11)
    axes[1].tick_params(colors='#999'); axes[1].legend(fontsize=8, facecolor='#0D1B2A', labelcolor='white')
    axes[1].grid(True, alpha=0.15, color='white')

    bars = axes[2].bar(['Literal ISRO\ncriterion\n(% pixels)', 'U-Net Test IoU\n(vs proxy label)'],
                        [100 * abs_fraction, 100 * mmi],
                        color=['#E74C3C', '#1D9E75'], width=0.4, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, [100 * abs_fraction, 100 * mmi]):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                      f'{val:.2f}%' if val < 5 else f'{val:.1f}%',
                      ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')
    axes[2].set_facecolor('#1A2A3A')
    axes[2].set_title('Two separate findings (not head-to-head)', color='white', fontsize=10)
    axes[2].set_ylim(0, 100); axes[2].grid(True, alpha=0.15, color='white', axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='#0D1B2A')
    print(f"Saved: {out_path}")


def plot_detection_comparison(X_test, Y_test, test_preds_ml, model_ious,
                               out_path='outputs/PS8_ML_comparison.png'):
    samples = np.argsort([Y_test[i, 0].sum() for i in range(len(X_test))])[::-1][:3]
    fig, axes = plt.subplots(3, 3, figsize=(16, 15))
    fig.patch.set_facecolor('#0D1B2A')
    fig.suptitle("PS 8 — ICE DETECTION: SAR Input -> Spatially-coherent Proxy Label -> CNN Output\n"
                  "Real Chandrayaan-2 DFSAR | South Polar Region | Doubly Shadowed Crater",
                  fontsize=12, color='white', fontweight='bold', y=0.99)

    ice_cm = mcolors.ListedColormap(['#1a1a2e', '#00b4d8'])
    for col, title in enumerate(["SAR Input (SC band)",
                                  "Proxy Label\n(spatially-coherent CPR anomaly)",
                                  "U-Net Output\n(spatially regularized)"]):
        axes[0, col].set_title(title, color='white', fontsize=10,
                                fontweight='bold' if col == 2 else 'normal', pad=6)

    for row, idx in enumerate(samples):
        sar = X_test[idx, 0]
        gt = Y_test[idx, 0]
        mp = test_preds_ml[idx]
        mb = (mp > 0.5).astype(np.float32)

        axes[row, 0].imshow(sar, cmap='gray', vmin=np.percentile(sar, 2), vmax=np.percentile(sar, 98))
        axes[row, 0].set_ylabel(f'Sample {row+1}', color='white', fontsize=10)
        axes[row, 1].imshow(gt, cmap=ice_cm, vmin=0, vmax=1)
        axes[row, 1].text(3, 118, f'Ice: {100*gt.mean():.2f}%', color='white', fontsize=8,
                           bbox=dict(facecolor='#0D1B2A', alpha=0.8, boxstyle='round'))
        axes[row, 2].imshow(mb, cmap=ice_cm, vmin=0, vmax=1)
        axes[row, 2].imshow(mp, cmap='Blues', alpha=0.3, vmin=0, vmax=1)
        axes[row, 2].text(3, 118, f'IoU: {model_ious[idx]:.3f}', color='#1D9E75', fontsize=9,
                           fontweight='bold', bbox=dict(facecolor='#0D1B2A', alpha=0.85, boxstyle='round'))

    for ax in axes.flat:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    fig.legend(handles=[
        mpatches.Patch(color='#00b4d8', label='Ice candidate (spatially coherent)'),
        mpatches.Patch(color='#1a1a2e', label='Rocky terrain'),
        Line2D([0], [0], color='#1D9E75', lw=3, label='U-Net prediction'),
    ], loc='lower center', ncol=3, fontsize=10, facecolor='#0D1B2A', labelcolor='white',
       framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='#0D1B2A')
    print(f"Saved: {out_path}")
