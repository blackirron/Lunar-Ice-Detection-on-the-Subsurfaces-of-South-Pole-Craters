"""
04_train.py

Training loop using focal loss (handles the extreme class imbalance from the sparse,
spatially-coherent ice label — ~24 positive pixels out of ~31M) combined with Dice loss
for region-overlap quality. Plain BCE was found unsuitable given label prevalence below
0.001%, since it would trivially minimize loss by predicting all-negative.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

EPOCHS = 40


class SARDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def focal_loss(pred, target, alpha=0.85, gamma=2.0):
    """alpha up-weights the rare positive class; gamma down-weights easy negatives
    so the loss isn't dominated by the overwhelming majority of trivial background px."""
    pred_sig = torch.sigmoid(pred)
    pt = torch.where(target == 1, pred_sig, 1 - pred_sig)
    alpha_t = torch.where(target == 1, alpha, 1 - alpha)
    loss = -alpha_t * (1 - pt) ** gamma * torch.log(pt.clamp(min=1e-7))
    return loss.mean()


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    p, t = pred.view(-1), target.view(-1)
    return 1 - (2 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)


def combined_loss(pred, target):
    return 0.6 * focal_loss(pred, target) + 0.4 * dice_loss(pred, target)


def iou_score(pred, target, th=0.5):
    pb = (torch.sigmoid(pred) > th).float()
    inter = (pb * target).sum()
    return (inter / (pb.sum() + target.sum() - inter + 1e-8)).item()


def train(model, X_train, Y_train, X_val, Y_val, device):
    train_loader = DataLoader(SARDataset(X_train, Y_train), batch_size=16, shuffle=True, num_workers=2)
    val_loader   = DataLoader(SARDataset(X_val, Y_val),     batch_size=16, shuffle=False, num_workers=2)

    optimizer = optim.Adam([
        {'params': [p for n, p in model.named_parameters() if 'encoder' in n and p.requires_grad], 'lr': 1e-4},
        {'params': [p for n, p in model.named_parameters() if 'encoder' not in n], 'lr': 3e-4},
    ], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    train_losses, val_losses, train_ious, val_ious = [], [], [], []
    best_val_iou, best_epoch = 0.0, 0

    print(f"Training for {EPOCHS} epochs...")
    print("-" * 60)
    for epoch in range(EPOCHS):
        model.train()
        el, ei = 0.0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = combined_loss(pred, yb)
            loss.backward()
            optimizer.step()
            el += loss.item(); ei += iou_score(pred, yb)
        train_losses.append(el / len(train_loader))
        train_ious.append(ei / len(train_loader))

        model.eval(); vl, vi = 0.0, 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vl += combined_loss(pred, yb).item()
                vi += iou_score(pred, yb)
        val_losses.append(vl / len(val_loader))
        val_ious.append(vi / len(val_loader))
        scheduler.step()

        if val_ious[-1] > best_val_iou:
            best_val_iou, best_epoch = val_ious[-1], epoch + 1
            torch.save(model.state_dict(), 'best_model.pth')

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {train_losses[-1]:.4f} | "
                  f"Val Loss: {val_losses[-1]:.4f} | Val IoU: {val_ious[-1]:.4f}")

    print("-" * 60)
    print(f"Best Val IoU: {best_val_iou:.4f} at epoch {best_epoch}")
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    print("Best weights loaded.")

    history = dict(train_losses=train_losses, val_losses=val_losses,
                    train_ious=train_ious, val_ious=val_ious,
                    best_val_iou=best_val_iou, best_epoch=best_epoch)
    return model, history


if __name__ == '__main__':
    from importlib import import_module
    model_mod = import_module('03_model')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load('data/ps8_split.npz')
    model = model_mod.build_model(device)
    model, history = train(model, data['X_train'], data['Y_train'],
                            data['X_val'], data['Y_val'], device)
    np.savez('data/ps8_history.npz', **history)
