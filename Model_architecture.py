"""
03_model.py

U-Net with a ResNet-34 encoder, ImageNet-pretrained, adapted for 3-channel SAR input
(SC backscatter + filtered CPR + filtered DOP). Early encoder layers are frozen to
retain general low-level feature extractors learned from natural images, while later
layers and the full decoder remain trainable for the SAR-specific task.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_model(device: torch.device) -> nn.Module:
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,       # SC + filtered CPR + filtered DOP
        classes=1,
        activation=None,     # raw logits — sigmoid applied in loss/inference
    ).to(device)

    # Freeze early encoder layers (general low-level features); keep deeper
    # layers + decoder trainable for SAR-specific adaptation.
    for name, param in model.named_parameters():
        if any(f"encoder.layer{i}" in name for i in [1, 2, 3]):
            param.requires_grad = False

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model ready | Total: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M")
    print("Input: 3 channels (SC + filtered-CPR + filtered-DOP)")
    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = build_model(device)
