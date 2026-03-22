"""
CIFAR-10 Image Classification with CNN and MLP (Baseline Models)
=================================================================

This script trains two baseline deep learning models on the CIFAR-10 dataset:
  - SimpleCNN: A convolutional neural network (recommended)
  - SimpleMLP: A multi-layer perceptron (fully-connected, for comparison)

CIFAR-10 Dataset
----------------
  - 60,000 colour images of size 32×32 pixels across 10 classes
  - 50,000 training images / 10,000 test images
  - Classes: airplane, automobile, bird, cat, deer,
             dog, frog, horse, ship, truck

Expected Accuracy Ceilings for These Baseline Models
------------------------------------------------------
The accuracy you can realistically expect depends on which model you use and
whether you apply any data augmentation or regularisation.

  Model       | Augmentation | Typical Test Accuracy
  ------------|--------------|----------------------
  SimpleMLP   | None         | ~45 – 55 %
  SimpleCNN   | None         | ~65 – 75 %
  SimpleCNN   | Basic *      | ~72 – 80 %

  * Basic augmentation = random horizontal flip + random crop.

These figures assume:
  - The simple architectures defined below (2 conv blocks / 2 FC layers).
  - Standard SGD or Adam optimiser, ~20–30 training epochs.
  - No batch normalisation, dropout, or learning-rate scheduling.

Factors That Can Raise Accuracy
---------------------------------
  1. Data augmentation (flips, crops, colour jitter) → +3 – 8 pp
  2. Batch normalisation after conv layers              → +2 – 5 pp
  3. Dropout for regularisation                        → +1 – 3 pp
  4. Deeper / wider architectures (VGG, ResNet)        → up to ~93 %
  5. Learning-rate scheduling (step decay, cosine)     → +1 – 3 pp
  6. Training for more epochs (50–200)                 → +2 – 5 pp
  7. Pre-trained weights (transfer learning)           → up to ~98 %

State-of-the-Art Reference
---------------------------
  As of 2024, the best published results on CIFAR-10 exceed 99 % accuracy,
  achieved with large models (Vision Transformers, EfficientNet, etc.),
  heavy augmentation, and regularisation techniques.
  The baseline SimpleCNN here is intentionally simple for learning purposes.

Usage
-----
  python cifar10_train.py --model cnn --epochs 20
  python cifar10_train.py --model mlp --epochs 20
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """A minimal convolutional neural network for CIFAR-10.

    Architecture
    ------------
    Input: 3 × 32 × 32 colour image
      Conv(3→32, 3×3) → ReLU → MaxPool(2×2)    # 32 × 16 × 16
      Conv(32→64, 3×3) → ReLU → MaxPool(2×2)   # 64 ×  8 ×  8
      Flatten
      FC(64*7*7 → 512) → ReLU
      FC(512 → 10)                               # 10 class scores

    Expected accuracy (no data augmentation, ~20 epochs)
    -----------------------------------------------------
      - Typical test accuracy: **65 – 75 %**
      - Hard ceiling for this architecture without augmentation: ~75 %
      - With basic augmentation (flip + crop): ~72 – 80 %
      - Accuracy is limited mainly by:
          * Shallow depth (only 2 conv blocks)
          * Small number of filters
          * No batch normalisation or dropout
          * No learning-rate scheduling
    """

    def __init__(self):
        super().__init__()
        # Block 1: extracts low-level features (edges, colours)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 32×32 → 16×16
        )
        # Block 2: extracts higher-level features (shapes, textures)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 16×16 → 8×8
        )
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


class SimpleMLP(nn.Module):
    """A minimal multi-layer perceptron (fully-connected) for CIFAR-10.

    Architecture
    ------------
    Input: 3 × 32 × 32 image  →  Flatten  →  3072-dim vector
      FC(3072 → 512) → ReLU
      FC(512  → 256) → ReLU
      FC(256  → 10)

    Expected accuracy (no data augmentation, ~20 epochs)
    -----------------------------------------------------
      - Typical test accuracy: **45 – 55 %**
      - Hard ceiling for this architecture: ~58 %
      - MLPs are significantly weaker than CNNs on images because they
        discard all spatial structure by flattening the input.
      - The large first layer (3072 → 512) also makes the model prone to
        over-fitting without regularisation (dropout, weight decay).

    Use this model mainly as a baseline to appreciate the benefit of
    convolutional feature extraction (compare with SimpleCNN above).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_dataloaders(batch_size: int = 128):
    """Return CIFAR-10 train and test DataLoaders.

    Preprocessing
    -------------
    Only basic normalisation is applied (mean/std per channel computed over
    the CIFAR-10 training set).  No data augmentation is used here, which is
    one of the main reasons the baseline accuracy is limited (see model docs).

    Adding transforms such as ``RandomHorizontalFlip`` and ``RandomCrop``
    before the ``ToTensor`` call would typically improve accuracy by 3–8 pp.
    """
    # Per-channel mean and std for CIFAR-10 training set
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    # No augmentation – baseline setting
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=transform
    )
    test_set  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training and evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += images.size(0)

    return total_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 baseline trainer")
    parser.add_argument(
        "--model", choices=["cnn", "mlp"], default="cnn",
        help="Model architecture to train (default: cnn)"
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate for Adam optimiser (default: 0.001)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Mini-batch size (default: 128)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print accuracy expectations before training so users know what to expect
    if args.model == "cnn":
        print(
            "\n[INFO] Training SimpleCNN (baseline, no augmentation)\n"
            "       Expected test accuracy after training:\n"
            "         ~65 – 75 % (no augmentation, simple 2-block CNN)\n"
            "       To improve: add data augmentation, batch norm, or dropout.\n"
        )
    else:
        print(
            "\n[INFO] Training SimpleMLP (baseline, no augmentation)\n"
            "       Expected test accuracy after training:\n"
            "         ~45 – 55 % (MLP discards spatial structure)\n"
            "       Compare with SimpleCNN to see the benefit of convolutions.\n"
        )

    train_loader, test_loader = get_dataloaders(args.batch_size)

    model = (SimpleCNN() if args.model == "cnn" else SimpleMLP()).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8}")
    print("-" * 57)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)
        print(
            f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2f}% | "
            f"{test_loss:>9.4f} | {test_acc:>7.2f}%"
        )

    print(f"\nFinal test accuracy: {test_acc:.2f}%")
    print(
        "\nNote: Accuracy may vary ±2–3 pp across runs due to random weight\n"
        "      initialisation and data shuffling. See model docstrings and\n"
        "      README.md for a full discussion of accuracy ranges and\n"
        "      techniques to push beyond this baseline.\n"
    )


if __name__ == "__main__":
    main()
