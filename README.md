# CIFAR-10 Baseline: CNN & MLP Image Classifier

This repository contains a minimal deep-learning baseline for the
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) image-classification
benchmark.  Two model architectures are provided:

| Model       | Description                                   |
|-------------|-----------------------------------------------|
| `SimpleCNN` | 2-block convolutional network (**recommended**) |
| `SimpleMLP` | Fully-connected (multi-layer perceptron)      |

---

## Quick Start

```bash
# Install dependencies
pip install torch torchvision

# Train the CNN (default)
python cifar10_train.py --model cnn --epochs 20

# Train the MLP for comparison
python cifar10_train.py --model mlp --epochs 20
```

---

## Expected Accuracy

> **TL;DR** — expect roughly **65–75 %** for the CNN and **45–55 %** for the
> MLP when training from scratch without data augmentation.

### Detailed ranges

| Model       | Data augmentation | Typical test accuracy |
|-------------|-------------------|-----------------------|
| `SimpleMLP` | None              | 45 – 55 %             |
| `SimpleCNN` | None              | 65 – 75 %             |
| `SimpleCNN` | Basic †           | 72 – 80 %             |

† Basic augmentation = `RandomHorizontalFlip` + `RandomCrop(32, padding=4)`.

These numbers assume ~20 training epochs with the Adam optimiser (lr = 0.001)
and no regularisation techniques.  Accuracy can vary ±2–3 percentage points
across runs due to random weight initialisation and mini-batch sampling.

### Why is accuracy limited?

The baseline intentionally uses the simplest possible architecture so that
the code is easy to read and understand.  The main limiting factors are:

1. **Shallow architecture** – only 2 convolutional blocks; deeper networks
   (ResNet, VGG) learn richer feature hierarchies.
2. **No data augmentation** – the model only sees each training image in its
   original form, which increases over-fitting.
3. **No batch normalisation** – without BN, training can be unstable and
   generalisation suffers.
4. **No dropout** – no explicit regularisation means the network memorises
   training examples rather than generalising.
5. **No learning-rate scheduling** – a constant learning rate is suboptimal
   for convergence.

### How to improve accuracy

| Technique                        | Typical gain |
|----------------------------------|-------------|
| Random horizontal flip + crop    | +3 – 8 pp   |
| Batch normalisation              | +2 – 5 pp   |
| Dropout (p = 0.3–0.5)            | +1 – 3 pp   |
| LR scheduling (step / cosine)    | +1 – 3 pp   |
| Wider / deeper CNN (ResNet-20)   | up to ~93 % |
| Transfer learning (pre-trained)  | up to ~98 % |

### State-of-the-art reference

As of 2024, the best published results on CIFAR-10 exceed **99 %** accuracy,
achieved with large models (Vision Transformers, EfficientNet), aggressive
data augmentation (AutoAugment, CutMix), and techniques like knowledge
distillation.  The `SimpleCNN` baseline here is intentionally simple and is
not meant to compete with state-of-the-art — it is a starting point for
understanding the fundamentals of image classification.

---

## Model Architecture Details

### SimpleCNN

```
Input: 3 × 32 × 32
  Conv(3→32, 3×3, padding=1) → ReLU → MaxPool(2×2)  →  32 × 16 × 16
  Conv(32→64, 3×3, padding=1) → ReLU → MaxPool(2×2) →  64 ×  8 ×  8
  Flatten  →  4096-dim vector
  Linear(4096 → 512) → ReLU
  Linear(512 → 10)
```

Total trainable parameters: ~2.1 M

### SimpleMLP

```
Input: 3 × 32 × 32  →  Flatten  →  3072-dim vector
  Linear(3072 → 512) → ReLU
  Linear(512  → 256) → ReLU
  Linear(256  → 10)
```

Total trainable parameters: ~1.7 M

The MLP achieves noticeably lower accuracy because flattening the image
discards all spatial relationships between pixels, which are essential for
recognising objects.

---

## Dataset

| Property        | Value                     |
|-----------------|---------------------------|
| Images          | 60 000 (50 k train / 10 k test) |
| Resolution      | 32 × 32 pixels, RGB       |
| Classes         | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Download size   | ~163 MB                   |

The dataset is downloaded automatically to `./data/` on the first run.
