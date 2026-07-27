# VGG16 Transfer Learning for Fashion-MNIST

This project fine-tunes a pretrained **VGG16** model in **PyTorch** for **Fashion-MNIST** classification.
The convolutional feature extractor is frozen, and a custom classifier head is trained on Fashion-MNIST images loaded from CSV.

The trained model reaches about **92.7% validation accuracy**.

---

## Features

- Transfer learning with pretrained **VGG16 (ImageNet)**
- Frozen convolutional base and trainable custom classifier
- Fashion-MNIST loaded from CSV format
- Custom **Dataset** and **DataLoader** pipeline
- Converts grayscale images to **3-channel RGB-style input**
- Image preprocessing for VGG16:
  - Resize to 256
  - Center crop to 224
  - ImageNet normalization
- Training with **Adam** optimizer and **CrossEntropyLoss**
- Automatic GPU support when CUDA is available
- Exported full model for inference
- Separate prediction script for custom images

---

## Project Files

- `transfer_learning.ipynb` - Full training notebook (transfer learning + evaluation + model export)
- `Test_vgg16_trained.py` - Loads the trained model and predicts the class of an input image
- `vgg16_fashion_mnist_full.pth` - Exported trained model (~156 MB)
- `requirements.txt` - Project dependencies
- `README.md` - Project documentation

---

## Model Summary

- Backbone: `torchvision.models.vgg16(pretrained=True)`
- Frozen layers: all `features` parameters
- Custom classifier:
  - Linear(25088 → 1024) + ReLU + Dropout(0.5)
  - Linear(1024 → 512) + ReLU + Dropout(0.5)
  - Linear(512 → 10)
- Optimizer: Adam (`lr=0.0001`)
- Epochs: 15
- Batch size: 32
- Reported validation accuracy: **0.927**

---

## Class Labels

- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

The training notebook expects Fashion-MNIST CSV files:

- `fashion-mnist_train.csv`
- `fashion-mnist_test.csv`

In the notebook, update `dataset_path` if needed. The original notebook used:

```python
dataset_path = "/fashion-mnist_train"
```

---

## Training

Open and run the notebook:

```bash
jupyter notebook transfer_learning.ipynb
```

The notebook will:

1. Load Fashion-MNIST CSV data
2. Split training data into train/validation sets
3. Build a custom dataset with VGG16 transforms
4. Freeze VGG16 feature layers
5. Replace the classifier head for 10 Fashion-MNIST classes
6. Train for 15 epochs
7. Evaluate validation accuracy
8. Save the full model as `vgg16_fashion_mnist_full.pth`

---

## Inference

Make sure `vgg16_fashion_mnist_full.pth` is in the same folder, then run:

```bash
python Test_vgg16_trained.py
```

Update the image path inside the script if needed:

```python
image_path = "shirt.jfif"
```

