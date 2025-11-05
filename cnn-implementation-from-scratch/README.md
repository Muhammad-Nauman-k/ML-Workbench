# PyTorch CNN Classifier from Scratch

This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** to classify the Fashion-MNIST dataset (from a CSV file instead of torchvision).  
The model reaches around **91% test accuracy** and demonstrates how to build and train a CNN **from scratch** without high-level training utilities.

---

##  Features
- Loads Fashion-MNIST directly from CSV format  
- Custom **Dataset & DataLoader** implementation  
- Convolutional Neural Network with:
  - ReLU activations  
  - Batch Normalization  
  - MaxPooling  
  - Dropout regularization  
- L2 Weight Decay to reduce overfitting  
- Manual training loop using **SGD optimizer** and **CrossEntropyLoss**  
- Automatic GPU support (uses CUDA if available)  
- Easily modifiable for other image classification datasets  

---
##  Installation dependencies

pip install -r requirements.txt

##  Installation

Clone the repository:
```bash
git clone https://github.com/your-username/pytorch-cnn-from-scratch.git
cd pytorch-cnn-from-scratch
