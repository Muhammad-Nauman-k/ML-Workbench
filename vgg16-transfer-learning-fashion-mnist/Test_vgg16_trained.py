import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Class names
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(
    "vgg16_fashion_mnist_full.pth", map_location=device, weights_only=False
)
model.eval()

# Transform (same as training)
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def predict(image_path):
    # Convert grayscale Fashion-MNIST style input to 3 channels for VGG16.
    img = Image.open(image_path).convert("L")
    img = np.stack([np.array(img)] * 3, axis=-1)
    img = Image.fromarray(img.astype(np.uint8))
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    return class_names[predicted.item()]


image_path = "shirt.jfif"  # Change this to your image path
result = predict(image_path)
print(f"Predicted class: {result}")
